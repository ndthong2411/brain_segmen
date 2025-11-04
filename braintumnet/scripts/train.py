import os, argparse, sys
from pathlib import Path

# Tắt TensorFlow warnings khi dùng TensorBoard
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Chỉ hiện errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Tắt oneDNN messages

ROOT = Path(__file__).resolve().parents[1]  # braintumnet/
sys.path.append(str(ROOT / "src"))

from braintumnet.utils.io import load_yaml
from braintumnet.utils.seed import set_seed
from braintumnet.engine.trainer import train_one_fold, prepare_artifact_dirs


def merge_configs(base_cfg, override_cfg):
    """Deep merge two configs, override_cfg takes precedence"""
    import copy
    merged = copy.deepcopy(base_cfg)

    for key, value in override_cfg.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge dicts
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged


def load_config_auto(model_name=None, hardware=None):
    """
    Auto-load and merge configs based on model and hardware

    Priority (later overrides earlier):
    1. base.yaml (common settings)
    2. models/{model}.yaml (model-specific)
    3. hardware_{hardware}.yaml (hardware-specific)

    Args:
        model_name: Model name (segunetv2, swin_unetr, nnunet, unetr)
        hardware: Hardware config (a100, or None for default)

    Returns:
        Merged config dict
    """
    configs_dir = ROOT / "configs"

    # 1. Load base config
    base_path = configs_dir / "base.yaml"
    print(f"Loading base config: {base_path}")
    cfg = load_yaml(str(base_path))

    # 2. Load model-specific config if model is specified
    if model_name:
        model_path = configs_dir / "models" / f"{model_name}.yaml"
        if model_path.exists():
            print(f"Loading model config: {model_path}")
            model_cfg = load_yaml(str(model_path))
            cfg = merge_configs(cfg, model_cfg)
        else:
            print(f"Warning: Model config not found: {model_path}")

    # 3. Load hardware-specific config if specified
    if hardware:
        hw_path = configs_dir / f"hardware_{hardware}.yaml"
        if hw_path.exists():
            print(f"Loading hardware config: {hw_path}")
            hw_cfg = load_yaml(str(hw_path))
            cfg = merge_configs(cfg, hw_cfg)

            # Apply model-specific batch size for this hardware
            if model_name and "model_batch_sizes" in hw_cfg:
                batch_sizes = hw_cfg["model_batch_sizes"]
                if model_name in batch_sizes:
                    cfg["train"]["batch_size"] = batch_sizes[model_name]
                    print(f"  Using {hardware} batch size for {model_name}: {batch_sizes[model_name]}")
        else:
            print(f"Warning: Hardware config not found: {hw_path}")

    return cfg


def main():

    print(f"Process PID: {os.getpid()}", flush=True)

    ap = argparse.ArgumentParser(
        description="Train brain tumor segmentation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train SegUNetV2 Phase 1 (OPTIMIZED) on A100
  python scripts/train.py --model segunetv2_phase1 --cfg a100 --fold 0

  # Train SegUNetV2 baseline
  python scripts/train.py --model segunetv2 --fold 0

  # Train Swin-UNETR on default hardware (3090)
  python scripts/train.py --model swin_unetr --fold 0

  # Train Swin-UNETR on A100
  python scripts/train.py --model swin_unetr --cfg a100 --fold 0

  # Resume training
  python scripts/train.py --model segunetv2_phase1 --fold 0 --resume
        """
    )

    ap.add_argument("--model", type=str, default="segunetv2",
                    choices=['segunetv2', 'segunetv2_phase1', 'segunetv2_phase2', 'v2', 'swin_unetr', 'nnunet', 'unetr', 'transunet', 'lg_unetr'],
                    help="Model architecture (default: segunetv2)")
    ap.add_argument("--cfg", type=str, default=None,
                    choices=['a100'],
                    help="Hardware config (default: None for 3090/local, 'a100' for A100 server)")
    ap.add_argument("--fold", type=int, default=None,
                    help="Fold number 0-4 (default: 0)")
    ap.add_argument("--resume", type=str, nargs='?', const='auto', default=None,
                    help="Resume training. Use --resume (auto-find) or --resume <path>")

    args = ap.parse_args()

    # Auto-load config based on model and hardware
    print("="*60)
    print("Configuration Loading")
    print("="*60)
    cfg = load_config_auto(model_name=args.model, hardware=args.cfg)
    print(f"Config loaded successfully!")

    # Set fold
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold
    else:
        cfg["data"]["fold"] = 0  # Default to fold 0

    fold = cfg["data"]["fold"]

    # Set experiment name based on model and fold
    model_name = cfg["model"]["model_type"]
    cfg["logging"]["exp_name"] = f"{model_name}_fold{fold}"
    artifact_dirs = prepare_artifact_dirs(cfg)

    # Print config summary
    print("\n" + "="*60)
    print("Training Configuration Summary")
    print("="*60)
    print(f"Model:           {cfg['model']['model_type']}")
    print(f"Hardware:        {'A100' if args.cfg == 'a100' else 'Default (3090/Local)'}")
    print(f"Fold:            {fold}")
    print(f"Batch size:      {cfg['train']['batch_size']}")
    print(f"Workers:         {cfg['train']['workers']}")
    print(f"AMP dtype:       {cfg['train']['amp_dtype']}")
    print(f"Optimizer fused: {cfg['train'].get('optimizer_fused', False)}")
    print(f"Channels last:   {cfg['train'].get('channels_last', False)}")
    print(f"Data backend:    {cfg['data']['backend']}")
    print(f"Experiment name: {cfg['logging']['exp_name']}")
    print(f"Log dir:         {artifact_dirs['log_dir']}")
    print(f"Checkpoint dir:  {artifact_dirs['save_dir']}")
    print(f"TensorBoard dir: {artifact_dirs['out_dir']}")
    print("="*60 + "\n")

    # Auto-find checkpoint if --resume flag is used without path
    resume_path = args.resume
    if args.resume == 'auto':
        # Auto-detect checkpoint path based on fold and config
        ckpt_dir = artifact_dirs["save_dir"]
        auto_ckpt_path = os.path.join(ckpt_dir, f"last_fold{fold}.pth")

        if os.path.exists(auto_ckpt_path):
            resume_path = auto_ckpt_path
            print(f"✓ Auto-detected checkpoint: {auto_ckpt_path}")
            print(f"  Will resume training for fold {fold}")
        else:
            print(f"✗ No checkpoint found at: {auto_ckpt_path}")
            print(f"  Starting training from scratch for fold {fold}...")
            resume_path = None
    elif args.resume is not None:
        # Manual path provided - validate it exists
        if not os.path.exists(args.resume):
            print(f"✗ Error: Checkpoint not found at: {args.resume}")
            print(f"  Please check the path and try again.")
            sys.exit(1)
        print(f"✓ Using checkpoint: {args.resume}")
        print(f"  WARNING: Make sure this checkpoint is for fold {fold}!")

    print(f"Setting seed...", flush=True)
    set_seed(42, deterministic=False)
    print(f"Starting training...", flush=True)
    best_iou = train_one_fold(cfg, fold,
                               config_path=None,  # We built config programmatically
                               resume_from=resume_path)
    print(f"\n{'='*60}")
    print(f"Training completed! Best IoU: {best_iou:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
