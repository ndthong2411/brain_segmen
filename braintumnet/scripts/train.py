import os, argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # braintumnet/
sys.path.append(str(ROOT / "src"))

from braintumnet.utils.io import load_yaml
from braintumnet.utils.seed import set_seed
from braintumnet.engine.trainer import train_one_fold

def main():
    
    print(f"Process PID: {os.getpid()}")
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default=str(ROOT / "configs" / "default.yaml"),
                    help="Path to config YAML file")
    ap.add_argument("--fold", type=int, default=None,
                    help="0..K-1. If not set, uses cfg.data.fold")
    ap.add_argument("--resume", type=str, nargs='?', const='auto', default=None,
                    help="Resume training. Use --resume (auto-find) or --resume <path>")
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold

    fold = cfg["data"]["fold"]

    # Auto-find checkpoint if --resume flag is used without path
    resume_path = args.resume
    if args.resume == 'auto':
        # Auto-detect checkpoint path based on fold and config
        ckpt_dir = cfg["logging"]["save_dir"]
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

    set_seed(42, deterministic=False)
    best_iou = train_one_fold(cfg, fold,
                               config_path=args.cfg,
                               resume_from=resume_path)
    print("Best IoU:", best_iou)

if __name__ == "__main__":
    main()
