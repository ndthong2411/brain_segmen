#!/usr/bin/env python3
"""
Simple evaluation script - just pass checkpoint path.

Usage:
    python braintumnet/scripts/evaluate.py checkpoints/braintumnet_best_fold0.pth
    python braintumnet/scripts/evaluate.py checkpoints/braintumnet_best_fold4.pth
"""

import os, argparse, sys, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from braintumnet.utils.io import load_yaml
from braintumnet.engine.evaluator import evaluate


def extract_fold_from_path(ckpt_path):
    """Extract fold number from checkpoint filename.

    Examples:
        braintumnet_best_fold0.pth -> 0
        braintumnet_best_fold4.pth -> 4
        last_fold2.pth -> 2
    """
    match = re.search(r'fold(\d+)', ckpt_path)
    if match:
        return int(match.group(1))
    return 0  # Default to fold 0


def detect_model_type_from_path(ckpt_path):
    """Detect model type from checkpoint path.

    Examples:
        checkpoints/nnunet/nnunet_fold4/... -> nnunet
        checkpoints/swin_unetr/... -> swin_unetr
        checkpoints/braintumnet_best_fold0.pth -> segunetv2 (default)
    """
    ckpt_path_lower = ckpt_path.lower()

    # Check for model type in path
    known_models = ['nnunet', 'swin_unetr', 'unetr', 'transunet', 'lg_unetr', 'segunetv2']
    for model in known_models:
        if model in ckpt_path_lower:
            return model

    # Default to segunetv2
    return 'segunetv2'


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


def main():
    ap = argparse.ArgumentParser(
        description='Evaluate BrainTumNet checkpoint with comprehensive metrics (DSC, IoU, HD, HD95)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python braintumnet/scripts/evaluate.py checkpoints/braintumnet_best_fold0.pth
  python braintumnet/scripts/evaluate.py checkpoints/braintumnet_best_fold4.pth
        """
    )
    ap.add_argument("checkpoint", type=str, help='Path to checkpoint file')
    args = ap.parse_args()

    ckpt_path = args.checkpoint

    # Verify checkpoint exists
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Extract fold number from checkpoint path
    fold = extract_fold_from_path(ckpt_path)
    print(f"\nDetected fold: {fold}")

    # Detect model type from checkpoint path
    model_type = detect_model_type_from_path(ckpt_path)
    print(f"Detected model type: {model_type}")

    # Load base config
    cfg_path = ROOT / "configs" / "base.yaml"
    print(f"Loading base config: {cfg_path}")
    cfg = load_yaml(str(cfg_path))

    # Load model-specific config if available
    model_cfg_path = ROOT / "configs" / "models" / f"{model_type}.yaml"
    if model_cfg_path.exists():
        print(f"Loading model config: {model_cfg_path}")
        model_cfg = load_yaml(str(model_cfg_path))
        cfg = merge_configs(cfg, model_cfg)
    else:
        print(f"No model-specific config found at {model_cfg_path}, using base config only")

    # Override fold in config
    cfg['data']['fold'] = fold

    # Evaluate
    print(f"\nEvaluating checkpoint: {ckpt_path}")
    print("="*70)

    results = evaluate(cfg, fold, ckpt_path)

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"  IoU (Jaccard):  {results['iou']:.4f}")
    print(f"  Dice (F1):      {results['dice']:.4f}")
    print(f"  HD:             {results['hd']:.2f} px")
    print(f"  HD95:           {results['hd95']:.2f} px")
    print(f"  Accuracy:       {results['acc']:.4f}")
    print(f"  F1 Score:       {results['f1']:.4f}")
    print(f"  AUC-ROC:        {results['auc']:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
