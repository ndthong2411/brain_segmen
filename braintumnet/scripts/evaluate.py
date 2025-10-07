#!/usr/bin/env python3
"""
Comprehensive evaluation script with DSC, IoU, HD, and HD95 metrics.

Usage:
    # Evaluate a specific checkpoint
    python scripts/evaluate.py --cfg configs/full_dataset.yaml --ckpt checkpoints/braintumnet_best_fold0.pth --fold 0

    # Evaluate all folds
    python scripts/evaluate.py --cfg configs/full_dataset.yaml --all_folds
"""

import os, argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from braintumnet.utils.io import load_yaml
from braintumnet.engine.evaluator import evaluate


def main():
    ap = argparse.ArgumentParser(description='Evaluate BrainTumNet model with comprehensive metrics (DSC, IoU, HD, HD95)')
    ap.add_argument("--cfg", type=str, default=str(ROOT / "configs" / "default.yaml"), help='Path to config YAML file')
    ap.add_argument("--ckpt", type=str, help='Path to checkpoint file (required if not using --all_folds)')
    ap.add_argument("--fold", type=int, default=0, help='Fold number to evaluate (default: 0)')
    ap.add_argument("--all_folds", action='store_true', help='Evaluate all folds using best checkpoints')
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)

    if args.all_folds:
        # Evaluate all folds
        print("\n" + "="*70)
        print("EVALUATING ALL FOLDS")
        print("="*70)

        all_results = []
        for fold in range(cfg['data']['num_folds']):
            ckpt_path = os.path.join(cfg['logging']['save_dir'], f"braintumnet_best_fold{fold}.pth")
            if not os.path.exists(ckpt_path):
                print(f"\nWarning: Checkpoint not found for fold {fold}: {ckpt_path}")
                print(f"Skipping fold {fold}...")
                continue

            print(f"\n{'='*70}")
            print(f"FOLD {fold}")
            print(f"{'='*70}")
            print(f"Checkpoint: {ckpt_path}")

            results = evaluate(cfg, fold, ckpt_path)
            results['fold'] = fold
            all_results.append(results)

        # Aggregate results
        if len(all_results) > 0:
            print("\n" + "="*70)
            print("AGGREGATED RESULTS ACROSS ALL FOLDS")
            print("="*70)

            metrics = ['iou', 'dice', 'hd', 'hd95', 'acc', 'f1', 'auc']
            for metric in metrics:
                values = [r[metric] for r in all_results if not np.isnan(r[metric])]
                if len(values) > 0:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"  {metric.upper():6s}: {mean_val:.4f} ± {std_val:.4f}")
                else:
                    print(f"  {metric.upper():6s}: N/A")
            print("="*70 + "\n")
        else:
            print("\nNo folds were successfully evaluated.")

    else:
        # Evaluate single fold
        if args.ckpt is None:
            # Try to use default checkpoint path
            ckpt_path = os.path.join(cfg['logging']['save_dir'], f"braintumnet_best_fold{args.fold}.pth")
            if not os.path.exists(ckpt_path):
                print(f"Error: No checkpoint specified and default not found: {ckpt_path}")
                print("Please specify --ckpt or ensure the checkpoint exists at the default location.")
                sys.exit(1)
        else:
            ckpt_path = args.ckpt

        if not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint not found: {ckpt_path}")
            sys.exit(1)

        print(f"\nEvaluating checkpoint: {ckpt_path}")
        results = evaluate(cfg, args.fold, ckpt_path)

        # Print summary
        print("\nFinal Metrics Summary:")
        print(f"  IoU:   {results['iou']:.4f}")
        print(f"  Dice:  {results['dice']:.4f}")
        print(f"  HD:    {results['hd']:.2f} px")
        print(f"  HD95:  {results['hd95']:.2f} px")
        print(f"  Acc:   {results['acc']:.4f}")
        print(f"  F1:    {results['f1']:.4f}")
        print(f"  AUC:   {results['auc']:.4f}\n")


if __name__ == "__main__":
    main()
