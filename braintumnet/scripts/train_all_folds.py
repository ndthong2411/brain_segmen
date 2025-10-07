#!/usr/bin/env python3
"""
Train all K folds sequentially.

This script trains all cross-validation folds one after another.
Useful for comprehensive model evaluation and ensemble predictions.

Usage:
    # Train all 5 folds with default config
    python scripts/train_all_folds.py --config configs/full_dataset.yaml

    # Train specific folds only
    python scripts/train_all_folds.py --config configs/full_dataset.yaml --folds 0 1 2
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from braintumnet.utils.io import load_yaml


def main():
    parser = argparse.ArgumentParser(description='Train all cross-validation folds')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='Specific folds to train (e.g., --folds 0 1 2). If not specified, trains all folds.')
    args = parser.parse_args()

    # Load config to get number of folds
    cfg = load_yaml(args.config)
    num_folds = cfg['data']['num_folds']

    # Determine which folds to train
    if args.folds is None:
        folds_to_train = list(range(num_folds))
    else:
        folds_to_train = args.folds
        # Validate fold numbers
        for fold in folds_to_train:
            if fold < 0 or fold >= num_folds:
                print(f"Error: Fold {fold} is out of range [0, {num_folds-1}]")
                sys.exit(1)

    print("="*70)
    print(f"TRAINING ALL FOLDS")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Total folds: {num_folds}")
    print(f"Folds to train: {folds_to_train}")
    print("="*70 + "\n")

    # Train each fold
    results = []
    total_start_time = time.time()

    for i, fold in enumerate(folds_to_train):
        print("\n" + "="*70)
        print(f"TRAINING FOLD {fold} ({i+1}/{len(folds_to_train)})")
        print("="*70)

        fold_start_time = time.time()

        # Run training script for this fold
        cmd = [
            sys.executable,  # Use same Python interpreter
            str(ROOT / "scripts" / "train.py"),
            "--cfg", args.config,
            "--fold", str(fold)
        ]

        print(f"Command: {' '.join(cmd)}\n")

        try:
            result = subprocess.run(cmd, check=True)
            fold_time = time.time() - fold_start_time

            results.append({
                'fold': fold,
                'status': 'SUCCESS',
                'time': fold_time
            })

            print(f"\n✓ Fold {fold} completed in {fold_time/3600:.2f} hours")

        except subprocess.CalledProcessError as e:
            fold_time = time.time() - fold_start_time

            results.append({
                'fold': fold,
                'status': 'FAILED',
                'time': fold_time
            })

            print(f"\n✗ Fold {fold} failed after {fold_time/3600:.2f} hours")
            print(f"Error: {e}")

            # Ask user if they want to continue
            response = input("\nContinue with remaining folds? (y/n): ").strip().lower()
            if response != 'y':
                print("Stopping training.")
                break

    # Summary
    total_time = time.time() - total_start_time

    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)

    for r in results:
        status_symbol = "✓" if r['status'] == 'SUCCESS' else "✗"
        print(f"{status_symbol} Fold {r['fold']:2d}: {r['status']:8s} ({r['time']/3600:.2f} hours)")

    print("-"*70)
    successful = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed = sum(1 for r in results if r['status'] == 'FAILED')

    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed:     {failed}/{len(results)}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print("="*70 + "\n")

    if successful == len(results):
        print("✓ All folds trained successfully!")
        print("\nNext step: Evaluate all folds")
        print(f"  python scripts/evaluate.py --cfg {args.config} --all_folds\n")
    else:
        print("⚠ Some folds failed. Check the logs above.")


if __name__ == "__main__":
    main()
