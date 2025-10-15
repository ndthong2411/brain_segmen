#!/usr/bin/env python3
"""
Monitor Training Progress - Easy Way to Track Metrics

Usage:
    python scripts/monitor_training.py
    python scripts/monitor_training.py --fold 4
    python scripts/monitor_training.py --last 10  # Show last 10 epochs
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict


def parse_log_line(line: str) -> Dict:
    """Parse a summary log line into metrics dict."""
    if "SUMMARY" not in line:
        return None

    # Extract metrics using regex
    metrics = {}

    # Epoch number
    epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
    if epoch_match:
        metrics['epoch'] = int(epoch_match.group(1))
        metrics['total_epochs'] = int(epoch_match.group(2))

    # Key metrics
    patterns = {
        'train_loss': r'train_loss: ([-\d.]+)',
        'val_iou': r'val_iou: ([\d.]+)',
        'val_dice': r'val_dice: ([\d.]+)',
        'WT_iou': r'WT_iou: ([\d.]+)',
        'TC_iou': r'TC_iou: ([\d.]+)',
        'ED_iou': r'ED_iou: ([\d.]+)',
        'lr': r'lr: ([\d.e+-]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except ValueError:
                metrics[key] = 0.0

    return metrics if metrics else None


def find_latest_log(fold: int = None) -> str:
    """Find the most recent log file."""
    log_dir = Path("logs")
    if not log_dir.exists():
        print("❌ Logs directory not found!")
        return None

    # Search pattern
    if fold is not None:
        pattern = f"*_fold{fold}_*.log"
    else:
        pattern = "*.log"

    log_files = sorted(log_dir.glob(pattern), key=os.path.getmtime, reverse=True)

    if not log_files:
        print(f"❌ No log files found matching: {pattern}")
        return None

    return str(log_files[0])


def print_training_summary(log_file: str, last_n: int = None):
    """Print a nice summary of training progress."""

    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return

    # Read and parse log
    metrics_history = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            metrics = parse_log_line(line)
            if metrics:
                metrics_history.append(metrics)

    if not metrics_history:
        print("❌ No metrics found in log file")
        return

    # Get subset if requested
    if last_n:
        metrics_history = metrics_history[-last_n:]

    # Extract info from filename
    filename = Path(log_file).name
    fold_match = re.search(r'fold(\d+)', filename)
    fold = fold_match.group(1) if fold_match else "?"

    # Print header
    print("=" * 120)
    print(f"📊 TRAINING MONITOR - Fold {fold}")
    print(f"📁 Log: {Path(log_file).name}")
    print("=" * 120)
    print()

    # Get best metrics
    best_epoch = max(metrics_history, key=lambda x: x.get('val_iou', 0))
    latest_epoch = metrics_history[-1]
    first_epoch = metrics_history[0]

    # Summary stats
    print("🎯 CURRENT STATUS:")
    print(f"  Epoch: {latest_epoch['epoch']}/{latest_epoch.get('total_epochs', '?')}")
    print(f"  Val IoU: {latest_epoch.get('val_iou', 0):.4f}")
    print(f"  TC IoU:  {latest_epoch.get('TC_iou', 0):.4f} (hardest class)")
    print(f"  ED IoU:  {latest_epoch.get('ED_iou', 0):.4f}")
    print(f"  WT IoU:  {latest_epoch.get('WT_iou', 0):.4f}")
    print()

    print("⭐ BEST RESULT:")
    print(f"  Epoch: {best_epoch['epoch']}")
    print(f"  Val IoU: {best_epoch.get('val_iou', 0):.4f} ✅")
    print(f"  TC IoU:  {best_epoch.get('TC_iou', 0):.4f}")
    print(f"  ED IoU:  {best_epoch.get('ED_iou', 0):.4f}")
    print(f"  WT IoU:  {best_epoch.get('WT_iou', 0):.4f}")
    print()

    print("📈 PROGRESS:")
    print(f"  Starting IoU: {first_epoch.get('val_iou', 0):.4f}")
    print(f"  Current IoU:  {latest_epoch.get('val_iou', 0):.4f}")
    print(f"  Best IoU:     {best_epoch.get('val_iou', 0):.4f}")
    print(f"  Improvement:  +{(best_epoch.get('val_iou', 0) - first_epoch.get('val_iou', 0)):.4f}")
    print()

    # Targets
    print("🎯 TARGETS:")
    current_iou = latest_epoch.get('val_iou', 0)
    if current_iou < 0.70:
        status = "🔴 Need improvement"
        target = "Target: 0.70+"
    elif current_iou < 0.75:
        status = "🟡 Good"
        target = "Target: 0.75+ (Phase 1 goal)"
    elif current_iou < 0.80:
        status = "🟢 Very Good"
        target = "Target: 0.80+ (Phase 2 Small goal)"
    elif current_iou < 0.85:
        status = "🟢 Excellent"
        target = "Target: 0.85+ (Phase 2 A100 goal)"
    else:
        status = "🏆 Outstanding"
        target = "Target: 0.90 (Final goal with ensemble)"

    print(f"  Status: {status}")
    print(f"  {target}")
    print()

    # Recent history table
    print("📊 RECENT EPOCHS:")
    print("-" * 120)
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val IoU':>9} | {'TC IoU':>8} | {'ED IoU':>8} | {'WT IoU':>8} | {'Status':>10}")
    print("-" * 120)

    # Show last N epochs
    display_epochs = metrics_history[-min(20, len(metrics_history)):]

    for m in display_epochs:
        epoch = m.get('epoch', 0)
        train_loss = m.get('train_loss', 0)
        val_iou = m.get('val_iou', 0)
        tc_iou = m.get('TC_iou', 0)
        ed_iou = m.get('ED_iou', 0)
        wt_iou = m.get('WT_iou', 0)

        # Determine status
        is_best = (m == best_epoch)
        status = "⭐ BEST" if is_best else ""

        print(f"{epoch:>6} | {train_loss:>12.4f} | {val_iou:>9.4f} | {tc_iou:>8.4f} | {ed_iou:>8.4f} | {wt_iou:>8.4f} | {status:>10}")

    print("-" * 120)
    print()

    # Trend analysis
    if len(metrics_history) >= 5:
        recent_5 = metrics_history[-5:]
        avg_recent_iou = sum(m.get('val_iou', 0) for m in recent_5) / 5

        older_5 = metrics_history[-10:-5] if len(metrics_history) >= 10 else metrics_history[:5]
        avg_older_iou = sum(m.get('val_iou', 0) for m in older_5) / len(older_5)

        trend = avg_recent_iou - avg_older_iou

        print("📉 TREND ANALYSIS (Last 5 vs Previous 5):")
        if trend > 0.01:
            print(f"  🟢 IMPROVING: +{trend:.4f} (Good!)")
        elif trend > -0.01:
            print(f"  🟡 STABLE: {trend:+.4f} (Plateauing)")
        else:
            print(f"  🔴 DECLINING: {trend:+.4f} (May need adjustment)")
        print()

    # Recommendations
    print("💡 RECOMMENDATIONS:")
    tc_iou = latest_epoch.get('TC_iou', 0)
    ed_iou = latest_epoch.get('ED_iou', 0)

    if tc_iou < 0.65:
        print("  ⚠️  TC IoU is low - consider increasing TC class weight")
    if ed_iou < 0.70:
        print("  ⚠️  ED IoU is low - check data augmentation")
    if current_iou > 0.82:
        print("  🎉 Great results! Consider training other folds and applying ensemble")
    elif current_iou > 0.75:
        print("  👍 Good progress! Continue training to reach 0.80+")
    else:
        print("  📚 Keep training - model is still learning")

    print()
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument('--fold', type=int, default=None, help='Fold number to monitor')
    parser.add_argument('--log', type=str, default=None, help='Specific log file to analyze')
    parser.add_argument('--last', type=int, default=None, help='Show only last N epochs in detail')

    args = parser.parse_args()

    # Find log file
    if args.log:
        log_file = args.log
    else:
        log_file = find_latest_log(args.fold)

    if not log_file:
        return

    # Print summary
    print_training_summary(log_file, args.last)


if __name__ == "__main__":
    main()
