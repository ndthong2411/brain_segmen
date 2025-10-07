#!/usr/bin/env python
"""
Compare multiple training runs side-by-side.
Useful for hyperparameter tuning and model selection.
"""
import os
import sys
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not installed. Install with: pip install tensorboard")


def load_run_data(run_dir):
    """Load metrics from a single run."""
    if not HAS_TENSORBOARD:
        return None

    try:
        # Find event file
        event_files = list(Path(run_dir).glob('events.out.tfevents.*'))
        if not event_files:
            return None

        event_file = str(max(event_files, key=os.path.getmtime))
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        data = {}

        # Load validation metrics
        if 'val/iou' in ea.Tags()['scalars']:
            events = ea.Scalars('val/iou')
            data['epochs'] = [e.step for e in events]
            data['val_iou'] = [e.value for e in events]

        if 'val/dice' in ea.Tags()['scalars']:
            events = ea.Scalars('val/dice')
            data['val_dice'] = [e.value for e in events]

        if 'val/cls_acc' in ea.Tags()['scalars']:
            events = ea.Scalars('val/cls_acc')
            data['val_acc'] = [e.value for e in events]

        if 'epoch/train_loss' in ea.Tags()['scalars']:
            events = ea.Scalars('epoch/train_loss')
            data['train_loss'] = [e.value for e in events]

        return data

    except Exception as e:
        print(f"Error loading {run_dir}: {e}")
        return None


def compare_runs(run_dirs, run_names=None, output_path=None):
    """Compare multiple training runs."""
    if run_names is None:
        run_names = [os.path.basename(d) for d in run_dirs]

    # Load all runs
    all_data = []
    valid_names = []
    for run_dir, run_name in zip(run_dirs, run_names):
        data = load_run_data(run_dir)
        if data:
            all_data.append(data)
            valid_names.append(run_name)
            print(f"Loaded: {run_name}")
        else:
            print(f"Skipped: {run_name} (no data)")

    if not all_data:
        print("Error: No valid runs found")
        return

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Runs Comparison', fontsize=14, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Plot 1: Validation IoU
    ax1 = axes[0, 0]
    for idx, (data, name) in enumerate(zip(all_data, valid_names)):
        if 'val_iou' in data:
            color = colors[idx % len(colors)]
            ax1.plot(data['epochs'], data['val_iou'], '-o', linewidth=2,
                    markersize=4, label=name, color=color, alpha=0.8)
            # Annotate best value
            best_iou = max(data['val_iou'])
            best_epoch = data['epochs'][data['val_iou'].index(best_iou)]
            ax1.plot(best_epoch, best_iou, '*', markersize=12, color=color)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('IoU Score')
    ax1.set_title('Validation IoU')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Plot 2: Validation Dice
    ax2 = axes[0, 1]
    for idx, (data, name) in enumerate(zip(all_data, valid_names)):
        if 'val_dice' in data:
            color = colors[idx % len(colors)]
            ax2.plot(data['epochs'], data['val_dice'], '-s', linewidth=2,
                    markersize=4, label=name, color=color, alpha=0.8)
            # Annotate best value
            best_dice = max(data['val_dice'])
            best_epoch = data['epochs'][data['val_dice'].index(best_dice)]
            ax2.plot(best_epoch, best_dice, '*', markersize=12, color=color)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Validation Dice')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Plot 3: Classification Accuracy
    ax3 = axes[1, 0]
    for idx, (data, name) in enumerate(zip(all_data, valid_names)):
        if 'val_acc' in data:
            color = colors[idx % len(colors)]
            ax3.plot(data['epochs'], data['val_acc'], '-^', linewidth=2,
                    markersize=4, label=name, color=color, alpha=0.8)
            # Annotate best value
            best_acc = max(data['val_acc'])
            best_epoch = data['epochs'][data['val_acc'].index(best_acc)]
            ax3.plot(best_epoch, best_acc, '*', markersize=12, color=color)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Classification Accuracy')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    # Plot 4: Training Loss
    ax4 = axes[1, 1]
    for idx, (data, name) in enumerate(zip(all_data, valid_names)):
        if 'train_loss' in data:
            color = colors[idx % len(colors)]
            ax4.plot(data['epochs'], data['train_loss'], '-', linewidth=2,
                    label=name, color=color, alpha=0.8)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Training Loss')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved comparison to: {output_path}")
    else:
        plt.show()

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Run Name':<40} {'Best IoU':>10} {'Best Dice':>10} {'Best Acc':>10}")
    print("-"*80)

    for data, name in zip(all_data, valid_names):
        best_iou = max(data.get('val_iou', [0]))
        best_dice = max(data.get('val_dice', [0]))
        best_acc = max(data.get('val_acc', [0]))
        print(f"{name[:40]:<40} {best_iou:>10.4f} {best_dice:>10.4f} {best_acc:>10.4f}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare multiple BrainTumNet training runs")
    parser.add_argument("--logdir", type=str, default="runs",
                       help="Base directory containing run logs")
    parser.add_argument("--runs", type=str, nargs='+', default=None,
                       help="Specific run names to compare (if not specified, compares all)")
    parser.add_argument("--save", type=str, default=None,
                       help="Save comparison plot to file")
    args = parser.parse_args()

    # Find all runs
    if args.runs:
        run_dirs = [os.path.join(args.logdir, r) for r in args.runs]
        run_names = args.runs
    else:
        # Auto-detect all runs
        if not os.path.exists(args.logdir):
            print(f"Error: Log directory not found: {args.logdir}")
            return

        run_dirs = [os.path.join(args.logdir, d) for d in os.listdir(args.logdir)
                   if os.path.isdir(os.path.join(args.logdir, d))]
        run_names = [os.path.basename(d) for d in run_dirs]

        if not run_dirs:
            print(f"Error: No runs found in {args.logdir}")
            return

        print(f"Found {len(run_dirs)} runs in {args.logdir}")

    compare_runs(run_dirs, run_names, args.save)


if __name__ == "__main__":
    main()
