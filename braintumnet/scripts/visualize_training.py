#!/usr/bin/env python
"""
Real-time training visualization tool.
Monitors TensorBoard logs and creates live plots of training progress.
"""
import os
import sys
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not installed. Install with: pip install tensorboard")


class TrainingVisualizer:
    """Live visualization of training metrics from TensorBoard logs."""

    def __init__(self, log_dir, refresh_interval=5):
        """
        Args:
            log_dir: Path to TensorBoard log directory
            refresh_interval: Seconds between plot updates
        """
        self.log_dir = log_dir
        self.refresh_interval = refresh_interval
        self.ea = None

        # Data storage
        self.data = {
            'step': [],
            'train_loss': [],
            'train_loss_seg': [],
            'train_loss_cls': [],
            'lr': [],
            'epoch': [],
            'val_iou': [],
            'val_dice': [],
            'val_acc': []
        }

        # Setup plot
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle(f'Training Progress: {os.path.basename(log_dir)}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    def load_logs(self):
        """Load data from TensorBoard event files."""
        if not HAS_TENSORBOARD:
            return False

        try:
            # Find the most recent event file
            event_files = list(Path(self.log_dir).glob('events.out.tfevents.*'))
            if not event_files:
                return False

            event_file = str(max(event_files, key=os.path.getmtime))

            # Load events
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()

            # Extract training metrics
            if 'train/loss_total' in ea.Tags()['scalars']:
                events = ea.Scalars('train/loss_total')
                self.data['step'] = [e.step for e in events]
                self.data['train_loss'] = [e.value for e in events]

            if 'train/loss_seg' in ea.Tags()['scalars']:
                events = ea.Scalars('train/loss_seg')
                self.data['train_loss_seg'] = [e.value for e in events]

            if 'train/loss_cls' in ea.Tags()['scalars']:
                events = ea.Scalars('train/loss_cls')
                self.data['train_loss_cls'] = [e.value for e in events]

            if 'train/lr' in ea.Tags()['scalars']:
                events = ea.Scalars('train/lr')
                self.data['lr'] = [e.value for e in events]

            # Extract validation metrics
            if 'val/iou' in ea.Tags()['scalars']:
                events = ea.Scalars('val/iou')
                self.data['epoch'] = [e.step for e in events]
                self.data['val_iou'] = [e.value for e in events]

            if 'val/dice' in ea.Tags()['scalars']:
                events = ea.Scalars('val/dice')
                self.data['val_dice'] = [e.value for e in events]

            if 'val/cls_acc' in ea.Tags()['scalars']:
                events = ea.Scalars('val/cls_acc')
                self.data['val_acc'] = [e.value for e in events]

            return True

        except Exception as e:
            print(f"Error loading logs: {e}")
            return False

    def update_plots(self, frame=None):
        """Update all plots with latest data."""
        self.load_logs()

        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()

        # Plot 1: Training Loss
        ax1 = self.axes[0, 0]
        if self.data['step'] and self.data['train_loss']:
            ax1.plot(self.data['step'], self.data['train_loss'], 'b-', linewidth=2, label='Total Loss')
            if self.data['train_loss_seg']:
                ax1.plot(self.data['step'], self.data['train_loss_seg'], 'g--', alpha=0.7, label='Seg Loss')
            if self.data['train_loss_cls']:
                ax1.plot(self.data['step'], self.data['train_loss_cls'], 'r--', alpha=0.7, label='Cls Loss')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Show current values
            if self.data['train_loss']:
                current_loss = self.data['train_loss'][-1]
                ax1.text(0.02, 0.98, f'Current: {current_loss:.4f}',
                        transform=ax1.transAxes, va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax1.text(0.5, 0.5, 'Waiting for training data...',
                    ha='center', va='center', transform=ax1.transAxes)

        # Plot 2: Learning Rate
        ax2 = self.axes[0, 1]
        if self.data['step'] and self.data['lr']:
            ax2.plot(self.data['step'], self.data['lr'], 'orange', linewidth=2)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)

            # Show current LR
            current_lr = self.data['lr'][-1]
            ax2.text(0.02, 0.98, f'Current: {current_lr:.2e}',
                    transform=ax2.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax2.text(0.5, 0.5, 'Waiting for LR data...',
                    ha='center', va='center', transform=ax2.transAxes)

        # Plot 3: Validation Metrics (IoU & Dice)
        ax3 = self.axes[1, 0]
        if self.data['epoch'] and self.data['val_iou']:
            ax3.plot(self.data['epoch'], self.data['val_iou'], 'b-o', linewidth=2, markersize=6, label='IoU')
            if self.data['val_dice']:
                ax3.plot(self.data['epoch'], self.data['val_dice'], 'g-s', linewidth=2, markersize=6, label='Dice')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Score')
            ax3.set_title('Segmentation Performance')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0, 1])

            # Show best values
            best_iou = max(self.data['val_iou'])
            best_dice = max(self.data['val_dice']) if self.data['val_dice'] else 0
            ax3.text(0.02, 0.98, f'Best IoU: {best_iou:.4f}\nBest Dice: {best_dice:.4f}',
                    transform=ax3.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        else:
            ax3.text(0.5, 0.5, 'Waiting for validation data...',
                    ha='center', va='center', transform=ax3.transAxes)

        # Plot 4: Classification Accuracy
        ax4 = self.axes[1, 1]
        if self.data['epoch'] and self.data['val_acc']:
            ax4.plot(self.data['epoch'], self.data['val_acc'], 'r-^', linewidth=2, markersize=6)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Classification Accuracy')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1])

            # Show best accuracy
            best_acc = max(self.data['val_acc'])
            ax4.text(0.02, 0.98, f'Best: {best_acc:.4f} ({best_acc*100:.1f}%)',
                    transform=ax4.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        else:
            ax4.text(0.5, 0.5, 'Waiting for validation data...',
                    ha='center', va='center', transform=ax4.transAxes)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    def start_live(self):
        """Start live updating visualization."""
        print(f"Monitoring: {self.log_dir}")
        print(f"Refresh interval: {self.refresh_interval}s")
        print("Close the plot window to stop monitoring.\n")

        ani = animation.FuncAnimation(
            self.fig,
            self.update_plots,
            interval=self.refresh_interval * 1000,
            cache_frame_data=False
        )
        plt.show()

    def save_snapshot(self, output_path):
        """Save current plots to file."""
        self.update_plots()
        self.fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize BrainTumNet training progress")
    parser.add_argument("--logdir", type=str, default="runs",
                       help="TensorBoard log directory or specific run directory")
    parser.add_argument("--refresh", type=int, default=5,
                       help="Refresh interval in seconds (default: 5)")
    parser.add_argument("--save", type=str, default=None,
                       help="Save snapshot to file instead of live view")
    parser.add_argument("--run", type=str, default=None,
                       help="Specific run name (e.g., braintumnet_quick_test_fold0)")
    args = parser.parse_args()

    # Determine log directory
    log_dir = args.logdir
    if args.run:
        log_dir = os.path.join(args.logdir, args.run)

    # If log_dir contains multiple runs, use the most recent
    if os.path.isdir(log_dir):
        subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir)
                   if os.path.isdir(os.path.join(log_dir, d))]
        if subdirs and not args.run:
            log_dir = max(subdirs, key=os.path.getmtime)
            print(f"Using most recent run: {os.path.basename(log_dir)}")

    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found: {log_dir}")
        print(f"\nAvailable runs in {args.logdir}:")
        if os.path.exists(args.logdir):
            for d in os.listdir(args.logdir):
                if os.path.isdir(os.path.join(args.logdir, d)):
                    print(f"  - {d}")
        return

    # Create visualizer
    viz = TrainingVisualizer(log_dir, args.refresh)

    if args.save:
        # Save snapshot mode
        viz.save_snapshot(args.save)
    else:
        # Live monitoring mode
        viz.start_live()


if __name__ == "__main__":
    main()
