"""
Logging utilities for training progress and metrics.
Provides file logging, console output, and structured logging.
"""
import os
import sys
import json
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    """
    Comprehensive logger for training process.
    Logs to both console and file with timestamps.
    """

    def __init__(self, log_dir, exp_name, fold, console=True):
        """
        Args:
            log_dir: Directory to save log files
            exp_name: Experiment name
            fold: Fold number
            console: Whether to also print to console
        """
        self.console = console
        self.exp_name = exp_name
        self.fold = fold

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{exp_name}_fold{fold}_{timestamp}.log")

        # Initialize log file
        self.start_time = datetime.now()
        self._write_header()

    def _write_header(self):
        """Write log file header."""
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BrainTumNet Training Log\n")
            f.write("=" * 80 + "\n")
            f.write(f"Experiment: {self.exp_name}\n")
            f.write(f"Fold: {self.fold}\n")
            f.write(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 80 + "\n\n")

    def log(self, message, level="INFO"):
        """
        Log a message with timestamp.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, SUCCESS)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"

        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(formatted_msg + "\n")

        # Print to console
        if self.console:
            print(formatted_msg)

    def info(self, message):
        """Log info message."""
        self.log(message, "INFO")

    def warning(self, message):
        """Log warning message."""
        self.log(message, "WARNING")

    def error(self, message):
        """Log error message."""
        self.log(message, "ERROR")

    def success(self, message):
        """Log success message."""
        self.log(message, "SUCCESS")

    def section(self, title):
        """Log a section header."""
        with open(self.log_file, 'a') as f:
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"{title}\n")
            f.write("-" * 80 + "\n")

        if self.console:
            print("\n" + "-" * 80)
            print(f"{title}")
            print("-" * 80)

    def epoch_start(self, epoch, total_epochs, phase="TRAIN"):
        """Log epoch start."""
        msg = f"Epoch {epoch+1}/{total_epochs} - {phase}"
        self.section(msg)

    def epoch_end(self, epoch, total_epochs, metrics, phase="TRAIN"):
        """
        Log epoch end with metrics.

        Args:
            epoch: Current epoch
            total_epochs: Total epochs
            metrics: Dictionary of metrics
            phase: TRAIN or VALIDATION
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        msg = f"[{timestamp}] Epoch {epoch+1}/{total_epochs} - {phase} - {metrics_str}"

        with open(self.log_file, 'a') as f:
            f.write(msg + "\n")

        if self.console:
            print(msg)

    def best_checkpoint(self, metric_name, metric_value, epoch):
        """Log new best checkpoint."""
        msg = f"*** NEW BEST {metric_name.upper()}: {metric_value:.4f} (epoch {epoch+1}) - Checkpoint saved ***"
        self.success(msg)

    def save_config(self, config, config_path):
        """
        Save configuration to log directory.

        Args:
            config: Configuration dictionary
            config_path: Original config file path
        """
        import shutil
        import yaml

        # Copy original config
        log_dir = os.path.dirname(self.log_file)
        config_copy = os.path.join(log_dir, f"config_fold{self.fold}.yaml")
        shutil.copy(config_path, config_copy)

        # Also save as JSON for easy parsing
        config_json = os.path.join(log_dir, f"config_fold{self.fold}.json")
        with open(config_json, 'w') as f:
            json.dump(config, f, indent=2)

        self.info(f"Configuration saved to: {config_copy}")
        self.info(f"Configuration (JSON) saved to: {config_json}")

    def training_summary(self, best_metrics, total_time):
        """
        Log training summary.

        Args:
            best_metrics: Dictionary of best metrics
            total_time: Total training time in seconds
        """
        with open(self.log_file, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("Training Complete!\n")
            f.write("=" * 80 + "\n")

            # Format time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            f.write(f"Total Time: {hours}h {minutes}m {seconds}s\n")

            # Best metrics
            f.write("\nBest Metrics:\n")
            for metric, value in best_metrics.items():
                if isinstance(value, tuple):
                    val, ep = value
                    f.write(f"  {metric}: {val:.4f} (epoch {ep+1})\n")
                else:
                    f.write(f"  {metric}: {value:.4f}\n")

            f.write("\nLog file: " + self.log_file + "\n")
            f.write("=" * 80 + "\n")

        if self.console:
            print("\n" + "=" * 80)
            print("Training Complete!")
            print("=" * 80)
            print(f"Total Time: {hours}h {minutes}m {seconds}s")
            print("\nBest Metrics:")
            for metric, value in best_metrics.items():
                if isinstance(value, tuple):
                    val, ep = value
                    print(f"  {metric}: {val:.4f} (epoch {ep+1})")
                else:
                    print(f"  {metric}: {value:.4f}")
            print(f"\nLog saved to: {self.log_file}")
            print("=" * 80)

    def close(self):
        """Close logger."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        self.training_summary({}, duration)
