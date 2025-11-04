"""
Metrics logging and export utilities.
Saves training metrics to CSV and JSON for easy analysis.
"""
import os
import csv
import json
from pathlib import Path


class MetricsLogger:
    """
    Logger for training metrics with CSV/JSON export.
    Tracks metrics across epochs for analysis and plotting.
    """

    def __init__(self, log_dir, exp_name, fold):
        """
        Args:
            log_dir: Directory to save metric files
            exp_name: Experiment name
            fold: Fold number
        """
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.fold = fold

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Metric storage
        self.metrics_history = []
        self.best_metrics = {}

        # File paths
        self.csv_path = os.path.join(log_dir, f"metrics_{exp_name}_fold{fold}.csv")
        self.json_path = os.path.join(log_dir, f"metrics_{exp_name}_fold{fold}.json")

        # CSV file initialization
        self.csv_initialized = False
        self.csv_headers = None

    def log_epoch(self, epoch, metrics_dict):
        """
        Log metrics for one epoch.

        Args:
            epoch: Epoch number
            metrics_dict: Dictionary of metric_name -> value
        """
        # Add epoch number
        metrics_dict['epoch'] = epoch

        # Store in history
        self.metrics_history.append(metrics_dict.copy())

        # Update best metrics
        for key, value in metrics_dict.items():
            if key == 'epoch':
                continue

            # For loss, lower is better; for others, higher is better
            if 'loss' in key.lower():
                if key not in self.best_metrics or value < self.best_metrics[key][0]:
                    self.best_metrics[key] = (value, epoch)
            else:
                if key not in self.best_metrics or value > self.best_metrics[key][0]:
                    self.best_metrics[key] = (value, epoch)

        # Write to CSV
        self._write_csv(metrics_dict)

    def _write_csv(self, metrics_dict):
        """Write metrics to CSV file."""
        # Initialize CSV headers on first write
        if not self.csv_initialized:
            self.csv_headers = sorted(metrics_dict.keys())
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                writer.writeheader()
            self.csv_initialized = True

        # Append metrics
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writerow(metrics_dict)

    def save_json(self):
        """Save all metrics to JSON file."""
        output = {
            'experiment': self.exp_name,
            'fold': self.fold,
            'history': self.metrics_history,
            'best_metrics': {k: {'value': v[0], 'epoch': v[1]}
                            for k, v in self.best_metrics.items()}
        }

        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

    def get_best_metrics(self):
        """Get dictionary of best metrics."""
        return self.best_metrics.copy()

    def get_history(self):
        """Get full metrics history."""
        return self.metrics_history.copy()

    def close(self):
        """Close logger and save final JSON."""
        self.save_json()

    def print_summary(self):
        """Print summary of best metrics."""
        print("\n" + "=" * 60)
        print("Best Metrics Summary")
        print("=" * 60)
        for metric, (value, epoch) in sorted(self.best_metrics.items()):
            print(f"{metric:20s}: {value:.4f} (epoch {epoch+1})")
        print("=" * 60)
        print(f"CSV saved to: {self.csv_path}")
        print(f"JSON saved to: {self.json_path}")
        print("=" * 60 + "\n")
