"""Test multiclass HD95 computation"""
import torch
import numpy as np
from src.braintumnet.multiclass_metrics import MulticlassMetricsAccumulator

def test_multiclass_hd95():
    print("Testing Multiclass HD95 computation...")

    # Create synthetic multiclass data
    # 3 classes: 0=background, 1=TC, 2=ED
    batch_size = 2
    num_classes = 3
    H, W = 128, 128

    # Create logits (B, C, H, W)
    logits = torch.randn(batch_size, num_classes, H, W) * 3

    # Make TC (class 1) high probability in region
    logits[0, 1, 40:80, 40:80] = 10.0
    logits[1, 1, 50:90, 50:90] = 10.0

    # Make ED (class 2) high probability in region
    logits[0, 2, 60:100, 60:100] = 10.0
    logits[1, 2, 70:110, 70:110] = 10.0

    # Create ground truth (B, 1, H, W) - integer labels
    target = torch.zeros(batch_size, 1, H, W, dtype=torch.long)

    # TC regions (class 1)
    target[0, 0, 42:82, 42:82] = 1
    target[1, 0, 52:92, 52:92] = 1

    # ED regions (class 2)
    target[0, 0, 62:102, 62:102] = 2
    target[1, 0, 72:112, 72:112] = 2

    print(f"Input shapes: logits={logits.shape}, target={target.shape}")
    print(f"Target unique values: {torch.unique(target)}")

    # Test with HD95 enabled
    print("\n=== Testing with HD95 enabled ===")
    metrics_acc = MulticlassMetricsAccumulator(num_classes=3, compute_hd95=True)
    metrics_acc.update(logits, target)

    metrics = metrics_acc.get_metrics()

    print("\nMetrics:")
    for key, value in metrics.items():
        if 'hd95' in key.lower():
            val_str = f"{value:.4f}" if value >= 0 else "N/A"
            print(f"  {key}: {val_str}")
        else:
            print(f"  {key}: {value:.4f}")

    # Test with HD95 disabled
    print("\n=== Testing with HD95 disabled ===")
    metrics_acc_no_hd95 = MulticlassMetricsAccumulator(num_classes=3, compute_hd95=False)
    metrics_acc_no_hd95.update(logits, target)

    metrics_no_hd95 = metrics_acc_no_hd95.get_metrics()

    print("\nMetrics (no HD95):")
    for key, value in metrics_no_hd95.items():
        if 'hd95' in key.lower():
            val_str = f"{value:.4f}" if value >= 0 else "N/A"
            print(f"  {key}: {val_str}")
        else:
            print(f"  {key}: {value:.4f}")

    print("\n=== Test completed successfully! ===")

    # Check that HD95 values are reasonable
    if metrics['mean_hd95'] >= 0:
        print(f"\nMean HD95: {metrics['mean_hd95']:.4f} - VALID")
    else:
        print(f"\nMean HD95: N/A - This means no valid samples (all empty masks)")

    return metrics

if __name__ == "__main__":
    test_multiclass_hd95()
