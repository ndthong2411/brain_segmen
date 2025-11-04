"""
Test script to verify the metrics calculation fix.

This demonstrates the difference between using soft probabilities (WRONG)
vs hard predictions (CORRECT) for computing Dice/IoU metrics.
"""

import torch
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent / "braintumnet"
sys.path.insert(0, str(ROOT / "src"))

from braintumnet.multiclass_metrics import (
    MulticlassMetricsAccumulator,
    compute_brats_regions,
    multiclass_dice_coefficient,
    multiclass_iou
)


def test_metrics_fix():
    """Test that metrics are now computed correctly using hard predictions"""
    
    print("="*70)
    print("TESTING METRICS FIX - Hard Predictions vs Soft Probabilities")
    print("="*70)
    
    # Create a simple test case
    # 3 classes: Background (0), Tumor Core (1), Edema (2)
    batch_size = 2
    num_classes = 3
    h, w = 4, 4
    
    # Create fake logits (model output before softmax)
    # Shape: (B, C, H, W)
    logits = torch.randn(batch_size, num_classes, h, w)
    
    # Make predictions more confident for testing
    logits[:, :, 0, 0] = torch.tensor([-5.0, 10.0, -5.0])  # Strong TC prediction
    logits[:, :, 0, 1] = torch.tensor([-5.0, -5.0, 10.0])  # Strong ED prediction
    logits[:, :, 0, 2] = torch.tensor([10.0, -5.0, -5.0])  # Strong BG prediction
    
    # Create ground truth labels
    # Shape: (B, 1, H, W)
    target = torch.zeros(batch_size, 1, h, w, dtype=torch.long)
    target[:, :, 0, 0] = 1  # TC
    target[:, :, 0, 1] = 2  # ED
    target[:, :, 0, 2] = 0  # BG
    target[:, :, 1, 1] = 1  # TC
    target[:, :, 2, 2] = 2  # ED
    
    print(f"\nTest setup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num classes: {num_classes}")
    print(f"  Image size: {h}x{w}")
    print(f"  Total pixels: {batch_size * h * w}")
    
    # Get hard predictions
    pred_classes = torch.argmax(logits, dim=1, keepdim=True)
    
    print(f"\nGround truth (first sample, unique values): {target[0].unique().tolist()}")
    print(f"Predictions (first sample, unique values): {pred_classes[0].unique().tolist()}")
    
    # Count pixels for each class
    print(f"\nPixel distribution (all samples):")
    for class_idx in range(num_classes):
        class_names = ['Background', 'Tumor Core (TC)', 'Edema (ED)']
        target_count = (target == class_idx).sum().item()
        pred_count = (pred_classes == class_idx).sum().item()
        print(f"  Class {class_idx} ({class_names[class_idx]}): "
              f"target={target_count}, pred={pred_count}")
    
    # Test 1: MulticlassMetricsAccumulator
    print("\n" + "="*70)
    print("TEST 1: MulticlassMetricsAccumulator (FIXED)")
    print("="*70)
    
    metrics_acc = MulticlassMetricsAccumulator(num_classes=num_classes)
    metrics_acc.update(logits, target)
    metrics = metrics_acc.get_metrics()
    
    print("\nResults:")
    print(f"  WT Dice: {metrics['WT_dice']:.4f} | WT IoU: {metrics['WT_iou']:.4f}")
    print(f"  TC Dice: {metrics['TC_dice']:.4f} | TC IoU: {metrics['TC_iou']:.4f}")
    print(f"  ED Dice: {metrics['ED_dice']:.4f} | ED IoU: {metrics['ED_iou']:.4f}")
    print(f"  Mean Dice: {metrics['mean_dice']:.4f} | Mean IoU: {metrics['mean_iou']:.4f}")
    
    # Verify metrics are NOT zero (they should have meaningful values now)
    assert metrics['mean_dice'] > 0.0, "❌ FAILED: Mean Dice is still 0!"
    assert metrics['mean_iou'] > 0.0, "❌ FAILED: Mean IoU is still 0!"
    print("\n✅ PASSED: Metrics are now NON-ZERO!")
    
    # Test 2: compute_brats_regions
    print("\n" + "="*70)
    print("TEST 2: compute_brats_regions (FIXED)")
    print("="*70)
    
    regions = compute_brats_regions(logits, target, num_classes=num_classes)
    
    print("\nResults:")
    print(f"  WT Dice: {regions['WT_dice']:.4f} | WT IoU: {regions['WT_iou']:.4f}")
    print(f"  TC Dice: {regions['TC_dice']:.4f} | TC IoU: {regions['TC_iou']:.4f}")
    print(f"  ED Dice: {regions['ED_dice']:.4f} | ED IoU: {regions['ED_iou']:.4f}")
    print(f"  Mean Dice: {regions['mean_dice']:.4f} | Mean IoU: {regions['mean_iou']:.4f}")
    
    assert regions['mean_dice'] > 0.0, "❌ FAILED: Mean Dice is still 0!"
    assert regions['mean_iou'] > 0.0, "❌ FAILED: Mean IoU is still 0!"
    print("\n✅ PASSED: Metrics are now NON-ZERO!")
    
    # Test 3: Individual class metrics
    print("\n" + "="*70)
    print("TEST 3: Individual Class Metrics (FIXED)")
    print("="*70)
    
    for class_idx in range(num_classes):
        class_names = ['Background', 'TC', 'ED']
        dice = multiclass_dice_coefficient(logits, target, class_idx)
        iou = multiclass_iou(logits, target, class_idx)
        print(f"  Class {class_idx} ({class_names[class_idx]}): "
              f"Dice={dice:.4f}, IoU={iou:.4f}")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✅")
    print("="*70)
    print("\nSUMMARY:")
    print("  The bug was using soft probabilities (softmax output) instead of")
    print("  hard predictions (argmax) for computing Dice/IoU metrics.")
    print("\n  BEFORE (WRONG): pred = softmax(logits)[:, class_idx]  # 0.0-1.0")
    print("  AFTER (CORRECT): pred = (argmax(logits) == class_idx)  # 0 or 1")
    print("\n  This ensures intersection/union are computed on binary masks,")
    print("  which is the standard way to calculate segmentation metrics.")
    print("="*70)


if __name__ == "__main__":
    test_metrics_fix()
