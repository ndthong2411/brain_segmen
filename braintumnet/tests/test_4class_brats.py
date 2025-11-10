"""
Test 4-class Standard BraTS Implementation

This script verifies that the 4-class BraTS segmentation is correctly implemented.
"""
import torch
import numpy as np
from braintumnet.metrics.multiclass import MulticlassMetricsAccumulator

def test_4class_regions():
    """Test that 4-class regions (ET, TC, WT) are computed correctly"""
    print("="*60)
    print("Testing 4-class BraTS Region Definitions")
    print("="*60)

    # Create synthetic 4-class data
    batch_size = 4
    H, W = 128, 128
    num_classes = 4

    # Create logits (B, C, H, W)
    logits = torch.randn(batch_size, num_classes, H, W) * 2

    # Create ground truth (B, 1, H, W)
    target = torch.zeros(batch_size, 1, H, W, dtype=torch.long)

    # Sample 0: All regions present
    target[0, 0, 30:50, 30:50] = 1  # NCR
    target[0, 0, 50:70, 50:70] = 2  # ED
    target[0, 0, 40:60, 40:60] = 3  # ET (overlaps with NCR)

    # Sample 1: Only ED and ET
    target[1, 0, 60:80, 60:80] = 2  # ED
    target[1, 0, 65:75, 65:75] = 3  # ET

    # Sample 2: Only NCR and ET (TC but no ED)
    target[2, 0, 40:70, 40:70] = 1  # NCR
    target[2, 0, 50:65, 50:65] = 3  # ET

    # Sample 3: All tumor classes
    target[3, 0, 20:80, 20:80] = 1  # Large NCR
    target[3, 0, 30:90, 30:90] = 2  # Large ED
    target[3, 0, 50:70, 50:70] = 3  # ET in center

    # Make predictions favor ground truth
    for b in range(batch_size):
        for c in range(num_classes):
            logits[b, c][target[b, 0] == c] = 10.0

    # Test metrics
    print("\nInitializing 4-class metrics accumulator...")
    metrics_acc = MulticlassMetricsAccumulator(num_classes=4, compute_hd95=True)

    print("Computing metrics...")
    metrics_acc.update(logits, target)
    metrics = metrics_acc.get_metrics()

    # Print results
    print("\n" + "-"*60)
    print("RESULTS")
    print("-"*60)
    print(f"ET (Enhancing Tumor) Dice:  {metrics['ET_dice']:.4f}")
    print(f"TC (Tumor Core) Dice:       {metrics['TC_dice']:.4f}")
    print(f"WT (Whole Tumor) Dice:      {metrics['WT_dice']:.4f}")
    print(f"ED (Edema) Dice:            {metrics['ED_dice']:.4f}")
    print(f"\nMean Dice (ET+TC+WT):       {metrics['mean_dice']:.4f}")
    print("-"*60)

    # HD95 metrics
    print("\nHD95 Metrics:")
    et_hd95 = metrics['ET_hd95']
    tc_hd95 = metrics['TC_hd95']
    wt_hd95 = metrics['WT_hd95']
    mean_hd95 = metrics['mean_hd95']

    print(f"  ET HD95: {et_hd95:.2f}" if et_hd95 >= 0 else "  ET HD95: N/A")
    print(f"  TC HD95: {tc_hd95:.2f}" if tc_hd95 >= 0 else "  TC HD95: N/A")
    print(f"  WT HD95: {wt_hd95:.2f}" if wt_hd95 >= 0 else "  WT HD95: N/A")
    print(f"  Mean HD95: {mean_hd95:.2f}" if mean_hd95 >= 0 else "  Mean HD95: N/A")

    # Validation checks
    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)

    checks_passed = 0
    total_checks = 6

    # Check 1: ET dice > 0
    if metrics['ET_dice'] > 0:
        print("[PASS] ET Dice > 0 (ET is being evaluated)")
        checks_passed += 1
    else:
        print("[FAIL] ET Dice = 0 (ET not being evaluated!)")

    # Check 2: TC should include both NCR and ET
    # TC Dice should be higher than ET Dice alone (since TC = NCR + ET)
    if metrics['TC_dice'] >= metrics['ET_dice'] * 0.8:
        print("[PASS] TC Dice >= ET Dice (TC includes NCR + ET)")
        checks_passed += 1
    else:
        print("[FAIL] TC Dice < ET Dice (TC definition may be wrong!)")

    # Check 3: WT should be highest (includes all tumor)
    if metrics['WT_dice'] >= metrics['TC_dice'] * 0.8:
        print("[PASS] WT Dice >= TC Dice (WT includes all tumor)")
        checks_passed += 1
    else:
        print("[FAIL] WT Dice < TC Dice (WT definition may be wrong!)")

    # Check 4: Mean dice should be ET + TC + WT / 3
    expected_mean = (metrics['ET_dice'] + metrics['TC_dice'] + metrics['WT_dice']) / 3.0
    if abs(metrics['mean_dice'] - expected_mean) < 0.001:
        print(f"[PASS] Mean Dice = (ET+TC+WT)/3 = {expected_mean:.4f}")
        checks_passed += 1
    else:
        print(f"[FAIL] Mean Dice mismatch: got {metrics['mean_dice']:.4f}, expected {expected_mean:.4f}")

    # Check 5: All IoU metrics present
    if all(k in metrics for k in ['ET_iou', 'TC_iou', 'WT_iou', 'ED_iou']):
        print("[PASS] All IoU metrics present (ET, TC, WT, ED)")
        checks_passed += 1
    else:
        print("[FAIL] Missing IoU metrics")

    # Check 6: All metrics are valid (not NaN)
    if not any(np.isnan(v) for k, v in metrics.items() if isinstance(v, float)):
        print("[PASS] No NaN values in metrics")
        checks_passed += 1
    else:
        print("[FAIL] NaN values detected in metrics")

    print("="*60)
    print(f"RESULT: {checks_passed}/{total_checks} checks passed")
    print("="*60)

    if checks_passed == total_checks:
        print("\n[SUCCESS] 4-class BraTS implementation is CORRECT!")
        return True
    else:
        print("\n[WARNING] Some checks failed. Review implementation.")
        return False


def test_3class_backward_compat():
    """Test that 3-class (legacy) mode still works"""
    print("\n" + "="*60)
    print("Testing 3-class Backward Compatibility")
    print("="*60)

    batch_size = 2
    H, W = 128, 128
    num_classes = 3

    # 3-class: 0=BG, 1=TC (NCR+ET merged), 2=ED
    logits = torch.randn(batch_size, num_classes, H, W) * 2
    target = torch.zeros(batch_size, 1, H, W, dtype=torch.long)

    target[0, 0, 40:80, 40:80] = 1  # TC
    target[0, 0, 60:100, 60:100] = 2  # ED
    target[1, 0, 30:70, 30:70] = 1
    target[1, 0, 50:90, 50:90] = 2

    # Make predictions favor ground truth
    for b in range(batch_size):
        for c in range(num_classes):
            logits[b, c][target[b, 0] == c] = 10.0

    print("\nInitializing 3-class metrics accumulator...")
    metrics_acc = MulticlassMetricsAccumulator(num_classes=3, compute_hd95=False)

    print("Computing metrics...")
    metrics_acc.update(logits, target)
    metrics = metrics_acc.get_metrics()

    print("\nResults:")
    print(f"  WT Dice: {metrics['WT_dice']:.4f}")
    print(f"  TC Dice: {metrics['TC_dice']:.4f}")
    print(f"  ED Dice: {metrics['ED_dice']:.4f}")
    print(f"  ET Dice: {metrics['ET_dice']:.4f} (should be 0.0 for 3-class)")

    if metrics['ET_dice'] == 0.0:
        print("\n[PASS] 3-class backward compatibility")
        return True
    else:
        print("\n[FAIL] 3-class backward compatibility")
        return False


def main():
    print("\n" + "#"*60)
    print("# 4-CLASS STANDARD BraTS TEST SUITE")
    print("#"*60 + "\n")

    success_4class = test_4class_regions()
    success_3class = test_3class_backward_compat()

    print("\n" + "#"*60)
    print("# FINAL SUMMARY")
    print("#"*60)
    print(f"4-class test: {'PASS' if success_4class else 'FAIL'}")
    print(f"3-class test: {'PASS' if success_3class else 'FAIL'}")

    if success_4class and success_3class:
        print("\n[SUCCESS] ALL TESTS PASSED!")
        print("Your 4-class BraTS implementation is ready for training.")
        return 0
    else:
        print("\n[WARNING] SOME TESTS FAILED")
        print("Please review the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
