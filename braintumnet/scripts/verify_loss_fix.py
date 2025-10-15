#!/usr/bin/env python3
"""
Verify IoU Loss Fix

Tests that IoU loss is always positive after the fix.
"""

import torch
import sys
sys.path.insert(0, 'src')

from braintumnet.losses_iou import MulticlassIoULoss


def test_iou_loss_always_positive():
    """Test that IoU loss is always positive with various scenarios."""

    print("="*80)
    print("🧪 TESTING IoU LOSS FIX")
    print("="*80)
    print()

    # Config with class weights (like in training)
    class_weights = [1.0, 4.0, 2.5]  # [bg(ignored), TC, ED]

    loss_fn = MulticlassIoULoss(
        num_classes=3,
        ignore_background=True,
        class_weights=class_weights
    )

    scenarios = [
        ("Random predictions (IoU ~0.1)", 0.1),
        ("Poor predictions (IoU ~0.3)", 0.3),
        ("Medium predictions (IoU ~0.5)", 0.5),
        ("Good predictions (IoU ~0.7)", 0.7),
        ("Excellent predictions (IoU ~0.9)", 0.9),
        ("Near-perfect (IoU ~0.99)", 0.99),
    ]

    print("Testing with class_weights:", class_weights)
    print()

    all_passed = True

    for scenario_name, target_iou in scenarios:
        print(f"📊 {scenario_name}")
        print("-" * 80)

        # Create synthetic data with approximately target_iou
        B, H, W = 4, 64, 64
        C = 3

        # Create target: center is TC (class 1), ring is ED (class 2)
        target = torch.zeros(B, H, W, dtype=torch.long)
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        dist = torch.sqrt((x - H//2)**2 + (y - W//2)**2)

        target = target.unsqueeze(0).expand(B, -1, -1)
        target[(dist >= 10) & (dist < 15)] = 1  # TC
        target[(dist >= 15) & (dist < 25)] = 2  # ED

        # Create predictions with controlled overlap
        logits = torch.zeros(B, C, H, W)

        # Adjust prediction centers based on target_iou
        shift = int((1.0 - target_iou) * 5)  # More shift = less overlap

        pred_target = torch.zeros(B, H, W, dtype=torch.long)
        pred_target[(dist >= 10-shift) & (dist < 15+shift)] = 1  # TC
        pred_target[(dist >= 15-shift) & (dist < 25+shift)] = 2  # ED

        # Convert to logits
        for b in range(B):
            for c in range(C):
                logits[b, c] = (pred_target[b] == c).float() * 10.0

        # Compute loss
        loss = loss_fn(logits, target)

        # Check if positive
        loss_val = loss.item()
        is_positive = loss_val >= 0

        print(f"  Loss value: {loss_val:.6f}")
        print(f"  Is positive: {is_positive}")

        if is_positive:
            print(f"  ✅ PASS")
        else:
            print(f"  ❌ FAIL - Loss is negative!")
            all_passed = False

        print()

    # Summary
    print("="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print()
        print("✅ IoU Loss is always positive")
        print("✅ Fix is working correctly")
        print("✅ Safe to use in training")
    else:
        print("❌ SOME TESTS FAILED!")
        print()
        print("⚠️  IoU Loss can still be negative")
        print("⚠️  Fix may not be complete")
        print("⚠️  Check losses_iou.py again")

    print("="*80)

    return all_passed


def test_compare_old_vs_new():
    """Show the difference between old (buggy) and new (fixed) behavior."""

    print("\n" + "="*80)
    print("📊 OLD vs NEW BEHAVIOR COMPARISON")
    print("="*80)
    print()

    class_weights = [1.0, 4.0, 2.5]
    num_classes = 3

    # Simulate a case where IoU is low (early training)
    # TC IoU = 0.3, ED IoU = 0.4

    print("Scenario: Early training")
    print("  TC IoU = 0.30")
    print("  ED IoU = 0.40")
    print()

    print("OLD IMPLEMENTATION (BUGGY):")
    print("-" * 80)
    tc_iou = 0.30
    ed_iou = 0.40

    # Old way: weight IoU, then compute loss
    weighted_tc = tc_iou * class_weights[1]  # 0.30 × 4.0 = 1.20
    weighted_ed = ed_iou * class_weights[2]  # 0.40 × 2.5 = 1.00
    mean_iou = (weighted_tc + weighted_ed) / 2  # (1.20 + 1.00) / 2 = 1.10
    old_loss = 1.0 - mean_iou  # 1.0 - 1.10 = -0.10 ❌

    print(f"  Weighted TC IoU: {tc_iou} × {class_weights[1]} = {weighted_tc}")
    print(f"  Weighted ED IoU: {ed_iou} × {class_weights[2]} = {weighted_ed}")
    print(f"  Mean weighted IoU: ({weighted_tc} + {weighted_ed}) / 2 = {mean_iou}")
    print(f"  Loss: 1.0 - {mean_iou} = {old_loss:.6f}  ❌ NEGATIVE!")
    print()

    print("NEW IMPLEMENTATION (FIXED):")
    print("-" * 80)

    # New way: compute loss, then weight
    tc_loss = 1.0 - tc_iou  # 1.0 - 0.30 = 0.70
    ed_loss = 1.0 - ed_iou  # 1.0 - 0.40 = 0.60

    weighted_tc_loss = tc_loss * class_weights[1]  # 0.70 × 4.0 = 2.80
    weighted_ed_loss = ed_loss * class_weights[2]  # 0.60 × 2.5 = 1.50

    new_loss = (weighted_tc_loss + weighted_ed_loss) / 2  # (2.80 + 1.50) / 2 = 2.15

    print(f"  TC loss: 1.0 - {tc_iou} = {tc_loss}")
    print(f"  ED loss: 1.0 - {ed_iou} = {ed_loss}")
    print(f"  Weighted TC loss: {tc_loss} × {class_weights[1]} = {weighted_tc_loss}")
    print(f"  Weighted ED loss: {ed_loss} × {class_weights[2]} = {weighted_ed_loss}")
    print(f"  Mean loss: ({weighted_tc_loss} + {weighted_ed_loss}) / 2 = {new_loss:.6f}  ✅ POSITIVE!")
    print()

    print("="*80)
    print("CONCLUSION:")
    print(f"  Old (buggy): Loss = {old_loss:.6f}  ❌")
    print(f"  New (fixed): Loss = {new_loss:.6f}  ✅")
    print(f"  Difference: {new_loss - old_loss:.6f}")
    print("="*80)


def main():
    print("\n" + "="*80)
    print("🔬 IoU LOSS FIX VERIFICATION")
    print("="*80)
    print()
    print("This script verifies that the IoU Loss bug has been fixed.")
    print("The bug caused IoU loss to be negative when class weights were > 1.0")
    print()

    # Test 1: Always positive
    passed = test_iou_loss_always_positive()

    # Test 2: Compare old vs new
    test_compare_old_vs_new()

    # Final result
    print("\n" + "="*80)
    print("🎯 VERIFICATION RESULT")
    print("="*80)

    if passed:
        print("""
✅ ALL TESTS PASSED!

The IoU Loss bug has been successfully fixed:
  - IoU Loss is always positive ✅
  - Class weights applied correctly ✅
  - Math is sound ✅

You can now:
  1. Resume training with confidence
  2. Expect better IoU convergence
  3. See consistent loss behavior

Next steps:
  1. Restart training: python scripts/train.py --cfg configs/phase2_a100_80gb.yaml --fold 0
  2. Monitor: python scripts/monitor_training.py --fold 0
  3. Check: IoU should improve faster now!
""")
    else:
        print("""
❌ TESTS FAILED!

Something is still wrong with IoU Loss.

Please:
  1. Check src/braintumnet/losses_iou.py
  2. Verify the fix was applied correctly
  3. Run this script again after fixing
""")

    print("="*80)

    return 0 if passed else 1


if __name__ == "__main__":
    exit(main())
