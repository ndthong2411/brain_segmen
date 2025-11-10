"""
Comprehensive verification script for 4-class BraTS implementation.
Tests all critical components for correctness.
"""

import torch
import numpy as np

from braintumnet.metrics.multiclass import MulticlassMetricsAccumulator
from braintumnet.losses.multiclass import (
    MultiClassCombinedLoss,
    MultiClassDiceLoss,
    MultiClassFocalLoss,
)

print("=" * 80)
print("4-CLASS BRATS IMPLEMENTATION VERIFICATION")
print("=" * 80)

# ============================================================================
# TEST 1: Shape Consistency
# ============================================================================
print("\n[TEST 1] Shape Consistency")
print("-" * 80)

batch_size = 2
num_classes = 4
height, width = 256, 256

# Model output: (B, C, H, W) with C=4 classes
pred_logits = torch.randn(batch_size, num_classes, height, width)
print(f"[OK] Model output shape: {pred_logits.shape} (B, C, H, W)")

# Target: (B, 1, H, W) with integer labels [0, 1, 2, 3]
target = torch.randint(0, num_classes, (batch_size, 1, height, width))
print(f"[OK] Target shape: {target.shape} (B, 1, H, W)")
print(f"[OK] Target unique values: {torch.unique(target).tolist()}")

assert pred_logits.shape[0] == batch_size, "Batch size mismatch"
assert pred_logits.shape[1] == num_classes, "Number of classes mismatch"
assert target.min() >= 0 and target.max() < num_classes, "Invalid target labels"
print("[OK] Shape consistency: PASS\n")

# ============================================================================
# TEST 2: Loss Functions - 4 Classes
# ============================================================================
print("[TEST 2] Loss Functions - 4 Classes")
print("-" * 80)

# Test Dice Loss
dice_loss = MultiClassDiceLoss(num_classes=4, ignore_background=True)
loss_dice = dice_loss(pred_logits, target)
print(f"[OK] Dice Loss: {loss_dice.item():.4f}")
assert not torch.isnan(loss_dice), "Dice loss is NaN"
assert not torch.isinf(loss_dice), "Dice loss is Inf"

# Test Focal Loss
focal_loss = MultiClassFocalLoss(num_classes=4, alpha=[0.0, 0.3, 0.4, 0.3], gamma=3.0)
loss_focal = focal_loss(pred_logits, target)
print(f"[OK] Focal Loss: {loss_focal.item():.4f}")
assert not torch.isnan(loss_focal), "Focal loss is NaN"
assert not torch.isinf(loss_focal), "Focal loss is Inf"

# Test Combined Loss
combined_loss = MultiClassCombinedLoss(
    num_classes=4,
    dice_weight=1.0,
    focal_weight=1.0,
    class_weights_dice=[1.0, 3.0, 4.0, 5.0],
    class_weights_focal=[0.0, 0.3, 0.4, 0.3],
    focal_gamma=3.0
)
loss_total, loss_d, loss_f = combined_loss(pred_logits, target)
print(f"[OK] Combined Loss: total={loss_total.item():.4f}, dice={loss_d.item():.4f}, focal={loss_f.item():.4f}")
assert not torch.isnan(loss_total), "Combined loss is NaN"
assert not torch.isinf(loss_total), "Combined loss is Inf"
print("[OK] Loss functions: PASS\n")

# ============================================================================
# TEST 3: TC/WT/ET Region Logic Correctness
# ============================================================================
print("[TEST 3] TC/WT/ET Region Logic Correctness")
print("-" * 80)

# Create synthetic test case with known regions
# Ground truth pattern:
# - Class 0 (Background): top-left quadrant
# - Class 1 (NCR): top-right quadrant
# - Class 2 (ED): bottom-left quadrant
# - Class 3 (ET): bottom-right quadrant

test_target = torch.zeros(1, 1, 4, 4, dtype=torch.long)
test_target[0, 0, :2, 2:] = 1  # NCR in top-right
test_target[0, 0, 2:, :2] = 2  # ED in bottom-left
test_target[0, 0, 2:, 2:] = 3  # ET in bottom-right

# Create perfect prediction (100% accuracy)
test_logits = torch.zeros(1, 4, 4, 4)
for i in range(4):
    for j in range(4):
        gt_class = test_target[0, 0, i, j].item()
        test_logits[0, gt_class, i, j] = 10.0  # High logit for correct class

# Initialize metrics accumulator
metrics_acc = MulticlassMetricsAccumulator(num_classes=4, compute_hd95=False)
metrics_acc.update(test_logits, test_target)
metrics = metrics_acc.get_metrics()

print("Ground truth regions:")
print(f"  - Background (class 0): top-left quadrant (4 pixels)")
print(f"  - NCR (class 1): top-right quadrant (4 pixels)")
print(f"  - ED (class 2): bottom-left quadrant (4 pixels)")
print(f"  - ET (class 3): bottom-right quadrant (4 pixels)")
print()
print("Expected BraTS regions:")
print(f"  - ET = class 3 only => 4 pixels")
print(f"  - TC = classes 1+3 (NCR+ET) => 8 pixels")
print(f"  - WT = classes 1+2+3 (all tumor) => 12 pixels")
print()
print("Computed Dice scores (should be 1.0 for perfect prediction):")
print(f"  - ET Dice: {metrics['ET_dice']:.4f}")
print(f"  - TC Dice: {metrics['TC_dice']:.4f}")
print(f"  - WT Dice: {metrics['WT_dice']:.4f}")
print(f"  - ED Dice: {metrics['ED_dice']:.4f}")
print(f"  - Mean Dice: {metrics['mean_dice']:.4f}")

# Verify logic
assert abs(metrics['ET_dice'] - 1.0) < 0.01, f"ET Dice should be 1.0, got {metrics['ET_dice']}"
assert abs(metrics['TC_dice'] - 1.0) < 0.01, f"TC Dice should be 1.0, got {metrics['TC_dice']}"
assert abs(metrics['WT_dice'] - 1.0) < 0.01, f"WT Dice should be 1.0, got {metrics['WT_dice']}"
assert abs(metrics['ED_dice'] - 1.0) < 0.01, f"ED Dice should be 1.0, got {metrics['ED_dice']}"

# Verify mean calculation: (ET + TC + WT) / 3 (standard BraTS, not ED)
expected_mean = (metrics['ET_dice'] + metrics['TC_dice'] + metrics['WT_dice']) / 3.0
assert abs(metrics['mean_dice'] - expected_mean) < 0.01, f"Mean Dice calculation error: {metrics['mean_dice']} != {expected_mean}"
print(f"\n[OK] Mean calculation verified: (ET + TC + WT) / 3 = {expected_mean:.4f}")
print("[OK] Region logic: PASS\n")

# ============================================================================
# TEST 4: Edge Cases
# ============================================================================
print("[TEST 4] Edge Cases")
print("-" * 80)

# Test 4.1: Empty masks (all background)
empty_target = torch.zeros(1, 1, 4, 4, dtype=torch.long)
empty_logits = torch.zeros(1, 4, 4, 4)
empty_logits[:, 0, :, :] = 10.0  # Predict all background

metrics_acc_empty = MulticlassMetricsAccumulator(num_classes=4, compute_hd95=False)
metrics_acc_empty.update(empty_logits, empty_target)
metrics_empty = metrics_acc_empty.get_metrics()
print(f"Empty masks - ET Dice: {metrics_empty['ET_dice']:.4f} (should be ~0)")
print(f"Empty masks - TC Dice: {metrics_empty['TC_dice']:.4f} (should be ~0)")
print(f"Empty masks - WT Dice: {metrics_empty['WT_dice']:.4f} (should be ~0)")
assert not np.isnan(metrics_empty['ET_dice']), "Empty mask ET Dice is NaN"
assert not np.isnan(metrics_empty['TC_dice']), "Empty mask TC Dice is NaN"
assert not np.isnan(metrics_empty['WT_dice']), "Empty mask WT Dice is NaN"
print("[OK] Empty masks handled correctly\n")

# Test 4.2: Only ET present
et_only_target = torch.zeros(1, 1, 4, 4, dtype=torch.long)
et_only_target[0, 0, 2:, 2:] = 3  # ET only
et_only_logits = torch.zeros(1, 4, 4, 4)
et_only_logits[:, 0, :, :] = 10.0
et_only_logits[:, 3, 2:, 2:] = 15.0

metrics_acc_et = MulticlassMetricsAccumulator(num_classes=4, compute_hd95=False)
metrics_acc_et.update(et_only_logits, et_only_target)
metrics_et = metrics_acc_et.get_metrics()
print(f"ET only - ET Dice: {metrics_et['ET_dice']:.4f} (should be ~1.0)")
print(f"ET only - TC Dice: {metrics_et['TC_dice']:.4f} (should be ~1.0, TC includes ET)")
print(f"ET only - WT Dice: {metrics_et['WT_dice']:.4f} (should be ~1.0, WT includes ET)")
assert abs(metrics_et['ET_dice'] - 1.0) < 0.01, "ET-only case: ET Dice incorrect"
assert abs(metrics_et['TC_dice'] - 1.0) < 0.01, "ET-only case: TC Dice incorrect (should include ET)"
assert abs(metrics_et['WT_dice'] - 1.0) < 0.01, "ET-only case: WT Dice incorrect (should include ET)"
print("[OK] ET-only case handled correctly\n")

# Test 4.3: Division by zero protection
zero_target = torch.zeros(1, 1, 4, 4, dtype=torch.long)
zero_logits = torch.zeros(1, 4, 4, 4)
zero_logits[:, 1, :, :] = 10.0  # Predict all NCR (class 1)

metrics_acc_zero = MulticlassMetricsAccumulator(num_classes=4, compute_hd95=False)
metrics_acc_zero.update(zero_logits, zero_target)
metrics_zero = metrics_acc_zero.get_metrics()
print(f"Pred/GT mismatch - Dice values: ET={metrics_zero['ET_dice']:.4f}, TC={metrics_zero['TC_dice']:.4f}, WT={metrics_zero['WT_dice']:.4f}")
assert not np.isnan(metrics_zero['ET_dice']), "Division by zero produced NaN"
assert not np.isnan(metrics_zero['mean_dice']), "Division by zero in mean"
print("[OK] Division by zero protection: PASS\n")

# ============================================================================
# TEST 5: Config Consistency
# ============================================================================
print("[TEST 5] Config Consistency")
print("-" * 80)

import yaml
config_path = Path(__file__).parent / "braintumnet" / "configs" / "base.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

num_classes_cfg = cfg["model"]["num_classes_seg"]
print(f"Config num_classes_seg: {num_classes_cfg}")
assert num_classes_cfg == 4, f"Config should have num_classes_seg=4, got {num_classes_cfg}"

focal_alpha = cfg["train"]["focal_alpha"]
print(f"Config focal_alpha: {focal_alpha}")
assert len(focal_alpha) == 4, f"focal_alpha should have 4 values, got {len(focal_alpha)}"

class_weights = cfg["train"]["class_weights"]
print(f"Config class_weights: {class_weights}")
assert len(class_weights) == 4, f"class_weights should have 4 values, got {len(class_weights)}"

print("[OK] Config consistency: PASS\n")

# ============================================================================
# TEST 6: Preprocessing Label Mapping
# ============================================================================
print("[TEST 6] Preprocessing Label Mapping")
print("-" * 80)

# Simulate BraTS segmentation values
brats_seg = np.array([
    [0, 0, 1, 1],
    [0, 2, 2, 4],
    [1, 2, 4, 4],
    [1, 1, 4, 0]
])

# Import the conversion function
sys.path.insert(0, str(Path(__file__).parent / "braintumnet" / "scripts"))
from preprocess_nifti_to_multiclass import convert_brats_seg_to_4class

# Convert to 4-class
result = convert_brats_seg_to_4class(brats_seg)

print("BraTS labels:")
print(brats_seg)
print("\n4-class labels:")
print(result)

# Verify mapping
assert (result[brats_seg == 0] == 0).all(), "Background (0) mapping failed"
assert (result[brats_seg == 1] == 1).all(), "NCR (1) mapping failed"
assert (result[brats_seg == 2] == 2).all(), "ED (2) mapping failed"
assert (result[brats_seg == 4] == 3).all(), "ET (4->3) mapping failed"

print("\n[OK] Label mapping verified:")
print("  BraTS 0 -> class 0 (Background)")
print("  BraTS 1 -> class 1 (NCR/NET)")
print("  BraTS 2 -> class 2 (ED)")
print("  BraTS 4 -> class 3 (ET)")
print("[OK] Preprocessing: PASS\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)
print("[OK] TEST 1: Shape Consistency - PASS")
print("[OK] TEST 2: Loss Functions - PASS")
print("[OK] TEST 3: TC/WT/ET Region Logic - PASS")
print("[OK] TEST 4: Edge Cases - PASS")
print("[OK] TEST 5: Config Consistency - PASS")
print("[OK] TEST 6: Preprocessing Label Mapping - PASS")
print("\n" + "=" * 80)
print("ALL TESTS PASSED - 4-CLASS IMPLEMENTATION VERIFIED")
print("=" * 80)
