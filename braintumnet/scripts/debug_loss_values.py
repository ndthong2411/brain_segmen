#!/usr/bin/env python3
"""
Debug Loss Values - Understand Why Loss Can Be Negative

This script simulates different IoU scenarios and shows how each loss component behaves.
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, 'src')

from braintumnet.losses_multiclass import MultiClassDiceLoss, MultiClassFocalLoss
from braintumnet.losses_iou import MulticlassIoULoss
from braintumnet.losses_boundary import BoundaryLoss


def create_synthetic_data(iou_level='low'):
    """
    Create synthetic predictions and targets with specific IoU level.

    Args:
        iou_level: 'low' (IoU~0.3), 'medium' (IoU~0.6), 'high' (IoU~0.9)

    Returns:
        logits: (1, 3, 64, 64) raw predictions
        target: (1, 64, 64) ground truth
    """
    B, H, W = 1, 64, 64
    C = 3  # classes: bg, TC, ED

    # Create ground truth: center region is TC (class 1), outer ring is ED (class 2)
    target = torch.zeros(B, H, W, dtype=torch.long)
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    dist = torch.sqrt((x - H//2)**2 + (y - W//2)**2)

    # TC: radius 10-15
    target[(dist >= 10) & (dist < 15)] = 1  # TC
    # ED: radius 15-25
    target[(dist >= 15) & (dist < 25)] = 2  # ED
    # Rest is background (0)

    # Create predictions based on IoU level
    logits = torch.zeros(B, C, H, W)

    if iou_level == 'low':
        # Poor prediction: very different from target
        pred_target = torch.zeros(B, H, W, dtype=torch.long)
        pred_target[(dist >= 8) & (dist < 12)] = 1  # TC shifted
        pred_target[(dist >= 18) & (dist < 28)] = 2  # ED shifted

    elif iou_level == 'medium':
        # Medium prediction: some overlap
        pred_target = torch.zeros(B, H, W, dtype=torch.long)
        pred_target[(dist >= 9) & (dist < 16)] = 1  # TC overlaps ~60%
        pred_target[(dist >= 14) & (dist < 26)] = 2  # ED overlaps ~60%

    elif iou_level == 'high':
        # Good prediction: very close to target
        pred_target = torch.zeros(B, H, W, dtype=torch.long)
        pred_target[(dist >= 10) & (dist < 15)] = 1  # TC exact
        pred_target[(dist >= 15) & (dist < 25)] = 2  # ED exact

    elif iou_level == 'perfect':
        # Perfect prediction
        pred_target = target.clone()

    # Convert pred_target to logits with high confidence
    for b in range(B):
        for c in range(C):
            logits[b, c] = (pred_target[b] == c).float() * 10.0  # High logit value

    return logits, target


def analyze_loss_components(iou_level='medium'):
    """Analyze each loss component for a given IoU level."""

    print(f"\n{'='*80}")
    print(f"SCENARIO: {iou_level.upper()} IoU")
    print(f"{'='*80}\n")

    # Create data
    logits, target = create_synthetic_data(iou_level)

    # Calculate actual IoU for reference
    pred_probs = F.softmax(logits, dim=1)
    pred_classes = pred_probs.argmax(dim=1)

    # Calculate IoU per class
    iou_values = {}
    for c in range(1, 3):  # Skip background
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        iou_c = (intersection / (union + 1e-6)).item()
        iou_values[c] = iou_c

    mean_iou = sum(iou_values.values()) / len(iou_values)

    print(f"📊 Actual IoU:")
    print(f"  TC IoU (class 1): {iou_values[1]:.4f}")
    print(f"  ED IoU (class 2): {iou_values[2]:.4f}")
    print(f"  Mean IoU:         {mean_iou:.4f}")
    print()

    # Config matching your training
    config = {
        'num_classes': 3,
        'ignore_background': True,
        'class_weights': [1.0, 3.0, 2.0],  # [bg, TC, ED]
        'focal_alpha': [0.0, 0.4, 0.1],    # [bg, TC, ED]
        'focal_gamma': 3.0,
        'dice_weight': 1.0,
        'focal_weight': 1.0,
        'iou_weight': 2.0,
        'boundary_weight': 0.5
    }

    # Initialize loss functions
    dice_loss_fn = MultiClassDiceLoss(
        num_classes=config['num_classes'],
        ignore_background=config['ignore_background'],
        class_weights=config['class_weights']
    )

    focal_loss_fn = MultiClassFocalLoss(
        num_classes=config['num_classes'],
        alpha=config['focal_alpha'],
        gamma=config['focal_gamma'],
        ignore_background=config['ignore_background']
    )

    iou_loss_fn = MulticlassIoULoss(
        num_classes=config['num_classes'],
        ignore_background=config['ignore_background'],
        class_weights=config['class_weights']
    )

    boundary_loss_fn = BoundaryLoss(
        ignore_background=config['ignore_background']
    )

    # Compute losses
    print(f"🔬 Loss Components (BEFORE weighting):")
    print(f"-" * 80)

    dice_loss = dice_loss_fn(logits, target).item()
    print(f"  Dice Loss:      {dice_loss:>10.4f}  (formula: 1 - dice_score)")
    print(f"                              → dice_score = {1 - dice_loss:.4f}")

    focal_loss = focal_loss_fn(logits, target).item()
    print(f"  Focal Loss:     {focal_loss:>10.4f}  (handles class imbalance)")

    iou_loss = iou_loss_fn(logits, target).item()
    print(f"  IoU Loss:       {iou_loss:>10.4f}  (formula: 1 - IoU)")
    print(f"                              → IoU = {1 - iou_loss:.4f}")

    boundary_loss = boundary_loss_fn(logits, target).item()
    print(f"  Boundary Loss:  {boundary_loss:>10.4f}  (Hausdorff distance)")

    print(f"\n⚖️  Weighted Loss Components (what gets summed):")
    print(f"-" * 80)

    weighted_dice = config['dice_weight'] * dice_loss
    weighted_focal = config['focal_weight'] * focal_loss
    weighted_iou = config['iou_weight'] * iou_loss
    weighted_boundary = config['boundary_weight'] * boundary_loss

    print(f"  Dice     × {config['dice_weight']:.1f}  = {weighted_dice:>10.4f}")
    print(f"  Focal    × {config['focal_weight']:.1f}  = {weighted_focal:>10.4f}")
    print(f"  IoU      × {config['iou_weight']:.1f}  = {weighted_iou:>10.4f}")
    print(f"  Boundary × {config['boundary_weight']:.1f}  = {weighted_boundary:>10.4f}")
    print(f"  " + "-" * 40)

    total_loss = weighted_dice + weighted_focal + weighted_iou + weighted_boundary
    print(f"  TOTAL LOSS      = {total_loss:>10.4f}")

    if total_loss < 0:
        print(f"\n  ⚠️  TOTAL LOSS IS NEGATIVE!")
    else:
        print(f"\n  ✅ Total loss is positive")

    print()
    return total_loss, mean_iou


def main():
    print("\n" + "="*80)
    print("🔍 LOSS DEBUG TOOL - Understanding Negative Loss")
    print("="*80)

    scenarios = [
        ('low', 'IoU ~0.3 (early training)'),
        ('medium', 'IoU ~0.6 (mid training)'),
        ('high', 'IoU ~0.9 (late training)'),
        ('perfect', 'IoU ~1.0 (perfect prediction)')
    ]

    results = []

    for scenario, description in scenarios:
        total_loss, mean_iou = analyze_loss_components(scenario)
        results.append((scenario, total_loss, mean_iou))

    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"\n{'Scenario':<15} | {'Mean IoU':>10} | {'Total Loss':>12} | {'Status'}")
    print("-" * 80)

    for scenario, loss, iou in results:
        status = "✅ Positive" if loss >= 0 else "⚠️  NEGATIVE"
        print(f"{scenario:<15} | {iou:>10.4f} | {loss:>12.4f} | {status}")

    print("\n" + "="*80)
    print("💡 KEY INSIGHTS:")
    print("="*80)
    print("""
1. DICE LOSS: Always 0 ≤ dice_loss ≤ 1
   - dice_loss = 1 - dice_score
   - When dice_score = 1.0 (perfect), dice_loss = 0.0

2. FOCAL LOSS: Always focal_loss ≥ 0
   - Handles class imbalance
   - Decreases as predictions improve

3. IoU LOSS: Always 0 ≤ iou_loss ≤ 1
   - iou_loss = 1 - IoU
   - When IoU = 1.0 (perfect), iou_loss = 0.0

4. BOUNDARY LOSS: Always boundary_loss ≥ 0
   - Measures Hausdorff distance
   - Decreases as boundaries get more precise

5. TOTAL LOSS = 1.0×dice + 1.0×focal + 2.0×iou + 0.5×boundary

   ⚠️  CÓ THỂ ÂM vì:
   - Không có thành phần nào THỰC SỰ âm
   - Nhưng nếu có BUG trong code hoặc:
   - Numerical instability (chia cho số rất nhỏ)
   - Smoothing term gây overflow
   - Gradient accumulation artifacts

6. TRONG THỰC TẾ:
   - Nếu loss âm → CÓ VẤN ĐỀ trong implementation
   - Cần kiểm tra lại code loss functions
   - Hoặc có numerical issue (NaN, Inf)

7. NHƯNG val_iou LẠ TỐT (0.70+):
   - Model đang train tốt BẤT CHẤP loss âm
   - Có thể là cách log loss có vấn đề
   - Hoặc loss_dict return sai
""")

    print("\n" + "="*80)
    print("🔧 RECOMMENDED ACTIONS:")
    print("="*80)
    print("""
1. Add debug logging to see EACH component separately:
   - Log dice_loss, focal_loss, iou_loss, boundary_loss
   - Check if ANY component is negative

2. Add assertions in loss functions:
   - assert dice_loss >= 0
   - assert focal_loss >= 0
   - assert iou_loss >= 0
   - assert boundary_loss >= 0

3. Check for NaN/Inf:
   - torch.isnan(loss).any()
   - torch.isinf(loss).any()

4. For NOW: Use val_iou as the main metric!
   - val_iou = 0.70+ → Training is WORKING
   - Ignore the negative train_loss
""")


if __name__ == "__main__":
    main()
