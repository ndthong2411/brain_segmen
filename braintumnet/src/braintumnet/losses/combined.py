"""
Ultimate Combined Loss for IoU 0.90 Target

This module combines all loss components optimized for reaching IoU 0.90:
1. Dice Loss - Region overlap optimization
2. Focal Loss - Hard example focus and class imbalance
3. IoU Loss - Direct IoU metric optimization
4. Boundary Loss - Precise boundary delineation

The combination directly targets the weaknesses identified in baseline:
- IoU loss fixes "optimizing wrong metric" issue
- Boundary loss fixes boundary errors that hurt IoU
- Focal loss handles class imbalance (TC is hardest)
- Dice loss provides stable base optimization

Author: BrainTumNet V2.0 Upgrade
Date: 2025-01-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UltimateLoss(nn.Module):
    """
    Ultimate Combined Loss: Dice + Focal + IoU + Boundary

    This is the Phase 1 loss function targeting IoU 0.75-0.80.

    Loss = w_dice * Dice + w_focal * Focal + w_iou * IoU + w_boundary * Boundary

    Recommended weights (optimized for IoU):
    - dice_weight: 1.0 (baseline stability)
    - focal_weight: 1.0 (hard examples)
    - iou_weight: 2.0 (⭐ emphasize target metric)
    - boundary_weight: 0.5 (precision without overwhelming)

    Args:
        num_classes: Number of segmentation classes (3 for BraTS)
        dice_weight: Weight for Dice loss component
        focal_weight: Weight for Focal loss component
        iou_weight: Weight for IoU loss component
        boundary_weight: Weight for Boundary loss component
        focal_alpha: Class weights for Focal loss
        focal_gamma: Focusing parameter for Focal loss
        class_weights: Optional class weights for Dice/IoU
        ignore_background: If True, skip background in loss computation
    """
    def __init__(self,
                 num_classes=3,
                 dice_weight=1.0,
                 focal_weight=1.0,
                 iou_weight=2.0,  # ⭐ Emphasize IoU
                 boundary_weight=0.5,
                 focal_alpha=None,
                 focal_gamma=2.0,
                 class_weights=None,
                 ignore_background=True):
        super().__init__()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight
        self.boundary_weight = boundary_weight

        # Import individual loss components
        from .multiclass import MultiClassDiceLoss, MultiClassFocalLoss
        from .iou import MulticlassIoULoss
        from .boundary import BoundaryLoss

        self.dice_loss = MultiClassDiceLoss(
            num_classes=num_classes,
            ignore_background=ignore_background,
            class_weights=class_weights
        )

        self.focal_loss = MultiClassFocalLoss(
            num_classes=num_classes,
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_background=ignore_background
        )

        self.iou_loss = MulticlassIoULoss(
            num_classes=num_classes,
            ignore_background=ignore_background,
            class_weights=class_weights
        )

        self.boundary_loss = BoundaryLoss(
            ignore_background=ignore_background
        )

    def forward(self, logits, target):
        """
        Args:
            logits: (B, C, H, W) raw predictions
            target: (B, H, W) or (B, 1, H, W) class indices

        Returns:
            total_loss: Weighted combination of all losses
            loss_dict: Dictionary of individual loss values (for logging)
        """
        # Compute individual losses
        dice_l = self.dice_loss(logits, target)
        focal_l = self.focal_loss(logits, target)
        iou_l = self.iou_loss(logits, target)
        boundary_l = self.boundary_loss(logits, target)

        # Debug: Check for negative components (should NEVER be negative)
        if dice_l < 0 or focal_l < 0 or iou_l < 0 or boundary_l < 0:
            print(f"⚠️  WARNING: Negative loss component detected!")
            print(f"  dice_l: {dice_l.item():.6f}")
            print(f"  focal_l: {focal_l.item():.6f}")
            print(f"  iou_l: {iou_l.item():.6f}")
            print(f"  boundary_l: {boundary_l.item():.6f}")

        # Weighted combination
        total = (self.dice_weight * dice_l +
                 self.focal_weight * focal_l +
                 self.iou_weight * iou_l +
                 self.boundary_weight * boundary_l)

        # Return total loss and individual components for logging
        loss_dict = {
            'dice': dice_l.item(),
            'focal': focal_l.item(),
            'iou': iou_l.item(),
            'boundary': boundary_l.item(),
            'total': total.item()
        }

        return total, loss_dict


class UltimateMultiTaskLoss(nn.Module):
    """
    Ultimate Multi-Task Loss: Segmentation (Ultimate) + Classification

    Combines:
    - Segmentation: UltimateLoss (Dice + Focal + IoU + Boundary)
    - Classification: CrossEntropy for HGG vs LGG

    With deep supervision support for multi-scale training.

    Args:
        seg_loss_weight: Weight for segmentation loss
        cls_loss_weight: Weight for classification loss
        deep_supervision: If True, apply loss to auxiliary outputs
        aux_weight: Weight for auxiliary outputs (typically 0.3-0.4)
        **kwargs: Arguments passed to UltimateLoss
    """
    def __init__(self,
                 seg_loss_weight=1.0,
                 cls_loss_weight=0.5,
                 deep_supervision=True,
                 aux_weight=0.3,
                 **ultimate_loss_kwargs):
        super().__init__()

        self.seg_loss_weight = seg_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.deep_supervision = deep_supervision
        self.aux_weight = aux_weight

        # Segmentation loss
        self.seg_loss = UltimateLoss(**ultimate_loss_kwargs)

        # Classification loss
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, seg_logits, seg_target, cls_logits, cls_target, aux_outputs=None):
        """
        Args:
            seg_logits: (B, C, H, W) main segmentation output
            seg_target: (B, H, W) segmentation ground truth
            cls_logits: (B, num_classes_cls) classification output
            cls_target: (B,) classification ground truth
            aux_outputs: List of auxiliary segmentation outputs (optional)
                        [aux3, aux2, aux1] at different resolutions

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of all loss components
        """
        # Main segmentation loss
        seg_l, seg_dict = self.seg_loss(seg_logits, seg_target)

        # Auxiliary losses (deep supervision)
        if self.deep_supervision and aux_outputs is not None:
            aux_losses = []
            for i, aux_out in enumerate(aux_outputs):
                # Resize target to match auxiliary output resolution
                H, W = aux_out.shape[2:]

                # Ensure target is 3D (B, H, W) before interpolation
                if seg_target.dim() == 4:
                    target_3d = seg_target.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
                else:
                    target_3d = seg_target  # Already (B, H, W)

                # Resize: (B, H, W) -> (B, 1, H, W) -> interpolate -> (B, H_aux, W_aux)
                target_resized = F.interpolate(
                    target_3d.unsqueeze(1).float(),
                    size=(H, W),
                    mode='nearest'
                ).squeeze(1).long()

                # Compute loss for this auxiliary output
                aux_l, _ = self.seg_loss(aux_out, target_resized)
                aux_losses.append(aux_l)

            # Add weighted auxiliary losses
            aux_total = sum(aux_losses) * self.aux_weight
            seg_l = seg_l + aux_total

            # Add to loss dict
            seg_dict['aux_total'] = aux_total.item()
            for i, aux_l in enumerate(aux_losses):
                seg_dict[f'aux{3-i}'] = aux_l.item()  # aux3, aux2, aux1

        # Classification loss
        # CrossEntropyLoss expects: input (B, C) and target (B,)
        # Ensure cls_target is 1D
        if cls_target.dim() > 1:
            cls_target = cls_target.squeeze()
        cls_l = self.cls_loss(cls_logits, cls_target.long())

        # Total multi-task loss
        total = self.seg_loss_weight * seg_l + self.cls_loss_weight * cls_l

        # Combine loss dicts
        loss_dict = {
            **seg_dict,
            'cls': cls_l.item(),
            'seg_weighted': (self.seg_loss_weight * seg_l).item(),
            'cls_weighted': (self.cls_loss_weight * cls_l).item(),
            'total': total.item()
        }

        return total, loss_dict


# ============================================================================
# Loss Factory Function
# ============================================================================

def create_loss_from_config(cfg):
    """
    Factory function to create loss from configuration.

    Args:
        cfg: Configuration dictionary with loss parameters

    Returns:
        loss_fn: Configured loss function

    Example config:
        loss:
          type: "ultimate_multitask"
          seg_loss_weight: 1.0
          cls_loss_weight: 0.5
          dice_weight: 1.0
          focal_weight: 1.0
          iou_weight: 2.0
          boundary_weight: 0.5
          focal_alpha: [0.5, 0.4, 0.1]  # [bg, TC, ED] - emphasize TC
          focal_gamma: 3.0
          class_weights: [1.0, 3.0, 2.0]  # Emphasize TC
          deep_supervision: true
          aux_weight: 0.3
    """
    loss_cfg = cfg.get('train', {})
    model_cfg = cfg.get('model', {})

    loss_type = loss_cfg.get('loss_type', 'ultimate_multitask')
    num_classes_seg = model_cfg.get('num_classes_seg', 3)

    if loss_type == 'ultimate':
        # Segmentation only
        return UltimateLoss(
            num_classes=num_classes_seg,
            dice_weight=loss_cfg.get('dice_weight', 1.0),
            focal_weight=loss_cfg.get('focal_weight', 1.0),
            iou_weight=loss_cfg.get('iou_weight', 2.0),
            boundary_weight=loss_cfg.get('boundary_weight', 0.5),
            focal_alpha=loss_cfg.get('focal_alpha', None),
            focal_gamma=loss_cfg.get('focal_gamma', 2.0),
            class_weights=loss_cfg.get('class_weights', None),
            ignore_background=loss_cfg.get('ignore_background', True)
        )

    elif loss_type == 'ultimate_multitask':
        # Segmentation + Classification
        return UltimateMultiTaskLoss(
            seg_loss_weight=loss_cfg.get('seg_loss_weight', 1.0),
            cls_loss_weight=loss_cfg.get('cls_loss_weight', 0.5),
            deep_supervision=model_cfg.get('deep_supervision', True),
            aux_weight=loss_cfg.get('aux_weight', 0.3),
            num_classes=num_classes_seg,
            dice_weight=loss_cfg.get('dice_weight', 1.0),
            focal_weight=loss_cfg.get('focal_weight', 1.0),
            iou_weight=loss_cfg.get('iou_weight', 2.0),
            boundary_weight=loss_cfg.get('boundary_weight', 0.5),
            focal_alpha=loss_cfg.get('focal_alpha', None),
            focal_gamma=loss_cfg.get('focal_gamma', 3.0),
            class_weights=loss_cfg.get('class_weights', None),
            ignore_background=loss_cfg.get('ignore_background', True)
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ============================================================================
# Unit Tests
# ============================================================================

def test_ultimate_loss():
    """Test ultimate combined loss"""
    print("Testing UltimateLoss...")

    batch_size = 2
    num_classes = 3
    height, width = 64, 64

    logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
    target = torch.randint(0, num_classes, (batch_size, height, width))

    loss_fn = UltimateLoss(
        num_classes=num_classes,
        dice_weight=1.0,
        focal_weight=1.0,
        iou_weight=2.0,
        boundary_weight=0.5,
        focal_alpha=[0.5, 0.4, 0.1],
        focal_gamma=3.0,
        class_weights=[1.0, 3.0, 2.0]
    )

    loss, loss_dict = loss_fn(logits, target)

    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Components:")
    print(f"    - Dice: {loss_dict['dice']:.4f}")
    print(f"    - Focal: {loss_dict['focal']:.4f}")
    print(f"    - IoU: {loss_dict['iou']:.4f}")
    print(f"    - Boundary: {loss_dict['boundary']:.4f}")

    print(f"  Backward pass: ", end="")
    loss.backward()
    print("✓ OK")

    print("✓ UltimateLoss tests passed!\n")


def test_ultimate_multitask_loss():
    """Test ultimate multi-task loss with deep supervision"""
    print("Testing UltimateMultiTaskLoss...")

    batch_size = 2
    num_classes_seg = 3
    num_classes_cls = 2
    height, width = 256, 256

    # Main outputs
    seg_logits = torch.randn(batch_size, num_classes_seg, height, width, requires_grad=True)
    cls_logits = torch.randn(batch_size, num_classes_cls, requires_grad=True)

    # Targets
    seg_target = torch.randint(0, num_classes_seg, (batch_size, height, width))
    cls_target = torch.randint(0, num_classes_cls, (batch_size,))

    # Auxiliary outputs (deep supervision)
    aux3 = torch.randn(batch_size, num_classes_seg, 64, 64, requires_grad=True)
    aux2 = torch.randn(batch_size, num_classes_seg, 128, 128, requires_grad=True)
    aux1 = torch.randn(batch_size, num_classes_seg, 256, 256, requires_grad=True)
    aux_outputs = [aux3, aux2, aux1]

    loss_fn = UltimateMultiTaskLoss(
        seg_loss_weight=1.0,
        cls_loss_weight=0.5,
        deep_supervision=True,
        aux_weight=0.3,
        num_classes=num_classes_seg,
        dice_weight=1.0,
        focal_weight=1.0,
        iou_weight=2.0,
        boundary_weight=0.5
    )

    loss, loss_dict = loss_fn(seg_logits, seg_target, cls_logits, cls_target, aux_outputs)

    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Segmentation components:")
    print(f"    - Dice: {loss_dict['dice']:.4f}")
    print(f"    - Focal: {loss_dict['focal']:.4f}")
    print(f"    - IoU: {loss_dict['iou']:.4f}")
    print(f"    - Boundary: {loss_dict['boundary']:.4f}")
    print(f"  Auxiliary losses:")
    print(f"    - Aux3: {loss_dict['aux3']:.4f}")
    print(f"    - Aux2: {loss_dict['aux2']:.4f}")
    print(f"    - Aux1: {loss_dict['aux1']:.4f}")
    print(f"  Classification: {loss_dict['cls']:.4f}")

    print(f"  Backward pass: ", end="")
    loss.backward()
    print("✓ OK")

    print("✓ UltimateMultiTaskLoss tests passed!\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Ultimate Combined Loss Module Unit Tests")
    print("=" * 70 + "\n")

    test_ultimate_loss()
    test_ultimate_multitask_loss()

    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
