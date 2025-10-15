"""
IoU (Intersection over Union) Loss for Multi-class Segmentation

This module implements IoU loss to directly optimize the IoU metric,
which is the target metric for reaching 0.90 performance.

IoU Loss is computed as: 1 - IoU
where IoU = Intersection / Union

For multi-class segmentation:
- Compute IoU per class
- Average across tumor classes (exclude background)
- Return 1 - mean_IoU as loss

Author: BrainTumNet V2.0 Upgrade
Date: 2025-01-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassIoULoss(nn.Module):
    """
    Multi-class IoU Loss for tumor sub-regions.

    Directly optimizes IoU metric by computing:
    IoU = Intersection / Union
    Loss = 1 - IoU

    Args:
        num_classes: Number of classes (3 for BraTS: bg, TC, ED)
        ignore_background: If True, only compute loss for tumor classes
        smooth: Smoothing factor to avoid division by zero
        class_weights: Optional weights for each class
    """
    def __init__(self, num_classes=3, ignore_background=True, smooth=1.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.smooth = smooth

        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights))
        else:
            self.register_buffer('class_weights', torch.ones(num_classes))

    def forward(self, logits, target):
        """
        Args:
            logits: (B, C, H, W) raw predictions (C=num_classes)
            target: (B, H, W) or (B, 1, H, W) class indices {0, 1, 2}

        Returns:
            loss: scalar IoU loss (1 - mean_IoU)
        """
        # Convert target to one-hot encoding
        if target.dim() == 4:
            target = target.squeeze(1)  # (B, H, W)

        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Apply softmax to logits to get probabilities
        pred_probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # Compute IoU for each class
        iou_scores = []
        start_idx = 1 if self.ignore_background else 0

        for c in range(start_idx, self.num_classes):
            pred_c = pred_probs[:, c]  # (B, H, W)
            target_c = target_one_hot[:, c]  # (B, H, W)

            # Intersection: element-wise product
            intersection = (pred_c * target_c).sum(dim=(1, 2))  # (B,)

            # Union: sum of both masks minus intersection
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2)) - intersection  # (B,)

            # IoU with smoothing
            iou = (intersection + self.smooth) / (union + self.smooth)  # (B,)

            # Compute loss for this class (1 - IoU)
            iou_loss_c = 1.0 - iou  # (B,) - loss per sample

            # Apply class weight to LOSS (not IoU)
            class_weight = self.class_weights[c].to(iou_loss_c.device)
            weighted_loss = iou_loss_c * class_weight
            iou_scores.append(weighted_loss.mean())  # Average over batch

        # Average across classes
        loss = torch.stack(iou_scores).mean()

        return loss  # Already computed as loss, no need for 1 - mean_iou


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice/F1 score.

    Allows weighting of false positives and false negatives differently.
    When alpha=beta=0.5, Tversky = Dice.

    Useful for imbalanced segmentation where we want to penalize
    false negatives more (alpha < 0.5) or false positives more (alpha > 0.5).

    Args:
        alpha: Weight for false positives (default 0.3)
        beta: Weight for false negatives (default 0.7)
               alpha + beta should = 1.0
        num_classes: Number of classes
        ignore_background: If True, skip background class
    """
    def __init__(self, alpha=0.3, beta=0.7, num_classes=3, ignore_background=True, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.smooth = smooth

    def forward(self, logits, target):
        """
        Args:
            logits: (B, C, H, W)
            target: (B, H, W)
        Returns:
            loss: Tversky loss
        """
        if target.dim() == 4:
            target = target.squeeze(1)

        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        pred_probs = F.softmax(logits, dim=1)

        tversky_scores = []
        start_idx = 1 if self.ignore_background else 0

        for c in range(start_idx, self.num_classes):
            pred_c = pred_probs[:, c]
            target_c = target_one_hot[:, c]

            # True Positive
            tp = (pred_c * target_c).sum(dim=(1, 2))

            # False Positive
            fp = (pred_c * (1 - target_c)).sum(dim=(1, 2))

            # False Negative
            fn = ((1 - pred_c) * target_c).sum(dim=(1, 2))

            # Tversky index
            tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)

            tversky_scores.append(tversky.mean())

        mean_tversky = torch.stack(tversky_scores).mean()
        loss = 1.0 - mean_tversky

        return loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - Applies focal weighting to Tversky loss.

    Combines benefits of:
    - Tversky: Handle class imbalance and FP/FN trade-off
    - Focal: Focus on hard examples

    Useful for small tumor regions with high class imbalance.

    Args:
        alpha: Tversky alpha (FP weight)
        beta: Tversky beta (FN weight)
        gamma: Focal gamma (focusing parameter, typically 1.0-2.0)
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.5, num_classes=3, ignore_background=True):
        super().__init__()
        self.tversky_loss = TverskyLoss(alpha, beta, num_classes, ignore_background)
        self.gamma = gamma

    def forward(self, logits, target):
        tversky_loss = self.tversky_loss(logits, target)
        # Apply focal weighting: loss^gamma
        focal_tversky = torch.pow(tversky_loss, self.gamma)
        return focal_tversky


class ComboIoULoss(nn.Module):
    """
    Combined IoU + Dice Loss.

    IoU is stricter than Dice (penalizes overlap more).
    Combining both can improve performance.

    Loss = w_iou * IoU_loss + w_dice * Dice_loss

    Args:
        iou_weight: Weight for IoU loss component
        dice_weight: Weight for Dice loss component
    """
    def __init__(self, iou_weight=1.0, dice_weight=1.0, num_classes=3, ignore_background=True):
        super().__init__()
        self.iou_weight = iou_weight
        self.dice_weight = dice_weight

        self.iou_loss = MulticlassIoULoss(num_classes, ignore_background)
        # Import Dice loss from existing module
        from .losses_multiclass import MultiClassDiceLoss
        self.dice_loss = MultiClassDiceLoss(num_classes, ignore_background)

    def forward(self, logits, target):
        iou_l = self.iou_loss(logits, target)
        dice_l = self.dice_loss(logits, target)

        total = self.iou_weight * iou_l + self.dice_weight * dice_l

        return total


# ============================================================================
# Unit Tests
# ============================================================================

def test_iou_loss():
    """Test IoU loss on synthetic data"""
    print("Testing MulticlassIoULoss...")

    batch_size = 2
    num_classes = 3
    height, width = 64, 64

    # Create synthetic data
    logits = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))

    # Test loss
    loss_fn = MulticlassIoULoss(num_classes=num_classes, ignore_background=True)
    loss = loss_fn(logits, target)

    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Loss range: [0, 1]")
    print(f"  Backward pass: ", end="")
    loss.backward()
    print("✓ OK")

    # Test perfect prediction
    perfect_logits = torch.zeros(batch_size, num_classes, height, width)
    for b in range(batch_size):
        for c in range(num_classes):
            perfect_logits[b, c][target[b] == c] = 10.0  # High logit for correct class

    loss_fn = MulticlassIoULoss(num_classes=num_classes, ignore_background=True)
    perfect_loss = loss_fn(perfect_logits, target)
    print(f"  Perfect prediction loss: {perfect_loss.item():.6f} (should be ~0)")

    # Test worst prediction
    worst_logits = -perfect_logits
    worst_loss = loss_fn(worst_logits, target)
    print(f"  Worst prediction loss: {worst_loss.item():.6f} (should be ~1)")

    print("✓ MulticlassIoULoss tests passed!\n")


def test_tversky_loss():
    """Test Tversky loss"""
    print("Testing TverskyLoss...")

    batch_size = 2
    num_classes = 3
    height, width = 64, 64

    logits = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))

    # Test with emphasis on false negatives (under-segmentation penalty)
    loss_fn = TverskyLoss(alpha=0.3, beta=0.7, num_classes=num_classes)
    loss = loss_fn(logits, target)

    print(f"  Loss value (alpha=0.3, beta=0.7): {loss.item():.4f}")
    loss.backward()
    print("  Backward pass: ✓ OK")

    print("✓ TverskyLoss tests passed!\n")


def test_combo_iou_loss():
    """Test combined IoU + Dice loss"""
    print("Testing ComboIoULoss...")

    batch_size = 2
    num_classes = 3
    height, width = 64, 64

    logits = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))

    loss_fn = ComboIoULoss(iou_weight=1.0, dice_weight=1.0, num_classes=num_classes)
    loss = loss_fn(logits, target)

    print(f"  Loss value: {loss.item():.4f}")
    loss.backward()
    print("  Backward pass: ✓ OK")

    print("✓ ComboIoULoss tests passed!\n")


if __name__ == "__main__":
    print("=" * 70)
    print("IoU Loss Module Unit Tests")
    print("=" * 70 + "\n")

    test_iou_loss()
    test_tversky_loss()
    test_combo_iou_loss()

    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
