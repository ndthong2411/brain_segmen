"""
Multi-class segmentation losses for BraTS tumor sub-regions.

Classes:
- 0: Background
- 1: Tumor Core (NCR + ET)
- 2: Edema (ED)

Regions evaluated:
- WT (Whole Tumor) = TC + ED (classes 1,2)
- TC (Tumor Core) = class 1
- ED (Edema) = class 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassDiceLoss(nn.Module):
    """
    Multi-class Dice Loss for tumor sub-regions.

    Computes Dice loss for each class separately, then averages.
    Ignores background class (class 0) to avoid dominance.

    Args:
        num_classes: Number of classes (3 for BraTS: bg, TC, ED)
        ignore_background: If True, only compute loss for tumor classes
        class_weights: Weights for each class (None = equal weight)
    """
    def __init__(self, num_classes=3, ignore_background=True, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background

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
            loss: scalar Dice loss
        """
        # Convert target to one-hot encoding
        if target.dim() == 4:
            target = target.squeeze(1)  # (B, H, W)

        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Apply softmax to logits
        pred = F.softmax(logits, dim=1)  # (B, C, H, W)

        # Compute Dice for each class
        dice_scores = []
        start_idx = 1 if self.ignore_background else 0

        for c in range(start_idx, self.num_classes):
            pred_c = pred[:, c]  # (B, H, W)
            target_c = target_one_hot[:, c]  # (B, H, W)

            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
            dice_loss = 1.0 - dice

            # Apply class weight - ensure same device
            class_weight = self.class_weights[c].to(dice_loss.device)
            weighted_loss = dice_loss * class_weight
            dice_scores.append(weighted_loss.mean())

        # Average across classes
        total_loss = torch.stack(dice_scores).mean()

        return total_loss


class MultiClassFocalLoss(nn.Module):
    """
    Multi-class Focal Loss for handling class imbalance.

    Args:
        num_classes: Number of classes
        alpha: Weights for each class (list/tensor of length num_classes)
        gamma: Focusing parameter (default 2.0)
        ignore_background: If True, set background weight to 0
    """
    def __init__(self, num_classes=3, alpha=None, gamma=2.0, ignore_background=True):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma

        if alpha is None:
            # Default: equal weights, but 0 for background if ignored
            alpha = [0.0 if ignore_background else 1.0] + [1.0] * (num_classes - 1)

        self.register_buffer('alpha', torch.tensor(alpha))

    def forward(self, logits, target):
        """
        Args:
            logits: (B, C, H, W)
            target: (B, H, W) class indices
        """
        if target.dim() == 4:
            target = target.squeeze(1)

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # Get probability of true class
        B, C, H, W = probs.shape
        probs_flat = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (B*H*W, C)
        target_flat = target.view(-1)  # (B*H*W)

        pt = probs_flat[torch.arange(len(target_flat), device=target_flat.device), target_flat]  # (B*H*W)

        # Focal term
        focal_weight = (1 - pt) ** self.gamma

        # Cross entropy
        ce = -torch.log(pt + 1e-7)

        # Alpha weighting - ensure alpha tensor is on same device as target
        alpha_t = self.alpha.to(target_flat.device)[target_flat]

        # Focal loss
        loss = alpha_t * focal_weight * ce

        return loss.mean()


class MultiClassCombinedLoss(nn.Module):
    """
    Combined Dice + Focal Loss for multi-class segmentation.

    Best for BraTS with class imbalance (bg >> tumor >> sub-regions).

    Args:
        num_classes: Number of classes
        dice_weight: Weight for Dice loss component
        focal_weight: Weight for Focal loss component
        class_weights_dice: Class weights for Dice loss
        class_weights_focal: Alpha values for Focal loss
        focal_gamma: Gamma parameter for Focal loss
    """
    def __init__(self, num_classes=3,
                 dice_weight=1.0, focal_weight=1.0,
                 class_weights_dice=None,
                 class_weights_focal=None,
                 focal_gamma=2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.dice_loss = MultiClassDiceLoss(
            num_classes=num_classes,
            ignore_background=True,
            class_weights=class_weights_dice
        )

        self.focal_loss = MultiClassFocalLoss(
            num_classes=num_classes,
            alpha=class_weights_focal,
            gamma=focal_gamma,
            ignore_background=True
        )

    def forward(self, logits, target):
        dice = self.dice_loss(logits, target)
        focal = self.focal_loss(logits, target)

        total = self.dice_weight * dice + self.focal_weight * focal

        return total, dice.detach(), focal.detach()


class MultiTaskMultiClassLoss(nn.Module):
    """
    Multi-task loss for multi-class segmentation + classification.

    Args:
        num_classes_seg: Number of segmentation classes
        seg_loss_type: 'dice', 'focal', or 'combined'
        seg_w: Weight for segmentation loss
        cls_w: Weight for classification loss
    """
    def __init__(self, num_classes_seg=3,
                 seg_loss_type='combined',
                 seg_w=1.0, cls_w=0.5,
                 dice_weight=1.0, focal_weight=1.0,
                 focal_gamma=2.0):
        super().__init__()
        self.seg_w = seg_w
        self.cls_w = cls_w

        # Segmentation loss
        if seg_loss_type == 'dice':
            self.seg_loss = MultiClassDiceLoss(num_classes=num_classes_seg)
        elif seg_loss_type == 'focal':
            self.seg_loss = MultiClassFocalLoss(num_classes=num_classes_seg, gamma=focal_gamma)
        elif seg_loss_type == 'combined':
            self.seg_loss = MultiClassCombinedLoss(
                num_classes=num_classes_seg,
                dice_weight=dice_weight,
                focal_weight=focal_weight,
                focal_gamma=focal_gamma
            )
        else:
            raise ValueError(f"Unknown seg_loss_type: {seg_loss_type}")

        # Classification loss
        self.cls_loss = nn.CrossEntropyLoss()
        self.seg_loss_type = seg_loss_type

    def forward(self, seg_logits, seg_mask, cls_logits, cls_label):
        # Segmentation loss
        if self.seg_loss_type == 'combined':
            l_seg_total, l_dice, l_focal = self.seg_loss(seg_logits, seg_mask)
            l_seg = l_seg_total
        else:
            l_seg = self.seg_loss(seg_logits, seg_mask)

        # Classification loss
        l_cls = self.cls_loss(cls_logits, cls_label)

        # Total loss
        total = self.seg_w * l_seg + self.cls_w * l_cls

        return total, l_seg.detach(), l_cls.detach()
