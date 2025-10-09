import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

def dice_loss_with_logits(logits, target, eps=1e-6):
    pred = torch.sigmoid(logits)
    num = 2 * (pred * target).sum(dim=(2,3))
    den = (pred.pow(2).sum(dim=(2,3)) + target.pow(2).sum(dim=(2,3))) + eps
    dice = 1 - (num + eps) / den
    return dice.mean()

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal Loss down-weights easy examples and focuses on hard examples.
    For tumor segmentation with 97% background, this helps the model
    focus on the minority tumor class.

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)

    Formula: FL(p) = -α(1-p)^γ * log(p)
    - α: class weight (balance pos/neg)
    - γ: focusing parameter (down-weight easy examples)
    - p: predicted probability

    Args:
        alpha: Weight for positive class (tumor). Default 0.25
        gamma: Focusing parameter. Default 2.0
               Higher gamma = more focus on hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        """
        Args:
            logits: (B, 1, H, W) raw predictions
            target: (B, 1, H, W) binary ground truth {0, 1}
        """
        # Convert to probabilities
        probs = torch.sigmoid(logits)

        # For numerical stability
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)

        # Focal loss computation
        # pt = p if y=1, else (1-p)
        pt = torch.where(target == 1, probs, 1 - probs)

        # Alpha weighting
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)

        # Focal term: (1-pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Binary cross entropy: -log(pt)
        bce = -torch.log(pt)

        # Final focal loss
        loss = alpha_t * focal_weight * bce

        return loss.mean()


class DiceCELoss(nn.Module):
    """
    Dice + Cross Entropy Loss with optional class weighting.

    Args:
        pos_weight: Weight for positive class (tumor).
                   None = no weighting
                   Float = weight for tumor class
                   For 97% bg / 3% tumor, use pos_weight=32.0
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            # Ensure pos_weight is tensor
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor([pos_weight])
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, seg_logits, seg_mask):
        dice = dice_loss_with_logits(seg_logits, seg_mask)
        bce = self.bce(seg_logits, seg_mask)
        return dice + bce


class DiceFocalLoss(nn.Module):
    """
    Dice + Focal Loss - Best for severe class imbalance.

    Combines:
    - Dice Loss: Good for overlap, handles imbalance
    - Focal Loss: Focuses on hard examples, down-weights easy background

    Args:
        focal_alpha: Weight for positive class in focal loss
        focal_gamma: Focusing parameter
        dice_weight: Weight for dice loss component
        focal_weight: Weight for focal loss component
    """
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0,
                 dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, seg_logits, seg_mask):
        dice = dice_loss_with_logits(seg_logits, seg_mask)
        focal = self.focal(seg_logits, seg_mask)
        return self.dice_weight * dice + self.focal_weight * focal

class BoundaryLoss(nn.Module):
    """
    Boundary Loss - penalizes predictions that are far from true boundaries.

    This loss computes distance maps from ground truth boundaries and uses them
    to weight the prediction errors. Errors near boundaries have more impact.

    Reference: "Boundary loss for highly unbalanced segmentation"
               (Kervadec et al., MIDL 2019)

    How it works:
    1. Compute distance map from boundaries (using distance transform)
    2. Multiply prediction errors by distance values
    3. Errors at boundaries (distance=0) have minimal impact
    4. Errors far from boundaries have higher impact

    This encourages the model to predict boundaries accurately.
    """
    def __init__(self, cache_distance_maps=True):
        super().__init__()
        self.cache = {} if cache_distance_maps else None

    def compute_distance_map(self, mask):
        """
        Compute signed distance map from boundaries.

        Args:
            mask: (B, 1, H, W) binary mask tensor {0, 1}

        Returns:
            distance_map: (B, 1, H, W) signed distance map
                         Positive inside tumor, negative outside
        """
        B = mask.shape[0]
        distance_maps = []

        for b in range(B):
            mask_np = mask[b, 0].cpu().numpy().astype(bool)

            # Check cache (using hash of mask)
            if self.cache is not None:
                mask_hash = hash(mask_np.tobytes())
                if mask_hash in self.cache:
                    distance_maps.append(self.cache[mask_hash])
                    continue

            # Compute distance transform
            if mask_np.any():
                # Distance from foreground (inside tumor)
                pos_dist = distance_transform_edt(mask_np)
                # Distance from background (outside tumor)
                neg_dist = distance_transform_edt(~mask_np)
                # Signed distance: positive inside, negative outside
                distance_map = neg_dist.astype(np.float32) - pos_dist.astype(np.float32)
            else:
                # Empty mask
                distance_map = np.zeros_like(mask_np, dtype=np.float32)

            # Cache result
            if self.cache is not None:
                self.cache[mask_hash] = distance_map

            distance_maps.append(distance_map)

        return torch.from_numpy(np.stack(distance_maps)).unsqueeze(1).to(mask.device)

    def forward(self, pred_logits, target):
        """
        Compute boundary loss.

        Args:
            pred_logits: (B, 1, H, W) raw logits (before sigmoid)
            target: (B, 1, H, W) binary ground truth {0, 1}

        Returns:
            loss: scalar boundary loss value
        """
        # Convert logits to probabilities
        pred_prob = torch.sigmoid(pred_logits)

        # Compute distance map (expensive, but cached)
        with torch.no_grad():
            dist_map = self.compute_distance_map(target)

        # Boundary loss: (pred - target) * distance_map
        # This weights errors by their distance from boundaries
        boundary_term = (pred_prob - target) * dist_map

        # Take absolute value and mean
        loss = boundary_term.abs().mean()

        return loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for segmentation + classification.

    Args:
        seg_w: Weight for segmentation loss
        cls_w: Weight for classification loss
        boundary_w: Weight for boundary loss (optional)
        loss_type: Type of segmentation loss
                  'dice_ce' - Dice + BCE (default)
                  'dice_ce_weighted' - Dice + Weighted BCE
                  'dice_focal' - Dice + Focal Loss (best for imbalance)
        pos_weight: Positive class weight for BCE (if loss_type='dice_ce_weighted')
                   For 97% bg / 3% tumor, use 32.0
        focal_alpha: Alpha for focal loss (if loss_type='dice_focal')
        focal_gamma: Gamma for focal loss (if loss_type='dice_focal')
    """
    def __init__(self, seg_w=1.0, cls_w=0.7, boundary_w=0.0,
                 loss_type='dice_ce', pos_weight=None,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.seg_w = seg_w
        self.cls_w = cls_w
        self.boundary_w = boundary_w
        self.loss_type = loss_type

        # Select segmentation loss based on type
        if loss_type == 'dice_ce':
            self.seg_loss = DiceCELoss()
        elif loss_type == 'dice_ce_weighted':
            self.seg_loss = DiceCELoss(pos_weight=pos_weight)
        elif loss_type == 'dice_focal':
            self.seg_loss = DiceFocalLoss(focal_alpha=focal_alpha,
                                         focal_gamma=focal_gamma)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. "
                           f"Must be 'dice_ce', 'dice_ce_weighted', or 'dice_focal'")

        self.cls_loss = nn.CrossEntropyLoss()

        # Only initialize boundary loss if weight > 0
        if self.boundary_w > 0:
            self.boundary_loss = BoundaryLoss(cache_distance_maps=True)
        else:
            self.boundary_loss = None

    def forward(self, seg_logits, seg_mask, cls_logits, cls_label):
        # Segmentation loss (Dice + BCE)
        l_seg = self.seg_loss(seg_logits, seg_mask)

        # Boundary loss (optional)
        if self.boundary_loss is not None and self.boundary_w > 0:
            l_boundary = self.boundary_loss(seg_logits, seg_mask)
            l_seg = l_seg + self.boundary_w * l_boundary
        else:
            l_boundary = torch.tensor(0.0, device=seg_logits.device)

        # Classification loss
        l_cls = self.cls_loss(cls_logits, cls_label)

        # Total loss
        total_loss = self.seg_w * l_seg + self.cls_w * l_cls

        return total_loss, l_seg.detach(), l_cls.detach()
