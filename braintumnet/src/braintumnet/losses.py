import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
import warnings

def dice_loss_with_logits(logits, target, eps=1e-6):
    """
    Stable dice loss computation that works under mixed precision.
    Cast to float32 for the math to avoid float16 underflow, then return
    the result in the original dtype for downstream scaling.
    """
    orig_dtype = logits.dtype
    logits = logits.float()
    target = target.float()

    pred = torch.sigmoid(logits)
    num = 2 * (pred * target).sum(dim=(2, 3))
    den = pred.pow(2).sum(dim=(2, 3)) + target.pow(2).sum(dim=(2, 3)) + eps
    dice = 1 - (num + eps) / den
    return dice.mean().to(orig_dtype)

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
        self.alpha_pos, self.alpha_neg = self._parse_alpha(alpha)
        self.gamma = gamma

    @staticmethod
    def _parse_alpha(alpha):
        """Normalize incoming alpha weights to (pos_alpha, neg_alpha) floats."""
        if isinstance(alpha, torch.Tensor):
            values = alpha.detach().cpu().flatten().tolist()
        elif isinstance(alpha, (list, tuple)):
            values = [float(a) for a in alpha]
        else:
            pos_alpha = float(alpha)
            return pos_alpha, 1.0 - pos_alpha

        if not values:
            raise ValueError("alpha must contain at least one value")

        if len(values) == 1:
            pos_alpha = values[0]
            return pos_alpha, 1.0 - pos_alpha

        neg_alpha = values[0]
        pos_components = values[1:]
        if not pos_components:
            raise ValueError("alpha list must include at least one positive-class weight")
        pos_alpha = float(sum(pos_components) / len(pos_components))
        return pos_alpha, float(neg_alpha)

    def forward(self, logits, target):
        """
        Args:
            logits: (B, 1, H, W) raw predictions
            target: (B, 1, H, W) binary ground truth {0, 1}
        """
        orig_dtype = logits.dtype
        logits = logits.float()

        # Convert to probabilities
        probs = torch.sigmoid(logits)

        # For numerical stability (respect current dtype limits)
        finfo = torch.finfo(probs.dtype)
        probs = torch.clamp(probs, min=finfo.tiny, max=1 - finfo.eps)

        # Focal loss computation
        # pt = p if y=1, else (1-p)
        pt = torch.where(target == 1, probs, 1 - probs)

        # Alpha weighting
        alpha_pos = logits.new_tensor(self.alpha_pos)
        alpha_neg = logits.new_tensor(self.alpha_neg)
        alpha_t = torch.where(target == 1, alpha_pos, alpha_neg)

        # Focal term: (1-pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Binary cross entropy: -log(pt)
        bce = -torch.log(pt)

        # Final focal loss
        loss = alpha_t * focal_weight * bce

        return loss.mean().to(orig_dtype)


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
        loss_type: Type of segmentation loss.
                  For multi-class segmentation (num_classes>1), binary loss names
                  are automatically mapped to their multiclass counterparts.
                  - 'dice_ce'            -> Dice + CE
                  - 'dice_ce_weighted'   -> Dice + CE with class_weights
                  - 'dice_focal'         -> Dice + Focal Loss
        pos_weight: Positive class weight for BCE (if loss_type='dice_ce_weighted')
                   For 97% bg / 3% tumor, use 32.0
        focal_alpha: Alpha for focal loss (if loss_type='dice_focal')
        focal_gamma: Gamma for focal loss (if loss_type='dice_focal')
    """
    def __init__(self, seg_w=1.0, cls_w=0.7, boundary_w=0.0,
                 loss_type='dice_ce', pos_weight=None,
                 focal_alpha=0.25, focal_gamma=2.0,
                 # Multiclass params
                 num_classes=1, ignore_background=True, class_weights=None):
        super().__init__()
        self.seg_w = seg_w
        self.cls_w = cls_w
        self.boundary_w = boundary_w
        self.loss_type = loss_type
        self.num_classes = num_classes

        self.seg_loss, self.loss_name = self._build_seg_loss(
            loss_type=loss_type,
            num_classes=num_classes,
            ignore_background=ignore_background,
            class_weights=class_weights,
            pos_weight=pos_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma
        )

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

        # Classification loss (only if cls_logits is not None)
        if cls_logits is not None and self.cls_w > 0:
            l_cls = self.cls_loss(cls_logits, cls_label)
        else:
            l_cls = torch.tensor(0.0, device=seg_logits.device)

        # Total loss
        total_loss = self.seg_w * l_seg + self.cls_w * l_cls

        return total_loss, l_seg.detach(), l_cls.detach()

    def _build_seg_loss(self, loss_type, num_classes, ignore_background,
                        class_weights, pos_weight, focal_alpha, focal_gamma):
        """
        Build segmentation loss, automatically selecting multiclass variants
        when the model predicts multiple classes.
        """
        is_multiclass = num_classes > 1

        # Explicit multiclass selections
        if loss_type == 'multiclass_dice_ce':
            seg_loss = MulticlassDiceCELoss(
                num_classes=num_classes,
                ignore_background=ignore_background,
                class_weights=class_weights
            )
            return seg_loss, 'multiclass_dice_ce'

        if loss_type == 'multiclass_dice_focal':
            alpha_list = self._normalize_multiclass_alpha(
                focal_alpha, num_classes, ignore_background
            )
            seg_loss = MulticlassDiceFocalLoss(
                num_classes=num_classes,
                ignore_background=ignore_background,
                focal_alpha=alpha_list,
                focal_gamma=focal_gamma
            )
            return seg_loss, 'multiclass_dice_focal'

        # Auto-upgrade binary losses when dealing with multiclass targets
        if is_multiclass and loss_type in {'dice_ce', 'dice_ce_weighted'}:
            if loss_type == 'dice_ce_weighted' and pos_weight is not None:
                warnings.warn(
                    "pos_weight is ignored for multiclass DiceCE; provide class_weights instead.",
                    RuntimeWarning
                )
            seg_loss = MulticlassDiceCELoss(
                num_classes=num_classes,
                ignore_background=ignore_background,
                class_weights=class_weights
            )
            return seg_loss, 'multiclass_dice_ce(auto)'

        if is_multiclass and loss_type == 'dice_focal':
            alpha_list = self._normalize_multiclass_alpha(
                focal_alpha, num_classes, ignore_background
            )
            seg_loss = MulticlassDiceFocalLoss(
                num_classes=num_classes,
                ignore_background=ignore_background,
                focal_alpha=alpha_list,
                focal_gamma=focal_gamma
            )
            return seg_loss, 'multiclass_dice_focal(auto)'

        # Binary (single-class) losses retain their original definitions
        if loss_type == 'dice_ce':
            return DiceCELoss(), 'dice_ce'
        if loss_type == 'dice_ce_weighted':
            return DiceCELoss(pos_weight=pos_weight), 'dice_ce_weighted'
        if loss_type == 'dice_focal':
            return DiceFocalLoss(focal_alpha=focal_alpha, focal_gamma=focal_gamma), 'dice_focal'

        raise ValueError(
            f"Unknown loss_type: {loss_type}. Supported values: "
            "'dice_ce', 'dice_ce_weighted', 'dice_focal', "
            "'multiclass_dice_ce', 'multiclass_dice_focal'"
        )

    @staticmethod
    def _normalize_multiclass_alpha(focal_alpha, num_classes, ignore_background):
        """Ensure focal alpha is a list of length num_classes for multiclass losses."""
        if focal_alpha is None:
            return None

        if isinstance(focal_alpha, torch.Tensor):
            alpha = focal_alpha.detach().cpu().flatten().tolist()
        elif isinstance(focal_alpha, (list, tuple)):
            alpha = [float(a) for a in focal_alpha]
        else:
            alpha = [float(focal_alpha)] * num_classes

        if len(alpha) < num_classes:
            alpha = alpha + [alpha[-1]] * (num_classes - len(alpha))
        elif len(alpha) > num_classes:
            alpha = alpha[:num_classes]

        if ignore_background and num_classes > 0:
            alpha[0] = 0.0

        return alpha


# ============================================================================
# MULTI-CLASS SEGMENTATION LOSSES (for 3+ class segmentation)
# ============================================================================

def multiclass_dice_loss(logits, targets, num_classes, ignore_background=True, smooth=1.0):
    """
    Multi-class Dice Loss.

    Args:
        logits: (B, C, H, W) - raw logits from model
        targets: (B, 1, H, W) - integer class labels [0, C-1]
        num_classes: Number of classes
        ignore_background: If True, exclude background (class 0) from loss
        smooth: Smoothing factor

    Returns:
        Dice loss (scalar)
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1)  # (B, C, H, W)

    # Convert targets to one-hot encoding
    targets = targets.squeeze(1).long()  # (B, H, W)
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes)  # (B, H, W, C)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

    # Compute Dice for each class
    dice_scores = []
    start_class = 1 if ignore_background else 0

    for c in range(start_class, num_classes):
        pred_c = probs[:, c, :, :]  # (B, H, W)
        target_c = targets_one_hot[:, c, :, :]  # (B, H, W)

        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.mean())

    # Average Dice across classes
    mean_dice = torch.stack(dice_scores).mean()
    return 1.0 - mean_dice  # Return loss (1 - Dice)


class MulticlassDiceCELoss(nn.Module):
    """
    Multi-class Dice + CrossEntropy Loss.

    Combines:
    - Multi-class Dice Loss: Good for overlap, class imbalance
    - CrossEntropy Loss: Standard multi-class classification loss

    Args:
        num_classes: Number of segmentation classes
        ignore_background: If True, exclude background from Dice computation
        dice_weight: Weight for Dice loss component
        ce_weight: Weight for CrossEntropy loss component
        class_weights: Optional class weights for CE loss
    """
    def __init__(self, num_classes=3, ignore_background=True,
                 dice_weight=1.0, ce_weight=1.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        # CrossEntropy loss
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - raw logits
            targets: (B, 1, H, W) - integer class labels
        """
        # Dice loss
        dice = multiclass_dice_loss(logits, targets, self.num_classes,
                                     self.ignore_background)

        # CE loss
        targets_ce = targets.squeeze(1).long()  # (B, H, W)
        ce = self.ce_loss(logits, targets_ce)

        return self.dice_weight * dice + self.ce_weight * ce


class MulticlassFocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Focuses on hard examples by down-weighting easy ones.
    Good for handling class imbalance and hard boundary pixels.

    Args:
        num_classes: Number of classes
        alpha: Class weights (list of length num_classes)
        gamma: Focusing parameter (higher = more focus on hard examples)
        ignore_background: If True, set background weight to 0
    """
    def __init__(self, num_classes=3, alpha=None, gamma=2.0, ignore_background=True):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma

        # Set alpha (class weights)
        if alpha is None:
            alpha = [1.0] * num_classes
        if ignore_background:
            alpha[0] = 0.0  # Ignore background

        self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, 1, H, W)
        """
        # Move alpha to same device as logits
        self.alpha = self.alpha.to(logits.device)

        # Get probabilities
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)

        # Get targets as class indices
        targets = targets.squeeze(1).long()  # (B, H, W)

        # Flatten
        probs = probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # (B*H*W, C)
        targets = targets.reshape(-1)  # (B*H*W,)

        # Get probabilities of true class
        pt = probs[torch.arange(len(targets)), targets]  # (B*H*W,)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Class weight
        alpha_t = self.alpha[targets]

        # CrossEntropy loss
        ce_loss = torch.nn.functional.cross_entropy(
            probs, targets, reduction='none'
        )

        # Combine
        focal_loss = alpha_t * focal_weight * ce_loss

        return focal_loss.mean()


class MulticlassDiceFocalLoss(nn.Module):
    """
    Multi-class Dice + Focal Loss (RECOMMENDED for multi-class segmentation).

    Combines:
    - Multi-class Dice Loss: Good for overlap and class imbalance
    - Multi-class Focal Loss: Focuses on hard examples

    Args:
        num_classes: Number of classes
        ignore_background: If True, exclude background from metrics
        dice_weight: Weight for Dice component
        focal_weight: Weight for Focal component
        focal_alpha: Class weights for focal loss
        focal_gamma: Focusing parameter
    """
    def __init__(self, num_classes=3, ignore_background=True,
                 dice_weight=1.0, focal_weight=1.0,
                 focal_alpha=None, focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.focal_loss = MulticlassFocalLoss(
            num_classes=num_classes,
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_background=ignore_background
        )

    def forward(self, logits, targets):
        # Dice loss
        dice = multiclass_dice_loss(logits, targets, self.num_classes,
                                     self.ignore_background)

        # Focal loss
        focal = self.focal_loss(logits, targets)

        return self.dice_weight * dice + self.focal_weight * focal
