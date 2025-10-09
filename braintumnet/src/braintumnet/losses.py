import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

def dice_loss_with_logits(logits, target, eps=1e-6):
    pred = torch.sigmoid(logits)
    num = 2 * (pred * target).sum(dim=(2,3))
    den = (pred.pow(2).sum(dim=(2,3)) + target.pow(2).sum(dim=(2,3))) + eps
    dice = 1 - (num + eps) / den
    return dice.mean()

class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, seg_logits, seg_mask):
        return dice_loss_with_logits(seg_logits, seg_mask) + self.bce(seg_logits, seg_mask)

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
    def __init__(self, seg_w=1.0, cls_w=0.7, boundary_w=0.0):
        super().__init__()
        self.seg_w = seg_w
        self.cls_w = cls_w
        self.boundary_w = boundary_w

        self.seg_loss = DiceCELoss()
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
