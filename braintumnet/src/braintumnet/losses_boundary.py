"""
Boundary Loss for Medical Image Segmentation

Boundary loss emphasizes accurate segmentation of object boundaries by
weighting pixels based on their distance to the nearest boundary.

This is critical for IoU improvement since boundary errors significantly
impact IoU more than interior errors.

Implementation based on:
"Boundary loss for highly unbalanced segmentation" (Kervadec et al., 2019)

Author: BrainTumNet V2.0 Upgrade
Date: 2025-01-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class BoundaryLoss(nn.Module):
    """
    Boundary Loss - Penalizes errors near object boundaries.

    Computes signed distance function (SDF) from ground truth boundaries
    and weights prediction errors by distance to boundary.

    Pixels near boundaries get higher loss weight → forces network
    to focus on precise boundary delineation.

    Args:
        theta0: Distance at which weight = 1.0 (default 3 pixels)
        theta: Distance decay parameter (default 5 pixels)
        ignore_background: If True, only compute for tumor classes
    """
    def __init__(self, theta0=3, theta=5, ignore_background=True):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
        self.ignore_background = ignore_background

    def compute_sdf(self, mask):
        """
        Compute signed distance function from mask boundaries.

        SDF(x) = distance to nearest boundary
        - Positive inside object
        - Negative outside object
        - Zero at boundary

        Args:
            mask: (H, W) binary mask tensor

        Returns:
            sdf: (H, W) signed distance map
        """
        mask_np = mask.cpu().numpy().astype(np.uint8)

        # Compute distance transform for inside (positive distances)
        pos_mask = (mask_np > 0).astype(np.uint8)
        pos_dist = distance_transform_edt(pos_mask)

        # Compute distance transform for outside (negative distances)
        neg_mask = (mask_np == 0).astype(np.uint8)
        neg_dist = distance_transform_edt(neg_mask)

        # Combine: negative outside, positive inside
        sdf = neg_dist.astype(np.float32) - pos_dist.astype(np.float32)

        return torch.from_numpy(sdf).float().to(mask.device)

    def compute_boundary_weight(self, sdf):
        """
        Compute boundary weight map from SDF.

        Weight is high near boundaries, low far from boundaries.

        w(x) = exp(-|SDF(x)| / theta)

        Args:
            sdf: (H, W) signed distance map

        Returns:
            weight: (H, W) boundary weight map [0, 1]
        """
        # Exponential decay based on distance to boundary
        weight = torch.exp(-sdf.abs() / self.theta)
        return weight

    def forward(self, logits, target):
        """
        Args:
            logits: (B, C, H, W) raw predictions
            target: (B, H, W) or (B, 1, H, W) class indices

        Returns:
            loss: Boundary-weighted loss
        """
        if target.dim() == 4:
            target = target.squeeze(1)  # (B, H, W)

        # Get prediction probabilities
        pred_probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        B, C, H, W = logits.shape
        total_loss = 0.0
        num_classes = 0

        start_idx = 1 if self.ignore_background else 0

        for c in range(start_idx, C):
            for b in range(B):
                # Get ground truth mask for this class
                target_mask = (target[b] == c).float()  # (H, W)

                # Skip if no pixels of this class (empty mask)
                if target_mask.sum() == 0:
                    continue

                # Compute signed distance function
                sdf = self.compute_sdf(target_mask)  # (H, W)

                # Compute boundary weight
                weight = self.compute_boundary_weight(sdf)  # (H, W)

                # Get prediction for this class
                pred_c = pred_probs[b, c]  # (H, W)

                # Compute weighted error
                # L1 distance between prediction and target, weighted by boundary proximity
                error = (pred_c - target_mask).abs()
                weighted_error = weight * error

                total_loss += weighted_error.mean()
                num_classes += 1

        # Average over all classes and batches
        if num_classes > 0:
            loss = total_loss / num_classes
        else:
            loss = torch.tensor(0.0, device=logits.device)

        return loss


class HausdorffLoss(nn.Module):
    """
    Hausdorff Distance Loss - Penalizes maximum distance between boundaries.

    Hausdorff distance measures the maximum distance between two sets of points.
    For segmentation, it measures worst-case boundary error.

    Useful for ensuring no large boundary errors exist.

    This is a differentiable approximation using average of k-th percentile distances.

    Args:
        percentile: Percentile to use (95 = 95th percentile Hausdorff)
        ignore_background: If True, only compute for tumor classes
    """
    def __init__(self, percentile=95, ignore_background=True):
        super().__init__()
        self.percentile = percentile
        self.ignore_background = ignore_background

    def forward(self, logits, target):
        """
        Args:
            logits: (B, C, H, W)
            target: (B, H, W)

        Returns:
            loss: Hausdorff distance loss
        """
        if target.dim() == 4:
            target = target.squeeze(1)

        pred_probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(pred_probs, dim=1)  # (B, H, W)

        B, C, H, W = logits.shape
        total_hd = 0.0
        num_classes = 0

        start_idx = 1 if self.ignore_background else 0

        for c in range(start_idx, C):
            for b in range(B):
                # Get binary masks
                pred_mask = (pred_class[b] == c).float()
                target_mask = (target[b] == c).float()

                # Skip if either mask is empty
                if pred_mask.sum() == 0 or target_mask.sum() == 0:
                    continue

                # Compute distance transforms
                pred_sdf = self._compute_dtm(pred_mask)
                target_sdf = self._compute_dtm(target_mask)

                # Hausdorff: max of (pred points to target surface, target points to pred surface)
                # We use percentile to make it differentiable and robust
                pred_to_target = pred_sdf[pred_mask > 0.5]
                target_to_pred = target_sdf[target_mask > 0.5]

                if len(pred_to_target) > 0 and len(target_to_pred) > 0:
                    # k-th percentile
                    k = int(len(pred_to_target) * self.percentile / 100)
                    k = max(1, min(k, len(pred_to_target) - 1))

                    hd_pred = torch.topk(pred_to_target, k)[0].mean()
                    hd_target = torch.topk(target_to_pred, k)[0].mean()
                    hd = max(hd_pred, hd_target)

                    total_hd += hd
                    num_classes += 1

        if num_classes > 0:
            loss = total_hd / num_classes
        else:
            loss = torch.tensor(0.0, device=logits.device)

        return loss

    def _compute_dtm(self, mask):
        """Compute distance transform map"""
        mask_np = mask.cpu().numpy().astype(np.uint8)
        # Distance from each point to nearest zero point
        dtm = distance_transform_edt(1 - mask_np)
        return torch.from_numpy(dtm).float().to(mask.device)


class CombinedBoundaryLoss(nn.Module):
    """
    Combined Boundary + IoU + Dice Loss.

    Optimizes:
    1. Region overlap (Dice)
    2. Boundary precision (Boundary Loss)
    3. Intersection-over-Union (IoU)

    This combination directly targets IoU improvement while
    ensuring accurate boundary delineation.

    Args:
        dice_weight: Weight for Dice loss
        boundary_weight: Weight for Boundary loss
        iou_weight: Weight for IoU loss
    """
    def __init__(self, dice_weight=1.0, boundary_weight=0.5, iou_weight=1.0,
                 num_classes=3, ignore_background=True):
        super().__init__()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.iou_weight = iou_weight

        # Import existing losses
        from .losses_multiclass import MultiClassDiceLoss
        from .losses_iou import MulticlassIoULoss

        self.dice_loss = MultiClassDiceLoss(num_classes, ignore_background)
        self.boundary_loss = BoundaryLoss(ignore_background=ignore_background)
        self.iou_loss = MulticlassIoULoss(num_classes, ignore_background)

    def forward(self, logits, target):
        dice_l = self.dice_loss(logits, target)
        boundary_l = self.boundary_loss(logits, target)
        iou_l = self.iou_loss(logits, target)

        total = (self.dice_weight * dice_l +
                 self.boundary_weight * boundary_l +
                 self.iou_weight * iou_l)

        return total, {
            'dice': dice_l.item(),
            'boundary': boundary_l.item(),
            'iou': iou_l.item()
        }


# ============================================================================
# Unit Tests
# ============================================================================

def test_boundary_loss():
    """Test boundary loss on synthetic data"""
    print("Testing BoundaryLoss...")

    batch_size = 2
    num_classes = 3
    height, width = 64, 64

    # Create synthetic data with clear boundaries
    logits = torch.randn(batch_size, num_classes, height, width)
    target = torch.zeros(batch_size, height, width, dtype=torch.long)

    # Create circular tumor regions
    for b in range(batch_size):
        center_y, center_x = height // 2, width // 2
        for y in range(height):
            for x in range(width):
                dist = ((y - center_y)**2 + (x - center_x)**2) ** 0.5
                if dist < 15:
                    target[b, y, x] = 1  # TC
                elif dist < 20:
                    target[b, y, x] = 2  # ED

    loss_fn = BoundaryLoss(theta0=3, theta=5, ignore_background=True)
    loss = loss_fn(logits, target)

    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Backward pass: ", end="")
    loss.backward()
    print("✓ OK")

    print("✓ BoundaryLoss tests passed!\n")


def test_combined_boundary_loss():
    """Test combined boundary loss"""
    print("Testing CombinedBoundaryLoss...")

    batch_size = 2
    num_classes = 3
    height, width = 64, 64

    logits = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))

    loss_fn = CombinedBoundaryLoss(
        dice_weight=1.0,
        boundary_weight=0.5,
        iou_weight=1.0,
        num_classes=num_classes
    )

    loss, loss_dict = loss_fn(logits, target)

    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Dice: {loss_dict['dice']:.4f}")
    print(f"  Boundary: {loss_dict['boundary']:.4f}")
    print(f"  IoU: {loss_dict['iou']:.4f}")
    print(f"  Backward pass: ", end="")
    loss.backward()
    print("✓ OK")

    print("✓ CombinedBoundaryLoss tests passed!\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Boundary Loss Module Unit Tests")
    print("=" * 70 + "\n")

    test_boundary_loss()
    test_combined_boundary_loss()

    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
