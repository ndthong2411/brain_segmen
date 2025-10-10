"""
Multi-class metrics for BraTS tumor sub-regions.

Evaluates:
- Whole Tumor (WT) = TC + ED (classes 1 + 2)
- Tumor Core (TC) = class 1 only
- Edema (ED) = class 2 only

This avoids background dominance in metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np


def get_tumor_regions(pred, target):
    """
    Extract tumor regions from multi-class predictions.

    Args:
        pred: (B, C, H, W) predicted logits or (B, H, W) predicted classes
        target: (B, H, W) ground truth class indices {0, 1, 2}

    Returns:
        dict with binary masks for each region:
            - 'WT': Whole Tumor (TC + ED)
            - 'TC': Tumor Core
            - 'ED': Edema
    """
    # Convert prediction to class indices
    if pred.dim() == 4:  # (B, C, H, W) logits
        pred_classes = torch.argmax(pred, dim=1)  # (B, H, W)
    else:  # (B, H, W) already class indices
        pred_classes = pred

    # Ensure target is (B, H, W)
    if target.dim() == 4:
        target = target.squeeze(1)

    # Extract regions
    regions_pred = {}
    regions_target = {}

    # Whole Tumor (WT) = TC (1) + ED (2)
    regions_pred['WT'] = ((pred_classes == 1) | (pred_classes == 2)).float()
    regions_target['WT'] = ((target == 1) | (target == 2)).float()

    # Tumor Core (TC) = class 1
    regions_pred['TC'] = (pred_classes == 1).float()
    regions_target['TC'] = (target == 1).float()

    # Edema (ED) = class 2
    regions_pred['ED'] = (pred_classes == 2).float()
    regions_target['ED'] = (target == 2).float()

    return regions_pred, regions_target


def dice_score_multiclass(pred, target, region='WT', eps=1e-6):
    """
    Compute Dice score for a specific tumor region.

    Args:
        pred: (B, C, H, W) logits or (B, H, W) class indices
        target: (B, H, W) ground truth classes
        region: 'WT', 'TC', or 'ED'
        eps: Small value for numerical stability

    Returns:
        dice: Scalar Dice score for the region
    """
    regions_pred, regions_target = get_tumor_regions(pred, target)

    pred_region = regions_pred[region]
    target_region = regions_target[region]

    intersection = (pred_region * target_region).sum(dim=(1, 2))
    union = pred_region.sum(dim=(1, 2)) + target_region.sum(dim=(1, 2))

    dice = (2.0 * intersection + eps) / (union + eps)

    return dice.mean().item()


def iou_score_multiclass(pred, target, region='WT', eps=1e-6):
    """
    Compute IoU score for a specific tumor region.

    Args:
        pred: (B, C, H, W) logits or (B, H, W) class indices
        target: (B, H, W) ground truth classes
        region: 'WT', 'TC', or 'ED'
        eps: Small value for numerical stability

    Returns:
        iou: Scalar IoU score for the region
    """
    regions_pred, regions_target = get_tumor_regions(pred, target)

    pred_region = regions_pred[region]
    target_region = regions_target[region]

    intersection = (pred_region * target_region).sum(dim=(1, 2))
    union = pred_region.sum(dim=(1, 2)) + target_region.sum(dim=(1, 2)) - intersection

    iou = (intersection + eps) / (union + eps)

    return iou.mean().item()


def compute_all_region_metrics(pred, target, eps=1e-6):
    """
    Compute Dice and IoU for all tumor regions.

    Args:
        pred: (B, C, H, W) predicted logits
        target: (B, H, W) ground truth classes

    Returns:
        dict with metrics for each region:
            {
                'WT_dice': float,
                'WT_iou': float,
                'TC_dice': float,
                'TC_iou': float,
                'ED_dice': float,
                'ED_iou': float,
                'mean_dice': float,  # Average of WT, TC, ED
                'mean_iou': float
            }
    """
    metrics = {}

    for region in ['WT', 'TC', 'ED']:
        metrics[f'{region}_dice'] = dice_score_multiclass(pred, target, region, eps)
        metrics[f'{region}_iou'] = iou_score_multiclass(pred, target, region, eps)

    # Mean scores (average across regions)
    metrics['mean_dice'] = (metrics['WT_dice'] + metrics['TC_dice'] + metrics['ED_dice']) / 3.0
    metrics['mean_iou'] = (metrics['WT_iou'] + metrics['TC_iou'] + metrics['ED_iou']) / 3.0

    return metrics


def compute_region_intersection_union(pred, target, region='WT'):
    """
    Compute intersection and union for a region (for accumulation across batches).

    Args:
        pred: (B, C, H, W) predicted logits
        target: (B, H, W) ground truth classes
        region: 'WT', 'TC', or 'ED'

    Returns:
        intersection: scalar
        union: scalar
    """
    regions_pred, regions_target = get_tumor_regions(pred, target)

    pred_region = regions_pred[region]
    target_region = regions_target[region]

    intersection = (pred_region * target_region).sum().item()
    union = pred_region.sum().item() + target_region.sum().item()

    return intersection, union


class RegionMetricsAccumulator:
    """
    Accumulator for computing metrics across entire dataset.

    Usage:
        acc = RegionMetricsAccumulator()
        for batch in loader:
            pred, target = model(batch)
            acc.update(pred, target)
        metrics = acc.compute()
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {
            'WT': {'intersection': 0.0, 'union': 0.0},
            'TC': {'intersection': 0.0, 'union': 0.0},
            'ED': {'intersection': 0.0, 'union': 0.0},
        }

    def update(self, pred, target):
        """
        Update accumulated metrics with a batch.

        Args:
            pred: (B, C, H, W) predicted logits
            target: (B, H, W) ground truth classes
        """
        for region in ['WT', 'TC', 'ED']:
            inter, union = compute_region_intersection_union(pred, target, region)
            self.metrics[region]['intersection'] += inter
            self.metrics[region]['union'] += union

    def compute(self, eps=1e-6):
        """
        Compute final Dice and IoU scores.

        Returns:
            dict with metrics for all regions
        """
        results = {}

        for region in ['WT', 'TC', 'ED']:
            inter = self.metrics[region]['intersection']
            union = self.metrics[region]['union']

            # Dice
            dice = (2.0 * inter + eps) / (union + eps)
            results[f'{region}_dice'] = dice

            # IoU
            iou = (inter + eps) / (union - inter + eps)
            results[f'{region}_iou'] = iou

        # Mean scores
        results['mean_dice'] = (results['WT_dice'] + results['TC_dice'] + results['ED_dice']) / 3.0
        results['mean_iou'] = (results['WT_iou'] + results['TC_iou'] + results['ED_iou']) / 3.0

        return results


# Backward compatibility aliases
def dice_score_wt(pred, target, eps=1e-6):
    """Whole Tumor Dice score."""
    return dice_score_multiclass(pred, target, 'WT', eps)


def dice_score_tc(pred, target, eps=1e-6):
    """Tumor Core Dice score."""
    return dice_score_multiclass(pred, target, 'TC', eps)


def dice_score_ed(pred, target, eps=1e-6):
    """Edema Dice score."""
    return dice_score_multiclass(pred, target, 'ED', eps)


def iou_score_wt(pred, target, eps=1e-6):
    """Whole Tumor IoU score."""
    return iou_score_multiclass(pred, target, 'WT', eps)


def iou_score_tc(pred, target, eps=1e-6):
    """Tumor Core IoU score."""
    return iou_score_multiclass(pred, target, 'TC', eps)


def iou_score_ed(pred, target, eps=1e-6):
    """Edema IoU score."""
    return iou_score_multiclass(pred, target, 'ED', eps)
