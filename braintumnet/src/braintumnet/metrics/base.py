import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.spatial.distance import directed_hausdorff
from typing import Tuple, Optional

def binarize(logits: torch.Tensor, thr: float=0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > thr).float()

def compute_intersection_union(logits: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """
    Compute intersection and union for global IoU/Dice calculation.
    This is the CORRECT way to compute metrics across batches.

    Returns:
        intersection: Total intersection count
        union: Total union count (pred + target)
    """
    pred = binarize(logits)
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item()
    return inter, union

def iou_score(logits: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
    """
    DEPRECATED: Use compute_intersection_union() for correct global metrics.
    This function averages batch scores which is mathematically incorrect.
    Kept for backward compatibility only.
    """
    pred = binarize(logits)
    inter = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) - inter + eps
    return ((inter + eps) / union).mean().item()

def dice_score(logits: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
    """
    DEPRECATED: Use compute_intersection_union() for correct global metrics.
    This function averages batch scores which is mathematically incorrect.
    Kept for backward compatibility only.
    """
    pred = binarize(logits)
    num = 2 * (pred * target).sum(dim=(2,3))
    den = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps
    return (num/den).mean().item()

def hd95_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    try:
        from scipy.spatial.distance import cdist
        pred_pts = np.argwhere(pred_mask > 0)
        gt_pts = np.argwhere(gt_mask > 0)
        if len(pred_pts)==0 or len(gt_pts)==0:
            return float("inf")
        D = cdist(pred_pts, gt_pts)
        return np.percentile(np.hstack([D.min(axis=1), D.min(axis=0)]), 95)
    except Exception:
        return float("nan")

def cls_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Tuple[float,float,float]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    auc = float("nan")
    try:
        ncls = y_prob.shape[1]
        if ncls == 2:
            auc = roc_auc_score(y_true, y_prob[:,1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        pass
    return acc, f1, auc


# ============================================================================
# ENHANCED EVALUATION METRICS
# ============================================================================

def compute_dice_coefficient(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """
    Dice Similarity Coefficient (DSC).

    DSC = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        pred: Binary prediction mask (0 or 1)
        target: Binary ground truth mask (0 or 1)
        eps: Small epsilon to avoid division by zero

    Returns:
        Dice coefficient [0, 1], where 1 is perfect overlap
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()

    # Handle edge cases
    if pred_sum == 0 and target_sum == 0:
        return 1.0  # Both empty = perfect match
    if pred_sum == 0 or target_sum == 0:
        return 0.0  # One empty, one not = no overlap

    dice = (2.0 * intersection) / (pred_sum + target_sum + eps)
    return float(dice)


def compute_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """
    Intersection over Union (IoU / Jaccard Index).

    IoU = |A ∩ B| / |A ∪ B|

    Args:
        pred: Binary prediction mask (0 or 1)
        target: Binary ground truth mask (0 or 1)
        eps: Small epsilon to avoid division by zero

    Returns:
        IoU score [0, 1], where 1 is perfect overlap
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    # Handle edge cases
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / (union + eps)
    return float(iou)


def compute_hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Hausdorff Distance (HD) - maximum distance from a point in one set
    to the nearest point in the other set.

    HD(A, B) = max(max_a min_b d(a,b), max_b min_a d(b,a))

    Lower is better (0 = perfect overlap).

    Args:
        pred: Binary prediction mask (0 or 1)
        target: Binary ground truth mask (0 or 1)

    Returns:
        Hausdorff distance in pixels. Returns inf if either mask is empty.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    # Get boundary points
    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)

    # Handle empty masks
    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')

    # Compute symmetric Hausdorff distance
    hd_forward = directed_hausdorff(pred_points, target_points)[0]
    hd_backward = directed_hausdorff(target_points, pred_points)[0]
    hd = max(hd_forward, hd_backward)

    return float(hd)


def compute_hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    95th percentile Hausdorff Distance (HD95) - more robust to outliers.

    Instead of using the maximum distance (which is sensitive to outliers),
    uses the 95th percentile of distances.

    Lower is better (0 = perfect overlap).

    Args:
        pred: Binary prediction mask (0 or 1)
        target: Binary ground truth mask (0 or 1)

    Returns:
        95th percentile Hausdorff distance in pixels. Returns inf if either mask is empty.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    # Get boundary points
    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)

    # Handle empty masks
    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')

    # Compute distances from each point to nearest point in other set
    from scipy.spatial.distance import cdist
    distances_matrix = cdist(pred_points, target_points)

    # Minimum distance from each pred point to any target point
    min_dist_pred_to_target = distances_matrix.min(axis=1)
    # Minimum distance from each target point to any pred point
    min_dist_target_to_pred = distances_matrix.min(axis=0)

    # Combine all minimum distances
    all_distances = np.concatenate([min_dist_pred_to_target, min_dist_target_to_pred])

    # Return 95th percentile
    hd95 = np.percentile(all_distances, 95)
    return float(hd95)


def compute_segmentation_metrics(pred: np.ndarray, target: np.ndarray,
                                  compute_hd: bool = True,
                                  compute_hd95: bool = True) -> dict:
    """
    Compute all segmentation metrics for a single prediction-target pair.

    Args:
        pred: Binary prediction mask (0 or 1)
        target: Binary ground truth mask (0 or 1)
        compute_hd: Whether to compute Hausdorff Distance (can be slow)
        compute_hd95: Whether to compute HD95 (can be slow)

    Returns:
        Dictionary with keys: 'dice', 'iou', 'hd', 'hd95'
    """
    metrics = {
        'dice': compute_dice_coefficient(pred, target),
        'iou': compute_iou(pred, target),
    }

    if compute_hd:
        try:
            metrics['hd'] = compute_hausdorff_distance(pred, target)
        except Exception:
            metrics['hd'] = float('nan')

    if compute_hd95:
        try:
            metrics['hd95'] = compute_hausdorff_distance_95(pred, target)
        except Exception:
            metrics['hd95'] = float('nan')

    return metrics
