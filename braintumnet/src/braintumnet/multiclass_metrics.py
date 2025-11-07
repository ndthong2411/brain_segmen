"""
Multi-class Segmentation Metrics for BraTS Tumor Regions (STANDARD 4-class)

For 4-class segmentation:
- Class 0: Background
- Class 1: NCR/NET (Necrotic/Non-enhancing)
- Class 2: ED (Edema/Peritumoral)
- Class 3: ET (Enhancing Tumor)

Evaluation regions (STANDARD BraTS):
- ET (Enhancing Tumor) = class 3 only
- TC (Tumor Core) = classes 1 + 3 (NCR + ET)
- WT (Whole Tumor) = classes 1 + 2 + 3 (all tumor)
"""

import torch
import numpy as np
from typing import Dict, Tuple
from .metrics import compute_hausdorff_distance_95


def multiclass_dice_coefficient(pred: torch.Tensor, target: torch.Tensor,
                                  class_idx: int, eps: float = 1e-6) -> float:
    """
    Compute Dice coefficient for a specific class.

    Args:
        pred: (B, C, H, W) - predicted logits
        target: (B, 1, H, W) - integer class labels [0, C-1]
        class_idx: Which class to compute Dice for
        eps: Smoothing factor

    Returns:
        Dice coefficient for the specified class
    """
    # Get HARD predictions using argmax
    pred_classes = torch.argmax(pred, dim=1)  # (B, H, W) - integer [0, C-1]
    pred_class = (pred_classes == class_idx).float()  # (B, H, W) - binary 0/1

    # Get ground truth for this class
    target_class = (target.squeeze(1) == class_idx).float()  # (B, H, W)

    # Compute intersection and union
    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum()

    # Compute Dice
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.item()


def multiclass_iou(pred: torch.Tensor, target: torch.Tensor,
                   class_idx: int, eps: float = 1e-6) -> float:
    """
    Compute IoU for a specific class.

    Args:
        pred: (B, C, H, W) - predicted logits
        target: (B, 1, H, W) - integer class labels [0, C-1]
        class_idx: Which class to compute IoU for
        eps: Smoothing factor

    Returns:
        IoU for the specified class
    """
    # Get HARD predictions using argmax
    pred_classes = torch.argmax(pred, dim=1)  # (B, H, W) - integer [0, C-1]
    pred_class = (pred_classes == class_idx).float()  # (B, H, W) - binary 0/1

    # Get ground truth for this class
    target_class = (target.squeeze(1) == class_idx).float()  # (B, H, W)

    # Compute intersection and union
    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum() - intersection

    # Compute IoU
    iou = (intersection + eps) / (union + eps)
    return iou.item()


def compute_brats_regions(pred: torch.Tensor, target: torch.Tensor,
                          num_classes: int = 3) -> Dict[str, Tuple[float, float]]:
    """
    Compute Dice and IoU for BraTS tumor regions.

    For 3-class segmentation:
    - WT (Whole Tumor) = TC + ED (classes 1, 2)
    - TC (Tumor Core) = class 1 only
    - ED (Edema) = class 2 only

    Args:
        pred: (B, C, H, W) - predicted logits
        target: (B, 1, H, W) - integer class labels [0, 1, 2]
        num_classes: Number of classes (default 3)

    Returns:
        Dictionary with metrics:
        {
            'WT_dice': float, 'WT_iou': float,
            'TC_dice': float, 'TC_iou': float,
            'ED_dice': float, 'ED_iou': float,
            'mean_dice': float, 'mean_iou': float
        }
    """
    metrics = {}
    eps = 1e-6

    # Get HARD predictions using argmax (not soft probabilities!)
    pred_classes = torch.argmax(pred, dim=1)  # (B, H, W) - integer [0, C-1]
    target_squeezed = target.squeeze(1)  # (B, H, W)

    # --- TC (Tumor Core) = class 1 only ---
    pred_tc = (pred_classes == 1).float()  # (B, H, W) - binary 0/1
    target_tc = (target_squeezed == 1).float()  # (B, H, W)

    inter_tc = (pred_tc * target_tc).sum()
    union_tc = pred_tc.sum() + target_tc.sum()

    dice_tc = (2.0 * inter_tc + eps) / (union_tc + eps)
    iou_tc = (inter_tc + eps) / (union_tc - inter_tc + eps)

    metrics['TC_dice'] = dice_tc.item()
    metrics['TC_iou'] = iou_tc.item()

    # --- ED (Edema) = class 2 only ---
    if num_classes >= 3:
        pred_ed = (pred_classes == 2).float()  # (B, H, W) - binary 0/1
        target_ed = (target_squeezed == 2).float()  # (B, H, W)

        inter_ed = (pred_ed * target_ed).sum()
        union_ed = pred_ed.sum() + target_ed.sum()

        dice_ed = (2.0 * inter_ed + eps) / (union_ed + eps)
        iou_ed = (inter_ed + eps) / (union_ed - inter_ed + eps)

        metrics['ED_dice'] = dice_ed.item()
        metrics['ED_iou'] = iou_ed.item()
    else:
        metrics['ED_dice'] = 0.0
        metrics['ED_iou'] = 0.0

    # --- WT (Whole Tumor) = TC + ED (classes 1, 2) ---
    if num_classes >= 3:
        pred_wt = (pred_classes >= 1).float()  # (B, H, W) - any tumor class
        target_wt = (target_squeezed >= 1).float()  # (B, H, W) - any tumor class
    else:
        pred_wt = pred_tc
        target_wt = target_tc

    inter_wt = (pred_wt * target_wt).sum()
    union_wt = pred_wt.sum() + target_wt.sum()

    dice_wt = (2.0 * inter_wt + eps) / (union_wt + eps)
    iou_wt = (inter_wt + eps) / (union_wt - inter_wt + eps)

    metrics['WT_dice'] = dice_wt.item()
    metrics['WT_iou'] = iou_wt.item()

    # --- Mean metrics (average of WT, TC, ED) ---
    if num_classes >= 3:
        metrics['mean_dice'] = (metrics['WT_dice'] + metrics['TC_dice'] + metrics['ED_dice']) / 3.0
        metrics['mean_iou'] = (metrics['WT_iou'] + metrics['TC_iou'] + metrics['ED_iou']) / 3.0
    else:
        metrics['mean_dice'] = (metrics['WT_dice'] + metrics['TC_dice']) / 2.0
        metrics['mean_iou'] = (metrics['WT_iou'] + metrics['TC_iou']) / 2.0

    return metrics


class MulticlassMetricsAccumulator:
    """
    Accumulates intersection and union for each region across batches,
    then computes global Dice/IoU/HD95 at the end.

    Supports both 3-class (legacy) and 4-class (standard BraTS) schemes.

    This is the CORRECT way to compute metrics - accumulate then divide,
    not average per-batch metrics.
    """
    def __init__(self, num_classes: int = 4, compute_hd95: bool = True):
        self.num_classes = num_classes
        self.compute_hd95 = compute_hd95
        self.reset()

    def reset(self):
        """Reset all accumulators"""
        self.inter_wt = 0.0
        self.union_wt = 0.0
        self.inter_tc = 0.0
        self.union_tc = 0.0
        self.inter_ed = 0.0
        self.union_ed = 0.0
        self.inter_et = 0.0  # New for 4-class
        self.union_et = 0.0  # New for 4-class
        self.n_batches = 0

        # HD95 accumulators
        self.hd95_wt_sum = 0.0
        self.hd95_wt_count = 0
        self.hd95_tc_sum = 0.0
        self.hd95_tc_count = 0
        self.hd95_ed_sum = 0.0
        self.hd95_ed_count = 0
        self.hd95_et_sum = 0.0  # New for 4-class
        self.hd95_et_count = 0  # New for 4-class

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update accumulators with a new batch.

        Supports both 3-class and 4-class:
        - 3-class: 0=BG, 1=TC, 2=ED (legacy, ET merged into TC)
        - 4-class: 0=BG, 1=NCR, 2=ED, 3=ET (standard BraTS)

        Args:
            pred: (B, C, H, W) - predicted logits
            target: (B, 1, H, W) - integer class labels
        """
        # Get HARD predictions using argmax (not soft probabilities!)
        # This is critical for correct Dice/IoU calculation
        pred_classes = torch.argmax(pred, dim=1)  # (B, H, W) - integer [0, C-1]
        target_squeezed = target.squeeze(1)  # (B, H, W)

        if self.num_classes == 4:
            # ============ 4-CLASS (STANDARD BraTS) ============
            # ET (Enhancing Tumor) = class 3 only
            pred_et = (pred_classes == 3).float()
            target_et = (target_squeezed == 3).float()

            self.inter_et += (pred_et * target_et).sum().item()
            self.union_et += (pred_et.sum() + target_et.sum()).item()

            # TC (Tumor Core) = classes 1 + 3 (NCR + ET)
            pred_tc = ((pred_classes == 1) | (pred_classes == 3)).float()
            target_tc = ((target_squeezed == 1) | (target_squeezed == 3)).float()

            self.inter_tc += (pred_tc * target_tc).sum().item()
            self.union_tc += (pred_tc.sum() + target_tc.sum()).item()

            # ED (Edema) = class 2 only
            pred_ed = (pred_classes == 2).float()
            target_ed = (target_squeezed == 2).float()

            self.inter_ed += (pred_ed * target_ed).sum().item()
            self.union_ed += (pred_ed.sum() + target_ed.sum()).item()

            # WT (Whole Tumor) = classes 1 + 2 + 3 (all tumor)
            pred_wt = (pred_classes >= 1).float()
            target_wt = (target_squeezed >= 1).float()

            self.inter_wt += (pred_wt * target_wt).sum().item()
            self.union_wt += (pred_wt.sum() + target_wt.sum()).item()

        else:
            # ============ 3-CLASS (LEGACY) ============
            # TC = class 1 only (ET already merged)
            pred_tc = (pred_classes == 1).float()
            target_tc = (target_squeezed == 1).float()

            self.inter_tc += (pred_tc * target_tc).sum().item()
            self.union_tc += (pred_tc.sum() + target_tc.sum()).item()

            # ED = class 2 only
            if self.num_classes >= 3:
                pred_ed = (pred_classes == 2).float()
                target_ed = (target_squeezed == 2).float()

                self.inter_ed += (pred_ed * target_ed).sum().item()
                self.union_ed += (pred_ed.sum() + target_ed.sum()).item()

            # WT = TC + ED
            if self.num_classes >= 3:
                pred_wt = (pred_classes >= 1).float()
                target_wt = (target_squeezed >= 1).float()
            else:
                pred_wt = pred_tc
                target_wt = target_tc

            self.inter_wt += (pred_wt * target_wt).sum().item()
            self.union_wt += (pred_wt.sum() + target_wt.sum()).item()

        # --- Compute HD95 if enabled ---
        if self.compute_hd95:
            # Convert to numpy for HD95 computation
            pred_wt_np = pred_wt.cpu().numpy()
            target_wt_np = target_wt.cpu().numpy()
            pred_tc_np = pred_tc.cpu().numpy()
            target_tc_np = target_tc.cpu().numpy()

            batch_size = pred_wt_np.shape[0]

            for i in range(batch_size):
                # WT HD95
                pred_count_wt = np.sum(pred_wt_np[i] > 0)
                target_count_wt = np.sum(target_wt_np[i] > 0)
                if pred_count_wt > 0 and target_count_wt > 0:
                    try:
                        hd95_val = compute_hausdorff_distance_95(pred_wt_np[i], target_wt_np[i])
                        if not np.isinf(hd95_val) and not np.isnan(hd95_val):
                            self.hd95_wt_sum += hd95_val
                            self.hd95_wt_count += 1
                    except Exception:
                        pass

                # TC HD95
                pred_count_tc = np.sum(pred_tc_np[i] > 0)
                target_count_tc = np.sum(target_tc_np[i] > 0)
                if pred_count_tc > 0 and target_count_tc > 0:
                    try:
                        hd95_val = compute_hausdorff_distance_95(pred_tc_np[i], target_tc_np[i])
                        if not np.isinf(hd95_val) and not np.isnan(hd95_val):
                            self.hd95_tc_sum += hd95_val
                            self.hd95_tc_count += 1
                    except Exception:
                        pass

                # ED HD95
                if self.num_classes >= 3:
                    pred_ed_np = pred_ed.cpu().numpy()
                    target_ed_np = target_ed.cpu().numpy()
                    pred_count_ed = np.sum(pred_ed_np[i] > 0)
                    target_count_ed = np.sum(target_ed_np[i] > 0)
                    if pred_count_ed > 0 and target_count_ed > 0:
                        try:
                            hd95_val = compute_hausdorff_distance_95(pred_ed_np[i], target_ed_np[i])
                            if not np.isinf(hd95_val) and not np.isnan(hd95_val):
                                self.hd95_ed_sum += hd95_val
                                self.hd95_ed_count += 1
                        except Exception:
                            pass

                # ET HD95 (only for 4-class)
                if self.num_classes == 4:
                    pred_et_np = pred_et.cpu().numpy()
                    target_et_np = target_et.cpu().numpy()
                    pred_count_et = np.sum(pred_et_np[i] > 0)
                    target_count_et = np.sum(target_et_np[i] > 0)
                    if pred_count_et > 0 and target_count_et > 0:
                        try:
                            hd95_val = compute_hausdorff_distance_95(pred_et_np[i], target_et_np[i])
                            if not np.isinf(hd95_val) and not np.isnan(hd95_val):
                                self.hd95_et_sum += hd95_val
                                self.hd95_et_count += 1
                        except Exception:
                            pass

        self.n_batches += 1

    def get_metrics(self) -> Dict[str, float]:
        """
        Compute final global metrics from accumulated values.

        Returns:
            Dictionary with all metrics including HD95.
            For 4-class: includes ET metrics (STANDARD BraTS).
            For 3-class: ET metrics are 0.0 (legacy).
        """
        eps = 1e-6

        # Compute Dice and IoU for WT, TC, ED
        dice_wt = (2.0 * self.inter_wt + eps) / (self.union_wt + eps)
        iou_wt = (self.inter_wt + eps) / (self.union_wt - self.inter_wt + eps)

        dice_tc = (2.0 * self.inter_tc + eps) / (self.union_tc + eps)
        iou_tc = (self.inter_tc + eps) / (self.union_tc - self.inter_tc + eps)

        dice_ed = (2.0 * self.inter_ed + eps) / (self.union_ed + eps)
        iou_ed = (self.inter_ed + eps) / (self.union_ed - self.inter_ed + eps)

        # Compute ET metrics (only for 4-class)
        if self.num_classes == 4:
            dice_et = (2.0 * self.inter_et + eps) / (self.union_et + eps)
            iou_et = (self.inter_et + eps) / (self.union_et - self.inter_et + eps)
            # Standard BraTS: mean of ET, TC, WT (not ED)
            mean_dice = (dice_et + dice_tc + dice_wt) / 3.0
            mean_iou = (iou_et + iou_tc + iou_wt) / 3.0
        else:
            dice_et = 0.0
            iou_et = 0.0
            # Legacy 3-class: mean of WT, TC, ED
            mean_dice = (dice_wt + dice_tc + dice_ed) / 3.0
            mean_iou = (iou_wt + iou_tc + iou_ed) / 3.0

        # Compute HD95 for each region
        hd95_wt = self.hd95_wt_sum / self.hd95_wt_count if self.hd95_wt_count > 0 else -1.0
        hd95_tc = self.hd95_tc_sum / self.hd95_tc_count if self.hd95_tc_count > 0 else -1.0
        hd95_ed = self.hd95_ed_sum / self.hd95_ed_count if self.hd95_ed_count > 0 else -1.0
        hd95_et = self.hd95_et_sum / self.hd95_et_count if self.hd95_et_count > 0 else -1.0

        # Compute mean HD95 (only from valid values)
        valid_hd95_values = []
        if hd95_wt >= 0:
            valid_hd95_values.append(hd95_wt)
        if hd95_tc >= 0:
            valid_hd95_values.append(hd95_tc)
        if hd95_et >= 0 and self.num_classes == 4:
            valid_hd95_values.append(hd95_et)
        # Note: ED not included in standard BraTS mean

        mean_hd95 = np.mean(valid_hd95_values) if len(valid_hd95_values) > 0 else -1.0

        return {
            'ET_dice': dice_et,  # NEW for 4-class
            'ET_iou': iou_et,    # NEW for 4-class
            'ET_hd95': hd95_et,  # NEW for 4-class
            'TC_dice': dice_tc,
            'TC_iou': iou_tc,
            'TC_hd95': hd95_tc,
            'WT_dice': dice_wt,
            'WT_iou': iou_wt,
            'WT_hd95': hd95_wt,
            'ED_dice': dice_ed,
            'ED_iou': iou_ed,
            'ED_hd95': hd95_ed,
            'mean_dice': mean_dice,
            'mean_iou': mean_iou,
            'mean_hd95': mean_hd95,
        }

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metrics (same as get_metrics, for compatibility)"""
        return self.get_metrics()


def get_multiclass_predictions(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert multiclass logits to predicted class labels.

    Args:
        logits: (B, C, H, W) - raw logits from model

    Returns:
        (B, 1, H, W) - predicted class labels [0, C-1]
    """
    pred_classes = torch.argmax(logits, dim=1, keepdim=True)  # (B, 1, H, W)
    return pred_classes


def visualize_multiclass_prediction(class_labels: torch.Tensor) -> torch.Tensor:
    """
    Convert integer class labels to RGB visualization.

    For 3-class:
    - Background (0) = Black (0, 0, 0)
    - Tumor Core (1) = Red (1, 0, 0)
    - Edema (2) = Green (0, 1, 0)

    Args:
        class_labels: (B, H, W) or (B, 1, H, W) - integer class labels [0, 1, 2]

    Returns:
        rgb: (B, 3, H, W) - RGB visualization in range [0, 1]
    """
    # Handle both (B, H, W) and (B, 1, H, W) shapes
    if class_labels.ndim == 4:
        class_labels = class_labels.squeeze(1)  # (B, H, W)

    B, H, W = class_labels.shape

    # Create RGB image
    rgb = torch.zeros(B, 3, H, W, device=class_labels.device, dtype=torch.float32)

    # Color mapping: Background=Black, TC=Red, ED=Green
    # Class 0 (Background): RGB = (0, 0, 0) - already zero
    # Class 1 (TC): RGB = (1, 0, 0) - Red
    rgb[:, 0, :, :][class_labels == 1] = 1.0
    # Class 2 (ED): RGB = (0, 1, 0) - Green
    rgb[:, 1, :, :][class_labels == 2] = 1.0

    return rgb
