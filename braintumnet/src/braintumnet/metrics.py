import torch, numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Tuple

def binarize(logits: torch.Tensor, thr: float=0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > thr).float()

def iou_score(logits: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
    pred = binarize(logits)
    inter = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) - inter + eps
    return ((inter + eps) / union).mean().item()

def dice_score(logits: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
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
