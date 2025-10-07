import os, torch, numpy as np
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm
from ..models.braintumnet import BrainTumNet
from ..data.brats2020_dataset import SliceDataset
from ..metrics import (cls_metrics, compute_intersection_union,
                       compute_segmentation_metrics, binarize)
from ..utils.io import load_ckpt

def evaluate(cfg: Dict, fold: int, ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = cfg["data"]["proc_root"]
    val_list = os.path.join(proc, f"split_val_fold{fold}.txt")
    ds = SliceDataset(proc, val_list, cfg["data"]["img_size"], train=False, in_channels=cfg["model"]["in_channels"])
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["workers"])
    model = BrainTumNet(in_ch=cfg["model"]["in_channels"], num_cls=cfg["model"]["num_classes_cls"],
                        base=cfg["model"]["base"], dim=cfg["model"]["dim"], patch=cfg["model"]["patch_size"],
                        depth=cfg["model"]["depth"], n_heads=cfg["model"]["n_heads"],
                        roi_stop_grad=cfg["model"]["roi_stop_grad"]).to(device)
    load_ckpt(model, ckpt_path, map_location=device)
    model.eval()

    # Classification metrics
    y_true, y_pred, y_prob = [], [], []

    # Segmentation metrics (global)
    total_inter, total_union = 0.0, 0.0

    # Per-slice metrics for HD and HD95 (accumulated)
    hd_scores = []
    hd95_scores = []

    import torch.nn.functional as F
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"Evaluating Fold {fold}"):
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)
            lab = batch["label"].cpu().numpy()
            seg, cls = model(img)

            # Classification
            prob = F.softmax(cls, dim=1).cpu().numpy()
            y_true.extend(lab.tolist())
            y_pred.extend(prob.argmax(1).tolist())
            y_prob.extend(prob.tolist())

            # Segmentation (accumulate global metrics)
            inter, union = compute_intersection_union(seg, msk)
            total_inter += inter
            total_union += union

            # Per-slice HD and HD95 (on CPU)
            pred_masks = binarize(seg).cpu().numpy()
            target_masks = msk.cpu().numpy()

            for pred_slice, target_slice in zip(pred_masks, target_masks):
                # Only compute HD/HD95 if there's tumor in ground truth
                if target_slice.sum() > 0:
                    metrics = compute_segmentation_metrics(
                        pred_slice.squeeze(),
                        target_slice.squeeze(),
                        compute_hd=True,
                        compute_hd95=True
                    )
                    if not np.isinf(metrics['hd']) and not np.isnan(metrics['hd']):
                        hd_scores.append(metrics['hd'])
                    if not np.isinf(metrics['hd95']) and not np.isnan(metrics['hd95']):
                        hd95_scores.append(metrics['hd95'])

    # Compute classification metrics
    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
    acc, f1, auc = cls_metrics(y_true, y_pred, y_prob)

    # Compute segmentation metrics
    eps = 1e-6
    iou = total_inter / (total_union - total_inter + eps)
    dice = (2 * total_inter) / (total_union + eps)

    # Average HD and HD95
    hd_mean = np.mean(hd_scores) if len(hd_scores) > 0 else float('nan')
    hd95_mean = np.mean(hd95_scores) if len(hd95_scores) > 0 else float('nan')
    hd_std = np.std(hd_scores) if len(hd_scores) > 0 else float('nan')
    hd95_std = np.std(hd95_scores) if len(hd95_scores) > 0 else float('nan')

    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS - Fold {fold}")
    print("=" * 70)
    print("\nSegmentation Metrics:")
    print(f"  IoU (Jaccard):        {iou:.4f}")
    print(f"  Dice (F1):            {dice:.4f}")
    print(f"  Hausdorff Distance:   {hd_mean:.2f} ± {hd_std:.2f} pixels")
    print(f"  HD95 (95th percentile): {hd95_mean:.2f} ± {hd95_std:.2f} pixels")
    print(f"  (HD computed on {len(hd_scores)} slices with tumor)")
    print("\nClassification Metrics:")
    print(f"  Accuracy:             {acc:.4f}")
    print(f"  F1 Score:             {f1:.4f}")
    print(f"  AUC-ROC:              {auc:.4f}")
    print("=" * 70 + "\n")

    return {
        'iou': iou,
        'dice': dice,
        'hd': hd_mean,
        'hd95': hd95_mean,
        'hd_std': hd_std,
        'hd95_std': hd95_std,
        'acc': acc,
        'f1': f1,
        'auc': auc
    }
