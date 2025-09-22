import os, torch, numpy as np
from torch.utils.data import DataLoader
from typing import Dict
from ..models.braintumnet import BrainTumNet
from ..data.brats2020_dataset import SliceDataset
from ..metrics import cls_metrics
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
    y_true, y_pred, y_prob = [], [], []
    import torch.nn.functional as F
    with torch.no_grad():
        for batch in dl:
            img = batch["image"].to(device)
            lab = batch["label"].cpu().numpy()
            seg, cls = model(img)
            prob = F.softmax(cls, dim=1).cpu().numpy()
            y_true.extend(lab.tolist())
            y_pred.extend(prob.argmax(1).tolist())
            y_prob.extend(prob.tolist())
    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
    acc, f1, auc = cls_metrics(y_true, y_pred, y_prob)
    print(f"[Fold {fold}] ACC {acc:.4f} | F1 {f1:.4f} | AUC {auc:.4f}")
