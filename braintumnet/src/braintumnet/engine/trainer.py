import os, math, torch, time
from torch.utils.data import DataLoader
from typing import Dict
from ..models.braintumnet import BrainTumNet
from ..data.brats2020_dataset import SliceDataset
from ..losses import MultiTaskLoss
from ..metrics import iou_score, dice_score
from ..utils.io import ensure_dir, save_ckpt

def _cosine_lr(optimizer, base_lr, t, T):
    lr = 0.5 * base_lr * (1 + math.cos(math.pi * t / T))
    for pg in optimizer.param_groups: pg["lr"] = lr

def build_dataloaders(cfg: Dict, fold: int):
    proc = cfg["data"]["proc_root"]
    img_size = cfg["data"]["img_size"]
    train_list = os.path.join(proc, f"split_train_fold{fold}.txt")
    val_list   = os.path.join(proc, f"split_val_fold{fold}.txt")
    train_ds = SliceDataset(proc, train_list, img_size, cfg["augment"]["rotate_deg"],
                            cfg["augment"]["hflip_p"], cfg["augment"]["vflip_p"], True, cfg["model"]["in_channels"])
    val_ds   = SliceDataset(proc, val_list, img_size, 0,0,0, False, cfg["model"]["in_channels"])
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["workers"])
    val_loader   = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["workers"])
    return train_loader, val_loader

def build_model(cfg: Dict):
    mcfg = cfg["model"]
    return BrainTumNet(in_ch=mcfg["in_channels"], num_cls=mcfg["num_classes_cls"], base=mcfg["base"],
                       dim=mcfg["dim"], patch=mcfg["patch_size"], depth=mcfg["depth"], n_heads=mcfg["n_heads"],
                       roi_stop_grad=mcfg["roi_stop_grad"])

def train_one_fold(cfg: Dict, fold: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = build_dataloaders(cfg, fold)
    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    crit = MultiTaskLoss(cfg["train"]["seg_loss_weight"], cfg["train"]["cls_loss_weight"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("amp", False))

    total_steps = cfg["train"]["epochs"] * max(1, len(train_loader))
    step = 0
    best_iou = -1.0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for batch in train_loader:
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)
            lab = batch["label"].to(device)
            with torch.cuda.amp.autocast(enabled=cfg["train"].get("amp", False)):
                seg, cls = model(img)
                loss, _, _ = crit(seg, msk, cls, lab)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if cfg["train"]["scheduler"] == "cosine":
                _cosine_lr(opt, cfg["train"]["lr"], step, total_steps)
            step += 1

        # validation
        model.eval()
        iou_m, dice_m, acc_m, n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                msk = batch["mask"].to(device)
                lab = batch["label"].to(device)
                seg, cls = model(img)
                iou_m += iou_score(seg, msk)
                dice_m += dice_score(seg, msk)
                acc_m += (cls.argmax(1)==lab).float().mean().item()
                n += 1
        iou_m /= n; dice_m /= n; acc_m /= n
        print(f"[Fold {fold}] Epoch {epoch+1}/{cfg['train']['epochs']} | IoU {iou_m:.4f} | Dice {dice_m:.4f} | ClsAcc {acc_m:.4f}")

        if iou_m > best_iou:
            best_iou = iou_m
            ckpt_dir = cfg["logging"]["save_dir"]
            ensure_dir(ckpt_dir)
            save_ckpt(model, os.path.join(ckpt_dir, f"braintumnet_best_fold{fold}.pth"))

    return best_iou
