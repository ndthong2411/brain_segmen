import os, math, torch, time
from torch.utils.data import DataLoader
from typing import Dict
from ..models.braintumnet import BrainTumNet
from ..data.brats2020_dataset import SliceDataset
from ..losses import MultiTaskLoss
from ..metrics import iou_score, dice_score
from ..utils.io import ensure_dir, save_ckpt
from ..utils.logger import TrainingLogger
from ..utils.metrics_logger import MetricsLogger
from ..metrics import compute_intersection_union

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

def _cosine_lr_with_warmup(optimizer, base_lr, t, T, warmup_steps=500, min_lr=1e-6):
    """Cosine learning rate with warmup and minimum LR to prevent hitting zero"""
    if t < warmup_steps:
        # Linear warmup
        lr = base_lr * (t / warmup_steps)
    else:
        # Cosine decay with minimum LR
        progress = (t - warmup_steps) / (T - warmup_steps)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
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

def train_one_fold(cfg: Dict, fold: int, config_path: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize loggers
    log_dir = cfg["logging"].get("log_dir", "logs")
    logger = TrainingLogger(log_dir, cfg["exp_name"], fold)
    metrics_logger = MetricsLogger(log_dir, cfg["exp_name"], fold)

    # Save configuration
    if config_path and os.path.exists(config_path):
        logger.save_config(cfg, config_path)

    logger.info(f"Training on device: {device}")

    train_loader, val_loader = build_dataloaders(cfg, fold)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model(cfg).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    crit = MultiTaskLoss(cfg["train"]["seg_loss_weight"], cfg["train"]["cls_loss_weight"])
    scaler = torch.amp.GradScaler(device='cuda', enabled=cfg["train"].get("amp", False))

    # ReduceLROnPlateau scheduler for adaptive learning rate
    plateau_scheduler = None
    if cfg["train"]["scheduler"] == "plateau":
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=0.5, patience=10, min_lr=1e-7
        )

    # TensorBoard
    writer = None
    if HAS_TENSORBOARD and cfg["logging"].get("use_tensorboard", True):
        tb_log_dir = os.path.join(cfg["logging"]["out_dir"], f"{cfg['exp_name']}_fold{fold}")
        ensure_dir(tb_log_dir)
        writer = SummaryWriter(tb_log_dir)
        logger.info(f"TensorBoard logging to: {tb_log_dir}")

    total_steps = cfg["train"]["epochs"] * max(1, len(train_loader))
    step = 0
    best_iou = -1.0
    best_iou_epoch = 0
    start_time = time.time()

    # Early stopping
    early_stop_patience = cfg["train"].get("early_stop_patience", 30)
    epochs_without_improvement = 0

    logger.info(f"Starting training for {cfg['train']['epochs']} epochs...")

    for epoch in range(cfg["train"]["epochs"]):
        epoch_start_time = time.time()
        logger.epoch_start(epoch, cfg["train"]["epochs"], "TRAIN")

        model.train()
        train_loss_sum = 0.0

        # Progress bar for training
        if HAS_TQDM:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Train]", ncols=100)
        else:
            pbar = train_loader

        for batch_idx, batch in enumerate(pbar):
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)
            lab = batch["label"].to(device)
            with torch.amp.autocast(device_type='cuda', enabled=cfg["train"].get("amp", False)):
                seg, cls = model(img)
                loss, l_seg, l_cls = crit(seg, msk, cls, lab)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if cfg["train"]["scheduler"] == "cosine":
                _cosine_lr_with_warmup(opt, cfg["train"]["lr"], step, total_steps,
                                      warmup_steps=cfg["train"].get("warmup_steps", 500),
                                      min_lr=cfg["train"].get("min_lr", 1e-6))

            train_loss_sum += loss.item()

            # Update progress bar
            if HAS_TQDM:
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{opt.param_groups[0]["lr"]:.2e}'})

            # Log to TensorBoard
            if writer and step % 10 == 0:
                writer.add_scalar('train/loss_total', loss.item(), step)
                writer.add_scalar('train/loss_seg', l_seg.item(), step)
                writer.add_scalar('train/loss_cls', l_cls.item(), step)
                writer.add_scalar('train/lr', opt.param_groups[0]['lr'], step)

            step += 1

        # validation
        model.eval()
        total_inter, total_union = 0.0, 0.0
        acc_m, n = 0.0, 0
        sample_imgs, sample_masks, sample_preds = None, None, None

        # Progress bar for validation
        if HAS_TQDM:
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Val]", ncols=100)
        else:
            val_pbar = val_loader

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                img = batch["image"].to(device)
                msk = batch["mask"].to(device)
                lab = batch["label"].to(device)
                seg, cls = model(img)

                # Accumulate intersection and union for global metrics
                inter, union = compute_intersection_union(seg, msk)
                total_inter += inter
                total_union += union

                acc_m += (cls.argmax(1)==lab).float().mean().item()
                n += 1

                # Compute current metrics for progress bar
                if HAS_TQDM and total_union > 0:
                    curr_iou = total_inter / (total_union - total_inter + 1e-6)
                    curr_dice = (2 * total_inter) / (total_union + 1e-6)
                    val_pbar.set_postfix({'iou': f'{curr_iou:.4f}', 'dice': f'{curr_dice:.4f}'})

                # Save first batch for visualization
                if batch_idx == 0 and writer:
                    sample_imgs = img[:4].cpu()
                    sample_masks = msk[:4].cpu()
                    sample_preds = (seg[:4] > 0.5).float().cpu()

        # Compute final global metrics
        eps = 1e-6
        iou_m = total_inter / (total_union - total_inter + eps)
        dice_m = (2 * total_inter) / (total_union + eps)
        acc_m /= n
        avg_train_loss = train_loss_sum / len(train_loader)
        epoch_time = time.time() - epoch_start_time

        # Log to file logger
        logger.epoch_end(epoch, cfg["train"]["epochs"], {
            'train_loss': avg_train_loss,
            'val_iou': iou_m,
            'val_dice': dice_m,
            'val_acc': acc_m,
            'lr': opt.param_groups[0]['lr'],
            'time_s': epoch_time
        }, "SUMMARY")

        # Log to metrics logger (CSV/JSON)
        metrics_logger.log_epoch(epoch, {
            'train_loss': avg_train_loss,
            'val_iou': iou_m,
            'val_dice': dice_m,
            'val_acc': acc_m,
            'learning_rate': opt.param_groups[0]['lr'],
            'epoch_time_s': epoch_time
        })

        # Console output
        print(f"[Fold {fold}] Epoch {epoch+1}/{cfg['train']['epochs']} | Train Loss {avg_train_loss:.4f} | Val IoU {iou_m:.4f} | Dice {dice_m:.4f} | ClsAcc {acc_m:.4f}")

        # Log validation metrics to TensorBoard
        if writer:
            writer.add_scalar('val/iou', iou_m, epoch)
            writer.add_scalar('val/dice', dice_m, epoch)
            writer.add_scalar('val/cls_acc', acc_m, epoch)
            writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)

            # Log sample predictions every 10 epochs
            if sample_imgs is not None and epoch % 10 == 0:
                import torchvision
                # Create grid: [input | ground truth | prediction]
                grid_img = torchvision.utils.make_grid(sample_imgs, nrow=4, normalize=True)
                grid_mask = torchvision.utils.make_grid(sample_masks, nrow=4)
                grid_pred = torchvision.utils.make_grid(sample_preds, nrow=4)
                writer.add_image('samples/input', grid_img, epoch)
                writer.add_image('samples/ground_truth', grid_mask, epoch)
                writer.add_image('samples/prediction', grid_pred, epoch)

        # Check for improvement
        if iou_m > best_iou:
            best_iou = iou_m
            best_iou_epoch = epoch
            epochs_without_improvement = 0
            ckpt_dir = cfg["logging"]["save_dir"]
            ensure_dir(ckpt_dir)
            save_ckpt(model, os.path.join(ckpt_dir, f"braintumnet_best_fold{fold}.pth"))
            logger.best_checkpoint("IoU", best_iou, epoch)
            print(f"  -> New best IoU: {best_iou:.4f}, checkpoint saved")
        else:
            epochs_without_improvement += 1

        # ReduceLROnPlateau step
        if plateau_scheduler is not None:
            old_lr = opt.param_groups[0]['lr']
            plateau_scheduler.step(iou_m)
            new_lr = opt.param_groups[0]['lr']
            if new_lr != old_lr:
                logger.info(f"ReduceLROnPlateau: Reducing learning rate {old_lr:.2e} -> {new_lr:.2e}")
                print(f"  -> Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

        # Early stopping check
        if epochs_without_improvement >= early_stop_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs ({epochs_without_improvement} epochs without improvement)")
            print(f"\n[Early Stop] No improvement for {early_stop_patience} epochs. Best IoU: {best_iou:.4f} at epoch {best_iou_epoch+1}")
            break

    # Training complete
    total_time = time.time() - start_time

    # Get best metrics from metrics logger
    best_metrics = metrics_logger.get_best_metrics()
    best_metrics['iou'] = (best_iou, best_iou_epoch)

    # Log summary
    logger.training_summary(best_metrics, total_time)
    metrics_logger.print_summary()

    # Close loggers
    metrics_logger.close()

    if writer:
        writer.close()

    return best_iou
