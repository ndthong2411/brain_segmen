import os, math, torch, time, sys
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict
from ..models.braintumnet import BrainTumNet
from ..data.brats2020_dataset import SliceDataset
from ..losses import MultiTaskLoss, dice_loss_with_logits
from ..metrics import iou_score, dice_score
from ..utils.io import ensure_dir, save_ckpt, save_training_state
from ..utils.logger import TrainingLogger
from ..utils.metrics_logger import MetricsLogger
from ..metrics import compute_intersection_union
from ..multiclass_metrics import MulticlassMetricsAccumulator, get_multiclass_predictions, visualize_multiclass_prediction

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
    train_list = os.path.join(proc, f"train_fold{fold}.csv")
    val_list   = os.path.join(proc, f"val_fold{fold}.csv")
    train_ds = SliceDataset(proc, train_list, img_size, cfg["augment"]["rotate_deg"],
                            cfg["augment"]["hflip_p"], cfg["augment"]["vflip_p"], True, cfg["model"]["in_channels"])
    val_ds   = SliceDataset(proc, val_list, img_size, 0,0,0, False, cfg["model"]["in_channels"])

    # Optimized DataLoader for A100 GPU
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["workers"],
        pin_memory=True,  # Faster CPU->GPU transfer
        persistent_workers=True if cfg["train"]["workers"] > 0 else False,  # Avoid worker recreation overhead
        prefetch_factor=2 if cfg["train"]["workers"] > 0 else None  # Preload 2 batches per worker
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"].get("val_batch_size", cfg["train"]["batch_size"]),  # Can be larger in validation
        shuffle=False,
        num_workers=cfg["train"]["workers"],
        pin_memory=True,
        persistent_workers=True if cfg["train"]["workers"] > 0 else False,
        prefetch_factor=2 if cfg["train"]["workers"] > 0 else None
    )
    return train_loader, val_loader

def build_model(cfg: Dict):
    mcfg = cfg["model"]
    return BrainTumNet(in_ch=mcfg["in_channels"], num_cls=mcfg["num_classes_cls"], base=mcfg["base"],
                       dim=mcfg["dim"], patch=mcfg["patch_size"], depth=mcfg["depth"], n_heads=mcfg["n_heads"],
                       roi_stop_grad=mcfg["roi_stop_grad"], deep_supervision=mcfg.get("deep_supervision", False),
                       num_classes_seg=mcfg.get("num_classes_seg", 1))

def train_one_fold(cfg: Dict, fold: int, config_path: str = None, resume_from: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize loggers
    log_dir = cfg["logging"].get("log_dir", "logs")
    exp_name = cfg["logging"].get("exp_name", cfg.get("exp_name", "braintumnet"))
    logger = TrainingLogger(log_dir, exp_name, fold)
    metrics_logger = MetricsLogger(log_dir, exp_name, fold)

    # Save configuration
    if config_path and os.path.exists(config_path):
        logger.save_config(cfg, config_path)

    logger.info(f"Training on device: {device}")

    train_loader, val_loader = build_dataloaders(cfg, fold)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model(cfg).to(device)

    # A100 Optimizations: channels_last memory format
    use_channels_last = cfg["train"].get("channels_last", cfg["train"].get("use_channels_last", False))
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
        logger.info("Using channels_last memory format for A100 optimization")

    # cuDNN benchmark for A100 optimization
    if cfg["train"].get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmark for A100 optimization")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")

    # PyTorch 2.0+ compilation for speedup
    if cfg["train"].get("use_compile", False):
        try:
            logger.info("Compiling model with torch.compile()...")
            model = torch.compile(model, mode='max-autotune')
            logger.info("Model compilation successful")
        except Exception as e:
            logger.warning(f"torch.compile() failed: {e}. Proceeding without compilation.")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    # Setup loss function with class imbalance handling
    loss_type = cfg["train"].get("loss_type", "dice_ce")

    # Check if using new Ultimate loss system (Phase 1+)
    if loss_type in ["ultimate", "ultimate_multitask"]:
        from ..losses_combined import create_loss_from_config
        crit = create_loss_from_config(cfg)
        logger.info(f"Using loss type: {loss_type} (Phase 1+ Ultimate Loss)")
        logger.info(f"  Loss components: Dice + Focal + IoU + Boundary")
        logger.info(f"  IoU weight: {cfg['train'].get('iou_weight', 2.0)}")
        logger.info(f"  Boundary weight: {cfg['train'].get('boundary_weight', 0.5)}")
    else:
        # Original loss system (baseline)
        crit = MultiTaskLoss(
            seg_w=cfg["train"]["seg_loss_weight"],
            cls_w=cfg["train"]["cls_loss_weight"],
            boundary_w=cfg["train"].get("boundary_loss_weight", 0.0),
            loss_type=loss_type,
            pos_weight=cfg["train"].get("pos_weight", None),
            focal_alpha=cfg["train"].get("focal_alpha", 0.25),
            focal_gamma=cfg["train"].get("focal_gamma", 2.0),
            # Multiclass params
            num_classes=cfg["model"].get("num_classes_seg", 1),
            ignore_background=cfg["train"].get("ignore_background", True),
            class_weights=cfg["train"].get("class_weights", None)
        )
        logger.info(f"Using loss type: {loss_type}")
    if loss_type == "dice_ce_weighted":
        logger.info(f"  Positive class weight: {cfg['train'].get('pos_weight', 'None')}")
    elif loss_type == "dice_focal":
        logger.info(f"  Focal alpha: {cfg['train'].get('focal_alpha', 0.25)}, gamma: {cfg['train'].get('focal_gamma', 2.0)}")

    # Mixed precision setup with dtype support (bfloat16 for A100)
    use_amp = cfg["train"].get("amp", False)
    amp_dtype_str = cfg["train"].get("amp_dtype", "float16")
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16

    # GradScaler not needed for bfloat16 (A100)
    scaler = torch.amp.GradScaler(device='cuda', enabled=(use_amp and amp_dtype == torch.float16))

    if use_amp:
        logger.info(f"Mixed precision enabled: {amp_dtype_str}")

    # Learning rate scheduler setup
    plateau_scheduler = None
    onecycle_scheduler = None

    if cfg["train"]["scheduler"] == "plateau":
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=0.5, patience=10, min_lr=1e-7
        )
    elif cfg["train"]["scheduler"] == "onecycle":
        # OneCycleLR for faster convergence (A100 optimized)
        onecycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=cfg["train"]["lr"],
            epochs=cfg["train"]["epochs"],
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # 30% warmup
            anneal_strategy='cos',
            div_factor=25.0,  # initial_lr = max_lr/25
            final_div_factor=1e4  # min_lr = initial_lr/1e4
        )

    # TensorBoard
    writer = None
    if HAS_TENSORBOARD and cfg["logging"].get("use_tensorboard", True):
        out_dir = cfg["logging"].get("out_dir", "runs")
        tb_log_dir = os.path.join(out_dir, f"{exp_name}_fold{fold}")
        ensure_dir(tb_log_dir)
        writer = SummaryWriter(tb_log_dir)
        logger.info(f"TensorBoard logging to: {tb_log_dir}")

    total_steps = cfg["train"]["epochs"] * max(1, len(train_loader))
    step = 0
    best_iou = -1.0
    best_iou_epoch = 0
    start_epoch = 0
    start_time = time.time()

    # Resume from checkpoint if specified
    if resume_from is not None:
        logger.info(f"Resuming training from checkpoint: {resume_from}")
        from ..utils.io import load_training_state
        # Pass the active scheduler (plateau or onecycle)
        active_scheduler = plateau_scheduler if plateau_scheduler is not None else onecycle_scheduler
        resume_info = load_training_state(resume_from, model, opt, active_scheduler, scaler, device, expected_fold=fold)
        start_epoch = resume_info['epoch'] + 1  # Start from next epoch
        best_iou = resume_info['best_iou']
        best_iou_epoch = resume_info['best_iou_epoch']
        step = start_epoch * len(train_loader)
        logger.info(f"  Starting from epoch {start_epoch}")
        logger.info(f"  Previous best IoU: {best_iou:.4f} at epoch {best_iou_epoch + 1}")

    # Early stopping
    early_stop_patience = cfg["train"].get("early_stop_patience", 30)
    epochs_without_improvement = 0

    logger.info(f"Starting training for {cfg['train']['epochs']} epochs...")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        epoch_start_time = time.time()
        logger.epoch_start(epoch, cfg["train"]["epochs"], "TRAIN")

        model.train()
        train_loss_sum = 0.0

        # Progress bar for training
        if HAS_TQDM:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Train]",
                       ncols=120, leave=True, file=sys.stdout)
        else:
            pbar = train_loader

        for batch_idx, batch in enumerate(pbar):
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)
            lab = batch["label"].to(device)

            # Convert to channels_last if enabled
            if use_channels_last:
                img = img.to(memory_format=torch.channels_last)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp, dtype=amp_dtype):
                model_output = model(img)

                # Handle deep supervision output
                if cfg["model"].get("deep_supervision", False):
                    seg, cls, aux_outputs = model_output  # seg: main output, aux_outputs: [aux3, aux2, aux1]
                else:
                    seg, cls = model_output
                    aux_outputs = None

                # Main loss (segmentation + classification)
                # Handle both old and new loss formats
                if loss_type in ["ultimate", "ultimate_multitask"]:
                    # New loss format: returns (total_loss, loss_dict)
                    loss, loss_dict = crit(seg, msk, cls, lab, aux_outputs)
                    l_seg = loss_dict.get('dice', 0.0) + loss_dict.get('focal', 0.0) + loss_dict.get('iou', 0.0) + loss_dict.get('boundary', 0.0)
                    l_cls = loss_dict.get('cls', 0.0)
                else:
                    # Old loss format: returns (total_loss, seg_loss, cls_loss)
                    loss, l_seg, l_cls = crit(seg, msk, cls, lab)

                    # Deep supervision auxiliary losses (only for old loss system)
                    if aux_outputs is not None:
                        aux_weights = cfg["train"].get("aux_loss_weights", [0.5, 0.25, 0.125])
                        for i, aux_output in enumerate(aux_outputs):
                            # Resize auxiliary output to match mask size
                            aux_resized = F.interpolate(aux_output, size=msk.shape[-2:],
                                                        mode='bilinear', align_corners=False)
                            # Compute auxiliary loss using the same seg_loss as main task
                            aux_loss_val = crit.seg_loss(aux_resized, msk)
                            # Add weighted auxiliary loss
                            weight = aux_weights[i] if i < len(aux_weights) else 0.125
                            loss = loss + weight * aux_loss_val

                # Gradient accumulation support
                grad_accum_steps = cfg["train"].get("grad_accum_steps", 1)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            # Apply gradient clipping before optimizer step
            if (batch_idx + 1) % grad_accum_steps == 0:
                if cfg["train"].get("grad_clip_norm", 0) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                # OneCycleLR step (per batch)
                if onecycle_scheduler is not None:
                    onecycle_scheduler.step()

            if cfg["train"]["scheduler"] == "cosine":
                _cosine_lr_with_warmup(opt, cfg["train"]["lr"], step, total_steps,
                                      warmup_steps=cfg["train"].get("warmup_steps", 500),
                                      min_lr=cfg["train"].get("min_lr", 1e-6))

            train_loss_sum += loss.item()

            # Update progress bar
            if HAS_TQDM:
                actual_loss = loss.item() * cfg["train"].get("grad_accum_steps", 1)  # Show unscaled loss
                pbar.set_postfix({'loss': f'{actual_loss:.4f}', 'lr': f'{opt.param_groups[0]["lr"]:.2e}'})
            else:
                # Fallback: print progress every 10 batches if no tqdm
                if (batch_idx + 1) % 10 == 0:
                    actual_loss = loss.item() * cfg["train"].get("grad_accum_steps", 1)
                    print(f"  [{batch_idx+1}/{len(train_loader)}] loss: {actual_loss:.4f}, lr: {opt.param_groups[0]['lr']:.2e}", flush=True)

            # Log to TensorBoard (reduced frequency for A100 optimization)
            log_interval = cfg["train"].get("log_interval", 10)
            if writer and step % log_interval == 0:
                actual_loss = loss.item() * cfg["train"].get("grad_accum_steps", 1)
                writer.add_scalar('train/loss_total', actual_loss, step)
                # Handle both tensor and float types for l_seg and l_cls
                l_seg_val = l_seg.item() if isinstance(l_seg, torch.Tensor) else l_seg
                l_cls_val = l_cls.item() if isinstance(l_cls, torch.Tensor) else l_cls
                writer.add_scalar('train/loss_seg', l_seg_val, step)
                writer.add_scalar('train/loss_cls', l_cls_val, step)
                writer.add_scalar('train/lr', opt.param_groups[0]['lr'], step)

            step += 1

        # Validation (skip if val_interval > 1 and not this epoch)
        val_interval = cfg["train"].get("val_interval", 1)
        should_validate = (epoch % val_interval == 0) or (epoch == cfg["train"]["epochs"] - 1)

        if should_validate:
            model.eval()

            # Initialize multiclass metrics accumulator if num_classes > 1, else use binary
            num_classes_seg = cfg["model"].get("num_classes_seg", 1)
            if num_classes_seg > 1:
                metrics_acc = MulticlassMetricsAccumulator(num_classes=num_classes_seg)
            else:
                total_inter, total_union = 0.0, 0.0

            acc_m, n = 0.0, 0
            sample_imgs, sample_masks, sample_preds = None, None, None

            # Progress bar for validation
            if HAS_TQDM:
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Val]",
                               ncols=120, leave=True, file=sys.stdout)
            else:
                val_pbar = val_loader

            # Use inference_mode for better performance than no_grad
            with torch.inference_mode():
                for batch_idx, batch in enumerate(val_pbar):
                    img = batch["image"].to(device)
                    msk = batch["mask"].to(device)
                    lab = batch["label"].to(device)

                    # Convert to channels_last if enabled
                    if use_channels_last:
                        img = img.to(memory_format=torch.channels_last)

                    # Handle deep supervision outputs
                    model_output = model(img)
                    if cfg["model"].get("deep_supervision", False):
                        seg, cls, _ = model_output  # Ignore aux outputs in validation
                    else:
                        seg, cls = model_output

                    # Accumulate metrics (multiclass or binary)
                    if num_classes_seg > 1:
                        metrics_acc.update(seg, msk)
                    else:
                        inter, union = compute_intersection_union(seg, msk)
                        total_inter += inter
                        total_union += union

                    acc_m += (cls.argmax(1)==lab).float().mean().item()
                    n += 1

                    # Compute current metrics for progress bar
                    if HAS_TQDM:
                        if num_classes_seg > 1:
                            # Show running multiclass metrics
                            curr_metrics = metrics_acc.get_metrics()
                            val_pbar.set_postfix({
                                'WT': f'{curr_metrics["WT_dice"]:.4f}',
                                'TC': f'{curr_metrics["TC_dice"]:.4f}',
                                'ED': f'{curr_metrics["ED_dice"]:.4f}'
                            })
                        elif total_union > 0:
                            curr_iou = total_inter / (total_union - total_inter + 1e-6)
                            curr_dice = (2 * total_inter) / (total_union + 1e-6)
                            val_pbar.set_postfix({'iou': f'{curr_iou:.4f}', 'dice': f'{curr_dice:.4f}'})

                    # Save first batch for visualization
                    if batch_idx == 0 and writer:
                        sample_imgs = img[:4].cpu()
                        sample_masks = msk[:4].cpu()
                        if num_classes_seg > 1:
                            sample_preds = get_multiclass_predictions(seg[:4]).cpu()
                        else:
                            sample_preds = (seg[:4] > 0.5).float().cpu()

            # Compute final global metrics
            eps = 1e-6
            if num_classes_seg > 1:
                # Get all multiclass metrics
                final_metrics = metrics_acc.get_metrics()
                # Use mean_dice as main metric for checkpointing
                iou_m = final_metrics['mean_iou']
                dice_m = final_metrics['mean_dice']
                # Extract region-specific metrics
                wt_dice = final_metrics['WT_dice']
                wt_iou = final_metrics['WT_iou']
                tc_dice = final_metrics['TC_dice']
                tc_iou = final_metrics['TC_iou']
                ed_dice = final_metrics['ED_dice']
                ed_iou = final_metrics['ED_iou']
            else:
                # Binary metrics
                iou_m = total_inter / (total_union - total_inter + eps)
                dice_m = (2 * total_inter) / (total_union + eps)
                wt_dice = wt_iou = tc_dice = tc_iou = ed_dice = ed_iou = 0.0
            acc_m /= n
        else:
            # Skip validation this epoch
            iou_m = best_iou  # Use previous best
            dice_m = 0.0
            acc_m = 0.0
            wt_dice = wt_iou = tc_dice = tc_iou = ed_dice = ed_iou = 0.0
        avg_train_loss = train_loss_sum / len(train_loader)
        epoch_time = time.time() - epoch_start_time

        # Log to file logger
        log_dict = {
            'train_loss': avg_train_loss,
            'val_iou': iou_m,
            'val_dice': dice_m,
            'val_acc': acc_m,
            'lr': opt.param_groups[0]['lr'],
            'time_s': epoch_time
        }
        if num_classes_seg > 1:
            log_dict.update({
                'WT_dice': wt_dice, 'WT_iou': wt_iou,
                'TC_dice': tc_dice, 'TC_iou': tc_iou,
                'ED_dice': ed_dice, 'ED_iou': ed_iou
            })
        logger.epoch_end(epoch, cfg["train"]["epochs"], log_dict, "SUMMARY")

        # Log to metrics logger (CSV/JSON)
        metrics_dict = {
            'train_loss': avg_train_loss,
            'val_iou': iou_m,
            'val_dice': dice_m,
            'val_acc': acc_m,
            'learning_rate': opt.param_groups[0]['lr'],
            'epoch_time_s': epoch_time
        }
        if num_classes_seg > 1:
            metrics_dict.update({
                'WT_dice': wt_dice, 'WT_iou': wt_iou,
                'TC_dice': tc_dice, 'TC_iou': tc_iou,
                'ED_dice': ed_dice, 'ED_iou': ed_iou
            })
        metrics_logger.log_epoch(epoch, metrics_dict)

        # Console output
        if num_classes_seg > 1:
            print(f"[Fold {fold}] Epoch {epoch+1}/{cfg['train']['epochs']} | Train Loss {avg_train_loss:.4f} | WT {wt_dice:.4f} | TC {tc_dice:.4f} | ED {ed_dice:.4f} | Mean {dice_m:.4f} | ClsAcc {acc_m:.4f}")
        else:
            print(f"[Fold {fold}] Epoch {epoch+1}/{cfg['train']['epochs']} | Train Loss {avg_train_loss:.4f} | Val IoU {iou_m:.4f} | Dice {dice_m:.4f} | ClsAcc {acc_m:.4f}")

        # Log validation metrics to TensorBoard
        if writer:
            writer.add_scalar('val/iou', iou_m, epoch)
            writer.add_scalar('val/dice', dice_m, epoch)
            writer.add_scalar('val/cls_acc', acc_m, epoch)
            writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)

            # Log multiclass metrics if applicable
            if num_classes_seg > 1:
                writer.add_scalar('val/WT_dice', wt_dice, epoch)
                writer.add_scalar('val/WT_iou', wt_iou, epoch)
                writer.add_scalar('val/TC_dice', tc_dice, epoch)
                writer.add_scalar('val/TC_iou', tc_iou, epoch)
                writer.add_scalar('val/ED_dice', ed_dice, epoch)
                writer.add_scalar('val/ED_iou', ed_iou, epoch)

            # Log sample predictions every 10 epochs
            if sample_imgs is not None and epoch % 10 == 0:
                import torchvision
                # Create grid: [input | ground truth | prediction]
                grid_img = torchvision.utils.make_grid(sample_imgs, nrow=4, normalize=True)

                if num_classes_seg > 1:
                    # Convert multiclass masks to RGB for visualization
                    grid_mask = visualize_multiclass_prediction(sample_masks.squeeze(1).long())
                    grid_pred = visualize_multiclass_prediction(sample_preds.long())
                    grid_mask = torchvision.utils.make_grid(grid_mask, nrow=4)
                    grid_pred = torchvision.utils.make_grid(grid_pred, nrow=4)
                else:
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

        # Save "last" checkpoint every epoch for resume capability
        ckpt_dir = cfg["logging"]["save_dir"]
        ensure_dir(ckpt_dir)
        last_ckpt_path = os.path.join(ckpt_dir, f"last_fold{fold}.pth")
        # Save the active scheduler (plateau or onecycle)
        active_scheduler = plateau_scheduler if plateau_scheduler is not None else onecycle_scheduler
        save_training_state(last_ckpt_path, epoch, model, opt, active_scheduler, scaler,
                           best_iou, best_iou_epoch, cfg, fold=fold)

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
