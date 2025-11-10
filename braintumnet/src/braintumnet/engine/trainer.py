import os, math, torch, time, sys
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict
from pathlib import Path
from ..data.dataset_factory import create_dataset, get_data_root
from ..losses import MultiTaskLoss, dice_loss_with_logits
from ..metrics import (
    compute_hausdorff_distance_95,
    compute_intersection_union,
    dice_score,
    iou_score,
)
from ..utils.io import ensure_dir, save_ckpt, save_training_state
from ..utils.logger import TrainingLogger
from ..utils.metrics_logger import MetricsLogger
from ..metrics.multiclass import (
    MulticlassMetricsAccumulator,
    get_multiclass_predictions,
    visualize_multiclass_prediction,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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


def centralized_gradient(optimizer):
    """
    Apply Gradient Centralization (Phase 1 Optimization)

    Gradient Centralization (GC) centralizes the gradient vectors to have zero mean
    before applying to weight updates. This simple technique improves:
    - Optimization stability
    - Convergence speed
    - Generalization performance

    Expected improvement: +0.3-0.7% Dice

    Reference: Yong et al. "Gradient Centralization: A New Optimization Technique
               for Deep Neural Networks" (ECCV 2020)

    Args:
        optimizer: PyTorch optimizer with gradients already computed
    """
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            # Only apply to convolutional and linear layers (ndim > 1)
            if len(p.grad.shape) > 1:
                # Subtract mean across all dimensions except output channel
                # For Conv: (out_ch, in_ch, h, w) -> mean over (in_ch, h, w)
                # For Linear: (out_features, in_features) -> mean over (in_features,)
                dims_to_reduce = tuple(range(1, len(p.grad.shape)))
                grad_mean = p.grad.mean(dim=dims_to_reduce, keepdim=True)
                p.grad.sub_(grad_mean)


class DeepSupervisionScheduler:
    """
    Deep Supervision Weight Scheduler (Phase 1 Optimization)

    Gradually reduces auxiliary loss weight during training.
    Early epochs need strong deep supervision for better gradient flow.
    Late epochs should focus on main output for fine-tuning.

    Expected improvement: +0.5-1% Dice through better training dynamics

    Args:
        initial_weight: Starting auxiliary weight (default: 0.5)
        final_weight: Ending auxiliary weight (default: 0.1)
        total_epochs: Total number of training epochs (default: 400)
        schedule_type: "linear" or "cosine" decay (default: "linear")
    """
    def __init__(self, initial_weight=0.5, final_weight=0.1, total_epochs=400, schedule_type="linear"):
        self.initial = initial_weight
        self.final = final_weight
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type

    def get_weight(self, epoch):
        """
        Get auxiliary loss weight for current epoch

        Args:
            epoch: Current epoch number (0-indexed)

        Returns:
            weight: Auxiliary loss weight for this epoch
        """
        progress = min(epoch / self.total_epochs, 1.0)

        if self.schedule_type == "cosine":
            # Cosine decay (smoother transition)
            weight = self.final + (self.initial - self.final) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            # Linear decay (default)
            weight = self.initial + (self.final - self.initial) * progress

        return max(weight, self.final)  # Ensure we don't go below final weight

def build_dataloaders(cfg: Dict, fold: int):
    # Get backend type and data root
    backend = cfg["data"].get("backend", "png")  # Default to PNG for backward compatibility
    data_root = get_data_root(cfg)

    print(f"\n[DataLoader] Backend: {backend}")
    print(f"[DataLoader] Data root: {data_root}")

    # Build split file paths
    train_list = os.path.join(data_root, f"train_fold{fold}.csv")
    val_list   = os.path.join(data_root, f"val_fold{fold}.csv")

    # Create datasets using factory
    train_ds = create_dataset(backend, data_root, train_list, cfg, train=True)
    val_ds   = create_dataset(backend, data_root, val_list, cfg, train=False)

    # Optimized DataLoader for A100 GPU
    prefetch_factor = cfg["train"].get("prefetch_factor", 2)
    pin_memory = cfg["train"].get("pin_memory", True)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["workers"],
        pin_memory=pin_memory,  # Faster CPU->GPU transfer
        persistent_workers=True if cfg["train"]["workers"] > 0 else False,  # Avoid worker recreation overhead
        prefetch_factor=prefetch_factor if cfg["train"]["workers"] > 0 else None  # Configurable prefetch
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"].get("val_batch_size", cfg["train"]["batch_size"]),  # Can be larger in validation
        shuffle=False,
        num_workers=cfg["train"]["workers"],
        pin_memory=pin_memory,
        persistent_workers=True if cfg["train"]["workers"] > 0 else False,
        prefetch_factor=prefetch_factor if cfg["train"]["workers"] > 0 else None
    )
    return train_loader, val_loader

def build_model(cfg: Dict):
    """Build model using factory pattern"""
    from ..models import build_model as model_factory
    return model_factory(cfg)


def _sanitize_artifact_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(name))
    safe = safe.strip("_").lower()
    return safe or "model"


def _resolve_project_path(path_str: str) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def prepare_artifact_dirs(cfg: Dict) -> Dict[str, str]:
    logging_cfg = cfg.setdefault("logging", {})
    exp_name = logging_cfg.get("exp_name", cfg.get("exp_name", "braintumnet"))
    model_cfg = cfg.get("model", {})
    raw_model_name = model_cfg.get("model_name") or model_cfg.get("name") or model_cfg.get("model_type", "model")
    model_identifier = _sanitize_artifact_name(raw_model_name)

    if logging_cfg.get("_artifact_dirs_prepared"):
        log_dir = logging_cfg["log_dir"]
        out_dir = logging_cfg["out_dir"]
        save_dir = logging_cfg["save_dir"]
        ensure_dir(log_dir)
        ensure_dir(out_dir)
        ensure_dir(save_dir)
        return {
            "log_dir": log_dir,
            "out_dir": out_dir,
            "save_dir": save_dir,
            "exp_name": exp_name,
            "model_identifier": model_identifier,
            "raw_model_name": raw_model_name,
        }

    base_log_dir = logging_cfg.get("_base_log_dir")
    base_out_dir = logging_cfg.get("_base_out_dir")
    base_save_dir = logging_cfg.get("_base_save_dir")

    if base_log_dir is None:
        base_log_dir = logging_cfg.get("log_dir", "logs")
    base_log_dir = _resolve_project_path(base_log_dir)
    logging_cfg["_base_log_dir"] = base_log_dir

    if base_out_dir is None:
        base_out_dir = logging_cfg.get("out_dir", "runs")
    base_out_dir = _resolve_project_path(base_out_dir)
    logging_cfg["_base_out_dir"] = base_out_dir

    if base_save_dir is None:
        base_save_dir = logging_cfg.get("save_dir", "checkpoints")
    base_save_dir = _resolve_project_path(base_save_dir)
    logging_cfg["_base_save_dir"] = base_save_dir

    log_dir = os.path.join(base_log_dir, model_identifier, exp_name)
    out_dir = os.path.join(base_out_dir, model_identifier, exp_name)
    save_dir = os.path.join(base_save_dir, model_identifier, exp_name)

    ensure_dir(log_dir)
    ensure_dir(out_dir)
    ensure_dir(save_dir)

    logging_cfg["log_dir"] = log_dir
    logging_cfg["out_dir"] = out_dir
    logging_cfg["save_dir"] = save_dir
    logging_cfg["_artifact_dirs_prepared"] = True

    return {
        "log_dir": log_dir,
        "out_dir": out_dir,
        "save_dir": save_dir,
        "exp_name": exp_name,
        "model_identifier": model_identifier,
        "raw_model_name": raw_model_name,
    }

def train_one_fold(cfg: Dict, fold: int, config_path: str = None, resume_from: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    artifact_dirs = prepare_artifact_dirs(cfg)
    log_dir = artifact_dirs["log_dir"]
    out_dir = artifact_dirs["out_dir"]
    save_dir = artifact_dirs["save_dir"]
    exp_name = artifact_dirs["exp_name"]
    raw_model_name = artifact_dirs["raw_model_name"]

    # Initialize loggers
    logger = TrainingLogger(log_dir, exp_name, fold)
    metrics_logger = MetricsLogger(log_dir, exp_name, fold)

    # Save configuration
    if config_path and os.path.exists(config_path):
        logger.save_config(cfg, config_path)

    logger.info(f"Training on device: {device}")
    logger.info(f"Artifact directories ({raw_model_name}) | logs: {log_dir} | tensorboard: {out_dir} | checkpoints: {save_dir}")

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
            compile_mode = cfg["train"].get("compile_mode", "max-autotune")
            logger.info(f"Compiling model with torch.compile(mode='{compile_mode}')...")
            model = torch.compile(model, mode=compile_mode)
            logger.info("Model compilation successful")
        except Exception as e:
            logger.warning(f"torch.compile() failed: {e}. Proceeding without compilation.")

    # Optimizer setup with A100 optimizations
    optimizer_type = cfg["train"].get("optimizer", "adam").lower()
    optimizer_fused = cfg["train"].get("optimizer_fused", False)
    
    if optimizer_type == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
            fused=optimizer_fused  # A100 optimization
        )
        logger.info(f"Using AdamW optimizer (fused={optimizer_fused})")
    else:
        opt = torch.optim.Adam(
            model.parameters(),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
            fused=optimizer_fused  # A100 optimization
        )
        logger.info(f"Using Adam optimizer (fused={optimizer_fused})")

    # Setup loss function with class imbalance handling
    loss_type = cfg["train"].get("loss_type", "dice_ce")
    compute_hd95 = cfg["train"].get("compute_hd95", True)

    # Check if using new Ultimate loss system (Phase 1+)
    if loss_type in ["ultimate", "ultimate_multitask"]:
        from ..losses.combined import create_loss_from_config
        crit = create_loss_from_config(cfg)
        logger.info(f"Using loss type: {loss_type} (Phase 1+ Ultimate Loss)")
        logger.info(f"  Loss components: Dice + Focal + IoU + Boundary")
        logger.info(f"  IoU weight: {cfg['train'].get('iou_weight', 2.0)}")
        logger.info(f"  Boundary weight: {cfg['train'].get('boundary_weight', 0.5)}")
        effective_loss = loss_type
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
        effective_loss = getattr(crit, "loss_name", loss_type)
        logger.info(f"Using loss type: {effective_loss}")
        if effective_loss != loss_type:
            logger.info(
                f"  Requested loss_type '{loss_type}' auto-adjusted for num_classes="
                f"{cfg['model'].get('num_classes_seg', 1)}"
            )
        if "dice_ce_weighted" in effective_loss:
            logger.info(f"  Positive class weight: {cfg['train'].get('pos_weight', 'None')}")
        if "dice_focal" in effective_loss:
            logger.info(
                f"  Focal alpha: {cfg['train'].get('focal_alpha', 0.25)}, "
                f"gamma: {cfg['train'].get('focal_gamma', 2.0)}"
            )

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
    cosine_restarts_scheduler = None

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
    elif cfg["train"]["scheduler"] == "cosine_restarts":
        # SGDR: Cosine Annealing with Warm Restarts (Phase 1 fix for plateau)
        T_0 = cfg["train"].get("T_0", 50)  # Initial restart period
        T_mult = cfg["train"].get("T_mult", 2)  # Period multiplier after each restart
        eta_min = cfg["train"].get("min_lr", 1e-5)  # Minimum learning rate
        cosine_restarts_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
        logger.info(f"Using SGDR scheduler: T_0={T_0}, T_mult={T_mult}, eta_min={eta_min}")

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
        # Pass the active scheduler (plateau, onecycle, or cosine_restarts)
        active_scheduler = plateau_scheduler if plateau_scheduler is not None else (
            onecycle_scheduler if onecycle_scheduler is not None else cosine_restarts_scheduler
        )
        resume_info = load_training_state(resume_from, model, opt, active_scheduler, scaler, device, expected_fold=fold)
        start_epoch = resume_info['epoch'] + 1  # Start from next epoch
        best_iou = resume_info['best_iou']
        best_iou_epoch = resume_info['best_iou_epoch']
        step = start_epoch * len(train_loader)
        logger.info(f"  Starting from epoch {start_epoch}")
        logger.info(f"  Previous best IoU: {best_iou:.4f} at epoch {best_iou_epoch + 1}")

    # Deep Supervision Scheduler (Phase 1 optimization)
    ds_scheduler = None
    if cfg["train"].get("aux_weight_initial") and cfg["train"].get("aux_weight_final"):
        ds_scheduler = DeepSupervisionScheduler(
            initial_weight=cfg["train"]["aux_weight_initial"],
            final_weight=cfg["train"]["aux_weight_final"],
            total_epochs=cfg["train"]["epochs"],
            schedule_type="linear"
        )
        logger.info(f"Using Deep Supervision Scheduler: {cfg['train']['aux_weight_initial']} → {cfg['train']['aux_weight_final']}")

    # Early stopping
    early_stop_patience = cfg["train"].get("early_stop_patience", 30)
    epochs_without_improvement = 0

    logger.info(f"Starting training for {cfg['train']['epochs']} epochs...")
    last_val_metrics = None  # Cache last computed validation metrics when val is skipped

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
                # Check if model actually returns 3 values (seg, cls, aux_outputs)
                if isinstance(model_output, tuple):
                    if len(model_output) == 3:
                        seg, cls, aux_outputs = model_output
                    elif len(model_output) == 2:
                        seg, cls = model_output
                        aux_outputs = None
                    else:
                        raise ValueError(f"Unexpected model output format: {len(model_output)} values")
                else:
                    # Single output (only segmentation)
                    seg = model_output
                    cls = None
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
                        # Get dynamic aux weight from scheduler if available
                        if ds_scheduler is not None:
                            base_aux_weight = ds_scheduler.get_weight(epoch)
                            aux_weights = [base_aux_weight * 1.0, base_aux_weight * 0.5, base_aux_weight * 0.25]
                        else:
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

            # Apply gradient clipping and centralization before optimizer step
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Unscale gradients first (needed for both clipping and centralization)
                scaler.unscale_(opt)

                # Phase 1: Gradient Centralization
                if cfg["train"].get("gradient_centralization", False):
                    centralized_gradient(opt)

                # Gradient clipping (after centralization)
                if cfg["train"].get("grad_clip_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                # OneCycleLR step (per batch)
                if onecycle_scheduler is not None:
                    onecycle_scheduler.step()

                # CosineAnnealingWarmRestarts step (per batch)
                if cosine_restarts_scheduler is not None:
                    cosine_restarts_scheduler.step()

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

                # Log individual loss components if available (ultimate loss)
                if loss_type in ["ultimate", "ultimate_multitask"] and 'loss_dict' in locals():
                    writer.add_scalar('train/loss_dice', loss_dict.get('dice', 0.0), step)
                    writer.add_scalar('train/loss_focal', loss_dict.get('focal', 0.0), step)
                    writer.add_scalar('train/loss_iou', loss_dict.get('iou', 0.0), step)
                    writer.add_scalar('train/loss_boundary', loss_dict.get('boundary', 0.0), step)

            step += 1

        # Validation (skip if val_interval > 1 and not this epoch)
        val_interval = cfg["train"].get("val_interval", 1)
        should_validate = (epoch % val_interval == 0) or (epoch == cfg["train"]["epochs"] - 1)

        if should_validate:
            model.eval()

            # Initialize multiclass metrics accumulator if num_classes > 1, else use binary
            num_classes_seg = cfg["model"].get("num_classes_seg", 1)
            if num_classes_seg > 1:
                metrics_acc = MulticlassMetricsAccumulator(
                    num_classes=num_classes_seg,
                    compute_hd95=compute_hd95
                )
            else:
                total_inter, total_union = 0.0, 0.0
                hd95_sum, hd95_count = 0.0, 0

            cls_acc_sum, cls_batches = 0.0, 0
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

                    if isinstance(model_output, tuple):
                        if len(model_output) >= 3:
                            seg, cls, _ = model_output[:3]
                        elif len(model_output) == 2:
                            seg, cls = model_output
                        elif len(model_output) == 1:
                            seg = model_output[0]
                            cls = None
                        else:
                            raise ValueError(f"Unexpected model output format: {len(model_output)} values")
                    else:
                        seg = model_output
                        cls = None

                    # Accumulate metrics (multiclass or binary)
                    if num_classes_seg > 1:
                        metrics_acc.update(seg, msk)
                    else:
                        inter, union = compute_intersection_union(seg, msk)
                        total_inter += inter
                        total_union += union

                        # Compute HD95 for each sample in batch
                        pred_np = (torch.sigmoid(seg) > 0.5).cpu().numpy()
                        target_np = msk.cpu().numpy()

                        # Debug logging for first batch
                        if batch_idx == 0 and epoch == start_epoch:
                            logger.info(f"HD95 Debug - pred_np shape: {pred_np.shape}, target_np shape: {target_np.shape}")
                            logger.info(f"HD95 Debug - pred_np unique values: {np.unique(pred_np)}")
                            logger.info(f"HD95 Debug - target_np unique values: {np.unique(target_np)}")

                        for i in range(pred_np.shape[0]):
                            try:
                                # Extract 2D slice (remove channel dimension if present)
                                pred_slice = pred_np[i, 0] if pred_np.ndim == 4 else pred_np[i]
                                target_slice = target_np[i, 0] if target_np.ndim == 4 else target_np[i]

                                # Check if masks are not empty
                                pred_count = np.sum(pred_slice > 0)
                                target_count = np.sum(target_slice > 0)

                                if pred_count == 0 or target_count == 0:
                                    # Skip empty masks
                                    continue

                                hd95_val = compute_hausdorff_distance_95(pred_slice, target_slice)
                                if not np.isinf(hd95_val) and not np.isnan(hd95_val):
                                    hd95_sum += hd95_val
                                    hd95_count += 1

                                    # Debug logging for first successful computation
                                    if hd95_count == 1 and batch_idx == 0:
                                        logger.info(f"HD95 Debug - First valid HD95: {hd95_val:.4f} (pred_pixels: {pred_count}, target_pixels: {target_count})")
                            except Exception as e:
                                # Log first error for debugging
                                if batch_idx == 0 and epoch == start_epoch:
                                    logger.warning(f"HD95 computation failed: {e}, pred_shape: {pred_slice.shape}, target_shape: {target_slice.shape}")
                                continue

                    if cls is not None:
                        cls_acc_sum += (cls.argmax(1) == lab).float().mean().item()
                        cls_batches += 1

                    # Compute current metrics for progress bar
                    if HAS_TQDM:
                        if num_classes_seg > 1:
                            # Show running multiclass metrics, adapt labels for 4-class BraTS
                            curr_metrics = metrics_acc.get_metrics()
                            postfix = {
                                'WT': f'{curr_metrics["WT_dice"]:.4f}',
                                'TC': f'{curr_metrics["TC_dice"]:.4f}',
                            }
                            if num_classes_seg == 4:
                                postfix['ET'] = f'{curr_metrics["ET_dice"]:.4f}'
                            else:
                                postfix['ED'] = f'{curr_metrics["ED_dice"]:.4f}'
                            val_pbar.set_postfix(postfix)
                        elif total_union > 0:
                            curr_dice = (2 * total_inter) / (total_union + 1e-6)
                            curr_hd95 = hd95_sum / hd95_count if hd95_count > 0 else 0.0
                            val_pbar.set_postfix({'dice': f'{curr_dice:.4f}', 'hd95': f'{curr_hd95:.2f}'})

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
                # Get all multiclass metrics including HD95
                final_metrics = metrics_acc.get_metrics()
                # Use mean_dice as main metric for checkpointing
                iou_m = final_metrics['mean_iou']
                dice_m = final_metrics['mean_dice']
                hd95_m = final_metrics['mean_hd95']
                # Extract region-specific metrics
                et_dice = final_metrics['ET_dice']  # NEW for 4-class
                et_iou = final_metrics['ET_iou']
                et_hd95 = final_metrics['ET_hd95']
                tc_dice = final_metrics['TC_dice']
                tc_iou = final_metrics['TC_iou']
                tc_hd95 = final_metrics['TC_hd95']
                wt_dice = final_metrics['WT_dice']
                wt_iou = final_metrics['WT_iou']
                wt_hd95 = final_metrics['WT_hd95']
                ed_dice = final_metrics['ED_dice']
                ed_iou = final_metrics['ED_iou']
                ed_hd95 = final_metrics['ED_hd95']
            else:
                # Binary metrics
                iou_m = total_inter / (total_union - total_inter + eps)
                dice_m = (2 * total_inter) / (total_union + eps)

                # Compute HD95 (use -1.0 as sentinel for "not computed")
                if hd95_count > 0:
                    hd95_m = hd95_sum / hd95_count
                    logger.info(f"HD95: Computed from {hd95_count} valid samples, mean = {hd95_m:.4f}")
                else:
                    hd95_m = -1.0  # Sentinel value
                    logger.warning(f"HD95: No valid samples computed (all masks may be empty or model not predicting)")

                et_dice = et_iou = et_hd95 = wt_dice = wt_iou = wt_hd95 = tc_dice = tc_iou = tc_hd95 = ed_dice = ed_iou = ed_hd95 = 0.0
            acc_m = cls_acc_sum / cls_batches if cls_batches > 0 else 0.0
            last_val_metrics = {
                'dice': dice_m,
                'iou': iou_m,
                'hd95': hd95_m,
                'acc': acc_m,
                'et_dice': et_dice, 'et_iou': et_iou, 'et_hd95': et_hd95,
                'tc_dice': tc_dice, 'tc_iou': tc_iou, 'tc_hd95': tc_hd95,
                'wt_dice': wt_dice, 'wt_iou': wt_iou, 'wt_hd95': wt_hd95,
                'ed_dice': ed_dice, 'ed_iou': ed_iou, 'ed_hd95': ed_hd95,
            }
        else:
            # Skip validation this epoch
            if last_val_metrics is not None:
                dice_m = last_val_metrics['dice']
                iou_m = last_val_metrics['iou']
                hd95_m = last_val_metrics['hd95']
                acc_m = last_val_metrics['acc']
                et_dice = last_val_metrics['et_dice']
                et_iou = last_val_metrics['et_iou']
                et_hd95 = last_val_metrics['et_hd95']
                tc_dice = last_val_metrics['tc_dice']
                tc_iou = last_val_metrics['tc_iou']
                tc_hd95 = last_val_metrics['tc_hd95']
                wt_dice = last_val_metrics['wt_dice']
                wt_iou = last_val_metrics['wt_iou']
                wt_hd95 = last_val_metrics['wt_hd95']
                ed_dice = last_val_metrics['ed_dice']
                ed_iou = last_val_metrics['ed_iou']
                ed_hd95 = last_val_metrics['ed_hd95']
            else:
                iou_m = best_iou  # Use previous best
                dice_m = 0.0
                acc_m = 0.0
                hd95_m = -1.0
                et_dice = et_iou = et_hd95 = wt_dice = wt_iou = wt_hd95 = tc_dice = tc_iou = tc_hd95 = ed_dice = ed_iou = ed_hd95 = 0.0
        avg_train_loss = train_loss_sum / len(train_loader)
        epoch_time = time.time() - epoch_start_time

        # Log to file logger
        log_dict = {
            'train_loss': avg_train_loss,
            'val_dice': dice_m,
            'val_hd95': hd95_m,
            'val_acc': acc_m,
            'lr': opt.param_groups[0]['lr'],
            'time_s': epoch_time
        }
        if num_classes_seg > 1:
            log_dict.update({
                'ET_dice': et_dice, 'ET_iou': et_iou, 'ET_hd95': et_hd95,
                'TC_dice': tc_dice, 'TC_iou': tc_iou, 'TC_hd95': tc_hd95,
                'WT_dice': wt_dice, 'WT_iou': wt_iou, 'WT_hd95': wt_hd95,
                'ED_dice': ed_dice, 'ED_iou': ed_iou, 'ED_hd95': ed_hd95
            })
        logger.epoch_end(epoch, cfg["train"]["epochs"], log_dict, "SUMMARY")

        # Log to metrics logger (CSV/JSON)
        metrics_dict = {
            'train_loss': avg_train_loss,
            'val_dice': dice_m,
            'val_hd95': hd95_m,
            'val_acc': acc_m,
            'learning_rate': opt.param_groups[0]['lr'],
            'epoch_time_s': epoch_time
        }
        if num_classes_seg > 1:
            metrics_dict.update({
                'ET_dice': et_dice, 'ET_iou': et_iou, 'ET_hd95': et_hd95,
                'TC_dice': tc_dice, 'TC_iou': tc_iou, 'TC_hd95': tc_hd95,
                'WT_dice': wt_dice, 'WT_iou': wt_iou, 'WT_hd95': wt_hd95,
                'ED_dice': ed_dice, 'ED_iou': ed_iou, 'ED_hd95': ed_hd95
            })
        metrics_logger.log_epoch(epoch, metrics_dict)

        # Console output
        if num_classes_seg > 1:
            hd95_str = f"{hd95_m:.2f}" if hd95_m >= 0 else "N/A"
            # 4-class: show ET, TC, WT (standard BraTS) - ED optional
            if num_classes_seg == 4:
                print(f"[Fold {fold}] Epoch {epoch+1}/{cfg['train']['epochs']} | Loss {avg_train_loss:.4f} | ET {et_dice:.4f} | TC {tc_dice:.4f} | WT {wt_dice:.4f} | Mean {dice_m:.4f} | HD95 {hd95_str}")
            else:
                # 3-class: show WT, TC, ED
                print(f"[Fold {fold}] Epoch {epoch+1}/{cfg['train']['epochs']} | Loss {avg_train_loss:.4f} | WT {wt_dice:.4f} | TC {tc_dice:.4f} | ED {ed_dice:.4f} | Mean {dice_m:.4f} | HD95 {hd95_str}")
        else:
            hd95_str = f"{hd95_m:.2f}" if hd95_m >= 0 else "N/A"
            print(f"[Fold {fold}] Epoch {epoch+1}/{cfg['train']['epochs']} | Train Loss {avg_train_loss:.4f} | Dice {dice_m:.4f} | HD95 {hd95_str} | ClsAcc {acc_m:.4f}")

        # Log validation metrics to TensorBoard
        if writer:
            writer.add_scalar('val/dice', dice_m, epoch)
            if hd95_m >= 0:  # Only log if valid
                writer.add_scalar('val/hd95', hd95_m, epoch)
            writer.add_scalar('val/cls_acc', acc_m, epoch)
            writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)

            # Log multiclass metrics if applicable
            if num_classes_seg > 1:
                # ET metrics (4-class only)
                if num_classes_seg == 4 and et_dice > 0:
                    writer.add_scalar('val/ET_dice', et_dice, epoch)
                    writer.add_scalar('val/ET_iou', et_iou, epoch)
                    if et_hd95 >= 0:
                        writer.add_scalar('val/ET_hd95', et_hd95, epoch)
                # TC metrics
                writer.add_scalar('val/TC_dice', tc_dice, epoch)
                writer.add_scalar('val/TC_iou', tc_iou, epoch)
                if tc_hd95 >= 0:
                    writer.add_scalar('val/TC_hd95', tc_hd95, epoch)
                # WT metrics
                writer.add_scalar('val/WT_dice', wt_dice, epoch)
                writer.add_scalar('val/WT_iou', wt_iou, epoch)
                if wt_hd95 >= 0:
                    writer.add_scalar('val/WT_hd95', wt_hd95, epoch)
                # ED metrics
                writer.add_scalar('val/ED_dice', ed_dice, epoch)
                writer.add_scalar('val/ED_iou', ed_iou, epoch)
                if ed_hd95 >= 0:
                    writer.add_scalar('val/ED_hd95', ed_hd95, epoch)

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
        # Save the active scheduler (plateau, onecycle, or cosine_restarts)
        active_scheduler = plateau_scheduler if plateau_scheduler is not None else (
            onecycle_scheduler if onecycle_scheduler is not None else cosine_restarts_scheduler
        )
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
