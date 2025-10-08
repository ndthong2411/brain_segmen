import os, yaml, torch
from typing import Any, Dict

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_ckpt(model, path: str):
    """Save only model weights (for best checkpoint)"""
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)

def load_ckpt(model, path: str, map_location="cpu"):
    """Load only model weights"""
    sd = torch.load(path, map_location=map_location)
    model.load_state_dict(sd)
    return model

def save_training_state(path: str, epoch: int, model, optimizer, scheduler, scaler,
                       best_iou: float, best_iou_epoch: int, config: Dict = None, fold: int = None):
    """
    Save complete training state for resuming.

    Args:
        path: Path to save checkpoint
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state (can be None)
        scaler: GradScaler state (can be None)
        best_iou: Best IoU so far
        best_iou_epoch: Epoch with best IoU
        config: Training config (optional)
    """
    ensure_dir(os.path.dirname(path))

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': best_iou,
        'best_iou_epoch': best_iou_epoch,
        'fold': fold,  # Store fold number for validation
    }

    # Add scheduler state if exists
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Add scaler state if exists
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    # Add config if provided
    if config is not None:
        checkpoint['config'] = config

    torch.save(checkpoint, path)
    print(f"Saved training state to: {path}")

def load_training_state(path: str, model, optimizer, scheduler=None, scaler=None, map_location="cpu", expected_fold=None):
    """
    Load complete training state for resuming.

    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: LR scheduler to load state into (optional)
        scaler: GradScaler to load state into (optional)
        map_location: Device to map tensors to

    Returns:
        dict with keys: epoch, best_iou, best_iou_epoch, config
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)

    # Validate fold if provided
    checkpoint_fold = checkpoint.get('fold', None)
    if expected_fold is not None and checkpoint_fold is not None:
        if checkpoint_fold != expected_fold:
            raise ValueError(
                f"Fold mismatch! Checkpoint is for fold {checkpoint_fold}, "
                f"but you're trying to resume fold {expected_fold}. "
                f"Please use the correct checkpoint: last_fold{expected_fold}.pth"
            )

    # Load model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load scaler if provided
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Return training info
    info = {
        'epoch': checkpoint['epoch'],
        'best_iou': checkpoint.get('best_iou', -1.0),
        'best_iou_epoch': checkpoint.get('best_iou_epoch', 0),
        'config': checkpoint.get('config', None),
    }

    print(f"Loaded training state from: {path}")
    print(f"  Resuming from epoch {info['epoch'] + 1}")
    print(f"  Best IoU so far: {info['best_iou']:.4f} (epoch {info['best_iou_epoch'] + 1})")

    return info
