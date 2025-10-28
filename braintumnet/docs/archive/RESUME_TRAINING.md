# Resume Training Guide

## Overview

Your training can now be **interrupted and resumed** without losing progress. The code automatically saves checkpoints every epoch.

---

## 🎯 Key Features

✅ **Automatic checkpoint saving** - Every epoch saves complete training state
✅ **Resume from any epoch** - Continue exactly where you left off
✅ **Preserves optimizer state** - Learning rate, momentum, etc.
✅ **Preserves scheduler state** - LR schedule continues correctly
✅ **Preserves training stats** - Best IoU, epoch numbers, etc.

---

## 📁 Checkpoint Files

### **Two Types of Checkpoints**:

1. **`best_fold{N}.pth`** - Best model by IoU (model weights only)
   - Used for evaluation
   - Lightweight (~60MB)
   - Only contains model.state_dict()

2. **`last_fold{N}.pth`** - Latest training state (complete state)
   - Used for resuming training
   - Larger (~120MB)
   - Contains: model + optimizer + scheduler + scaler + metadata

**Location**: `checkpoints/`

---

## 🚀 How to Use

### **Normal Training** (from scratch):
```bash
cd braintumnet
python scripts/train.py --cfg configs/full_dataset.yaml --fold 0
```

### **Resume Training** (AUTO - RECOMMENDED ✨):
```bash
cd braintumnet
python scripts/train.py --cfg configs/full_dataset.yaml --fold 0 --resume
```
**Tự động tìm `checkpoints/last_fold0.pth`!** Không cần chỉ định path.

**Output when resuming**:
```
✓ Auto-detected checkpoint: checkpoints/last_fold0.pth
Loaded training state from: checkpoints/last_fold0.pth
  Resuming from epoch 51
  Best IoU so far: 0.6543 (epoch 45)
Starting training for 150 epochs...
Epoch 51/150 - TRAIN
...
```

### **Resume Training** (Manual path - optional):
```bash
cd braintumnet
python scripts/train.py --cfg configs/full_dataset.yaml --fold 0 --resume checkpoints/last_fold0.pth
```
⚠️ Chỉ cần khi muốn resume từ checkpoint khác (không phải `last_fold{N}.pth`)

---

## 💡 Common Scenarios

### **Scenario 1: Training Interrupted by Accident**
```bash
# Training was running, but crashed/stopped at epoch 75
# Just add --resume flag:

python scripts/train.py --cfg configs/full_dataset.yaml --fold 0 --resume
```
✓ Auto-finds `checkpoints/last_fold0.pth`
✓ Continues from epoch 76
✓ Preserves best IoU and all training state

---

### **Scenario 2: Want to Train Longer**
```bash
# Trained 150 epochs, but want to continue to 200

# 1. Edit config: epochs: 150 → 200
# 2. Resume:

python scripts/train.py --cfg configs/full_dataset.yaml --fold 0 --resume
```
✓ Auto-finds checkpoint
✓ Continues from epoch 151 to 200
✓ Uses same learning rate schedule

---

### **Scenario 3: System Maintenance**
```bash
# Need to restart computer during training

# 1. Stop training (Ctrl+C or let it finish epoch)
# 2. Do system maintenance
# 3. Resume:

python scripts/train.py --cfg configs/full_dataset.yaml --fold 0 --resume
```
✓ Auto-finds checkpoint
✓ No progress lost (only that epoch if interrupted mid-epoch)

---

### **Scenario 4: Experiment with Learning Rate**
```bash
# Trained 100 epochs, want to lower LR and continue

# 1. Edit config: lr: 2e-4 → 1e-4
# 2. Resume (optimizer LR will be overridden by config):

python scripts/train.py --cfg configs/full_dataset.yaml --fold 0 --resume checkpoints/last_fold0.pth
```
⚠️ Note: Config LR takes precedence when resuming

---

## 📊 What's Saved in Checkpoint

```python
checkpoint = {
    'epoch': 75,                        # Last completed epoch
    'model_state_dict': {...},          # Model weights
    'optimizer_state_dict': {...},      # Optimizer state (LR, momentum, etc.)
    'scheduler_state_dict': {...},      # LR scheduler state
    'scaler_state_dict': {...},         # AMP scaler state
    'best_iou': 0.6543,                 # Best IoU so far
    'best_iou_epoch': 68,               # Epoch with best IoU
    'config': {...}                     # Training config (reference)
}
```

---

## ⚙️ Advanced Usage

### **Resume Specific Checkpoint**
```bash
# Resume from a specific epoch (if you saved manually)
python scripts/train.py --cfg configs/full_dataset.yaml --fold 0 --resume checkpoints/epoch_100_fold0.pth
```

### **Resume All Folds**
```bash
# If training all 5 folds was interrupted, resume each:
python scripts/train.py --cfg configs/full_dataset.yaml --fold 0 --resume checkpoints/last_fold0.pth
python scripts/train.py --cfg configs/full_dataset.yaml --fold 1 --resume checkpoints/last_fold1.pth
python scripts/train.py --cfg configs/full_dataset.yaml --fold 2 --resume checkpoints/last_fold2.pth
python scripts/train.py --cfg configs/full_dataset.yaml --fold 3 --resume checkpoints/last_fold3.pth
python scripts/train.py --cfg configs/full_dataset.yaml --fold 4 --resume checkpoints/last_fold4.pth
```

### **Resume Multi-Modal Training**
```bash
python scripts/train.py --cfg configs/full_dataset_multimodal.yaml --fold 0 --resume checkpoints/last_fold0.pth
```

---

## 🔍 Verify Checkpoint

### **Check what's in a checkpoint**:
```python
import torch

ckpt = torch.load("checkpoints/last_fold0.pth")
print(f"Epoch: {ckpt['epoch']}")
print(f"Best IoU: {ckpt['best_iou']:.4f}")
print(f"Best IoU Epoch: {ckpt['best_iou_epoch']}")
print(f"Keys: {ckpt.keys()}")
```

---

## ⚠️ Important Notes

### **1. Checkpoint Compatibility**
- Checkpoints are **tied to the model architecture**
- If you change model config (base, dim, depth), old checkpoints won't work
- Best practice: Keep same model architecture when resuming

### **2. Config Changes**
When resuming, these config changes are **applied**:
- ✓ epochs (can train longer)
- ✓ learning rate (optimizer state overridden)
- ✓ early_stop_patience

These config changes are **NOT applied** (uses checkpoint values):
- ❌ model architecture (must match checkpoint)
- ❌ data augmentation (already trained with old settings)

### **3. Disk Space**
- `last_fold{N}.pth` files are larger (~2× size of best checkpoint)
- They are overwritten every epoch (only keeps latest)
- If disk space is tight, can disable by commenting out save_training_state() call

### **4. Best Checkpoint**
- `best_fold{N}.pth` is still saved separately (for evaluation)
- Resuming doesn't affect best checkpoint
- Best checkpoint is always the best IoU across all epochs

---

## 🐛 Troubleshooting

### **Error: "Checkpoint not found"**
```
FileNotFoundError: Checkpoint not found: checkpoints/last_fold0.pth
```
**Solution**: Make sure checkpoint path is correct. Use absolute path if needed:
```bash
python scripts/train.py --cfg configs/full_dataset.yaml --fold 0 \
    --resume "E:\thong\code\brain_segmen\braintumnet\checkpoints\last_fold0.pth"
```

### **Error: "Size mismatch"**
```
RuntimeError: Error(s) in loading state_dict for BrainTumNet:
    size mismatch for seg.e1.block.0.0.weight...
```
**Solution**: Model architecture changed. Cannot resume from this checkpoint. Either:
1. Use checkpoint with matching architecture
2. Start training from scratch

### **Resume starts from epoch 1 instead of saved epoch**
**Issue**: Checkpoint might be corrupted or from different fold.
**Solution**: Check checkpoint content:
```python
import torch
ckpt = torch.load("checkpoints/last_fold0.pth")
print(ckpt['epoch'])  # Should show the epoch number
```

---

## 📝 API Reference

### **Save Training State**
```python
from braintumnet.utils.io import save_training_state

save_training_state(
    path="checkpoints/last_fold0.pth",
    epoch=75,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,      # Can be None
    scaler=scaler,            # Can be None
    best_iou=0.6543,
    best_iou_epoch=68,
    config=cfg                # Optional
)
```

### **Load Training State**
```python
from braintumnet.utils.io import load_training_state

info = load_training_state(
    path="checkpoints/last_fold0.pth",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,      # Can be None
    scaler=scaler,            # Can be None
    map_location="cuda"       # or "cpu"
)

# Returns dict with:
# {
#     'epoch': 75,
#     'best_iou': 0.6543,
#     'best_iou_epoch': 68,
#     'config': {...}
# }
```

---

## ✅ Summary

| Feature | Status |
|---------|--------|
| **Auto-save every epoch** | ✅ Enabled |
| **Resume from checkpoint** | ✅ `--resume` flag |
| **Preserve optimizer** | ✅ Yes |
| **Preserve scheduler** | ✅ Yes |
| **Preserve best IoU** | ✅ Yes |
| **Multiple folds** | ✅ Supported |
| **Multi-modal** | ✅ Supported |

**No data loss!** Train confidently knowing you can resume anytime. 🎉

---

**Created**: 2025-10-06
**Files Modified**:
- [utils/io.py](../src/braintumnet/utils/io.py) - Added save/load_training_state()
- [engine/trainer.py](../src/braintumnet/engine/trainer.py) - Added resume logic
- [scripts/train.py](../scripts/train.py) - Added --resume argument
