# Part 9: Troubleshooting Guide

**Navigation**: [[TECHNICAL_REPORT_INDEX|← Back to Index]]

---

## Table of Contents

1. [Common Errors](#common-errors)
2. [Installation Issues](#installation-issues)
3. [Data Preprocessing Problems](#data-preprocessing-problems)
4. [Training Issues](#training-issues)
5. [Memory Problems](#memory-problems)
6. [Performance Issues](#performance-issues)
7. [Inference Problems](#inference-problems)
8. [Debugging Strategies](#debugging-strategies)

---

## Common Errors

### Error: "CUDA out of memory"

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB
(GPU 0; 11.00 GiB total capacity; 9.12 GiB already allocated; ...)
```

**Causes**:
1. Batch size too large
2. Model too big
3. Image resolution too high
4. Memory leak (not freeing tensors)

**Solutions**:

**Solution 1**: Reduce batch size
```yaml
# In config file
train:
  batch_size: 8  # Reduce from 12
```

**Solution 2**: Enable mixed precision
```yaml
train:
  amp: true  # FP16 uses 50% less memory
```

**Solution 3**: Reduce model size
```yaml
model:
  base: 16  # Reduce from 32
```

**Solution 4**: Reduce image size
```yaml
data:
  img_size: 128  # Reduce from 256
```

**Solution 5**: Clear GPU cache
```python
# Add to training loop
import torch
torch.cuda.empty_cache()
```

**Solution 6**: Use gradient accumulation
```python
# In trainer.py, modify training loop
accumulation_steps = 4  # Effective batch size = 12 × 4 = 48

for i, batch in enumerate(train_loader):
    loss = compute_loss(...)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

### Error: "Checkpoint fold mismatch"

**Symptoms**:
```
ValueError: Fold mismatch! Checkpoint is for fold 0,
but you're trying to resume fold 1.
```

**Cause**: Resuming training with wrong checkpoint

**Solution**:
```bash
# Check which fold checkpoint is for
python -c "import torch; print(torch.load('checkpoints/last_fold0.pth')['fold'])"
# Output: 0

# Use correct checkpoint
python train.py --cfg configs/default.yaml --fold 0 --resume checkpoints/last_fold0.pth
```

---

### Error: "File not found: split_train_fold0.txt"

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory:
'data/processed_full_multimodal/split_train_fold0.txt'
```

**Cause**: Forgot to run preprocessing

**Solution**:
```bash
# Run preprocessing first
python scripts/prepare_brats2020_h5.py \
    --h5 data/raw/brats2020_training.h5 \
    --out data/processed_full_multimodal \
    --modality multi \
    --img_size 256 \
    --slices_per_case 30 \
    --num_folds 5

# Then train
python train.py --cfg configs/full_dataset_multimodal.yaml --fold 0
```

---

### Error: "NaN loss during training"

**Symptoms**:
```
Epoch 5/100 | Train Loss nan | Val IoU nan | Dice nan
```

**Causes**:
1. Learning rate too high
2. Gradient explosion
3. Numerical instability
4. Bad data (inf/nan values)

**Solutions**:

**Solution 1**: Lower learning rate
```yaml
train:
  lr: 5.0e-5  # Reduce from 1.5e-4
```

**Solution 2**: Add gradient clipping
```python
# In trainer.py
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Solution 3**: Check data
```python
# Add to dataloader
import torch
import numpy as np

def check_batch(batch):
    img = batch["image"]
    msk = batch["mask"]

    if torch.isnan(img).any():
        print("NaN in images!")
    if torch.isinf(img).any():
        print("Inf in images!")
    if img.max() > 10 or img.min() < -10:
        print(f"Suspicious values: min={img.min()}, max={img.max()}")

# Use in training loop
for batch in train_loader:
    check_batch(batch)
    ...
```

**Solution 4**: Enable anomaly detection
```python
# At start of train.py
import torch
torch.autograd.set_detect_anomaly(True)
```

---

### Error: "Expected 4D tensor, got 3D"

**Symptoms**:
```
RuntimeError: Expected 4D (unbatched 3D) or 5D (batched 4D) input to conv2d,
but got input of size: [256, 256]
```

**Cause**: Missing batch or channel dimension

**Solution**:
```python
# Correct shapes:
# Training: (B, C, H, W) = (12, 4, 256, 256)
# Inference: (1, C, H, W) = (1, 4, 256, 256)

# Add batch dimension
img_tensor = img_tensor.unsqueeze(0)  # (C, H, W) → (1, C, H, W)

# Add channel dimension
img_tensor = img_tensor.unsqueeze(0)  # (H, W) → (1, H, W)
```

---

## Installation Issues

### Error: "No module named 'torch'"

**Symptoms**:
```
ModuleNotFoundError: No module named 'torch'
```

**Solution**:
```bash
# Install PyTorch (check https://pytorch.org for your system)

# CPU only
pip install torch torchvision

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### Error: "CUDA not available"

**Symptoms**:
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Diagnosis**:
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Check if PyTorch built with CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**Solutions**:

**Solution 1**: Install CUDA-enabled PyTorch
```bash
# Reinstall with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Solution 2**: Update NVIDIA drivers
```bash
# Ubuntu/Linux
sudo ubuntu-drivers autoinstall
sudo reboot

# Windows
# Download from https://www.nvidia.com/Download/index.aspx
```

**Solution 3**: Use CPU (slower)
```yaml
# In training, model automatically uses CPU if CUDA unavailable
# Expect 10-20× slower training
```

---

## Data Preprocessing Problems

### Error: "HDF5 file corrupt"

**Symptoms**:
```
OSError: Unable to open file (file signature not found)
```

**Solution**:
```bash
# Re-download HDF5 file
# Check file integrity
md5sum data/raw/brats2020_training.h5

# If corrupt, download again from BraTS
```

---

### Error: "Preprocessing crashes midway"

**Symptoms**:
```
Processing patient 234/369...
Killed
```

**Cause**: Out of RAM

**Solution**:
```bash
# Process in smaller batches
python scripts/prepare_brats2020_h5.py \
    --h5 data/raw/brats2020_training.h5 \
    --out data/processed_full_multimodal \
    --modality multi \
    --start_idx 0 \
    --end_idx 100  # Process 100 patients at a time

# Then continue
python scripts/prepare_brats2020_h5.py \
    ... \
    --start_idx 100 \
    --end_idx 200

# (Code modification required to support --start_idx/--end_idx)
```

---

### Warning: "Imbalanced dataset"

**Symptoms**:
```
WARNING: Tumor slices: 2340 (21%), Non-tumor: 8730 (79%)
```

**Solution**: Already handled by `tumor_slice_ratio` in config
```yaml
data:
  tumor_slice_ratio: 0.5  # Balances to 50-50
```

---

## Training Issues

### Error: "Training stuck at low accuracy"

**Symptoms**:
```
Epoch 50/100 | Dice 0.65 | Not improving
```

**Diagnosis**:
```python
# Check learning rate
print(f"Current LR: {optimizer.param_groups[0]['lr']}")

# Check if model updates
before = model.seg.e1.block[0].weight.clone()
# Train one batch
after = model.seg.e1.block[0].weight
print(f"Weight changed: {not torch.equal(before, after)}")
```

**Solutions**:

**Solution 1**: Increase learning rate
```yaml
train:
  lr: 3.0e-4  # Increase from 1.5e-4
```

**Solution 2**: Check if frozen layers
```python
# All parameters should be trainable
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"Frozen: {name}")
        param.requires_grad = True
```

**Solution 3**: Reinitialize model
```python
# Sometimes bad initialization
model = build_model(cfg).to(device)
# Retrain from scratch
```

---

### Error: "Validation worse than training"

**Symptoms**:
```
Train Dice: 0.95 | Val Dice: 0.75 (15% gap)
```

**Cause**: Overfitting

**Solutions**:

**Solution 1**: Increase weight decay
```yaml
train:
  weight_decay: 5.0e-4  # Increase from 1.0e-4
```

**Solution 2**: Add dropout
```python
# In model definition
self.dropout = nn.Dropout(0.3)

# In forward pass
x = self.dropout(x)
```

**Solution 3**: More augmentation
```yaml
augment:
  rotate_deg: 30  # Increase from 20
  hflip_p: 0.7    # Increase from 0.5
```

**Solution 4**: Use early stopping (already in config)
```yaml
train:
  early_stop_patience: 20  # Reduce from 30
```

---

### Error: "Loss oscillating wildly"

**Symptoms**:
```
Epoch 10: loss=0.25
Epoch 11: loss=0.45  ← Big jump
Epoch 12: loss=0.22
Epoch 13: loss=0.52  ← Another jump
```

**Cause**: Learning rate too high

**Solution**:
```yaml
train:
  lr: 5.0e-5  # Reduce from 1.5e-4
```

---

## Memory Problems

### Error: "System RAM full"

**Symptoms**:
```
MemoryError: Unable to allocate array
```

**Cause**: Data loading using too much RAM

**Solutions**:

**Solution 1**: Reduce num_workers
```yaml
train:
  workers: 2  # Reduce from 4
```

**Solution 2**: Use lazy loading (already implemented in Dataset)

**Solution 3**: Reduce batch size
```yaml
train:
  batch_size: 8  # Reduce from 12
```

---

### Error: "Disk space full"

**Symptoms**:
```
OSError: [Errno 28] No space left on device
```

**Solutions**:

**Solution 1**: Clean up old checkpoints
```bash
# Keep only best checkpoints
cd checkpoints
rm last_fold*.pth  # Remove intermediate checkpoints
```

**Solution 2**: Reduce TensorBoard logging
```yaml
logging:
  use_tensorboard: false  # Disable TensorBoard
```

**Solution 3**: Use external drive
```yaml
logging:
  out_dir: "/mnt/external/runs"
  save_dir: "/mnt/external/checkpoints"
```

---

## Performance Issues

### Issue: "Training too slow"

**Symptoms**:
```
2.5 it/s (expected 4-5 it/s)
```

**Diagnosis**:
```python
import time

# Time data loading
start = time.time()
for i, batch in enumerate(train_loader):
    if i == 100:
        break
data_time = time.time() - start
print(f"Data loading: {data_time/100:.3f}s per batch")

# Time forward pass
start = time.time()
with torch.no_grad():
    for i, batch in enumerate(train_loader):
        img = batch["image"].to(device)
        model(img)
        if i == 100:
            break
forward_time = time.time() - start
print(f"Forward pass: {forward_time/100:.3f}s per batch")
```

**Solutions**:

**Solution 1**: Enable mixed precision (if not already)
```yaml
train:
  amp: true  # 2× speedup
```

**Solution 2**: Increase num_workers (if data loading slow)
```yaml
train:
  workers: 8  # Increase from 4 (if CPU has cores)
```

**Solution 3**: Pin memory
```python
# In dataloader
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # Add this for faster GPU transfer
)
```

**Solution 4**: Use faster GPU
```
RTX 3050:  2.5 it/s
RTX 3060:  4.7 it/s  ← Current
RTX 3090:  9.2 it/s
RTX 4090: 14.5 it/s
```

---

### Issue: "Poor segmentation accuracy"

**Symptoms**:
```
Dice < 0.80 (expected > 0.90)
```

**Diagnosis**:
```python
# Check dataset
print(f"Num train samples: {len(train_ds)}")
print(f"Num val samples: {len(val_ds)}")

# Check data distribution
labels = []
for i in range(len(train_ds)):
    labels.append(train_ds[i]["label"])
print(f"Class distribution: {np.bincount(labels)}")

# Visualize predictions
import matplotlib.pyplot as plt
img, msk = val_ds[0]["image"], val_ds[0]["mask"]
with torch.no_grad():
    pred = model(img.unsqueeze(0).to(device))
    pred = torch.sigmoid(pred[0]).cpu().squeeze()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img.squeeze(), cmap='gray')
axes[0].set_title('Input')
axes[1].imshow(msk.squeeze(), cmap='gray')
axes[1].set_title('Ground Truth')
axes[2].imshow(pred, cmap='hot')
axes[2].set_title('Prediction')
plt.show()
```

**Solutions**:

**Solution 1**: Train longer
```yaml
train:
  epochs: 200  # Increase from 150
  early_stop_patience: 50
```

**Solution 2**: Use multi-modal (if using single-modal)
```yaml
data:
  modality: "multi"  # Use all 4 modalities
model:
  in_channels: 4
```

**Solution 3**: Tune loss weights
```yaml
train:
  seg_loss_weight: 1.5  # Emphasize segmentation
  cls_loss_weight: 0.3
```

---

## Inference Problems

### Error: "Model outputs wrong shape"

**Symptoms**:
```
Expected (1, 1, 256, 256), got (1, 2, 256, 256)
```

**Cause**: Model returns both seg and cls

**Solution**:
```python
# Unpack both outputs
seg_logits, cls_logits = model(img)

# Use only segmentation
seg_prob = torch.sigmoid(seg_logits)
```

---

### Error: "Prediction all zeros"

**Symptoms**:
```
Prediction mask is blank (all zeros)
```

**Diagnosis**:
```python
# Check model output range
seg_logits, _ = model(img)
print(f"Logits range: [{seg_logits.min():.2f}, {seg_logits.max():.2f}]")

seg_prob = torch.sigmoid(seg_logits)
print(f"Prob range: [{seg_prob.min():.4f}, {seg_prob.max():.4f}]")

# If prob max < 0.5, no pixels will be predicted
```

**Solutions**:

**Solution 1**: Lower threshold
```python
threshold = 0.3  # Lower from 0.5
seg_binary = (seg_prob > threshold).float()
```

**Solution 2**: Check if model loaded correctly
```python
# Verify checkpoint
checkpoint = torch.load(ckpt_path)
print(f"Checkpoint keys: {checkpoint.keys()}")

# Load and verify
load_ckpt(model, ckpt_path)
print("Model loaded successfully")
```

**Solution 3**: Check input normalization
```python
# Input should be in [0, 1]
print(f"Input range: [{img.min():.2f}, {img.max():.2f}]")

# If not, normalize
img = (img - img.min()) / (img.max() - img.min() + 1e-6)
```

---

## Debugging Strategies

### Strategy 1: Overfit on One Batch

**Purpose**: Verify model can learn

```python
# In training script
single_batch = next(iter(train_loader))

for epoch in range(100):
    optimizer.zero_grad()
    seg, cls = model(single_batch["image"].to(device))
    loss, _, _ = criterion(seg, single_batch["mask"].to(device),
                          cls, single_batch["label"].to(device))
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

# Expected: Loss should go to near 0
# If not: Problem with model/optimizer
```

---

### Strategy 2: Visualize Intermediate Features

**Purpose**: Understand what model learns

```python
import matplotlib.pyplot as plt

# Hook to capture activations
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks
model.seg.e1.block.register_forward_hook(get_activation('e1'))
model.seg.e2.block.register_forward_hook(get_activation('e2'))
model.seg.e3.block.register_forward_hook(get_activation('e3'))

# Forward pass
with torch.no_grad():
    model(img.unsqueeze(0).to(device))

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(activations['e1'][0, 0].cpu(), cmap='viridis')
axes[0].set_title('Encoder 1 (32 channels)')
axes[1].imshow(activations['e2'][0, 0].cpu(), cmap='viridis')
axes[1].set_title('Encoder 2 (64 channels)')
axes[2].imshow(activations['e3'][0, 0].cpu(), cmap='viridis')
axes[2].set_title('Encoder 3 (128 channels)')
plt.show()
```

---

### Strategy 3: Compare with Baseline

**Purpose**: Isolate problem

```python
# Train simple U-Net (no attention, no transformer)
from braintumnet.models.seg_unet import SegUNetMasked

simple_unet = SegUNetMasked(in_ch=4, base=32, dim=256,
                             patch=8, depth=0, n_heads=4)  # depth=0 disables transformer

# Train and compare
# If simple U-Net works but full model doesn't → problem in transformer/attention
# If simple U-Net also fails → problem in data/training
```

---

### Strategy 4: Check Gradients

**Purpose**: Verify backprop works

```python
# After loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm == 0:
            print(f"Zero gradient: {name}")
        elif grad_norm > 100:
            print(f"Exploding gradient: {name} (norm={grad_norm:.2f})")
```

---

### Strategy 5: Log Everything

**Purpose**: Track training details

```python
# In training loop
import wandb  # Or TensorBoard

wandb.init(project="braintumnet-debug")

for epoch in range(epochs):
    for i, batch in enumerate(train_loader):
        # ... training code ...

        # Log extensively
        wandb.log({
            'loss': loss.item(),
            'seg_loss': l_seg.item(),
            'cls_loss': l_cls.item(),
            'lr': optimizer.param_groups[0]['lr'],
            'grad_norm': sum(p.grad.norm().item() for p in model.parameters()),
            'weight_norm': sum(p.norm().item() for p in model.parameters()),
        })

        if i % 100 == 0:
            # Log images
            wandb.log({
                'input': wandb.Image(batch["image"][0]),
                'mask': wandb.Image(batch["mask"][0]),
                'pred': wandb.Image(torch.sigmoid(seg[0]) > 0.5),
            })
```

---

## Quick Reference

### Checklist for Training Issues

- [ ] CUDA available? (`torch.cuda.is_available()`)
- [ ] Data preprocessed? (check `data/processed_*/`)
- [ ] Correct config? (check `exp_name`, `in_channels`, etc.)
- [ ] Sufficient GPU memory? (reduce `batch_size` if OOM)
- [ ] Mixed precision enabled? (`amp: true`)
- [ ] Learning rate reasonable? (1e-5 to 1e-3)
- [ ] Model updating? (check gradients)
- [ ] Data loading fast? (check `num_workers`)
- [ ] Augmentation reasonable? (not too extreme)
- [ ] Checkpoints saving? (check `checkpoints/`)

### Common Parameter Adjustments

| Issue | Parameter | Direction |
|-------|-----------|-----------|
| OOM | `batch_size` | ↓ Decrease |
| OOM | `base` | ↓ Decrease |
| OOM | `img_size` | ↓ Decrease |
| Slow training | `amp` | Enable |
| Slow training | `workers` | ↑ Increase |
| Overfitting | `weight_decay` | ↑ Increase |
| Overfitting | `rotate_deg` | ↑ Increase |
| Underfitting | `epochs` | ↑ Increase |
| Underfitting | `base` | ↑ Increase |
| Unstable | `lr` | ↓ Decrease |
| Unstable | Gradient clip | Add |

---

**Next**: [[10_EXTENSION_GUIDE|Part 10: Extension Guide →]]

**Back**: [[08_RESULTS_ANALYSIS|← Part 8: Results Analysis]] | [[TECHNICAL_REPORT_INDEX|Index]]
