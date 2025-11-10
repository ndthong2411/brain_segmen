# Phần 8: Troubleshooting và Best Practices

> **🔧 Xử Lý Sự Cố và Kinh Nghiệm Thực Tế**
>
> Tài liệu này tổng hợp các vấn đề thường gặp và cách giải quyết.

---

## Mục Lục

1. [Training Issues](#1-training-issues)
2. [Memory Issues](#2-memory-issues)
3. [Performance Issues](#3-performance-issues)
4. [Data Issues](#4-data-issues)
5. [Best Practices](#5-best-practices)
6. [Common Mistakes](#6-common-mistakes)

---

## 1. Training Issues

### Loss Không Giảm

**Symptoms**:
```
Epoch 1: Loss = 1.85
Epoch 10: Loss = 1.82
Epoch 20: Loss = 1.80
Epoch 50: Loss = 1.78  ← Stuck!
```

**Possible Causes và Solutions**:

**1. Learning Rate quá nhỏ**
```python
# Check learning rate
current_lr = optimizer.param_groups[0]['lr']
print(f"Current LR: {current_lr}")

# Solution: Increase LR
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # từ 1e-4
```

**2. Learning Rate quá lớn (loss oscillates)**
```
Epoch 1: Loss = 1.85
Epoch 2: Loss = 2.10
Epoch 3: Loss = 1.75
Epoch 4: Loss = 2.20  ← Unstable!
```
```python
# Solution: Decrease LR
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # từ 1e-4
```

**3. Gradient vanishing/exploding**
```python
# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: {grad_norm}")

# Very small (< 1e-6): Vanishing
# Very large (> 100): Exploding

# Solution: Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**4. Wrong loss function weights**
```python
# Check class weights
class_weights = [0.344, 5.865, 3.981]  # From data statistics

# Too aggressive weights có thể cause instability
# Solution: Normalize hoặc reduce range
class_weights = [1.0, 2.0, 1.5]  # More conservative
```

### Overfitting

**Symptoms**:
```
Epoch 50:  Train Dice = 0.95, Val Dice = 0.75
Epoch 100: Train Dice = 0.98, Val Dice = 0.72  ← Gap increasing!
```

**Solutions**:

**1. Regularization**
```python
# Increase weight decay
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-4, 
    weight_decay=1e-4  # Increased từ 1e-5
)

# Add dropout
model = BrainTumNetV2(
    dropout=0.2  # Increased từ 0.15
)
```

**2. Data Augmentation**
```python
# More aggressive augmentation
augmentation = BraTSAugmentation(
    horizontal_flip=0.7,    # từ 0.5
    vertical_flip=0.7,
    rotation=25,            # từ 15
    elastic_alpha=100,      # từ 50
    gaussian_noise=0.5      # từ 0.3
)
```

**3. Early Stopping**
```python
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
    
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        else:
            self.best_score = val_score
            self.counter = 0
        return False

# Usage
early_stopping = EarlyStopping(patience=20)
for epoch in range(num_epochs):
    train()
    val_dice = validate()
    
    if early_stopping(val_dice):
        print(f"Early stopping at epoch {epoch}")
        break
```

### NaN Loss

**Symptoms**:
```
Epoch 1: Loss = 1.85
Epoch 2: Loss = 1.72
Epoch 3: Loss = nan  ← Exploded!
```

**Causes và Solutions**:

**1. Learning rate quá cao**
```python
# Solution
lr = 1e-5  # Start lower
```

**2. Mixed precision issues (FP16)**
```python
# Check loss scaling
if scaler is not None:
    print(f"Loss scale: {scaler.get_scale()}")
    
    # If scale = 65536 (max) và vẫn nan:
    # → Gradients quá lớn
    
    # Solution: Use BFloat16 instead
    amp_dtype = 'bfloat16'  # No scaling needed
```

**3. Division by zero trong loss**
```python
# Check Dice loss smooth parameter
dice_loss = MultiClassDiceLoss(smooth=1.0)  # Ensure > 0

# Check metrics computation
dice = (2 * intersection + smooth) / (pred + target + smooth)
# Never: dice = 2 * intersection / (pred + target)  ← Can be 0/0!
```

**4. Invalid input data**
```python
# Check for NaN/Inf trong data
assert not torch.isnan(images).any(), "NaN in images!"
assert not torch.isinf(images).any(), "Inf in images!"

# Check labels range
assert masks.min() >= 0 and masks.max() < num_classes, "Invalid labels!"
```

---

## 2. Memory Issues

### CUDA Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 512 MiB 
(GPU 0; 23.70 GiB total capacity; 21.24 GiB already allocated; 
89.31 MiB free; 22.15 GiB reserved in total by PyTorch)
```

**Solutions**:

**1. Reduce Batch Size**
```python
# RTX 3090 (24GB): batch_size = 12
# RTX 3080 (10GB): batch_size = 4
# V100 (16GB): batch_size = 8

# In config
train:
  batch_size: 4  # Reduced
```

**2. Gradient Accumulation**
```python
# Effective batch = batch_size × accumulation_steps
# batch_size=4, accumulation=3 → effective=12

gradient_accumulation_steps = 3

for batch_idx, (images, masks, labels) in enumerate(train_loader):
    loss = compute_loss(images, masks, labels)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    
    if (batch_idx + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**3. Reduce Model Size**
```python
# Use smaller model
model = BrainTumNetV2(
    base=32,   # từ 48
    dim=256,   # từ 384
    depth=2,   # từ 4
)
```

**4. Mixed Precision**
```python
# Enable AMP
use_amp = True
amp_dtype = 'float16'  # 2× memory reduction
```

**5. Clear Cache**
```python
# After each epoch
torch.cuda.empty_cache()

# Inside training loop (every N batches)
if batch_idx % 100 == 0:
    torch.cuda.empty_cache()
```

**6. Disable Deep Supervision**
```python
# Deep supervision tạo auxiliary outputs → more memory
model = BrainTumNetV2(
    deep_supervision=False  # Disable
)
```

### Memory Leak

**Symptoms**:
```
Epoch 1:  GPU Memory = 8GB
Epoch 10: GPU Memory = 12GB
Epoch 20: GPU Memory = 16GB  ← Increasing!
```

**Common Causes**:

**1. Accumulating tensors trong loop**
```python
# ❌ Wrong
losses = []
for epoch in range(num_epochs):
    loss = train()
    losses.append(loss)  # Keeps tensor in memory!

# ✓ Correct
losses = []
for epoch in range(num_epochs):
    loss = train()
    losses.append(loss.item())  # Convert to Python float
```

**2. Not detaching từ computation graph**
```python
# ❌ Wrong
running_loss += loss

# ✓ Correct
running_loss += loss.item()
# hoặc
running_loss += loss.detach().cpu().item()
```

**3. TensorBoard logging**
```python
# ❌ Wrong
writer.add_scalar('Loss', loss, epoch)  # Keeps tensor!

# ✓ Correct
writer.add_scalar('Loss', loss.item(), epoch)
```

---

## 3. Performance Issues

### Slow Training

**Symptoms**:
```
Expected: ~7 min/epoch
Actual: ~25 min/epoch
```

**Optimizations**:

**1. Enable cuDNN benchmark**
```python
torch.backends.cudnn.benchmark = True
```

**2. Increase num_workers**
```python
train_loader = DataLoader(
    dataset,
    batch_size=12,
    num_workers=8,  # Increased từ 4
    pin_memory=True
)
```

**3. Pin memory**
```python
train_loader = DataLoader(
    dataset,
    pin_memory=True  # Faster data transfer
)
```

**4. Channels last memory format (A100)**
```python
model = model.to(memory_format=torch.channels_last)
images = images.to(memory_format=torch.channels_last)
```

**5. Compile model (PyTorch 2.0+)**
```python
model = torch.compile(model)
```

**6. Profiling**
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for i in range(10):
        output = model(input)
        loss = criterion(output, target)
        loss.backward()

print(prof.key_averages().table())
```

### Low Dice Score

**Symptoms**:
```
WT Dice: 0.65 (expected: 0.88+)
TC Dice: 0.55 (expected: 0.82+)
ED Dice: 0.45 (expected: 0.75+)
```

**Debugging Steps**:

**1. Verify data preprocessing**
```python
# Load và visualize sample
import matplotlib.pyplot as plt

flair = cv2.imread('data/flair/sample.png', cv2.IMREAD_GRAYSCALE)
seg = cv2.imread('data/seg/sample.png', cv2.IMREAD_GRAYSCALE)

print(f"FLAIR range: [{flair.min()}, {flair.max()}]")
print(f"Seg unique: {np.unique(seg)}")  # Should be [0, 1, 2]

plt.subplot(121); plt.imshow(flair, cmap='gray')
plt.subplot(122); plt.imshow(seg, cmap='jet')
plt.show()
```

**2. Check class distribution**
```python
# Label distribution
from collections import Counter
seg_flat = seg.flatten()
counts = Counter(seg_flat)

for cls, count in counts.items():
    pct = 100 * count / seg_flat.size
    print(f"Class {cls}: {pct:.2f}%")

# Should match expected:
# Class 0 (BG): ~87%
# Class 1 (TC): ~5%
# Class 2 (ED): ~7%
```

**3. Verify model output**
```python
# Forward pass
seg_logits, cls_logits = model(images)

print(f"Seg logits shape: {seg_logits.shape}")  # (B, 3, H, W)
print(f"Seg logits range: [{seg_logits.min():.2f}, {seg_logits.max():.2f}]")

# Check predictions
seg_pred = seg_logits.argmax(dim=1)
print(f"Pred unique: {torch.unique(seg_pred)}")  # Should be [0, 1, 2]

# Check prediction distribution
for cls in [0, 1, 2]:
    count = (seg_pred == cls).sum().item()
    pct = 100 * count / seg_pred.numel()
    print(f"Predicted class {cls}: {pct:.2f}%")
```

**4. Verify loss computation**
```python
# Compute loss manually
seg_loss, loss_dict = seg_loss_fn(seg_logits, masks)

print(f"Seg loss: {seg_loss.item():.4f}")
print(f"Dice: {loss_dict['dice']:.4f}")
print(f"Focal: {loss_dict['focal']:.4f}")

# If Dice loss ≈ 1.0: Model predicting all background!
# If Focal loss very high: Hard examples dominating
```

---

## 4. Data Issues

### FileNotFoundError

**Symptoms**:
```
FileNotFoundError: data/processed_multiclass/flair/BraTS20_001_0050.png
```

**Solutions**:

**1. Verify preprocessing**
```python
# Check if preprocessing completed
import os

data_dir = 'data/processed_multiclass'
for modality in ['flair', 't1', 't1ce', 't2', 'seg']:
    mod_dir = os.path.join(data_dir, modality)
    num_files = len(os.listdir(mod_dir))
    print(f"{modality}: {num_files} files")

# All should have same number (57,195)
```

**2. Check fold CSVs**
```python
# Verify fold split files exist
for fold in range(5):
    train_csv = f"{data_dir}/train_fold{fold}.csv"
    val_csv = f"{data_dir}/val_fold{fold}.csv"
    
    assert os.path.exists(train_csv), f"Missing {train_csv}"
    assert os.path.exists(val_csv), f"Missing {val_csv}"

# If missing: Run create_fold_splits.py
```

### Label Mismatch

**Symptoms**:
```
AssertionError: Invalid seg values: [0 1 2 4]  # Expected: [0 1 2]
```

**Solution**:
```python
# Preprocessing không convert đúng BraTS labels
# Re-run preprocess_h5_to_multiclass.py

# Verify conversion function
def convert_to_multiclass(seg):
    seg_mc = np.zeros_like(seg)
    seg_mc[seg == 0] = 0  # Background
    seg_mc[seg == 1] = 1  # TC (NCR/NET)
    seg_mc[seg == 2] = 2  # Edema
    seg_mc[seg == 4] = 1  # TC (ET)
    return seg_mc
```

### Class Imbalance

**Symptoms**:
```
Model predicts all background (class 0)
WT Dice = 0.0, TC Dice = 0.0, ED Dice = 0.0
```

**Solutions**:

**1. Class weights**
```python
# Compute từ data
class_counts = [87.35, 5.12, 7.53]  # percentages
class_weights = [1/c for c in class_counts]
class_weights = [w/sum(class_weights)*3 for w in class_weights]

# Use trong loss
loss_fn = MultiClassCombinedLoss(
    class_weights=class_weights
)
```

**2. Focal loss gamma**
```python
# Increase gamma để focus on hard examples
focal_loss = MultiClassFocalLoss(gamma=3.0)  # từ 2.0
```

---

## 5. Best Practices

### Project Organization

```
braintumnet/
├── configs/              # All configs
│   ├── base.yaml
│   ├── multiclass.yaml
│   └── a100.yaml
│
├── src/braintumnet/      # Source code
│   ├── models/
│   ├── data/
│   ├── engine/
│   └── losses/base.py
│
├── scripts/              # Standalone scripts
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── data/                 # Data (gitignored)
│   └── processed_multiclass/
│
├── checkpoints/          # Models (gitignored)
├── logs/                 # Logs (gitignored)
├── runs/                 # TensorBoard (gitignored)
│
├── tests/                # Unit tests
├── docs/                 # Documentation
├── requirements.txt
└── README.md
```

### Code Quality

**1. Docstrings**
```python
def compute_dice_score(pred, target, num_classes=3):
    """
    Compute Dice score per class.
    
    Args:
        pred (Tensor): Predictions (B, H, W) with class indices
        target (Tensor): Ground truth (B, H, W)
        num_classes (int): Number of classes (default: 3)
    
    Returns:
        dict: Dice scores per class
        
    Example:
        >>> pred = torch.randint(0, 3, (4, 256, 256))
        >>> target = torch.randint(0, 3, (4, 256, 256))
        >>> scores = compute_dice_score(pred, target)
        >>> print(scores)
        {'class_1': 0.85, 'class_2': 0.78}
    """
    pass
```

**2. Type hints**
```python
from typing import Tuple, Dict, Optional
import torch
from torch import Tensor

def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = 'cuda'
) -> Dict[str, float]:
    """Train one epoch"""
    pass
```

**3. Assertions**
```python
def predict(model, image):
    assert image.ndim == 4, f"Expected 4D tensor, got {image.ndim}D"
    assert image.shape[1] == 4, f"Expected 4 channels, got {image.shape[1]}"
    # ...
```

### Reproducibility

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call at start
set_seed(42)
```

### Logging

```python
import logging

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use
logger.info("Starting training...")
logger.warning("Learning rate very high!")
logger.error("CUDA out of memory!")
```

---

## 6. Common Mistakes

### ❌ Mistake 1: Data Leakage

```python
# ❌ Wrong: Slice-level split
all_slices = [...]  # 57,195 slices
train_slices = all_slices[:45000]
val_slices = all_slices[45000:]

# Problem: Same patient trong cả train và val!
```

```python
# ✓ Correct: Patient-level split
patients = [...]  # 369 patients
train_patients = patients[:295]
val_patients = patients[295:]

train_slices = get_slices_from_patients(train_patients)
val_slices = get_slices_from_patients(val_patients)
```

### ❌ Mistake 2: Wrong Metrics Computation

```python
# ❌ Wrong: Averaging Dice per batch
dice_scores = []
for batch in val_loader:
    pred, target = ...
    dice = compute_dice(pred, target)
    dice_scores.append(dice)

mean_dice = np.mean(dice_scores)  # BIASED!
```

```python
# ✓ Correct: Global accumulation
total_intersection = 0
total_union = 0

for batch in val_loader:
    pred, target = ...
    total_intersection += (pred & target).sum()
    total_union += (pred | target).sum()

dice = 2 * total_intersection / (total_union + total_intersection)
```

### ❌ Mistake 3: Not Normalizing Inputs

```python
# ❌ Wrong: Use raw pixel values
image = cv2.imread('flair.png')  # [0, 255]
output = model(image)  # Poor results!
```

```python
# ✓ Correct: Z-score normalization
image = cv2.imread('flair.png')
image = (image - image.mean()) / (image.std() + 1e-6)
output = model(image)
```

### ❌ Mistake 4: Ignoring Class Imbalance

```python
# ❌ Wrong: Uniform weights
loss_fn = nn.CrossEntropyLoss()  # Treats all classes equally
# → Model predicts all background!
```

```python
# ✓ Correct: Class weights
class_weights = [0.344, 5.865, 3.981]
loss_fn = MultiClassCombinedLoss(class_weights=class_weights)
```

### ❌ Mistake 5: eval() Mode

```python
# ❌ Wrong: Training trong eval mode
model.eval()
for epoch in range(num_epochs):
    for images, masks in train_loader:
        output = model(images)  # BatchNorm/Dropout disabled!
        loss.backward()
```

```python
# ✓ Correct
model.train()  # Training mode
for epoch in range(num_epochs):
    for images, masks in train_loader:
        output = model(images)
        loss.backward()

model.eval()  # Eval mode cho validation
with torch.no_grad():
    for images, masks in val_loader:
        output = model(images)
```

---

**[← Phần 7: Inference](v_07_INFERENCE.md)** | **[Phần 9: Results →](v_09_RESULTS.md)**
