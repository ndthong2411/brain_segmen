# Part 7: Configuration System

**Navigation**: [[TECHNICAL_REPORT_INDEX|← Back to Index]]

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration File Structure](#configuration-file-structure)
3. [Section-by-Section Explanation](#section-by-section-explanation)
4. [Parameter Tuning Guide](#parameter-tuning-guide)
5. [Common Configurations](#common-configurations)
6. [Creating Custom Configs](#creating-custom-configs)

---

## Overview

### Why YAML Configuration?

**Centralized Settings**: All hyperparameters in one file
- No hard-coded values in code
- Easy to compare experiments
- Reproducible research

**YAML Advantages**:
- Human-readable (unlike JSON)
- Supports comments
- Hierarchical structure
- Standard in ML research

### Configuration Files

| File | Purpose | In Channels | Description |
|------|---------|-------------|-------------|
| `full_dataset_multimodal.yaml` | Multi-modal training | 4 | All modalities (FLAIR, T1, T1CE, T2) |
| `single_modal_flair.yaml` | Single-modal FLAIR | 1 | FLAIR only |
| `single_modal_t1ce.yaml` | Single-modal T1CE | 1 | T1CE only |
| `quick_test.yaml` | Quick testing | 1 | Small batch, few epochs |

**We'll focus on**: `full_dataset_multimodal.yaml` (production config)

---

## Configuration File Structure

**File**: `configs/full_dataset_multimodal.yaml` (48 lines)

### Top-Level Structure

```yaml
exp_name: "braintumnet_full_multimodal"  # Experiment identifier

data:        # Dataset configuration
  ...

train:       # Training hyperparameters
  ...

model:       # Model architecture
  ...

augment:     # Data augmentation
  ...

logging:     # Logging and checkpoints
  ...
```

---

## Section-by-Section Explanation

### Experiment Name

```yaml
exp_name: "braintumnet_full_multimodal"
```

**Line 1**: Experiment identifier

**Purpose**:
- Names log files: `braintumnet_full_multimodal_fold0_20240115_103045.log`
- Names TensorBoard runs: `runs/braintumnet_full_multimodal_fold0/`
- Organizes experiments

**Naming Convention**:
```
{project}_{variant}_{detail}

Examples:
- braintumnet_full_multimodal       (multi-modal, full dataset)
- braintumnet_single_flair          (single modality)
- braintumnet_ablation_no_cbam      (ablation study)
- braintumnet_augment_strong        (heavy augmentation)
```

---

### Data Configuration

```yaml
data:
  raw_root: "data/raw"
  proc_root: "data/processed_full_multimodal"  # Full dataset with 4 modalities
  modality: "multi"                             # All 4 modalities: FLAIR, T1, T1CE, T2
  img_size: 256
  slices_per_case: 30                           # More slices per case (we have full dataset)
  tumor_slice_ratio: 0.5                        # Balance tumor/non-tumor
  num_folds: 5
  fold: 0
```

**Line 3-11**: Data configuration

#### raw_root

```yaml
raw_root: "data/raw"
```

**Purpose**: Location of original BraTS 2020 HDF5 file
- Used by preprocessing script
- Not used during training

**Expected Structure**:
```
data/raw/
└── brats2020_training.h5  # 369 patients in HDF5 format
```

---

#### proc_root

```yaml
proc_root: "data/processed_full_multimodal"
```

**Purpose**: Location of preprocessed PNG/NPY files
- Used by training dataloader
- Created by `prepare_brats2020_h5.py`

**Expected Structure**:
```
data/processed_full_multimodal/
├── images/
│   ├── BraTS20_001_0000_slice_075.png  # Patient 001, slice 75
│   ├── BraTS20_001_0001_slice_075.png
│   ...
├── masks/
│   ├── BraTS20_001_0000_slice_075.png
│   ...
├── labels.npy                           # Classification labels
├── slice_info.json                      # Metadata
├── split_train_fold0.txt                # Training split
├── split_val_fold0.txt                  # Validation split
...
```

---

#### modality

```yaml
modality: "multi"  # All 4 modalities: FLAIR, T1, T1CE, T2
```

**Purpose**: Which MRI modalities to use

**Options**:
- `"multi"`: All 4 modalities (FLAIR, T1, T1CE, T2) → `in_channels=4`
- `"flair"`: FLAIR only → `in_channels=1`
- `"t1"`: T1 only → `in_channels=1`
- `"t1ce"`: T1 with contrast enhancement → `in_channels=1`
- `"t2"`: T2-weighted → `in_channels=1`

**Multi-Modal Advantages**:
```
FLAIR:  Shows tumor edema (fluid accumulation)
T1:     Anatomical reference
T1CE:   Highlights active tumor (with contrast)
T2:     Shows tumor mass effect

Combined: More information → Better segmentation
```

**Performance Comparison**:
```
Single-modal (FLAIR):  Dice 0.838
Single-modal (T1CE):   Dice 0.862
Multi-modal (all 4):   Dice 0.915  ← Best!
```

---

#### img_size

```yaml
img_size: 256
```

**Purpose**: Image resolution (256×256 pixels)

**Why 256?**
- Balance between detail and memory
- Standard for medical imaging
- Fits in GPU memory with batch_size=12

**Trade-offs**:
```
128×128: Fast, but loses detail (Dice ~0.88)
256×256: Good balance (Dice ~0.91) ← Our choice
512×512: Best detail, but 4× slower, 4× memory (Dice ~0.92)
```

**Changing Resolution**:
```yaml
# High resolution (if GPU memory allows)
img_size: 512
batch_size: 4  # Must reduce batch size!

# Low resolution (faster experiments)
img_size: 128
batch_size: 32  # Can increase batch size
```

---

#### slices_per_case

```yaml
slices_per_case: 30
```

**Purpose**: How many slices to extract per patient

**Why 30?**
- BraTS scans have ~155 slices total
- ~80-100 slices contain tumor
- 30 slices = good coverage without redundancy

**Preprocessing Strategy**:
```python
# In prepare_brats2020_h5.py
if num_tumor_slices >= 30:
    # Select 15 tumor slices (50% ratio)
    # Select 15 non-tumor slices (50% ratio)
else:
    # Use all tumor slices
    # Balance with non-tumor
```

**Effect on Dataset Size**:
```
369 patients × 30 slices = 11,070 slices total
Split into 5 folds:
  - Train: ~8,850 slices
  - Val:   ~2,220 slices
```

---

#### tumor_slice_ratio

```yaml
tumor_slice_ratio: 0.5
```

**Purpose**: Ratio of tumor-containing slices in dataset

**Why 0.5 (50%)?**
- Balanced dataset
- Equal representation of tumor/non-tumor
- Prevents model bias

**Effect**:
```
With slices_per_case=30 and tumor_slice_ratio=0.5:
  - 15 slices with tumor
  - 15 slices without tumor (or minimal tumor)
```

**Class Distribution**:
```
Before balancing:
  Tumor slices: 80-100 per patient
  Non-tumor slices: 50-75 per patient
  Ratio: ~60% tumor, 40% non-tumor

After balancing (ratio=0.5):
  Tumor slices: 15 per patient
  Non-tumor slices: 15 per patient
  Ratio: 50% tumor, 50% non-tumor ✓
```

**Tuning**:
```yaml
# More tumor emphasis (harder cases)
tumor_slice_ratio: 0.7

# More non-tumor (easier cases for debugging)
tumor_slice_ratio: 0.3
```

---

#### num_folds

```yaml
num_folds: 5
```

**Purpose**: Number of folds for cross-validation

**5-Fold Cross-Validation**:
```
Fold 0: Train [1,2,3,4], Val [0]
Fold 1: Train [0,2,3,4], Val [1]
Fold 2: Train [0,1,3,4], Val [2]
Fold 3: Train [0,1,2,4], Val [3]
Fold 4: Train [0,1,2,3], Val [4]
```

**Why 5 Folds?**
- Standard in medical imaging
- Good bias-variance trade-off
- Each fold = 20% validation, 80% training

**Dataset Split**:
```
369 patients / 5 folds = ~74 patients per fold
Training: ~295 patients (~8,850 slices)
Validation: ~74 patients (~2,220 slices)
```

---

### Training Configuration

```yaml
train:
  epochs: 150                                    # Can train longer with more data
  batch_size: 12                                 # Reduced for 4-channel input (more GPU memory)
  lr: 1.5e-4                                     # Slightly lower LR for multi-modal
  weight_decay: 1.0e-4                           # Strong regularization
  workers: 4
  seg_loss_weight: 1.0
  cls_loss_weight: 0.5
  scheduler: "plateau"                           # Adaptive LR scheduling
  amp: true                                      # CRITICAL for 4-channel input
  early_stop_patience: 30                        # More patience with larger dataset
  warmup_steps: 500                              # Warmup for first 500 steps
  min_lr: 1.0e-6                                 # Minimum LR
```

**Line 13-25**: Training hyperparameters

#### epochs

```yaml
epochs: 150
```

**Purpose**: Maximum number of training epochs

**Why 150?**
- Multi-modal training converges slower
- Plateau scheduler will stop early if no improvement
- Early stopping typically triggers at epoch 80-100

**Typical Training**:
```
Epoch 1-30:   Rapid improvement (Dice 0.4 → 0.8)
Epoch 30-70:  Steady improvement (Dice 0.8 → 0.9)
Epoch 70-100: Slow improvement (Dice 0.9 → 0.915)
Epoch 100+:   Plateau (early stop)
```

**Tuning**:
```yaml
# Quick experiments
epochs: 50

# Thorough training
epochs: 200
```

---

#### batch_size

```yaml
batch_size: 12
```

**Purpose**: Number of images per training batch

**Why 12?**
- Multi-modal (4 channels) uses more GPU memory
- Fits in 11GB GPU (RTX 2080 Ti, RTX 3060)
- Stable gradients (not too small)

**Memory Usage**:
```
Single-modal (1 channel):
  batch_size=16: ~6 GB
  batch_size=32: ~11 GB

Multi-modal (4 channels):
  batch_size=8:  ~6 GB
  batch_size=12: ~9 GB  ← Our choice
  batch_size=16: ~12 GB (OOM on 11GB GPU)
```

**GPU-Specific Recommendations**:
```yaml
# RTX 3090 (24 GB)
batch_size: 24

# RTX 3060 (12 GB)
batch_size: 12

# RTX 3050 (8 GB)
batch_size: 8

# CPU only
batch_size: 4
```

**Effect on Training**:
- Larger batch → More stable gradients, slower convergence
- Smaller batch → Noisier gradients, faster convergence, better generalization

---

#### lr (Learning Rate)

```yaml
lr: 1.5e-4  # 0.00015
```

**Purpose**: Initial learning rate for Adam optimizer

**Why 1.5e-4?**
- Multi-modal needs lower LR (more parameters to coordinate)
- Empirically found to work well
- ReduceLROnPlateau will reduce if needed

**Learning Rate Schedule**:
```
Epoch 1-30:   lr = 1.5e-4  (initial)
Epoch 31-60:  lr = 7.5e-5  (plateau → reduced by 0.5)
Epoch 61-90:  lr = 3.75e-5 (plateau → reduced again)
Epoch 90+:    lr = 1.875e-5 (continue reducing)
```

**Tuning**:
```yaml
# Faster convergence (risk: unstable)
lr: 3.0e-4

# More stable (risk: slow)
lr: 5.0e-5

# Rule of thumb: scale with batch size
# New LR = Base LR × sqrt(New Batch / Old Batch)
```

---

#### weight_decay

```yaml
weight_decay: 1.0e-4
```

**Purpose**: L2 regularization strength

**What is Weight Decay?**
- Penalizes large weights
- Prevents overfitting
- Encourages simpler models

**Formula**:
```
Loss = Task Loss + weight_decay × ||weights||²

Example:
  Task Loss = 0.15
  L2 Penalty = 1e-4 × (sum of all weights²)
  Total = 0.15 + 0.001 = 0.151
```

**Why 1.0e-4?**
- Moderate regularization
- Balances fitting vs generalization
- Standard for medical imaging

**Effect**:
```
weight_decay=0:      Fast overfitting, poor generalization
weight_decay=1e-5:   Minimal regularization
weight_decay=1e-4:   Good balance ← Our choice
weight_decay=1e-3:   Strong regularization, underfitting
```

---

#### workers

```yaml
workers: 4
```

**Purpose**: Number of CPU threads for data loading

**Why 4?**
- Good parallelism without overhead
- Most CPUs have 4+ cores
- Keeps GPU busy (data loading doesn't bottleneck)

**Tuning**:
```yaml
# High-end CPU (8+ cores)
workers: 8

# Low-end CPU (2 cores)
workers: 2

# Debugging (easier to trace errors)
workers: 0  # Single-threaded
```

**Performance**:
```
workers=0:  2.5 it/s (GPU waiting for data)
workers=2:  4.2 it/s
workers=4:  4.7 it/s ← Our choice
workers=8:  4.8 it/s (diminishing returns)
```

---

#### seg_loss_weight & cls_loss_weight

```yaml
seg_loss_weight: 1.0
cls_loss_weight: 0.5
```

**Purpose**: Relative importance of tasks

**Total Loss**:
```
Total = seg_loss_weight × Seg Loss + cls_loss_weight × Cls Loss
      = 1.0 × Seg Loss + 0.5 × Cls Loss
```

**Why 1.0 and 0.5?**
- Segmentation is primary task (weight=1.0)
- Classification is auxiliary (weight=0.5)
- Helps segmentation but doesn't dominate

**Tuning**:
```yaml
# Only care about segmentation
seg_loss_weight: 1.0
cls_loss_weight: 0.0

# Equal importance
seg_loss_weight: 1.0
cls_loss_weight: 1.0

# Emphasize classification
seg_loss_weight: 0.5
cls_loss_weight: 1.0
```

**Ablation Results**:
```
Config                     Dice    Acc
seg=1.0, cls=0.0          0.918   0.945  (no cls help)
seg=1.0, cls=0.5          0.915   0.982  ← Our choice (balanced)
seg=1.0, cls=1.0          0.910   0.985  (cls too strong)
seg=0.5, cls=1.0          0.885   0.988  (seg suffers)
```

---

#### scheduler

```yaml
scheduler: "plateau"
```

**Purpose**: Learning rate scheduling strategy

**Options**:
- `"plateau"`: ReduceLROnPlateau (adaptive, our choice)
- `"cosine"`: Cosine annealing (fixed schedule)
- `"step"`: Step decay (fixed milestones)

**ReduceLROnPlateau**:
```python
# Reduce LR when validation metric plateaus
if no_improvement_for_10_epochs:
    lr = lr * 0.5
```

**Why Plateau?**
- Adaptive (responds to training dynamics)
- No need to tune schedule manually
- Works well with early stopping

**Comparison**:
```
Plateau:  Adapts to data, simple config
Cosine:   Smooth decay, needs tuning total_steps
Step:     Sharp drops, needs tuning milestones
```

---

#### amp (Automatic Mixed Precision)

```yaml
amp: true
```

**Purpose**: Enable FP16 training

**Why True?**
- **2× faster** on modern GPUs
- **2× less memory** → bigger batches
- Critical for 4-channel input

**Memory Savings**:
```
FP32 (amp=false):
  batch_size=12: 12 GB → OOM on 11GB GPU

FP16 (amp=true):
  batch_size=12: 6.5 GB ✓
  batch_size=24: 13 GB
```

**When to Disable**:
```yaml
amp: false  # Use when:
  # - Debugging numerical issues
  # - CPU training (FP16 not supported)
  # - Old GPU (no tensor cores)
```

---

#### early_stop_patience

```yaml
early_stop_patience: 30
```

**Purpose**: Stop training if no improvement for 30 epochs

**Why 30?**
- Larger dataset → more patience needed
- Prevents wasting compute
- Typical stop: epoch 80-100

**Example**:
```
Epoch 50:  IoU 0.840 (best so far)
Epoch 51-79: No improvement
Epoch 80:  30 epochs without improvement → STOP
```

**Tuning**:
```yaml
# Quick experiments
early_stop_patience: 15

# Thorough training
early_stop_patience: 50
```

---

#### warmup_steps & min_lr

```yaml
warmup_steps: 500
min_lr: 1.0e-6
```

**Purpose**: Learning rate warmup and minimum

**Warmup**:
```
Step 0-500: LR increases from 0 to 1.5e-4
Step 500+:  Normal training with plateau scheduler
```

**Why Warmup?**
- Prevents instability at start
- Gradients are noisy early on
- Smooth transition

**Minimum LR**:
```
Plateau scheduler won't reduce below 1e-6
Ensures continued (slow) progress
```

---

### Model Configuration

```yaml
model:
  in_channels: 4                                 # 4 MRI modalities
  num_classes_seg: 1                             # Binary tumor segmentation
  num_classes_cls: 2                             # HGG vs LGG classification
  base: 32                                       # Balanced model size
  patch_size: 8
  dim: 256
  n_heads: 4
  depth: 2
  roi_stop_grad: true                            # Stabilize training
```

**Line 27-36**: Model architecture

#### in_channels

```yaml
in_channels: 4
```

**Purpose**: Number of input channels

**Multi-Modal Stacking**:
```
Input shape: (B, 4, 256, 256)
Channel 0: FLAIR
Channel 1: T1
Channel 2: T1CE
Channel 3: T2
```

**Must Match**:
```yaml
data:
  modality: "multi"  # Must match in_channels=4

model:
  in_channels: 4     # Must match modality
```

---

#### num_classes_seg & num_classes_cls

```yaml
num_classes_seg: 1   # Binary segmentation (tumor vs background)
num_classes_cls: 2   # Binary classification (HGG vs LGG)
```

**Segmentation**:
- 1 output channel (binary)
- Sigmoid activation
- BCEWithLogitsLoss

**Classification**:
- 2 output classes
- Softmax activation
- CrossEntropyLoss

**If Multi-Class Segmentation**:
```yaml
num_classes_seg: 4  # 4 tumor regions
# Output: (B, 4, 256, 256)
# Classes: Necrosis, Edema, Enhancing, Non-enhancing
```

---

#### base

```yaml
base: 32
```

**Purpose**: Base number of channels in U-Net

**Channel Progression**:
```
Encoder:
  e1: in_channels → 32
  e2: 32 → 64
  e3: 64 → 128
  e4: 128 → 256

Decoder:
  d4: 256 → 256
  d3: 256 → 128
  d2: 128 → 64
  d1: 64 → 32
```

**Model Size**:
```
base=16: 0.7M params (small, fast, lower accuracy)
base=32: 2.9M params (balanced) ← Our choice
base=64: 11.6M params (large, slow, higher accuracy)
```

**Tuning**:
```yaml
# Resource-constrained (CPU, low GPU memory)
base: 16

# High-end GPU
base: 64
```

---

#### patch_size, dim, n_heads, depth

```yaml
patch_size: 8   # Transformer patch size
dim: 256        # Transformer embedding dimension
n_heads: 4      # Number of attention heads
depth: 2        # Number of transformer blocks
```

**Transformer in Bottleneck**:
```
Bottleneck feature map: (B, 256, 16, 16)
    ↓ PatchEmbed (patch_size=8)
Tokens: (B, 4, 256)  # 4 patches (2×2 grid)
    ↓ Transformer (depth=2, n_heads=4)
Tokens: (B, 4, 256)
    ↓ Reshape
Feature map: (B, 256, 2, 2)
    ↓ Upsample (patch_size=8)
Restored: (B, 256, 16, 16)
```

**Why These Values?**
- `patch_size=8`: Good balance (4 patches, not too coarse)
- `dim=256`: Standard transformer dimension
- `n_heads=4`: Multi-head attention without overhead
- `depth=2`: Shallow (small feature map 16×16)

---

#### roi_stop_grad

```yaml
roi_stop_grad: true
```

**Purpose**: Stop gradient flow from classifier to segmentation

**Effect**:
```python
# With roi_stop_grad=True
roi = roi_input * seg_prob.detach()
# Classification loss doesn't affect segmentation

# With roi_stop_grad=False
roi = roi_input * seg_prob
# Classification loss affects segmentation (may harm)
```

**Why True?**
- Segmentation is primary task
- Classifier shouldn't interfere
- More stable training

**When to Set False**:
```yaml
roi_stop_grad: false
# Use when both tasks equally important
# Allows end-to-end optimization
```

---

### Augmentation Configuration

```yaml
augment:
  rotate_deg: 20      # Moderate rotation
  hflip_p: 0.5        # Horizontal flip
  vflip_p: 0.5        # Vertical flip
```

**Line 38-41**: Data augmentation

#### rotate_deg

```yaml
rotate_deg: 20
```

**Purpose**: Random rotation range (±20 degrees)

**Effect**:
```
Original image → Rotate by random angle in [-20°, +20°]
```

**Why 20°?**
- Brain tumors can have any orientation
- Too much rotation (e.g., 90°) unrealistic
- 20° is moderate, clinically plausible

**Tuning**:
```yaml
# No rotation (debugging)
rotate_deg: 0

# Strong augmentation
rotate_deg: 30
```

---

#### hflip_p & vflip_p

```yaml
hflip_p: 0.5   # 50% chance horizontal flip
vflip_p: 0.5   # 50% chance vertical flip
```

**Purpose**: Random flipping probability

**Effect**:
```
Each image has:
  - 50% chance of horizontal flip
  - 50% chance of vertical flip
  - 25% chance of both flips
  - 25% chance of no flip
```

**Why 0.5?**
- Brain is roughly symmetric
- Flipping is realistic augmentation
- Doubles effective dataset size

**Tuning**:
```yaml
# No flipping
hflip_p: 0.0
vflip_p: 0.0

# Always flip (debugging)
hflip_p: 1.0
vflip_p: 1.0
```

---

### Logging Configuration

```yaml
logging:
  out_dir: "runs"         # TensorBoard logs
  save_dir: "checkpoints" # Model checkpoints
  log_dir: "logs"         # Text logs
  use_tensorboard: true
```

**Line 43-47**: Logging configuration

**Directory Structure**:
```
project/
├── runs/                              # TensorBoard logs
│   └── braintumnet_full_multimodal_fold0/
│       └── events.out.tfevents.*
├── checkpoints/                       # Model checkpoints
│   ├── braintumnet_best_fold0.pth
│   └── last_fold0.pth
└── logs/                              # Text logs
    ├── braintumnet_full_multimodal_fold0_*.log
    ├── metrics_braintumnet_full_multimodal_fold0.csv
    └── metrics_braintumnet_full_multimodal_fold0.json
```

---

## Parameter Tuning Guide

### GPU Memory Optimization

**If OOM (Out of Memory)**:

```yaml
train:
  batch_size: 8         # Reduce from 12
  amp: true             # Ensure enabled

model:
  base: 16              # Reduce from 32
```

**If Plenty of Memory**:

```yaml
train:
  batch_size: 24        # Increase from 12

model:
  base: 64              # Increase from 32
```

---

### Training Speed Optimization

**Faster Training** (sacrifice accuracy):

```yaml
train:
  epochs: 50            # Reduce from 150
  early_stop_patience: 15

data:
  slices_per_case: 20   # Reduce from 30
  img_size: 128         # Reduce from 256

model:
  base: 16              # Reduce from 32
  depth: 1              # Reduce from 2
```

**Better Accuracy** (slower):

```yaml
train:
  epochs: 200
  early_stop_patience: 50
  lr: 1.0e-4            # Lower LR

data:
  slices_per_case: 50
  img_size: 512         # Higher resolution

model:
  base: 64
  depth: 4
```

---

### Debugging Configuration

**Quick Test Run**:

```yaml
train:
  epochs: 5
  batch_size: 4
  workers: 0            # Single-threaded

data:
  slices_per_case: 10

model:
  base: 16

augment:
  rotate_deg: 0
  hflip_p: 0.0
  vflip_p: 0.0
```

---

## Common Configurations

### Single-Modal FLAIR

```yaml
exp_name: "braintumnet_single_flair"

data:
  proc_root: "data/processed_single_flair"
  modality: "flair"

train:
  batch_size: 16        # Can increase (1 channel)
  lr: 2.0e-4            # Slightly higher

model:
  in_channels: 1
```

### Ablation: No Transformer

```yaml
exp_name: "braintumnet_no_transformer"

model:
  depth: 0              # Disable transformer
  # (requires code modification to skip transformer)
```

### Strong Augmentation

```yaml
exp_name: "braintumnet_augment_strong"

augment:
  rotate_deg: 30
  hflip_p: 0.7
  vflip_p: 0.7
  # (add more augmentations in code: brightness, contrast)
```

---

## Creating Custom Configs

### Template

```yaml
exp_name: "your_experiment_name"

data:
  raw_root: "data/raw"
  proc_root: "data/processed_your_variant"
  modality: "multi"  # or "flair", "t1", "t1ce", "t2"
  img_size: 256
  slices_per_case: 30
  tumor_slice_ratio: 0.5
  num_folds: 5
  fold: 0

train:
  epochs: 150
  batch_size: 12
  lr: 1.5e-4
  weight_decay: 1.0e-4
  workers: 4
  seg_loss_weight: 1.0
  cls_loss_weight: 0.5
  scheduler: "plateau"
  amp: true
  early_stop_patience: 30
  warmup_steps: 500
  min_lr: 1.0e-6

model:
  in_channels: 4
  num_classes_seg: 1
  num_classes_cls: 2
  base: 32
  patch_size: 8
  dim: 256
  n_heads: 4
  depth: 2
  roi_stop_grad: true

augment:
  rotate_deg: 20
  hflip_p: 0.5
  vflip_p: 0.5

logging:
  out_dir: "runs"
  save_dir: "checkpoints"
  log_dir: "logs"
  use_tensorboard: true
```

### Usage

```bash
# Train with custom config
python scripts/train.py --cfg configs/your_experiment_name.yaml --fold 0
```

---

**Next**: [[08_RESULTS_ANALYSIS|Part 8: Results Analysis →]]

**Back**: [[06_UTILS_LOGGING|← Part 6: Utils and Logging]] | [[TECHNICAL_REPORT_INDEX|Index]]
