# Phần 6: Configuration và Experiments

> **⚙️ Hệ Thống Cấu Hình - YAML Configs, Hyperparameters, Experiment Tracking**
>
> Tài liệu này giải thích chi tiết các file config và cách tuning hyperparameters.

---

## Mục Lục

1. [Config System Overview](#1-config-system-overview)
2. [MultiClass Config](#2-multiclass-config)
3. [Phase 2 Small Config](#3-phase-2-small-config)
4. [A100 Optimized Config](#4-a100-optimized-config)
5. [Hyperparameter Tuning](#5-hyperparameter-tuning)
6. [Experiment Management](#6-experiment-management)

---

## 1. Config System Overview

### File Structure

```
configs/
├── multiclass.yaml           # Standard 3-class config (RTX 3090)
├── phase2_small.yaml         # V2 small (base=48, dim=384)
└── phase2_a100.yaml          # V2 large (base=64, dim=512, A100)
```

### Config Loading

```python
import yaml

def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

# Usage
cfg = load_config('configs/multiclass.yaml')

# Access config values
batch_size = cfg['train']['batch_size']
learning_rate = cfg['optimizer']['lr']
num_epochs = cfg['train']['num_epochs']
```

---

## 2. MultiClass Config

### File: `configs/multiclass.yaml`

```yaml
# ============================================
# BrainTumNet Multi-Class Segmentation Config
# Target: RTX 3090 (24GB VRAM)
# ============================================

# Model Architecture
model:
  name: 'BrainTumNet'
  in_ch: 4                    # 4 MRI modalities
  num_cls: 2                  # HGG/LGG classification
  base: 32                    # Base channels (encoder start)
  dim: 256                    # Transformer embedding dim
  patch: 8                    # Transformer patch size
  depth: 2                    # Transformer depth (layers)
  n_heads: 4                  # Attention heads
  num_classes_seg: 3          # Segmentation classes (BG, TC, ED)
  roi_stop_grad: true         # Stop gradient qua ROI path
  deep_supervision: false     # Deep supervision (V1: false)

# Dataset
data:
  data_dir: 'data/processed_multiclass'
  img_size: 256
  num_folds: 5
  
  # Data splits
  train_split: 'train'
  val_split: 'val'

# Training
train:
  num_epochs: 250
  batch_size: 12              # RTX 3090: 12 samples
  num_workers: 4
  pin_memory: true
  gradient_accumulation_steps: 1
  
  # Mixed precision
  use_amp: true
  amp_dtype: 'float16'        # 'float16' or 'bfloat16'
  
  # Device
  device: 'cuda'

# Optimizer
optimizer:
  type: 'AdamW'
  lr: 1.0e-4                  # Learning rate
  weight_decay: 1.0e-5        # L2 regularization
  betas: [0.9, 0.999]
  eps: 1.0e-8

# Learning Rate Scheduler
scheduler:
  type: 'CosineAnnealingLR'
  T_max: 250                  # Max epochs
  eta_min: 1.0e-6             # Min LR
  warmup_epochs: 10

# Loss Functions
loss:
  # Segmentation loss
  seg_loss:
    type: 'MultiClassCombinedLoss'
    num_classes: 3
    dice_weight: 1.0
    focal_weight: 1.0
    class_weights: [0.344, 5.865, 3.981]  # Từ data statistics
    ignore_background: true
    dice_smooth: 1.0
    focal_gamma: 2.0
  
  # Classification loss
  cls_loss:
    type: 'CrossEntropyLoss'
    weight: null              # Uniform weights

# Augmentation
augmentation:
  train:
    horizontal_flip: 0.5
    vertical_flip: 0.5
    rotation: 15              # degrees
    scale: 0.1                # ±10%
    elastic_alpha: 50
    elastic_sigma: 5
    brightness: 0.2
    contrast: 0.2
    gaussian_noise: 0.3
  
  val:
    enabled: false            # No augmentation for validation

# Checkpointing
checkpoint:
  save_dir: 'checkpoints'
  save_freq: 10               # Save mỗi 10 epochs
  save_best: true             # Save best model
  metric: 'val_dice_mean'     # Metric để track best

# Logging
logging:
  log_dir: 'logs'
  tensorboard_dir: 'runs'
  log_freq: 10                # Log mỗi 10 batches
  
  # Metrics to log
  metrics:
    - 'train_loss'
    - 'val_loss'
    - 'val_dice_wt'
    - 'val_dice_tc'
    - 'val_dice_ed'
    - 'val_dice_mean'
    - 'val_cls_acc'

# Evaluation
evaluation:
  save_predictions: true
  prediction_dir: 'predictions'
  visualize: true
  num_vis_samples: 10
```

### Config Usage

```python
# Load config
cfg = load_config('configs/multiclass.yaml')

# Create model
model = BrainTumNet(
    in_ch=cfg['model']['in_ch'],
    num_cls=cfg['model']['num_cls'],
    base=cfg['model']['base'],
    dim=cfg['model']['dim'],
    patch=cfg['model']['patch'],
    depth=cfg['model']['depth'],
    n_heads=cfg['model']['n_heads'],
    num_classes_seg=cfg['model']['num_classes_seg'],
    roi_stop_grad=cfg['model']['roi_stop_grad'],
    deep_supervision=cfg['model']['deep_supervision']
)

# Create optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg['optimizer']['lr'],
    weight_decay=cfg['optimizer']['weight_decay'],
    betas=cfg['optimizer']['betas'],
    eps=cfg['optimizer']['eps']
)

# Create loss
seg_loss_fn = MultiClassCombinedLoss(
    num_classes=cfg['loss']['seg_loss']['num_classes'],
    dice_weight=cfg['loss']['seg_loss']['dice_weight'],
    focal_weight=cfg['loss']['seg_loss']['focal_weight'],
    class_weights=cfg['loss']['seg_loss']['class_weights'],
    ignore_background=cfg['loss']['seg_loss']['ignore_background']
)
```

---

## 3. Phase 2 Small Config

### File: `configs/phase2_small.yaml`

```yaml
# ============================================
# BrainTumNet V2 Small - Phase 2 Enhancements
# Target: RTX 3090 / RTX 4090 (24GB VRAM)
# ============================================

# Model Architecture (V2 Enhancements)
model:
  name: 'BrainTumNetV2'
  in_ch: 4
  num_cls: 2
  base: 48                    # ↑ Increased từ 32
  dim: 384                    # ↑ Increased từ 256
  patch: 8
  depth: 4                    # ↑ Increased từ 2
  n_heads: 8                  # ↑ Increased từ 4
  num_classes_seg: 3
  dropout: 0.15               # ✓ NEW: Regularization
  roi_stop_grad: true
  deep_supervision: true      # ✓ NEW: Deep supervision
  multi_scale_fusion: true    # ✓ NEW: Multi-scale fusion
  
  # V2 specific
  norm: 'instance'            # ✓ NEW: InstanceNorm instead of BatchNorm
  activation: 'leakyrelu'     # ✓ NEW: LeakyReLU instead of ReLU

# Dataset (same)
data:
  data_dir: 'data/processed_multiclass'
  img_size: 256
  num_folds: 5

# Training
train:
  num_epochs: 250
  batch_size: 12              # Same as V1
  num_workers: 4
  pin_memory: true
  gradient_accumulation_steps: 1
  use_amp: true
  amp_dtype: 'float16'
  device: 'cuda'

# Optimizer (slightly higher LR cho larger model)
optimizer:
  type: 'AdamW'
  lr: 1.5e-4                  # ↑ Increased từ 1e-4
  weight_decay: 1.0e-5
  betas: [0.9, 0.999]
  eps: 1.0e-8

# Scheduler (same)
scheduler:
  type: 'CosineAnnealingLR'
  T_max: 250
  eta_min: 1.0e-6
  warmup_epochs: 10

# Loss Functions (with deep supervision)
loss:
  seg_loss:
    type: 'MultiClassCombinedLoss'
    num_classes: 3
    dice_weight: 1.0
    focal_weight: 1.0
    class_weights: [0.344, 5.865, 3.981]
    ignore_background: true
    dice_smooth: 1.0
    focal_gamma: 2.0
  
  cls_loss:
    type: 'CrossEntropyLoss'
  
  # Deep supervision weights
  deep_supervision:
    enabled: true
    aux_weights: [0.2, 0.3, 0.5]  # [aux3, aux2, aux1]

# Augmentation (more aggressive)
augmentation:
  train:
    horizontal_flip: 0.5
    vertical_flip: 0.5
    rotation: 20              # ↑ Increased
    scale: 0.15               # ↑ Increased
    elastic_alpha: 80         # ↑ Increased
    elastic_sigma: 8
    brightness: 0.25
    contrast: 0.25
    gaussian_noise: 0.4
  
  val:
    enabled: false

# Checkpointing (same)
checkpoint:
  save_dir: 'checkpoints'
  save_freq: 10
  save_best: true
  metric: 'val_dice_mean'

# Logging (same)
logging:
  log_dir: 'logs'
  tensorboard_dir: 'runs'
  log_freq: 10
  metrics:
    - 'train_loss'
    - 'train_seg_loss'
    - 'train_cls_loss'
    - 'val_loss'
    - 'val_dice_wt'
    - 'val_dice_tc'
    - 'val_dice_ed'
    - 'val_dice_mean'
    - 'val_cls_acc'
```

---

## 4. A100 Optimized Config

### File: `configs/phase2_a100.yaml`

```yaml
# ============================================
# BrainTumNet V2 Large - A100 Optimized
# Target: A100 (40GB/80GB VRAM)
# ============================================

# Model Architecture (Larger capacity)
model:
  name: 'BrainTumNetV2'
  in_ch: 4
  num_cls: 2
  base: 64                    # ↑ Maximum capacity
  dim: 512                    # ↑ Maximum transformer dim
  patch: 8
  depth: 4
  n_heads: 8
  num_classes_seg: 3
  dropout: 0.15
  roi_stop_grad: true
  deep_supervision: true
  multi_scale_fusion: true
  norm: 'instance'
  activation: 'leakyrelu'

# Dataset
data:
  data_dir: 'data/processed_full_multimodal'
  img_size: 256
  num_folds: 5

# Training (A100 optimizations)
train:
  num_epochs: 250
  batch_size: 64              # ↑ Large batch for A100
  num_workers: 8              # ↑ More workers
  pin_memory: true
  gradient_accumulation_steps: 1
  
  # Mixed precision with BFloat16
  use_amp: true
  amp_dtype: 'bfloat16'       # ✓ BF16 for A100 (no scaling needed)
  
  # A100 specific optimizations
  cudnn_benchmark: true
  channels_last: true         # ✓ Memory format optimization
  
  device: 'cuda'

# Optimizer (higher LR cho large batch)
optimizer:
  type: 'AdamW'
  lr: 3.0e-4                  # ↑ Higher LR (batch 64 vs 12)
  weight_decay: 1.0e-5
  betas: [0.9, 0.999]
  eps: 1.0e-8

# Scheduler
scheduler:
  type: 'CosineAnnealingLR'
  T_max: 250
  eta_min: 1.0e-6
  warmup_epochs: 15           # ↑ Longer warmup for large batch

# Loss Functions
loss:
  seg_loss:
    type: 'MultiClassCombinedLoss'
    num_classes: 3
    dice_weight: 1.0
    focal_weight: 1.0
    class_weights: [0.344, 5.865, 3.981]
    ignore_background: true
    dice_smooth: 1.0
    focal_gamma: 2.0
  
  cls_loss:
    type: 'CrossEntropyLoss'
  
  deep_supervision:
    enabled: true
    aux_weights: [0.2, 0.3, 0.5]

# Augmentation
augmentation:
  train:
    horizontal_flip: 0.5
    vertical_flip: 0.5
    rotation: 20
    scale: 0.15
    elastic_alpha: 80
    elastic_sigma: 8
    brightness: 0.25
    contrast: 0.25
    gaussian_noise: 0.4
  
  val:
    enabled: false

# Checkpointing
checkpoint:
  save_dir: 'checkpoints'
  save_freq: 5                # ↓ Save more frequently
  save_best: true
  metric: 'val_dice_mean'

# Logging
logging:
  log_dir: 'logs'
  tensorboard_dir: 'runs'
  log_freq: 5                 # ↓ Log more frequently
  metrics:
    - 'train_loss'
    - 'train_seg_loss'
    - 'train_cls_loss'
    - 'val_loss'
    - 'val_dice_wt'
    - 'val_dice_tc'
    - 'val_dice_ed'
    - 'val_dice_mean'
    - 'val_cls_acc'

# Evaluation
evaluation:
  save_predictions: true
  prediction_dir: 'predictions'
  visualize: true
  num_vis_samples: 20         # ↑ More visualizations
  
  # Compute additional metrics
  compute_hd95: true          # Hausdorff Distance
  compute_sensitivity: true
  compute_specificity: true
```

### A100 Specific Optimizations

```python
# Channels Last Memory Format (A100 optimized)
model = model.to(memory_format=torch.channels_last)
images = images.to(memory_format=torch.channels_last)

# cuDNN Benchmark
torch.backends.cudnn.benchmark = True

# BFloat16 (no scaler needed)
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    outputs = model(images)
    loss = compute_loss(outputs, targets)

loss.backward()  # No scaling!
optimizer.step()
```

---

## 5. Hyperparameter Tuning

### Key Hyperparameters

**Model Architecture**:
```yaml
# Capacity scaling
base: [32, 48, 64]          # 32: V1, 48: V2 small, 64: V2 large
dim: [256, 384, 512]        # Transformer dimension
depth: [2, 4, 6]            # Transformer layers
n_heads: [4, 8, 12]         # Attention heads

# Trade-offs:
# Higher values → Better accuracy, more memory, slower
# Lower values → Faster, less memory, lower accuracy
```

**Training**:
```yaml
# Batch size
batch_size: [8, 12, 16, 32, 64]
# GPU dependent:
# RTX 3090 (24GB): 12-16
# A100 (40GB): 32-64
# V100 (16GB): 8-12

# Learning rate (scales với batch size)
lr: 1e-4 × (batch_size / 12)
# batch_size=12: lr=1e-4
# batch_size=64: lr=5.3e-4 ≈ 3e-4 (reduced slightly)
```

**Loss Weights**:
```yaml
# Dice vs Focal balance
dice_weight: [0.5, 1.0, 2.0]
focal_weight: [0.5, 1.0, 2.0]

# Recommendations:
# dice=1.0, focal=1.0: Balanced (default)
# dice=2.0, focal=1.0: Emphasize overlap
# dice=1.0, focal=2.0: Emphasize boundaries
```

**Class Weights**:
```yaml
# Compute từ data statistics
class_weights: [w_bg, w_tc, w_ed]

# Inverse frequency:
w_i = 1 / (freq_i + eps)

# BraTS 2020 (ví dụ):
# BG: 87.35% → w=0.344
# TC: 5.12% → w=5.865
# ED: 7.53% → w=3.981
```

### Tuning Strategy

**Step 1: Baseline**
```yaml
# Start với standard config
base: 32
dim: 256
depth: 2
n_heads: 4
lr: 1e-4
batch_size: 12
```

**Step 2: Scale capacity**
```yaml
# Increase model size
base: 48        # +50%
dim: 384        # +50%
depth: 4        # 2×
n_heads: 8      # 2×

# Expected: +2-3% Dice
```

**Step 3: Optimize training**
```yaml
# Enable deep supervision
deep_supervision: true
aux_weights: [0.2, 0.3, 0.5]

# Expected: +1-2% Dice
```

**Step 4: Fine-tune**
```yaml
# Adjust loss weights
dice_weight: 1.5
focal_weight: 1.0

# More aggressive augmentation
rotation: 20    # từ 15
scale: 0.15     # từ 0.1

# Expected: +0.5-1% Dice
```

---

## 6. Experiment Management

### Naming Convention

```
Format: {model}_{dataset}_{fold}_{date}_{time}

Examples:
- braintumnet_multiclass_fold0_20250128_103045
- braintumnet_v2_small_fold2_20250128_141523
- braintumnet_v2_a100_fold4_20250128_203018
```

### Directory Structure

```
experiments/
├── exp_001_baseline/
│   ├── config.yaml
│   ├── checkpoints/
│   │   ├── last_fold0.pth
│   │   └── best_fold0.pth
│   ├── logs/
│   │   └── metrics_fold0.csv
│   └── runs/
│       └── tensorboard_events
│
├── exp_002_v2_small/
│   ├── config.yaml
│   ├── checkpoints/
│   ├── logs/
│   └── runs/
│
└── exp_003_v2_large_a100/
    ├── config.yaml
    ├── checkpoints/
    ├── logs/
    └── runs/
```

### Experiment Tracking

```python
import json
from datetime import datetime

def create_experiment(name, config):
    """
    Create new experiment với tracking
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{name}_{timestamp}"
    exp_dir = f"experiments/{exp_name}"
    
    # Create directories
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{exp_dir}/logs", exist_ok=True)
    os.makedirs(f"{exp_dir}/runs", exist_ok=True)
    
    # Save config
    with open(f"{exp_dir}/config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Create metadata
    metadata = {
        'name': exp_name,
        'created_at': timestamp,
        'config': config,
        'status': 'running'
    }
    
    with open(f"{exp_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return exp_dir

# Usage
cfg = load_config('configs/phase2_small.yaml')
exp_dir = create_experiment('braintumnet_v2_small_fold0', cfg)
```

### Results Comparison

```python
import pandas as pd

def compare_experiments(exp_dirs):
    """
    Compare multiple experiments
    """
    results = []
    
    for exp_dir in exp_dirs:
        # Load metadata
        with open(f"{exp_dir}/metadata.json") as f:
            meta = json.load(f)
        
        # Load best metrics
        metrics_csv = f"{exp_dir}/logs/metrics_fold0.csv"
        df = pd.read_csv(metrics_csv)
        
        best_row = df.loc[df['mean_dice'].idxmax()]
        
        results.append({
            'experiment': meta['name'],
            'base': meta['config']['model']['base'],
            'dim': meta['config']['model']['dim'],
            'batch_size': meta['config']['train']['batch_size'],
            'best_epoch': best_row['epoch'],
            'wt_dice': best_row['wt_dice'],
            'tc_dice': best_row['tc_dice'],
            'ed_dice': best_row['ed_dice'],
            'mean_dice': best_row['mean_dice']
        })
    
    # Create comparison table
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('mean_dice', ascending=False)
    
    return comparison_df

# Usage
exp_dirs = [
    'experiments/exp_001_baseline',
    'experiments/exp_002_v2_small',
    'experiments/exp_003_v2_large'
]

comparison = compare_experiments(exp_dirs)
print(comparison)
```

**Output**:
```
experiment                      base  dim  batch_size  best_epoch  wt_dice  tc_dice  ed_dice  mean_dice
exp_003_v2_large                64    512  64          198         0.8956   0.8478   0.7834   0.8423
exp_002_v2_small                48    384  12          215         0.8823   0.8234   0.7612   0.8223
exp_001_baseline                32    256  12          237         0.8612   0.7845   0.7134   0.7864
```

---

**[← Phần 5: Training System](v_05_TRAINING_SYSTEM.md)** | **[Phần 7: Inference →](v_07_INFERENCE.md)**
