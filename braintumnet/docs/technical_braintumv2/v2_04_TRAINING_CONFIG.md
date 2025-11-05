# Phase 2 Training Configuration

> **Hướng dẫn cấu hình training cho Phase 2 Small và Large**

---

## Configuration Files

- **phase2_small.yaml**: RTX 3090 24GB, 37M params
- **phase2_a100.yaml**: A100 80GB, 87M params
- **models/segunetv2_phase2.yaml**: Model-specific config

---

## phase2_small.yaml - RTX 3090 Recommended

### Model Configuration

```yaml
model:
  model_type: "v2"              # Use BrainTumNetV2

  # Input/Output
  in_channels: 4                # FLAIR, T1, T1CE, T2
  num_classes_seg: 3            # bg, TC, ED
  num_classes_cls: 2            # HGG, LGG

  # Architecture - Phase 2 Small
  base: 48                      # +50% from V1 (32→48)
  dim: 384                      # +50% from V1 (256→384)
  patch_size: 8
  depth: 4                      # 2x from V1
  n_heads: 8                    # 2x from V1

  # Phase 2 features
  dropout: 0.15                 # Regularization
  norm: "instance"              # Medical imaging standard
  roi_stop_grad: true
  deep_supervision: true        # Auxiliary losses
  multi_scale_fusion: true      # Multi-scale features
```

**Parameters**: 37M (2.6x V1)

### Training Configuration

```yaml
train:
  epochs: 350                   # Longer for larger model
  batch_size: 8                 # Reduced from 16 (larger model)
  lr: 3.0e-5                    # Lower for stability
  weight_decay: 1.0e-4
  workers: 4

  # Loss configuration
  loss_type: "ultimate_multitask"
  seg_loss_weight: 1.0
  cls_loss_weight: 0.5

  # Loss components
  dice_weight: 1.0
  focal_weight: 1.0
  iou_weight: 2.0               # Emphasize IoU
  boundary_weight: 0.5

  # Class-specific weights
  focal_alpha: [0.0, 0.4, 0.1]  # [bg, TC, ED]
  class_weights: [1.0, 3.0, 2.0]
  ignore_background: true

  # Deep supervision
  aux_weight: 0.3               # Auxiliary loss weight

  # Optimizer
  optimizer: "adamw"
  grad_clip_norm: 1.0

  # Scheduler
  scheduler: "cosine"
  warmup_steps: 1000            # Longer warmup
  min_lr: 1.0e-6

  # Mixed precision
  amp: true
  grad_accum_steps: 2           # Effective batch=16

  # Early stopping
  early_stop_patience: 80
  val_interval: 1
```

### Data Configuration

```yaml
data:
  proc_root: "braintumnet/data/processed_multiclass"
  modality: "multi"             # 4 modalities
  img_size: 256
  slices_per_case: 30
  tumor_slice_ratio: 0.7        # Focus on tumor slices
  num_folds: 5
  fold: 0
```

### Augmentation

```yaml
augment:
  # Geometric
  rotate_deg: 30
  hflip_p: 0.5
  vflip_p: 0.5

  # Intensity
  brightness_range: [0.8, 1.2]
  contrast_range: [0.8, 1.2]
```

### Expected Results

```
Training time: ~48 hours (350 epochs)
GPU memory: ~12-16GB
Target IoU: 0.80-0.82
Multi-class:
  - WT: 0.83-0.86
  - TC: 0.80-0.83
  - ED: 0.82-0.85
```

---

## phase2_a100.yaml - A100 Large Model

### Model Configuration

```yaml
model:
  model_type: "v2"

  # Architecture - Phase 2 Large
  base: 64                      # 2x from V1
  dim: 512                      # 2x from V1
  patch_size: 8
  depth: 4
  n_heads: 8
  dropout: 0.2                  # Higher for larger model

  # Same features as small
  norm: "instance"
  roi_stop_grad: true
  deep_supervision: true
  multi_scale_fusion: true
```

**Parameters**: 87M (6.2x V1)

### Training Configuration

```yaml
train:
  epochs: 400
  batch_size: 16                # A100 can handle larger
  lr: 5.0e-5                    # Scaled for batch size
  weight_decay: 1.5e-4
  workers: 16                   # More workers

  # Loss (FIXED for multi-class)
  focal_alpha: [0.0, 0.4, 0.3]  # ED increased: 0.15→0.3
  class_weights: [1.0, 3.0, 4.0] # ED increased: 2.5→4.0

  # Optimizer
  optimizer: "adamw"
  optimizer_fused: true         # A100 optimization
  grad_clip_norm: 1.0

  # Scheduler
  scheduler: "cosine"
  warmup_steps: 2000            # Longer warmup
  min_lr: 5.0e-7

  # Mixed precision - A100 BFloat16
  amp: true
  amp_dtype: "bfloat16"         # Native on A100

  # A100 optimizations
  channels_last: true           # Memory format
  cudnn_benchmark: true
  pin_memory: true
  prefetch_factor: 8
  persistent_workers: true
```

### Expected Results

```
Training time: ~200 hours (400 epochs, ~8 days)
GPU memory: ~28GB
Target IoU: 0.85-0.90
Multi-class:
  - WT: 0.85-0.88
  - TC: 0.83-0.86
  - ED: 0.84-0.87
```

---

## Key Configuration Differences

### V1 vs Phase 2 Small

| Parameter | V1 | Phase 2 Small | Change |
|-----------|----|--------------| -------|
| base | 32 | 48 | +50% |
| dim | 256 | 384 | +50% |
| depth | 2 | 4 | 2x |
| n_heads | 4 | 8 | 2x |
| batch_size | 16 | 8 | Halved (larger model) |
| lr | 5e-5 | 3e-5 | Lower (stability) |
| warmup_steps | 500 | 1000 | Longer |
| dropout | 0.0 | 0.15 | Added |
| multi_scale_fusion | false | true | Added |
| deep_supervision | false | true | Added |

### Phase 2 Small vs Large

| Parameter | Small | Large | Reason |
|-----------|-------|-------|--------|
| base | 48 | 64 | More capacity |
| dim | 384 | 512 | Larger transformer |
| batch_size | 8 | 16 | A100 can handle |
| lr | 3e-5 | 5e-5 | Scaled with batch |
| dropout | 0.15 | 0.2 | More regularization |
| workers | 4 | 16 | A100 throughput |
| warmup | 1000 | 2000 | Longer for larger |
| amp_dtype | float16 | bfloat16 | A100 native |

---

## Loss Configuration Details

### Ultimate Multi-Task Loss

```python
Total Loss = seg_loss_weight * SegLoss + cls_loss_weight * ClsLoss

SegLoss = dice_weight * DiceLoss
        + focal_weight * FocalLoss
        + iou_weight * IoULoss
        + boundary_weight * BoundaryLoss
        + aux_weight * AuxLosses  # If deep_supervision
```

### Multi-Class Weights

**Phase 2 Small** (balanced):
```yaml
focal_alpha: [0.0, 0.4, 0.1]    # TC emphasized more
class_weights: [1.0, 3.0, 2.0]  # TC harder than ED
```
- Background: Ignored (focal_alpha=0, always present)
- TC (Tumor Core): High weight (hardest class)
- ED (Edema): Medium weight

**Phase 2 Large** (ED fixed):
```yaml
focal_alpha: [0.0, 0.4, 0.3]    # ED increased
class_weights: [1.0, 3.0, 4.0]  # ED highest weight
```
- ED weight increased (0.1→0.3, 2.0→4.0)
- Fixes ED class learning (was 0.009 Dice ❌)
- Expected ED: 0.82-0.85 ✅

### Why Different Weights?

```
V1 Results:
- WT: 0.04  ❌ (sum of TC+ED, both failed)
- TC: 0.81  ✓  (only TC learned)
- ED: 0.009 ❌ (ED not learned)

Problem: ED class too weak
→ Model ignores ED, only learns TC
→ Need higher ED weight

Phase 2 Fix:
- Increase ED focal_alpha: 0.1 → 0.3
- Increase ED class_weight: 2.0 → 4.0
→ Force model to learn ED
→ Expected ED: 0.82-0.85 ✅
```

---

## Optimizer and Scheduler

### AdamW Optimizer

```yaml
optimizer: "adamw"
lr: 3.0e-5 (small) or 5.0e-5 (large)
weight_decay: 1.0e-4 (small) or 1.5e-4 (large)
betas: [0.9, 0.999]  # Default
eps: 1.0e-8          # Default
```

**Why AdamW?**
- Better generalization than Adam
- Decoupled weight decay
- Standard for transformers
- Works well with large models

**A100 Fused Optimizer**:
```yaml
optimizer_fused: true  # A100 only
```
- ~15% faster on A100
- Not available on RTX 3090

### Cosine Scheduler with Warmup

```python
# Warmup phase (linear increase)
lr = base_lr * (step / warmup_steps)  # 0 → base_lr

# Cosine decay phase
lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
```

**Warmup**:
- Small: 1000 steps (~5 epochs)
- Large: 2000 steps (~5 epochs)
- Prevents early instability
- Larger models need longer warmup

**Cosine decay**:
- Smooth decay to min_lr
- No sudden drops
- Good for convergence

---

## Mixed Precision Training

### RTX 3090 (Float16)

```yaml
amp: true
amp_dtype: "float16"  # Default
```

**Benefits**:
- 2x faster training
- ~30% less memory
- Minimal accuracy loss

**Considerations**:
- Use gradient scaling
- Some ops fallback to FP32
- Loss scaling may need tuning

### A100 (BFloat16)

```yaml
amp: true
amp_dtype: "bfloat16"  # A100 native
```

**Benefits over Float16**:
- No gradient scaling needed
- Larger dynamic range
- More stable training
- Native A100 support

**Trade-offs**:
- Slightly less precision than FP16
- But better stability
- Overall better for large models

---

## Gradient Accumulation

**Phase 2 Small**:
```yaml
batch_size: 8
grad_accum_steps: 2
# Effective batch size = 8 * 2 = 16
```

**Why accumulate?**
- Model too large for batch=16
- Accumulate gradients over 2 steps
- Same effect as batch=16 but less memory

**How it works**:
```python
for step in range(grad_accum_steps):
    loss = model(batch) / grad_accum_steps
    loss.backward()  # Accumulate gradients

optimizer.step()     # Update after accumulation
optimizer.zero_grad()
```

---

## Data Loading Optimization

### RTX 3090

```yaml
workers: 4                # DataLoader workers
pin_memory: true          # Faster GPU transfer
prefetch_factor: 2        # Default
persistent_workers: false # Save memory
```

### A100

```yaml
workers: 16               # More workers for throughput
pin_memory: true
prefetch_factor: 8        # Aggressive prefetch
persistent_workers: true  # Keep workers alive
```

**Why more workers on A100?**
- A100 processes batches faster
- Need more data throughput
- CPU becomes bottleneck otherwise

---

## Monitoring and Checkpointing

### Logging

```yaml
logging:
  log_dir: "logs"
  out_dir: "runs"
  save_dir: "checkpoints"
  exp_name: "braintumnet_phase2_small"
  use_tensorboard: true
  log_every_n_steps: 50
  save_top_k: 3            # Keep best 3 checkpoints
```

### Validation

```yaml
train:
  val_interval: 1          # Validate every epoch
  early_stop_patience: 80  # Stop if no improvement
  save_interval: 10        # Save checkpoint every 10 epochs
```

---

## Quick Start Commands

### Phase 2 Small (RTX 3090)

```bash
# Train fold 0
python scripts/train.py --cfg configs/phase2_small.yaml --fold 0

# Resume from checkpoint
python scripts/train.py --cfg configs/phase2_small.yaml --fold 0 \
  --resume checkpoints/braintumnet_phase2_small_fold0_best.pth

# Train all folds
python scripts/train_all_folds.py --cfg configs/phase2_small.yaml
```

### Phase 2 Large (A100)

```bash
# Train fold 0
python scripts/train.py --cfg configs/phase2_a100.yaml --fold 0

# With optional features
python scripts/train.py --cfg configs/phase2_a100.yaml --fold 0 \
  --use-multiscale-transformer --use-attention-gates
```

---

## Troubleshooting

### OOM (Out of Memory)

**Solutions**:
1. Reduce batch_size: 8 → 4
2. Increase grad_accum_steps: 2 → 4
3. Disable optional features
4. Reduce model size: base 48 → 40

### Slow Training

**Solutions**:
1. Increase workers: 4 → 8
2. Use channels_last memory format
3. Enable cudnn_benchmark
4. Check GPU utilization (nvidia-smi)

### NaN Loss

**Solutions**:
1. Lower learning rate: 3e-5 → 1e-5
2. Increase warmup: 1000 → 2000
3. Check data normalization
4. Use gradient clipping (already enabled)

### Poor Multi-Class Results

**Check**:
1. Class weights correct?
2. focal_alpha for ED high enough?
3. Using deep_supervision?
4. Enough training epochs?

---

## Summary

### Phase 2 Small (Recommended)

```yaml
Model: 37M params, base=48, dim=384
Hardware: RTX 3090 24GB
Batch: 8 (effective 16 with grad_accum)
Time: ~48 hours
Target: IoU 0.80-0.82
```

### Phase 2 Large (Best)

```yaml
Model: 87M params, base=64, dim=512
Hardware: A100 80GB
Batch: 16
Time: ~200 hours
Target: IoU 0.85-0.90
```

### Key Differences from V1

- 2.6-6.2x more parameters
- InstanceNorm + LeakyReLU
- Multi-scale fusion + Deep supervision
- Lower LR, longer warmup
- Gradient accumulation
- BFloat16 on A100

---

**Next**: [Upgrade Reasoning →](v2_06_UPGRADE_REASONING.md)

**Back**: [← Phase 2 Features](v2_03_PHASE2_FEATURES.md)
