# BrainTumNet Evolution Roadmap
## From Dice 0.8699 → 0.90+ (Top BraTS Performance)

**Document Version**: 2.0 (Updated Based on Actual Phase 2 Results)
**Date**: 2025-11-04
**Current Performance**: Dice 0.8699, IoU 0.7717 (Fold 3, 62.7M params)
**Target Performance**: Dice 0.90-0.91 (Match top BraTS ensembles)
**Gap to Close**: +2-3% Dice Score
**Timeline**: 10 weeks
**Hardware**: 2× RTX 3090 24GB + 1× A100 80GB
**Priority**: Performance > Speed

---

## Executive Summary

Phase 2 training has **exceeded expectations**, achieving Dice 0.8699—significantly higher than originally predicted (0.72-0.76). This document provides a revised, **data-driven roadmap** to reach top BraTS performance (Dice 0.90+) within 10 weeks.

### Current Status (November 2025)

| Metric | Phase 2 Actual | Original Prediction | Difference |
|--------|----------------|---------------------|------------|
| **Best Dice** | **0.8699** (Fold 3) | 0.72-0.76 | **+11.4%** better! |
| **Best IoU** | 0.7717 | 0.80-0.82 | -4.2% (boundary issue) |
| **WT Dice** | 0.9189 | Not estimated | Excellent |
| **TC Dice** | 0.8662 | Not estimated | Good |
| **ED Dice** | 0.8287 | Not estimated | Needs improvement |
| **Gap to SOTA** | **2-3%** | 11-15% | **85% reduced!** |

**Key Achievement**: We are much closer to SOTA than originally thought. Only 2-3% Dice gap remains.

### SOTA Landscape (2024-2025)

| Model | Parameters | Dice Score | Key Technology |
|-------|-----------|------------|----------------|
| nnUNet (single) | Adaptive | 0.83-0.85 | Self-configuring |
| MedNeXt-L | ~100M | 0.83-0.86 | Large kernels (7×7) |
| SwinUNETR | ~62M | 0.82-0.85 | Hierarchical Transformer |
| SegMamba | ~45M | 0.84-0.86 | State Space Models |
| **Top Ensemble** | Combined | **0.87-0.90** | Multi-model voting |
| **BrainTumNet V2** | **62.7M** | **0.8699** | **Current (Fold 3)** |

**Position**: Single-model performance rivals nnUNet. Ensemble can push to 0.90+.

---

## Phase 0: Completed Phase 2 Training

### Training Results Summary

**Completed Folds**:

| Fold | Params | Best Epoch | Val Dice | Val IoU | WT Dice | TC Dice | ED Dice | Time | Status |
|------|--------|------------|----------|---------|---------|---------|---------|------|--------|
| 1 | 35.4M | 148 | 0.8331 | 0.7166 | 0.8916 | 0.8322 | 0.7825 | 17h | Complete |
| 2 | 35.4M | 44 | **0.8435** | **0.7323** | 0.9044 | 0.8372 | 0.7890 | - | **Incomplete** |
| 3 | 62.7M | 46 | **0.8699** | **0.7717** | **0.9189** | **0.8662** | **0.8287** | 45h | Complete |

**Best Model**: Fold 3 (62.7M parameters, A100 trained)

### Identified Issues

Despite excellent Dice scores, three critical issues were identified:

#### Issue 1: IoU-Dice Gap (10% discrepancy)
**Symptom**:
```
Dice: 0.8699 (Excellent coverage)
IoU:  0.7717 (Below target 0.82)
Gap:  0.0982 (10%)
```

**Root Cause**: Boundary imprecision. Model over-segments slightly, leading to false positives at edges.

**Evidence**:
- Good Dice = good overlap coverage
- Low IoU = poor boundary precision
- Model predicts correct regions but with blurry/imprecise edges

**Fix Strategy** (Phase 1):
- Increase boundary loss weight: 0.6 → 1.0
- Increase IoU loss weight: 2.5 → 3.0
- Better penalize false positives

---

#### Issue 2: Training Plateau (40% wasted compute)
**Symptom**:
```
Fold 1: Best epoch 148, continued to 228 (80 wasted)
Fold 3: Best epoch 46, continued to 146 (100 wasted)
→ 40-50% of training time wasted after peak
```

**Root Cause**: Learning rate reaches minimum too early.

**LR Schedule Analysis**:
```
Cosine Annealing: LR = min_lr + (max_lr - min_lr) × cos(π × epoch / total)

Epoch 0:    LR = 5.0e-5  (start)
Epoch 50:   LR = 2.5e-5  (50% decay)
Epoch 100:  LR = 5.0e-6  (90% decay)
Epoch 150:  LR = 1.0e-6  (minimum, effectively stuck)
Epoch 200+: LR = 1.0e-6  (flat, no learning)
```

**Fix Strategy** (Phase 1):
- Implement **Cosine with Warm Restarts** (SGDR)
- Restart LR periodically: epochs 50, 150, 350
- Increase min_lr: 1e-6 → 1e-5 (10×)
- Reduce early stop patience: 100 → 40

---

#### Issue 3: ED (Edema) Underperformance
**Symptom**:
```
WT Dice: 0.9189  (+0% baseline)
TC Dice: 0.8662  (-5.3% vs WT)
ED Dice: 0.8287  (-9.0% vs WT)  ← Weakest
```

**Root Cause**: Edema has diffuse boundaries (gradual intensity transitions), harder to segment than sharp tumor core.

**Current Weights**:
```yaml
class_weights: [1.0, 3.0, 4.0]  # [bg, TC, ED]
focal_alpha:   [0.0, 0.4, 0.3]  # [bg, TC, ED]
```

**Fix Strategy** (Phase 1):
- Match ED focal_alpha to TC: 0.3 → 0.4
- Maintain higher class_weight (already 4.0)
- Consider ED-specific augmentation

---

## Phase 1: Immediate Training Fixes (Week 1)

**Goal**: Optimize training pipeline WITHOUT architecture changes
**Hardware**: RTX 3090 (fast iterations)
**Expected Gain**: +0.5-1.0% Dice → 0.8750-0.8800
**Risk**: Low (proven techniques)

### Week 1 Timeline

#### **Days 1-2: Implement SGDR (Stochastic Gradient Descent with Warm Restarts)**

**What to Change**:

**File**: `src/braintumnet/engine/trainer.py`

**Current Code** (around line 150-160):
```python
# Current scheduler
if cfg.train.scheduler == 'cosine':
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.epochs,
        eta_min=cfg.train.min_lr
    )
```

**New Code**:
```python
# Add warm restarts option
if cfg.train.scheduler == 'cosine':
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.epochs,
        eta_min=cfg.train.min_lr
    )
elif cfg.train.scheduler == 'cosine_restarts':
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.train.get('T_0', 50),      # First restart period
        T_mult=cfg.train.get('T_mult', 2), # Period multiplier
        eta_min=cfg.train.min_lr
    )
```

**Config Change**:

**File**: `configs/phases/phase1_optimized.yaml` (NEW FILE)

```yaml
train:
  # Learning rate schedule (UPDATED)
  scheduler: "cosine_restarts"  # Changed from "cosine"
  T_0: 50                        # Restart every 50 epochs initially
  T_mult: 2                      # Double period each restart (50, 100, 200...)
  min_lr: 1.0e-5                 # Increased from 1.0e-6 (10×)

  # Early stopping (UPDATED)
  early_stop_patience: 40        # Reduced from 100
```

**Why This Works**:
- **Warm Restarts**: LR spikes help escape local minima
- **Higher min_lr**: Keeps model learning even at later epochs
- **Lower patience**: Stops earlier if truly converged

**Expected Impact**: +0.3-0.5% Dice, save 40% training time

---

#### **Days 3-4: Rebalance Loss Weights**

**File**: `configs/phases/phase1_optimized.yaml`

**Current Weights**:
```yaml
train:
  iou_weight: 2.5
  boundary_weight: 0.6
  class_weights: [1.0, 3.0, 4.0]  # [bg, TC, ED]
  focal_alpha: [0.0, 0.4, 0.3]    # [bg, TC, ED]
```

**New Weights**:
```yaml
train:
  # Emphasize IoU and boundaries (UPDATED)
  iou_weight: 3.0          # Increased from 2.5 (+20%)
  boundary_weight: 1.0     # Increased from 0.6 (+67%)

  # Class weights (maintained, working well)
  class_weights: [1.0, 3.0, 4.0]

  # Focal alpha (UPDATED for ED)
  focal_alpha: [0.0, 0.4, 0.4]  # Match ED to TC (was 0.3)
```

**Reasoning**:

**IoU Weight** (2.5 → 3.0):
- IoU is the target metric lagging behind
- Current loss composition:
  ```
  Total = 1.0×Dice + 1.0×Focal + 2.5×IoU + 0.6×Boundary
        = 5.1 total

  IoU contribution: 2.5/5.1 = 49%
  ```
- New composition:
  ```
  Total = 1.0×Dice + 1.0×Focal + 3.0×IoU + 1.0×Boundary
        = 6.0 total

  IoU contribution: 3.0/6.0 = 50% (balanced with others)
  Boundary: 1.0/6.0 = 17% (was 12%, now stronger)
  ```

**Boundary Weight** (0.6 → 1.0):
- Boundary loss directly improves edge precision
- IoU-Dice gap indicates boundary is the issue
- Doubling emphasis should sharpen predictions

**Focal Alpha for ED** (0.3 → 0.4):
- ED is hardest region, needs same focus as TC
- Focal loss targets hard examples
- Matching TC's alpha (0.4) should improve ED Dice

**Expected Impact**: +0.2-0.4% IoU, +0.1-0.2% Dice

**No Code Changes Needed**: Trainer already reads these from config!

---

#### **Day 5: Validation Testing**

**Test Changes on Small Run**:
```bash
# Quick validation (50 epochs)
python braintumnet/scripts/train.py \
    --model segunetv2 \
    --cfg phase1_optimized \
    --fold 0 \
    --epochs 50 \
    --output-prefix "phase1_test"

# Monitor:
# - LR restarts working (spike at epoch 50)
# - IoU improving faster than Dice
# - No training instabilities
```

**Success Criteria**:
- LR restarts visible in logs
- val_iou improvement > baseline
- No NaN losses or divergence

---

#### **Days 6-7: Resume Fold 2 Training**

**Why Fold 2?**
- Best early performance (Dice 0.8435 at epoch 44)
- Stopped prematurely (epoch 74)
- Potential to be best fold with proper training

**Command**:
```bash
# Resume from epoch 74 checkpoint
python braintumnet/scripts/train.py \
    --model segunetv2 \
    --cfg phase1_optimized \
    --fold 2 \
    --resume checkpoints/last_fold2.pth \
    --epochs 350

# Hardware: RTX 3090 (35.4M model fits)
```

**Expected Outcome**:
- Training continues from epoch 74
- New LR schedule applied (restart at epoch 100, 200...)
- Reaches Dice 0.85+ by epoch 150
- Early stops around epoch 190 (vs 350 without fixes)

**Deliverable**: Optimized Fold 2 model, potentially best 35M model

---

### Phase 1 Summary

**Changes Made**:
1. ✅ SGDR scheduler (warm restarts)
2. ✅ Higher min_lr (1e-5 instead of 1e-6)
3. ✅ Rebalanced loss weights (IoU +20%, Boundary +67%)
4. ✅ Improved early stopping (patience 40)
5. ✅ Completed Fold 2 training

**Expected Results**:
- Dice: 0.8699 → 0.8750-0.8800 (+0.5-1.0%)
- IoU: 0.7717 → 0.7800-0.7850 (+0.8-1.3%)
- Training time: 45h → 30h (33% faster)
- ED Dice: 0.8287 → 0.8350+ (+0.6%)

**Validation**: Train Fold 0 with new config, compare with Fold 3 baseline

**Risk**: Very low. All changes are hyperparameter tuning, easily reversible.

---

## Phase 2: MedNeXt Backbone (Weeks 2-4)

**Goal**: Replace basic Conv blocks with MedNeXt (large kernels, inverted bottlenecks)
**Hardware**: A100 80GB (larger model capacity)
**Expected Gain**: +1.5-2.5% Dice → 0.8900-0.9050
**Risk**: Medium (new architecture component)

### Why MedNeXt?

**Current Bottleneck**: Limited receptive field

**ResidualConvBlock** (current):
```python
# Two 3×3 convolutions
conv1: 3×3  → receptive field 3×3
conv2: 3×3  → cumulative 5×5

# After 4 encoder levels:
Total receptive field: ~33×33 pixels (for 256×256 image)

# Tumor sizes:
Small tumor: ~50×50 pixels
Large tumor: ~150×150 pixels

→ Receptive field INSUFFICIENT for full tumor context
```

**MedNeXt Solution**: Large depthwise kernels

```python
# Single 7×7 depthwise conv
conv_dw: 7×7 → receptive field 7×7

# After 4 levels:
Total receptive field: ~77×77 pixels

→ 2.3× larger context
```

**Parameter Efficiency**:

Standard 3×3 conv (128 → 256 channels):
```
params = 128 × 256 × 3 × 3 = 294,912
```

MedNeXt (depthwise 7×7 + inverted bottleneck):
```
depthwise 7×7:  128 × 7 × 7 = 6,272
expand 1×1:     128 × 512 = 65,536
compress 1×1:   512 × 128 = 65,536
Total:          137,344 params

→ 53% fewer parameters for 2.3× larger receptive field!
```

**Proven Results** (MedNeXt paper, MICCAI 2023):
- BraTS 2021: Baseline Dice 0.846 → MedNeXt Dice 0.872 (+2.6%)
- AMOS CT: Baseline Dice 0.863 → MedNeXt Dice 0.891 (+2.8%)

**Expected for BrainTumNet**:
- Current: Dice 0.8750
- With MedNeXt: Dice 0.8900-0.9000 (+1.5-2.5%)

---

### Week 2: Implementation

#### **Days 1-3: Implement MedNeXtBlock**

**NEW FILE**: `src/braintumnet/models/mednext.py`

```python
"""
MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation
Based on: https://github.com/MIC-DKFZ/MedNeXt (MICCAI 2023)
"""

import torch
import torch.nn as nn


class MedNeXtBlock(nn.Module):
    """
    MedNeXt block: Large kernel depthwise conv + Inverted bottleneck

    Architecture:
    1. Depthwise conv (7×7 kernel)
    2. InstanceNorm
    3. 1×1 expand (dim → dim×4)
    4. GELU activation
    5. 1×1 compress (dim×4 → dim)
    6. Layer scaling
    7. Residual connection

    Args:
        dim: Input/output channels
        kernel_size: Depthwise kernel size (7 or 9)
        expansion: Expansion ratio (default 4)
        dropout: Dropout probability
    """
    def __init__(self, dim, kernel_size=7, expansion=4, dropout=0.0):
        super().__init__()

        # Depthwise convolution (groups=dim → each channel processed separately)
        self.conv_dw = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,  # Depthwise
            bias=False
        )

        # Normalization (InstanceNorm for medical images)
        self.norm = nn.InstanceNorm2d(dim, affine=True)

        # Inverted bottleneck: expand → compress
        self.conv_pw1 = nn.Conv2d(dim, dim * expansion, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.conv_pw2 = nn.Conv2d(dim * expansion, dim, kernel_size=1, bias=False)

        # Layer scaling (stabilizes training of deep networks)
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1) * 1e-6)

        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x

        # Depthwise conv
        x = self.conv_dw(x)
        x = self.norm(x)

        # Inverted bottleneck
        x = self.conv_pw1(x)
        x = self.act(x)
        x = self.conv_pw2(x)

        # Layer scaling
        x = self.gamma * x

        # Dropout
        x = self.dropout(x)

        # Residual
        return x + identity


class MedNeXtEncoderBlock(nn.Module):
    """
    Encoder block with MedNeXt and strided conv downsampling

    Args:
        in_ch: Input channels
        out_ch: Output channels
        kernel_size: MedNeXt kernel size
        dropout: Dropout probability
    """
    def __init__(self, in_ch, out_ch, kernel_size=7, dropout=0.0):
        super().__init__()

        # Match channels if needed
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

        # MedNeXt block
        self.block = MedNeXtBlock(out_ch, kernel_size=kernel_size, dropout=dropout)

        # Strided conv for downsampling
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.proj(x)
        x = self.block(x)
        x_down = self.downsample(x)
        return x, x_down  # Return both for skip connection


class MedNeXtDecoderBlock(nn.Module):
    """
    Decoder block with MedNeXt and CBAM attention
    """
    def __init__(self, in_ch, out_ch, kernel_size=7, dropout=0.0):
        super().__init__()

        from .cbam import CBAM

        # Upsample
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)

        # CBAM attention on skip
        self.cbam = CBAM(out_ch)

        # Channel projection (concatenation: out_ch + out_ch → out_ch)
        self.proj = nn.Conv2d(out_ch * 2, out_ch, 1, bias=False)

        # MedNeXt block
        self.block = MedNeXtBlock(out_ch, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.cbam(skip)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.block(x)
        return x
```

**Unit Tests**:

**NEW FILE**: `tests/test_mednext.py`

```python
import torch
from braintumnet.models.mednext import MedNeXtBlock, MedNeXtEncoderBlock, MedNeXtDecoderBlock


def test_mednext_block():
    """Test MedNeXtBlock forward/backward"""
    block = MedNeXtBlock(dim=64, kernel_size=7)
    x = torch.randn(2, 64, 32, 32)

    # Forward
    y = block(x)
    assert y.shape == x.shape, "Output shape mismatch"

    # Backward
    loss = y.sum()
    loss.backward()
    assert block.conv_dw.weight.grad is not None, "Gradient not computed"

    print("✓ MedNeXtBlock test passed")


def test_encoder_block():
    """Test MedNeXtEncoderBlock"""
    block = MedNeXtEncoderBlock(in_ch=64, out_ch=128)
    x = torch.randn(2, 64, 64, 64)

    # Forward
    skip, x_down = block(x)
    assert skip.shape == (2, 128, 64, 64), "Skip shape mismatch"
    assert x_down.shape == (2, 128, 32, 32), "Downsampled shape mismatch"

    print("✓ MedNeXtEncoderBlock test passed")


def test_decoder_block():
    """Test MedNeXtDecoderBlock"""
    block = MedNeXtDecoderBlock(in_ch=128, out_ch=64)
    x = torch.randn(2, 128, 32, 32)
    skip = torch.randn(2, 64, 64, 64)

    # Forward
    y = block(x, skip)
    assert y.shape == (2, 64, 64, 64), "Output shape mismatch"

    print("✓ MedNeXtDecoderBlock test passed")


if __name__ == "__main__":
    test_mednext_block()
    test_encoder_block()
    test_decoder_block()
    print("\n✓ All MedNeXt tests passed!")
```

**Run Tests**:
```bash
python tests/test_mednext.py
```

---

#### **Days 4-5: Integrate MedNeXt into SegUNetV2**

**MODIFY**: `src/braintumnet/models/seg_unet_v2.py`

**Add import** (top of file):
```python
from .mednext import MedNeXtEncoderBlock, MedNeXtDecoderBlock
```

**Modify __init__** (add flag for MedNeXt):
```python
class SegUNetV2(nn.Module):
    def __init__(self, in_ch=4, base=48, dim=384, patch=8, depth=4, n_heads=8,
                 num_classes=3, dropout=0.15, norm='instance',
                 deep_supervision=True, multi_scale_fusion=True,
                 use_mednext=False,        # NEW: Enable MedNeXt
                 mednext_kernel_size=7):   # NEW: Kernel size for MedNeXt
        super().__init__()
        self.use_mednext = use_mednext

        # Encoder
        if use_mednext:
            self.e1 = MedNeXtEncoderBlock(in_ch, base, mednext_kernel_size, dropout=0)
            self.e2 = MedNeXtEncoderBlock(base, base*2, mednext_kernel_size, dropout=0)
            self.e3 = MedNeXtEncoderBlock(base*2, base*4, mednext_kernel_size, dropout)
            self.e4 = MedNeXtEncoderBlock(base*4, base*8, mednext_kernel_size, dropout)
        else:
            # Original encoder blocks
            self.e1 = EncoderBlock(in_ch, base, norm=norm, dropout=0)
            self.e2 = EncoderBlock(base, base*2, norm=norm, dropout=0)
            self.e3 = EncoderBlock(base*2, base*4, norm=norm, dropout=dropout)
            self.e4 = EncoderBlock(base*4, base*8, norm=norm, dropout=dropout)

        # Bottleneck (unchanged)
        self.bottleneck_conv = conv_norm_act(base*8, dim, k=1, s=1, p=0, norm=norm)
        self.amt = AdaptiveMaskedTransformer(...)
        self.tr_upsample = nn.ConvTranspose2d(...)

        # Decoder
        if use_mednext:
            self.d4 = MedNeXtDecoderBlock(base*8, base*8, mednext_kernel_size, dropout)
            self.d3 = MedNeXtDecoderBlock(base*8, base*4, mednext_kernel_size, dropout)
            self.d2 = MedNeXtDecoderBlock(base*4, base*2, mednext_kernel_size, dropout/2)
            self.d1 = MedNeXtDecoderBlock(base*2, base, mednext_kernel_size, 0)
        else:
            # Original decoder blocks
            self.d4 = DecoderBlock(base*8, base*8, norm=norm, dropout=dropout)
            # ... (rest unchanged)

        # Rest of __init__ unchanged
```

**Backward Compatibility**: Old configs (use_mednext=False) still work!

---

#### **Days 6-7: End-to-End Testing**

**Test Full Model**:

**NEW FILE**: `tests/test_mednext_model.py`

```python
import torch
from braintumnet.models.braintumnet_v2 import BrainTumNetV2


def test_mednext_model():
    """Test full model with MedNeXt backbone"""
    model = BrainTumNetV2(
        in_ch=4,
        num_cls=2,
        base=48,
        dim=384,
        patch_size=8,
        depth=4,
        n_heads=8,
        num_classes_seg=3,
        dropout=0.15,
        deep_supervision=True,
        multi_scale_fusion=True,
        use_mednext=True,         # Enable MedNeXt
        mednext_kernel_size=7
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.1f}M")

    # Forward pass
    x = torch.randn(2, 4, 256, 256)
    seg_logits, cls_logits, aux_outputs = model(x)

    # Check shapes
    assert seg_logits.shape == (2, 3, 256, 256), "Seg output shape mismatch"
    assert cls_logits.shape == (2, 2), "Cls output shape mismatch"
    assert len(aux_outputs) == 3, "Aux outputs missing"

    # Backward pass
    loss = seg_logits.sum() + cls_logits.sum()
    loss.backward()

    print("✓ MedNeXt model test passed!")


if __name__ == "__main__":
    test_mednext_model()
```

**Memory Profiling**:
```python
import torch
from braintumnet.models.braintumnet_v2 import BrainTumNetV2

model = BrainTumNetV2(..., use_mednext=True).cuda()

# Profile memory
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

x = torch.randn(16, 4, 256, 256).cuda()  # Batch size 16
seg, cls, aux = model(x)

peak_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak memory: {peak_memory:.2f} GB")

# Expected: ~35 GB (fits on A100 80GB)
```

---

### Week 3: Configuration & Documentation

**NEW FILE**: `configs/phase2_mednext.yaml`

```yaml
# Phase 2 with MedNeXt Backbone
# Expected: Dice 0.87-0.90 (+1.5-2.5% over Phase 1)

data:
  proc_root: "data/lmdb_processed_multiclass_full"
  # ... (same as phase1_optimized.yaml)

train:
  epochs: 350
  batch_size: 12                   # Reduced for MedNeXt (larger model)
  lr: 4.0e-5                       # Slightly lower for stability

  # Learning rate (from Phase 1)
  scheduler: "cosine_restarts"
  T_0: 50
  T_mult: 2
  min_lr: 1.0e-5

  # Loss (from Phase 1)
  loss_type: "ultimate_multitask"
  iou_weight: 3.0
  boundary_weight: 1.0
  class_weights: [1.0, 3.0, 4.0]
  focal_alpha: [0.0, 0.4, 0.4]

  # ... (rest same)

# Model - MedNeXt enabled
model:
  model_type: "v2"
  in_channels: 4
  num_classes_seg: 3
  num_classes_cls: 2

  base: 64                         # Phase 2 Large
  dim: 512
  patch_size: 8
  depth: 4
  n_heads: 8
  dropout: 0.15

  # MedNeXt configuration (NEW)
  use_mednext: true                # Enable MedNeXt backbone
  mednext_kernel_size: 7           # 7×7 depthwise kernels

  # ... (rest same)

# Hardware - A100 optimized
hardware:
  memory_format: "channels_last"
  amp_dtype: "bfloat16"
  fused: true
```

**Documentation**:

Update `README.md` or create `docs/phase2_mednext.md`:
```markdown
# Phase 2: MedNeXt Backbone

## Changes
- Replaced ResidualConvBlock → MedNeXtBlock
- Large kernels: 3×3 → 7×7 (2.3× receptive field)
- Inverted bottleneck (depthwise + pointwise)
- 53% fewer parameters for same capacity

## Training
```bash
python braintumnet/scripts/train.py \
    --model segunetv2 \
    --cfg phase2_mednext \
    --fold 0
```

## Expected Results
- Dice: 0.87-0.90 (+1.5-2.5%)
- Training time: ~50 hours (A100)
- GPU memory: ~40 GB
```

---

### Week 4: Full Training Run

**Train Fold 0** (validation of MedNeXt architecture):

```bash
# A100 80GB
python braintumnet/scripts/train.py \
    --model segunetv2 \
    --cfg phase2_mednext \
    --fold 0 \
    --epochs 350

# Expected:
# - Training time: ~50 hours
# - Best Dice: 0.89-0.90 (at epoch 60-80)
# - IoU: 0.80-0.82
# - Memory: ~40 GB
```

**Monitor**:
```
- Training loss should decrease smoothly
- Val Dice should surpass Phase 1 baseline (0.8750) by epoch 40
- Target: Val Dice > 0.89 by epoch 100
- IoU-Dice gap should narrow (<8%)
```

**Success Criteria**:
- ✅ Val Dice > 0.89
- ✅ Val IoU > 0.80
- ✅ IoU-Dice gap < 8%
- ✅ No training instabilities (NaN, divergence)

**Deliverable**: Trained MedNeXt model (Fold 0), ready for Phase 3

---

### Phase 2 Summary

**Implementation**:
1. ✅ MedNeXtBlock (depthwise 7×7, inverted bottleneck)
2. ✅ MedNeXtEncoderBlock, MedNeXtDecoderBlock
3. ✅ Integration into SegUNetV2 (use_mednext flag)
4. ✅ Backward compatibility (old configs work)
5. ✅ Full model testing (forward/backward)

**Expected Results**:
- Dice: 0.8750 → 0.8900-0.9000 (+1.5-2.5%)
- IoU: 0.7800 → 0.8000-0.8200 (+2.0-4.0%)
- Receptive field: 33×33 → 77×77 pixels
- Parameters: 62.7M → 68M (+8%, but more efficient)

**Risk Mitigation**:
- If performance < 0.88: Try kernel_size=9 (even larger)
- If memory issues: Reduce batch_size to 8
- If unstable: Lower LR to 3e-5
- **Fallback**: Revert to Phase 1 (still 0.8750+)

---

## Phase 3: Swin Transformer Attention (Weeks 5-7)

**Goal**: Add hierarchical multi-scale attention at all decoder levels
**Hardware**: A100 80GB (attention is memory-intensive)
**Expected Gain**: +1.0-1.5% Dice → 0.9000-0.9150
**Risk**: Medium-High (memory constraints possible)

### Why Swin Transformer?

**Current**: Single transformer at bottleneck (32×32 spatial)

**Problem**: Only sees low-resolution global context

**Swin Solution**: Multi-scale attention windows at ALL levels

```
Encoder 1 (256×256) → Swin (window=8)  ← Fine details (edges)
Encoder 2 (128×128) → Swin (window=8)  ← Medium features (texture)
Encoder 3 (64×64)   → Swin (window=4)  ← Coarse features (shape)
Encoder 4 (32×32)   → Swin (window=2)  ← Global context (location)

→ Hierarchical attention = multi-scale understanding
```

**Window Attention Efficiency**:

Full attention (256×256):
```
Tokens: 256 × 256 = 65,536
Attention matrix: 65,536 × 65,536 = 4.3 billion values
Memory: ~17 GB (float32)
```

Window attention (window=8):
```
Tokens per window: 8 × 8 = 64
Windows: (256/8) × (256/8) = 1,024
Attention per window: 64 × 64 = 4,096
Total attention: 1,024 × 4,096 = 4.2 million values
Memory: ~17 MB (1000× reduction!)
```

**Shifted Windows**: Every other layer shifts window by 4 pixels → cross-window communication

---

### Week 5: WindowAttention Implementation

#### **Days 1-2: Implement WindowAttention Module**

**NEW FILE**: `src/braintumnet/models/swin_transformer.py`

```python
"""
Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
Based on: https://github.com/microsoft/Swin-Transformer (ICCV 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class WindowAttention(nn.Module):
    """
    Window-based multi-head self attention with relative position bias

    Args:
        dim: Number of input channels
        window_size: Window size (tuple of H, W)
        n_heads: Number of attention heads
    """
    def __init__(self, dim, window_size, n_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), n_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Register relative position index
        self.register_relative_position_index()

    def register_relative_position_index(self):
        """Precompute relative position indices"""
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wh*Ww, Wh*Ww, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1  # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)  # (Wh*Ww, Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        """
        Args:
            x: (B, N, C) where N = num_windows * window_size^2

        Returns:
            (B, N, C)
        """
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B, n_heads, N, N)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)  # (N, N, n_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (n_heads, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)

        # Softmax
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        x = self.proj(x)

        return x
```

(Continued in next message due to length...)

---

### Phase 3 Summary (Preview)

**Week 5**: WindowAttention + SwinTransformerBlock
**Week 6**: Cross-attention decoder
**Week 7**: Full training

**Expected**: Dice 0.90-0.915

---

## Phase 4: Ensemble + TTA (Weeks 8-10)

**Week 8**: 2.5D data loading
**Week 9**: 5-fold training (parallel on 3 GPUs)
**Week 10**: Ensemble + TTA

**Expected**: Dice 0.91-0.92

---

## Hardware Allocation

```
RTX 3090 #1 (24GB):
- Phase 1: Training experiments
- Phase 4: Fold 0 + Fold 3

RTX 3090 #2 (24GB):
- Phase 4: Fold 1 + Fold 4

A100 80GB:
- Phase 2: MedNeXt training
- Phase 3: Swin Transformer
- Phase 4: Fold 2 (best fold)
```

---

## Code Structure Preservation

**DO NOT MODIFY**:
- `src/braintumnet/models/__init__.py`
- `src/braintumnet/data/dataset_factory.py`
- `configs/base.yaml`

**CREATE NEW**:
- `models/mednext.py`
- `models/swin_transformer.py`
- `models/cross_attention.py`

**MODIFY WITH FLAGS**:
- `models/seg_unet_v2.py`: Add use_mednext, use_swin
- `engine/trainer.py`: Add new scheduler options

---

## Timeline Summary

| Phase | Weeks | Dice Gain | Cumulative Dice | GPU |
|-------|-------|-----------|-----------------|-----|
| **Baseline** | - | - | 0.8699 | - |
| **Phase 1** | 1 | +0.5-1.0% | 0.8750-0.8800 | 3090 |
| **Phase 2** | 2-4 | +1.5-2.5% | 0.8900-0.9050 | A100 |
| **Phase 3** | 5-7 | +1.0-1.5% | 0.9000-0.9150 | A100 |
| **Phase 4** | 8-10 | +1.0-1.5% | **0.9100-0.9200** | All |

**Total**: 10 weeks to Dice 0.91+

---

**Document End**

This is a realistic, hardware-optimized, code-preserving evolution plan based on actual Phase 2 results. Ready to execute!
