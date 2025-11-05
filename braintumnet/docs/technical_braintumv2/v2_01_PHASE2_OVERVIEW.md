# Phase 2 Overview - Tổng Quan Phase 2

> **Giải thích Phase 2 là gì và tại sao cần nó**

---

## Phase 2 Là Gì?

**Phase 2** là giai đoạn nâng cấp model từ baseline V1 lên **SegUNetV2** với nhiều cải tiến để:
- Tăng accuracy cho cả binary và multi-class segmentation
- Scale up model capacity (14M → 37M → 87M parameters)
- Áp dụng medical imaging best practices (InstanceNorm, LeakyReLU)
- Thêm advanced features (multi-scale fusion, deep supervision, attention gates)

---

## Timeline Phát Triển

### V1 Baseline (2025-10-15)
**Model**: seg_unet.py (67 dòng)
**Architecture**:
- U-Net với CBAM attention
- Adaptive Masked Transformer bottleneck
- BatchNorm + ReLU
- MaxPool downsampling

**Results**:
- Binary Dice: **0.9148** ✅ (rất tốt!)
- Multi-class IoU: **0.7263** (chấp nhận được)
- HD95: **2.73mm**

**Vấn đề**:
- Multi-class segmentation chưa tốt (WT=0.04, ED=0.009)
- Model capacity nhỏ (14M params)
- BatchNorm không optimal cho medical imaging
- Thiếu residual connections → khó train deeper

---

### Phase 1 Optimization (2025-10-15 - 2025-12-31)

**Improvements**:
1. **Ultimate Loss** - combined loss cho multi-class
2. **Boundary Refinement Module** - cải thiện IoU-Dice gap
3. **Training optimizations** - better scheduler, gradient centralization

**Results**:
- IoU cải thiện lên **0.75-0.80** (+3-8%)
- Boundary precision tốt hơn
- Nhưng vẫn chưa đủ cho multi-class tốt

**Config**: phase1_optimized.yaml, segunetv2_p1.yaml

---

### Phase 2 - SegUNetV2 (2025-01-04 - hiện tại)

**Major Changes**: Thiết kế lại toàn bộ architecture

**File mới**:
- `seg_unet_v2.py` (478 dòng) - complete redesign
- `braintumnet_v2.py` (170 dòng) - V2 wrapper
- `multiscale_transformer.py` (243 dòng) - multi-scale bottleneck

**7 Core Improvements**:
1. InstanceNorm thay BatchNorm
2. LeakyReLU thay ReLU
3. Residual blocks trong tất cả layers
4. Strided conv thay MaxPool
5. Multi-scale fusion module
6. Deep supervision
7. Dropout regularization

**3 Optional Features**:
8. Multi-scale transformer bottleneck
9. Attention gates for skip connections
10. Boundary refinement module (from Phase 1)

**Configs**:
- `phase2_small.yaml` - 37M params (RTX 3090)
- `phase2_a100.yaml` - 87M params (A100 80GB)

**Target Results**:
- Multi-class WT: **0.83-0.86** (vs 0.04 ❌)
- Multi-class TC: **0.80-0.83** (vs 0.81 ✓)
- Multi-class ED: **0.82-0.85** (vs 0.009 ❌)

---

## Tại Sao Cần Phase 2?

### Vấn Đề 1: Multi-Class Segmentation Thất Bại

**V1 Results** (multi-class):
```
WT (Whole Tumor): Dice 0.04  ❌ (should be ~0.85)
TC (Tumor Core):  Dice 0.81  ✓  (acceptable)
ED (Edema):       Dice 0.009 ❌ (should be ~0.83)
```

**Tại sao?**
- Model capacity quá nhỏ (14M params)
- Binary segmentation đơn giản hơn multi-class
- BatchNorm không ổn định với batch size nhỏ
- Thiếu multi-scale information

**Phase 2 Solution**:
- Scale up: 14M → 37M/87M params
- InstanceNorm (không phụ thuộc batch size)
- Multi-scale fusion (combine all decoder levels)
- Deep supervision (better gradient flow)

---

### Vấn Đề 2: Model Capacity Không Đủ

**V1 Baseline**:
```python
base = 32   # Too small for complex segmentation
dim = 256   # Transformer capacity limited
depth = 2   # Shallow transformer
n_heads = 4 # Limited attention
```

**Phase 2 Small**:
```python
base = 48   # 1.5x larger (32 → 48)
dim = 384   # 1.5x transformer capacity
depth = 4   # 2x deeper
n_heads = 8 # 2x more attention heads
```

**Kết quả**: 14M → 37M params (2.6x)

**Phase 2 Large**:
```python
base = 64   # 2x larger
dim = 512   # 2x transformer capacity
depth = 4   # Same depth
n_heads = 8 # Same heads
```

**Kết quả**: 14M → 87M params (6.2x)

---

### Vấn Đề 3: BatchNorm Không Tốt Cho Medical Imaging

**BatchNorm**:
```python
mean = x.mean(dim=(0, 2, 3))  # Across batch
var = x.var(dim=(0, 2, 3))
```

**Vấn đề**:
- Medical imaging: batch size nhỏ (4-8) → statistics không ổn định
- Training vs Inference khác nhau (batch stats vs running stats)
- Không phù hợp với patient-specific imaging

**InstanceNorm** (Phase 2):
```python
mean = x.mean(dim=(2, 3), keepdim=True)  # Per sample
var = x.var(dim=(2, 3), keepdim=True)
```

**Lợi ích**:
- Không phụ thuộc batch size
- Training == Inference
- Standard trong medical imaging (nnU-Net, MedicalNet)
- Tốt hơn với augmentation

---

### Vấn Đề 4: Single-Scale Features

**V1**: Chỉ dùng final decoder output (d1)
```
d4 → d3 → d2 → d1 → HEAD
              Only this!
```

**Phase 2**: Multi-scale fusion
```
d4 (high-level) ──┐
d3 (mid-level)  ──┼→ FUSION → HEAD
d2 (low-level)  ──┤
d1 (details)    ──┘
```

**Benefit**:
- d4: Semantic information (tumor vs not)
- d3: Structural boundaries
- d2: Fine details
- d1: Precise localization

---

## Model Configurations Chi Tiết

### V1 Baseline (Reference)

```yaml
# seg_unet.py
Parameters: 14M
Memory: ~12GB (batch=16)
Training: ~2.5s/epoch

Architecture:
  base: 32
  dim: 256
  depth: 2
  n_heads: 4
  norm: batch
  activation: relu
  downsampling: maxpool
  residuals: false
  multi_scale_fusion: false
  deep_supervision: false
  dropout: 0.0

Results (binary):
  Dice: 0.9148
  IoU: 0.8430
  HD95: 2.73mm

Results (multi-class):
  WT: 0.04 ❌
  TC: 0.81 ✓
  ED: 0.009 ❌
```

---

### Phase 2 Small (Recommended)

```yaml
# seg_unet_v2.py
Parameters: 37M (2.6x V1)
Memory: ~16GB (batch=8)
Training: ~3.5s/epoch
Hardware: RTX 3090 24GB

Architecture:
  base: 48        # +50% from V1
  dim: 384        # +50% from V1
  depth: 4        # 2x from V1
  n_heads: 8      # 2x from V1
  norm: instance  # Changed from batch
  activation: leakyrelu  # Changed from relu
  downsampling: strided_conv  # Changed from maxpool
  residuals: true  # NEW
  multi_scale_fusion: true  # NEW
  deep_supervision: true    # NEW
  dropout: 0.15   # NEW

Target Results (binary):
  Dice: 0.92-0.93 (+1-2%)
  IoU: 0.85-0.87 (+2-3%)
  HD95: 2.2-2.5mm

Target Results (multi-class):
  WT: 0.83-0.86 (+0.82!)
  TC: 0.80-0.83 (stable)
  ED: 0.82-0.85 (+0.81!)

Training Time:
  Per epoch: ~20 minutes (350 epochs)
  Total: ~116 hours (~5 days)

Config: phase2_small.yaml
```

**Khi nào dùng**:
- RTX 3090 hoặc tương đương
- Cần balance giữa accuracy và speed
- Multi-class segmentation
- Production deployment

---

### Phase 2 Large (Best Performance)

```yaml
# seg_unet_v2.py
Parameters: 87M (6.2x V1)
Memory: ~28GB (batch=16)
Training: ~5s/epoch
Hardware: A100 80GB

Architecture:
  base: 64        # 2x from V1
  dim: 512        # 2x from V1
  depth: 4        # 2x from V1
  n_heads: 8      # 2x from V1
  norm: instance
  activation: leakyrelu
  downsampling: strided_conv
  residuals: true
  multi_scale_fusion: true
  deep_supervision: true
  dropout: 0.2    # Higher for larger model

Target Results (binary):
  Dice: 0.93-0.94 (+2-3%)
  IoU: 0.87-0.89 (+4-5%)
  HD95: 2.0-2.3mm

Target Results (multi-class):
  WT: 0.85-0.88
  TC: 0.83-0.86
  ED: 0.84-0.87

Training Time:
  Per epoch: ~30 minutes (400 epochs)
  Total: ~200 hours (~8 days)

Config: phase2_a100.yaml
```

**Khi nào dùng**:
- A100 hoặc high-end GPU
- Cần accuracy cao nhất
- Research experiments
- Có thời gian training dài

---

## Phase 2 Optional Features

Ngoài 7 core improvements, Phase 2 có 3 features tùy chọn:

### 1. Multi-Scale Transformer Bottleneck

**File**: multiscale_transformer.py (243 dòng)

**Concept**: Thay vì 1 patch size, dùng nhiều patch sizes (4, 8, 16)
```
Patch 4  → Small receptive field  → Fine details
Patch 8  → Medium receptive field → Normal features
Patch 16 → Large receptive field  → Global context
         ↓
       FUSION
```

**Benefits**:
- Better multi-resolution reasoning
- Capture both local and global patterns
- Expected: +1.5-2.5% Dice

**Trade-offs**:
- +40% parameters ở bottleneck
- +30% slower training
- +2GB memory

**Enable**:
```yaml
model:
  use_multiscale_transformer: true
```

**Khi nào dùng**:
- Có GPU tốt (A100)
- Cần accuracy cao nhất
- Tumors nhiều kích thước khác nhau

---

### 2. Attention Gates

**File**: seg_unet_v2.py (AttentionGate class)

**Concept**: nnU-Net style attention cho skip connections
```
Decoder signal (g) → tells what to focus on
Skip connection (x) → encoder features
                  ↓
            Attention Map
                  ↓
      Weighted Skip Connection
```

**Benefits**:
- Suppress irrelevant regions
- Focus on salient features
- Expected: +1-2% Dice

**Trade-offs**:
- +5% parameters
- +10% slower
- +0.5GB memory

**Enable**:
```yaml
model:
  use_attention_gates: true
```

**Khi nào dùng**:
- Multi-class segmentation phức tạp
- Nhiều background noise
- Small tumors

---

### 3. Boundary Refinement Module

**File**: seg_unet_v2.py (BoundaryRefinementModule)

**Concept**: Edge detection + boundary attention
```
Features → Edge Detector (Sobel-like)
        → Boundary Attention
        → Refined Features
```

**Benefits**:
- Better edge precision
- Reduces IoU-Dice gap (10% → 5%)
- Expected: +2-3% Dice

**Trade-offs**:
- +3% parameters
- +5% slower
- Minimal memory

**Enable**:
```yaml
model:
  boundary_refinement: true
```

**Khi nào dùng**:
- Cần edge precision cao
- IoU-Dice gap lớn
- Clinical applications

---

## Hardware Requirements

### Minimum (Phase 2 Small)
- **GPU**: RTX 3090 24GB (hoặc tương đương)
- **RAM**: 32GB system RAM
- **Storage**: 20GB cho dataset + checkpoints
- **Training time**: ~5 days (350 epochs)

### Recommended (Phase 2 Large)
- **GPU**: A100 40GB/80GB
- **RAM**: 64GB system RAM
- **Storage**: 50GB (dataset + multiple experiments)
- **Training time**: ~8 days (400 epochs)

### For Development/Testing
- **GPU**: RTX 3060 12GB (reduce batch size to 4)
- **RAM**: 16GB
- **Use**: batch_size=4, base=32, dim=256

---

## Expected Performance Gains

### Binary Segmentation

| Metric | V1 | Phase 2 Small | Phase 2 Large | Gain (Large) |
|--------|----|--------------|--------------|----|
| Dice | 0.9148 | 0.92-0.93 | 0.93-0.94 | +1.5-2.5% |
| IoU | 0.8430 | 0.85-0.87 | 0.87-0.89 | +2.7-4.7% |
| HD95 | 2.73mm | 2.2-2.5mm | 2.0-2.3mm | -0.43-0.73mm |

### Multi-Class Segmentation

**V1 (failed)**:
- WT: 0.04, TC: 0.81, ED: 0.009

**Phase 2 Small (target)**:
- WT: 0.83-0.86 (+0.82)
- TC: 0.80-0.83 (stable)
- ED: 0.82-0.85 (+0.81)

**Phase 2 Large (target)**:
- WT: 0.85-0.88 (+0.84)
- TC: 0.83-0.86 (+0.02)
- ED: 0.84-0.87 (+0.83)

---

## Training Recommendations

### Phase 2 Small
```yaml
epochs: 350
batch_size: 8
lr: 3.0e-5
optimizer: adamw
scheduler: cosine
warmup_steps: 1000
amp: true
grad_accum_steps: 2  # effective batch=16
```

**Training curve**:
- Epochs 0-50: Rapid improvement
- Epochs 50-150: Steady progress
- Epochs 150-300: Fine-tuning
- Epochs 300-350: Convergence

### Phase 2 Large
```yaml
epochs: 400
batch_size: 16
lr: 5.0e-5
optimizer: adamw (fused)
scheduler: cosine
warmup_steps: 2000
amp: true (bfloat16 on A100)
channels_last: true
```

**A100 Optimizations**:
- BFloat16 (native support)
- Channels last memory format
- Fused optimizer
- Persistent workers
- Increased prefetch_factor

---

## Migration Path

### Từ V1 → Phase 2

**Step 1**: Test Phase 2 với baseline config
```yaml
# Tương đương V1 performance
base: 32
dim: 256
depth: 2
n_heads: 4
```
Expect: Similar results (~0.91 Dice)

**Step 2**: Enable Phase 2 improvements
```yaml
base: 48
dim: 384
depth: 4
dropout: 0.15
```
Expect: +2-3% improvement

**Step 3**: Add optional features
```yaml
multi_scale_fusion: true
deep_supervision: true
```
Expect: +1-2% more

**Step 4**: Try advanced features
```yaml
use_multiscale_transformer: true
use_attention_gates: true
```
Expect: +1-2% more (if GPU allows)

---

## Summary

**Phase 2 Goals**:
1. ✅ Fix multi-class segmentation (0.04 → 0.85)
2. ✅ Scale model capacity (14M → 37M/87M)
3. ✅ Apply medical imaging best practices
4. ✅ Add advanced features (optional)
5. ✅ Maintain backward compatibility

**Key Improvements**:
- 7 core improvements (always on)
- 3 optional features (enable as needed)
- 2 configurations (small, large)
- Expected: +5-10% Dice overall

**Trade-offs**:
- 2.6-6.2x more parameters
- 40-100% slower training
- +4-16GB more memory
- But significantly better results!

---

**Next**: [SegUNetV2 Architecture →](v2_02_SEGUNETV2_ARCHITECTURE.md)

**Back**: [← Index](v2_00_INDEX.md)
