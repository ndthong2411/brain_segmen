# AMT-UNet - Technical Summary

This document provides a quick reference for the key technical details of AMT-UNet (Adaptive Masked Transformer U-Net) as implemented in the code and described in the paper.

## Architecture Overview

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Channels | 48 |
| Transformer Dim | 384 |
| Transformer Depth | 4 |
| Transformer Heads | 8 |
| Dropout | 0.15 |
| Total Parameters | 37M |

### Input/Output Specifications

- **Input**: 4-channel MRI (FLAIR, T1, T1CE, T2) of size $256 \times 256$
- **Segmentation Output**: 3-class logits (Background, Tumor Core, Edema)
- **Classification Output**: 2-class logits (LGG, HGG)
- **Auxiliary Outputs**: 3 deep supervision outputs at resolutions $64 \times 64$, $128 \times 128$, $256 \times 256$

## Architecture Components (from Code)

### SegUNetV2 Encoder

```
Input (4 × 256 × 256)
  ↓
EncoderBlock1: ResConvBlock → StrideConv2d(s=2)
  → skip1 (48 × 256 × 256), x1 (48 × 128 × 128)
  ↓
EncoderBlock2: ResConvBlock → StrideConv2d(s=2)
  → skip2 (96 × 128 × 128), x2 (96 × 64 × 64)
  ↓
EncoderBlock3: ResConvBlock(dropout=0.15) → StrideConv2d(s=2)
  → skip3 (192 × 64 × 64), x3 (192 × 32 × 32)
  ↓
EncoderBlock4: ResConvBlock(dropout=0.15) → StrideConv2d(s=2)
  → skip4 (384 × 32 × 32), x4 (384 × 16 × 16)
```

### Residual Convolutional Block (ResConvBlock)

```python
# From seg_unet_v2.py lines 53-79
conv1 = Conv3x3-InstanceNorm-LeakyReLU(0.01)-Dropout
conv2 = Conv3x3-InstanceNorm
residual = Conv1x1 (if channel mismatch) else Identity
output = LeakyReLU(conv2(conv1(x)) + residual(x))
```

### Adaptive Masked Transformer Bottleneck

```
x4 (384 × 16 × 16)
  ↓
Conv1x1 → (384 × 16 × 16)
  ↓
PatchEmbed(patch_size=8) → tokens (4, 384) where 4 = (16/8)²
  ↓
SoftMaskGenerator: MLP(384 → 192 → 8) → masks (8, 4) for 8 heads
  ↓
4× TransformerBlock:
  - MaskedSelfAttention(heads=8, dim=384)
    Q, K, V = Linear(384 → 3×384)
    Attn = softmax((Q·K^T)/√48 + log(mask))
  - FFN: Linear(384 → 1536) → GELU → Linear(1536 → 384)
  ↓
Reshape → (384 × 2 × 2)
  ↓
ConvTranspose2d(kernel=8, stride=8) → (384 × 16 × 16)
```

### SegUNetV2 Decoder

```
Bottleneck output (384 × 16 × 16)
  ↓
DecoderBlock4: ConvTranspose2d(s=2) → Concat[up, CBAM(skip4)] → ResConvBlock
  → d4 (384 × 32 × 32) [+ aux_head3 for deep supervision]
  ↓
DecoderBlock3: ConvTranspose2d(s=2) → Concat[up, CBAM(skip3)] → ResConvBlock
  → d3 (192 × 64 × 64) [+ aux_head2 for deep supervision]
  ↓
DecoderBlock2: ConvTranspose2d(s=2) → Concat[up, CBAM(skip2)] → ResConvBlock
  → d2 (96 × 128 × 128) [+ aux_head1 for deep supervision]
  ↓
DecoderBlock1: ConvTranspose2d(s=2) → Concat[up, CBAM(skip1)] → ResConvBlock
  → d1 (48 × 256 × 256)
```

### CBAM Attention Module

```python
# From cbam.py
Channel Attention:
  avg_pool = AdaptiveAvgPool2d(1)  # (C, 1, 1)
  max_pool = AdaptiveMaxPool2d(1)  # (C, 1, 1)
  mlp = Conv1x1(C → C/16) → ReLU → Conv1x1(C/16 → C)
  channel_attn = sigmoid(mlp(avg_pool) + mlp(max_pool))

Spatial Attention:
  avg_channel = mean along channel dim  # (1, H, W)
  max_channel = max along channel dim   # (1, H, W)
  spatial_attn = sigmoid(Conv7x7(concat[avg_channel, max_channel]))

Output = x * channel_attn * spatial_attn
```

### Multi-Scale Fusion

```python
# From seg_unet_v2.py lines 250-292
# Fuse features from all decoder levels
f1 = Conv1x1(d1) → (48 × 256 × 256)
f2 = Conv1x1(d2) → Interpolate(256×256) → (48 × 256 × 256)
f3 = Conv1x1(d3) → Interpolate(256×256) → (48 × 256 × 256)
f4 = Conv1x1(d4) → Interpolate(256×256) → (48 × 256 × 256)

fused = InstanceNorm(LeakyReLU(f1 + f2 + f3 + f4))
final = ResConvBlock(Concat[d1, fused])  # (96 → 48)
seg_output = Conv1x1(final)  # (48 → 3)
```

### ROI-Guided Classification Network

```python
# From braintumnet_v2.py lines 99-119
# 1. Extract whole tumor probability
seg_prob = softmax(seg_logits)  # (B, 3, H, W)
wt_prob = seg_prob[:, 1:, :, :].sum(dim=1, keepdim=True)  # TC + ED

# 2. Reduce input channels and mask
roi_input = Conv1x1(input)  # (B, 4, H, W) → (B, 1, H, W)
roi_masked = roi_input * wt_prob.detach()  # Stop gradient

# 3. T-Inception classification
stem = Conv3x3-BN-ReLU(1 → 64)

TInceptionBlock1:
  b1 = Conv1x1(64 → 32)
  b2 = Conv3x3(64 → 32)
  b3 = Conv1x3(64 → 32)
  b4 = Conv3x1(64 → 32)
  concat = [b1, b2, b3, b4]  # (128)
  fuse = Conv1x1(128 → 128)

TInceptionBlock2: similar (128 → 256)

Global pool → Dropout(0.3) → Linear(256 → 2)
```

## Loss Functions (Exact Formulas from Code)

### Ultimate Combined Loss

```python
# From losses_combined.py
L_total = w_seg * L_seg + w_cls * L_cls
  where w_seg = 1.0, w_cls = 0.5

L_seg = w_D * L_Dice + w_F * L_Focal + w_I * L_IoU + w_B * L_Boundary
  where w_D = 1.0, w_F = 1.0, w_I = 2.5, w_B = 0.6
```

### Dice Loss (Multi-Class)

```python
# From losses_multiclass.py lines 42-83
# For each class c in {1, 2} (skip background):
pred_c = softmax(logits)[:, c, :, :]
target_c = one_hot(target)[:, c, :, :]

intersection = (pred_c * target_c).sum()
union = pred_c.sum() + target_c.sum()

dice_c = (2 * intersection + eps) / (union + eps)
loss_c = (1 - dice_c) * class_weight[c]

L_Dice = mean([loss_1, loss_2])
# class_weight = [1.0, 3.0, 4.0] for [BG, TC, ED]
```

### Focal Loss (Multi-Class)

```python
# From losses_multiclass.py lines 107-138
# Focal loss with class weights and focusing parameter
probs = softmax(logits)  # (B, C, H, W)
pt = probs[target_indices]  # Probability of true class

focal_weight = (1 - pt)^gamma  # gamma = 3.0
ce = -log(pt + eps)
alpha_t = alpha[target]  # alpha = [0.0, 0.4, 0.3]

L_Focal = mean(alpha_t * focal_weight * ce)
```

### IoU Loss (Multi-Class)

```python
# From losses_iou.py lines 49-96
# For each class c in {1, 2}:
pred_c = softmax(logits)[:, c, :, :]
target_c = one_hot(target)[:, c, :, :]

intersection = (pred_c * target_c).sum()
union = pred_c.sum() + target_c.sum() - intersection

iou_c = (intersection + smooth) / (union + smooth)
loss_c = (1 - iou_c) * class_weight[c]

L_IoU = mean([loss_1, loss_2])
# smooth = 1.0
```

### Boundary Loss

```python
# From losses_boundary.py lines 93-146
# For each class c and sample b:
target_mask = (target == c).float()

# Compute signed distance function
sdf = distance_transform(target_mask)  # scipy.ndimage

# Boundary weight (exponential decay)
weight = exp(-|sdf| / theta)  # theta = 5 pixels

# Weighted L1 error
pred_c = softmax(logits)[c]
error = |pred_c - target_mask|
weighted_error = weight * error

L_Boundary = mean(weighted_error)
```

### Deep Supervision

```python
# From losses_combined.py lines 191-221
# Main output loss
L_main = L_seg(seg_main, target)

# Auxiliary output losses (3 levels)
for aux_output in [aux3, aux2, aux1]:
  target_resized = interpolate(target, size=aux_output.shape[-2:])
  L_aux_i = L_seg(aux_output, target_resized)

L_seg_total = L_main + w_aux * (L_aux3 + L_aux2 + L_aux1)
# w_aux = 0.3
```

### Classification Loss

```python
# From losses_combined.py line 228
L_cls = CrossEntropyLoss(cls_logits, cls_target)
```

## Training Configuration

### Optimizer: AdamW

```python
optimizer = AdamW(
    params=model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1.5e-4,
    fused=True
)
```

### Learning Rate Schedule: Cosine with Warmup

```python
# Warmup for first 1000-2000 steps
if step <= warmup_steps:
    lr = lr_max * (step / warmup_steps)
else:
    # Cosine annealing
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * progress))

# lr_max = 5e-5, lr_min = 5e-7
```

### Training Hyperparameters

```yaml
epochs: 400
batch_size: 12-16 (depending on GPU memory)
gradient_clip_norm: 1.0
early_stopping_patience: 30

# Mixed precision
amp: true
amp_dtype: bfloat16 or float16

# Optimization settings
channels_last: true
cudnn_benchmark: true
persistent_workers: true
prefetch_factor: 8
```

### Data Augmentation

```python
# Geometric
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomRotation(degrees=(-45, 45))

# Intensity
RandomBrightnessContrast(
    brightness_limit=0.25,
    contrast_limit=0.25
)

# Noise
GaussianNoise(std=0.01, p=0.2)
```

## Evaluation Metrics (from multiclass_metrics.py)

### Dice Coefficient

```python
# For each region (WT, TC, ED):
pred_hard = argmax(logits, dim=1)  # Hard predictions!
pred_region = (pred_hard == class_idx).float()
target_region = (target == class_idx).float()

intersection = (pred_region * target_region).sum()
union = pred_region.sum() + target_region.sum()

Dice = (2 * intersection + eps) / (union + eps)
```

### IoU

```python
intersection = (pred_region * target_region).sum()
union = pred_region.sum() + target_region.sum() - intersection

IoU = (intersection + eps) / (union + eps)
```

### Region Definitions

```python
# Whole Tumor: Any tumor class
WT = (pred_hard >= 1).float()  # TC or ED

# Tumor Core: Class 1 only
TC = (pred_hard == 1).float()

# Edema: Class 2 only
ED = (pred_hard == 2).float()
```

## Data Processing (from Code Analysis)

### Preprocessing Pipeline

1. **Z-score normalization** (per modality, within brain mask):
   ```python
   brain_mask = (image > 0)
   mean = image[brain_mask].mean()
   std = image[brain_mask].std()
   normalized = (image - mean) / (std + 1e-8)
   ```

2. **Slice sampling** (20 slices per case):
   - 50% from tumor-containing slices
   - 50% from all slices

3. **Resize**: $240 \times 240 \rightarrow 256 \times 256$
   - Bilinear for images
   - Nearest-neighbor for masks

4. **Label conversion**:
   ```python
   # Original BraTS labels: 0, 1, 2, 4
   # Converted to 3-class:
   # 0 → 0 (Background)
   # 1 → 1 (NCR → Tumor Core)
   # 2 → 2 (Edema)
   # 4 → 1 (ET → Tumor Core)
   ```

### Dataset Statistics

- Total cases: 369 (293 HGG, 76 LGG)
- Volume size: $240 \times 240 \times 155$
- Modalities: 4 (FLAIR, T1, T1CE, T2)
- Slices per case: 20 (sampled)
- Total slices: ~7,380 (369 × 20)

### 5-Fold Cross-Validation

- Stratified by tumor grade
- Train: ~295 cases
- Validation: ~74 cases

## Key Implementation Details

### Instance Normalization

```python
# From seg_unet_v2.py line 36
# Statistics computed per-instance and per-channel
# Prevents batch size dependency
nn.InstanceNorm2d(num_features, affine=True)
```

### Learned Downsampling

```python
# From seg_unet_v2.py line 94
# Strided convolution instead of MaxPool
nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
```

### Gradient Stopping in ROI

```python
# From braintumnet_v2.py line 114
# Prevent classification gradients from affecting segmentation
roi = roi_input * seg_prob.detach()
```

## Hardware Requirements

### Minimum Requirements

- GPU: RTX 3090 24GB or similar
- RAM: 32GB
- Storage: 100GB for dataset + checkpoints

### Recommended Setup

- GPU: A100 80GB or RTX 4090
- RAM: 64GB
- Storage: 200GB

### Training Time

- AMT-UNet (37M parameters): ~40-50 hours depending on GPU

## Code Files Reference

All formulas and specifications extracted from:

- `braintumnet/src/braintumnet/models/braintumnet_v2.py` (170 lines)
- `braintumnet/src/braintumnet/models/seg_unet_v2.py` (478 lines)
- `braintumnet/src/braintumnet/models/masked_transformer.py` (105 lines)
- `braintumnet/src/braintumnet/models/cbam.py` (33 lines)
- `braintumnet/src/braintumnet/models/t_inception.py` (51 lines)
- `braintumnet/src/braintumnet/losses_combined.py` (425 lines)
- `braintumnet/src/braintumnet/losses_multiclass.py` (239 lines)
- `braintumnet/src/braintumnet/losses_iou.py` (325 lines)
- `braintumnet/src/braintumnet/losses_boundary.py` (364 lines)
- `braintumnet/src/braintumnet/multiclass_metrics.py` (324 lines)
- `braintumnet/configs/phase2_a100.yaml` (187 lines)

All numbers are exact as implemented in the code.
