# BrainTumNet Baseline Architecture V1.0

**Documentation Date**: 2025-01-14
**Purpose**: Baseline documentation before IoU 0.90 optimization
**Version**: 1.0 (Pre-Optimization)

---

## 📊 Performance Baseline

### Best Results (Fold 4, Epoch 149/250)

```
Training Date: 2025-10-11
Training Duration: 35h 53m (199 epochs, early stopped)
Best Checkpoint: checkpoints/braintumnet_best_fold4.pth
```

| Metric | Value | Notes |
|--------|-------|-------|
| **Mean IoU** | **0.7263** | Primary target metric |
| **Mean Dice** | **0.8412** | Overall segmentation quality |
| **Val Accuracy** | **1.0000** | Classification task (HGG vs LGG) |
| | | |
| **WT (Whole Tumor)** | | TC + ED combined |
| - WT Dice | 0.8476 | Target: 0.88-0.90 |
| - WT IoU | 0.7356 | Need: +0.16 to reach 0.90 |
| | | |
| **TC (Tumor Core)** | | NCR + ET |
| - TC Dice | 0.8199 | Target: 0.82-0.85 |
| - TC IoU | 0.6948 | Need: +0.21 to reach 0.90 (hardest) |
| | | |
| **ED (Edema)** | | Peritumoral edema |
| - ED Dice | **0.8561** | ✅ Exceeds target (0.75-0.80) |
| - ED IoU | **0.7483** | ✅ Best performing region |
| | | |
| **Training Loss** | 0.1243 | Still decreasing at stop |
| **Learning Rate** | 0.0000 | Hit minimum at ~epoch 100 |
| **Epoch Time** | ~11 min | RTX 3090 / A100 GPU |

### Cross-Validation Status

| Fold | Status | Checkpoint Path | Best Epoch | Notes |
|------|--------|----------------|------------|-------|
| 0 | ✅ Trained | `checkpoints/braintumnet_best_fold0.pth` | ? | Available |
| 1 | ❌ Not trained | - | - | Missing |
| 2 | ✅ Trained | `checkpoints/braintumnet_best_fold2.pth` | ? | Available |
| 3 | ✅ Trained | `checkpoints/braintumnet_best_fold3.pth` | ? | Available |
| 4 | ✅ Trained | `checkpoints/braintumnet_best_fold4.pth` | 149 | Documented above |

**Ensemble Capability**: 4/5 folds available (fold 1 missing)

---

## 🏗️ Model Architecture

### Overview

```
BrainTumNet = U-Net Encoder-Decoder + Transformer Bottleneck + Multi-Task Head

Input: 4-channel MRI (FLAIR, T1, T1CE, T2) @ 256×256
  ↓
[U-Net Encoder] 4 levels with CBAM attention on skip connections
  ↓
[Transformer Bottleneck] Adaptive masked self-attention (8×8 patches)
  ↓
[U-Net Decoder] 4 levels with skip connections + deep supervision
  ↓
[Multi-Task Heads]
  ├─ Segmentation: 3-class (Background, TC, ED) @ 256×256
  └─ Classification: HGG vs LGG (2-class)

Total Parameters: 14.29M (all trainable)
```

### Architecture Components

#### 1. Encoder (SegUNetMasked)

**File**: [src/braintumnet/models/seg_unet.py](../src/braintumnet/models/seg_unet.py)

```python
# Encoder structure
self.e1 = EncoderBlock(in_ch=4,   out_ch=32)   # Level 1: 256×256 → 128×128
self.e2 = EncoderBlock(in_ch=32,  out_ch=64)   # Level 2: 128×128 → 64×64
self.e3 = EncoderBlock(in_ch=64,  out_ch=128)  # Level 3: 64×64 → 32×32
self.e4 = EncoderBlock(in_ch=128, out_ch=256)  # Level 4: 32×32 → 16×16

# Channel progression: [32, 64, 128, 256]
```

**EncoderBlock Details**:
```python
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Two conv blocks (standard U-Net)
        self.block = nn.Sequential(
            conv_bn_relu(in_ch, out_ch),   # Conv → BatchNorm → ReLU
            conv_bn_relu(out_ch, out_ch)   # Conv → BatchNorm → ReLU
        )
        # Downsampling via max pooling
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.block(x)      # Process features
        x_down = self.pool(x)  # Downsample for next level
        return x, x_down       # Return both for skip connection
```

**Basic Convolution Block**:
```python
def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),        # ⚠️ BatchNorm (medical images prefer InstanceNorm)
        nn.ReLU(inplace=True),         # ⚠️ ReLU (SOTA uses LeakyReLU)
    )
```

**Key Characteristics**:
- ✅ Standard U-Net encoder structure
- ⚠️ No residual connections
- ⚠️ BatchNorm instead of InstanceNorm
- ⚠️ MaxPool instead of strided convolution
- ⚠️ ReLU instead of LeakyReLU

---

#### 2. Transformer Bottleneck (AdaptiveMaskedTransformer)

**File**: [src/braintumnet/models/masked_transformer.py](../src/braintumnet/models/masked_transformer.py)

```python
# Bottleneck structure (at 16×16 spatial resolution)
self.bottleneck_conv = conv_bn_relu(256, 256, k=1)  # Channel projection
self.amt = AdaptiveMaskedTransformer(
    in_ch=256,
    dim=256,
    patch_size=8,  # 16×16 → 2×2 patches = 4 tokens
    depth=2,       # 2 transformer blocks
    n_heads=4      # 4 attention heads
)
self.tr_upsample = nn.ConvTranspose2d(256, 256, kernel_size=8, stride=8)
```

**AdaptiveMaskedTransformer Details**:
```python
class AdaptiveMaskedTransformer(nn.Module):
    def __init__(self, in_ch, dim, patch_size=8, depth=2, n_heads=4):
        super().__init__()
        # Patch embedding: Spatial → Tokens
        self.pe = PatchEmbed(in_ch, dim, patch_size)  # 16×16 → 2×2 patches

        # Soft mask generator: Learn which patches matter
        self.mask_gen = SoftMaskGenerator(dim, hidden=dim//2, n_heads=n_heads)

        # Transformer blocks with masked attention
        self.blocks = nn.ModuleList([
            MaskedTransformerBlock(dim, n_heads) for _ in range(depth)
        ])

    def forward(self, x):
        tokens, (H, W) = self.pe(x)         # (B, N, C) where N=H*W patches
        softmask = self.mask_gen(tokens)    # (B, n_heads, N) soft attention mask

        for blk in self.blocks:
            tokens = blk(tokens, softmask)  # Masked self-attention

        # Reshape back to spatial
        feat = tokens.transpose(1,2).reshape(x.size(0), tokens.size(-1), H, W)
        return feat
```

**Soft Masking Innovation**:
```python
class SoftMaskGenerator(nn.Module):
    """Learn which patches are important (tumor vs background)"""
    def forward(self, tokens):  # (B, N, C)
        m = self.mlp(tokens)    # (B, N, n_heads)
        return m.sigmoid()      # Soft weights [0,1] per patch per head
```

**Masked Self-Attention**:
```python
class MaskedSelfAttention(nn.Module):
    def forward(self, x, softmask):
        # Standard Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, n_heads, head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, n_heads, N, head_dim)

        # Compute attention with soft masking
        attn = (q @ k.transpose(-2,-1)) / sqrt(head_dim)  # (B, n_heads, N, N)
        key_bias = torch.log(softmask.unsqueeze(-2) + 1e-6)  # (B, n_heads, 1, N)
        attn = attn + key_bias  # Downweight background patches
        attn = attn.softmax(-1)

        out = (attn @ v).transpose(1,2).reshape(B, N, C)
        return self.proj(out)
```

**Key Characteristics**:
- ✅ Novel adaptive masking (focus on tumor regions)
- ✅ Efficient patch-based attention (4 tokens only)
- ✅ Flash Attention 2 support (on A100)
- ⚠️ Only 2 transformer blocks (SOTA uses 6-12)
- ⚠️ Small patch count (2×2 = 4 tokens)

---

#### 3. Decoder with CBAM Attention

**File**: [src/braintumnet/models/seg_unet.py](../src/braintumnet/models/seg_unet.py)

```python
# Decoder structure (symmetric to encoder)
self.d4 = DecoderBlock(in_ch=256, out_ch=256)  # Level 4: 16×16 → 32×32
self.d3 = DecoderBlock(in_ch=256, out_ch=128)  # Level 3: 32×32 → 64×64
self.d2 = DecoderBlock(in_ch=128, out_ch=64)   # Level 2: 64×64 → 128×128
self.d1 = DecoderBlock(in_ch=64,  out_ch=32)   # Level 1: 128×128 → 256×256

# Output head
self.head = nn.Conv2d(32, num_classes=3, kernel_size=1)  # 3 classes: bg, TC, ED
```

**DecoderBlock with CBAM**:
```python
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Upsampling
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        # CBAM attention on skip connection
        self.cbam = CBAM(out_ch)  # ✅ Channel + Spatial attention

        # Processing after concatenation
        self.block = nn.Sequential(
            conv_bn_relu(out_ch + out_ch, out_ch),  # Concat doubles channels
            conv_bn_relu(out_ch, out_ch)
        )

    def forward(self, x, skip):
        x = self.up(x)                      # Upsample decoder features
        skip = self.cbam(skip)              # ✅ Attend to important skip features
        x = torch.cat([x, skip], dim=1)    # Concatenate
        x = self.block(x)                   # Process
        return x
```

**CBAM Attention Details**:
```python
# File: src/braintumnet/models/cbam.py

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16, k=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)  # Which features?
        self.sa = SpatialAttention(k)                       # Where to look?

    def forward(self, x):
        x = self.ca(x)  # Channel attention
        x = self.sa(x)  # Spatial attention
        return x

class ChannelAttention(nn.Module):
    """Learn which channels (features) are important"""
    def forward(self, x):
        avg_feat = self.avg(x)  # Global average pooling
        max_feat = self.max(x)  # Global max pooling
        att = torch.sigmoid(self.mlp(avg_feat) + self.mlp(max_feat))
        return x * att  # Weight channels

class SpatialAttention(nn.Module):
    """Learn which spatial locations are important"""
    def forward(self, x):
        att = torch.cat([x.mean(1, True), x.max(1, True)[0].unsqueeze(1)], dim=1)
        att = torch.sigmoid(self.conv(att))  # 7×7 conv
        return x * att  # Weight spatial locations
```

**Key Characteristics**:
- ✅ CBAM attention on all skip connections (unique feature)
- ✅ Standard U-Net decoder structure
- ⚠️ No residual connections
- ⚠️ No deep supervision at decoder levels (only auxiliary heads)

---

#### 4. Deep Supervision

```python
# Auxiliary segmentation heads at multiple resolutions
if self.deep_supervision:
    self.aux_head3 = nn.Conv2d(base*4, num_classes, 1)  # After d3: 64×64
    self.aux_head2 = nn.Conv2d(base*2, num_classes, 1)  # After d2: 128×128
    self.aux_head1 = nn.Conv2d(base, num_classes, 1)    # After d1: 256×256
```

**Deep Supervision Forward**:
```python
def forward(self, x):
    # ... encoder + bottleneck ...

    x = self.d4(b, s4)

    x = self.d3(x, s3)
    aux3 = self.aux_head3(x) if self.deep_supervision else None  # 64×64 prediction

    x = self.d2(x, s2)
    aux2 = self.aux_head2(x) if self.deep_supervision else None  # 128×128 prediction

    x = self.d1(x, s1)
    aux1 = self.aux_head1(x) if self.deep_supervision else None  # 256×256 prediction

    seg = self.head(x)  # Main 256×256 prediction

    if self.deep_supervision:
        return seg, [aux3, aux2, aux1]
    return seg
```

**Loss Computation** (inferred from trainer):
```python
# Main output loss + weighted auxiliary losses
loss = main_loss + 0.3*aux3_loss + 0.3*aux2_loss + 0.3*aux1_loss
```

**Key Characteristics**:
- ✅ 3 auxiliary outputs at different scales
- ✅ Improves gradient flow to encoder
- ⚠️ Fixed weights (0.3) for all auxiliary losses

---

#### 5. Multi-Task Learning

**File**: [src/braintumnet/models/braintumnet.py](../src/braintumnet/models/braintumnet.py)

```python
class BrainTumNet(nn.Module):
    def __init__(self, in_ch=1, num_cls=2, base=32, dim=256, ...):
        super().__init__()

        # Segmentation network
        self.seg = SegUNetMasked(
            in_ch=in_ch,
            base=base,
            dim=dim,
            num_classes=num_classes_seg  # 3 classes: bg, TC, ED
        )

        # Channel reduction for classification input
        self.reduce = nn.Conv2d(in_ch, 1, 1, bias=False) if in_ch > 1 else nn.Identity()

        # Classification network (HGG vs LGG)
        self.cls_backbone = TInceptionNet(in_ch=1, num_classes=num_cls)

    def forward(self, x):
        # Segmentation branch
        seg_output = self.seg(x)  # 3-class segmentation

        if self.deep_supervision:
            seg_logits, aux_outputs = seg_output
        else:
            seg_logits = seg_output

        # Compute Whole Tumor mask for ROI
        if self.num_classes_seg == 3:
            seg_prob = torch.softmax(seg_logits, dim=1)  # (B, 3, H, W)
            # WT = TC + ED (sum classes 1 and 2, exclude background 0)
            wt_mask = seg_prob[:, 1:, :, :].sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # ROI-guided classification
        roi_input = self.reduce(x)  # Multi-modal → single channel
        roi = roi_input * wt_mask.detach()  # Mask out non-tumor (no gradient)

        # Classification branch
        cls_logits = self.cls_backbone(roi)  # HGG vs LGG

        if self.deep_supervision:
            return seg_logits, cls_logits, aux_outputs
        return seg_logits, cls_logits
```

**Classification Network (TInceptionNet)**:
```python
# File: src/braintumnet/models/t_inception.py

class TInceptionNet(nn.Module):
    """Inception-style classifier for tumor grading"""
    def __init__(self, in_ch=1, num_classes=2):
        super().__init__()
        self.stem = Conv2d(in_ch, 64, 3) + BatchNorm + ReLU

        # Two inception blocks with multi-scale kernels
        self.b1 = TInceptionBlock(64, 128)   # 1×1, 3×3, 1×3, 3×1 branches
        self.b2 = TInceptionBlock(128, 256)

        self.pool = nn.AdaptiveAvgPool2d(1)   # Global pooling
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)

class TInceptionBlock(nn.Module):
    """Multi-scale feature extraction"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        c = out_ch // 4
        self.b1 = InceptionBranch(in_ch, c, kernel=(1,1))  # Point-wise
        self.b2 = InceptionBranch(in_ch, c, kernel=(3,3))  # Standard
        self.b3 = InceptionBranch(in_ch, c, kernel=(1,3))  # Horizontal
        self.b4 = InceptionBranch(in_ch, c, kernel=(3,1))  # Vertical
        self.fuse = Conv2d(c*4, out_ch, 1) + BatchNorm + ReLU

    def forward(self, x):
        return self.fuse(torch.cat([self.b1(x), self.b2(x),
                                     self.b3(x), self.b4(x)], dim=1))
```

**Multi-Task Loss**:
```python
# File: src/braintumnet/losses/multiclass.py (used in trainer)

class MultiTaskMultiClassLoss(nn.Module):
    def forward(self, seg_logits, seg_mask, cls_logits, cls_label):
        # Segmentation loss (Dice + Focal)
        l_seg = self.seg_loss(seg_logits, seg_mask)

        # Classification loss (CrossEntropy)
        l_cls = self.cls_loss(cls_logits, cls_label)

        # Combined with weights
        total = self.seg_w * l_seg + self.cls_w * l_cls  # seg_w=1.0, cls_w=0.5

        return total, l_seg.detach(), l_cls.detach()
```

**Key Characteristics**:
- ✅ Joint segmentation + classification (unique feature)
- ✅ ROI-guided classification (uses segmentation output)
- ✅ Gradient detachment (segmentation doesn't affect classification)
- ✅ Multi-scale inception features for classification

---

## 📐 Model Specifications

### Hyperparameters (from configs/multiclass.yaml)

```yaml
model:
  in_channels: 4              # FLAIR, T1, T1CE, T2
  num_classes_seg: 3          # Background, TC, ED
  num_classes_cls: 2          # HGG, LGG

  # Encoder/Decoder
  base: 32                    # Channel multiplier [32, 64, 128, 256]

  # Transformer
  dim: 256                    # Transformer embedding dimension
  patch_size: 8               # Patch size for tokenization
  depth: 2                    # Number of transformer blocks
  n_heads: 4                  # Attention heads

  # Training
  roi_stop_grad: true         # Detach seg mask for classification
  deep_supervision: true      # Auxiliary losses at 3 scales
```

### Parameter Count

```
Total Parameters: 14.29M
Trainable Parameters: 14.29M (100%)

Breakdown (estimated):
- Encoder (4 levels): ~2.5M
- Transformer (depth=2): ~3.0M
- Decoder (4 levels): ~2.5M
- CBAM modules (4×): ~0.5M
- Deep supervision heads: ~0.2M
- Classification network: ~5.5M
```

### Model Size

```
Checkpoint file size: ~55 MB (FP32)
Memory usage (training):
  - Batch size 12: ~8-10 GB GPU RAM
  - Batch size 16: ~10-12 GB GPU RAM
  - With AMP (FP16): ~50% reduction
```

---

## 🎓 Training Configuration

### Loss Function

**File**: [src/braintumnet/losses/multiclass.py](../src/braintumnet/losses/multiclass.py)

```yaml
train:
  loss_type: "multiclass_dice_focal"  # Combined Dice + Focal
  seg_loss_weight: 1.0                # Segmentation loss weight
  cls_loss_weight: 0.5                # Classification loss weight

  # Dice loss
  dice_weight: 1.0                    # Weight for Dice component

  # Focal loss
  focal_weight: 1.0                   # Weight for Focal component
  focal_alpha: [0.5, 0.3, 0.2]        # Class weights [bg, TC, ED]
  focal_gamma: 2.0                    # Focusing parameter

  ignore_background: true             # Ignore background in loss computation
  class_weights: null                 # Optional per-class weights (unused)
```

**Loss Components**:

1. **Multi-Class Dice Loss**:
```python
class MultiClassDiceLoss:
    def forward(self, logits, target):
        pred = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=3)

        dice_per_class = []
        for c in [1, 2]:  # TC, ED (skip background)
            intersection = (pred[:, c] * target_onehot[:, c]).sum()
            union = pred[:, c].sum() + target_onehot[:, c].sum()
            dice = (2 * intersection + 1e-6) / (union + 1e-6)
            dice_per_class.append(1.0 - dice)

        return mean(dice_per_class)
```

2. **Multi-Class Focal Loss**:
```python
class MultiClassFocalLoss:
    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        pt = probs[target]  # Probability of true class

        focal_weight = (1 - pt) ** gamma  # gamma=2.0
        ce = -log(pt + 1e-7)
        alpha_t = alpha[target]  # [0.5, 0.3, 0.2]

        loss = alpha_t * focal_weight * ce
        return loss.mean()
```

3. **Combined Loss**:
```python
total_seg_loss = 1.0 * dice_loss + 1.0 * focal_loss

# With deep supervision (3 auxiliary outputs)
main_loss = seg_loss(main_output, target)
aux3_loss = seg_loss(aux3_output, target_64x64)
aux2_loss = seg_loss(aux2_output, target_128x128)
aux1_loss = seg_loss(aux1_output, target_256x256)

total_seg_loss = main_loss + 0.3*aux3_loss + 0.3*aux2_loss + 0.3*aux1_loss

# Multi-task
cls_loss = CrossEntropyLoss(cls_logits, cls_label)
total_loss = 1.0 * total_seg_loss + 0.5 * cls_loss
```

**Key Characteristics**:
- ✅ Combined Dice + Focal (handles class imbalance)
- ✅ Ignores background (focuses on tumor)
- ✅ Weighted focal loss (emphasizes hard examples)
- ⚠️ No IoU loss component
- ⚠️ No boundary loss component

---

### Training Hyperparameters

```yaml
train:
  epochs: 250
  batch_size: 12              # Per GPU

  # Optimizer
  lr: 1.0e-4                  # Initial learning rate
  weight_decay: 1.0e-4        # L2 regularization
  optimizer: Adam             # Default (not AdamW)

  # Learning rate scheduler
  scheduler: "cosine"         # Cosine annealing with warmup
  warmup_steps: 1000          # Linear warmup for 1000 steps
  min_lr: 1.0e-6              # Minimum learning rate

  # Early stopping
  early_stop_patience: 50     # Stop if no improvement for 50 epochs
  early_stop_metric: "val_iou"  # Monitor validation IoU

  # Mixed precision
  amp: true                   # Automatic Mixed Precision (FP16)

  # Workers
  workers: 4                  # DataLoader num_workers
```

**Learning Rate Schedule**:
```python
def cosine_lr_with_warmup(t, T, warmup_steps, base_lr, min_lr):
    if t < warmup_steps:
        # Linear warmup: 0 → base_lr
        lr = base_lr * (t / warmup_steps)
    else:
        # Cosine decay: base_lr → min_lr
        progress = (t - warmup_steps) / (T - warmup_steps)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
    return lr

# Example schedule for 250 epochs:
# Epoch 0-10: 0 → 1e-4 (warmup)
# Epoch 10-250: 1e-4 → 1e-6 (cosine decay)
```

**Observed Issue**: LR hit min_lr ~1e-6 around epoch 100, preventing further optimization for last 99 epochs.

---

### Data Augmentation

**File**: [src/braintumnet/data/transforms.py](../src/braintumnet/data/transforms.py)

```yaml
augment:
  rotate_deg: 20              # Random rotation [-20°, +20°]
  hflip_p: 0.5                # Horizontal flip probability
  vflip_p: 0.5                # Vertical flip probability
```

**Augmentation Pipeline**:
```python
def augment_pair(img, mask, img_size, rotate_deg, hflip_p, vflip_p, train):
    # 1. Resize + pad to square
    img = resize_pad_to_square(img, img_size, is_mask=False)  # BILINEAR
    mask = resize_pad_to_square(mask, img_size, is_mask=True)  # NEAREST

    if train:
        # 2. Random rotation
        angle = random.uniform(-rotate_deg, rotate_deg)
        img = TF.rotate(img, angle)
        mask = TF.rotate(mask, angle)

        # 3. Horizontal flip
        if random.random() < hflip_p:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 4. Vertical flip
        if random.random() < vflip_p:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

    # 5. Convert to tensor [0, 1]
    img_tensor = torch.from_numpy(np.array(img) / 255.0).unsqueeze(0)
    mask_tensor = torch.from_numpy((np.array(mask) > 127).astype(float)).unsqueeze(0)

    return img_tensor, mask_tensor
```

**Key Characteristics**:
- ✅ Basic geometric augmentations
- ✅ Consistent image-mask transformations
- ⚠️ No intensity augmentations (brightness, contrast, gamma)
- ⚠️ No noise augmentation
- ⚠️ No elastic deformation
- ⚠️ Limited rotation range (±20° vs SOTA ±45°)

---

### Dataset Configuration

```yaml
data:
  raw_root: "data/raw"
  proc_root: "data/processed_multiclass"
  modality: "multi"           # All 4 modalities: FLAIR, T1, T1CE, T2
  img_size: 256               # Resize to 256×256
  slices_per_case: 30         # Extract 30 slices per 3D volume
  tumor_slice_ratio: 0.5      # 50% tumor slices, 50% non-tumor
  num_folds: 5                # 5-fold cross-validation
```

**Dataset Statistics**:
- Total cases: 369 (BraTS 2020 training set)
- Total slices: ~11,000 (30 per case)
- Classes: 3 (Background, TC, ED)
- Tumor labels: 2 (HGG, LGG)

**Data Split** (example fold 4):
- Training: ~8,800 slices (~73 cases)
- Validation: ~2,200 slices (~18 cases)

**Preprocessing**:
1. Load 3D MRI volumes from HDF5
2. Select 30 slices per case (15 with tumor, 15 without)
3. Resize to 256×256 with padding
4. Save as PNG (separate files for each modality)
5. Create fold split CSV files

---

## 💾 Codebase Structure

```
braintumnet/
├── configs/
│   ├── multiclass.yaml              # ⭐ Main config (used for baseline)
│   ├── multiclass_a100.yaml         # A100 optimized (batch_size=64)
│   └── improved_v4_focal.yaml       # Experimental
│
├── src/braintumnet/
│   ├── models/
│   │   ├── braintumnet.py           # ⭐ Main model (multi-task)
│   │   ├── seg_unet.py              # ⭐ U-Net encoder-decoder
│   │   ├── masked_transformer.py    # ⭐ Transformer bottleneck
│   │   ├── cbam.py                  # ⭐ CBAM attention
│   │   └── t_inception.py           # Classification network
│   │
│   ├── data/
│   │   ├── brats2020_dataset.py     # Dataset loader
│   │   ├── transforms.py            # Augmentation
│   │   └── preprocessing.py         # Data preprocessing
│   │
│   ├── engine/
│   │   ├── trainer.py               # ⭐ Training loop
│   │   └── evaluator.py             # Evaluation
│   │
│   ├── losses/multiclass.py         # ⭐ Loss functions
│   ├── metrics_multiclass.py        # ⭐ Metrics computation
│   └── metrics/multiclass.py        # Additional metrics
│
├── scripts/
│   ├── train.py                     # ⭐ Training script
│   ├── evaluate.py                  # Evaluation script
│   ├── predict.py                   # Inference script
│   ├── preprocess_h5_to_multiclass.py  # Preprocessing
│   └── create_fold_splits.py        # Create CV folds
│
├── checkpoints/                     # ⭐ Saved models
│   ├── braintumnet_best_fold0.pth   # Best fold 0 checkpoint
│   ├── braintumnet_best_fold2.pth   # Best fold 2 checkpoint
│   ├── braintumnet_best_fold3.pth   # Best fold 3 checkpoint
│   ├── braintumnet_best_fold4.pth   # ⭐ Best fold 4 (documented)
│   └── last_fold*.pth               # Last epoch checkpoints
│
├── logs/                            # Training logs
│   ├── braintumnet_multiclass_3class_fold4_*.log  # ⭐ Fold 4 log
│   ├── metrics_*.csv                # Metrics per epoch
│   └── config_fold*.yaml            # Saved configs
│
├── runs/                            # TensorBoard logs
│   └── braintumnet_multiclass_3class_fold*/
│
├── data/
│   ├── processed_multiclass/        # ⭐ Preprocessed data
│   │   ├── flair/                   # FLAIR modality PNGs
│   │   ├── t1/                      # T1 modality PNGs
│   │   ├── t1ce/                    # T1CE modality PNGs
│   │   ├── t2/                      # T2 modality PNGs
│   │   ├── seg/                     # Segmentation masks
│   │   ├── train_fold*.csv          # Training splits
│   │   ├── val_fold*.csv            # Validation splits
│   │   ├── labels.csv               # Case-level labels (HGG/LGG)
│   │   └── mapping.csv              # Slice-to-case mapping
│   └── raw/                         # Original BraTS2020 HDF5 files
│
└── docs/                            # Documentation
    ├── BASELINE_ARCHITECTURE_V1.md  # ⭐ This file
    ├── COMPARISON_BRAINTUMNET_VS_SOTA.md
    ├── ROADMAP_TO_IOU_090.md
    └── ...
```

---

## 🧪 Training Procedure

### Command

```bash
cd braintumnet
python scripts/train.py --cfg configs/multiclass.yaml --fold 4
```

### Training Loop (simplified)

```python
# From src/braintumnet/engine/trainer.py

for epoch in range(250):
    # 1. Training phase
    model.train()
    for batch in train_loader:
        images, masks, labels = batch  # (B, 4, 256, 256), (B, 1, 256, 256), (B,)

        # Forward pass
        seg_logits, cls_logits, aux_outputs = model(images)

        # Compute loss
        main_loss = seg_loss(seg_logits, masks)
        aux_losses = [seg_loss(aux, masks) for aux in aux_outputs]
        total_seg_loss = main_loss + 0.3*sum(aux_losses)
        cls_loss = CrossEntropyLoss(cls_logits, labels)
        total_loss = 1.0*total_seg_loss + 0.5*cls_loss

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()  # AMP
        scaler.step(optimizer)
        scaler.update()

        # Update LR
        cosine_lr_with_warmup(optimizer, step, total_steps)

    # 2. Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images, masks, labels = batch
            seg_logits, cls_logits = model(images)

            # Compute metrics
            seg_pred = torch.argmax(seg_logits, dim=1)  # (B, H, W)

            # Per-region metrics
            wt_dice, wt_iou = compute_metrics(seg_pred, masks, classes=[1,2])  # TC+ED
            tc_dice, tc_iou = compute_metrics(seg_pred, masks, classes=[1])    # TC only
            ed_dice, ed_iou = compute_metrics(seg_pred, masks, classes=[2])    # ED only

            mean_dice = (wt_dice + tc_dice + ed_dice) / 3
            mean_iou = (wt_iou + tc_iou + ed_iou) / 3

    # 3. Logging
    print(f"Epoch {epoch}: WT_dice={wt_dice:.4f}, TC_dice={tc_dice:.4f}, ED_dice={ed_dice:.4f}")

    # 4. Checkpointing
    if mean_iou > best_iou:
        save_checkpoint(f"checkpoints/braintumnet_best_fold{fold}.pth")
        best_iou = mean_iou
        patience_counter = 0
    else:
        patience_counter += 1

    # 5. Early stopping
    if patience_counter >= 50:
        print(f"Early stopping at epoch {epoch}")
        break
```

### Metrics Computation

```python
def compute_metrics(pred, target, classes):
    """
    pred: (B, H, W) class indices {0, 1, 2}
    target: (B, 1, H, W) class indices {0, 1, 2}
    classes: list of classes to include (e.g., [1] for TC only, [1,2] for WT)
    """
    # Create binary mask for specified classes
    pred_mask = torch.isin(pred, torch.tensor(classes))
    target_mask = torch.isin(target, torch.tensor(classes))

    # Compute intersection and union
    intersection = (pred_mask & target_mask).sum()
    union = pred_mask.sum() + target_mask.sum()

    # Dice coefficient
    dice = (2.0 * intersection) / (union + 1e-6)

    # IoU
    iou = intersection / (union - intersection + 1e-6)

    return dice.item(), iou.item()
```

---

## 📊 Results Summary Table

| Metric | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Std |
|--------|--------|--------|--------|--------|--------|------|-----|
| **Mean IoU** | ? | - | ? | ? | 0.7263 | - | - |
| **Mean Dice** | ? | - | ? | ? | 0.8412 | - | - |
| **WT Dice** | ? | - | ? | ? | 0.8476 | - | - |
| **TC Dice** | ? | - | ? | ? | 0.8199 | - | - |
| **ED Dice** | ? | - | ? | ? | 0.8561 | - | - |
| **WT IoU** | ? | - | ? | ? | 0.7356 | - | - |
| **TC IoU** | ? | - | ? | ? | 0.6948 | - | - |
| **ED IoU** | ? | - | ? | ? | 0.7483 | - | - |
| **Val Acc** | ? | - | ? | ? | 1.0000 | - | - |

**Note**: Only fold 4 has been fully documented. Other folds trained but metrics not extracted.

---

## 🎯 Key Strengths

### 1. Novel Architectural Features ✅
- **Adaptive Masked Transformer**: Learns to focus on tumor-relevant patches
- **CBAM Attention on Skip Connections**: Better than standard U-Net
- **Multi-Task Learning**: Joint segmentation + classification
- **ROI-Guided Classification**: Uses segmentation to guide grading

### 2. Well-Implemented Components ✅
- **Deep Supervision**: Multi-scale auxiliary losses
- **Combined Dice + Focal Loss**: Handles class imbalance
- **Mixed Precision Training**: Efficient memory usage
- **Comprehensive Logging**: TensorBoard + CSV metrics

### 3. Strong Performance ✅
- **ED Segmentation**: Exceeds target (0.8561 vs 0.75-0.80)
- **Classification**: Perfect validation accuracy (1.0)
- **Training Stability**: Smooth convergence, no collapse

---

## ⚠️ Known Limitations

### 1. Architecture Gaps
- ❌ No residual connections (SOTA standard)
- ❌ BatchNorm instead of InstanceNorm (medical imaging standard)
- ❌ ReLU instead of LeakyReLU (SOTA standard)
- ❌ MaxPool instead of strided conv (less learnable)
- ❌ Small model (14M vs SOTA 30-60M)

### 2. Training Issues
- ❌ LR hit minimum too early (epoch ~100)
- ❌ Limited augmentation (no intensity, elastic, noise)
- ❌ No IoU loss component (optimizing wrong metric)
- ❌ No boundary loss (boundary errors costly for IoU)

### 3. Inference Gaps
- ❌ No test-time augmentation (standard in SOTA)
- ❌ No ensemble inference (standard in SOTA)
- ❌ No post-processing (CRF, morphological ops)

### 4. Dataset Limitations
- ❌ 2D slices instead of 3D volumes (loses spatial context)
- ❌ Only 30 slices per case (vs full 155 slices)
- ❌ Single resolution (256×256, no multi-scale)

---

## 📝 Baseline Snapshot

### Git Hash (if available)
```bash
git log -1 --format="%H"
# (Document current commit before changes)
```

### Environment
```
Python: 3.8+
PyTorch: 2.0+
CUDA: 11.8+
GPU: RTX 3090 / A100
OS: Windows 11
```

### Key Dependencies
```
torch>=2.0.0
torchvision
pillow
pandas
h5py
pyyaml
tensorboard
tqdm
scikit-learn
```

---

## 🎓 Usage Examples

### 1. Load Baseline Model

```python
import torch
from braintumnet.models.braintumnet import BrainTumNet

# Load model
model = BrainTumNet(
    in_ch=4,
    num_cls=2,
    base=32,
    dim=256,
    patch=8,
    depth=2,
    n_heads=4,
    roi_stop_grad=True,
    deep_supervision=True,
    num_classes_seg=3
)

# Load checkpoint
checkpoint = torch.load('checkpoints/braintumnet_best_fold4.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Best IoU: {checkpoint['best_iou']:.4f}")
```

### 2. Inference

```python
# Prepare input (4-channel MRI)
import torch
from PIL import Image
import numpy as np

# Load 4 modalities
flair = np.array(Image.open('data/processed_multiclass/flair/case_001_slice_050.png'))
t1 = np.array(Image.open('data/processed_multiclass/t1/case_001_slice_050.png'))
t1ce = np.array(Image.open('data/processed_multiclass/t1ce/case_001_slice_050.png'))
t2 = np.array(Image.open('data/processed_multiclass/t2/case_001_slice_050.png'))

# Stack and normalize
img = np.stack([flair, t1, t1ce, t2], axis=0).astype(np.float32) / 255.0
img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 4, 256, 256)

# Predict
with torch.no_grad():
    seg_logits, cls_logits = model(img_tensor.cuda())

    # Segmentation
    seg_pred = torch.argmax(seg_logits, dim=1)  # (1, 256, 256)
    seg_pred = seg_pred.cpu().numpy()[0]

    # Classification
    cls_pred = torch.argmax(cls_logits, dim=1)  # 0=LGG, 1=HGG
    cls_prob = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]

print(f"Segmentation: {np.unique(seg_pred)}")  # [0, 1, 2] = bg, TC, ED
print(f"Grade: {'HGG' if cls_pred == 1 else 'LGG'} (prob: {cls_prob[cls_pred]:.2f})")
```

### 3. Reproduce Training

```bash
# Full reproduction from scratch

# 1. Preprocess data
python scripts/preprocessing/preprocess_h5_to_multiclass.py \
    --h5_dir "path/to/brats2020/h5" \
    --out_dir "data/processed_multiclass" \
    --img_size 256 \
    --num_folds 5

# 2. Train fold 4 (baseline)
python scripts/train.py \
    --cfg configs/multiclass.yaml \
    --fold 4

# 3. Monitor training
tensorboard --logdir=runs

# 4. Evaluate
python scripts/evaluate.py \
    --checkpoint checkpoints/braintumnet_best_fold4.pth \
    --fold 4
```

---

## 📌 Change Log

### Version 1.0 (2025-01-14) - Baseline Documentation
- Initial baseline documentation before IoU 0.90 optimization
- Documented fold 4 best results (IoU 0.7263)
- Complete architecture specification
- Training configuration captured
- Ready for comparison with upgraded version

---

## 📚 Related Documentation

- [COMPARISON_BRAINTUMNET_VS_SOTA.md](COMPARISON_BRAINTUMNET_VS_SOTA.md) - Detailed comparison with nnUNet and Swin-UNETR
- [ROADMAP_TO_IOU_090.md](ROADMAP_TO_IOU_090.md) - Upgrade plan to reach IoU 0.90
- [FOLD4_ANALYSIS_AND_SUGGESTIONS.md](FOLD4_ANALYSIS_AND_SUGGESTIONS.md) - Performance analysis
- [COMPARISON_PATCHBASED_VS_BRAINTUMNET.md](COMPARISON_PATCHBASED_VS_BRAINTUMNET.md) - Comparison with traditional methods

---

**This baseline documentation captures BrainTumNet V1.0 before any IoU 0.90 optimization changes.**

**Checksum**: 14.29M parameters | IoU 0.7263 | Dice 0.8412 | 4/5 folds trained
