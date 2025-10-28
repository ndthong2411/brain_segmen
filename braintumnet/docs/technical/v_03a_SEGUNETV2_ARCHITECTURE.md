# Phần 3a: SegUNetV2 Architecture - Phase 2 Improvements

⭐ **NEW**: Tài liệu này giải thích chi tiết **SegUNetV2**, phiên bản cải tiến của U-Net segmentation model với các nâng cấp Phase 2.

**File code**: `src/braintumnet/models/seg_unet_v2.py` (322 dòng)

---

## Mục Lục

1. [Tổng Quan V2 Improvements](#tổng-quan-v2-improvements)
2. [Enhanced Conv Block: conv_norm_act](#enhanced-conv-block-conv_norm_act)
3. [Residual Convolutional Blocks](#residual-convolutional-blocks)
4. [Enhanced Encoder Block](#enhanced-encoder-block)
5. [Enhanced Decoder Block](#enhanced-decoder-block)
6. [Multi-Scale Fusion Module](#multi-scale-fusion-module)
7. [SegUNetV2 Complete Architecture](#segunetv2-complete-architecture)
8. [Forward Pass với Deep Supervision](#forward-pass-với-deep-supervision)
9. [Model Configurations](#model-configurations)
10. [So Sánh V1 vs V2](#so-sánh-v1-vs-v2)

---

## Tổng Quan V2 Improvements

### Động Lực

**Tại sao cần V2?**
- V1 đạt Dice 0.9148 trên binary segmentation - rất tốt!
- Nhưng multi-class segmentation phức tạp hơn (phân biệt TC vs ED)
- Cần model capacity lớn hơn
- Cần regularization tốt hơn để tránh overfitting

### 7 Cải Tiến Chính

```
V1 → V2 Improvements:

1. BatchNorm       → InstanceNorm     (medical imaging standard)
2. ReLU            → LeakyReLU        (better gradients, slope=0.01)
3. No residuals    → Residual blocks  (deeper network training)
4. MaxPool         → Strided conv     (learned downsampling)
5. Single scale    → Multi-scale      (fuse features from all decoder levels)
6. No DS           → Deep supervision (auxiliary losses at d3, d2, d1)
7. No dropout      → Dropout 0.15     (regularization for large models)
```

### Architecture Comparison Diagram

**V1 Architecture**:
```
Input
  ↓ [EncoderBlock]
  ↓ MaxPool (fixed)
  ...
  ↓ [Bottleneck Transformer]
  ↓ [DecoderBlock]
  ↓ Upsample
  ...
  ↓ [1×1 Conv Head]
Output (1 or 3 classes)
```

**V2 Architecture**:
```
Input
  ↓ [ResidualConvBlock]
  ↓ Strided Conv (learned) ⭐
  ...
  ↓ [Bottleneck Transformer]
  ↓ [ResidualConvBlock + CBAM] ⭐
  ↓ Upsample
  ├→ Aux Head 3 (deep supervision) ⭐
  ...
  ├→ Multi-Scale Fusion ⭐
  ↓ [1×1 Conv Head]
Output (1 or 3 classes)
```

---

## Enhanced Conv Block: conv_norm_act

**V1**:
```python
def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )
```

**V2** ⭐:
```python
def conv_norm_act(in_ch, out_ch, k=3, s=1, p=1, norm='instance', dropout=0.0):
    """
    Improved convolution block: Conv + Norm + LeakyReLU + Dropout

    Args:
        norm: 'instance' (medical imaging), 'batch', or 'group'
        dropout: dropout probability (0.0 = no dropout)
    """
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)]

    # Normalization
    if norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    elif norm == 'batch':
        layers.append(nn.BatchNorm2d(out_ch))
    elif norm == 'group':
        num_groups = min(32, out_ch // 4)  # Adaptive group size
        layers.append(nn.GroupNorm(num_groups, out_ch))

    # Activation
    layers.append(nn.LeakyReLU(0.01, inplace=True))  # slope=0.01 (nnUNet style)

    # Dropout
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))

    return nn.Sequential(*layers)
```

**Dòng 24-51**: Enhanced convolution block với 4 improvements

### Improvements Explained

#### 1. InstanceNorm vs BatchNorm

**BatchNorm (V1)**:
- Normalize across batch dimension: E[x_batch]
- **Vấn đề**: Medical imaging thường có batch size nhỏ (2-8)
- Batch size nhỏ → statistics không ổn định
- Khác nhau giữa training vs inference (batch statistics vs running statistics)

**InstanceNorm (V2)**:
```python
# BatchNorm: normalize across (B, H, W)
mean = x.mean(dim=(0, 2, 3))  # Shape: (C,)
var = x.var(dim=(0, 2, 3))

# InstanceNorm: normalize across (H, W) cho mỗi sample
mean = x.mean(dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)
var = x.var(dim=(2, 3), keepdim=True)
```

**Ưu điểm**:
- ✅ Không phụ thuộc vào batch size
- ✅ Training == Inference (no running stats)
- ✅ Standard trong medical imaging (nnUNet, MedicalNet)
- ✅ Tốt hơn với augmentation (contrast changes)

#### 2. LeakyReLU vs ReLU

**ReLU (V1)**:
```python
ReLU(x) = max(0, x)
```
- Gradient = 0 khi x < 0 → **dying ReLU problem**

**LeakyReLU (V2)**:
```python
LeakyReLU(x) = max(0.01x, x)  # slope=0.01
```
- Gradient = 0.01 khi x < 0 → không bao giờ "chết"

**Tại sao slope=0.01?**
- nnUNet sử dụng 0.01 (medical imaging sota)
- 0.01 đủ nhỏ để không làm nhiễu activations
- 0.01 đủ lớn để gradients flow

**Benefit**:
```
Training deep networks:
V1 (ReLU):      20% neurons die → slow convergence
V2 (LeakyReLU): All neurons alive → faster, stable training
```

#### 3. Flexible Normalization

V2 hỗ trợ 3 loại normalization:

**InstanceNorm** (default):
- Best cho medical imaging
- Không phụ thuộc batch size
- Per-sample statistics

**BatchNorm**:
- Cho compatibility với V1
- Tốt nếu batch size lớn (>16)

**GroupNorm**:
- Middle ground giữa Instance và Batch
- Tốt cho batch size trung bình (8-16)
- Chia channels thành groups, normalize mỗi group

#### 4. Dropout Regularization

**Không có dropout (V1)**:
- Model nhỏ (~14M params) → không cần
- Binary segmentation → đơn giản hơn

**Có dropout (V2)**:
```python
if dropout > 0:
    layers.append(nn.Dropout2d(dropout))
```
- Model lớn (~35-60M params) → dễ overfit
- Multi-class → phức tạp hơn
- Dropout2d: drop toàn bộ feature maps (not individual pixels)

**Usage**:
- Dropout 0.0: Shallow layers (e1, e2)
- Dropout 0.15: Deep layers (e3, e4)
- Dropout 0.075: Middle decoder layers (d2)
- Dropout 0.0: Output layers (d1)

**Tại sao không uniform dropout?**
- Shallow features: low-level (edges, textures) → cần stable
- Deep features: high-level (semantics) → có thể regularize

---

## Residual Convolutional Blocks

V2 thay thế tất cả conv blocks bằng **residual blocks**:

```python
class ResidualConvBlock(nn.Module):
    """
    Residual convolutional block with InstanceNorm and LeakyReLU

    Structure: Conv-Norm-Act -> Conv-Norm -> Add-Act
    """
    def __init__(self, in_ch, out_ch, norm='instance', dropout=0.0):
        super().__init__()
        self.conv1 = conv_norm_act(in_ch, out_ch, norm=norm, dropout=dropout)
        # Second conv without activation (will be applied after residual add)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True) if norm == 'instance' else nn.BatchNorm2d(out_ch)
        )

        # Residual connection: 1×1 conv if channel mismatch
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        identity = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity  # Residual addition
        out = self.act(out)
        return out
```

**Dòng 53-80**: Residual convolutional block

### Why Residual Connections?

**Without Residuals (V1)**:
```
x → Conv → BN → ReLU → Conv → BN → ReLU → out
```
- Deep networks: gradients vanish
- Hard to train >20 layers

**With Residuals (V2)**:
```
x → Conv → BN → ReLU → Conv → BN → (+) → ReLU → out
↓                                      ↑
└──────────────────────────────────────┘ (identity/projection)
```

**Forward**:
```python
out = F(x) + x  # F(x) = learned residual, x = identity
```

**Backward**:
```python
∂loss/∂x = ∂loss/∂out * (∂F/∂x + 1)
#                          ↑     ↑
#                      learned  identity
```

**Key insight**: `+1` ensures gradient always flows!

### Residual Projection

**Case 1: Matching channels** (in_ch == out_ch):
```python
self.residual = nn.Identity()
out = F(x) + x  # Direct addition
```

**Case 2: Mismatched channels** (in_ch != out_ch):
```python
self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False)
out = F(x) + self.residual(x)  # 1×1 conv projects to matching dims
```

**Example**:
```python
# Encoder 1: 4 → 48 channels
block = ResidualConvBlock(4, 48)
# residual = Conv2d(4, 48, 1×1) projects input to 48 channels
```

### Benefit: Training Deep Networks

**V1 (No residuals)**:
```
Epoch 1-5:   Loss decreases slowly
Epoch 10:    Convergence slows
Epoch 20+:   Plateau, no improvement
```

**V2 (With residuals)**:
```
Epoch 1-5:   Loss decreases rapidly ⭐
Epoch 10:    Still improving
Epoch 20+:   Smooth convergence to better minimum
```

**ResNets showed**: Residuals enable training networks 100+ layers deep!

---

## Enhanced Encoder Block

**V1 Encoder**:
```python
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(conv_bn_relu(in_ch, out_ch), conv_bn_relu(out_ch, out_ch))
        self.pool = nn.MaxPool2d(2)  # Fixed downsampling

    def forward(self, x):
        x = self.block(x)
        return x, self.pool(x)
```

**V2 Encoder** ⭐:
```python
class EncoderBlock(nn.Module):
    """
    Encoder block with residual convolutions and strided conv downsampling

    Improvements:
    - Residual connections
    - Strided conv instead of MaxPool (learned downsampling)
    """
    def __init__(self, in_ch, out_ch, norm='instance', dropout=0.0):
        super().__init__()
        self.block = ResidualConvBlock(in_ch, out_ch, norm=norm, dropout=dropout)
        # Strided convolution for downsampling (learnable, better than MaxPool)
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.block(x)
        x_down = self.downsample(x)
        return x, x_down
```

**Dòng 82-100**: Enhanced encoder block

### Strided Conv vs MaxPool

**MaxPool (V1)**:
```python
nn.MaxPool2d(kernel_size=2, stride=2)
```
- **Fixed operation**: Always picks max in 2×2 window
- **Not learnable**: No parameters
- **Information loss**: Throws away 75% of activations

**Strided Conv (V2)**:
```python
nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
```
- **Learned operation**: Model learns best downsampling
- **Learnable**: Has parameters (weights)
- **Information preserved**: Weighted combination of all pixels

**Visual Comparison**:
```
Input 4×4:
┌─────┬─────┐
│ 1 2 │ 3 4 │
│ 5 6 │ 7 8 │
├─────┼─────┤
│ 9 A │ B C │
│ D E │ F G │
└─────┴─────┘

MaxPool 2×2 Output:
┌─────┬─────┐
│  6  │  8  │  ← max(1,2,5,6), max(3,4,7,8)
├─────┼─────┤
│  E  │  G  │  ← max(9,A,D,E), max(B,C,F,G)
└─────┴─────┘
(Fixed, no learning)

Strided Conv 3×3 Output:
┌────────┬────────┐
│ w₁·1 + │ w₁·3 + │
│ w₂·2 + │ w₂·4 + │
│ w₃·5 + │ w₃·7 + │  ← Learned weights
│ w₄·6 + │ w₄·8 + │
│ ...    │ ...    │
└────────┴────────┘
(Learned combination)
```

**Benefit**: Strided conv adapts downsampling to task!

### Example Usage

```python
encoder = EncoderBlock(48, 96, dropout=0.15)

x = torch.randn(4, 48, 128, 128)
skip, down = encoder(x)

print(skip.shape)  # (4, 96, 128, 128) - for skip connection
print(down.shape)  # (4, 96, 64, 64) - downsampled 2×
```

---

## Enhanced Decoder Block

**V1 Decoder**:
```python
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, 2)
        self.cbam = CBAM(in_ch)
        self.block = nn.Sequential(conv_bn_relu(in_ch*2, out_ch), conv_bn_relu(out_ch, out_ch))

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.cbam(skip)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)
```

**V2 Decoder** ⭐:
```python
class DecoderBlock(nn.Module):
    """
    Decoder block with residual convolutions and CBAM attention

    Improvements:
    - Residual connections
    - CBAM attention on skip connections
    - Dropout for regularization
    """
    def __init__(self, in_ch, out_ch, norm='instance', dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.cbam = CBAM(out_ch)
        self.block = ResidualConvBlock(out_ch * 2, out_ch, norm=norm, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, self.cbam(skip)], dim=1)
        x = self.block(x)
        return x
```

**Dòng 102-122**: Enhanced decoder block

### Key Changes

**1. Upsample first, then concat**:
```python
# V2: Cleaner
x = self.up(x)           # (in_ch, H, W) → (out_ch, 2H, 2W)
x = torch.cat([x, self.cbam(skip)], dim=1)  # (out_ch*2, 2H, 2W)

# V1: Less efficient
x = self.up(x)           # (in_ch, H, W) → (in_ch, 2H, 2W)
skip = self.cbam(in_ch)  # Must match input channels
x = torch.cat([x, skip], dim=1)  # (in_ch*2, 2H, 2W)
```

**2. ResidualConvBlock**:
- V1: Two separate convs
- V2: Single Residual block (better training)

**3. Dropout in decoder**:
```python
# Dropout strategy (Phase 2 Large):
d4 = DecoderBlock(..., dropout=0.15)  # Deepest → most regularization
d3 = DecoderBlock(..., dropout=0.15)
d2 = DecoderBlock(..., dropout=0.075) # Reduce dropout
d1 = DecoderBlock(..., dropout=0.0)   # No dropout near output
```

---

## Multi-Scale Fusion Module

⭐ **NEW in V2**: Fuse features từ nhiều decoder levels trước final head.

### Motivation

**Problem**: Final segmentation chỉ dùng d1 features
```
d4 (16×16) → d3 (32×32) → d2 (64×64) → d1 (128×128) → HEAD
                                           ↑
                                        Only this!
```

**Better**: Combine features từ tất cả levels
```
d4 (16×16) ──┐
d3 (32×32) ──┼→ [Fusion] → Fused Features (128×128)
d2 (64×64) ──┤                     ↓
d1 (128×128)─┘                   [HEAD]
```

**Benefit**: Multi-scale information!
- d4: High-level semantics (tumor vs no tumor)
- d3: Mid-level structures (tumor boundaries)
- d2: Low-level details (fine edges)
- d1: Spatial details (precise localization)

### Implementation

```python
class MultiScaleFusion(nn.Module):
    """
    Multi-scale feature fusion module

    Fuses features from multiple decoder levels to capture
    both fine-grained and coarse information.
    """
    def __init__(self, channels_list, out_channels):
        """
        Args:
            channels_list: List of channel dimensions [d1_ch, d2_ch, d3_ch, d4_ch]
            out_channels: Output channel dimension
        """
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1, bias=False) for ch in channels_list
        ])
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, features):
        """
        Args:
            features: List [d1, d2, d3, d4] with different spatial sizes
        Returns:
            fused: (B, out_channels, H, W) fused features
        """
        target_size = features[0].shape[2:]  # Use largest spatial size (d1)

        upsampled = []
        for i, feat in enumerate(features):
            # Project to same channel dimension
            feat = self.convs[i](feat)
            # Upsample to target size if needed
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled.append(feat)

        # Fuse by summation
        fused = sum(upsampled)
        fused = self.norm(fused)
        fused = self.act(fused)
        return fused
```

**Dòng 124-167**: Multi-scale fusion module

### Forward Pass Example

```python
# Decoder outputs (phase 2 small: base=48)
d1 = torch.randn(4, 48, 256, 256)   # base
d2 = torch.randn(4, 96, 128, 128)   # base*2
d3 = torch.randn(4, 192, 64, 64)    # base*4
d4 = torch.randn(4, 384, 32, 32)    # base*8

# Multi-scale fusion
fusion = MultiScaleFusion(
    channels_list=[48, 96, 192, 384],
    out_channels=48
)

decoder_features = [d1, d2, d3, d4]
fused = fusion(decoder_features)

print(fused.shape)  # (4, 48, 256, 256) - all scales combined!
```

### Fusion Process

**Step by step**:
```
1. Project all to same channels (48):
   d1: (B, 48, 256, 256) → conv1×1 → (B, 48, 256, 256)
   d2: (B, 96, 128, 128) → conv1×1 → (B, 48, 128, 128)
   d3: (B, 192, 64, 64)  → conv1×1 → (B, 48, 64, 64)
   d4: (B, 384, 32, 32)  → conv1×1 → (B, 48, 32, 32)

2. Upsample all to target size (256×256):
   d1: (B, 48, 256, 256) → no change
   d2: (B, 48, 128, 128) → bilinear → (B, 48, 256, 256)
   d3: (B, 48, 64, 64)   → bilinear → (B, 48, 256, 256)
   d4: (B, 48, 32, 32)   → bilinear → (B, 48, 256, 256)

3. Sum all features:
   fused = d1 + d2_up + d3_up + d4_up  # (B, 48, 256, 256)

4. Normalize and activate:
   fused = LeakyReLU(InstanceNorm(fused))
```

**Why summation not concatenation?**
- Concat: (B, 48*4=192, 256, 256) → too many channels
- Sum: (B, 48, 256, 256) → efficient, equally weights all scales

---

## Deep Supervision

⭐ **NEW in V2**: Auxiliary outputs at intermediate decoder levels.

### Motivation

**Problem with single output**:
```
Loss only backprops through final head
→ Gradients weaken as they flow backwards
→ Early layers learn slowly
```

**Solution: Deep Supervision**:
```
Main Output (d1) → Loss 1 (weight=1.0)
Aux Output 1 (d2) → Loss 2 (weight=0.5)
Aux Output 2 (d3) → Loss 3 (weight=0.25)
Aux Output 3 (d4) → Loss 4 (weight=0.125)

Total Loss = Loss1 + 0.5*Loss2 + 0.25*Loss3 + 0.125*Loss4
```

**Benefit**: Direct supervision cho intermediate layers!

### Implementation in V2

```python
# In SegUNetV2.__init__():
if self.deep_supervision:
    self.aux_head3 = nn.Conv2d(base*4, num_classes, 1)  # From d3
    self.aux_head2 = nn.Conv2d(base*2, num_classes, 1)  # From d2
    self.aux_head1 = nn.Conv2d(base, num_classes, 1)    # From d1

# In forward():
d3 = self.d3(d4, s3)
aux3 = self.aux_head3(d3) if self.deep_supervision else None

d2 = self.d2(d3, s2)
aux2 = self.aux_head2(d2) if self.deep_supervision else None

d1 = self.d1(d2, s1)
aux1 = self.aux_head1(d1) if self.deep_supervision else None

# Final
seg = self.head(final_features)

if self.deep_supervision:
    return seg, [aux3, aux2, aux1]
return seg
```

### Loss Computation

```python
# In training loop:
if deep_supervision:
    seg_logits, aux_outputs = model.seg(img)

    # Main loss
    seg_loss = criterion(seg_logits, mask)

    # Auxiliary losses with decreasing weights
    weights = [0.5, 0.25, 0.125]
    for i, aux in enumerate(aux_outputs):
        # Resize aux to match mask size
        aux_resized = F.interpolate(aux, size=mask.shape[-2:], mode='bilinear')
        seg_loss += weights[i] * criterion(aux_resized, mask)
```

### Why Decreasing Weights?

```
Main output (full resolution):    weight = 1.0   ← Most important
Aux1 (d1, 256×256):                weight = 0.5
Aux2 (d2, 128×128):                weight = 0.25
Aux3 (d3, 64×64):                  weight = 0.125 ← Least important
```

**Reasoning**:
- Main output: Full resolution, most accurate
- Auxiliary outputs: Lower resolution, less accurate
- Weight decay: Balance contribution fairly

---

## SegUNetV2 Complete Architecture

Bây giờ đi qua toàn bộ model:

```python
class SegUNetV2(nn.Module):
    def __init__(self, in_ch=4, base=48, dim=384, patch=8, depth=4, n_heads=8,
                 num_classes=3, dropout=0.15, norm='instance',
                 deep_supervision=True, multi_scale_fusion=True):
        super().__init__()
        # ... (initialization code)
```

**Dòng 169-197**: Main model class

### Constructor Parameters

**Core parameters**:
- `in_ch=4`: Input channels (4 MRI modalities)
- `base=48`: Base channels (V1=32, V2 Small=48, V2 Large=64)
- `dim=384`: Transformer dimension (V1=256, V2 Small=384, V2 Large=512)

**Transformer parameters**:
- `patch=8`: Patch size for transformer
- `depth=4`: Transformer blocks (V1=2, V2=4)
- `n_heads=8`: Attention heads (V1=4, V2=8)

**Segmentation parameters**:
- `num_classes=3`: Output classes (1=binary, 3=multi-class)
- `dropout=0.15`: Dropout probability (0.0-0.2)
- `norm='instance'`: Normalization type

**V2 features**:
- `deep_supervision=True`: Use auxiliary losses
- `multi_scale_fusion=True`: Fuse decoder features

### Full Forward Pass

```python
def forward(self, x):
    # Encoder
    s1, x1 = self.e1(x)      # base, H, W
    s2, x2 = self.e2(x1)     # base*2, H/2, W/2
    s3, x3 = self.e3(x2)     # base*4, H/4, W/4
    s4, x4 = self.e4(x3)     # base*8, H/8, W/8

    # Transformer bottleneck
    b = self.bottleneck_conv(x4)
    b = self.amt(b)
    b = self.tr_upsample(b)

    # Decoder
    d4 = self.d4(b, s4)      # base*8, H/8, W/8

    d3 = self.d3(d4, s3)     # base*4, H/4, W/4
    aux3 = self.aux_head3(d3) if self.deep_supervision else None

    d2 = self.d2(d3, s2)     # base*2, H/2, W/2
    aux2 = self.aux_head2(d2) if self.deep_supervision else None

    d1 = self.d1(d2, s1)     # base, H, W
    aux1 = self.aux_head1(d1) if self.deep_supervision else None

    # Multi-scale fusion (optional)
    if self.multi_scale_fusion:
        decoder_features = [d1, d2, d3, d4]
        fused = self.ms_fusion(decoder_features)
        # Combine fused features with final decoder output
        combined = torch.cat([d1, fused], dim=1)
        final_features = self.fusion_conv(combined)
    else:
        final_features = d1

    # Final segmentation
    seg = self.head(final_features)

    if self.deep_supervision:
        return seg, [aux3, aux2, aux1]
    return seg
```

**Dòng 240-280**: Complete forward pass

---

## Model Configurations

### Phase 2 Small (RTX 3090, 24GB)

```python
model = SegUNetV2(
    in_ch=4,
    base=48,
    dim=384,
    patch=8,
    depth=4,
    n_heads=8,
    num_classes=3,
    dropout=0.15,
    norm='instance',
    deep_supervision=True,
    multi_scale_fusion=True,
)
```

**Stats**:
- Parameters: ~35M
- Memory: ~16GB (batch size 12)
- Speed: ~3.5s/epoch on RTX 3090

### Phase 2 Large (A100, 40GB)

```python
model = SegUNetV2(
    in_ch=4,
    base=64,
    dim=512,
    patch=8,
    depth=4,
    n_heads=8,
    num_classes=3,
    dropout=0.15,
    norm='instance',
    deep_supervision=True,
    multi_scale_fusion=True,
)
```

**Stats**:
- Parameters: ~60M
- Memory: ~28GB (batch size 16)
- Speed: ~5s/epoch on A100

### Baseline (V1-compatible)

```python
model = SegUNetV2(
    in_ch=4,
    base=32,
    dim=256,
    patch=8,
    depth=2,
    n_heads=4,
    num_classes=1,  # Binary
    dropout=0.0,
    deep_supervision=False,
    multi_scale_fusion=False,
)
```

**Stats**:
- Parameters: ~14M (same as V1)
- But with V2 improvements (ResBlocks, InstanceNorm, LeakyReLU)

---

## So Sánh V1 vs V2

### Feature Comparison Table

| Feature | V1 | V2 Small | V2 Large |
|---------|----|-----------| ---------|
| **Base channels** | 32 | 48 | 64 |
| **Transformer dim** | 256 | 384 | 512 |
| **Transformer depth** | 2 | 4 | 4 |
| **Attention heads** | 4 | 8 | 8 |
| **Parameters** | 14M | 35M | 60M |
| **Normalization** | BatchNorm | InstanceNorm | InstanceNorm |
| **Activation** | ReLU | LeakyReLU | LeakyReLU |
| **Downsampling** | MaxPool | Strided Conv | Strided Conv |
| **Residual blocks** | ❌ | ✅ | ✅ |
| **Multi-scale fusion** | ❌ | ✅ | ✅ |
| **Deep supervision** | ❌ | ✅ | ✅ |
| **Dropout** | 0.0 | 0.15 | 0.15 |
| **Segmentation modes** | Binary | Binary + Multi-class | Binary + Multi-class |

### Performance Comparison

**Expected improvements** (based on Phase 2 design):

```
V1 Binary:
- Dice: 0.9148
- IoU: 0.8430

V2 Small Binary (expected):
- Dice: ~0.92-0.93
- IoU: ~0.85-0.87

V2 Large Binary (expected):
- Dice: ~0.93-0.94
- IoU: ~0.87-0.89

V2 Multi-Class (3 classes):
- Dice WT: ~0.88-0.90
- Dice TC: ~0.83-0.86
- Dice ED: ~0.82-0.85
```

### Training Time Comparison

**Per epoch** (on dataset with ~22k slices):

| Model | GPU | Batch Size | Time/Epoch | Memory |
|-------|-----|------------|------------|--------|
| V1 | RTX 3090 | 16 | ~2.5s | ~12GB |
| V2 Small | RTX 3090 | 12 | ~3.5s | ~16GB |
| V2 Large | A100 40GB | 16 | ~5.0s | ~28GB |

**Trade-off**: V2 models ~40-100% slower but significantly more accurate!

### Code Complexity

**V1**: 67 dòng (seg_unet.py)
**V2**: 322 dòng (seg_unet_v2.py)

**Why 5× more code?**
- Residual blocks: +30 dòng
- Multi-scale fusion: +45 dòng
- Deep supervision logic: +20 dòng
- Flexible normalization: +25 dòng
- Comments và documentation: +135 dòng

**But**: More features, better performance, more flexible!

---

## Summary

### V2 Key Takeaways

**7 Major Improvements**:
1. ✅ InstanceNorm → Better for medical imaging
2. ✅ LeakyReLU → No dying neurons
3. ✅ Residual blocks → Train deeper networks
4. ✅ Strided conv → Learned downsampling
5. ✅ Multi-scale fusion → Combine all decoder levels
6. ✅ Deep supervision → Better gradient flow
7. ✅ Dropout → Prevent overfitting

**Benefits**:
- Higher accuracy (expected +2-4% Dice)
- Multi-class segmentation support
- More stable training
- Better generalization

**Trade-offs**:
- Larger model size (14M → 35-60M)
- Slower training (~40-100% slower)
- More memory required (12GB → 16-28GB)

### When to Use V1 vs V2?

**Use V1 when**:
- Binary segmentation is enough
- Limited GPU memory (<12GB)
- Need fast training
- Already have good results

**Use V2 when**:
- Need multi-class segmentation
- Have good GPU (RTX 3090, A100)
- Want best possible accuracy
- Training time not critical

---

[[v_03_MODEL_ARCHITECTURE|← Back to Main Architecture]] | [[v_04_TRAINING_SYSTEM|Next: Training System →]]
