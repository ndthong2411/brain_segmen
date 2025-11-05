# SegUNetV2 Architecture - Kiến Trúc Chi Tiết

> **Giải thích kiến trúc SegUNetV2 từng component**
>
> **File code**: `src/braintumnet/models/seg_unet_v2.py` (478 dòng)

---

## Mục Lục

1. [Architecture Overview](#architecture-overview)
2. [Enhanced Conv Block](#enhanced-conv-block)
3. [Residual Convolutional Block](#residual-convolutional-block)
4. [Enhanced Encoder](#enhanced-encoder)
5. [Enhanced Decoder](#enhanced-decoder)
6. [Multi-Scale Fusion Module](#multi-scale-fusion-module)
7. [Deep Supervision](#deep-supervision)
8. [Complete Forward Pass](#complete-forward-pass)
9. [Tensor Shapes Example](#tensor-shapes-example)

---

## Architecture Overview

### High-Level Structure

```
Input (B, 4, 256, 256)
    ↓
┌─────────────────────────────────────┐
│         ENCODER (4 blocks)          │
│  e1: 4→48    (256×256)             │
│  e2: 48→96   (128×128)             │
│  e3: 96→192  (64×64)               │
│  e4: 192→384 (32×32)               │
│  [Strided Conv downsampling]       │
└─────────────────────────────────────┘
    ↓ skip connections (s1,s2,s3,s4)
┌─────────────────────────────────────┐
│    TRANSFORMER BOTTLENECK           │
│  Conv 1×1: 384→512                 │
│  Transformer: 512→512               │
│  ConvTranspose: 512→384             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│         DECODER (4 blocks)          │
│  d4: 384→384 (32×32)  + s4         │
│  d3: 384→192 (64×64)  + s3 → aux3  │
│  d2: 192→96  (128×128) + s2 → aux2 │
│  d1: 96→48   (256×256) + s1 → aux1 │
│  [CBAM attention on skips]         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│    MULTI-SCALE FUSION (optional)    │
│  Fuse [d1, d2, d3, d4]             │
│  → Single 48-channel feature        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│         SEGMENTATION HEAD           │
│  Conv 1×1: 48→3 (num_classes)      │
└─────────────────────────────────────┘
    ↓
Output (B, 3, 256, 256)
+ Auxiliary outputs (if deep_supervision)
```

### Key Differences from V1

| Component | V1 | V2 |
|-----------|----|----|
| **Conv blocks** | Conv-BN-ReLU | Conv-InstanceNorm-LeakyReLU-Dropout |
| **Block structure** | Sequential convs | Residual blocks |
| **Downsampling** | MaxPool (fixed) | Strided Conv (learned) |
| **Normalization** | BatchNorm | InstanceNorm |
| **Activation** | ReLU | LeakyReLU (0.01) |
| **Skip connections** | Direct | Through residual blocks |
| **Decoder fusion** | Only d1 | Multi-scale fusion [d1,d2,d3,d4] |
| **Supervision** | Single output | Deep supervision (3 auxiliary) |
| **Regularization** | None | Dropout (adaptive) |

---

## Enhanced Conv Block

### Function: conv_norm_act()

**Location**: seg_unet_v2.py lines 24-51

```python
def conv_norm_act(in_ch, out_ch, k=3, s=1, p=1, norm='instance', dropout=0.0):
    """
    Improved convolution block: Conv + Norm + LeakyReLU + Dropout

    Args:
        in_ch: Input channels
        out_ch: Output channels
        k: Kernel size (default 3×3)
        s: Stride (default 1)
        p: Padding (default 1)
        norm: 'instance', 'batch', or 'group'
        dropout: Dropout probability (0.0 = no dropout)

    Returns:
        nn.Sequential: Conv → Norm → Act → (Dropout)
    """
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)]

    # Normalization
    if norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    elif norm == 'batch':
        layers.append(nn.BatchNorm2d(out_ch))
    elif norm == 'group':
        num_groups = min(32, out_ch // 4)  # Adaptive
        layers.append(nn.GroupNorm(num_groups, out_ch))

    # Activation
    layers.append(nn.LeakyReLU(0.01, inplace=True))

    # Dropout (if specified)
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))

    return nn.Sequential(*layers)
```

### Component Breakdown

#### 1. Convolution
```python
nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
```
- **kernel_size=3**: 3×3 receptive field
- **stride=1**: No downsampling (done separately)
- **padding=1**: Keep spatial size
- **bias=False**: Bias không cần vì normalization có affine

#### 2. Normalization

**InstanceNorm (default)**:
```python
nn.InstanceNorm2d(out_ch, affine=True)

# Computation:
# For each sample and channel:
mean = x.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
var = x.var(dim=(2, 3), keepdim=True)
x_norm = (x - mean) / sqrt(var + eps)
x_out = gamma * x_norm + beta  # Affine transform
```

**Why InstanceNorm?**
- Medical imaging: patient-specific intensity ranges
- Batch size nhỏ (4-8): BatchNorm unstable
- Training == Inference (no running statistics)
- Standard trong medical imaging (nnU-Net uses InstanceNorm)

**BatchNorm (optional)**:
```python
nn.BatchNorm2d(out_ch)

# Computation:
# Across all samples in batch:
mean = x.mean(dim=(0, 2, 3))  # (C,)
var = x.var(dim=(0, 2, 3))
```
- Dùng nếu batch size lớn (>16)
- Có running statistics (training != inference)

**GroupNorm (optional)**:
```python
num_groups = min(32, out_ch // 4)
nn.GroupNorm(num_groups, out_ch)
```
- Middle ground giữa Instance và Batch
- Chia channels thành groups, normalize each group
- Tốt cho batch size trung bình (8-16)

#### 3. Activation

```python
nn.LeakyReLU(negative_slope=0.01, inplace=True)

# Computation:
# f(x) = x if x > 0 else 0.01 * x
```

**Why LeakyReLU?**
- Gradient luôn flow (slope=0.01 khi x<0)
- No "dying ReLU" problem
- nnU-Net standard (slope=0.01)

**vs ReLU**:
```python
# ReLU: f(x) = max(0, x)
# Gradient = 0 when x < 0 → neurons can "die"

# LeakyReLU: f(x) = max(0.01x, x)
# Gradient = 0.01 when x < 0 → always alive
```

#### 4. Dropout (Optional)

```python
nn.Dropout2d(dropout)

# Drops entire feature maps (not individual pixels)
# During training: randomly set maps to 0 with probability p
# During inference: no dropout (model.eval())
```

**Dropout Strategy**:
- `dropout=0.0`: Shallow layers (e1, e2, d1)
- `dropout=0.15`: Deep layers (e3, e4, d4, d3)
- `dropout=0.075`: Middle decoder (d2)

**Why Dropout2d not Dropout?**
- Spatial coherence: drop whole feature maps
- Better for convolutional layers
- Prevents co-adaptation

---

## Residual Convolutional Block

### Class: ResidualConvBlock

**Location**: seg_unet_v2.py lines 53-80

```python
class ResidualConvBlock(nn.Module):
    """
    Residual convolutional block: Conv-Norm-Act → Conv-Norm → Add-Act

    Structure:
        x → [Conv-Norm-Act] → [Conv-Norm] → (+) → Act → out
        ↓                                      ↑
        └────────[Residual Projection]─────────┘
    """
    def __init__(self, in_ch, out_ch, norm='instance', dropout=0.0):
        super().__init__()

        # First conv with activation
        self.conv1 = conv_norm_act(in_ch, out_ch, norm=norm, dropout=dropout)

        # Second conv WITHOUT activation (applied after residual add)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True) if norm == 'instance'
                else nn.BatchNorm2d(out_ch)
        )

        # Residual connection: 1×1 conv if channels mismatch
        self.residual = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                        if in_ch != out_ch else nn.Identity())

        # Final activation (after residual add)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        identity = self.residual(x)  # Project if needed

        out = self.conv1(x)          # Conv-Norm-Act
        out = self.conv2(out)        # Conv-Norm (no act)
        out = out + identity         # Residual addition
        out = self.act(out)          # Final activation

        return out
```

### Why Residual Connections?

**Problem without residuals**:
```python
# Deep network:
x → Conv → Conv → Conv → ... → Conv → out
# Gradient flow:
∂L/∂x_0 = ∂L/∂x_n * ∂x_n/∂x_{n-1} * ... * ∂x_1/∂x_0
# Each layer multiplies gradient → vanishing gradients
```

**Solution with residuals**:
```python
# Residual block:
out = F(x) + x  # F(x) = learned function, x = identity

# Gradient flow:
∂L/∂x = ∂L/∂out * (∂F/∂x + 1)
#                   ↑      ↑
#                 learned  identity
# +1 ensures gradient always flows!
```

### Residual Projection

**Case 1: Matching channels** (in_ch == out_ch)
```python
self.residual = nn.Identity()
# out = F(x) + x  (direct addition)
```

**Case 2: Mismatched channels** (in_ch != out_ch)
```python
self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
# out = F(x) + Conv1×1(x)  (project to matching dims)
```

**Example**:
```python
# Encoder 1: 4 → 48 channels
block = ResidualConvBlock(in_ch=4, out_ch=48)
# self.residual = Conv2d(4, 48, 1×1) to project input

x = torch.randn(2, 4, 256, 256)    # Input
identity = self.residual(x)         # (2, 48, 256, 256)
out = self.conv1(x)                 # (2, 48, 256, 256)
out = self.conv2(out)               # (2, 48, 256, 256)
out = out + identity                # Residual addition
out = self.act(out)                 # (2, 48, 256, 256)
```

### Forward Pass Detail

```python
def forward(self, x):
    # Step 1: Project identity (if needed)
    identity = self.residual(x)  # (B, out_ch, H, W)

    # Step 2: First conv block
    out = self.conv1(x)
    # x: (B, in_ch, H, W)
    # → Conv: (B, out_ch, H, W)
    # → InstanceNorm: normalized
    # → LeakyReLU: activated
    # → Dropout: regularized (if dropout > 0)

    # Step 3: Second conv block (no activation yet)
    out = self.conv2(out)
    # → Conv: (B, out_ch, H, W)
    # → InstanceNorm: normalized
    # (No activation - will be applied after residual add)

    # Step 4: Residual addition
    out = out + identity  # Element-wise addition

    # Step 5: Final activation
    out = self.act(out)  # LeakyReLU

    return out  # (B, out_ch, H, W)
```

---

## Enhanced Encoder

### Class: EncoderBlock

**Location**: seg_unet_v2.py lines 82-100

```python
class EncoderBlock(nn.Module):
    """
    Encoder block with residual convolutions and strided conv downsampling

    Improvements over V1:
    - Residual connections (instead of plain convs)
    - Strided conv (instead of MaxPool)
    """
    def __init__(self, in_ch, out_ch, norm='instance', dropout=0.0):
        super().__init__()

        # Residual convolutional block
        self.block = ResidualConvBlock(in_ch, out_ch, norm=norm, dropout=dropout)

        # Strided convolution for learned downsampling
        self.downsample = nn.Conv2d(
            out_ch, out_ch,
            kernel_size=3, stride=2, padding=1,
            bias=False
        )

    def forward(self, x):
        # Process features
        x = self.block(x)          # (B, out_ch, H, W)

        # Downsample for next level
        x_down = self.downsample(x)  # (B, out_ch, H/2, W/2)

        return x, x_down
        #      ↑  ↑
        #      │  └─ For next encoder level
        #      └──── Skip connection to decoder
```

### Strided Conv vs MaxPool

**MaxPool (V1)**:
```python
nn.MaxPool2d(kernel_size=2, stride=2)

# Operation:
# For each 2×2 window, take maximum value
# Example:
# [1 2]  → max = 6
# [5 6]

# Problems:
# - Fixed operation (not learnable)
# - Throws away 75% of information
# - No parameters to optimize
```

**Strided Conv (V2)**:
```python
nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

# Operation:
# Learnable 3×3 kernel with stride=2
# Weighted combination of all pixels in receptive field

# Benefits:
# - Learnable (has parameters)
# - Adapts to task
# - Preserves more information through weighted sum
```

**Visual Comparison**:
```
Input 4×4:
┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │
├───┼───┼───┼───┤
│ 5 │ 6 │ 7 │ 8 │
├───┼───┼───┼───┤
│ 9 │ A │ B │ C │
├───┼───┼───┼───┤
│ D │ E │ F │ G │
└───┴───┴───┴───┘

MaxPool 2×2 stride=2:
┌─────┬─────┐
│  6  │  8  │  ← max(1,2,5,6), max(3,4,7,8)
├─────┼─────┤
│  E  │  G  │  ← max(9,A,D,E), max(B,C,F,G)
└─────┴─────┘
(Fixed, no learning)

Strided Conv 3×3 stride=2:
┌──────────┬──────────┐
│ w₁·1 +   │ w₁·3 +   │
│ w₂·2 +   │ w₂·4 +   │
│ w₃·5 +   │ w₃·7 +   │  ← Learned weights
│ w₄·6 +   │ w₄·8 +   │
│ w₅·9 +   │ w₅·B +   │
│ ...      │ ...      │
└──────────┴──────────┘
(Learned combination)
```

### Example Usage

```python
# Create encoder block
encoder = EncoderBlock(in_ch=48, out_ch=96, dropout=0.15)

# Forward pass
x = torch.randn(4, 48, 128, 128)  # (B, C, H, W)
skip, down = encoder(x)

print(f"Input:  {x.shape}")      # (4, 48, 128, 128)
print(f"Skip:   {skip.shape}")   # (4, 96, 128, 128) - for decoder
print(f"Down:   {down.shape}")   # (4, 96, 64, 64) - downsampled 2×
```

---

## Enhanced Decoder

### Class: DecoderBlock

**Location**: seg_unet_v2.py lines 153-186

```python
class DecoderBlock(nn.Module):
    """
    Decoder block with residual convolutions and CBAM attention

    Improvements over V1:
    - Residual connections
    - CBAM attention on skip connections
    - Optional Attention Gates (Phase 2)
    - Dropout regularization
    """
    def __init__(self, in_ch, out_ch, norm='instance', dropout=0.0,
                 use_attention_gate=False):
        super().__init__()
        self.use_attention_gate = use_attention_gate

        # Upsample decoder features
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch,
            kernel_size=2, stride=2,
            bias=False
        )

        # Optional Attention Gate (Phase 2 feature)
        if use_attention_gate:
            self.attn_gate = AttentionGate(
                F_g=out_ch,    # Gating signal from decoder
                F_l=out_ch,    # Skip connection from encoder
                F_int=out_ch // 2  # Intermediate dimension
            )

        # CBAM attention (from V1, kept in V2)
        self.cbam = CBAM(out_ch)

        # Residual block processes concatenated features
        self.block = ResidualConvBlock(
            out_ch * 2,  # Concatenated: decoder + skip
            out_ch,
            norm=norm,
            dropout=dropout
        )

    def forward(self, x, skip):
        """
        Args:
            x: Decoder features from previous level (B, in_ch, H, W)
            skip: Skip connection from encoder (B, out_ch, 2H, 2W)

        Returns:
            Processed features (B, out_ch, 2H, 2W)
        """
        # Step 1: Upsample decoder features
        x = self.up(x)  # (B, out_ch, 2H, 2W)

        # Step 2: Apply Attention Gate (if enabled)
        if self.use_attention_gate:
            skip = self.attn_gate(g=x, x=skip)

        # Step 3: Apply CBAM to skip connection
        skip = self.cbam(skip)

        # Step 4: Concatenate decoder and skip
        x = torch.cat([x, skip], dim=1)  # (B, out_ch*2, 2H, 2W)

        # Step 5: Process through residual block
        x = self.block(x)  # (B, out_ch, 2H, 2W)

        return x
```

### Decoder Flow

```
Decoder Input (B, 384, 32, 32)
    ↓
┌─────────────────────────┐
│   ConvTranspose 2×2     │
│   384 → 192             │
└─────────────────────────┘
    ↓ (B, 192, 64, 64)

Skip Connection (B, 192, 64, 64)
    ↓
┌─────────────────────────┐
│   Attention Gate        │  ← Optional (Phase 2)
│   (g=decoder, x=skip)   │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│   CBAM Attention        │
│   Channel + Spatial     │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│   Concatenate           │
│   [decoder, skip]       │
│   (192*2=384 channels)  │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│   Residual Block        │
│   384 → 192             │
└─────────────────────────┘
    ↓
Output (B, 192, 64, 64)
```

### CBAM Attention (Inherited from V1)

```python
# Channel attention: "what" to focus on
channel_weights = GlobalAvgPool → FC → ReLU → FC → Sigmoid
features = features * channel_weights

# Spatial attention: "where" to focus on
spatial_weights = Conv → Sigmoid
features = features * spatial_weights
```

**Why keep CBAM in V2?**
- Proven effective in V1 (+1.86% Dice)
- Channel attention: emphasize important feature channels
- Spatial attention: focus on relevant spatial regions
- Low overhead (<1% parameters)

---

## Multi-Scale Fusion Module

### Class: MultiScaleFusion

**Location**: seg_unet_v2.py lines 250-293

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
            channels_list: [d1_ch, d2_ch, d3_ch, d4_ch]
                          e.g., [48, 96, 192, 384]
            out_channels: Output channels (usually base)
                         e.g., 48
        """
        super().__init__()

        # 1×1 convs to project all to same channel dimension
        self.convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, kernel_size=1, bias=False)
            for ch in channels_list
        ])

        # Normalization and activation
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, features):
        """
        Args:
            features: List [d1, d2, d3, d4] with different spatial sizes
                     d1: (B, 48, 256, 256)
                     d2: (B, 96, 128, 128)
                     d3: (B, 192, 64, 64)
                     d4: (B, 384, 32, 32)

        Returns:
            fused: (B, out_channels, 256, 256) - all scales combined
        """
        target_size = features[0].shape[2:]  # Use d1 size (largest)

        upsampled = []
        for i, feat in enumerate(features):
            # Project to same channel dimension
            feat = self.convs[i](feat)  # → (B, out_channels, H, W)

            # Upsample to target size if needed
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size,
                    mode='bilinear', align_corners=False
                )

            upsampled.append(feat)

        # Fuse by summation
        fused = sum(upsampled)  # Element-wise addition

        # Normalize and activate
        fused = self.norm(fused)
        fused = self.act(fused)

        return fused
```

### Multi-Scale Fusion Process

**Step-by-step**:

```
Input: [d1, d2, d3, d4] decoder features

Step 1: Project to same channels (48)
  d1: (B, 48, 256, 256)  → Conv1×1 → (B, 48, 256, 256)
  d2: (B, 96, 128, 128)  → Conv1×1 → (B, 48, 128, 128)
  d3: (B, 192, 64, 64)   → Conv1×1 → (B, 48, 64, 64)
  d4: (B, 384, 32, 32)   → Conv1×1 → (B, 48, 32, 32)

Step 2: Upsample to target size (256×256)
  d1: No change (already 256×256)
  d2: Bilinear ↑2× → (B, 48, 256, 256)
  d3: Bilinear ↑4× → (B, 48, 256, 256)
  d4: Bilinear ↑8× → (B, 48, 256, 256)

Step 3: Sum all features
  fused = d1 + d2_up + d3_up + d4_up
  → (B, 48, 256, 256)

Step 4: Normalize and activate
  fused = LeakyReLU(InstanceNorm(fused))
```

### Why Summation not Concatenation?

**Concatenation**:
```python
fused = torch.cat([d1, d2_up, d3_up, d4_up], dim=1)
# → (B, 48*4=192, 256, 256)  Too many channels!
# Need extra conv to reduce: 192 → 48
```

**Summation**:
```python
fused = d1 + d2_up + d3_up + d4_up
# → (B, 48, 256, 256)  Efficient!
# Equally weights all scales
# No extra parameters
```

### What Each Scale Contributes

```
d4 (32×32):   High-level semantic
              "Is this tumor or not?"
              Coarse localization

d3 (64×64):   Mid-level structural
              "Where are boundaries?"
              Structural information

d2 (128×128): Low-level features
              "Fine edge details"
              Texture information

d1 (256×256): Spatial details
              "Precise localization"
              Pixel-level accuracy

FUSION:       Combines all levels
              → Best of all scales!
```

---

## Deep Supervision

### Concept

**Problem**: Gradient chỉ backprop qua main head
```
Encoder → Bottleneck → Decoder → HEAD → Loss
                                         ↓
                                    Gradients
```
→ Gradients yếu dần khi flow backward
→ Early layers học chậm

**Solution**: Auxiliary outputs ở intermediate decoder levels
```
d4 → d3 → aux3 → Loss3 (weight 0.125)
     ↓
     d2 → aux2 → Loss2 (weight 0.25)
     ↓
     d1 → aux1 → Loss1 (weight 0.5)
     ↓
     Final → main → Loss (weight 1.0)

Total Loss = Loss + 0.5*Loss1 + 0.25*Loss2 + 0.125*Loss3
```

### Implementation

```python
# In SegUNetV2.__init__():
if self.deep_supervision:
    self.aux_head3 = nn.Conv2d(base*4, num_classes, kernel_size=1)
    self.aux_head2 = nn.Conv2d(base*2, num_classes, kernel_size=1)
    self.aux_head1 = nn.Conv2d(base, num_classes, kernel_size=1)

# In forward():
d3 = self.d3(d4, s3)
aux3 = self.aux_head3(d3) if self.deep_supervision else None

d2 = self.d2(d3, s2)
aux2 = self.aux_head2(d2) if self.deep_supervision else None

d1 = self.d1(d2, s1)
aux1 = self.aux_head1(d1) if self.deep_supervision else None

# Final output
seg = self.head(final_features)

if self.deep_supervision:
    return seg, [aux3, aux2, aux1]
return seg
```

### Loss Computation

```python
# In training loop:
if deep_supervision:
    seg_logits, aux_outputs = model(img)

    # Main loss
    loss = criterion(seg_logits, mask)

    # Auxiliary losses with decreasing weights
    aux_weights = [0.5, 0.25, 0.125]
    for aux, weight in zip(aux_outputs, aux_weights):
        # Resize aux to match mask size
        aux_resized = F.interpolate(
            aux, size=mask.shape[-2:],
            mode='bilinear', align_corners=False
        )
        loss += weight * criterion(aux_resized, mask)
```

### Why Decreasing Weights?

```
Main output (256×256):  weight = 1.0    ← Most accurate
aux1 (256×256):         weight = 0.5    ← Same resolution
aux2 (128×128):         weight = 0.25   ← Lower resolution
aux3 (64×64):           weight = 0.125  ← Lowest resolution

Reasoning:
- Main output: Full resolution, most important
- Auxiliary outputs: Lower resolution, less accurate
- Weight decay: Fair contribution balance
```

---

## Complete Forward Pass

### Full SegUNetV2 Forward

```python
def forward(self, x):
    """
    Args:
        x: (B, 4, 256, 256) input MRI (4 modalities)

    Returns:
        If deep_supervision:
            seg: (B, 3, 256, 256) main segmentation
            aux: [(B, 3, 64, 64), (B, 3, 128, 128), (B, 3, 256, 256)]
        Else:
            seg: (B, 3, 256, 256)
    """
    # ─────────────────────────────────────────────────────
    # ENCODER
    # ─────────────────────────────────────────────────────
    s1, x1 = self.e1(x)      # (B, 48, 256, 256), (B, 48, 128, 128)
    s2, x2 = self.e2(x1)     # (B, 96, 128, 128), (B, 96, 64, 64)
    s3, x3 = self.e3(x2)     # (B, 192, 64, 64), (B, 192, 32, 32)
    s4, x4 = self.e4(x3)     # (B, 384, 32, 32), (B, 384, 16, 16)

    # ─────────────────────────────────────────────────────
    # TRANSFORMER BOTTLENECK
    # ─────────────────────────────────────────────────────
    b = self.bottleneck_conv(x4)  # (B, 512, 16, 16)

    if self.use_multiscale_transformer:
        # Phase 2: Multi-scale transformer
        b = self.bottleneck(b)    # (B, 512, 16, 16)
    else:
        # Phase 1: Single-scale transformer
        b = self.amt(b)                # Flatten to tokens
        b = self.tr_upsample(b)        # Back to spatial

    # ─────────────────────────────────────────────────────
    # DECODER
    # ─────────────────────────────────────────────────────
    d4 = self.d4(b, s4)      # (B, 384, 32, 32)

    d3 = self.d3(d4, s3)     # (B, 192, 64, 64)
    aux3 = self.aux_head3(d3) if self.deep_supervision else None

    d2 = self.d2(d3, s2)     # (B, 96, 128, 128)
    aux2 = self.aux_head2(d2) if self.deep_supervision else None

    d1 = self.d1(d2, s1)     # (B, 48, 256, 256)
    aux1 = self.aux_head1(d1) if self.deep_supervision else None

    # ─────────────────────────────────────────────────────
    # MULTI-SCALE FUSION (optional)
    # ─────────────────────────────────────────────────────
    if self.multi_scale_fusion:
        decoder_features = [d1, d2, d3, d4]
        fused = self.ms_fusion(decoder_features)  # (B, 48, 256, 256)

        # Combine fused with d1
        combined = torch.cat([d1, fused], dim=1)  # (B, 96, 256, 256)
        final_features = self.fusion_conv(combined)  # (B, 48, 256, 256)
    else:
        final_features = d1

    # ─────────────────────────────────────────────────────
    # BOUNDARY REFINEMENT (optional, from Phase 1)
    # ─────────────────────────────────────────────────────
    if self.boundary_refinement:
        final_features = self.boundary_refine(final_features)

    # ─────────────────────────────────────────────────────
    # FINAL SEGMENTATION
    # ─────────────────────────────────────────────────────
    seg = self.head(final_features)  # (B, 3, 256, 256)

    if self.deep_supervision:
        return seg, [aux3, aux2, aux1]
    return seg
```

---

## Tensor Shapes Example

### Phase 2 Small (base=48)

```python
# Input
x = torch.randn(2, 4, 256, 256)  # (B, C, H, W)

# ─── ENCODER ───
s1, x1 = e1(x)    # s1: (2, 48, 256, 256)  x1: (2, 48, 128, 128)
s2, x2 = e2(x1)   # s2: (2, 96, 128, 128)  x2: (2, 96, 64, 64)
s3, x3 = e3(x2)   # s3: (2, 192, 64, 64)   x3: (2, 192, 32, 32)
s4, x4 = e4(x3)   # s4: (2, 384, 32, 32)   x4: (2, 384, 16, 16)

# ─── BOTTLENECK ───
b = bottleneck_conv(x4)  # (2, 512, 16, 16)
b = amt(b)               # Transformer
b = tr_upsample(b)       # (2, 384, 32, 32)

# ─── DECODER ───
d4 = d4(b, s4)     # (2, 384, 32, 32)
d3 = d3(d4, s3)    # (2, 192, 64, 64)   aux3: (2, 3, 64, 64)
d2 = d2(d3, s2)    # (2, 96, 128, 128)  aux2: (2, 3, 128, 128)
d1 = d1(d2, s1)    # (2, 48, 256, 256)  aux1: (2, 3, 256, 256)

# ─── MULTI-SCALE FUSION ───
decoder_features = [d1, d2, d3, d4]
# → [(2, 48, 256, 256), (2, 96, 128, 128),
#    (2, 192, 64, 64), (2, 384, 32, 32)]

fused = ms_fusion(decoder_features)  # (2, 48, 256, 256)
combined = cat([d1, fused], dim=1)   # (2, 96, 256, 256)
final = fusion_conv(combined)        # (2, 48, 256, 256)

# ─── OUTPUT ───
seg = head(final)  # (2, 3, 256, 256)
```

### Memory Footprint

**Phase 2 Small** (base=48):
```
Encoder outputs: ~100MB per sample
  s1: 48×256×256 = 3.1MB
  s2: 96×128×128 = 1.6MB
  s3: 192×64×64 = 0.8MB
  s4: 384×32×32 = 0.4MB

Decoder outputs: ~100MB per sample
  d4-d1: Same as encoder

Total per sample: ~200MB
Batch of 8: ~1.6GB (activations only)
Model parameters: ~150MB (37M × 4 bytes)
Optimizer states: ~600MB (AdamW)

Peak memory (batch=8): ~16GB
```

**Phase 2 Large** (base=64):
```
Peak memory (batch=16): ~28GB
Fits on A100 80GB comfortably
```

---

## Summary

### SegUNetV2 Key Components

1. **Enhanced Conv Block**: InstanceNorm + LeakyReLU + Dropout
2. **Residual Blocks**: Better gradient flow, deeper training
3. **Enhanced Encoder**: Strided conv downsampling
4. **Enhanced Decoder**: Residual blocks + CBAM + optional AttentionGate
5. **Multi-Scale Fusion**: Combine all decoder levels
6. **Deep Supervision**: Auxiliary losses for better training

### Improvements Over V1

| Aspect | Improvement |
|--------|-------------|
| **Normalization** | BatchNorm → InstanceNorm (+medical imaging standard) |
| **Activation** | ReLU → LeakyReLU (+gradient flow) |
| **Blocks** | Plain convs → Residual blocks (+depth capability) |
| **Downsampling** | MaxPool → Strided conv (+learned, adaptive) |
| **Feature fusion** | Single scale → Multi-scale (+multi-resolution) |
| **Training** | Single output → Deep supervision (+gradient flow) |
| **Regularization** | None → Dropout (+prevent overfitting) |

### Parameters Breakdown

**Phase 2 Small** (37M total):
- Encoder: ~8M
- Bottleneck: ~18M
- Decoder: ~8M
- Fusion: ~1M
- Heads: ~2M

---

**Next**: [Phase 2 Features →](v2_03_PHASE2_FEATURES.md)

**Back**: [← Phase 2 Overview](v2_01_PHASE2_OVERVIEW.md)
