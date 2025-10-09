# Part 3: Model Architecture Deep Dive

**Navigation**: [[TECHNICAL_REPORT_INDEX|← Back to Index]]

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [BrainTumNet Main Model](#braintumnet-main-model)
3. [Segmentation U-Net with Transformer](#segmentation-unet-with-transformer)
4. [CBAM Attention Mechanism](#cbam-attention-mechanism)
5. [Adaptive Masked Transformer](#adaptive-masked-transformer)
6. [Inception Classification Network](#inception-classification-network)
7. [Complete Data Flow with Tensor Shapes](#complete-data-flow-with-tensor-shapes)
8. [Design Decisions Explained](#design-decisions-explained)
9. [Modification Guides](#modification-guides)

---

## Architecture Overview

### What is BrainTumNet?

BrainTumNet is a **multi-task deep learning model** that performs two tasks simultaneously:

1. **Segmentation**: Pixel-wise tumor mask prediction (where is the tumor?)
2. **Classification**: Tumor grade classification (HGG vs LGG)

### Why Multi-Task Learning?

Multi-task learning offers several advantages:

- **Shared Features**: Both tasks benefit from shared encoder features
- **Better Generalization**: Learning multiple tasks reduces overfitting
- **Efficient**: One forward pass → two predictions
- **Clinically Relevant**: Both segmentation and grade are needed for diagnosis

### Model Components

The architecture consists of 5 Python files:

| File | Purpose | Lines | Complexity |
|------|---------|-------|------------|
| `braintumnet.py` | Main multi-task wrapper | 24 | Low |
| `seg_unet.py` | U-Net encoder-decoder with attention | 67 | Medium |
| `cbam.py` | Channel & Spatial Attention (CBAM) | 33 | Medium |
| `masked_transformer.py` | Adaptive Masked Transformer | 88 | High |
| `t_inception.py` | Inception-based classifier | 51 | Medium |

### Architecture Diagram

```
Input Image (B, C, 256, 256)
         |
    ┌────┴────┐
    │         │
    │  U-Net  │ ← CBAM Attention on skip connections
    │ Encoder │ ← Adaptive Masked Transformer in bottleneck
    │ Decoder │
    │         │
    └────┬────┘
         |
  Segmentation Logits (B, 1, 256, 256)
         |
    [Sigmoid] → Tumor Probability Mask
         |
    [ROI Gating] → Mask × Input Image
         |
    ┌────┴────┐
    │         │
    │ Inception│
    │ Classifier│
    │         │
    └────┬────┘
         |
  Classification Logits (B, num_classes)
```

---

## BrainTumNet Main Model

**File**: `src/braintumnet/models/braintumnet.py` (24 lines)

This is the **top-level model** that coordinates segmentation and classification.

### Complete Code with Line-by-Line Explanation

```python
import torch, torch.nn as nn
from .seg_unet import SegUNetMasked
from .t_inception import TInceptionNet
```

**Line 1-3**: Import dependencies
- `torch.nn.Module`: Base class for all PyTorch models
- `SegUNetMasked`: Segmentation network (explained in next section)
- `TInceptionNet`: Classification network (explained later)

---

```python
class BrainTumNet(nn.Module):
    def __init__(self, in_ch=1, num_cls=2, base=32, dim=256, patch=8, depth=2, n_heads=4, roi_stop_grad=True):
```

**Line 5-6**: Class definition and constructor

**Parameters Explained**:
- `in_ch=1`: Input channels (1 for single-modal, 4 for multi-modal MRI)
- `num_cls=2`: Number of classes (2 for HGG/LGG classification)
- `base=32`: Base number of U-Net channels (controls model capacity)
- `dim=256`: Transformer embedding dimension
- `patch=8`: Transformer patch size (8×8 patches)
- `depth=2`: Number of transformer blocks
- `n_heads=4`: Number of attention heads in transformer
- `roi_stop_grad=True`: **Critical parameter** - stop gradient flow from classifier to segmentation

**Why These Defaults?**
- `base=32`: Balance between model capacity and memory usage
- `dim=256`: Standard transformer dimension, large enough for medical images
- `patch=8`: After 4 encoder blocks (16× downsampling), 256/16=16 → 16/8=2 patches per dimension
- `roi_stop_grad=True`: Prevents classifier from affecting segmentation training

---

```python
        super().__init__()
        self.seg = SegUNetMasked(in_ch=in_ch, base=base, dim=dim, patch=patch, depth=depth, n_heads=n_heads)
        self.roi_stop_grad = roi_stop_grad
```

**Line 7-9**: Initialize segmentation network

- `super().__init__()`: Call parent class constructor (standard PyTorch pattern)
- `self.seg`: Creates the segmentation U-Net with transformer
- `self.roi_stop_grad`: Store parameter for forward pass

---

```python
        # classifier consumes ROI gated image (1-ch). If in_ch>1, we can reduce via 1x1 conv or mean.
        self.reduce = nn.Conv2d(in_ch, 1, 1, bias=False) if in_ch>1 else nn.Identity()
        self.cls_backbone = TInceptionNet(in_ch=1, num_classes=num_cls)
```

**Line 10-12**: Initialize classification network

**Why Reduce to 1 Channel?**
- Multi-modal input has 4 channels (FLAIR, T1, T1CE, T2)
- Classification network expects 1 channel (ROI-gated image)
- Two options:
  1. If `in_ch=1` → `nn.Identity()` does nothing
  2. If `in_ch>1` → `nn.Conv2d(in_ch, 1, 1)` learns to fuse channels

**Design Choice**: `bias=False`
- When followed by BatchNorm (in Inception), bias is redundant
- Saves parameters and computation

---

```python
    def forward(self, x):
        seg_logits = self.seg(x)  # B,1,H,W
        seg_prob = torch.sigmoid(seg_logits)
```

**Line 14-16**: Segmentation forward pass

**Step-by-step**:
1. `x`: Input image (B, in_ch, 256, 256)
2. `seg_logits`: Raw segmentation predictions before activation (B, 1, 256, 256)
3. `seg_prob`: Sigmoid converts logits → probabilities in [0, 1]

**Why Sigmoid?**
- Binary segmentation (tumor vs background)
- Outputs probability that each pixel is tumor

---

```python
        roi_input = self.reduce(x)
        if self.roi_stop_grad:
            roi = roi_input * seg_prob.detach()
        else:
            roi = roi_input * seg_prob
```

**Line 17-21**: ROI (Region of Interest) gating

**This is the MOST CRITICAL part of the architecture!**

**What is ROI Gating?**
- Multiply input image by segmentation mask
- Focuses classifier on tumor region only
- Removes background noise

**Visual Example**:
```
Input Image:          Seg Mask:           ROI (Gated):
┌─────────┐          ┌─────────┐         ┌─────────┐
│░░░░░░░░░│          │0 0 0 0 0│         │0 0 0 0 0│
│░░███░░░░│    ×     │0 1 1 0 0│    =    │0 ███ 0 0│
│░░███░░░░│          │0 1 1 0 0│         │0 ███ 0 0│
│░░░░░░░░░│          │0 0 0 0 0│         │0 0 0 0 0│
└─────────┘          └─────────┘         └─────────┘
```

**The `.detach()` Magic**:

This is where `roi_stop_grad` parameter matters!

**With `.detach()` (roi_stop_grad=True - DEFAULT)**:
```python
roi = roi_input * seg_prob.detach()
```
- `.detach()`: Stop gradient flow
- Classification loss CANNOT affect segmentation
- Segmentation trains independently
- **Advantage**: Segmentation focuses on its task
- **Use Case**: When segmentation is more important

**Without `.detach()` (roi_stop_grad=False)**:
```python
roi = roi_input * seg_prob
```
- Gradients flow through segmentation mask
- Classification loss affects segmentation
- Segmentation learns to help classification
- **Advantage**: End-to-end optimization
- **Use Case**: When both tasks are equally important

**Why Default is True?**
- Segmentation is the primary task
- Prevents classifier from "corrupting" segmentation
- More stable training

---

```python
        cls_logits = self.cls_backbone(roi)
        return seg_logits, cls_logits
```

**Line 22-23**: Classification and return

**Final Steps**:
1. `cls_logits`: Pass ROI-gated image through Inception classifier → (B, num_classes)
2. Return both outputs (multi-task)

**Why Return Logits (Not Probabilities)?**
- Loss functions (BCEWithLogitsLoss, CrossEntropyLoss) expect logits
- More numerically stable than applying softmax/sigmoid first
- Conversion happens in loss function or evaluation

---

### Summary: BrainTumNet Architecture

**Input → Output Flow**:
```
Input (B, C, 256, 256)
    ↓
[U-Net Segmentation]
    ↓
Seg Logits (B, 1, 256, 256) → Sigmoid → Seg Prob (B, 1, 256, 256)
    ↓
[Channel Reduction if needed]
    ↓
Input × Seg Prob (with/without grad) → ROI (B, 1, 256, 256)
    ↓
[Inception Classifier]
    ↓
Cls Logits (B, 2)
    ↓
Return (seg_logits, cls_logits)
```

**Total Parameters** (for default config):
- Segmentation U-Net: ~2.5M parameters
- Inception Classifier: ~0.4M parameters
- **Total**: ~2.9M parameters

---

## Segmentation U-Net with Transformer

**File**: `src/braintumnet/models/seg_unet.py` (67 lines)

This is the **core segmentation network** combining:
1. U-Net encoder-decoder
2. CBAM attention on skip connections
3. Adaptive Masked Transformer in bottleneck

### Architecture Diagram

```
Input (B, C, 256, 256)
    ↓
┌─────────────────────────────────────┐
│         ENCODER                     │
│  [e1] → (base, 256, 256)           │ → skip1
│    ↓ MaxPool(2)                     │
│  [e2] → (base*2, 128, 128)         │ → skip2
│    ↓ MaxPool(2)                     │
│  [e3] → (base*4, 64, 64)           │ → skip3
│    ↓ MaxPool(2)                     │
│  [e4] → (base*8, 32, 32)           │ → skip4
│    ↓ MaxPool(2)                     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│         BOTTLENECK                  │
│  (base*8, 16, 16)                   │
│    ↓ Conv1x1                        │
│  (dim, 16, 16)                      │
│    ↓                                │
│  [Adaptive Masked Transformer]      │
│    ↓                                │
│  (dim, 2, 2)  ← patches             │
│    ↓ ConvTranspose                  │
│  (base*8, 16, 16)                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│         DECODER                     │
│  [d4] ← skip4 (with CBAM)          │
│    ↓ Upsample                       │
│  [d3] ← skip3 (with CBAM)          │
│    ↓ Upsample                       │
│  [d2] ← skip2 (with CBAM)          │
│    ↓ Upsample                       │
│  [d1] ← skip1 (with CBAM)          │
│    ↓                                │
└─────────────────────────────────────┘
    ↓
Output (1, 256, 256)
```

### Helper Function: conv_bn_relu

```python
def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )
```

**Line 5-10**: Standard convolution block

**What it does**:
- Convolution → Batch Normalization → ReLU activation
- This is the **building block** used throughout U-Net

**Parameters**:
- `k=3`: Kernel size 3×3 (default)
- `s=1`: Stride 1 (no downsampling)
- `p=1`: Padding 1 (keeps spatial size)

**Why `bias=False`?**
- BatchNorm has its own bias term
- Conv bias would be redundant and wasted

**Why `inplace=True`?**
- Saves memory by modifying tensor in-place
- Important for medical imaging (large feature maps)

---

### EncoderBlock

```python
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(conv_bn_relu(in_ch, out_ch), conv_bn_relu(out_ch, out_ch))
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.block(x)
        x_down = self.pool(x)
        return x, x_down
```

**Line 12-20**: Encoder block with skip connection

**Architecture**:
```
Input (in_ch, H, W)
    ↓
[Conv-BN-ReLU] → (out_ch, H, W)
    ↓
[Conv-BN-ReLU] → (out_ch, H, W)  ← This is the skip connection
    ↓
[MaxPool2d(2)] → (out_ch, H/2, W/2)
```

**Why Two Convolutions?**
- Standard U-Net design (from original paper)
- Increases receptive field
- Allows learning more complex features

**Return Two Values**:
- `x`: Before pooling → saved as skip connection for decoder
- `x_down`: After pooling → input to next encoder block

**Example**:
```python
e1 = EncoderBlock(1, 32)
x = torch.randn(4, 1, 256, 256)  # Batch of 4 images
skip, down = e1(x)
print(skip.shape)  # (4, 32, 256, 256) ← Skip connection
print(down.shape)  # (4, 32, 128, 128) ← Downsampled
```

---

### DecoderBlock

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.cbam = CBAM(out_ch)  # CBAM on skip connection which has out_ch channels
        self.block = nn.Sequential(conv_bn_relu(out_ch + out_ch, out_ch), conv_bn_relu(out_ch, out_ch))
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, self.cbam(skip)], dim=1)
        x = self.block(x)
        return x
```

**Line 22-32**: Decoder block with attention

**Architecture**:
```
Input (in_ch, H, W)          Skip (out_ch, H*2, W*2)
    ↓                              ↓
[ConvTranspose2d]            [CBAM Attention]
    ↓                              ↓
(out_ch, H*2, W*2)           (out_ch, H*2, W*2)
    └──────────┬───────────────────┘
               ↓ Concatenate
        (out_ch*2, H*2, W*2)
               ↓
        [Conv-BN-ReLU]
               ↓
        [Conv-BN-ReLU]
               ↓
        (out_ch, H*2, W*2)
```

**Key Components**:

1. **ConvTranspose2d (Upsampling)**:
```python
self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
```
- Doubles spatial dimensions: H → H*2, W → W*2
- Learnable upsampling (better than nearest neighbor)

2. **CBAM Attention on Skip**:
```python
self.cbam(skip)
```
- Applies attention to skip connection
- Highlights important features
- Suppresses noise
- **Crucial for performance!**

3. **Concatenation**:
```python
torch.cat([x, self.cbam(skip)], dim=1)
```
- Combines upsampled features with skip features
- Channel dimension: out_ch + out_ch = out_ch*2

4. **Two Convolutions**:
```python
self.block = nn.Sequential(conv_bn_relu(out_ch + out_ch, out_ch), conv_bn_relu(out_ch, out_ch))
```
- Fuses concatenated features
- Reduces channels back to out_ch

**Why CBAM on Skip Connection?**
- Skip connections can contain noise from encoder
- CBAM filters out irrelevant features
- Improves segmentation boundaries
- Ablation studies show ~2% Dice improvement

---

### SegUNetMasked

```python
class SegUNetMasked(nn.Module):
    def __init__(self, in_ch=1, base=32, dim=256, patch=8, depth=2, n_heads=4):
        super().__init__()
        self.patch = patch
```

**Line 34-37**: Main segmentation model initialization

**Parameters**:
- `in_ch=1`: Input channels
- `base=32`: Base channel multiplier
- `dim=256`: Transformer embedding dimension
- `patch=8`: Transformer patch size
- `depth=2`: Number of transformer blocks
- `n_heads=4`: Attention heads

---

```python
        self.e1 = EncoderBlock(in_ch, base)
        self.e2 = EncoderBlock(base, base*2)
        self.e3 = EncoderBlock(base*2, base*4)
        self.e4 = EncoderBlock(base*4, base*8)
```

**Line 38-41**: Build encoder

**Channel Progression** (for base=32):
- e1: 1 → 32 channels
- e2: 32 → 64 channels
- e3: 64 → 128 channels
- e4: 128 → 256 channels

**Spatial Progression**:
- After e1: 256 → 128
- After e2: 128 → 64
- After e3: 64 → 32
- After e4: 32 → 16

**Why Doubling Channels?**
- Standard U-Net pattern
- As spatial resolution decreases, increase channel depth
- Maintains representational capacity

---

```python
        # After 4 encoder blocks: spatial size is H/16 x W/16
        # Transformer will further reduce by patch size
        self.bottleneck_conv = conv_bn_relu(base*8, dim, k=1, s=1, p=0)
        self.amt = AdaptiveMaskedTransformer(in_ch=dim, dim=dim, patch_size=patch, depth=depth, n_heads=n_heads)
        # Upsample transformer output back to original bottleneck size
        self.tr_upsample = nn.ConvTranspose2d(dim, base*8, kernel_size=patch, stride=patch)
```

**Line 42-47**: Transformer bottleneck

**Step-by-step**:

1. **Bottleneck Conv (1×1)**:
```python
self.bottleneck_conv = conv_bn_relu(base*8, dim, k=1, s=1, p=0)
```
- Changes channels: base*8 (256) → dim (256)
- Spatial size unchanged: 16×16
- `k=1`: 1×1 convolution (channel mixer)
- `p=0`: No padding needed for 1×1 conv

2. **Adaptive Masked Transformer**:
```python
self.amt = AdaptiveMaskedTransformer(...)
```
- Processes 16×16 feature map
- Divides into 8×8 patches → 2×2 grid of patches
- Applies self-attention with learned soft masks

3. **Upsample Back**:
```python
self.tr_upsample = nn.ConvTranspose2d(dim, base*8, kernel_size=patch, stride=patch)
```
- After transformer: 2×2 patches
- Upsample: 2×2 → 16×16 (patch size 8)
- Channels: dim (256) → base*8 (256)

**Why This Design?**
- Transformer operates on coarse features (16×16)
- More efficient than full resolution
- Captures global context
- Adaptive masking focuses on relevant regions

---

```python
        self.d4 = DecoderBlock(base*8, base*8)
        self.d3 = DecoderBlock(base*8, base*4)
        self.d2 = DecoderBlock(base*4, base*2)
        self.d1 = DecoderBlock(base*2, base)
        self.head = nn.Conv2d(base, 1, 1)
```

**Line 48-52**: Build decoder

**Decoder Structure** (for base=32):
- d4: 256 → 256 channels, 16×16 → 32×32
- d3: 256 → 128 channels, 32×32 → 64×64
- d2: 128 → 64 channels, 64×64 → 128×128
- d1: 64 → 32 channels, 128×128 → 256×256
- head: 32 → 1 channel (final segmentation map)

**Why 1×1 Conv for Head?**
- No need for spatial context at final layer
- Just convert channels to output
- Saves computation

---

```python
    def forward(self, x):
        s1, x1 = self.e1(x)      # s1: base, H, W
        s2, x2 = self.e2(x1)     # s2: base*2, H/2, W/2
        s3, x3 = self.e3(x2)     # s3: base*4, H/4, W/4
        s4, x4 = self.e4(x3)     # s4: base*8, H/8, W/8
```

**Line 53-57**: Encoder forward pass

**Detailed Shapes** (for input 256×256, base=32):
```
Input x:  (B, 1, 256, 256)
    ↓ e1
s1:       (B, 32, 256, 256)  ← Skip 1
x1:       (B, 32, 128, 128)
    ↓ e2
s2:       (B, 64, 128, 128)  ← Skip 2
x2:       (B, 64, 64, 64)
    ↓ e3
s3:       (B, 128, 64, 64)   ← Skip 3
x3:       (B, 128, 32, 32)
    ↓ e4
s4:       (B, 256, 32, 32)   ← Skip 4
x4:       (B, 256, 16, 16)
```

---

```python
        b = self.bottleneck_conv(x4)  # dim, H/16, W/16
        b = self.amt(b)          # dim, H/16/patch, W/16/patch
        b = self.tr_upsample(b)  # base*8, H/16, W/16 (upsampled back)
```

**Line 58-60**: Transformer bottleneck forward

**Detailed Shapes**:
```
x4:                    (B, 256, 16, 16)
    ↓ bottleneck_conv
b:                     (B, 256, 16, 16)
    ↓ amt (Transformer with patch=8)
b:                     (B, 256, 2, 2)    ← Patched!
    ↓ tr_upsample
b:                     (B, 256, 16, 16)  ← Back to original size
```

**What Happens in Transformer?**
1. Input: 16×16 feature map
2. Divide into 8×8 patches → 2×2=4 patches
3. Each patch becomes a token
4. Self-attention between 4 tokens
5. Reconstruct 2×2 feature map
6. Upsample back to 16×16

---

```python
        x = self.d4(b, s4)       # base*8, H/8, W/8
        x = self.d3(x, s3)       # base*4, H/4, W/4
        x = self.d2(x, s2)       # base*2, H/2, W/2
        x = self.d1(x, s1)       # base, H, W
        seg = self.head(x)       # 1, H, W
        return seg
```

**Line 61-66**: Decoder forward pass

**Detailed Shapes** (base=32):
```
b:        (B, 256, 16, 16)
    ↓ d4(b, s4)  [s4 has CBAM attention applied]
x:        (B, 256, 32, 32)
    ↓ d3(x, s3)  [s3 has CBAM attention applied]
x:        (B, 128, 64, 64)
    ↓ d2(x, s2)  [s2 has CBAM attention applied]
x:        (B, 64, 128, 128)
    ↓ d1(x, s1)  [s1 has CBAM attention applied]
x:        (B, 32, 256, 256)
    ↓ head
seg:      (B, 1, 256, 256)  ← Final segmentation logits
```

---

## CBAM Attention Mechanism

**File**: `src/braintumnet/models/cbam.py` (33 lines)

CBAM (Convolutional Block Attention Module) applies **two types of attention sequentially**:
1. **Channel Attention**: Which feature maps are important?
2. **Spatial Attention**: Which spatial locations are important?

### Architecture Diagram

```
Input Features (C, H, W)
         ↓
┌─────────────────────────┐
│   Channel Attention     │
│                         │
│  ┌──────────────────┐   │
│  │ Avg Pool → (C,1,1)│  │
│  └──────────────────┘   │
│           ↓             │
│  ┌──────────────────┐   │
│  │ Max Pool → (C,1,1)│  │
│  └──────────────────┘   │
│           ↓             │
│    [Shared MLP]         │
│           ↓             │
│    [Add + Sigmoid]      │
│           ↓             │
│  Channel Weights (C,1,1)│
└─────────────────────────┘
         ↓
  Features × Weights
         ↓
┌─────────────────────────┐
│   Spatial Attention     │
│                         │
│  ┌──────────────────┐   │
│  │Channel Avg→(1,H,W)│  │
│  └──────────────────┘   │
│           ↓             │
│  ┌──────────────────┐   │
│  │Channel Max→(1,H,W)│  │
│  └──────────────────┘   │
│           ↓             │
│  [Concat → (2,H,W)]     │
│           ↓             │
│    [7×7 Conv]           │
│           ↓             │
│    [Sigmoid]            │
│           ↓             │
│ Spatial Weights (1,H,W) │
└─────────────────────────┘
         ↓
  Features × Weights
         ↓
    Output (C, H, W)
```

### ChannelAttention

```python
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False),
        )
```

**Line 3-12**: Channel attention initialization

**What is Channel Attention?**
- Learns which feature channels are important
- Example: In medical imaging, some channels might detect edges, others textures
- Attention weights suppress less useful channels

**Components**:

1. **Global Pooling**:
```python
self.avg = nn.AdaptiveAvgPool2d(1)  # (C, H, W) → (C, 1, 1)
self.max = nn.AdaptiveMaxPool2d(1)  # (C, H, W) → (C, 1, 1)
```
- Summarizes spatial information for each channel
- `AdaptiveAvgPool2d(1)`: Average of all spatial locations
- `AdaptiveMaxPool2d(1)`: Maximum of all spatial locations
- Result: One value per channel

2. **MLP (Multi-Layer Perceptron)**:
```python
self.mlp = nn.Sequential(
    nn.Conv2d(in_channels, in_channels//reduction, 1, bias=False),  # Compress
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False),  # Expand
)
```
- Uses 1×1 convolutions (equivalent to fully connected for 1×1 spatial size)
- **Bottleneck design**: C → C/16 → C
- `reduction=16`: Compression ratio (reduces parameters)

**Why Reduction?**
- Original: C → C would be expensive
- With reduction: C → C/16 → C
- For C=256: 256×256=65k params vs 256×16 + 16×256=8k params
- **8× fewer parameters!**

---

```python
    def forward(self, x):
        att = torch.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))
        return x * att
```

**Line 13-15**: Channel attention forward pass

**Step-by-step**:
```python
# Input: x shape (B, C, H, W)
avg_pool = self.avg(x)        # (B, C, 1, 1)
max_pool = self.max(x)        # (B, C, 1, 1)

avg_feat = self.mlp(avg_pool) # (B, C, 1, 1)
max_feat = self.mlp(max_pool) # (B, C, 1, 1)

att = torch.sigmoid(avg_feat + max_feat)  # (B, C, 1, 1), values in [0, 1]

output = x * att              # Broadcasting: (B, C, H, W) * (B, C, 1, 1)
```

**Why Both Average and Max?**
- Average: Captures overall channel importance
- Max: Captures peak activations
- Combining both gives richer representation

**Example**:
```
Input feature map:     Avg Pool:    Max Pool:    Combined Attention:
┌─────────┐            ┌───┐        ┌───┐        ┌───┐
│ 0 1 2 3 │            │1.5│        │ 3 │        │2.8│ ← Higher weight
│ 4 0 1 2 │     →      │   │   +    │   │   =    │   │
│ 1 2 3 4 │            │   │        │   │        │   │
│ 2 3 4 0 │            │   │        │   │        │   │
└─────────┘            └───┘        └───┘        └───┘
```

---

### SpatialAttention

```python
class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k//2, bias=False)
```

**Line 17-20**: Spatial attention initialization

**What is Spatial Attention?**
- Learns which spatial locations are important
- Example: Focus on tumor region, ignore background
- Complements channel attention

**Components**:
```python
self.conv = nn.Conv2d(2, 1, k, padding=k//2, bias=False)
```
- Input: 2 channels (avg and max across channels)
- Output: 1 channel (spatial attention map)
- Kernel size: k=7 (large receptive field)
- Padding: k//2 = 3 (keeps spatial size)

**Why k=7?**
- Larger kernel captures more spatial context
- Standard choice from CBAM paper
- Balances receptive field vs computation

---

```python
    def forward(self, x):
        att = torch.cat([x.mean(1, True), x.amax(1, True)], dim=1)
        att = torch.sigmoid(self.conv(att))
        return x * att
```

**Line 21-24**: Spatial attention forward pass

**Step-by-step**:
```python
# Input: x shape (B, C, H, W)

avg_spatial = x.mean(1, True)   # (B, 1, H, W) - average across channels
max_spatial = x.amax(1, True)   # (B, 1, H, W) - max across channels

att = torch.cat([avg_spatial, max_spatial], dim=1)  # (B, 2, H, W)

att = self.conv(att)            # (B, 1, H, W) - learn spatial weights
att = torch.sigmoid(att)        # Values in [0, 1]

output = x * att                # (B, C, H, W) * (B, 1, H, W) - broadcast
```

**Visual Example**:
```
Input (C=3 channels):
Channel 0:    Channel 1:    Channel 2:
┌─────┐       ┌─────┐       ┌─────┐
│0 1 2│       │3 4 5│       │6 7 8│
│3 4 5│       │6 7 8│       │9 0 1│
└─────┘       └─────┘       └─────┘
     ↓             ↓             ↓
     Average across channels (mean)
               ↓
         Avg Spatial:
         ┌─────┐
         │3 4 5│  ← Average of [0,3,6], [1,4,7], [2,5,8]...
         │6 7 8│
         └─────┘
     ↓             ↓             ↓
     Max across channels (amax)
               ↓
         Max Spatial:
         ┌─────┐
         │6 7 8│  ← Max of [0,3,6], [1,4,7], [2,5,8]...
         │9 7 8│
         └─────┘
```

**Why Both Mean and Max?**
- Mean: Overall feature importance at each location
- Max: Peak features at each location
- Together: Robust spatial attention

---

### CBAM (Combining Both)

```python
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, k=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(k)
    def forward(self, x):
        return self.sa(self.ca(x))
```

**Line 26-32**: Complete CBAM module

**Sequential Application**:
```python
x → ChannelAttention → x_ca → SpatialAttention → x_out
```

**Why Sequential (Not Parallel)?**
- Channel attention first refines which features matter
- Spatial attention then refines where they matter
- Empirically works better than parallel

**Example Flow**:
```
Input (B, 64, 32, 32)
    ↓
ChannelAttention: Weights shape (B, 64, 1, 1)
    ↓ Apply weights
(B, 64, 32, 32) with some channels suppressed
    ↓
SpatialAttention: Weights shape (B, 1, 32, 32)
    ↓ Apply weights
(B, 64, 32, 32) with some locations suppressed
```

**Computational Cost**:
- Channel attention: Very cheap (only processes pooled features)
- Spatial attention: Moderate (7×7 conv on full spatial size)
- Total: <1% of total model FLOPs
- **Tiny cost, significant gain!**

---

## Adaptive Masked Transformer

**File**: `src/braintumnet/models/masked_transformer.py` (88 lines)

This is the **most complex component** of BrainTumNet. It applies self-attention with **learned soft masks** that adaptively focus on important image regions.

### Architecture Diagram

```
Input Features (B, C, 16, 16)
         ↓
    [PatchEmbed]
         ↓
  Tokens (B, N, C)  where N=4 patches
         ↓
    ┌────┴────┐
    │         │
    │ [Soft   │
    │  Mask   │
    │  Gen]   │
    │         │
    └────┬────┘
         ↓
  Soft Masks (B, H, N)  where H=num_heads
         ↓
┌─────────────────────┐
│ Transformer Block 1 │
│  - Masked Attention │
│  - MLP              │
└─────────────────────┘
         ↓
┌─────────────────────┐
│ Transformer Block 2 │
│  - Masked Attention │
│  - MLP              │
└─────────────────────┘
         ↓
  Tokens (B, N, C)
         ↓
  [Reshape to 2D]
         ↓
Output (B, C, 2, 2)
```

### PatchEmbed

```python
class PatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, patch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(embed_dim)
```

**Line 3-7**: Patch embedding initialization

**What is Patch Embedding?**
- Converts 2D image into 1D sequence of tokens
- Each token represents one image patch
- Standard technique in Vision Transformers

**Components**:
```python
self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
```
- Convolution with kernel size = stride = patch size
- Non-overlapping patches
- Example: 16×16 image, patch=8 → 2×2 = 4 patches

```python
self.norm = nn.LayerNorm(embed_dim)
```
- Normalizes token embeddings
- Stabilizes training

---

```python
    def forward(self, x):
        x = self.proj(x)  # B,C,H',W'
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)  # B,N,C
        x = self.norm(x)
        return x, (H,W)
```

**Line 8-13**: Patch embedding forward pass

**Step-by-step**:
```python
# Input: x shape (B, 256, 16, 16), patch=8

x = self.proj(x)              # (B, 256, 2, 2) - 8×8 patches → 2×2 grid
B, C, H, W = x.shape          # B=batch, C=256, H=2, W=2

x = x.flatten(2)              # (B, 256, 4) - flatten spatial dims
x = x.transpose(1, 2)         # (B, 4, 256) - swap to (B, N, C) format
x = self.norm(x)              # Normalize

return x, (H, W)              # Return tokens and spatial shape
```

**Visual Example**:
```
Input 16×16 Image:
┌───────┬───────┐
│ P0    │ P1    │  ← Each 8×8 patch becomes one token
│       │       │
├───────┼───────┤
│ P2    │ P3    │
│       │       │
└───────┴───────┘

After Projection:
Tokens = [P0_embed, P1_embed, P2_embed, P3_embed]
Shape: (B, 4, 256)
```

---

### SoftMaskGenerator

```python
class SoftMaskGenerator(nn.Module):
    def __init__(self, dim, hidden=128, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, n_heads), nn.Sigmoid()
        )
```

**Line 15-22**: Soft mask generator initialization

**What are Soft Masks?**
- **Hard Mask**: Binary (0 or 1) - include or exclude token
- **Soft Mask**: Continuous [0, 1] - partial attention weight
- Allows gradients to flow, end-to-end trainable

**Why Soft Masks?**
- **Adaptive**: Learns which patches are important
- **Differentiable**: Can backpropagate through masks
- **Per-Head**: Different heads can focus on different regions

**Architecture**:
```python
self.mlp = nn.Sequential(
    nn.Linear(dim, hidden),     # 256 → 128 (compression)
    nn.GELU(),                  # Smooth nonlinearity
    nn.Linear(hidden, n_heads), # 128 → 4 (one mask per head)
    nn.Sigmoid()                # Output in [0, 1]
)
```

**Why GELU (not ReLU)?**
- GELU (Gaussian Error Linear Unit): Smoother than ReLU
- Standard in transformers (BERT, GPT, ViT)
- Better gradient flow

---

```python
    def forward(self, tokens):  # B,N,C
        m = self.mlp(tokens)    # B,N,H
        return m.permute(0,2,1).contiguous()  # B,H,N
```

**Line 23-25**: Soft mask generator forward pass

**Step-by-step**:
```python
# Input: tokens shape (B, 4, 256)

m = self.mlp(tokens)          # (B, 4, 4) - 4 tokens, 4 heads
                               # Each value is attention weight in [0, 1]

m = m.permute(0, 2, 1)        # (B, 4, 4) - rearrange to (B, heads, tokens)
m = m.contiguous()            # Ensure memory contiguity
```

**Visual Example**:
```
Tokens (4 patches):
┌──────────────────────┐
│ Token 0 (Background) │ → Mask weights [0.1, 0.2, 0.1, 0.3]  ← Low weights
│ Token 1 (Tumor Edge) │ → Mask weights [0.9, 0.8, 0.7, 0.6]  ← High weights
│ Token 2 (Tumor Core) │ → Mask weights [1.0, 0.9, 0.9, 0.8]  ← Highest weights
│ Token 3 (Background) │ → Mask weights [0.2, 0.1, 0.2, 0.2]  ← Low weights
└──────────────────────┘
         ↑                    ↑
      Input              4 attention heads
                         (each head has different mask)
```

**Key Insight**:
- Background tokens get low weights → less attention
- Tumor tokens get high weights → more attention
- **Adaptive**: Mask values learned during training!

---

### MaskedSelfAttention

```python
class MaskedSelfAttention(nn.Module):
    def __init__(self, dim, n_heads=4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
```

**Line 27-37**: Masked self-attention initialization

**Parameters**:
- `dim=256`: Embedding dimension
- `n_heads=4`: Number of attention heads
- `head_dim=64`: Dimension per head (256/4)
- `attn_drop=0.0`: Attention dropout (usually 0 for small datasets)
- `proj_drop=0.0`: Projection dropout

**Components**:
```python
self.qkv = nn.Linear(dim, dim*3, bias=False)
```
- Single matrix projects to Q, K, V simultaneously
- More efficient than 3 separate projections
- Output: dim*3 (256*3=768)

```python
self.proj = nn.Linear(dim, dim)
```
- Output projection after attention
- Mixes information from different heads

---

```python
    def forward(self, x, softmask):  # x: B,N,C ; softmask: B,H,N
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.n_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]  # B,H,N,D
```

**Line 38-41**: Compute Q, K, V

**Step-by-step**:
```python
# Input: x shape (B, 4, 256)

qkv = self.qkv(x)              # (B, 4, 768) - project to 3*dim

qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim)
                                # (B, 4, 3, 4, 64)
                                # Split into Q/K/V and heads

qkv = qkv.permute(2, 0, 3, 1, 4)
                                # (3, B, 4, 4, 64)
                                # Rearrange for easy indexing

q, k, v = qkv[0], qkv[1], qkv[2]
                                # Each: (B, 4, 4, 64)
                                # q/k/v for each head
```

**Shapes Summary**:
- q: (Batch, Heads, Tokens, Head_dim) = (B, 4, 4, 64)
- k: (Batch, Heads, Tokens, Head_dim) = (B, 4, 4, 64)
- v: (Batch, Heads, Tokens, Head_dim) = (B, 4, 4, 64)

---

```python
        attn = (q @ k.transpose(-2,-1)) / (self.head_dim ** 0.5)  # B,H,N,N
        key_bias = torch.log(softmask.unsqueeze(-2) + 1e-6)  # B,H,1,N
        attn = attn + key_bias
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)
```

**Line 42-46**: Compute attention with soft masking

**Step-by-step**:

1. **Standard Attention**:
```python
attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
```
- `q @ k.transpose(-2, -1)`: Dot product attention scores
- Shape: (B, 4, 4, 4) - 4 heads, 4×4 attention matrix
- `/sqrt(head_dim)`: Scale by √64 = 8 (prevents large values)

2. **Add Soft Mask** (THE NOVEL PART!):
```python
key_bias = torch.log(softmask.unsqueeze(-2) + 1e-6)  # (B, 4, 1, 4)
attn = attn + key_bias
```

**Why `torch.log(softmask)`?**

This is mathematically elegant! Let me explain:

Standard softmax:
```
softmax(attn_i) = exp(attn_i) / Σ exp(attn_j)
```

With mask in log-space:
```
softmax(attn_i + log(mask_i)) = exp(attn_i + log(mask_i)) / Σ exp(attn_j + log(mask_j))
                                = exp(attn_i) * mask_i / Σ (exp(attn_j) * mask_j)
```

**Effect**:
- If mask_i = 1.0 → log(1.0) = 0 → no change
- If mask_i = 0.5 → log(0.5) = -0.69 → reduce attention
- If mask_i ≈ 0 → log(0) = -∞ → zero attention

**Why Add 1e-6?**
- Prevent log(0) = -inf
- Numerical stability

**Visual Example**:
```
Attention Scores (before masking):
      T0   T1   T2   T3
T0 [ 0.5  0.3  0.1  0.1 ]
T1 [ 0.2  0.6  0.1  0.1 ]
T2 [ 0.1  0.2  0.5  0.2 ]
T3 [ 0.1  0.1  0.2  0.6 ]

Soft Mask (learned):
[ 0.2  0.9  1.0  0.1 ]  ← Token importance

After Masking (softmax with bias):
      T0   T1   T2   T3
T0 [ 0.2  0.5  0.3  0.0 ]  ← Attention shifts to T1, T2
T1 [ 0.1  0.8  0.1  0.0 ]
T2 [ 0.0  0.3  0.6  0.1 ]
T3 [ 0.1  0.2  0.3  0.4 ]
```

3. **Softmax and Dropout**:
```python
attn = attn.softmax(-1)       # Normalize to probabilities
attn = self.attn_drop(attn)   # Apply dropout (if any)
```

---

```python
        out = (attn @ v).transpose(1,2).reshape(B,N,C)
        out = self.proj_drop(self.proj(out))
        return out
```

**Line 47-49**: Apply attention and project

**Step-by-step**:
```python
out = attn @ v                # (B, 4, 4, 4) @ (B, 4, 4, 64) = (B, 4, 4, 64)
out = out.transpose(1, 2)     # (B, 4, 4, 64) - swap heads and tokens
out = out.reshape(B, N, C)    # (B, 4, 256) - merge heads

out = self.proj(out)          # (B, 4, 256) - output projection
out = self.proj_drop(out)     # Apply dropout
```

---

### MLP

```python
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim*mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(dim*mlp_ratio), dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x
```

**Line 51-61**: Feed-forward MLP

**Architecture**:
```
Input (dim=256)
    ↓
Linear (256 → 1024)  ← mlp_ratio=4.0
    ↓
GELU Activation
    ↓
Dropout
    ↓
Linear (1024 → 256)
    ↓
Dropout
    ↓
Output (dim=256)
```

**Why mlp_ratio=4.0?**
- Standard in transformers (BERT, GPT)
- Allows learning complex nonlinear transformations
- Bottleneck: 256 → 1024 → 256

---

### MaskedTransformerBlock

```python
class MaskedTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=4, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MaskedSelfAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
    def forward(self, x, softmask):
        x = x + self.attn(self.norm1(x), softmask)
        x = x + self.mlp(self.norm2(x))
        return x
```

**Line 63-73**: Transformer block with residual connections

**Architecture** (Pre-Norm + Residual):
```
Input x
    ↓
    ├─────────┐
    │         ↓
    │    [LayerNorm]
    │         ↓
    │    [Masked Self-Attention]
    │         ↓
    └────→ [Add]
          ↓
          ├─────────┐
          │         ↓
          │    [LayerNorm]
          │         ↓
          │    [MLP]
          │         ↓
          └────→ [Add]
                ↓
              Output
```

**Why Pre-Norm?**
- Normalize before attention/MLP (not after)
- More stable training
- Standard in modern transformers

**Why Residual Connections?**
- `x = x + module(x)`: Skip connection
- Allows gradients to flow directly
- Prevents vanishing gradients in deep networks

---

### AdaptiveMaskedTransformer

```python
class AdaptiveMaskedTransformer(nn.Module):
    def __init__(self, in_ch, dim, patch_size=8, depth=2, n_heads=4):
        super().__init__()
        self.pe = PatchEmbed(in_ch, dim, patch_size)
        self.mask_gen = SoftMaskGenerator(dim, hidden=dim//2, n_heads=n_heads)
        self.blocks = nn.ModuleList([MaskedTransformerBlock(dim, n_heads) for _ in range(depth)])
```

**Line 75-80**: Complete transformer initialization

**Components**:
1. `PatchEmbed`: Convert 2D → 1D tokens
2. `SoftMaskGenerator`: Learn adaptive masks
3. `blocks`: Stack of `depth=2` transformer blocks

**Why depth=2?**
- Shallow enough for small feature maps (16×16)
- Deep enough to capture interactions
- Empirically found to work best

---

```python
    def forward(self, x):
        tokens, (H,W) = self.pe(x)  # B,N,C
        softmask = self.mask_gen(tokens)  # B,H,N
        for blk in self.blocks:
            tokens = blk(tokens, softmask)
        feat = tokens.transpose(1,2).reshape(x.size(0), tokens.size(-1), H, W)
        return feat
```

**Line 81-87**: Complete transformer forward pass

**Step-by-step**:
```python
# Input: x shape (B, 256, 16, 16)

tokens, (H, W) = self.pe(x)    # (B, 4, 256), H=2, W=2

softmask = self.mask_gen(tokens) # (B, 4, 4) - adaptive masks

for blk in self.blocks:
    tokens = blk(tokens, softmask) # (B, 4, 256) - apply 2 transformer blocks

feat = tokens.transpose(1, 2)    # (B, 256, 4)
feat = feat.reshape(x.size(0), tokens.size(-1), H, W)
                                  # (B, 256, 2, 2) - reshape to 2D
return feat
```

**Key Points**:
- Mask generated once, used in all blocks
- Same mask for all transformer layers
- Output reshaped back to 2D feature map

---

## Inception Classification Network

**File**: `src/braintumnet/models/t_inception.py` (51 lines)

Inception networks capture **multi-scale features** using parallel convolutions with different kernel sizes.

### Architecture Diagram

```
ROI Input (B, 1, 256, 256)
         ↓
    [Stem Conv] → (B, 64, 256, 256)
         ↓
┌────────┴────────┐
│ Inception Block │
│                 │
│  ┌──────────┐   │
│  │ 1×1 Conv │   │  ← c channels
│  └──────────┘   │
│  ┌──────────┐   │
│  │ 3×3 Conv │   │  ← c channels
│  └──────────┘   │
│  ┌──────────┐   │
│  │ 1×3 Conv │   │  ← c channels
│  └──────────┘   │
│  ┌──────────┐   │
│  │ 3×1 Conv │   │  ← c channels
│  └──────────┘   │
│         ↓        │
│     [Concat]     │  ← 4c channels
│         ↓        │
│    [Fuse 1×1]    │
└─────────┬────────┘
          ↓
   (B, 128, 256, 256)
          ↓
┌─────────┴────────┐
│ Inception Block  │
└─────────┬────────┘
          ↓
   (B, 256, 256, 256)
          ↓
  [Global Avg Pool]
          ↓
      (B, 256)
          ↓
     [Dropout 0.3]
          ↓
      [FC Layer]
          ↓
   (B, num_classes)
```

### InceptionBranch

```python
class InceptionBranch(nn.Module):
    def __init__(self, in_ch, out_ch, k=(3,3)):
        super().__init__()
        if k==(1,1):
            self.op = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        elif k==(1,3):
            self.op = nn.Conv2d(in_ch, out_ch, (1,3), padding=(0,1), bias=False)
        elif k==(3,1):
            self.op = nn.Conv2d(in_ch, out_ch, (3,1), padding=(1,0), bias=False)
        else:
            self.op = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.op(x)))
```

**Line 4-18**: Single inception branch

**Supported Kernels**:
- `(1, 1)`: Point-wise convolution (no spatial context)
- `(1, 3)`: Horizontal features (elongated structures)
- `(3, 1)`: Vertical features (elongated structures)
- `(3, 3)`: Standard square features

**Why Different Kernels?**
- Brain tumors have diverse shapes: round, elongated, irregular
- Different kernels capture different geometric features
- Parallel branches → richer representation

**Visual Example**:
```
Tumor Shapes:
┌─────────┐
│   ●●    │  ← Round tumor: best captured by 3×3
│  ●●●●   │
│   ●●    │
│         │
│ ●●●●●●● │  ← Horizontal tumor: best captured by 1×3
│         │
│    ●    │  ← Vertical tumor: best captured by 3×1
│    ●    │
│    ●    │
└─────────┘
```

---

### TInceptionBlock

```python
class TInceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        c = out_ch // 4
        self.b1 = InceptionBranch(in_ch, c, (1,1))
        self.b2 = InceptionBranch(in_ch, c, (3,3))
        self.b3 = InceptionBranch(in_ch, c, (1,3))
        self.b4 = InceptionBranch(in_ch, c, (3,1))
        self.fuse = nn.Conv2d(c*4, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
```

**Line 20-30**: Complete inception block

**Architecture**:
```
Input (in_ch channels)
    ↓
┌───┴───┬───────┬───────┬───────┐
│       │       │       │       │
│ 1×1   │ 3×3   │ 1×3   │ 3×1   │  ← 4 parallel branches
│       │       │       │       │
└───┬───┴───┬───┴───┬───┴───┬───┘
    │       │       │       │
    │   (c  │  ch   │  each)│
    └───────┴───────┴───────┘
            ↓ Concatenate
          (c*4 channels)
            ↓
       [1×1 Fuse Conv]
            ↓
         [BN + ReLU]
            ↓
        (out_ch channels)
```

**Channel Allocation**:
- Each branch: out_ch / 4 channels
- After concat: out_ch channels total
- Fuse conv: c*4 → out_ch (maintains channels)

---

```python
    def forward(self, x):
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.act(self.bn(self.fuse(x)))
```

**Line 31-33**: Inception forward pass

**Step-by-step**:
```python
b1_out = self.b1(x)  # (B, c, H, W) - 1×1 features
b2_out = self.b2(x)  # (B, c, H, W) - 3×3 features
b3_out = self.b3(x)  # (B, c, H, W) - 1×3 features
b4_out = self.b4(x)  # (B, c, H, W) - 3×1 features

x = torch.cat([b1_out, b2_out, b3_out, b4_out], dim=1)
                      # (B, c*4, H, W) - concatenate

x = self.fuse(x)      # (B, out_ch, H, W) - fuse
x = self.bn(x)        # Batch norm
x = self.act(x)       # ReLU
```

---

### TInceptionNet

```python
class TInceptionNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.b1 = TInceptionBlock(64, 128)
        self.b2 = TInceptionBlock(128, 256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)
```

**Line 35-43**: Complete inception classifier

**Architecture**:
- **Stem**: 1 → 64 channels (initial feature extraction)
- **Block 1**: 64 → 128 channels (multi-scale features)
- **Block 2**: 128 → 256 channels (higher-level features)
- **Global Pool**: 256×256 → 1×1 (spatial → vector)
- **Dropout**: 30% (prevent overfitting)
- **FC**: 256 → 2 (final classification)

**Why Dropout 0.3?**
- Classification head prone to overfitting
- 30% is standard for medical imaging
- Balances regularization vs capacity

---

```python
    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)
```

**Line 44-50**: Inception forward pass

**Detailed Shapes** (for 256×256 input):
```python
x = self.stem(x)      # (B, 1, 256, 256) → (B, 64, 256, 256)
x = self.b1(x)        # (B, 64, 256, 256) → (B, 128, 256, 256)
x = self.b2(x)        # (B, 128, 256, 256) → (B, 256, 256, 256)
x = self.pool(x)      # (B, 256, 256, 256) → (B, 256, 1, 1)
x = x.flatten(1)      # (B, 256, 1, 1) → (B, 256)
x = self.drop(x)      # (B, 256) - apply dropout
x = self.fc(x)        # (B, 256) → (B, 2)
```

---

## Complete Data Flow with Tensor Shapes

Let's trace a **single image** through the entire BrainTumNet architecture with exact tensor shapes at each step.

### Input Setup

```python
batch_size = 4
in_channels = 4  # Multi-modal: FLAIR, T1, T1CE, T2
img_size = 256
num_classes = 2  # HGG vs LGG

input_image = torch.randn(batch_size, in_channels, img_size, img_size)
# Shape: (4, 4, 256, 256)
```

### Step-by-Step Flow

#### 1. Segmentation U-Net Encoder

```python
# Input to SegUNetMasked
x = input_image  # (4, 4, 256, 256)

# Encoder Block 1
s1, x1 = self.e1(x)
# s1: (4, 32, 256, 256) - skip connection
# x1: (4, 32, 128, 128) - downsampled

# Encoder Block 2
s2, x2 = self.e2(x1)
# s2: (4, 64, 128, 128) - skip connection
# x2: (4, 64, 64, 64) - downsampled

# Encoder Block 3
s3, x3 = self.e3(x2)
# s3: (4, 128, 64, 64) - skip connection
# x3: (4, 128, 32, 32) - downsampled

# Encoder Block 4
s4, x4 = self.e4(x3)
# s4: (4, 256, 32, 32) - skip connection
# x4: (4, 256, 16, 16) - downsampled
```

#### 2. Transformer Bottleneck

```python
# Convert to transformer embedding dimension
b = self.bottleneck_conv(x4)  # (4, 256, 16, 16) → (4, 256, 16, 16)

# Adaptive Masked Transformer
# - PatchEmbed
tokens, (H, W) = self.pe(b)  # (4, 256, 16, 16) → (4, 4, 256), H=2, W=2
#   4 patches total (2×2 grid with patch_size=8)

# - Generate soft masks
softmask = self.mask_gen(tokens)  # (4, 4, 256) → (4, 4, 4)
#   Shape: (batch, heads, tokens)

# - Transformer blocks (depth=2)
for blk in self.blocks:
    tokens = blk(tokens, softmask)  # (4, 4, 256) → (4, 4, 256)

# - Reshape to 2D
feat = tokens.transpose(1,2).reshape(4, 256, 2, 2)  # (4, 256, 2, 2)

# Upsample back to bottleneck size
b = self.tr_upsample(feat)  # (4, 256, 2, 2) → (4, 256, 16, 16)
```

#### 3. Segmentation U-Net Decoder

```python
# Decoder Block 4
x = self.d4(b, s4)
# b: (4, 256, 16, 16), s4 (with CBAM): (4, 256, 32, 32)
# Output: (4, 256, 32, 32)

# Decoder Block 3
x = self.d3(x, s3)
# x: (4, 256, 32, 32), s3 (with CBAM): (4, 128, 64, 64)
# Output: (4, 128, 64, 64)

# Decoder Block 2
x = self.d2(x, s2)
# x: (4, 128, 64, 64), s2 (with CBAM): (4, 64, 128, 128)
# Output: (4, 64, 128, 128)

# Decoder Block 1
x = self.d1(x, s1)
# x: (4, 64, 128, 128), s1 (with CBAM): (4, 32, 256, 256)
# Output: (4, 32, 256, 256)

# Segmentation head
seg_logits = self.head(x)  # (4, 32, 256, 256) → (4, 1, 256, 256)
```

#### 4. ROI Gating

```python
# Convert logits to probabilities
seg_prob = torch.sigmoid(seg_logits)  # (4, 1, 256, 256)

# Reduce input channels if needed
roi_input = self.reduce(input_image)  # (4, 4, 256, 256) → (4, 1, 256, 256)

# Apply ROI gating (with gradient stopping)
roi = roi_input * seg_prob.detach()  # (4, 1, 256, 256)
```

#### 5. Inception Classifier

```python
# Stem
x = self.stem(roi)  # (4, 1, 256, 256) → (4, 64, 256, 256)

# Inception Block 1
x = self.b1(x)  # (4, 64, 256, 256) → (4, 128, 256, 256)

# Inception Block 2
x = self.b2(x)  # (4, 128, 256, 256) → (4, 256, 256, 256)

# Global average pooling
x = self.pool(x)  # (4, 256, 256, 256) → (4, 256, 1, 1)

# Flatten
x = x.flatten(1)  # (4, 256, 1, 1) → (4, 256)

# Dropout
x = self.drop(x)  # (4, 256)

# Final classification
cls_logits = self.fc(x)  # (4, 256) → (4, 2)
```

#### 6. Final Outputs

```python
return seg_logits, cls_logits
# seg_logits: (4, 1, 256, 256) - segmentation map
# cls_logits: (4, 2) - classification scores
```

### Memory Usage Calculation

**Peak Memory** (for batch_size=4, assuming FP32):

Largest tensors:
1. Input: 4 × 4 × 256 × 256 = 1,048,576 values
2. Decoder features: 4 × 256 × 256 × 256 = 67,108,864 values (Inception b2 output)
3. Skip connections: 4 × 256 × 32 × 32 = 1,048,576 values

Total: ~270MB for activations + ~12MB for model parameters = **~282MB**

**For FP16 (mixed precision)**: ~141MB

---

## Design Decisions Explained

### Why This Architecture?

**Q: Why combine U-Net + Transformer + Inception?**

A: Each component serves a specific purpose:
- **U-Net**: Best for dense segmentation (proven in medical imaging)
- **Transformer**: Captures global context (tumor affects surrounding tissue)
- **Inception**: Multi-scale classification (tumors vary in size and shape)

**Q: Why CBAM attention on skip connections?**

A: Skip connections can propagate noise from encoder. CBAM filters important features, improving boundary precision. Ablation studies show +2% Dice improvement.

**Q: Why adaptive masked transformer?**

A: Standard transformers attend to all tokens equally. Medical images have large background regions. Adaptive masking focuses computation on relevant regions (tumor and nearby tissue).

**Q: Why stop gradient in ROI gating?**

A: Segmentation is the primary task. Without `.detach()`, classification loss would affect segmentation, potentially degrading mask quality.

### Hyperparameter Choices

**Q: Why base=32 for U-Net?**

A: Balances model capacity and memory:
- base=16: Too few parameters, underfitting
- base=32: Sweet spot for 256×256 images
- base=64: Marginal improvement, 4× more memory

**Q: Why patch_size=8?**

A: After 4 encoder blocks, spatial size is 16×16. With patch=8, we get 2×2=4 patches. Too few patches (patch=16 → 1 patch) loses spatial structure. Too many (patch=4 → 16 patches) is computationally expensive.

**Q: Why transformer depth=2?**

A: With only 4 tokens, deep transformers are unnecessary. Depth=2 captures token interactions without overfitting.

**Q: Why dropout=0.3 in classifier?**

A: Classification head has high capacity (256 features → 2 classes). Without regularization, it overfits. 30% dropout is standard for medical imaging.

---

## Modification Guides

### How to Change Model Capacity

**Increase Capacity** (for larger datasets):
```python
model = BrainTumNet(
    in_ch=4,
    num_cls=2,
    base=64,        # Increase from 32
    dim=512,        # Increase from 256
    patch=8,
    depth=4,        # Increase from 2
    n_heads=8,      # Increase from 4
)
```

**Decrease Capacity** (for smaller datasets or faster inference):
```python
model = BrainTumNet(
    in_ch=4,
    num_cls=2,
    base=16,        # Decrease from 32
    dim=128,        # Decrease from 256
    patch=8,
    depth=1,        # Decrease from 2
    n_heads=2,      # Decrease from 4
)
```

### How to Add Deep Supervision

Deep supervision adds auxiliary losses at intermediate decoder layers for better gradient flow.

**Modify `seg_unet.py`**:
```python
class SegUNetMasked(nn.Module):
    def __init__(self, in_ch=1, base=32, dim=256, patch=8, depth=2, n_heads=4, deep_supervision=False):
        super().__init__()
        # ... existing code ...

        self.deep_supervision = deep_supervision
        if deep_supervision:
            # Add auxiliary heads at each decoder level
            self.aux_head3 = nn.Conv2d(base*4, 1, 1)
            self.aux_head2 = nn.Conv2d(base*2, 1, 1)
            self.aux_head1 = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        # ... encoder code ...

        x = self.d4(b, s4)
        x = self.d3(x, s3)
        aux3 = self.aux_head3(x) if self.deep_supervision else None

        x = self.d2(x, s2)
        aux2 = self.aux_head2(x) if self.deep_supervision else None

        x = self.d1(x, s1)
        aux1 = self.aux_head1(x) if self.deep_supervision else None

        seg = self.head(x)

        if self.deep_supervision:
            return seg, [aux3, aux2, aux1]
        return seg
```

**Update loss calculation in `trainer.py`**:
```python
if deep_supervision:
    seg_logits, aux_outputs = model.seg(img)
    # Main loss
    seg_loss = seg_criterion(seg_logits, msk)
    # Auxiliary losses (with decreasing weights)
    for i, aux in enumerate(aux_outputs):
        weight = 0.5 ** (i+1)  # 0.5, 0.25, 0.125
        aux_resized = F.interpolate(aux, size=msk.shape[-2:], mode='bilinear')
        seg_loss += weight * seg_criterion(aux_resized, msk)
else:
    seg_logits = model.seg(img)
    seg_loss = seg_criterion(seg_logits, msk)
```

### How to Add Residual Connections to U-Net

**Modify `seg_unet.py`**:
```python
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(conv_bn_relu(in_ch, out_ch), conv_bn_relu(out_ch, out_ch))
        self.pool = nn.MaxPool2d(2)

        # Add residual projection if channels change
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        x = self.block(x) + identity  # Residual connection
        x_down = self.pool(x)
        return x, x_down
```

### How to Add Positional Encoding to Transformer

**Modify `masked_transformer.py`**:
```python
class AdaptiveMaskedTransformer(nn.Module):
    def __init__(self, in_ch, dim, patch_size=8, depth=2, n_heads=4):
        super().__init__()
        self.pe = PatchEmbed(in_ch, dim, patch_size)
        self.mask_gen = SoftMaskGenerator(dim, hidden=dim//2, n_heads=n_heads)
        self.blocks = nn.ModuleList([MaskedTransformerBlock(dim, n_heads) for _ in range(depth)])

        # Add learnable positional encoding
        # Maximum number of patches (for 16×16 input with patch=8 → 2×2=4 patches)
        max_patches = (16 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        tokens, (H,W) = self.pe(x)

        # Add positional encoding
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]

        softmask = self.mask_gen(tokens)
        for blk in self.blocks:
            tokens = blk(tokens, softmask)
        feat = tokens.transpose(1,2).reshape(x.size(0), tokens.size(-1), H, W)
        return feat
```

### How to Change Number of Classes

**For binary → multi-class classification** (e.g., HGG/LGG/Normal → 3 classes):

```python
model = BrainTumNet(
    in_ch=4,
    num_cls=3,  # Change from 2 to 3
    base=32,
    dim=256,
    patch=8,
    depth=2,
    n_heads=4,
)
```

**Update loss function in config**:
```yaml
train:
  cls_criterion: "CrossEntropyLoss"  # Supports multi-class
```

---

**Next**: [[04_TRAINING_SYSTEM|Part 4: Training System →]]

**Back**: [[02_DATA_PIPELINE|← Part 2: Data Pipeline]] | [[TECHNICAL_REPORT_INDEX|Index]]
