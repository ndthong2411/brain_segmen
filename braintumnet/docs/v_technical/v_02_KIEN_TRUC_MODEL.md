# Phần 2: Kiến Trúc Model Chi Tiết

> **🏗️ Giải Thích Sâu Về Kiến Trúc BrainTumNet V1 và V2**
>
> Tài liệu này phân tích chi tiết từng component của model, từ U-Net, CBAM Attention, Transformer đến Inception Classifier.

---

## Mục Lục

1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [BrainTumNet V1 - Baseline](#2-braintumnet-v1---baseline)
3. [BrainTumNet V2 - Phase 2 Enhancements](#3-braintumnet-v2---phase-2-enhancements)
4. [U-Net Segmentation V1](#4-u-net-segmentation-v1)
5. [U-Net V2 - Enhanced Version](#5-u-net-v2---enhanced-version)
6. [CBAM Attention Mechanism](#6-cbam-attention-mechanism)
7. [Adaptive Masked Transformer](#7-adaptive-masked-transformer)
8. [Inception Classification Network](#8-inception-classification-network)
9. [Luồng Dữ Liệu Hoàn Chỉnh](#9-luồng-dữ-liệu-hoàn-chỉnh)
10. [So Sánh V1 vs V2](#10-so-sánh-v1-vs-v2)

---

## 1. Tổng Quan Kiến Trúc

### BrainTumNet Là Multi-Task Model

**Hai nhiệm vụ chính**:
1. **Segmentation**: Phân đoạn vùng khối u từ ảnh MRI
2. **Classification**: Phân loại khối u (HGG vs LGG)

### Sơ Đồ Tổng Thể

```
Input: Multi-modal MRI (B, 4, 256, 256)
         ↓
    ┌─────────────────────────────────────┐
    │    SEGMENTATION NETWORK             │
    │                                     │
    │  U-Net Encoder (4 blocks)          │
    │      ↓                              │
    │  Masked Transformer Bottleneck      │
    │      ↓                              │
    │  U-Net Decoder (4 blocks)          │
    │      ↓                              │
    │  CBAM Attention (skip connections)  │
    └─────────────────────────────────────┘
         ↓
    Segmentation Output (B, C, 256, 256)
    C = 1 (V1) hoặc 3 (V2)
         ↓
    ┌─────────────────────────────────────┐
    │      ROI GATING                     │
    │  • V1: Binary mask                  │
    │  • V2: Whole Tumor (TC+ED)          │
    └─────────────────────────────────────┘
         ↓
    ROI Masked Input (B, 1, 256, 256)
         ↓
    ┌─────────────────────────────────────┐
    │  CLASSIFICATION NETWORK             │
    │                                     │
    │  Inception Blocks                   │
    │      ↓                              │
    │  Global Average Pooling             │
    │      ↓                              │
    │  Fully Connected                    │
    └─────────────────────────────────────┘
         ↓
    Classification Output (B, 2)
    [HGG probability, LGG probability]
```

### Các Thành Phần Chính

**1. Segmentation Network**:
- **Encoder**: Trích xuất features từ thô đến trừu tượng
- **Transformer**: Bắt global context
- **Decoder**: Tái tạo spatial resolution
- **CBAM**: Attention mechanism lọc features quan trọng

**2. ROI Gating**:
- Tạo Region of Interest (ROI) từ segmentation
- Chỉ tập trung vào vùng khối u
- Ngăn gradient từ classification ảnh hưởng segmentation

**3. Classification Network**:
- **Inception**: Multi-scale feature extraction
- **Global Pooling**: Aggregate spatial information
- **FC Layer**: Binary classification (HGG/LGG)

---

## 2. BrainTumNet V1 - Baseline

### File Code

**File**: `src/braintumnet/models/braintumnet.py` (57 dòng)

### Class Definition

```python
class BrainTumNet(nn.Module):
    """
    Multi-task brain tumor segmentation and classification model
    
    Tasks:
    1. Segmentation: Binary (tumor vs background) hoặc 
       Multi-class (background, TC, ED)
    2. Classification: Binary (HGG vs LGG)
    """
    def __init__(self, in_ch=4, num_cls=2, base=32, dim=256, 
                 patch=8, depth=2, n_heads=4,
                 roi_stop_grad=True, deep_supervision=False, 
                 num_classes_seg=1):
        super().__init__()
        self.num_classes_seg = num_classes_seg
        self.roi_stop_grad = roi_stop_grad
        self.deep_supervision = deep_supervision
        
        # Segmentation U-Net
        self.seg = SegUNetMasked(
            in_ch=in_ch, base=base, dim=dim, 
            patch=patch, depth=depth, n_heads=n_heads
        )
        
        # Channel reduction: 4 modalities → 1 channel
        self.reduce = nn.Conv2d(in_ch, 1, 1)
        
        # Classification network
        self.cls = TInceptionNet(in_ch=1, num_classes=num_cls)
```

**Giải thích từng tham số**:

- `in_ch=4`: Số kênh đầu vào (4 MRI modalities)
- `num_cls=2`: Số lớp classification (HGG/LGG)
- `base=32`: Base feature channels (nhân lên ở mỗi encoder block)
- `dim=256`: Transformer embedding dimension
- `patch=8`: Patch size cho transformer
- `depth=2`: Số transformer blocks (V1), 4 (V2)
- `n_heads=4`: Số attention heads (V1), 8 (V2)
- `roi_stop_grad=True`: Stop gradient qua ROI path
- `deep_supervision=False`: Sử dụng deep supervision (V2 feature)
- `num_classes_seg=1`: Số lớp segmentation (1=binary, 3=multi-class)

### Forward Pass V1 (Binary)

```python
def forward(self, x):
    # x: (B, 4, 256, 256) - Multi-modal input
    
    # 1. Segmentation
    seg_logits = self.seg(x)  # (B, 1, 256, 256)
    
    # 2. Sigmoid activation
    seg_prob = torch.sigmoid(seg_logits)  # (B, 1, 256, 256)
    # Values in [0, 1] - probability of tumor
    
    # 3. ROI gating
    roi_input = self.reduce(x)  # (B, 4, 256, 256) → (B, 1, 256, 256)
    
    # Stop gradient: classification không ảnh hưởng segmentation
    roi = roi_input * seg_prob.detach()  # (B, 1, 256, 256)
    
    # 4. Classification
    cls_logits = self.cls(roi)  # (B, 2)
    
    return seg_logits, cls_logits
```

**Luồng tensor shapes**:
```
Input:        (4, 4, 256, 256)  # Batch=4, 4 modalities
    ↓ seg
Seg logits:   (4, 1, 256, 256)  # Binary segmentation
    ↓ sigmoid
Seg prob:     (4, 1, 256, 256)  # Probabilities [0,1]
    ↓ reduce + gating
ROI:          (4, 1, 256, 256)  # Masked input
    ↓ cls
Cls logits:   (4, 2)            # HGG/LGG scores
```

### Multi-Class Support (V1 Updated)

```python
def forward(self, x):
    # Handle deep supervision
    seg_output = self.seg(x)
    if self.deep_supervision:
        seg_logits, aux_outputs = seg_output
    else:
        seg_logits = seg_output
        aux_outputs = None
    
    # Compute ROI based on segmentation mode
    if self.num_classes_seg == 1:
        # Binary: use sigmoid
        seg_prob = torch.sigmoid(seg_logits)  # (B, 1, H, W)
    else:
        # Multi-class: use softmax
        seg_prob = torch.softmax(seg_logits, dim=1)  # (B, 3, H, W)
        
        # Whole Tumor = TC (class 1) + ED (class 2)
        seg_prob = seg_prob[:, 1:, :, :].sum(dim=1, keepdim=True)  # (B, 1, H, W)
    
    # ROI gating (same as before)
    roi_input = self.reduce(x)
    if self.roi_stop_grad:
        roi = roi_input * seg_prob.detach()
    else:
        roi = roi_input * seg_prob
    
    # Classification
    cls_logits = self.cls(roi)
    
    # Return based on deep supervision
    if self.deep_supervision:
        return seg_logits, cls_logits, aux_outputs
    return seg_logits, cls_logits
```

**Tại sao sum classes 1 và 2 cho ROI?**

```
Multi-class Output (3 channels):
Channel 0: Background probability
Channel 1: Tumor Core (TC) probability
Channel 2: Edema (ED) probability

Whole Tumor (WT) = TC + ED
→ Sum channels 1 and 2

Ví dụ:
seg_prob = [
    [[0.8, 0.7, 0.9], ...],  # Background
    [[0.1, 0.2, 0.05], ...], # TC
    [[0.1, 0.1, 0.05], ...]  # ED
]

WT prob = [
    [[0.2, 0.3, 0.1], ...]   # TC + ED
]
```

### Tại Sao `detach()`?

**Không có detach**:
```
Classification Loss → ∇ cls_logits → ∇ roi → ∇ seg_prob → ∇ seg_logits
                                                              ↑
                                                    Ảnh hưởng segmentation!
```

**Có detach**:
```
Classification Loss → ∇ cls_logits → ∇ roi → ✗ STOPPED
Segmentation Loss → ∇ seg_logits (only from seg loss)
```

**Lợi ích**:
- Segmentation chỉ optimize cho Dice/Focal loss
- Classification chỉ optimize cho CE loss
- Hai tasks độc lập về gradient flow
- Ngăn classification làm sai lệch segmentation

---

## 3. BrainTumNet V2 - Phase 2 Enhancements

### File Code

**File**: `src/braintumnet/models/braintumnet_v2.py` (170 dòng)

### Key Differences from V1

```python
class BrainTumNetV2(nn.Module):
    """
    Enhanced multi-task model with Phase 2 improvements
    
    Improvements over V1:
    1. Uses SegUNetV2 (InstanceNorm, LeakyReLU, residuals)
    2. Larger capacity options (base=48/64)
    3. Deep supervision support
    4. Multi-scale fusion
    """
    def __init__(self, in_ch=4, num_cls=2, base=48, dim=384, 
                 patch=8, depth=4, n_heads=8, num_classes_seg=3, 
                 dropout=0.15, roi_stop_grad=True, 
                 deep_supervision=True, multi_scale_fusion=True):
        super().__init__()
        
        # Enhanced segmentation network
        self.seg = SegUNetV2(
            in_ch=in_ch,
            base=base,  # 48 hoặc 64 (thay vì 32)
            dim=dim,    # 384 hoặc 512 (thay vì 256)
            patch=patch,
            depth=depth,  # 4 (thay vì 2)
            n_heads=n_heads,  # 8 (thay vì 4)
            num_classes=num_classes_seg,  # 3 (multi-class)
            dropout=dropout,  # 0.15 regularization
            norm='instance',  # InstanceNorm!
            deep_supervision=deep_supervision,  # True
            multi_scale_fusion=multi_scale_fusion  # True
        )
        
        # Same as V1
        self.reduce = nn.Conv2d(in_ch, 1, 1, bias=False) if in_ch > 1 else nn.Identity()
        self.cls_backbone = TInceptionNet(in_ch=1, num_classes=num_cls)
```

### V2 Forward Pass

```python
def forward(self, x):
    # Segmentation với deep supervision
    seg_output = self.seg(x)
    
    if self.deep_supervision:
        seg_logits, aux_outputs = seg_output
        # seg_logits: (B, 3, 256, 256) - main output
        # aux_outputs: [(B,3,64,64), (B,3,128,128), (B,3,256,256)]
    else:
        seg_logits = seg_output
        aux_outputs = None
    
    # ROI computation (multi-class)
    seg_prob = torch.softmax(seg_logits, dim=1)  # (B, 3, H, W)
    seg_prob_wt = seg_prob[:, 1:, :, :].sum(dim=1, keepdim=True)  # (B, 1, H, W)
    
    # ROI gating
    roi_input = self.reduce(x)
    if self.roi_stop_grad:
        roi = roi_input * seg_prob_wt.detach()
    else:
        roi = roi_input * seg_prob_wt
    
    # Classification
    cls_logits = self.cls_backbone(roi)
    
    # Return
    if self.deep_supervision:
        return seg_logits, cls_logits, aux_outputs
    return seg_logits, cls_logits
```

### Parameter Counts

```python
# V1 Baseline
model_v1 = BrainTumNet(
    in_ch=4, base=32, dim=256, depth=2, n_heads=4
)
# Parameters: ~14M

# V2 Small (RTX 3090 compatible)
model_v2_small = BrainTumNetV2(
    in_ch=4, base=48, dim=384, depth=4, n_heads=8,
    num_classes_seg=3, dropout=0.15
)
# Parameters: ~45M

# V2 Large (A100 optimized)
model_v2_large = BrainTumNetV2(
    in_ch=4, base=64, dim=512, depth=4, n_heads=8,
    num_classes_seg=3, dropout=0.15
)
# Parameters: ~87M
```

---

## 4. U-Net Segmentation V1

### File Code

**File**: `src/braintumnet/models/seg_unet.py` (67 dòng)

### Helper Function

```python
def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    """
    Standard convolution block: Conv → BatchNorm → ReLU
    
    Tại sao thứ tự này?
    1. Conv: Học spatial features
    2. BatchNorm: Normalize activations (ổn định training)
    3. ReLU: Non-linearity
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )
```

**Tại sao `bias=False`?**
- BatchNorm có learnable shift parameter (β)
- Bias trong conv sẽ bị loại bỏ bởi BatchNorm
- → Tiết kiệm parameters

### EncoderBlock V1

```python
class EncoderBlock(nn.Module):
    """
    Encoder block: 2 Conv blocks + MaxPool
    
    Input (in_ch)
        ↓
    Conv-BN-ReLU (in_ch → out_ch)
        ↓
    Conv-BN-ReLU (out_ch → out_ch)
        ↓ (skip connection)
    MaxPool2d (downsample 1/2)
        ↓
    Output (downsampled)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            conv_bn_relu(in_ch, out_ch),
            conv_bn_relu(out_ch, out_ch)
        )
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.block(x)  # Features
        return x, self.pool(x)  # (skip, downsampled)
```

**Ví dụ shapes**:
```python
encoder = EncoderBlock(64, 128)
x = torch.randn(4, 64, 128, 128)
skip, down = encoder(x)

print(skip.shape)  # (4, 128, 128, 128) - giữ nguyên spatial
print(down.shape)  # (4, 128, 64, 64)   - giảm 1/2
```

### DecoderBlock V1

```python
class DecoderBlock(nn.Module):
    """
    Decoder block: Upsample + Concat skip + CBAM + Conv
    
    Input (in_ch, H, W)
        ↓
    ConvTranspose2d (upsample 2×)
        ↓ (in_ch, 2H, 2W)
    CBAM Attention trên skip
        ↓
    Concatenate [upsampled, skip]
        ↓ (in_ch*2, 2H, 2W)
    2× Conv-BN-ReLU
        ↓
    Output (out_ch, 2H, 2W)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, 2)
        self.cbam = CBAM(in_ch)
        self.block = nn.Sequential(
            conv_bn_relu(in_ch*2, out_ch),
            conv_bn_relu(out_ch, out_ch)
        )
    
    def forward(self, x, skip):
        x = self.up(x)           # Upsample
        skip = self.cbam(skip)   # Attention on skip
        x = torch.cat([x, skip], dim=1)  # Concat
        return self.block(x)     # Refine
```

**Tại sao CBAM trên skip?**
- Skip connections mang thông tin từ encoder
- Không phải tất cả features đều hữu ích
- CBAM lọc ra features quan trọng
- → Decoder nhận thông tin tốt hơn

### SegUNetMasked V1

```python
class SegUNetMasked(nn.Module):
    """
    U-Net với Masked Transformer bottleneck
    
    Architecture:
    Input (1/4, 256, 256)
        ↓
    e1: (32, 256, 256) → (32, 128, 128)
    e2: (64, 128, 128) → (64, 64, 64)
    e3: (128, 64, 64) → (128, 32, 32)
    e4: (256, 32, 32) → (256, 16, 16)
        ↓
    Bottleneck: Transformer
        ↓
    d4: (256, 32, 32)
    d3: (128, 64, 64)
    d2: (64, 128, 128)
    d1: (32, 256, 256)
        ↓
    Head: (1, 256, 256)
    """
    def __init__(self, in_ch=1, base=32, dim=256, 
                 patch=8, depth=2, n_heads=4):
        super().__init__()
        self.patch = patch
        
        # Encoder
        self.e1 = EncoderBlock(in_ch, base)
        self.e2 = EncoderBlock(base, base*2)
        self.e3 = EncoderBlock(base*2, base*4)
        self.e4 = EncoderBlock(base*4, base*8)
        
        # Transformer bottleneck
        self.bottleneck_conv = conv_bn_relu(base*8, dim, k=1, s=1, p=0)
        self.amt = AdaptiveMaskedTransformer(
            in_ch=dim, dim=dim, patch_size=patch, 
            depth=depth, n_heads=n_heads
        )
        self.tr_upsample = nn.ConvTranspose2d(
            dim, base*8, kernel_size=patch, stride=patch
        )
        
        # Decoder
        self.d4 = DecoderBlock(base*8, base*8)
        self.d3 = DecoderBlock(base*8, base*4)
        self.d2 = DecoderBlock(base*4, base*2)
        self.d1 = DecoderBlock(base*2, base)
        
        # Segmentation head
        self.head = nn.Conv2d(base, 1, 1)
    
    def forward(self, x):
        # Encoder
        s1, x1 = self.e1(x)   # (32, 256, 256), (32, 128, 128)
        s2, x2 = self.e2(x1)  # (64, 128, 128), (64, 64, 64)
        s3, x3 = self.e3(x2)  # (128, 64, 64), (128, 32, 32)
        s4, x4 = self.e4(x3)  # (256, 32, 32), (256, 16, 16)
        
        # Transformer bottleneck
        b = self.bottleneck_conv(x4)  # (256, 16, 16)
        b = self.amt(b)                # (256, 2, 2) after patching
        b = self.tr_upsample(b)        # (256, 16, 16) restored
        
        # Decoder
        x = self.d4(b, s4)    # (256, 32, 32)
        x = self.d3(x, s3)    # (128, 64, 64)
        x = self.d2(x, s2)    # (64, 128, 128)
        x = self.d1(x, s1)    # (32, 256, 256)
        
        # Segmentation output
        seg = self.head(x)    # (1, 256, 256)
        return seg
```

---

## 5. U-Net V2 - Enhanced Version

### File Code

**File**: `src/braintumnet/models/seg_unet_v2.py` (322 dòng)

### Enhanced Conv Block

```python
def conv_norm_act(in_ch, out_ch, k=3, s=1, p=1, 
                  norm='instance', dropout=0.0):
    """
    Improved conv block: Conv → Norm → LeakyReLU → Dropout
    
    Changes from V1:
    1. BatchNorm → InstanceNorm (medical imaging standard)
    2. ReLU → LeakyReLU (better gradient flow)
    3. Added Dropout option (regularization)
    """
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)]
    
    # Normalization
    if norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    elif norm == 'batch':
        layers.append(nn.BatchNorm2d(out_ch))
    elif norm == 'group':
        num_groups = min(32, out_ch // 4)
        layers.append(nn.GroupNorm(num_groups, out_ch))
    
    # Activation
    layers.append(nn.LeakyReLU(0.01, inplace=True))
    
    # Dropout
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    
    return nn.Sequential(*layers)
```

**Tại sao InstanceNorm?**
```
BatchNorm: E[(x - μ_batch) / σ_batch]
- Phụ thuộc vào batch statistics
- Không ổn định với batch nhỏ (8-16)
- Medical imaging thường có batch nhỏ

InstanceNorm: E[(x - μ_instance) / σ_instance]
- Normalize mỗi sample riêng lẻ
- Không phụ thuộc batch size
- Ổn định hơn cho medical imaging
```

### Residual Convolutional Block

```python
class ResidualConvBlock(nn.Module):
    """
    Residual block: Input → Conv → Conv → Add → Output
    
    Structure:
    x ────────────────┐
    │                 │
    Conv-Norm-Act     │
    │                 │
    Conv-Norm         │
    │                 │
    └─────→ ADD ←─────┘
            │
            Act
            │
            Output
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
        
        # Residual connection: 1×1 conv nếu channels khác nhau
        self.residual = (nn.Conv2d(in_ch, out_ch, 1, bias=False) 
                         if in_ch != out_ch else nn.Identity())
        
        self.act = nn.LeakyReLU(0.01, inplace=True)
    
    def forward(self, x):
        identity = self.residual(x)  # Shortcut
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity         # Residual add
        out = self.act(out)
        return out
```

**Tại sao Residual?**
```
Không có residual:
x → Conv → Conv → y
Gradient: ∇y → ∇Conv2 → ∇Conv1 → ∇x
Problem: Vanishing gradient trong mạng sâu

Có residual:
x → Conv → Conv → y
  └─────────────→ (shortcut)
Gradient: ∇y → ∇(Conv + identity)
         = ∇Conv + ∇identity
         = ∇Conv + 1
→ Luôn có gradient flow qua shortcut!
```

### Enhanced EncoderBlock V2

```python
class EncoderBlock(nn.Module):
    """
    V2 Encoder: Residual block + Strided conv downsampling
    
    Changes from V1:
    1. 2 Conv blocks → Residual block
    2. MaxPool → Strided convolution
    """
    def __init__(self, in_ch, out_ch, norm='instance', dropout=0.0):
        super().__init__()
        self.block = ResidualConvBlock(in_ch, out_ch, norm, dropout)
        
        # Learnable downsampling (thay vì MaxPool)
        self.downsample = nn.Conv2d(
            out_ch, out_ch, 
            kernel_size=3, stride=2, padding=1, 
            bias=False
        )
    
    def forward(self, x):
        x = self.block(x)        # Features
        x_down = self.downsample(x)  # Learned downsampling
        return x, x_down
```

**Tại sao Strided Conv thay MaxPool?**
```
MaxPool:
- Fixed operation (không học)
- Chọn max value trong window
- Mất information

Strided Convolution:
- Learnable weights
- Học cách downsample tối ưu
- Preserve more information
- Flexible cho different patterns
```

### Enhanced DecoderBlock V2

```python
class DecoderBlock(nn.Module):
    """
    V2 Decoder: Same structure nhưng dùng residual blocks
    """
    def __init__(self, in_ch, out_ch, norm='instance', dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.cbam = CBAM(out_ch)
        self.block = ResidualConvBlock(out_ch*2, out_ch, norm, dropout)
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, self.cbam(skip)], dim=1)
        x = self.block(x)
        return x
```

### Multi-Scale Fusion Module

```python
class MultiScaleFusion(nn.Module):
    """
    Fuse features từ multiple decoder levels
    
    Input: [d1, d2, d3, d4]
    - d1: (base, 256, 256)    # Finest
    - d2: (base*2, 128, 128)
    - d3: (base*4, 64, 64)
    - d4: (base*8, 32, 32)    # Coarsest
    
    Output: (out_ch, 256, 256) # Fused features
    """
    def __init__(self, channels_list, out_channels):
        super().__init__()
        # 1×1 convs để project về same channels
        self.convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1, bias=False) 
            for ch in channels_list
        ])
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
    
    def forward(self, features):
        """
        features: List of [d1, d2, d3, d4]
        """
        target_size = features[0].shape[2:]  # Largest size (d1)
        
        upsampled = []
        for i, feat in enumerate(features):
            # Project channels
            feat = self.convs[i](feat)
            
            # Upsample to target size
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, 
                    mode='bilinear', align_corners=False
                )
            upsampled.append(feat)
        
        # Fuse by summation
        fused = sum(upsampled)
        fused = self.norm(fused)
        fused = self.act(fused)
        return fused
```

**Tại sao Multi-Scale Fusion?**
```
d1 (256×256): Fine details (edges, small structures)
d2 (128×128): Medium features
d3 (64×64):   Larger patterns
d4 (32×32):   Global context

Fusing all → Bắt được cả details VÀ context!

Example:
- Small tumor: d1 detects boundaries
- Large tumor: d4 provides overall shape
- Combined: Accurate segmentation cho cả small và large
```

### SegUNetV2 Complete

```python
class SegUNetV2(nn.Module):
    """
    Enhanced U-Net với tất cả Phase 2 improvements
    """
    def __init__(self, in_ch=4, base=48, dim=384, patch=8, 
                 depth=4, n_heads=8, num_classes=3, dropout=0.15, 
                 norm='instance', deep_supervision=True, 
                 multi_scale_fusion=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.multi_scale_fusion = multi_scale_fusion
        self.num_classes = num_classes
        
        # Encoder (với residual + strided conv)
        self.e1 = EncoderBlock(in_ch, base, norm, 0)
        self.e2 = EncoderBlock(base, base*2, norm, 0)
        self.e3 = EncoderBlock(base*2, base*4, norm, dropout)
        self.e4 = EncoderBlock(base*4, base*8, norm, dropout)
        
        # Transformer bottleneck (same as V1)
        self.bottleneck_conv = conv_norm_act(base*8, dim, k=1, s=1, p=0, norm=norm)
        self.amt = AdaptiveMaskedTransformer(dim, dim, patch, depth, n_heads)
        self.tr_upsample = nn.ConvTranspose2d(dim, base*8, patch, patch, bias=False)
        
        # Decoder (với residual blocks)
        self.d4 = DecoderBlock(base*8, base*8, norm, dropout)
        self.d3 = DecoderBlock(base*8, base*4, norm, dropout)
        self.d2 = DecoderBlock(base*4, base*2, norm, dropout/2)
        self.d1 = DecoderBlock(base*2, base, norm, 0)
        
        # Multi-scale fusion
        if self.multi_scale_fusion:
            self.ms_fusion = MultiScaleFusion(
                [base, base*2, base*4, base*8], base
            )
            self.fusion_conv = ResidualConvBlock(base*2, base, norm, 0)
        
        # Main segmentation head
        self.head = nn.Conv2d(base, num_classes, 1)
        
        # Deep supervision auxiliary heads
        if self.deep_supervision:
            self.aux_head3 = nn.Conv2d(base*4, num_classes, 1)
            self.aux_head2 = nn.Conv2d(base*2, num_classes, 1)
            self.aux_head1 = nn.Conv2d(base, num_classes, 1)
    
    def forward(self, x):
        # Encoder
        s1, x1 = self.e1(x)
        s2, x2 = self.e2(x1)
        s3, x3 = self.e3(x2)
        s4, x4 = self.e4(x3)
        
        # Transformer
        b = self.bottleneck_conv(x4)
        b = self.amt(b)
        b = self.tr_upsample(b)
        
        # Decoder với deep supervision
        d4 = self.d4(b, s4)
        aux3 = self.aux_head3(d3) if self.deep_supervision else None
        
        d3 = self.d3(d4, s3)
        aux2 = self.aux_head2(d2) if self.deep_supervision else None
        
        d2 = self.d2(d3, s2)
        aux1 = self.aux_head1(d1) if self.deep_supervision else None
        
        d1 = self.d1(d2, s1)
        
        # Multi-scale fusion
        if self.multi_scale_fusion:
            fused = self.ms_fusion([d1, d2, d3, d4])
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

---

## 6. CBAM Attention Mechanism

### File Code

**File**: `src/braintumnet/models/cbam.py` (33 dòng)

### Tổng Quan

**CBAM** = Convolutional Block Attention Module

**Hai loại attention tuần tự**:
1. **Channel Attention**: Features nào quan trọng?
2. **Spatial Attention**: Vị trí nào quan trọng?

### Channel Attention

```python
class ChannelAttention(nn.Module):
    """
    Channel Attention: Học channels nào chứa thông tin quan trọng
    
    Workflow:
    Input (C, H, W)
        ↓
    ┌─────────────┬──────────────┐
    │             │              │
    Avg Pool      Max Pool
    (C,1,1)       (C,1,1)
    │             │
    └──────┬──────┘
           ↓
        MLP (bottleneck)
        C → C/16 → C
           ↓
        Add + Sigmoid
           ↓
    Channel Weights (C,1,1)
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        
        # MLP với bottleneck
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False),
        )
    
    def forward(self, x):
        # Dual pooling
        avg_feat = self.mlp(self.avg(x))  # (B, C, 1, 1)
        max_feat = self.mlp(self.max(x))  # (B, C, 1, 1)
        
        # Combine và sigmoid
        att = torch.sigmoid(avg_feat + max_feat)  # [0, 1]
        
        # Apply attention
        return x * att  # Broadcasting
```

**Ví dụ cụ thể**:
```python
x = torch.randn(4, 128, 32, 32)  # (B, C, H, W)
ca = ChannelAttention(128, reduction=16)
out = ca(x)

# Attention weights example:
# Channel 0: weight = 0.9 (important - edge detector)
# Channel 1: weight = 0.1 (unimportant - noise)
# Channel 2: weight = 0.8 (important - texture)
# ...
# → Model tự học channels nào hữu ích!
```

### Spatial Attention

```python
class SpatialAttention(nn.Module):
    """
    Spatial Attention: Học vị trí không gian nào quan trọng
    
    Workflow:
    Input (C, H, W)
        ↓
    ┌─────────────┬──────────────┐
    │             │              │
    Channel Avg   Channel Max
    (1, H, W)     (1, H, W)
    │             │
    └──────┬──────┘
           ↓
        Concat
        (2, H, W)
           ↓
        7×7 Conv
           ↓
        Sigmoid
           ↓
    Spatial Weights (1, H, W)
    """
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k//2, bias=False)
    
    def forward(self, x):
        # Dual channel pooling
        avg_spatial = x.mean(1, True)  # (B, 1, H, W)
        max_spatial = x.amax(1, True)  # (B, 1, H, W)
        
        # Concat và conv
        att = torch.cat([avg_spatial, max_spatial], dim=1)  # (B, 2, H, W)
        att = torch.sigmoid(self.conv(att))  # (B, 1, H, W)
        
        # Apply attention
        return x * att  # Broadcasting
```

**Ví dụ cụ thể**:
```python
# Input: Brain MRI features
x = torch.randn(4, 64, 128, 128)

# Spatial attention output (simplified):
# High weights (0.8-1.0) tại vùng tumor
# Low weights (0.1-0.3) tại background
# → Model tập trung vào tumor region!
```

### CBAM Complete

```python
class CBAM(nn.Module):
    """
    CBAM: Channel Attention → Spatial Attention
    """
    def __init__(self, in_channels, reduction=16, k=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(k)
    
    def forward(self, x):
        x = self.ca(x)  # Channel attention first
        x = self.sa(x)  # Spatial attention second
        return x
```

**Tại sao tuần tự (không song song)?**
```
Sequential (channel → spatial):
1. Channel attention refines "WHAT" features quan trọng
2. Spatial attention refines "WHERE" chúng quan trọng
3. Hai stages complement each other

Parallel (channel || spatial):
- Hai attentions độc lập
- Không có interaction
- Experimental results: Sequential tốt hơn!
```

---

*[Phần còn lại sẽ được tiếp tục trong response tiếp theo do giới hạn độ dài]*

---

**[← Phần 1: Tổng Quan](v_01_TONG_QUAN_DU_AN.md)** | **[Phần 3: Xử Lý Dữ Liệu →](v_03_XU_LY_DU_LIEU.md)**
