# Phần 3: Kiến Trúc Model

**Tổng quan**: Tài liệu này giải thích chi tiết kiến trúc của BrainTumNet, bao gồm tất cả các thành phần model với giải thích từng dòng code và hình ảnh minh họa.

## 🆕 Model Versions

BrainTumNet hiện có **2 phiên bản** với các cải tiến khác nhau:

### Version 1 (Baseline)
**Files code V1**:
- `src/braintumnet/models/braintumnet.py` (57 dòng - updated)
- `src/braintumnet/models/seg_unet.py` (67 dòng)
- `src/braintumnet/models/cbam.py` (33 dòng)
- `src/braintumnet/models/masked_transformer.py` (88 dòng)
- `src/braintumnet/models/t_inception.py` (51 dòng)

**Features**:
- Binary segmentation (tumor vs background)
- BatchNorm, ReLU activations
- MaxPool downsampling
- ~14M parameters

### Version 2 (Phase 2 Upgrades) ⭐ NEW

**Additional files V2**:
- `src/braintumnet/models/seg_unet_v2.py` (322 dòng) ⭐ NEW

**Key improvements**:
- ✅ **Multi-class segmentation** (3 classes: Background, TC, ED)
- ✅ **InstanceNorm** instead of BatchNorm (medical imaging standard)
- ✅ **LeakyReLU** instead of ReLU (better gradients)
- ✅ **Residual connections** in all blocks
- ✅ **Strided convolution** instead of MaxPool (learned downsampling)
- ✅ **Multi-scale fusion** before final head
- ✅ **Deep supervision** with auxiliary outputs
- ✅ **Dropout** for regularization (0.15 for large models)
- ✅ **Larger capacity** options (base=48/64, dim=384/512)

**Parameter counts**:
- V2 Baseline (V1-like): ~14M parameters
- V2 Small (Phase 2): ~35M parameters (base=48, dim=384)
- V2 Large (Phase 2): ~60M parameters (base=64, dim=512)

**Tổng cộng**: 618 dòng code được giải thích chi tiết trong tài liệu này.

---

## Mục lục

### Core Architecture (V1 & V2)
1. [Tổng quan kiến trúc](#tổng-quan-kiến-trúc)
2. [BrainTumNet (Wrapper đa nhiệm vụ)](#braintumnet-wrapper-đa-nhiệm-vụ)
   - 2a. [Multi-Class Segmentation Support](#multi-class-segmentation-support) ⭐ NEW
   - 2b. [ROI Gating for Multi-Class](#roi-gating-for-multi-class) ⭐ NEW
3. [U-Net Segmentation V1 với Transformer](#u-net-segmentation-với-transformer)
4. [CBAM Attention Mechanism](#cbam-attention-mechanism)
5. [Adaptive Masked Transformer](#adaptive-masked-transformer)
6. [Inception Classification Network](#inception-classification-network)

### Version 2 Enhancements ⭐ NEW
7. [SegUNetV2 - Phase 2 Improvements](#segunetv2-phase-2-improvements)
   - 7a. [Residual Convolutional Blocks](#residual-convolutional-blocks)
   - 7b. [Enhanced Encoder & Decoder](#enhanced-encoder--decoder)
   - 7c. [Multi-Scale Fusion](#multi-scale-fusion)
   - 7d. [Deep Supervision](#deep-supervision)
   - 7e. [V2 Full Architecture](#v2-full-architecture)

### Analysis & Guides
8. [Luồng dữ liệu hoàn chỉnh với Tensor Shapes](#luồng-dữ-liệu-hoàn-chỉnh-với-tensor-shapes)
9. [Giải thích các quyết định thiết kế](#giải-thích-các-quyết-định-thiết-kế)
10. [Hướng dẫn chỉnh sửa](#hướng-dẫn-chỉnh-sửa)

---

## Tổng quan kiến trúc

BrainTumNet là một **multi-task learning model** thực hiện hai nhiệm vụ:
1. **Segmentation**: Phân đoạn vùng tumor (khối u não) từ ảnh MRI
2. **Classification**: Phân loại tumor thành High-Grade Glioma (HGG) hoặc Low-Grade Glioma (LGG)

### Sơ đồ kiến trúc tổng thể

```
Input MRI (4 modalities: FLAIR, T1, T1CE, T2)
              ↓
    ┌─────────┴─────────┐
    │                   │
    │  [Segmentation    │
    │   U-Net with      │
    │   CBAM + Masked   │
    │   Transformer]    │
    │                   │
    └─────────┬─────────┘
              ↓
    Segmentation Mask
              ↓
        [ROI Gating]
              ↓
    ┌─────────┴─────────┐
    │                   │
    │  [Inception       │
    │   Classifier]     │
    │                   │
    └─────────┬─────────┘
              ↓
    Classification (HGG/LGG)
```

### Các thành phần chính

1. **SegUNetMasked**: Encoder-decoder với CBAM attention và Transformer bottleneck
2. **CBAM**: Convolutional Block Attention Module (channel + spatial attention)
3. **AdaptiveMaskedTransformer**: Self-attention với learnable soft masks
4. **TInceptionNet**: Multi-scale classifier với parallel convolutions

### Triết lý thiết kế

**Tại sao kết hợp U-Net + Transformer + Inception?**

- **U-Net**: Kiến trúc đã được chứng minh hiệu quả cho phân đoạn y tế
- **Transformer**: Bắt global context (tumor ảnh hưởng đến mô xung quanh)
- **Inception**: Multi-scale features cho tumors có kích thước và hình dạng khác nhau
- **CBAM**: Lọc features quan trọng từ skip connections

---

## BrainTumNet (Wrapper đa nhiệm vụ)

**File**: `src/braintumnet/models/braintumnet.py` (43 dòng)

Đây là model wrapper chính kết nối segmentation và classification tasks.

### Sơ đồ kiến trúc

```
Multi-modal MRI Input (B, 4, H, W)
         ↓
    [Segmentation U-Net]
         ↓
Segmentation Logits (B, 1, H, W)
         ↓
    [Sigmoid] → Mask Probability
         ↓
    ┌────┴────┐
    │         │
    ↓         ↓ (detach gradient)
Output 1   [ROI Gating]
Seg Mask      ↓
         Input × Mask → ROI
              ↓
         [Channel Reduce]
              ↓
         (B, 1, H, W)
              ↓
       [Inception Classifier]
              ↓
         Classification Logits (B, num_classes)
              ↓
           Output 2
```

### Code chi tiết

```python
class BrainTumNet(nn.Module):
    def __init__(self, in_ch=4, num_cls=2, base=32, dim=256, patch=8, depth=2, n_heads=4):
        super().__init__()
        self.seg = SegUNetMasked(in_ch=in_ch, base=base, dim=dim, patch=patch, depth=depth, n_heads=n_heads)
        self.reduce = nn.Conv2d(in_ch, 1, 1)
        self.cls = TInceptionNet(in_ch=1, num_classes=num_cls)
```

**Dòng 3-7**: Khởi tạo model

**Tham số**:
- `in_ch=4`: Số kênh đầu vào (4 MRI modalities: FLAIR, T1, T1CE, T2)
- `num_cls=2`: Số lớp phân loại (HGG vs LGG)
- `base=32`: Base channel multiplier cho U-Net
- `dim=256`: Transformer embedding dimension
- `patch=8`: Patch size cho transformer
- `depth=2`: Số transformer blocks
- `n_heads=4`: Số attention heads

**Ba thành phần chính**:

1. **Segmentation U-Net** (`self.seg`):
```python
self.seg = SegUNetMasked(in_ch=in_ch, base=base, dim=dim, patch=patch, depth=depth, n_heads=n_heads)
```
- Nhận đầu vào: (B, 4, 256, 256) multi-modal MRI
- Trả về: (B, 1, 256, 256) segmentation logits

2. **Channel Reducer** (`self.reduce`):
```python
self.reduce = nn.Conv2d(in_ch, 1, 1)
```
- Giảm 4 kênh xuống 1 kênh trước khi gating
- Sử dụng 1×1 convolution (pointwise)
- Học cách kết hợp thông tin từ 4 modalities

3. **Inception Classifier** (`self.cls`):
```python
self.cls = TInceptionNet(in_ch=1, num_classes=num_cls)
```
- Nhận đầu vào: (B, 1, 256, 256) gated ROI
- Trả về: (B, 2) classification logits

---

```python
    def forward(self, x):
        seg_logits = self.seg(x)
        seg_prob = torch.sigmoid(seg_logits)
        roi = self.reduce(x) * seg_prob.detach()
        cls_logits = self.cls(roi)
        return seg_logits, cls_logits
```

**Dòng 8-13**: Forward pass

**Từng bước chi tiết**:

1. **Segmentation**:
```python
seg_logits = self.seg(x)  # (B, 4, 256, 256) → (B, 1, 256, 256)
```
- U-Net xử lý ảnh đầu vào và tạo segmentation mask

2. **Tạo ROI mask**:
```python
seg_prob = torch.sigmoid(seg_logits)  # Logits → Probabilities [0, 1]
```
- Chuyển đổi raw logits thành probabilities
- Giá trị cao = khả năng cao là tumor

3. **ROI Gating**:
```python
roi = self.reduce(x) * seg_prob.detach()
```

**Phân tích từng phần**:
- `self.reduce(x)`: (B, 4, H, W) → (B, 1, H, W) kết hợp 4 modalities
- `seg_prob.detach()`: **Quan trọng!** Ngăn gradients từ classifier ảnh hưởng đến segmentation
- `*`: Element-wise multiplication (gating)

**Tại sao `.detach()`?**

Không có `.detach()`:
```
Classification Loss → Backprop → seg_prob → seg_logits → Segmentation Network
```
⚠️ **Vấn đề**: Classifier có thể làm sai lệch segmentation để cải thiện accuracy của riêng nó.

Có `.detach()`:
```
Classification Loss → Backprop → roi (stopped) ✗ không đến segmentation network
Segmentation Loss → Backprop → Segmentation Network ✓
```
✅ **Kết quả**: Segmentation chỉ tối ưu cho dice loss, không bị ảnh hưởng bởi classification.

4. **Classification**:
```python
cls_logits = self.cls(roi)  # (B, 1, 256, 256) → (B, 2)
```
- Inception network xử lý gated ROI
- Tạo classification logits cho 2 lớp (HGG/LGG)

5. **Return cả hai outputs**:
```python
return seg_logits, cls_logits
```
- `seg_logits`: Dùng cho Dice+BCE loss
- `cls_logits`: Dùng cho CrossEntropy loss

---

### Ví dụ sử dụng

```python
# Khởi tạo model
model = BrainTumNet(
    in_ch=4,        # 4 MRI modalities
    num_cls=2,      # Binary classification (HGG/LGG)
    base=32,        # U-Net base channels
    dim=256,        # Transformer dimension
    patch=8,        # Transformer patch size
    depth=2,        # 2 transformer blocks
    n_heads=4,      # 4 attention heads
)

# Forward pass
img = torch.randn(4, 4, 256, 256)  # Batch of 4 images
seg_logits, cls_logits = model(img)

# Shapes
print(seg_logits.shape)  # (4, 1, 256, 256) - segmentation masks
print(cls_logits.shape)  # (4, 2) - classification scores
```

---

## Multi-Class Segmentation Support

⭐ **NEW in V2**: BrainTumNet now supports **multi-class segmentation** (3 classes) in addition to binary mode.

### Updated Constructor Parameters

```python
class BrainTumNet(nn.Module):
    def __init__(self, in_ch=1, num_cls=2, base=32, dim=256, patch=8, depth=2, n_heads=4,
                 roi_stop_grad=True, deep_supervision=False, num_classes_seg=1):
        """
        Args:
            num_classes_seg: Number of segmentation classes ⭐ NEW
                            1 = binary (tumor vs background)
                            3 = multi-class (background, TC, ED)
            deep_supervision: Use deep supervision (V2 feature) ⭐ NEW
        """
```

**Dòng 7-13**: Constructor mới với tham số multi-class

**Tham số mới**:
- `num_classes_seg=1`: Số lớp segmentation
  - `1`: Binary segmentation (tumor vs background) - DEFAULT
  - `3`: Multi-class (Background, Tumor Core, Edema)
- `deep_supervision=False`: Sử dụng deep supervision hay không
- `roi_stop_grad=True`: Stop gradient qua ROI gating (như cũ)

### ROI Gating for Multi-Class

⭐ **NEW**: Forward pass xử lý cả binary và multi-class:

```python
def forward(self, x):
    seg_output = self.seg(x)

    # Handle deep supervision output
    if self.deep_supervision:
        seg_logits, aux_outputs = seg_output  # (B, C, H, W) và [aux3, aux2, aux1]
    else:
        seg_logits = seg_output  # (B, C, H, W)
        aux_outputs = None

    # ROI computation: for multi-class, use Whole Tumor (sum of all tumor classes)
    if self.num_classes_seg == 1:
        # Binary: use sigmoid
        seg_prob = torch.sigmoid(seg_logits)  # (B, 1, H, W)
    else:
        # Multi-class: use softmax and sum tumor classes (exclude background class 0)
        seg_prob = torch.softmax(seg_logits, dim=1)  # (B, 3, H, W)
        # Whole Tumor = sum of all tumor classes (classes 1, 2)
        seg_prob = seg_prob[:, 1:, :, :].sum(dim=1, keepdim=True)  # (B, 1, H, W)

    roi_input = self.reduce(x)

    if self.roi_stop_grad:
        roi = roi_input * seg_prob.detach()
    else:
        roi = roi_input * seg_prob

    cls_logits = self.cls_backbone(roi)

    if self.deep_supervision:
        return seg_logits, cls_logits, aux_outputs
    return seg_logits, cls_logits
```

**Dòng 25-56**: Forward pass mới hỗ trợ multi-class

### Phân Tích Chi Tiết

#### Binary Mode (num_classes_seg=1):
```python
# Input: (B, 4, 256, 256)
seg_logits = self.seg(x)    # (B, 1, 256, 256) - binary output
seg_prob = torch.sigmoid(seg_logits)  # (B, 1, 256, 256) - probabilities
```

**Output shape**: `(B, 1, H, W)` - Single channel cho tumor probability

#### Multi-Class Mode (num_classes_seg=3):
```python
# Input: (B, 4, 256, 256)
seg_logits = self.seg(x)    # (B, 3, 256, 256) - 3 classes
seg_prob = torch.softmax(seg_logits, dim=1)  # (B, 3, 256, 256)
# seg_prob[:, 0] = Background probability
# seg_prob[:, 1] = Tumor Core probability
# seg_prob[:, 2] = Edema probability

# For ROI gating, we want Whole Tumor (TC + ED)
seg_prob_wt = seg_prob[:, 1:, :, :].sum(dim=1, keepdim=True)  # (B, 1, 256, 256)
```

**Output shape**: `(B, 3, H, W)` - Three channels
- Channel 0: Background
- Channel 1: Tumor Core (TC)
- Channel 2: Edema (ED)

**Tại sao sum tumor classes cho ROI?**
- Classifier cần biết **toàn bộ vùng tumor** (Whole Tumor = WT)
- WT = TC + ED (classes 1 và 2)
- Background (class 0) bị bỏ qua
- Kết quả: ROI mask bao gồm tất cả tumor regions

### Tensor Shape Examples

**Binary Mode**:
```
Input:        (4, 4, 256, 256)  # 4 images, 4 modalities
              ↓
Seg U-Net:    (4, 1, 256, 256)  # 1 channel: tumor probability
              ↓ sigmoid
Seg Prob:     (4, 1, 256, 256)  # probabilities [0, 1]
              ↓ ROI gating
ROI:          (4, 1, 256, 256)  # gated input
              ↓ Inception
Cls Logits:   (4, 2)            # HGG vs LGG
```

**Multi-Class Mode**:
```
Input:        (4, 4, 256, 256)  # 4 images, 4 modalities
              ↓
Seg U-Net:    (4, 3, 256, 256)  # 3 channels: bg, TC, ED
              ↓ softmax
Seg Prob:     (4, 3, 256, 256)  # probabilities [0, 1] sum to 1
              ↓ sum classes 1,2
Seg Prob WT:  (4, 1, 256, 256)  # Whole Tumor probability
              ↓ ROI gating
ROI:          (4, 1, 256, 256)  # gated input
              ↓ Inception
Cls Logits:   (4, 2)            # HGG vs LGG
```

### Ví Dụ Sử Dụng Multi-Class

```python
# Multi-class segmentation với V2
from braintumnet.models.seg_unet_v2 import SegUNetV2

model = BrainTumNet(
    in_ch=4,              # 4 MRI modalities
    num_cls=2,            # Binary classification (HGG/LGG)
    base=48,              # V2 base channels (larger)
    dim=384,              # V2 transformer dimension
    patch=8,
    depth=4,              # V2 depth (deeper)
    n_heads=8,            # V2 heads (more)
    num_classes_seg=3,    # ⭐ Multi-class: 3 classes
    deep_supervision=True, # ⭐ Deep supervision
)

# Forward pass
img = torch.randn(4, 4, 256, 256)
seg_logits, cls_logits, aux_outputs = model(img)

# Shapes
print(seg_logits.shape)     # (4, 3, 256, 256) - 3 class segmentation
print(cls_logits.shape)     # (4, 2) - classification scores
print(len(aux_outputs))     # 3 - auxiliary outputs from d3, d2, d1
print(aux_outputs[0].shape) # (4, 3, 64, 64) - aux from d3
```

### Key Differences: Binary vs Multi-Class

| Aspect | Binary (num_classes_seg=1) | Multi-Class (num_classes_seg=3) |
|--------|----------------------------|----------------------------------|
| **Output channels** | 1 (tumor probability) | 3 (bg, TC, ED probabilities) |
| **Activation** | Sigmoid | Softmax |
| **Loss function** | DiceCELoss or DiceFocalLoss | MultiClassDiceLoss or MultiClassFocalLoss |
| **ROI computation** | Direct sigmoid output | Sum of tumor classes (1, 2) |
| **Evaluation metrics** | Dice, IoU for WT | Dice, IoU for WT, TC, ED separately |
| **Clinical utility** | Tumor location | Tumor sub-regions (TC vs ED) |

---

## U-Net Segmentation với Transformer

**File**: `src/braintumnet/models/seg_unet.py` (67 dòng)

U-Net là kiến trúc encoder-decoder với skip connections. Phiên bản này được nâng cấp với:
- **CBAM attention** trên tất cả skip connections
- **Adaptive Masked Transformer** ở bottleneck

### Sơ đồ kiến trúc

```
Input (B, 4, 256, 256)
        ↓
   [EncoderBlock 1] → skip1 (CBAM) → (B, 32, 256, 256)
        ↓ MaxPool
   (B, 32, 128, 128)
        ↓
   [EncoderBlock 2] → skip2 (CBAM) → (B, 64, 128, 128)
        ↓ MaxPool
   (B, 64, 64, 64)
        ↓
   [EncoderBlock 3] → skip3 (CBAM) → (B, 128, 64, 64)
        ↓ MaxPool
   (B, 128, 32, 32)
        ↓
   [EncoderBlock 4] → skip4 (CBAM) → (B, 256, 32, 32)
        ↓ MaxPool
   (B, 256, 16, 16)
        ↓
   ┌────────────────────┐
   │  [Bottleneck Conv] │
   │         ↓          │
   │  [Masked Transform]│  ← Global context với soft masking
   │         ↓          │
   │  [Upsample Back]   │
   └────────┬───────────┘
        ↓
   (B, 256, 16, 16)
        ↓
   [DecoderBlock 4] ← skip4
        ↓ Upsample
   (B, 256, 32, 32)
        ↓
   [DecoderBlock 3] ← skip3
        ↓ Upsample
   (B, 128, 64, 64)
        ↓
   [DecoderBlock 2] ← skip2
        ↓ Upsample
   (B, 64, 128, 128)
        ↓
   [DecoderBlock 1] ← skip1
        ↓ Upsample
   (B, 32, 256, 256)
        ↓
   [1×1 Conv Head]
        ↓
Output Segmentation (B, 1, 256, 256)
```

### Helper Functions

```python
def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )
```

**Dòng 5-10**: Helper tạo Conv-BN-ReLU block

**Tại sao thứ tự này?**
1. **Conv**: Học spatial features
2. **BatchNorm**: Normalize activations (ổn định training)
3. **ReLU**: Nonlinearity

**Tại sao `bias=False`?**
- BatchNorm có learnable shift parameter
- Bias trong conv sẽ bị loại bỏ bởi BatchNorm
- Tiết kiệm parameters

**Tham số mặc định**:
- `k=3`: 3×3 kernel (standard cho CNNs)
- `s=1`: Stride 1 (không downsampling)
- `p=1`: Padding 1 (giữ kích thước không đổi)

---

### EncoderBlock

```python
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(conv_bn_relu(in_ch, out_ch), conv_bn_relu(out_ch, out_ch))
        self.pool = nn.MaxPool2d(2)
```

**Dòng 12-16**: Encoder block initialization

**Kiến trúc**:
```
Input (in_ch channels)
    ↓
[Conv-BN-ReLU] (in_ch → out_ch)
    ↓
[Conv-BN-ReLU] (out_ch → out_ch)
    ↓
Skip Connection Output
    ↓
[MaxPool2d] (giảm kích thước 1/2)
    ↓
Downsampled Output
```

**Tại sao 2 convolutions?**
- Tăng receptive field
- Học features phức tạp hơn
- Standard U-Net pattern

---

```python
    def forward(self, x):
        x = self.block(x)
        return x, self.pool(x)
```

**Dòng 17-19**: Encoder forward pass

**Returns**:
- `x`: Features trước pooling (cho skip connection)
- `self.pool(x)`: Features sau pooling (truyền đến layer tiếp theo)

**Ví dụ với shapes**:
```python
# Input: (B, 64, 128, 128)
encoder = EncoderBlock(64, 128)
skip, down = encoder(x)
# skip: (B, 128, 128, 128) - giữ nguyên kích thước không gian
# down: (B, 128, 64, 64) - giảm 1/2 bởi MaxPool
```

---

### DecoderBlock

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, 2)
        self.cbam = CBAM(in_ch)
        self.block = nn.Sequential(conv_bn_relu(in_ch*2, out_ch), conv_bn_relu(out_ch, out_ch))
```

**Dòng 21-26**: Decoder block initialization

**Ba thành phần chính**:

1. **Upsampling**:
```python
self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, 2)
```
- Tăng kích thước không gian 2×
- kernel_size=2, stride=2: đúng gấp đôi không có overlapping
- Giữ nguyên số kênh

2. **CBAM Attention**:
```python
self.cbam = CBAM(in_ch)
```
- Áp dụng cho skip connection
- Lọc features quan trọng từ encoder

3. **Convolution Blocks**:
```python
self.block = nn.Sequential(conv_bn_relu(in_ch*2, out_ch), conv_bn_relu(out_ch, out_ch))
```
- `in_ch*2`: Concatenate upsampled features + skip connection
- Hai convs để refine features

---

```python
    def forward(self, x, skip):
        x = self.up(x)
        skip = self.cbam(skip)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)
```

**Dòng 27-31**: Decoder forward pass

**Từng bước chi tiết**:

1. **Upsample**:
```python
x = self.up(x)  # (B, C, H, W) → (B, C, 2H, 2W)
```

2. **Apply CBAM attention lên skip**:
```python
skip = self.cbam(skip)  # Lọc features quan trọng
```

3. **Concatenate**:
```python
x = torch.cat([x, skip], dim=1)  # (B, C, H, W) + (B, C, H, W) → (B, 2C, H, W)
```

4. **Refine với convolutions**:
```python
return self.block(x)  # (B, 2C, H, W) → (B, out_ch, H, W)
```

**Ví dụ với shapes**:
```python
# Decoder input: (B, 128, 32, 32)
# Skip connection: (B, 128, 64, 64)
decoder = DecoderBlock(128, 64)
out = decoder(x, skip)
# out: (B, 64, 64, 64)
```

---

### SegUNetMasked

```python
class SegUNetMasked(nn.Module):
    def __init__(self, in_ch=1, base=32, dim=256, patch=8, depth=2, n_heads=4):
        super().__init__()
        self.patch = patch
```

**Dòng 34-37**: Khởi tạo main segmentation model

**Tham số**:
- `in_ch=1`: Số kênh đầu vào
- `base=32`: Base channel multiplier
- `dim=256`: Transformer embedding dimension
- `patch=8`: Transformer patch size
- `depth=2`: Số transformer blocks
- `n_heads=4`: Attention heads

---

```python
        self.e1 = EncoderBlock(in_ch, base)
        self.e2 = EncoderBlock(base, base*2)
        self.e3 = EncoderBlock(base*2, base*4)
        self.e4 = EncoderBlock(base*4, base*8)
```

**Dòng 38-41**: Xây dựng encoder

**Channel Progression** (với base=32):
- e1: 1 → 32 channels
- e2: 32 → 64 channels
- e3: 64 → 128 channels
- e4: 128 → 256 channels

**Spatial Progression**:
- Sau e1: 256 → 128
- Sau e2: 128 → 64
- Sau e3: 64 → 32
- Sau e4: 32 → 16

**Tại sao tăng gấp đôi số kênh?**
- Pattern chuẩn của U-Net
- Khi độ phân giải không gian giảm, tăng độ sâu kênh
- Duy trì khả năng biểu diễn

---

```python
        # Sau 4 encoder blocks: spatial size là H/16 x W/16
        # Transformer sẽ giảm thêm theo patch size
        self.bottleneck_conv = conv_bn_relu(base*8, dim, k=1, s=1, p=0)
        self.amt = AdaptiveMaskedTransformer(in_ch=dim, dim=dim, patch_size=patch, depth=depth, n_heads=n_heads)
        # Upsample transformer output về kích thước bottleneck ban đầu
        self.tr_upsample = nn.ConvTranspose2d(dim, base*8, kernel_size=patch, stride=patch)
```

**Dòng 42-47**: Transformer bottleneck

**Từng bước chi tiết**:

1. **Bottleneck Conv (1×1)**:
```python
self.bottleneck_conv = conv_bn_relu(base*8, dim, k=1, s=1, p=0)
```
- Thay đổi số kênh: base*8 (256) → dim (256)
- Kích thước không gian không đổi: 16×16
- `k=1`: 1×1 convolution (channel mixer)
- `p=0`: Không cần padding cho 1×1 conv

2. **Adaptive Masked Transformer**:
```python
self.amt = AdaptiveMaskedTransformer(...)
```
- Xử lý feature map 16×16
- Chia thành patches 8×8 → lưới 2×2 patches
- Áp dụng self-attention với learned soft masks

3. **Upsample trở lại**:
```python
self.tr_upsample = nn.ConvTranspose2d(dim, base*8, kernel_size=patch, stride=patch)
```
- Sau transformer: 2×2 patches
- Upsample: 2×2 → 16×16 (patch size 8)
- Số kênh: dim (256) → base*8 (256)

**Tại sao thiết kế này?**
- Transformer hoạt động trên coarse features (16×16)
- Hiệu quả hơn full resolution
- Bắt global context
- Adaptive masking tập trung vào vùng liên quan

---

```python
        self.d4 = DecoderBlock(base*8, base*8)
        self.d3 = DecoderBlock(base*8, base*4)
        self.d2 = DecoderBlock(base*4, base*2)
        self.d1 = DecoderBlock(base*2, base)
        self.head = nn.Conv2d(base, 1, 1)
```

**Dòng 48-52**: Xây dựng decoder

**Cấu trúc Decoder** (với base=32):
- d4: 256 → 256 channels, 16×16 → 32×32
- d3: 256 → 128 channels, 32×32 → 64×64
- d2: 128 → 64 channels, 64×64 → 128×128
- d1: 64 → 32 channels, 128×128 → 256×256
- head: 32 → 1 channel (final segmentation map)

**Tại sao 1×1 Conv cho Head?**
- Không cần spatial context ở layer cuối
- Chỉ chuyển đổi số kênh sang output
- Tiết kiệm tính toán

---

```python
    def forward(self, x):
        s1, x1 = self.e1(x)      # s1: base, H, W
        s2, x2 = self.e2(x1)     # s2: base*2, H/2, W/2
        s3, x3 = self.e3(x2)     # s3: base*4, H/4, W/4
        s4, x4 = self.e4(x3)     # s4: base*8, H/8, W/8
```

**Dòng 53-57**: Encoder forward pass

**Shapes chi tiết** (với input 256×256, base=32):
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
        b = self.tr_upsample(b)  # base*8, H/16, W/16 (upsampled lại)
```

**Dòng 58-60**: Transformer bottleneck forward

**Shapes chi tiết**:
```
x4:                    (B, 256, 16, 16)
    ↓ bottleneck_conv
b:                     (B, 256, 16, 16)
    ↓ amt (Transformer với patch=8)
b:                     (B, 256, 2, 2)    ← Đã chia patches!
    ↓ tr_upsample
b:                     (B, 256, 16, 16)  ← Trở về kích thước ban đầu
```

**Điều gì xảy ra trong Transformer?**
1. Input: feature map 16×16
2. Chia thành patches 8×8 → 2×2=4 patches
3. Mỗi patch trở thành một token
4. Self-attention giữa 4 tokens
5. Tái tạo feature map 2×2
6. Upsample lại về 16×16

---

```python
        x = self.d4(b, s4)       # base*8, H/8, W/8
        x = self.d3(x, s3)       # base*4, H/4, W/4
        x = self.d2(x, s2)       # base*2, H/2, W/2
        x = self.d1(x, s1)       # base, H, W
        seg = self.head(x)       # 1, H, W
        return seg
```

**Dòng 61-66**: Decoder forward pass

**Shapes chi tiết** (base=32):
```
b:        (B, 256, 16, 16)
    ↓ d4(b, s4)  [s4 có CBAM attention áp dụng]
x:        (B, 256, 32, 32)
    ↓ d3(x, s3)  [s3 có CBAM attention áp dụng]
x:        (B, 128, 64, 64)
    ↓ d2(x, s2)  [s2 có CBAM attention áp dụng]
x:        (B, 64, 128, 128)
    ↓ d1(x, s1)  [s1 có CBAM attention áp dụng]
x:        (B, 32, 256, 256)
    ↓ head
seg:      (B, 1, 256, 256)  ← Segmentation logits cuối cùng
```

---

## CBAM Attention Mechanism

**File**: `src/braintumnet/models/cbam.py` (33 dòng)

CBAM (Convolutional Block Attention Module) áp dụng **hai loại attention tuần tự**:
1. **Channel Attention**: Feature maps nào quan trọng?
2. **Spatial Attention**: Vị trí không gian nào quan trọng?

### Sơ đồ kiến trúc

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

**Dòng 3-12**: Khởi tạo channel attention

**Channel Attention là gì?**
- Học channel features nào quan trọng
- Ví dụ: Trong y tế, một số channels có thể phát hiện edges, channels khác phát hiện textures
- Attention weights giảm channels ít hữu ích

**Các thành phần**:

1. **Global Pooling**:
```python
self.avg = nn.AdaptiveAvgPool2d(1)  # (C, H, W) → (C, 1, 1)
self.max = nn.AdaptiveMaxPool2d(1)  # (C, H, W) → (C, 1, 1)
```
- Tóm tắt thông tin không gian cho mỗi channel
- `AdaptiveAvgPool2d(1)`: Trung bình của tất cả vị trí không gian
- `AdaptiveMaxPool2d(1)`: Maximum của tất cả vị trí không gian
- Kết quả: Một giá trị cho mỗi channel

2. **MLP (Multi-Layer Perceptron)**:
```python
self.mlp = nn.Sequential(
    nn.Conv2d(in_channels, in_channels//reduction, 1, bias=False),  # Nén
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False),  # Mở rộng
)
```
- Sử dụng 1×1 convolutions (tương đương fully connected cho kích thước không gian 1×1)
- **Thiết kế bottleneck**: C → C/16 → C
- `reduction=16`: Tỷ lệ nén (giảm parameters)

**Tại sao Reduction?**
- Original: C → C sẽ đắt
- Với reduction: C → C/16 → C
- Với C=256: 256×256=65k params vs 256×16 + 16×256=8k params
- **Giảm 8× parameters!**

---

```python
    def forward(self, x):
        att = torch.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))
        return x * att
```

**Dòng 13-15**: Channel attention forward pass

**Từng bước chi tiết**:
```python
# Input: x shape (B, C, H, W)
avg_pool = self.avg(x)        # (B, C, 1, 1)
max_pool = self.max(x)        # (B, C, 1, 1)

avg_feat = self.mlp(avg_pool) # (B, C, 1, 1)
max_feat = self.mlp(max_pool) # (B, C, 1, 1)

att = torch.sigmoid(avg_feat + max_feat)  # (B, C, 1, 1), giá trị trong [0, 1]

output = x * att              # Broadcasting: (B, C, H, W) * (B, C, 1, 1)
```

**Tại sao cả Average và Max?**
- Average: Bắt tầm quan trọng channel tổng thể
- Max: Bắt peak activations
- Kết hợp cả hai cho biểu diễn phong phú hơn

**Ví dụ**:
```
Input feature map:     Avg Pool:    Max Pool:    Combined Attention:
┌─────────┐            ┌───┐        ┌───┐        ┌───┐
│ 0 1 2 3 │            │1.5│        │ 3 │        │2.8│ ← Weight cao hơn
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

**Dòng 17-20**: Khởi tạo spatial attention

**Spatial Attention là gì?**
- Học vị trí không gian nào quan trọng
- Ví dụ: Tập trung vào vùng tumor, bỏ qua background
- Bổ sung cho channel attention

**Các thành phần**:
```python
self.conv = nn.Conv2d(2, 1, k, padding=k//2, bias=False)
```
- Input: 2 channels (avg và max qua các channels)
- Output: 1 channel (spatial attention map)
- Kernel size: k=7 (receptive field lớn)
- Padding: k//2 = 3 (giữ kích thước không gian)

**Tại sao k=7?**
- Kernel lớn hơn bắt nhiều spatial context hơn
- Lựa chọn chuẩn từ CBAM paper
- Cân bằng receptive field vs tính toán

---

```python
    def forward(self, x):
        att = torch.cat([x.mean(1, True), x.amax(1, True)], dim=1)
        att = torch.sigmoid(self.conv(att))
        return x * att
```

**Dòng 21-24**: Spatial attention forward pass

**Từng bước chi tiết**:
```python
# Input: x shape (B, C, H, W)

avg_spatial = x.mean(1, True)   # (B, 1, H, W) - trung bình qua các channels
max_spatial = x.amax(1, True)   # (B, 1, H, W) - max qua các channels

att = torch.cat([avg_spatial, max_spatial], dim=1)  # (B, 2, H, W)

att = self.conv(att)            # (B, 1, H, W) - học spatial weights
att = torch.sigmoid(att)        # Giá trị trong [0, 1]

output = x * att                # (B, C, H, W) * (B, 1, H, W) - broadcast
```

**Ví dụ trực quan**:
```
Input (C=3 channels):
Channel 0:    Channel 1:    Channel 2:
┌─────┐       ┌─────┐       ┌─────┐
│0 1 2│       │3 4 5│       │6 7 8│
│3 4 5│       │6 7 8│       │9 0 1│
└─────┘       └─────┘       └─────┘
     ↓             ↓             ↓
     Trung bình qua channels (mean)
               ↓
         Avg Spatial:
         ┌─────┐
         │3 4 5│  ← Trung bình của [0,3,6], [1,4,7], [2,5,8]...
         │6 7 8│
         └─────┘
     ↓             ↓             ↓
     Max qua channels (amax)
               ↓
         Max Spatial:
         ┌─────┐
         │6 7 8│  ← Max của [0,3,6], [1,4,7], [2,5,8]...
         │9 7 8│
         └─────┘
```

**Tại sao cả Mean và Max?**
- Mean: Tầm quan trọng feature tổng thể tại mỗi vị trí
- Max: Peak features tại mỗi vị trí
- Cùng nhau: Spatial attention mạnh mẽ

---

### CBAM (Kết hợp cả hai)

```python
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, k=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(k)
    def forward(self, x):
        return self.sa(self.ca(x))
```

**Dòng 26-32**: Module CBAM hoàn chỉnh

**Áp dụng tuần tự**:
```python
x → ChannelAttention → x_ca → SpatialAttention → x_out
```

**Tại sao tuần tự (không song song)?**
- Channel attention trước để refine features nào quan trọng
- Spatial attention sau để refine chúng quan trọng ở đâu
- Thực nghiệm cho thấy hoạt động tốt hơn song song

**Ví dụ luồng dữ liệu**:
```
Input (B, 64, 32, 32)
    ↓
ChannelAttention: Weights shape (B, 64, 1, 1)
    ↓ Áp dụng weights
(B, 64, 32, 32) với một số channels bị giảm
    ↓
SpatialAttention: Weights shape (B, 1, 32, 32)
    ↓ Áp dụng weights
(B, 64, 32, 32) với một số vị trí bị giảm
```

**Chi phí tính toán**:
- Channel attention: Rất rẻ (chỉ xử lý pooled features)
- Spatial attention: Vừa phải (7×7 conv trên full spatial size)
- Tổng: <1% tổng FLOPs của model
- **Chi phí nhỏ, cải thiện đáng kể!**

---

## Adaptive Masked Transformer

**File**: `src/braintumnet/models/masked_transformer.py` (88 dòng)

Đây là **thành phần phức tạp nhất** của BrainTumNet. Áp dụng self-attention với **learned soft masks** tập trung adaptive vào vùng ảnh quan trọng.

### Sơ đồ kiến trúc

```
Input Features (B, C, 16, 16)
         ↓
    [PatchEmbed]
         ↓
  Tokens (B, N, C)  với N=4 patches
         ↓
    ┌────┴────┐
    │         │
    │ [Soft   │
    │  Mask   │
    │  Gen]   │
    │         │
    └────┬────┘
         ↓
  Soft Masks (B, H, N)  với H=num_heads
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
  [Reshape về 2D]
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

**Dòng 3-7**: Khởi tạo patch embedding

**Patch Embedding là gì?**
- Chuyển đổi ảnh 2D thành chuỗi 1D của tokens
- Mỗi token đại diện cho một image patch
- Kỹ thuật chuẩn trong Vision Transformers

**Các thành phần**:
```python
self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
```
- Convolution với kernel size = stride = patch size
- Non-overlapping patches
- Ví dụ: ảnh 16×16, patch=8 → 2×2 = 4 patches

```python
self.norm = nn.LayerNorm(embed_dim)
```
- Normalize token embeddings
- Ổn định training

---

```python
    def forward(self, x):
        x = self.proj(x)  # B,C,H',W'
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)  # B,N,C
        x = self.norm(x)
        return x, (H,W)
```

**Dòng 8-13**: Patch embedding forward pass

**Từng bước chi tiết**:
```python
# Input: x shape (B, 256, 16, 16), patch=8

x = self.proj(x)              # (B, 256, 2, 2) - patches 8×8 → lưới 2×2
B, C, H, W = x.shape          # B=batch, C=256, H=2, W=2

x = x.flatten(2)              # (B, 256, 4) - flatten spatial dims
x = x.transpose(1, 2)         # (B, 4, 256) - swap sang format (B, N, C)
x = self.norm(x)              # Normalize

return x, (H, W)              # Trả về tokens và spatial shape
```

**Ví dụ trực quan**:
```
Input ảnh 16×16:
┌───────┬───────┐
│ P0    │ P1    │  ← Mỗi patch 8×8 trở thành một token
│       │       │
├───────┼───────┤
│ P2    │ P3    │
│       │       │
└───────┴───────┘

Sau Projection:
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

**Dòng 15-22**: Khởi tạo soft mask generator

**Soft Masks là gì?**
- **Hard Mask**: Binary (0 hoặc 1) - bao gồm hoặc loại trừ token
- **Soft Mask**: Liên tục [0, 1] - partial attention weight
- Cho phép gradients chảy, trainable end-to-end

**Tại sao Soft Masks?**
- **Adaptive**: Học patches nào quan trọng
- **Differentiable**: Có thể backpropagate qua masks
- **Per-Head**: Các head khác nhau có thể tập trung vào vùng khác nhau

**Kiến trúc**:
```python
self.mlp = nn.Sequential(
    nn.Linear(dim, hidden),     # 256 → 128 (nén)
    nn.GELU(),                  # Smooth nonlinearity
    nn.Linear(hidden, n_heads), # 128 → 4 (một mask cho mỗi head)
    nn.Sigmoid()                # Output trong [0, 1]
)
```

**Tại sao GELU (không phải ReLU)?**
- GELU (Gaussian Error Linear Unit): Mượt hơn ReLU
- Chuẩn trong transformers (BERT, GPT, ViT)
- Gradient flow tốt hơn

---

```python
    def forward(self, tokens):  # B,N,C
        m = self.mlp(tokens)    # B,N,H
        return m.permute(0,2,1).contiguous()  # B,H,N
```

**Dòng 23-25**: Soft mask generator forward pass

**Từng bước chi tiết**:
```python
# Input: tokens shape (B, 4, 256)

m = self.mlp(tokens)          # (B, 4, 4) - 4 tokens, 4 heads
                               # Mỗi giá trị là attention weight trong [0, 1]

m = m.permute(0, 2, 1)        # (B, 4, 4) - sắp xếp lại thành (B, heads, tokens)
m = m.contiguous()            # Đảm bảo memory contiguity
```

**Ví dụ trực quan**:
```
Tokens (4 patches):
┌──────────────────────┐
│ Token 0 (Background) │ → Mask weights [0.1, 0.2, 0.1, 0.3]  ← Weights thấp
│ Token 1 (Tumor Edge) │ → Mask weights [0.9, 0.8, 0.7, 0.6]  ← Weights cao
│ Token 2 (Tumor Core) │ → Mask weights [1.0, 0.9, 0.9, 0.8]  ← Weights cao nhất
│ Token 3 (Background) │ → Mask weights [0.2, 0.1, 0.2, 0.2]  ← Weights thấp
└──────────────────────┘
         ↑                    ↑
      Input              4 attention heads
                         (mỗi head có mask khác nhau)
```

**Ý tưởng chính**:
- Background tokens nhận weights thấp → ít attention
- Tumor tokens nhận weights cao → nhiều attention hơn
- **Adaptive**: Giá trị mask được học trong training!

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

**Dòng 27-37**: Khởi tạo masked self-attention

**Tham số**:
- `dim=256`: Embedding dimension
- `n_heads=4`: Số attention heads
- `head_dim=64`: Dimension mỗi head (256/4)
- `attn_drop=0.0`: Attention dropout (thường 0 cho datasets nhỏ)
- `proj_drop=0.0`: Projection dropout

**Các thành phần**:
```python
self.qkv = nn.Linear(dim, dim*3, bias=False)
```
- Single matrix project sang Q, K, V đồng thời
- Hiệu quả hơn 3 projections riêng biệt
- Output: dim*3 (256*3=768)

```python
self.proj = nn.Linear(dim, dim)
```
- Output projection sau attention
- Mix thông tin từ các heads khác nhau

---

```python
    def forward(self, x, softmask):  # x: B,N,C ; softmask: B,H,N
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.n_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]  # B,H,N,D
```

**Dòng 38-41**: Tính Q, K, V

**Từng bước chi tiết**:
```python
# Input: x shape (B, 4, 256)

qkv = self.qkv(x)              # (B, 4, 768) - project sang 3*dim

qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim)
                                # (B, 4, 3, 4, 64)
                                # Chia thành Q/K/V và heads

qkv = qkv.permute(2, 0, 3, 1, 4)
                                # (3, B, 4, 4, 64)
                                # Sắp xếp lại để dễ indexing

q, k, v = qkv[0], qkv[1], qkv[2]
                                # Mỗi cái: (B, 4, 4, 64)
                                # q/k/v cho mỗi head
```

**Tóm tắt Shapes**:
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

**Dòng 42-46**: Tính attention với soft masking

**Từng bước chi tiết**:

1. **Standard Attention**:
```python
attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
```
- `q @ k.transpose(-2, -1)`: Dot product attention scores
- Shape: (B, 4, 4, 4) - 4 heads, ma trận attention 4×4
- `/sqrt(head_dim)`: Scale bởi √64 = 8 (ngăn giá trị lớn)

2. **Thêm Soft Mask** (PHẦN MỚI!):
```python
key_bias = torch.log(softmask.unsqueeze(-2) + 1e-6)  # (B, 4, 1, 4)
attn = attn + key_bias
```

**Tại sao `torch.log(softmask)`?**

Điều này là toán học elegent! Để tôi giải thích:

Standard softmax:
```
softmax(attn_i) = exp(attn_i) / Σ exp(attn_j)
```

Với mask trong log-space:
```
softmax(attn_i + log(mask_i)) = exp(attn_i + log(mask_i)) / Σ exp(attn_j + log(mask_j))
                                = exp(attn_i) * mask_i / Σ (exp(attn_j) * mask_j)
```

**Hiệu ứng**:
- Nếu mask_i = 1.0 → log(1.0) = 0 → không thay đổi
- Nếu mask_i = 0.5 → log(0.5) = -0.69 → giảm attention
- Nếu mask_i ≈ 0 → log(0) = -∞ → zero attention

**Tại sao thêm 1e-6?**
- Ngăn log(0) = -inf
- Ổn định số học

**Ví dụ trực quan**:
```
Attention Scores (trước masking):
      T0   T1   T2   T3
T0 [ 0.5  0.3  0.1  0.1 ]
T1 [ 0.2  0.6  0.1  0.1 ]
T2 [ 0.1  0.2  0.5  0.2 ]
T3 [ 0.1  0.1  0.2  0.6 ]

Soft Mask (đã học):
[ 0.2  0.9  1.0  0.1 ]  ← Tầm quan trọng token

Sau Masking (softmax với bias):
      T0   T1   T2   T3
T0 [ 0.2  0.5  0.3  0.0 ]  ← Attention chuyển sang T1, T2
T1 [ 0.1  0.8  0.1  0.0 ]
T2 [ 0.0  0.3  0.6  0.1 ]
T3 [ 0.1  0.2  0.3  0.4 ]
```

3. **Softmax và Dropout**:
```python
attn = attn.softmax(-1)       # Normalize sang probabilities
attn = self.attn_drop(attn)   # Áp dụng dropout (nếu có)
```

---

```python
        out = (attn @ v).transpose(1,2).reshape(B,N,C)
        out = self.proj_drop(self.proj(out))
        return out
```

**Dòng 47-49**: Áp dụng attention và project

**Từng bước chi tiết**:
```python
out = attn @ v                # (B, 4, 4, 4) @ (B, 4, 4, 64) = (B, 4, 4, 64)
out = out.transpose(1, 2)     # (B, 4, 4, 64) - swap heads và tokens
out = out.reshape(B, N, C)    # (B, 4, 256) - merge heads

out = self.proj(out)          # (B, 4, 256) - output projection
out = self.proj_drop(out)     # Áp dụng dropout
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

**Dòng 51-61**: Feed-forward MLP

**Kiến trúc**:
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

**Tại sao mlp_ratio=4.0?**
- Chuẩn trong transformers (BERT, GPT)
- Cho phép học complex nonlinear transformations
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

**Dòng 63-73**: Transformer block với residual connections

**Kiến trúc** (Pre-Norm + Residual):
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

**Tại sao Pre-Norm?**
- Normalize trước attention/MLP (không phải sau)
- Training ổn định hơn
- Chuẩn trong modern transformers

**Tại sao Residual Connections?**
- `x = x + module(x)`: Skip connection
- Cho phép gradients chảy trực tiếp
- Ngăn vanishing gradients trong deep networks

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

**Dòng 75-80**: Khởi tạo complete transformer

**Các thành phần**:
1. `PatchEmbed`: Chuyển 2D → 1D tokens
2. `SoftMaskGenerator`: Học adaptive masks
3. `blocks`: Stack của `depth=2` transformer blocks

**Tại sao depth=2?**
- Đủ shallow cho feature maps nhỏ (16×16)
- Đủ deep để bắt interactions
- Thực nghiệm cho thấy hoạt động tốt nhất

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

**Dòng 81-87**: Complete transformer forward pass

**Từng bước chi tiết**:
```python
# Input: x shape (B, 256, 16, 16)

tokens, (H, W) = self.pe(x)    # (B, 4, 256), H=2, W=2

softmask = self.mask_gen(tokens) # (B, 4, 4) - adaptive masks

for blk in self.blocks:
    tokens = blk(tokens, softmask) # (B, 4, 256) - áp dụng 2 transformer blocks

feat = tokens.transpose(1, 2)    # (B, 256, 4)
feat = feat.reshape(x.size(0), tokens.size(-1), H, W)
                                  # (B, 256, 2, 2) - reshape về 2D
return feat
```

**Điểm chính**:
- Mask được tạo một lần, dùng trong tất cả blocks
- Cùng mask cho tất cả transformer layers
- Output reshape lại thành 2D feature map

---

## Inception Classification Network

**File**: `src/braintumnet/models/t_inception.py` (51 dòng)

Inception networks bắt **multi-scale features** sử dụng parallel convolutions với kernel sizes khác nhau.

### Sơ đồ kiến trúc

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

**Dòng 4-18**: Single inception branch

**Các kernels được hỗ trợ**:
- `(1, 1)`: Point-wise convolution (không có spatial context)
- `(1, 3)`: Horizontal features (cấu trúc kéo dài)
- `(3, 1)`: Vertical features (cấu trúc kéo dài)
- `(3, 3)`: Standard square features

**Tại sao các Kernels khác nhau?**
- Brain tumors có hình dạng đa dạng: tròn, kéo dài, không đều
- Các kernels khác nhau bắt các geometric features khác nhau
- Parallel branches → biểu diễn phong phú hơn

**Ví dụ trực quan**:
```
Hình dạng Tumor:
┌─────────┐
│   ●●    │  ← Tumor tròn: bắt tốt nhất bởi 3×3
│  ●●●●   │
│   ●●    │
│         │
│ ●●●●●●● │  ← Tumor ngang: bắt tốt nhất bởi 1×3
│         │
│    ●    │  ← Tumor dọc: bắt tốt nhất bởi 3×1
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

**Dòng 20-30**: Complete inception block

**Kiến trúc**:
```
Input (in_ch channels)
    ↓
┌───┴───┬───────┬───────┬───────┐
│       │       │       │       │
│ 1×1   │ 3×3   │ 1×3   │ 3×1   │  ← 4 parallel branches
│       │       │       │       │
└───┬───┴───┬───┴───┬───┴───┬───┘
    │       │       │       │
    │   (c  │  ch   │  mỗi) │
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

**Phân bổ Channels**:
- Mỗi branch: out_ch / 4 channels
- Sau concat: out_ch channels tổng
- Fuse conv: c*4 → out_ch (duy trì channels)

---

```python
    def forward(self, x):
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.act(self.bn(self.fuse(x)))
```

**Dòng 31-33**: Inception forward pass

**Từng bước chi tiết**:
```python
b1_out = self.b1(x)  # (B, c, H, W) - features 1×1
b2_out = self.b2(x)  # (B, c, H, W) - features 3×3
b3_out = self.b3(x)  # (B, c, H, W) - features 1×3
b4_out = self.b4(x)  # (B, c, H, W) - features 3×1

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

**Dòng 35-43**: Complete inception classifier

**Kiến trúc**:
- **Stem**: 1 → 64 channels (trích xuất features ban đầu)
- **Block 1**: 64 → 128 channels (multi-scale features)
- **Block 2**: 128 → 256 channels (higher-level features)
- **Global Pool**: 256×256 → 1×1 (spatial → vector)
- **Dropout**: 30% (ngăn overfitting)
- **FC**: 256 → 2 (phân loại cuối cùng)

**Tại sao Dropout 0.3?**
- Classification head dễ overfitting
- 30% là chuẩn cho medical imaging
- Cân bằng regularization vs capacity

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

**Dòng 44-50**: Inception forward pass

**Shapes chi tiết** (với input 256×256):
```python
x = self.stem(x)      # (B, 1, 256, 256) → (B, 64, 256, 256)
x = self.b1(x)        # (B, 64, 256, 256) → (B, 128, 256, 256)
x = self.b2(x)        # (B, 128, 256, 256) → (B, 256, 256, 256)
x = self.pool(x)      # (B, 256, 256, 256) → (B, 256, 1, 1)
x = x.flatten(1)      # (B, 256, 1, 1) → (B, 256)
x = self.drop(x)      # (B, 256) - áp dụng dropout
x = self.fc(x)        # (B, 256) → (B, 2)
```

---

## Luồng dữ liệu hoàn chỉnh với Tensor Shapes

Hãy trace một **single image** qua toàn bộ kiến trúc BrainTumNet với exact tensor shapes tại mỗi bước.

### Thiết lập Input

```python
batch_size = 4
in_channels = 4  # Multi-modal: FLAIR, T1, T1CE, T2
img_size = 256
num_classes = 2  # HGG vs LGG

input_image = torch.randn(batch_size, in_channels, img_size, img_size)
# Shape: (4, 4, 256, 256)
```

### Luồng từng bước

#### 1. Segmentation U-Net Encoder

```python
# Input vào SegUNetMasked
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
# Chuyển sang transformer embedding dimension
b = self.bottleneck_conv(x4)  # (4, 256, 16, 16) → (4, 256, 16, 16)

# Adaptive Masked Transformer
# - PatchEmbed
tokens, (H, W) = self.pe(b)  # (4, 256, 16, 16) → (4, 4, 256), H=2, W=2
#   4 patches tổng (lưới 2×2 với patch_size=8)

# - Tạo soft masks
softmask = self.mask_gen(tokens)  # (4, 4, 256) → (4, 4, 4)
#   Shape: (batch, heads, tokens)

# - Transformer blocks (depth=2)
for blk in self.blocks:
    tokens = blk(tokens, softmask)  # (4, 4, 256) → (4, 4, 256)

# - Reshape về 2D
feat = tokens.transpose(1,2).reshape(4, 256, 2, 2)  # (4, 256, 2, 2)

# Upsample lại về kích thước bottleneck
b = self.tr_upsample(feat)  # (4, 256, 2, 2) → (4, 256, 16, 16)
```

#### 3. Segmentation U-Net Decoder

```python
# Decoder Block 4
x = self.d4(b, s4)
# b: (4, 256, 16, 16), s4 (với CBAM): (4, 256, 32, 32)
# Output: (4, 256, 32, 32)

# Decoder Block 3
x = self.d3(x, s3)
# x: (4, 256, 32, 32), s3 (với CBAM): (4, 128, 64, 64)
# Output: (4, 128, 64, 64)

# Decoder Block 2
x = self.d2(x, s2)
# x: (4, 128, 64, 64), s2 (với CBAM): (4, 64, 128, 128)
# Output: (4, 64, 128, 128)

# Decoder Block 1
x = self.d1(x, s1)
# x: (4, 64, 128, 128), s1 (với CBAM): (4, 32, 256, 256)
# Output: (4, 32, 256, 256)

# Segmentation head
seg_logits = self.head(x)  # (4, 32, 256, 256) → (4, 1, 256, 256)
```

#### 4. ROI Gating

```python
# Chuyển logits thành probabilities
seg_prob = torch.sigmoid(seg_logits)  # (4, 1, 256, 256)

# Giảm số kênh input nếu cần
roi_input = self.reduce(input_image)  # (4, 4, 256, 256) → (4, 1, 256, 256)

# Áp dụng ROI gating (với gradient stopping)
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

### Tính toán Memory Usage

**Peak Memory** (với batch_size=4, giả sử FP32):

Tensors lớn nhất:
1. Input: 4 × 4 × 256 × 256 = 1,048,576 giá trị
2. Decoder features: 4 × 256 × 256 × 256 = 67,108,864 giá trị (Inception b2 output)
3. Skip connections: 4 × 256 × 32 × 32 = 1,048,576 giá trị

Tổng: ~270MB cho activations + ~12MB cho model parameters = **~282MB**

**Với FP16 (mixed precision)**: ~141MB

---

## Giải thích các quyết định thiết kế

### Tại sao kiến trúc này?

**Q: Tại sao kết hợp U-Net + Transformer + Inception?**

A: Mỗi thành phần phục vụ mục đích cụ thể:
- **U-Net**: Tốt nhất cho dense segmentation (đã được chứng minh trong medical imaging)
- **Transformer**: Bắt global context (tumor ảnh hưởng mô xung quanh)
- **Inception**: Multi-scale classification (tumors có kích thước và hình dạng khác nhau)

**Q: Tại sao CBAM attention trên skip connections?**

A: Skip connections có thể truyền noise từ encoder. CBAM lọc features quan trọng, cải thiện độ chính xác ranh giới. Ablation studies cho thấy cải thiện Dice +2%.

**Q: Tại sao adaptive masked transformer?**

A: Standard transformers attend đồng đều tất cả tokens. Medical images có vùng background lớn. Adaptive masking tập trung tính toán vào vùng liên quan (tumor và mô gần đó).

**Q: Tại sao stop gradient trong ROI gating?**

A: Segmentation là nhiệm vụ chính. Không có `.detach()`, classification loss sẽ ảnh hưởng segmentation, có thể làm giảm chất lượng mask.

### Lựa chọn Hyperparameter

**Q: Tại sao base=32 cho U-Net?**

A: Cân bằng model capacity và memory:
- base=16: Quá ít parameters, underfitting
- base=32: Sweet spot cho ảnh 256×256
- base=64: Cải thiện nhỏ, tốn 4× memory hơn

**Q: Tại sao patch_size=8?**

A: Sau 4 encoder blocks, kích thước không gian là 16×16. Với patch=8, ta có 2×2=4 patches. Quá ít patches (patch=16 → 1 patch) mất cấu trúc không gian. Quá nhiều (patch=4 → 16 patches) tốn tính toán.

**Q: Tại sao transformer depth=2?**

A: Với chỉ 4 tokens, deep transformers không cần thiết. Depth=2 bắt token interactions mà không overfitting.

**Q: Tại sao dropout=0.3 trong classifier?**

A: Classification head có capacity cao (256 features → 2 classes). Không có regularization, nó overfits. Dropout 30% là chuẩn cho medical imaging.

---

## Hướng dẫn chỉnh sửa

### Làm thế nào thay đổi Model Capacity

**Tăng Capacity** (cho datasets lớn hơn):
```python
model = BrainTumNet(
    in_ch=4,
    num_cls=2,
    base=64,        # Tăng từ 32
    dim=512,        # Tăng từ 256
    patch=8,
    depth=4,        # Tăng từ 2
    n_heads=8,      # Tăng từ 4
)
```

**Giảm Capacity** (cho datasets nhỏ hơn hoặc inference nhanh hơn):
```python
model = BrainTumNet(
    in_ch=4,
    num_cls=2,
    base=16,        # Giảm từ 32
    dim=128,        # Giảm từ 256
    patch=8,
    depth=1,        # Giảm từ 2
    n_heads=2,      # Giảm từ 4
)
```

### Làm thế nào thêm Deep Supervision

Deep supervision thêm auxiliary losses tại intermediate decoder layers để gradient flow tốt hơn.

**Chỉnh sửa `seg_unet.py`**:
```python
class SegUNetMasked(nn.Module):
    def __init__(self, in_ch=1, base=32, dim=256, patch=8, depth=2, n_heads=4, deep_supervision=False):
        super().__init__()
        # ... existing code ...

        self.deep_supervision = deep_supervision
        if deep_supervision:
            # Thêm auxiliary heads tại mỗi decoder level
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

**Cập nhật tính toán loss trong `trainer.py`**:
```python
if deep_supervision:
    seg_logits, aux_outputs = model.seg(img)
    # Main loss
    seg_loss = seg_criterion(seg_logits, msk)
    # Auxiliary losses (với weights giảm dần)
    for i, aux in enumerate(aux_outputs):
        weight = 0.5 ** (i+1)  # 0.5, 0.25, 0.125
        aux_resized = F.interpolate(aux, size=msk.shape[-2:], mode='bilinear')
        seg_loss += weight * seg_criterion(aux_resized, msk)
else:
    seg_logits = model.seg(img)
    seg_loss = seg_criterion(seg_logits, msk)
```

### Làm thế nào thêm Residual Connections vào U-Net

**Chỉnh sửa `seg_unet.py`**:
```python
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(conv_bn_relu(in_ch, out_ch), conv_bn_relu(out_ch, out_ch))
        self.pool = nn.MaxPool2d(2)

        # Thêm residual projection nếu channels thay đổi
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        x = self.block(x) + identity  # Residual connection
        x_down = self.pool(x)
        return x, x_down
```

### Làm thế nào thêm Positional Encoding vào Transformer

**Chỉnh sửa `masked_transformer.py`**:
```python
class AdaptiveMaskedTransformer(nn.Module):
    def __init__(self, in_ch, dim, patch_size=8, depth=2, n_heads=4):
        super().__init__()
        self.pe = PatchEmbed(in_ch, dim, patch_size)
        self.mask_gen = SoftMaskGenerator(dim, hidden=dim//2, n_heads=n_heads)
        self.blocks = nn.ModuleList([MaskedTransformerBlock(dim, n_heads) for _ in range(depth)])

        # Thêm learnable positional encoding
        # Số patches tối đa (với input 16×16 với patch=8 → 2×2=4 patches)
        max_patches = (16 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        tokens, (H,W) = self.pe(x)

        # Thêm positional encoding
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]

        softmask = self.mask_gen(tokens)
        for blk in self.blocks:
            tokens = blk(tokens, softmask)
        feat = tokens.transpose(1,2).reshape(x.size(0), tokens.size(-1), H, W)
        return feat
```

### Làm thế nào thay đổi Number of Classes

**Cho binary → multi-class classification** (ví dụ: HGG/LGG/Normal → 3 classes):

```python
model = BrainTumNet(
    in_ch=4,
    num_cls=3,  # Thay đổi từ 2 sang 3
    base=32,
    dim=256,
    patch=8,
    depth=2,
    n_heads=4,
)
```

**Cập nhật loss function trong config**:
```yaml
train:
  cls_criterion: "CrossEntropyLoss"  # Hỗ trợ multi-class
```

---

**Tiếp theo**: [[v_04_TRAINING_SYSTEM|Phần 4: Hệ thống Training →]]
