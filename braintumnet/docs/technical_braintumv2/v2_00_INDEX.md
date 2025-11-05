# BrainTumNet V2 - Phase 2 Technical Documentation

> **Tài Liệu Kỹ Thuật Chi Tiết Về Model SegUNetV2 Phase 2**
>
> **Phiên bản**: Phase 2 (SegUNetV2)
> **Ngày cập nhật**: 2025-01-14
> **Mục đích**: Giải thích chi tiết về model mới SegUNetV2 và các cải tiến Phase 2

---

## Tổng Quan

Đây là tài liệu kỹ thuật chuyên biệt về **SegUNetV2** - phiên bản model mới với nhiều cải tiến so với baseline V1. Tài liệu này tập trung vào:

- Model architecture mới (SegUNetV2)
- Các tính năng Phase 2 (multi-scale transformer, attention gates, boundary refinement)
- Training configuration tối ưu cho Phase 2
- So sánh chi tiết V1 vs V2
- Lý do thiết kế và quyết định kỹ thuật

---

## Cấu Trúc Tài Liệu

Tài liệu được chia thành 6 phần chính:

### 1. [Index - Trang Này](v2_00_INDEX.md)
**Nội dung**: Chỉ mục và hướng dẫn đọc tài liệu

### 2. [Phase 2 Overview](v2_01_PHASE2_OVERVIEW.md)
**Nội dung**:
- Tổng quan về Phase 2 là gì
- Timeline phát triển từ V1 đến Phase 2
- Mục tiêu và kết quả mong đợi
- Model configurations (Small, Large)
- Hardware requirements

### 3. [SegUNetV2 Architecture](v2_02_SEGUNETV2_ARCHITECTURE.md)
**Nội dung**:
- Kiến trúc SegUNetV2 chi tiết
- Enhanced Conv Blocks (InstanceNorm, LeakyReLU)
- Residual Convolutional Blocks
- Enhanced Encoder/Decoder
- Multi-Scale Fusion Module
- Deep Supervision
- Forward pass example với tensor shapes

### 4. [Phase 2 Features](v2_03_PHASE2_FEATURES.md)
**Nội dung**:
- Multi-Scale Transformer Bottleneck
- Attention Gates for skip connections
- Boundary Refinement Module
- Cách các features này hoạt động
- Khi nào nên enable/disable các features

### 5. [Training Configuration](v2_04_TRAINING_CONFIG.md)
**Nội dung**:
- phase2_small.yaml chi tiết
- phase2_a100.yaml chi tiết
- Loss configuration cho multi-class
- Optimizer và scheduler settings
- Augmentation strategy
- Hardware-specific optimizations

### 6. [Upgrade Reasoning - Tại Sao Thay Đổi](v2_06_UPGRADE_REASONING.md)
**Nội dung**:
- So sánh chi tiết V1 vs V2
- Lý do cho mỗi thay đổi
- Vấn đề V1 gặp phải và cách V2 giải quyết
- Trade-offs (memory, speed, accuracy)
- Khi nào nên dùng V1 vs V2

---

## Lộ Trình Đọc

### Cho Người Mới
1. **Bắt đầu với Phase 2 Overview** (v2_01) - hiểu tổng quan
2. **Đọc SegUNetV2 Architecture** (v2_02) - hiểu kiến trúc mới
3. **Xem Upgrade Reasoning** (v2_06) - hiểu tại sao thay đổi
4. **Đọc Training Config** (v2_04) - biết cách train

### Cho Lập Trình Viên Đã Biết V1
1. **Đọc Upgrade Reasoning** (v2_06) - xem điểm khác biệt
2. **Đọc Phase 2 Features** (v2_03) - tính năng mới
3. **Xem Training Config** (v2_04) - config thay đổi gì
4. **Tham khảo Architecture** (v2_02) - khi cần implementation details

### Cho Researcher
1. **Upgrade Reasoning** (v2_06) - motivation và ablation
2. **Phase 2 Features** (v2_03) - novel contributions
3. **Architecture** (v2_02) - technical details
4. **Training Config** (v2_04) - experimental setup

---

## File Code Liên Quan

### Model Implementation
```
src/braintumnet/models/
├── seg_unet_v2.py              [478 dòng] - SegUNetV2 main implementation
├── braintumnet_v2.py           [170 dòng] - BrainTumNetV2 wrapper
├── multiscale_transformer.py   [243 dòng] - Multi-scale transformer bottleneck
├── cbam.py                     [33 dòng]  - CBAM attention (reused from V1)
└── masked_transformer.py       [88 dòng]  - Adaptive transformer (reused from V1)
```

### Configuration Files
```
configs/
├── phase2_small.yaml           - Phase 2 Small (37M params, RTX 3090)
├── phase2_a100.yaml            - Phase 2 Large (87M params, A100)
├── models/segunetv2.yaml       - Base SegUNetV2 config
├── models/segunetv2_phase2.yaml - Phase 2 specific config
└── models/segunetv2_p1.yaml    - Phase 1 config (baseline improvements)
```

---

## Điểm Nổi Bật Phase 2

### 7 Cải Tiến Chính

1. **InstanceNorm thay BatchNorm**
   - Tốt hơn cho medical imaging
   - Không phụ thuộc batch size
   - Training == Inference (no running stats)

2. **LeakyReLU thay ReLU**
   - Gradient luôn flow (slope=0.01)
   - Không có dying ReLU problem
   - nnU-Net style activation

3. **Residual Connections**
   - Train được mạng sâu hơn
   - Gradient flow tốt hơn
   - Hội tụ nhanh hơn

4. **Strided Conv thay MaxPool**
   - Learned downsampling
   - Không mất information
   - Flexible và adaptive

5. **Multi-Scale Fusion**
   - Kết hợp features từ tất cả decoder levels
   - Multi-resolution information
   - Tốt hơn cho boundaries

6. **Deep Supervision**
   - Auxiliary losses ở intermediate layers
   - Gradient flow trực tiếp đến early layers
   - Training ổn định hơn

7. **Dropout Regularization**
   - Prevent overfitting cho model lớn
   - Dropout2d (drop whole feature maps)
   - Adaptive dropout (high in deep layers)

### Phase 2 Features (Optional)

8. **Multi-Scale Transformer Bottleneck**
   - Multiple patch sizes (4, 8, 16)
   - Better global context
   - Expected: +1.5-2.5% Dice

9. **Attention Gates**
   - nnU-Net style attention
   - Suppress irrelevant skip connections
   - Expected: +1-2% Dice

10. **Boundary Refinement**
    - Edge detection + boundary attention
    - Reduces IoU-Dice gap
    - Expected: +2-3% Dice

---

## Model Configurations

### Baseline V1 (Reference)
```yaml
Parameters: 14M
base: 32
dim: 256
depth: 2
n_heads: 4
dropout: 0.0
deep_supervision: false
multi_scale_fusion: false
```
**Results**: Dice 0.9148 (binary), IoU 0.7263 (multi-class)

### Phase 2 Small (Recommended)
```yaml
Parameters: 37M (2.6x V1)
base: 48
dim: 384
depth: 4
n_heads: 8
dropout: 0.15
deep_supervision: true
multi_scale_fusion: true
```
**Target**: IoU 0.80-0.85 (+5-6% from Phase 1)
**Hardware**: RTX 3090 24GB, batch_size=8

### Phase 2 Large (Best Performance)
```yaml
Parameters: 87M (6.2x V1)
base: 64
dim: 512
depth: 4
n_heads: 8
dropout: 0.2
deep_supervision: true
multi_scale_fusion: true
```
**Target**: IoU 0.85-0.90 (+10-15% from Phase 1)
**Hardware**: A100 80GB, batch_size=16

---

## Kết Quả Mong Đợi

### Binary Segmentation (Whole Tumor)

| Model | Dice | IoU | HD95 |
|-------|------|-----|------|
| V1 Baseline | 0.9148 | 0.8430 | 2.73mm |
| Phase 2 Small | 0.92-0.93 | 0.85-0.87 | 2.2-2.5mm |
| Phase 2 Large | 0.93-0.94 | 0.87-0.89 | 2.0-2.3mm |

### Multi-Class Segmentation (3 classes)

**V1** (không tốt cho multi-class):
- WT: 0.04 ❌
- TC: 0.81 ✓
- ED: 0.009 ❌

**Phase 2 Small** (expected):
- WT Dice: 0.83-0.86
- TC Dice: 0.80-0.83
- ED Dice: 0.82-0.85

**Phase 2 Large** (expected):
- WT Dice: 0.85-0.88
- TC Dice: 0.83-0.86
- ED Dice: 0.84-0.87

---

## Training Time

### Per Epoch (BraTS2020, ~22k slices)

| Model | GPU | Batch Size | Time/Epoch | Total (350 epochs) |
|-------|-----|------------|------------|-------------------|
| V1 | RTX 3090 | 16 | ~2.5s | ~15 minutes |
| Phase 2 Small | RTX 3090 | 8 | ~3.5s | ~20 minutes |
| Phase 2 Large | A100 80GB | 16 | ~5.0s | ~30 minutes |

**Note**: Phase 2 models ~40-100% chậm hơn nhưng accuracy cao hơn đáng kể!

---

## Sử Dụng Tài Liệu

### Conventions

- **Code blocks**: Là actual code có thể chạy
- **Pseudo code**: Được đánh dấu rõ ràng
- **Tensor shapes**: Luôn được chú thích (B, C, H, W)
- **Comments**: Giải thích tại sao, không chỉ là gì

### Symbols

- ✅ = Đã implement và test
- ⭐ = Feature mới trong Phase 2
- ❌ = Không khuyến nghị
- 🔧 = Có thể customize
- 💡 = Tip quan trọng

---

## Liên Kết Với Tài Liệu Cũ

Tài liệu này là **bổ sung**, không thay thế tài liệu technical cũ:

- **Technical docs** (docs/technical/): Giải thích toàn bộ project
- **Technical_braintumv2** (folder này): Chuyên về SegUNetV2 và Phase 2

**Đọc cả hai để hiểu đầy đủ!**

Tham khảo:
- [v_01_PROJECT_OVERVIEW](../technical/v_01_PROJECT_OVERVIEW.md) - Project overview
- [v_03_MODEL_ARCHITECTURE](../technical/v_03_MODEL_ARCHITECTURE.md) - V1 architecture
- [v_03a_SEGUNETV2_ARCHITECTURE](../technical/v_03a_SEGUNETV2_ARCHITECTURE.md) - V2 architecture overview

---

## Hỗ Trợ

Nếu có câu hỏi:
1. Đọc lại [Upgrade Reasoning](v2_06_UPGRADE_REASONING.md) - giải thích tại sao
2. Xem code comments trong implementation files
3. Kiểm tra config files cho examples

---

## Bắt Đầu

**Recommended**: Đọc theo thứ tự v2_01 → v2_02 → v2_06 → v2_04

**Next**: [Phase 2 Overview →](v2_01_PHASE2_OVERVIEW.md)

---

*Tài liệu này tập trung vào model mới và các cải tiến. Để hiểu toàn bộ project, đọc cả technical docs gốc.*
