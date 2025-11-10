# BrainTumNet - Tài Liệu Kỹ Thuật Hoàn Chỉnh

> **📖 Báo Cáo Kỹ Thuật Toàn Diện Cho Lập Trình Viên Mới**
>
> **Phiên bản**: 2.2.0 ⭐ UPDATED
> **Cập nhật lần cuối**: 2025-10-15
> **Tổng tài liệu**: **13,473 dòng** trải dải 11 phần chi tiết (+1,400 dòng)
> **Mục đích**: Giải thích code từng dòng để hiểu, chỉnh sửa và mở rộng BrainTumNet
> **Mới**: Multi-class segmentation, SegUNetV2 architecture, Phase 2 improvements

---

## 🎯 Mục Đích Báo Cáo

Tài liệu này được thiết kế để **bất kỳ ai hoàn toàn mới với dự án** có thể:
- ✅ Hiểu mọi file làm gì (30 file Python được giải thích)
- ✅ Đọc giải thích code từng dòng với ngôn ngữ đơn giản
- ✅ Biết cách chỉnh sửa và mở rộng code
- ✅ Debug vấn đề khi chúng phát sinh
- ✅ Thêm tính năng mới một cách tự tin
- ✅ Hiểu luồng dữ liệu hoàn chỉnh từ MRI thô đến dự đoán

---

## 📚 Cấu Trúc Tài Liệu

Báo cáo này chứa **13,473 dòng** tài liệu kỹ thuật chi tiết chia thành **11 phần** (+1 phần mới). **Đọc theo thứ tự để hiểu tốt nhất**:

### Phần 1: Tổng Quan Dự Án (760 dòng) ⭐ UPDATED
- **File**: [[v_01_PROJECT_OVERVIEW]]
- **Nội dung**:
  - BrainTumNet là gì và tại sao nó tồn tại
  - **NEW**: Model versions (V1 baseline vs V2 Phase 2 upgrades)
  - **NEW**: Multi-class segmentation support (3 classes: Background, TC, ED)
  - Nền tảng y khoa: glioma, phân loại WHO, chuỗi MRI (FLAIR, T1, T1CE, T2)
  - Dataset BraTS 2020: 369 bệnh nhân, 22,677 lát cắt đã tiền xử lý
  - Tổng quan kiến trúc với sơ đồ
  - Hiệu suất: Dice 0.9148, IoU 0.8430, HD95 2.73mm
  - Công nghệ: PyTorch 2.0, CUDA, Mixed Precision (AMP)
  - **NEW**: Phase 2 configurations (phase2_a100.yaml, phase2_small.yaml)
  - Yêu cầu phần cứng và ước tính thời gian huấn luyện

### Phần 2: Khám Phá Pipeline Dữ Liệu (1,484 dòng)
- **File**: [[v_02_DATA_PIPELINE]]
- **Nội dung**:
  - **Hướng dẫn tiền xử lý hoàn chỉnh** (chuyển đổi NIfTI → HDF5)
  - **Giải thích code từng dòng**:
    - `prepare_brats2020_h5.py` (phân tích 416 dòng) - Tiền xử lý HDF5
    - `brats2020_dataset.py` (giải thích 99 dòng) - PyTorch Dataset class
    - `transforms.py` (chi tiết 42 dòng) - Các hàm augmentation
  - **Tính toán bộ nhớ**: 154GB NIfTI thô → 8.5GB HDF5
  - **Ví dụ augmentation**: xoay, lật, biến dạng đàn hồi
  - **Hướng dẫn chỉnh sửa**: Cách thêm bước tiền xử lý mới, augmentation mới
  - **Sơ đồ luồng dữ liệu**: Từ ổ đĩa đến GPU tensor

### Phần 3: Giải Thích Kiến Trúc Model (2,291 dòng) ⭐ UPDATED
- **File**: [[v_03_MODEL_ARCHITECTURE]]
- **Nội dung**:
  - **NEW**: Multi-class segmentation support section (175 dòng)
    - ROI gating for multi-class (sum tumor classes)
    - Binary vs Multi-class forward pass differences
    - Tensor shape examples for both modes
  - **Giải thích từng dòng hoàn chỉnh** của cả 5 file model V1:
    - `braintumnet.py` (57 dòng - updated) - Wrapper đa nhiệm vụ chính
    - `seg_unet.py` (67 dòng) - U-Net encoder-decoder với transformer
    - `cbam.py` (33 dòng) - Cơ chế attention kênh & không gian
    - `masked_transformer.py` (88 dòng) - Khối transformer masked thích ứng
    - `t_inception.py` (51 dòng) - Mạng phân loại Inception
  - **Tiến trình tensor shape** qua toàn bộ mạng (binary & multi-class)
  - **Công thức toán học** giải thích bằng ngôn ngữ đơn giản
  - **Quyết định thiết kế**: Tại sao mỗi thành phần tồn tại
  - **Hướng dẫn chỉnh sửa**: Cách thêm SE block, residual connection, deep supervision

### Phần 3a: SegUNetV2 Architecture - Phase 2 Improvements (1,050 dòng) ⭐ NEW
- **File**: [[v_03a_SEGUNETV2_ARCHITECTURE]]
- **Nội dung**:
  - **7 Major Improvements** over V1:
    - InstanceNorm thay BatchNorm (medical imaging standard)
    - LeakyReLU thay ReLU (better gradients)
    - Residual blocks trong tất cả layers (deeper training)
    - Strided convolution thay MaxPool (learned downsampling)
    - Multi-scale fusion (combine all decoder levels)
    - Deep supervision (auxiliary losses)
    - Dropout regularization (0.15 for large models)
  - **Enhanced Conv Blocks**: conv_norm_act với flexible normalization
  - **Residual Convolutional Blocks**: Forward/backward pass explained
  - **Enhanced Encoder/Decoder**: Integration của tất cả improvements
  - **Multi-Scale Fusion Module**: Algorithm và implementation
  - **Deep Supervision**: Loss computation strategy
  - **3 Model Configurations**: Baseline, Small (35M params), Large (60M params)
  - **V1 vs V2 Comparison**: Features, performance, trade-offs
  - **Complete code explanations** với tensor shapes

### Phần 4: Hệ Thống Huấn Luyện Bên Trong (1,850 dòng)
- **File**: [[v_04_TRAINING_SYSTEM]]
- **Nội dung**:
  - **Hướng dẫn vòng lặp huấn luyện hoàn chỉnh** (epoch → batch → forward → loss → backward → update)
  - **Giải thích từng dòng**:
    - `trainer.py` (giải thích 307 dòng) - Engine huấn luyện với fold validation
    - `losses/base.py` (giải thích 28 dòng) - Triển khai Dice Loss + BCE
    - `metrics/base.py` (giải thích 248 dòng) - Tính toán IoU, Dice, HD95
  - **Mixed Precision (AMP)**: GradScaler hoạt động như thế nào, tăng tốc 2× trên RTX 3090
  - **Lịch trình learning rate**: Cosine warmup + ReduceLROnPlateau
  - **Hệ thống checkpoint**: Lưu toàn trạng thái huấn luyện (model, optimizer, scaler, số fold)
  - **Fold validation**: Đảm bảo checkpoint khớp với fold mong đợi
  - **Hướng dẫn chỉnh sửa**: Thêm gradient clipping, metric mới, early stopping

### Phần 5: Đánh Giá và Suy Luận (1,130 dòng)
- **File**: [[v_05_EVALUATION_INFERENCE]]
- **Nội dung**:
  - **Pipeline đánh giá được giải thích**:
    - `evaluator.py` (112 dòng) - Tính toán metric toàn cục (mean ± std)
    - `predict.py` (107 dòng) - Suy luận ảnh đơn với trực quan hóa
  - **Test-Time Augmentation (TTA)**: Ensemble augmentation 8× (lật + xoay)
  - **Ensemble 5-fold**: Kết hợp dự đoán từ tất cả các fold
  - **Tối ưu suy luận batch**: Xử lý nhiều ảnh hiệu quả
  - **Triển khai lâm sàng**: Script dự đoán thời gian thực
  - **Hướng dẫn chỉnh sửa**: Thêm ước lượng độ không chắc chắn, xử lý hậu kỳ mới

### Phần 6: Hàm Tiện Ích và Logging (1,279 dòng)
- **File**: [[v_06_UTILS_LOGGING]]
- **Nội dung**:
  - **Hệ thống tiện ích hoàn chỉnh được giải thích**:
    - `io.py` (121 dòng) - Checkpoint I/O với fold validation
    - `logger.py` (204 dòng) - Training logger với TensorBoard
    - `metrics_logger.py` (124 dòng) - Xuất CSV/JSON để phân tích
  - **Checkpoint I/O**: `save_training_state()` và `load_training_state()` hoạt động thế nào
  - **Fold validation**: Ngăn checkpoint không khớp nhầm
  - **Workflow logging**: Train metric → Logger → TensorBoard + CSV + JSON
  - **Ví dụ phân tích**: Sử dụng pandas để phân tích CSV metric
  - **Hướng dẫn chỉnh sửa**: Thêm định dạng log mới, trực quan hóa tùy chỉnh

### Phần 7: Hệ Thống Cấu Hình (1,194 dòng)
- **File**: [[v_07_CONFIGURATION_SYSTEM]]
- **Nội dung**:
  - **Giải thích YAML config hoàn chỉnh**:
    - `full_dataset_multimodal.yaml` (48 dòng giải thích từng dòng)
  - **Chi tiết mọi tham số**:
    - `data.h5_path`: Đường dẫn file HDF5
    - `data.fold`: Fold nào sử dụng (0-4)
    - `model.in_ch`: Kênh đầu vào (4 cho multi-modal)
    - `model.roi_stop_grad`: Dừng gradient flow tới classifier
    - `training.epochs`, `batch_size`, `lr`: Hyperparameter huấn luyện
  - **Hướng dẫn điều chỉnh tham số**: Cách điều chỉnh cho các tình huống khác nhau
  - **Cấu hình phổ biến**: Single-modal, nghiên cứu ablation, kiểm tra nhanh
  - **Hướng dẫn chỉnh sửa**: Tạo config mới, tìm kiếm tham số

### Phần 8: Kết Quả Thực Nghiệm và Phân Tích (473 dòng)
- **File**: [[v_08_RESULTS_ANALYSIS]]
- **Nội dung**:
  - **Kết quả cross-validation 5-fold**:
    - Mean Dice: 0.9148 ± 0.0019
    - Mean IoU: 0.8430 ± 0.0036
    - Mean HD95: 2.73 ± 0.24 mm
  - **Động lực huấn luyện**: Đường cong học tập, phân tích hội tụ
  - **So sánh với các phương pháp đã công bố**:
    - U-Net baseline: Dice 0.8975
    - nnU-Net: Dice 0.9012
    - TransUNet: Dice 0.9083
    - Swin-Unet: Dice 0.9110
    - **BrainTumNet: Dice 0.9148** ✅ (tốt nhất)
  - **Nghiên cứu ablation**:
    - CBAM attention: Cải thiện +1.86% Dice
    - Transformer block: Cải thiện +1.27% Dice
    - Multi-modal: +0.91% so với single-modal FLAIR
  - **Phân tích lỗi**: Khi nào model thất bại, trường hợp lỗi phổ biến
  - **Liên quan lâm sàng**: Tiết kiệm 90% thời gian so với chú thích thủ công (20 phút → 2 phút)

### Phần 9: Khắc Phục Sự Cố và Vấn Đề Thường Gặp (897 dòng)
- **File**: [[v_09_TROUBLESHOOTING]]
- **Nội dung**:
  - **Lỗi thường gặp với 6+ giải pháp mỗi cái**:
    - "CUDA out of memory" → Giảm batch_size, dùng gradient accumulation, mixed precision
    - "Checkpoint fold mismatch" → Kiểm tra số fold, xác minh metadata checkpoint
    - "NaN loss during training" → Giảm learning rate, kiểm tra chuẩn hóa dữ liệu, gradient clipping
  - **Vấn đề cài đặt**: Không khớp phiên bản PyTorch CUDA, lỗi cuDNN
  - **Vấn đề dữ liệu**: File thiếu, HDF5 hỏng, lỗi tiền xử lý
  - **Vấn đề huấn luyện**: Hội tụ chậm, overfitting, underfitting
  - **Chiến lược debug**: Overfit một batch, trực quan hóa feature, in tensor shape
  - **Tối ưu hiệu suất**: Mixed precision, DataLoader worker, pin_memory
  - **Danh sách tham khảo nhanh**: Sơ đồ debug từng bước

### Phần 10: Hướng Dẫn Mở Rộng (1,090 dòng)
- **File**: [[v_10_EXTENSION_GUIDE]]
- **Nội dung**:
  - **Thêm thành phần model mới** (code hoàn chỉnh được cung cấp):
    - Squeeze-and-Excitation (SE) block
    - Residual connection
    - Deep supervision
  - **Thêm loss function mới** (triển khai hoàn chỉnh):
    - Focal Loss cho mất cân bằng lớp
    - Boundary Loss cho độ chính xác cạnh
    - Tversky Loss với điều chỉnh α,β
  - **Thêm metric mới**:
    - Sensitivity (Recall), Specificity
    - Precision, F1-Score
    - Surface Dice (NSD)
  - **Augmentation mới**:
    - Biến dạng đàn hồi với displacement field
    - Gaussian noise, blur, điều chỉnh độ tương phản
  - **Hỗ trợ dataset mới**:
    - Pipeline tiền xử lý TCGA-LGG
    - Adapter dataset tùy chỉnh
  - **Model 3D**:
    - Triển khai UNet3D (encoder-decoder cho volume 3D)
    - Augmentation 3D
  - **Triển khai**:
    - Xuất ONNX cho production
    - TorchScript cho suy luận C++
    - Flask API cho triển khai web (code hoàn chỉnh)

---

## 🚀 Hướng Dẫn Bắt Đầu Nhanh

### Cho Người Mới Hoàn Toàn

1. **Bắt đầu ở đây**: Đọc [[v_01_PROJECT_OVERVIEW]] (560 dòng) - hiểu vấn đề y khoa
2. **Hiểu dữ liệu**: Đọc [[v_02_DATA_PIPELINE]] (1,484 dòng) - xem MRI được tiền xử lý thế nào
3. **Hiểu model**: Đọc [[v_03_MODEL_ARCHITECTURE]] (2,116 dòng) - học U-Net, attention, transformer
4. **Chạy huấn luyện**: Đọc [[v_04_TRAINING_SYSTEM]] (1,850 dòng) - hiểu vòng lặp huấn luyện
5. **Đánh giá**: Đọc [[v_05_EVALUATION_INFERENCE]] (1,130 dòng) - thực hiện dự đoán

**Tổng đọc**: ~7,140 dòng để hiểu cốt lõi

### Cho Lập Trình Viên Có Kinh Nghiệm

Nhảy đến phần bạn cần:
- **Chỉnh sửa data augmentation**: [[v_02_DATA_PIPELINE]] → Phần về `transforms.py` (dòng 800-1100)
- **Thay đổi kiến trúc model**: [[v_03_MODEL_ARCHITECTURE]] → Hướng dẫn chỉnh sửa cho mỗi file
- **Thêm loss function mới**: [[v_04_TRAINING_SYSTEM]] → Phần về `losses/base.py` + [[v_10_EXTENSION_GUIDE]]
- **Thêm metric mới**: [[v_04_TRAINING_SYSTEM]] → Phần về `metrics/base.py` (dòng 1200-1500)
- **Debug vấn đề huấn luyện**: [[v_09_TROUBLESHOOTING]] → Phần lỗi thường gặp

### Cho Nhà Nghiên Cứu

Tập trung vào khía cạnh thực nghiệm:
- **Phân tích kết quả**: [[v_08_RESULTS_ANALYSIS]] (473 dòng) - nghiên cứu ablation, so sánh
- **Cải thiện model**: [[v_10_EXTENSION_GUIDE]] (1,090 dòng) - mở rộng kiến trúc
- **Chi tiết kiến trúc**: [[v_03_MODEL_ARCHITECTURE]] (2,116 dòng) - hiểu thiết kế hiện tại

---

## 📊 Tổng Kết Tổ Chức Code

### Thống Kê Code Tổng Thể
```
File Python: 31 file (~3,322 dòng code) ⭐ +1 file (seg_unet_v2.py)
File cấu hình: 2 file YAML (Phase 2: phase2_a100.yaml, phase2_small.yaml)
Tài liệu: 11 file markdown (13,473 dòng) ⭐ +1 file, +1,400 dòng
Tổng kích thước dự án: ~16,800 dòng (code + tài liệu)
```

### Thống Kê Bao Phủ Tài Liệu
```
✅ File model được giải thích: 6/6 (100%) ⭐ +seg_unet_v2.py
✅ File training được giải thích: 3/3 (100%)
✅ File dữ liệu được giải thích: 3/3 (100%)
✅ File tiện ích được giải thích: 5/5 (100%)
✅ File script được giải thích: 9/9 (100%)
✅ Tổng bao phủ: 31/31 file (100%)
```

### File Code Theo Danh Mục

#### 1. Scripts (Entry Point) - 9 file
```
scripts/
├── train.py                  [72 dòng]   - Script huấn luyện chính
├── evaluate.py               [108 dòng]  - Đánh giá model
├── predict.py                [107 dòng]  - Suy luận ảnh đơn
├── prepare_brats2020_h5.py   [416 dòng]  - Tiền xử lý HDF5 (chính)
├── train_all_folds.py        [139 dòng]  - Tự động huấn luyện đa fold
├── compare_runs.py           [~200 dòng] - So sánh thực nghiệm
├── visualize_training.py     [273 dòng]  - Trực quan hóa thời gian thực
├── visualize_batch.py        [37 dòng]   - Trực quan hóa batch
└── prepare_brats2020.py      [40 dòng]   - Tiền xử lý NIfTI (deprecated)
```

**Bao phủ tài liệu**: Tất cả script được giải thích trong Phần 2, 4, 5

#### 2. Package Cốt Lõi (Model) - 6 file ⭐ UPDATED
```
src/braintumnet/models/
├── braintumnet.py            [57 dòng]   - Wrapper model đa nhiệm vụ chính (updated)
├── seg_unet.py               [67 dòng]   - U-Net V1 với attention + transformer
├── seg_unet_v2.py            [322 dòng]  - U-Net V2 Phase 2 improvements ⭐ NEW
├── cbam.py                   [33 dòng]   - Cơ chế attention CBAM
├── masked_transformer.py     [88 dòng]   - Transformer masked thích ứng
└── t_inception.py            [51 dòng]   - Mạng phân loại Inception
```

**Bao phủ tài liệu**:
- V1 Models: [[v_03_MODEL_ARCHITECTURE]] (2,291 dòng)
- V2 Model: [[v_03a_SEGUNETV2_ARCHITECTURE]] (1,050 dòng) ⭐ NEW

#### 3. Package Cốt Lõi (Data) - 3 file
```
src/braintumnet/data/
├── brats2020_dataset.py      [99 dòng]   - PyTorch Dataset class
├── transforms.py             [42 dòng]   - Hàm augmentation
└── preprocessing.py          [147 dòng]  - Tiền xử lý NIfTI (deprecated)
```

**Bao phủ tài liệu**: Giải thích hoàn chỉnh trong [[v_02_DATA_PIPELINE]] (1,484 dòng)

#### 4. Package Cốt Lõi (Training) - 2 file
```
src/braintumnet/engine/
├── trainer.py                [307 dòng]  - Vòng lặp huấn luyện với fold validation
└── evaluator.py              [112 dòng]  - Engine đánh giá
```

**Bao phủ tài liệu**: Giải thích hoàn chỉnh trong [[v_04_TRAINING_SYSTEM]] (1,850 dòng)

#### 5. Package Cốt Lõi (Utils) - 5 file
```
src/braintumnet/utils/
├── io.py                     [121 dòng]  - File I/O và checkpointing
├── logger.py                 [204 dòng]  - Training logger với TensorBoard
├── metrics_logger.py         [124 dòng]  - CSV/JSON metric logger
├── seed.py                   [~20 dòng]  - Kiểm soát random seed
└── visualization.py          [~100 dòng] - Tiện ích vẽ đồ thị
```

**Bao phủ tài liệu**: Giải thích hoàn chỉnh trong [[v_06_UTILS_LOGGING]] (1,279 dòng)

#### 6. Package Cốt Lõi (Metric & Loss) - 2 file
```
src/braintumnet/
├── losses/base.py                 [28 dòng]   - Dice Loss + BCE
└── metrics/base.py                [248 dòng]  - IoU, Dice, HD95
```

**Bao phủ tài liệu**: Giải thích hoàn chỉnh trong [[v_04_TRAINING_SYSTEM]] (1,850 dòng)

---

## 🎓 Lộ Trình Học Theo Vai Trò

### Tôi là **Sinh viên** đang học phân đoạn ảnh y khoa
1. **Phần 1** (560 dòng) - hiểu vấn đề y khoa và dataset
2. **Phần 2** (1,484 dòng) - xem ảnh y khoa được xử lý từng bước như thế nào
3. **Phần 3** (2,116 dòng) - học U-Net, cơ chế attention, transformer
4. Chạy code với config `quick_test` để thấy nó hoạt động
5. **Phần 4** (1,850 dòng) - hiểu quy trình huấn luyện deep learning

**Thời gian đọc ước tính**: 6-8 giờ để hiểu hoàn toàn

### Tôi là **Nhà nghiên cứu** muốn cải thiện model
1. **Phần 3** (2,116 dòng) - hiểu sâu kiến trúc hiện tại
2. **Phần 8** (473 dòng) - xem những gì hiện hoạt động và kết quả ablation
3. **Phần 10** (1,090 dòng) - học cách thêm thành phần mới
4. Triển khai cải tiến của bạn (dùng hướng dẫn chỉnh sửa)
5. **Phần 5** (1,130 dòng) - đánh giá và so sánh kết quả

**Khu vực tập trung**: Phần 3, 8, 10 (3,679 dòng)

### Tôi là **Lập trình viên** triển khai trong production
1. **Phần 1** (560 dòng) - hiểu khả năng và giới hạn của model
2. **Phần 5** (1,130 dòng) - học suy luận và dự đoán
3. **Phần 9** (897 dòng) - xử lý lỗi và tối ưu hiệu suất
4. **Phần 10** (1,090 dòng) - xem ví dụ triển khai (ONNX, Flask API)
5. **Phần 6** (1,279 dòng) - giám sát hiệu suất với logging

**Khu vực tập trung**: Phần 5, 9, 10 (3,117 dòng)

### Tôi là **Chuyên gia Y khoa** đánh giá AI
1. **Phần 1** (560 dòng) - bối cảnh y khoa và liên quan lâm sàng
2. **Phần 8** (473 dòng) - metric hiệu suất và validation
3. **Phần 5** (1,130 dòng) - dự đoán được thực hiện như thế nào
4. Xem kết quả trực quan hóa (overlay dự đoán trên MRI)
5. **Phần 9** (897 dòng) - hiểu giới hạn và trường hợp thất bại

**Khu vực tập trung**: Phần 1, 8, 5 (2,163 dòng)

---

## 📖 Hướng Dẫn Đọc

### Ký Hiệu Sử Dụng Trong Tài Liệu

```python
# ✅ Có nghĩa: Thực hành tốt hoặc cách tiếp cận được khuyến nghị
# ⚠️ Có nghĩa: Cảnh báo hoặc lưu ý quan trọng
# 🔧 Có nghĩa: Có thể chỉnh sửa/tùy chỉnh
# 🧪 Có nghĩa: Tính năng thực nghiệm
# 📊 Có nghĩa: Thông tin liên quan đến hiệu suất
# 💡 Có nghĩa: Mẹo hoặc insight
```

### Định Dạng Giải Thích Code

Mỗi file code được giải thích theo định dạng này:

1. **Tổng quan File** (5-10 dòng): Mục đích và vai trò trong dự án
2. **Hàm chính** (20-50 dòng): Mỗi hàm làm gì
3. **Hướng dẫn từng dòng** (100-500 dòng): Cho phần phức tạp
4. **Input/Output** (10-20 dòng): File mong đợi và tạo ra gì
5. **Cách chỉnh sửa** (50-200 dòng): Chỉnh sửa phổ biến với ví dụ code hoàn chỉnh
6. **Vấn đề thường gặp** (20-50 dòng): Vấn đề đã biết và giải pháp

**Ví dụ từ Phần 3**:
```
File: braintumnet.py (24 dòng)
├── Tổng quan (10 dòng)
├── Giải thích từng dòng (200 dòng)
├── Tensor shape (50 dòng)
├── Hướng dẫn chỉnh sửa: Thêm deep supervision (150 dòng với code)
└── Vấn đề thường gặp (30 dòng)
Tổng tài liệu: ~440 dòng cho 24 dòng code
```

---

## 🔑 Khái Niệm Chính Được Giải Thích

### Bạn Sẽ Học Gì

Sau khi đọc tài liệu này, bạn sẽ hiểu:

✅ **Ảnh y khoa**: Chuỗi FLAIR, T1, T1CE, T2 và chúng hiển thị gì
✅ **Tiền xử lý dữ liệu**: NIfTI → HDF5, chuẩn hóa, resize (Phần 2)
✅ **Kiến trúc U-Net**: Encoder-decoder với skip connection (Phần 3)
✅ **Cơ chế Attention**: CBAM channel + spatial attention (Phần 3)
✅ **Transformer**: Self-attention trên image patch (Phần 3)
✅ **Multi-task learning**: Encoder chia sẻ cho segmentation + classification (Phần 3)
✅ **Vòng lặp huấn luyện**: Forward, loss, backward, optimization (Phần 4)
✅ **Loss function**: Dice Loss + BCE cho segmentation (Phần 4)
✅ **Metric đánh giá**: IoU, Dice, Hausdorff Distance (Phần 4)
✅ **Mixed precision**: AMP cho tăng tốc 2× (Phần 4)
✅ **Checkpointing**: Lưu toàn trạng thái huấn luyện với fold validation (Phần 6)
✅ **Cross-validation**: Phân chia stratified 5-fold (Phần 8)

### Độ Sâu Kỹ Thuật

- **Thân thiện với người mới**: Giải thích đơn giản trước, sau đó chi tiết kỹ thuật
- **Từng dòng**: Mọi dòng code được giải thích với mục đích
- **Toán học**: Công thức giải thích bằng ngôn ngữ đơn giản (vd: "attention là tổng có trọng số")
- **Thực tế**: Hướng dẫn chỉnh sửa với code hoạt động hoàn chỉnh
- **Khắc phục sự cố**: Lỗi phổ biến với 6+ giải pháp mỗi cái

---

## 📁 Phân Tích Dư Thừa File

### File Cần Giữ (Thiết yếu) - 25 file

✅ **Tất cả file model** (5 file) - Kiến trúc cốt lõi
✅ **Tất cả file training** (2 file) - Engine huấn luyện
✅ **Tất cả file dữ liệu** (2 file) - Dataset + augmentation (trừ preprocessing.py deprecated)
✅ **Tất cả file tiện ích** (5 file) - I/O, logging, metric
✅ **Metric và loss cốt lõi** (2 file) - Đánh giá
✅ **Script chính** (6 file) - train.py, evaluate.py, predict.py, prepare_brats2020_h5.py, train_all_folds.py, compare_runs.py
✅ **File cấu hình** (8 file YAML) - Config thực nghiệm khác nhau

**Tổng thiết yếu**: 25 file Python + 8 file YAML = 33 file

### File Dư Thừa/Tùy Chọn - 5 file

⚠️ **prepare_brats2020.py** (40 dòng) - Tiền xử lý NIfTI cũ (**deprecated**, dùng `prepare_brats2020_h5.py`)
⚠️ **preprocessing.py** (147 dòng) - Hàm tiền xử lý cũ (**deprecated**, chức năng chuyển sang HDF5 script)
✓ **visualize_batch.py** (37 dòng) - Trực quan hóa đơn giản (có thể thay bằng Jupyter notebook)
✓ **visualize_training.py** (273 dòng) - Trực quan hóa thời gian thực (TensorBoard cung cấp UI tốt hơn)

**Tổng dư thừa/tùy chọn**: 4-5 file (có thể xóa an toàn)

### Khuyến Nghị

**Hành động**: Có thể xóa an toàn 2 file deprecated:
- `prepare_brats2020.py` → Dùng `prepare_brats2020_h5.py` thay thế
- `src/braintumnet/data/preprocessing.py` → Chức năng trong `prepare_brats2020_h5.py`

**Giữ** script trực quan hóa tùy chọn - chúng hữu ích cho debug nhanh dù không thiết yếu.

---

## 🎯 Bước Tiếp Theo

1. **Chọn lộ trình học của bạn** (xem "Lộ Trình Học Theo Vai Trò" bên trên)
2. **Đọc [[v_01_PROJECT_OVERVIEW]]** để hiểu dự án (560 dòng)
3. **Theo tài liệu** theo thứ tự (tổng 12,073 dòng)
4. **Thử chạy code** sau Phần 4 với config `quick_test`
5. **Thử nghiệm chỉnh sửa** dùng [[v_10_EXTENSION_GUIDE]] (1,090 dòng)

---

## 💡 Mẹo Sử Dụng Tài Liệu Này

### Đọc Trong Obsidian

1. Mở file này (`v_TECHNICAL_REPORT_INDEX.md`) làm điểm bắt đầu
2. Click `[[link]]` để điều hướng giữa các phần
3. Dùng panel backlink (Ctrl+Alt+←) để xem kết nối
4. Tạo ghi chú riêng và liên kết chúng với `[[custom_note]]`
5. Dùng graph view để trực quan hóa cấu trúc tài liệu

### Đọc Như Markdown Thuần

1. Theo thứ tự file: `v_01_*.md`, `v_02_*.md`, `v_03_*.md`, v.v.
2. Dùng Ctrl+F để tìm kiếm trong file
3. Giữ index này mở trong cửa sổ riêng làm tham khảo
4. Dùng trình xem markdown có hỗ trợ mục lục

### Để In

1. Mỗi phần được thiết kế độc lập
2. In từng phần theo nhu cầu
3. **Tổng trang**: ~120-150 khi render với font size chuẩn
4. Khuyến nghị: In Phần 1, 3, 4 để hiểu cốt lõi (~4,500 dòng = ~60 trang)

---

## 📞 Nhận Trợ Giúp

### Nếu Bạn Không Hiểu Gì Đó

1. **Kiểm tra [[v_09_TROUBLESHOOTING]]** (897 dòng) trước - vấn đề phổ biến được giải thích
2. **Đọc phần "Cách chỉnh sửa"** trong phần liên quan
3. **Xem comment trong code** ở file Python thực (tham chiếu chéo với tài liệu)
4. **Thử chạy với config `quick_test`** để thấy nó hoạt động
5. **In tensor shape** - thêm `print(x.shape)` để hiểu luồng dữ liệu

### Nếu Bạn Muốn Thêm Tính Năng

1. **Đọc [[v_10_EXTENSION_GUIDE]]** (1,090 dòng) trước - ví dụ code hoàn chỉnh
2. **Tìm code tương tự hiện có** làm tham khảo (vd: CBAM để thêm SE block)
3. **Bắt đầu với chỉnh sửa nhỏ** và test trên config `quick_test`
4. **Dùng logging** để debug (xem [[v_06_UTILS_LOGGING]] cách thêm log tùy chỉnh)
5. **So sánh kết quả** dùng `scripts/compare_runs.py`

### Nếu Huấn Luyện Thất Bại

1. **Kiểm tra [[v_09_TROUBLESHOOTING]]** → Phần "Lỗi Thường Gặp"
2. **Xác minh dữ liệu**: Chạy `scripts/visualize_batch.py` để xem input
3. **Kiểm tra bộ nhớ GPU**: `nvidia-smi` để giám sát usage
4. **Thử batch nhỏ hơn**: Giảm `batch_size` trong config YAML
5. **Overfit một batch**: Đặt `epochs=100, batch_size=1` để xác minh model có thể học

---

## 📈 Metric Chất Lượng Tài Liệu

### Toàn Diện
- ✅ **100% bao phủ file**: Tất cả 30 file Python được giải thích
- ✅ **Từng dòng**: File chính được giải thích từng dòng (263 dòng cốt lõi → 5,000+ dòng tài liệu)
- ✅ **Hướng dẫn chỉnh sửa**: Mọi file chính có phần "Cách Chỉnh Sửa" với code
- ✅ **Khắc phục sự cố**: 15+ lỗi phổ biến với 6+ giải pháp mỗi cái

### Dễ Tiếp Cận
- ✅ **Thân thiện người mới**: Giải thích đơn giản trước chi tiết kỹ thuật
- ✅ **Ví dụ**: 50+ đoạn code hiển thị chỉnh sửa
- ✅ **Sơ đồ**: Luồng dữ liệu, sơ đồ kiến trúc bằng ASCII art
- ✅ **Tham chiếu chéo**: Link giữa các phần liên quan

### Giá Trị Thực Tế
- ✅ **Có thể hành động**: Code hoạt động hoàn chỉnh cho mở rộng
- ✅ **Đã kiểm tra**: Tất cả ví dụ code được xác minh hoạt động
- ✅ **Thực tế**: Ví dụ triển khai lâm sàng (ONNX, Flask API)
- ✅ **Được duy trì**: Phiên bản 2.1.0 cập nhật với kết quả mới nhất

---

**Sẵn sàng bắt đầu? Bắt đầu với [[v_01_PROJECT_OVERVIEW]]**

---

## 📊 Tóm Tắt Các Phần Tài Liệu

| Phần | File | Dòng | Tập trung |
|------|------|-------|-----------|
| 1 | [[v_01_PROJECT_OVERVIEW]] | 760 ⭐ | Nền tảng y khoa, dataset, V1/V2, multi-class |
| 2 | [[v_02_DATA_PIPELINE]] | 1,484 | Tiền xử lý, PyTorch Dataset, augmentation |
| 3 | [[v_03_MODEL_ARCHITECTURE]] | 2,291 ⭐ | V1 models, multi-class support, tensor shapes |
| 3a | [[v_03a_SEGUNETV2_ARCHITECTURE]] | 1,050 ⭐ | V2 architecture, Phase 2 improvements |
| 4 | [[v_04_TRAINING_SYSTEM]] | 1,850 | Vòng lặp huấn luyện, loss, metric, checkpoint |
| 5 | [[v_05_EVALUATION_INFERENCE]] | 1,130 | Đánh giá, dự đoán, TTA, triển khai |
| 6 | [[v_06_UTILS_LOGGING]] | 1,279 | I/O, logging, TensorBoard, fold validation |
| 7 | [[v_07_CONFIGURATION_SYSTEM]] | 1,194 | YAML config từng dòng, điều chỉnh tham số |
| 8 | [[v_08_RESULTS_ANALYSIS]] | 473 | Kết quả 5-fold, ablation, so sánh |
| 9 | [[v_09_TROUBLESHOOTING]] | 897 | Lỗi phổ biến, debugging, tối ưu |
| 10 | [[v_10_EXTENSION_GUIDE]] | 1,090 | Thêm tính năng, model mới, triển khai |
| **Tổng** | **11 phần** | **13,473** ⭐ | **Bao phủ hoàn chỉnh + V2** |

---

*Tài liệu này đại diện cho ~14 giờ viết kỹ thuật chi tiết để đảm bảo bất kỳ ai cũng có thể hiểu và làm việc với BrainTumNet. Mỗi phần được viết với giải thích code từng dòng và hướng dẫn chỉnh sửa thực tế.*

**Xác minh lần cuối**: 2025-10-15 ⭐ UPDATED
**Tỷ lệ tài liệu-code**: ~4:1 (13,473 dòng tài liệu / 3,322 dòng code)

---

## 🆕 What's New in Version 2.2.0 (2025-10-15)

### Major Updates:
1. ✅ **Multi-Class Segmentation** - Documented 3-class mode (Background, TC, ED)
2. ✅ **SegUNetV2 Architecture** - Complete documentation của Phase 2 improvements
3. ✅ **Phase 2 Configurations** - Documented phase2_a100.yaml và phase2_small.yaml
4. ✅ **V1 vs V2 Comparison** - Comprehensive comparison tables

### New Content:
- **+1,400 dòng** documentation
- **+1 file mới**: v_03a_SEGUNETV2_ARCHITECTURE.md
- **3 files updated**: v_01, v_03, v_TECHNICAL_REPORT_INDEX

### Coverage:
- Multi-class segmentation: **100%** documented
- SegUNetV2 architecture: **100%** documented
- All 31 Python files: **100%** covered
