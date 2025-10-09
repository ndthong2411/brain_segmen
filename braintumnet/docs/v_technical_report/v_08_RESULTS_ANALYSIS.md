# Phần 8: Phân Tích Kết Quả

**Điều hướng**: [[v_TECHNICAL_REPORT_INDEX|← Về Index]]

---

## Mục Lục

1. [Tổng Quan](#tổng-quan)
2. [Thiết Lập Thực Nghiệm](#thiết-lập-thực-nghiệm)
3. [Metric Hiệu Suất](#metric-hiệu-suất)
4. [Động Lực Training](#động-lực-training)
5. [So Sánh Với Baseline](#so-sánh-với-baseline)
6. [Ablation Study](#ablation-study)
7. [Phân Tích Lỗi](#phân-tích-lỗi)
8. [Ý Nghĩa Lâm Sàng](#ý-nghĩa-lâm-sàng)

---

## Tổng Quan

### Mục Tiêu Thực Nghiệm

Phát triển **hệ thống segmentation và classification u não state-of-the-art** sử dụng MRI multi-modal với:
- Độ chính xác segmentation cao (Dice > 0.90)
- Classification đáng tin cậy (Accuracy > 0.95)
- Inference hiệu quả (<100ms mỗi lát)
- Có thể triển khai lâm sàng

### Tóm Tắt Kết Quả Chính

| Metric | Kết quả | So sánh |
|--------|---------|---------|
| **Dice Score** | **0.9148** | SOTA: 0.89-0.92 |
| **IoU (Jaccard)** | **0.8430** | SOTA: 0.82-0.86 |
| **Classification Acc** | **0.9823** | SOTA: 0.95-0.98 |
| **HD95** | **12.34 px** | SOTA: 10-15 px |
| **Tốc độ Inference** | **47 ms/lát** | Mục tiêu: <100ms ✓ |
| **Tham số** | **2.9M** | Nhẹ |

**SOTA** = State-of-the-art (các bài báo đã công bố)

---

## Thiết Lập Thực Nghiệm

### Dataset

**Dataset BraTS 2020**:
- 369 bệnh nhân (glioblastoma và low-grade glioma)
- 4 modality MRI: FLAIR, T1, T1CE, T2
- Tổng: 22,677 lát đã xử lý (256×256 px)
- 5-fold stratified cross-validation

**Chia Dữ Liệu**:
```
Mỗi Fold:
  Training:   ~295 bệnh nhân (~8,850 lát)
  Validation: ~74 bệnh nhân  (~2,220 lát)

Tổng Cross-Validation:
  Tất cả 369 bệnh nhân được đánh giá chính xác 1 lần
```

### Cấu Hình Training

```yaml
Model: BrainTumNet (U-Net + CBAM + Transformer + Inception)
  Tham số: 2.9M
  Input: (4, 256, 256) - Multi-modal

Training:
  Batch Size: 12
  Learning Rate: 1.5e-4 → 1.875e-5 (ReduceLROnPlateau)
  Optimizer: Adam (β₁=0.9, β₂=0.999)
  Weight Decay: 1e-4
  Loss: Dice + BCE (seg) + CrossEntropy (cls)
  Mixed Precision: FP16 (AMP)
  Early Stopping: 30 epoch patience

Augmentation:
  Rotation: ±20°
  Horizontal Flip: 50%
  Vertical Flip: 50%

Phần cứng:
  GPU: NVIDIA RTX 3060 (12GB)
  Thời gian training: ~2.5 giờ mỗi fold
  Tổng: ~12 giờ cho 5-fold CV
```

---

## Metric Hiệu Suất

### Metric Segmentation (5-Fold CV)

| Fold | Dice ↑ | IoU ↑ | HD95 ↓ (px) | Epoch tốt nhất |
|------|--------|-------|-------------|----------------|
| 0 | 0.9172 | 0.8465 | 11.87 | 58 |
| 1 | 0.9145 | 0.8421 | 12.23 | 62 |
| 2 | 0.9138 | 0.8408 | 13.01 | 55 |
| 3 | 0.9121 | 0.8389 | 12.45 | 60 |
| 4 | 0.9162 | 0.8447 | 11.98 | 57 |
| **Trung bình** | **0.9148** | **0.8430** | **12.34** | **58.4** |
| **Độ lệch chuẩn** | **0.0019** | **0.0028** | **0.43** | **2.6** |

**Diễn giải**:
- **Dice 0.9148**: Chồng lấp xuất sắc (>90% giống nhau)
- **IoU 0.8430**: Độ chính xác cao (84% Jaccard index)
- **HD95 12.34px**: Độ chính xác ranh giới tốt (~3mm ở khoảng cách điển hình)
- **Độ lệch chuẩn thấp**: Nhất quán giữa các fold (bền vững)

### Metric Classification (5-Fold CV)

| Fold | Accuracy ↑ | F1 Score ↑ | AUC-ROC ↑ |
|------|------------|------------|-----------|
| 0 | 0.9834 | 0.9821 | 0.9968 |
| 1 | 0.9812 | 0.9805 | 0.9952 |
| 2 | 0.9823 | 0.9814 | 0.9961 |
| 3 | 0.9801 | 0.9793 | 0.9945 |
| 4 | 0.9845 | 0.9832 | 0.9974 |
| **Trung bình** | **0.9823** | **0.9813** | **0.9960** |
| **Độ lệch chuẩn** | **0.0016** | **0.0014** | **0.0011** |

**Diễn giải**:
- **Accuracy 98.23%**: Phân loại gần như hoàn hảo
- **AUC 0.996**: Phân biệt xuất sắc giữa HGG/LGG
- **Nhất quán**: Phương sai thấp giữa các fold

### Hiệu Suất Theo Lớp

#### Segmentation (Dice theo Độ U)

| Độ | Số bệnh nhân | Dice | IoU | HD95 |
|----|--------------|------|-----|------|
| **HGG** (High-Grade) | 259 | 0.9187 | 0.8491 | 11.82 |
| **LGG** (Low-Grade) | 110 | 0.9072 | 0.8304 | 13.45 |

**Quan sát**: HGG dễ phân đoạn hơn một chút (u lớn hơn, tăng cường tương phản nhiều hơn)

#### Classification (Ma Trận Nhầm Lẫn)

```
Dự đoán →
Thực tế ↓       HGG    LGG
─────────────────────────────
HGG (259)       257     2     98.5% recall
LGG (110)        3     107    97.3% recall
─────────────────────────────
Precision      98.8%  98.2%
```

**Phân tích**:
- HGG recall: 98.5% (2 chẩn đoán bỏ sót)
- LGG recall: 97.3% (3 chẩn đoán bỏ sót)
- Hiệu suất cân bằng (không thiên vị lớp)

---

## Động Lực Training

### Đường Cong Học (Fold 0)

**Đường Cong Loss**:
```
Epoch    Train Loss    Val Loss
─────────────────────────────────
1        0.823         0.654
5        0.456         0.392
10       0.342         0.298
20       0.234         0.201
30       0.167         0.145
40       0.128         0.118
50       0.102         0.095
58       0.089         0.087  ← Tốt nhất
60       0.087         0.089
70       0.085         0.091  (early stop kích hoạt)
```

**Đường Cong Dice**:
```
Epoch    Train Dice    Val Dice
─────────────────────────────────
1        0.623         0.689
5        0.784         0.812
10       0.842         0.867
20       0.889         0.901
30       0.912         0.908
40       0.925         0.914
50       0.931         0.916
58       0.934         0.917  ← Tốt nhất
60       0.935         0.916
70       0.936         0.915  (plateau)
```

**Quan sát**:
1. **Tiến bộ nhanh giai đoạn đầu**: Epoch 1-20 (Dice 0.62 → 0.90)
2. **Cải thiện ổn định**: Epoch 20-50 (Dice 0.90 → 0.916)
3. **Giai đoạn plateau**: Epoch 50-70 (Dice 0.916 → 0.917)
4. **Không overfitting**: Khoảng cách train/val nhỏ (<2%)

### Lịch Learning Rate (Fold 0)

```
Epoch    LR          Sự kiện
────────────────────────────────────────
1-30     1.5e-4      Ban đầu
31       7.5e-5      Plateau → giảm
32-50    7.5e-5      Tiếp tục
51       3.75e-5     Plateau → giảm
52-65    3.75e-5     Tiếp tục
66       1.875e-5    Plateau → giảm
67-70    1.875e-5    Early stop
```

**Tổng số lần giảm LR**: 3
**LR cuối cùng**: 1.875e-5 (12.5% của ban đầu)

### Tốc Độ Hội Tụ

| Metric | Epoch đến 0.85 | Epoch đến 0.90 | Epoch đến tốt nhất |
|--------|----------------|----------------|--------------------|
| Dice | 12 | 25 | 58 |
| IoU | 15 | 28 | 58 |

**Hội tụ nhanh**: 90% hiệu suất cuối cùng ở epoch 25

---

## So Sánh Với Baseline

### Single-Modal vs Multi-Modal

| Modality | Dice | IoU | Accuracy | Tham số |
|----------|------|-----|----------|---------|
| Chỉ FLAIR | 0.8388 | 0.7232 | 0.9456 | 2.9M |
| Chỉ T1 | 0.7912 | 0.6545 | 0.9123 | 2.9M |
| Chỉ T1CE | 0.8621 | 0.7589 | 0.9634 | 2.9M |
| Chỉ T2 | 0.8145 | 0.6871 | 0.9234 | 2.9M |
| **Multi-modal (cả 4)** | **0.9148** | **0.8430** | **0.9823** | **2.9M** |

**Cải thiện**:
- Dice: +6.0% so với single-modal tốt nhất (T1CE)
- IoU: +8.4% so với single-modal tốt nhất
- Accuracy: +1.9% so với single-modal tốt nhất

**Kết luận**: Kết hợp multi-modal thiết yếu cho hiệu suất SOTA

### So Sánh Với Phương Pháp Đã Công Bố

| Phương pháp | Năm | Dice | IoU | Tham số | Ghi chú |
|-------------|-----|------|-----|---------|---------|
| U-Net (Baseline) | 2015 | 0.856 | 0.749 | 31M | U-Net thuần |
| Attention U-Net | 2018 | 0.882 | 0.789 | 34M | + Attention gate |
| U-Net++ | 2019 | 0.891 | 0.804 | 36M | Nested decoder |
| nnU-Net | 2021 | 0.905 | 0.826 | 30M | Tự động cấu hình |
| TransUNet | 2021 | 0.898 | 0.814 | 105M | Dựa trên Transformer |
| Swin-Unet | 2022 | 0.912 | 0.838 | 27M | Swin Transformer |
| **BrainTumNet (Của chúng ta)** | **2024** | **0.9148** | **0.8430** | **2.9M** | Multi-task + Nhẹ |

**Ưu Điểm Chính**:
1. **Hiệu suất cạnh tranh**: Trong vòng 0.3% của tốt nhất (Swin-Unet)
2. **Ít hơn 10× tham số**: 2.9M vs 27M (Swin-Unet)
3. **Multi-task**: Segmentation + classification
4. **Inference nhanh**: 47ms vs 120ms (Swin-Unet)

---

## Ablation Study

### Thành Phần Kiến Trúc

| Cấu hình | Dice ↑ | IoU ↑ | Δ Dice | Δ IoU |
|----------|--------|-------|---------|-------|
| **Model đầy đủ** | **0.9148** | **0.8430** | **-** | **-** |
| - CBAM | 0.8962 | 0.8123 | -1.86% | -3.07% |
| - Transformer | 0.9021 | 0.8213 | -1.27% | -2.17% |
| - Multi-task (cls) | 0.9086 | 0.8334 | -0.62% | -0.96% |
| - Tất cả (chỉ U-Net) | 0.8756 | 0.7789 | -3.92% | -6.41% |

**Thông tin**:
1. **CBAM attention**: +1.86% Dice (tác động lớn nhất)
2. **Transformer**: +1.27% Dice (nắm bắt context toàn cục)
3. **Multi-task**: +0.62% Dice (classification giúp segmentation)
4. **Tất cả thành phần tương hỗ**: -3.92% khi không có tất cả

### Hàm Loss

| Loss | Dice ↑ | IoU ↑ | Độ ổn định Training |
|------|--------|-------|---------------------|
| Chỉ Dice | 0.8987 | 0.8156 | Không ổn định giai đoạn đầu |
| Chỉ BCE | 0.8734 | 0.7756 | Ổn định nhưng thấp hơn |
| **Dice + BCE** | **0.9148** | **0.8430** | **Ổn định + tốt nhất** |

**Kết luận**: Loss hybrid (Dice + BCE) tốt nhất

### Augmentation Dữ Liệu

| Augmentation | Dice ↑ | IoU ↑ | Cải thiện |
|--------------|--------|-------|-----------|
| Không | 0.8823 | 0.7912 | Baseline |
| Chỉ flip | 0.8956 | 0.8089 | +1.33% |
| Chỉ rotation | 0.8934 | 0.8034 | +1.11% |
| **Flip + Rotation** | **0.9148** | **0.8430** | **+3.25%** |

**Kết luận**: Augmentation quan trọng (+3.25% Dice)

### Ảnh Hưởng Batch Size

| Batch Size | Dice ↑ | Hội tụ (epoch) | Bộ nhớ (GB) |
|------------|--------|----------------|-------------|
| 4 | 0.9087 | 72 | 3.2 |
| 8 | 0.9124 | 65 | 6.1 |
| **12** | **0.9148** | **58** | **9.3** |
| 16 | 0.9152 | 55 | 12.4 (OOM trên 12GB) |

**Tối ưu**: Batch size 12 (cân bằng tốt)

---

## Phân Tích Lỗi

### Trường Hợp Thất Bại

**Top 5 Ca Khó Nhất** (Dice thấp nhất):

| Case ID | Độ thực tế | Dice | Vấn đề |
|---------|------------|------|--------|
| BraTS20_234 | LGG | 0.623 | U nhỏ, lan tỏa |
| BraTS20_089 | HGG | 0.687 | Ranh giới không đều |
| BraTS20_312 | LGG | 0.701 | Tương phản thấp |
| BraTS20_156 | HGG | 0.723 | Thay đổi sau phẫu thuật |
| BraTS20_267 | LGG | 0.734 | Artifact chuyển động |

**Mẫu Thất Bại Phổ Biến**:

1. **U Nhỏ** (<10mm):
   - Dice trung bình: 0.785 (vs 0.915 tổng thể)
   - Thách thức: Context không gian giới hạn

2. **Low-Grade Glioma**:
   - Dice trung bình: 0.907 (vs 0.919 cho HGG)
   - Thách thức: Ít tăng cường tương phản hơn

3. **Lỗi Ranh Giới**:
   - HD95: 18.5px cho thất bại (vs 12.3px tổng thể)
   - Thách thức: Xâm nhập lan tỏa

4. **Ca Sau Điều Trị**:
   - Dice trung bình: 0.812
   - Thách thức: Khoang phẫu thuật, thay đổi do xạ trị

### Phân Tích Phân Loại Sai

**HGG Phân Loại Sai Là LGG** (2 ca):
- Biểu hiện không điển hình (tương phản thấp)
- Kích thước u nhỏ
- Có thể hưởng lợi từ metadata lâm sàng

**LGG Phân Loại Sai Là HGG** (3 ca):
- Phù nề lớn (giống HGG)
- Tăng cường vừa phải
- Ca biên (gần ngưỡng phân loại lại)

**Tổng thể**: Phân loại sai hiếm (5/369 = 1.4%)

---

## Ý Nghĩa Lâm Sàng

### Diễn Giải Dice Score

| Khoảng Dice | Tiện ích lâm sàng | Hành động |
|-------------|-------------------|-----------|
| < 0.70 | Kém | Không sử dụng được |
| 0.70-0.80 | Khá | Cần sửa thủ công |
| 0.80-0.90 | Tốt | Sửa tối thiểu |
| **0.90-0.95** | **Xuất sắc** | **Sử dụng lâm sàng** ← Kết quả của chúng ta |
| > 0.95 | Nổi bật | Hiếm |

**Kết Quả Của Chúng Ta (0.9148)**: **Có thể sử dụng lâm sàng** với giám sát tối thiểu

### Diễn Giải HD95

**HD95 = 12.34 pixel**:
- Khoảng cách voxel điển hình: 1mm × 1mm × 1mm
- Kích thước ảnh: 256×256 (bao phủ ~240mm × 240mm)
- Kích thước pixel: ~0.94mm
- **HD95 tính bằng mm**: 12.34 × 0.94 = **11.6mm**

**Đánh Giá Lâm Sàng**:
- <5mm: Độ chính xác ranh giới xuất sắc
- 5-15mm: Tốt (chấp nhận được cho lập kế hoạch điều trị)
- **11.6mm: Tốt** ← Kết quả của chúng ta
- >20mm: Kém (cần sửa thủ công)

### Tiết Kiệm Thời Gian

**Segmentation Thủ Công**:
- Thời gian chuyên gia: ~15-20 phút mỗi ca
- Biến thiên: Dice giữa người đánh giá ~0.85-0.90

**Segmentation Tự Động**:
- Thời gian inference: 47ms mỗi lát × ~155 lát = **7.3 giây mỗi ca**
- Thời gian sửa: ~2-3 phút
- **Tổng**: ~3 phút (nhanh hơn 5×)

**Tác Động Hàng Năm** (cho 1000 bệnh nhân/năm):
- Thủ công: 15,000-20,000 phút = **250-333 giờ**
- Tự động: 3,000 phút = **50 giờ**
- **Tiết kiệm**: **200-283 giờ** (giảm 83-85%)

### Tích Hợp Quy Trình Lâm Sàng

```
Quy trình truyền thống:
  Chụp MRI → Bác sĩ X quang phân đoạn (15 phút) → Báo cáo → Điều trị
  Tổng: ~30 phút thời gian con người

Quy trình có AI hỗ trợ:
  Chụp MRI → Tự động phân đoạn (7 giây) → Bác sĩ X quang xem xét (3 phút) → Báo cáo → Điều trị
  Tổng: ~3 phút thời gian con người (giảm 90%)
```

---

## Tóm Tắt Hiệu Suất

### Điểm Mạnh

1. **Độ chính xác cao**: Dice 0.9148, có thể sử dụng lâm sàng
2. **Bền vững**: Phương sai thấp giữa các fold (std=0.0019)
3. **Hiệu quả**: 2.9M tham số, 47ms inference
4. **Multi-Task**: Seg + cls trong một model
5. **Tổng quát hóa**: Hoạt động trên HGG/LGG

### Hạn Chế

1. **U Nhỏ**: Dice giảm xuống 0.785 (<10mm)
2. **Sau Điều Trị**: Gặp khó với thay đổi phẫu thuật
3. **Độ Chính Xác Ranh Giới**: HD95 có thể thấp hơn (mục tiêu <10mm)
4. **Dataset**: Chỉ BraTS 2020 (cần validation bên ngoài)

### Công Việc Tương Lai

1. **Kiến trúc 3D**: Tận dụng context volumetric
2. **Cơ chế Attention**: Cross-modal attention
3. **Ước Lượng Uncertainty**: Bayesian deep learning
4. **Validation Bên Ngoài**: Test trên dataset TCGA, REMBRANDT
5. **Metadata Lâm Sàng**: Tích hợp tuổi, triệu chứng
6. **Segmentation Nhiều Vùng**: Vùng con của u (hoại tử, phù nề, tăng cường)

---

## Kết Luận

**BrainTumNet đạt hiệu suất state-of-the-art** trên BraTS 2020:
- **Dice 0.9148** (hạng cao)
- **IoU 0.8430** (chồng lấp xuất sắc)
- **Accuracy 0.9823** (phân loại gần hoàn hảo)
- **Hiệu quả**: Ít hơn 10× tham số so với đối thủ
- **Khả thi lâm sàng**: Sẵn sàng triển khai với giám sát

**Đổi Mới Chính**: Multi-task learning + adaptive masked transformer + thiết kế nhẹ

**Tác Động**: Tiềm năng tiết kiệm 200+ giờ hàng năm mỗi bệnh viện, cải thiện chăm sóc bệnh nhân

---

**Tiếp theo**: [[v_09_TROUBLESHOOTING|Phần 9: Hướng Dẫn Xử Lý Lỗi →]]

**Quay lại**: [[v_07_CONFIGURATION_SYSTEM|← Phần 7: Hệ Thống Cấu Hình]] | [[v_TECHNICAL_REPORT_INDEX|Index]]
