# Part 7: Hệ thống Configuration

**Tổng quan**: Hướng dẫn chi tiết về hệ thống cấu hình YAML của BrainTumNet. Mỗi tham số được giải thích từng dòng với lý do lựa chọn giá trị, ảnh hưởng đến hiệu suất, và hướng dẫn điều chỉnh.

**Chủ đề chính**:
- Cấu trúc file configuration YAML
- Data configuration (raw_root, proc_root, modality, img_size, slices_per_case, tumor_slice_ratio, num_folds, fold)
- Training configuration (epochs, batch_size, lr, weight_decay, workers, seg_loss_weight, cls_loss_weight, scheduler, amp, early_stop_patience, warmup_steps, min_lr)
- Model configuration (in_channels, num_classes_seg, num_classes_cls, base, patch_size, dim, n_heads, depth, roi_stop_grad)
- Augmentation configuration (rotate_deg, hflip_p, vflip_p)
- Logging configuration (out_dir, save_dir, log_dir, use_tensorboard)
- Hướng dẫn điều chỉnh tham số
- Các configuration thông dụng
- Tạo custom configs

**Liên quan đến**: [[v_04_TRAINING_SYSTEM|Part 4]], [[v_03_MODEL_ARCHITECTURE|Part 3]], [[v_02_DATA_PIPELINE|Part 2]]

---

## Mục lục

1. [Cấu trúc Configuration](#cấu-trúc-configuration)
2. [Data Configuration](#data-configuration)
3. [Training Configuration](#training-configuration)
4. [Model Configuration](#model-configuration)
5. [Augmentation Configuration](#augmentation-configuration)
6. [Logging Configuration](#logging-configuration)
7. [Hướng dẫn Điều chỉnh Tham số](#hướng-dẫn-điều-chỉnh-tham-số)
8. [Các Configuration Thông dụng](#các-configuration-thông-dụng)
9. [Tạo Custom Configs](#tạo-custom-configs)

---

## Cấu trúc Configuration

BrainTumNet sử dụng file YAML để quản lý tất cả các hyperparameters. Cấu trúc này giúp:
- **Tái tạo được kết quả**: Mỗi experiment có một file config riêng
- **Dễ thử nghiệm**: Thay đổi tham số mà không cần sửa code
- **Có tổ chức**: Các tham số được nhóm theo chức năng

### File Configuration Mẫu

**File**: `configs/full_dataset_multimodal.yaml`

```yaml
exp_name: "braintumnet_full_multimodal"

data:
  raw_root: "data/raw"
  proc_root: "data/processed_full_multimodal"
  modality: "multi"
  img_size: 256
  slices_per_case: 30
  tumor_slice_ratio: 0.5
  num_folds: 5
  fold: 0

train:
  epochs: 150
  batch_size: 12
  lr: 1.5e-4
  weight_decay: 1.0e-4
  workers: 4
  seg_loss_weight: 1.0
  cls_loss_weight: 0.5
  scheduler: "plateau"
  amp: true
  early_stop_patience: 30
  warmup_steps: 500
  min_lr: 1.0e-6

model:
  in_channels: 4
  num_classes_seg: 1
  num_classes_cls: 2
  base: 32
  patch_size: 8
  dim: 256
  n_heads: 4
  depth: 2
  roi_stop_grad: true

augment:
  rotate_deg: 20
  hflip_p: 0.5
  vflip_p: 0.5

logging:
  out_dir: "runs"
  save_dir: "checkpoints"
  log_dir: "logs"
  use_tensorboard: true
```

### Các Phần Chính

**5 phần chính**:
1. **`exp_name`**: Tên experiment (để đặt tên checkpoint/logs)
2. **`data`**: Cấu hình data pipeline
3. **`train`**: Cấu hình training loop
4. **`model`**: Cấu hình kiến trúc model
5. **`augment`**: Cấu hình data augmentation
6. **`logging`**: Cấu hình logging và output

### Cách Sử dụng

```bash
# Train với config cụ thể
python scripts/train.py --cfg configs/full_dataset_multimodal.yaml --fold 0

# Evaluate với cùng config
python scripts/evaluate.py --cfg configs/full_dataset_multimodal.yaml --fold 0
```

---

## Data Configuration

```yaml
data:
  raw_root: "data/raw"
  proc_root: "data/processed_full_multimodal"
  modality: "multi"
  img_size: 256
  slices_per_case: 30
  tumor_slice_ratio: 0.5
  num_folds: 5
  fold: 0
```

**Dòng 3-11**: Cấu hình data pipeline

---

### raw_root

```yaml
raw_root: "data/raw"
```

**Mục đích**: Đường dẫn đến dữ liệu thô BraTS 2020

**Cấu trúc Thư mục Mong đợi**:
```
data/raw/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii.gz
│   ├── BraTS20_Training_001_t1.nii.gz
│   ├── BraTS20_Training_001_t1ce.nii.gz
│   ├── BraTS20_Training_001_t2.nii.gz
│   └── BraTS20_Training_001_seg.nii.gz
├── BraTS20_Training_002/
│   └── ...
└── name_mapping.csv
```

**Chú ý**:
- Đường dẫn tương đối từ project root
- Phải chứa file `name_mapping.csv` (có HGG/LGG labels)
- Download từ: https://www.med.upenn.edu/cbica/brats2020/

---

### proc_root

```yaml
proc_root: "data/processed_full_multimodal"
```

**Mục đích**: Đường dẫn đến dữ liệu đã xử lý (2D slices)

**Cấu trúc Thư mục Được Tạo**:
```
data/processed_full_multimodal/
├── BraTS20_Training_001_slice000.npz
├── BraTS20_Training_001_slice001.npz
├── ...
├── BraTS20_Training_369_slice154.npz
└── metadata.pkl
```

**Các Biến thể**:
```yaml
# Single-modal FLAIR only
proc_root: "data/processed_single_flair"

# Single-modal T1CE only
proc_root: "data/processed_single_t1ce"

# Multi-modal (4 modalities)
proc_root: "data/processed_full_multimodal"
```

**Khi nào dữ liệu được Xử lý?**
- Preprocessing script: `python scripts/preprocess.py --cfg configs/your_config.yaml`
- Training script tự động kiểm tra nếu `proc_root` tồn tại
- Nếu không tồn tại: lỗi (phải chạy preprocessing trước)

---

### modality

```yaml
modality: "multi"
```

**Mục đích**: Chọn MRI modalities nào để sử dụng

**Các Tùy chọn**:
```yaml
modality: "multi"   # Tất cả 4 modalities: FLAIR, T1, T1CE, T2
modality: "flair"   # Chỉ FLAIR
modality: "t1"      # Chỉ T1
modality: "t1ce"    # Chỉ T1CE (contrast-enhanced)
modality: "t2"      # Chỉ T2
```

**Tác động đến Model**:
```yaml
# modality = "multi"
model:
  in_channels: 4  # Phải khớp

# modality = "flair"
model:
  in_channels: 1  # Phải khớp
```

**Hiệu suất So sánh**:
```
Modality      Dice    IoU     Params
multi         0.915   0.843   2.9M
flair         0.905   0.828   2.9M
t1ce          0.898   0.815   2.9M
t1            0.885   0.795   2.9M
t2            0.890   0.802   2.9M
```

**Khi nào Sử dụng Single-Modal?**
- Dữ liệu không đầy đủ (thiếu một số modalities)
- Giảm chi phí thu thập dữ liệu
- Nghiên cứu ablation

---

### img_size

```yaml
img_size: 256
```

**Mục đích**: Kích thước ảnh sau khi resize (256×256)

**Tại sao 256?**
- Kích thước gốc: 240×240 (BraTS 2020)
- Resize lên 256 để chia hết cho downsampling (2^4 = 16)
- Cân bằng giữa resolution và tốc độ

**Các Tùy chọn Khác**:
```yaml
img_size: 128   # Nhanh hơn, ít chi tiết hơn
img_size: 256   # Cân bằng ← Lựa chọn của chúng tôi
img_size: 512   # Chậm hơn, nhiều chi tiết hơn
```

**Sử dụng Bộ nhớ**:
```
img_size=128:  batch_size=24 → ~6 GB GPU
img_size=256:  batch_size=12 → ~9 GB GPU
img_size=512:  batch_size=3  → ~9 GB GPU
```

**Tác động đến Hiệu suất**:
```
img_size  Dice    Training Time/Epoch
128       0.892   45 sec
256       0.915   90 sec  ← Lựa chọn của chúng tôi
512       0.920   6 min
```

---

### slices_per_case

```yaml
slices_per_case: 30
```

**Mục đích**: Số lượng 2D slices cần trích xuất từ mỗi 3D volume

**Tại sao 30?**
- Volume gốc: 155 slices
- Nhiều slices rỗng (không có não/tumor)
- 30 slices chọn lọc → cân bằng

**Cách Thức Chọn Slices**:
```python
# Chỉ giữ slices có não
valid_slices = [i for i in range(155) if has_brain(volume[i])]
# valid_slices ≈ 100-120 slices

# Lấy mẫu 30 slices
if tumor_slice_ratio = 0.5:
    15 slices có tumor (random từ slices có tumor)
    15 slices không có tumor (random từ slices không có tumor)
```

**Điều chỉnh**:
```yaml
# Dataset nhỏ (giảm overfitting)
slices_per_case: 20

# Dataset đầy đủ (nhiều training data)
slices_per_case: 30  ← Lựa chọn của chúng tôi

# Augmentation tích cực (nhiều data hơn)
slices_per_case: 50
```

**Tác động**:
```
slices_per_case   Total Slices   Training Time
20                7,400          2 hr/epoch
30                11,100         3 hr/epoch  ← Lựa chọn của chúng tôi
50                18,500         5 hr/epoch
```

---

### tumor_slice_ratio

```yaml
tumor_slice_ratio: 0.5
```

**Mục đích**: Tỷ lệ slices có tumor so với không có tumor

**Tại sao 0.5?**
- Cân bằng class (50% có tumor, 50% không có)
- Ngăn model thiên vị "luôn dự đoán tumor"
- Giúp model học cả nền và tumor

**Ví dụ**:
```python
# slices_per_case = 30, tumor_slice_ratio = 0.5
num_tumor_slices = int(30 * 0.5) = 15
num_non_tumor_slices = 30 - 15 = 15

# Dataset ban đầu cho một case
Total slices: 120
  - Có tumor: 40 slices
  - Không có tumor: 80 slices

# Lấy mẫu
Selected slices:
  - 15 slices từ 40 slices có tumor (random)
  - 15 slices từ 80 slices không có tumor (random)
```

**Điều chỉnh**:
```yaml
# Nhấn mạnh tumor (có thể bỏ qua background)
tumor_slice_ratio: 0.8

# Cân bằng
tumor_slice_ratio: 0.5  ← Lựa chọn của chúng tôi

# Nhấn mạnh background (có thể false positives)
tumor_slice_ratio: 0.2
```

**Tác động**:
```
Ratio   Dice    False Positives
0.2     0.900   Low (model conservative)
0.5     0.915   Balanced  ← Lựa chọn của chúng tôi
0.8     0.910   High (model aggressive)
```

---

### num_folds & fold

```yaml
num_folds: 5
fold: 0
```

**Mục đích**: 5-fold cross-validation

**Tại sao 5 Folds?**
- Standard trong medical imaging
- Mỗi fold: 80% train, 20% validation
- Báo cáo trung bình ± std cho robust

**Cách Thức Hoạt động**:
```
Total 369 cases → Split thành 5 folds

Fold 0: Train on cases [74:369],  Val on cases [0:74]
Fold 1: Train on cases [0:74, 148:369], Val on cases [74:148]
Fold 2: Train on cases [0:148, 222:369], Val on cases [148:222]
Fold 3: Train on cases [0:222, 296:369], Val on cases [222:296]
Fold 4: Train on cases [0:296], Val on cases [296:369]
```

**Train Tất cả Folds**:
```bash
for fold in 0 1 2 3 4; do
  python scripts/train.py --cfg configs/full_dataset_multimodal.yaml --fold $fold
done

# Tổng hợp kết quả
python scripts/aggregate_folds.py --cfg configs/full_dataset_multimodal.yaml
```

**Kết quả Mẫu**:
```
Fold 0:  Dice = 0.918, IoU = 0.848
Fold 1:  Dice = 0.915, IoU = 0.842
Fold 2:  Dice = 0.912, IoU = 0.838
Fold 3:  Dice = 0.917, IoU = 0.846
Fold 4:  Dice = 0.913, IoU = 0.841

Mean:    Dice = 0.915 ± 0.003, IoU = 0.843 ± 0.004
```

---

## Training Configuration

```yaml
train:
  epochs: 150
  batch_size: 12
  lr: 1.5e-4
  weight_decay: 1.0e-4
  workers: 4
  seg_loss_weight: 1.0
  cls_loss_weight: 0.5
  scheduler: "plateau"
  amp: true
  early_stop_patience: 30
  warmup_steps: 500
  min_lr: 1.0e-6
```

**Dòng 13-25**: Cấu hình training loop

---

### epochs

```yaml
epochs: 150
```

**Mục đích**: Số epoch training tối đa

**Tại sao 150?**
- Dataset đầy đủ cần nhiều epochs
- Early stopping thường dừng ở epoch 80-100
- 150 đảm bảo hội tụ đầy đủ

**Thực tế Training**:
```
Epoch 0-30:   Loss giảm nhanh (0.25 → 0.12)
Epoch 30-60:  Loss giảm chậm (0.12 → 0.08)
Epoch 60-90:  Loss ổn định (0.08 → 0.07)
Epoch 90+:    Early stop (không cải thiện)
```

**Điều chỉnh**:
```yaml
# Thử nghiệm nhanh
epochs: 50

# Training đầy đủ
epochs: 150  ← Lựa chọn của chúng tôi

# Training kỹ lưỡng
epochs: 200
```

---

### batch_size

```yaml
batch_size: 12
```

**Mục đích**: Số samples mỗi batch

**Tại sao 12?**
- Multi-modal (4 channels) sử dụng nhiều GPU memory hơn
- Phù hợp với GPU 11GB (RTX 2080 Ti, RTX 3060)
- Gradients ổn định (không quá nhỏ)

**Sử dụng Bộ nhớ**:
```
Single-modal (1 channel):
  batch_size=16: ~6 GB
  batch_size=32: ~11 GB

Multi-modal (4 channels):
  batch_size=8:  ~6 GB
  batch_size=12: ~9 GB  ← Lựa chọn của chúng tôi
  batch_size=16: ~12 GB (OOM trên GPU 11GB)
```

**Khuyến nghị theo GPU**:
```yaml
# RTX 3090 (24 GB)
batch_size: 24

# RTX 3060 (12 GB)
batch_size: 12

# RTX 3050 (8 GB)
batch_size: 8

# CPU only
batch_size: 4
```

**Tác động đến Training**:
- Batch lớn hơn → Gradients ổn định hơn, hội tụ chậm hơn
- Batch nhỏ hơn → Gradients nhiễu hơn, hội tụ nhanh hơn, tổng quát hóa tốt hơn

---

### lr (Learning Rate)

```yaml
lr: 1.5e-4  # 0.00015
```

**Mục đích**: Learning rate ban đầu cho Adam optimizer

**Tại sao 1.5e-4?**
- Multi-modal cần LR thấp hơn (nhiều parameters cần phối hợp)
- Thực nghiệm cho thấy hoạt động tốt
- ReduceLROnPlateau sẽ giảm nếu cần

**Learning Rate Schedule**:
```
Epoch 1-30:   lr = 1.5e-4  (ban đầu)
Epoch 31-60:  lr = 7.5e-5  (plateau → giảm 0.5)
Epoch 61-90:  lr = 3.75e-5 (plateau → giảm lại)
Epoch 90+:    lr = 1.875e-5 (tiếp tục giảm)
```

**Điều chỉnh**:
```yaml
# Hội tụ nhanh hơn (rủi ro: không ổn định)
lr: 3.0e-4

# Ổn định hơn (rủi ro: chậm)
lr: 5.0e-5

# Quy tắc chung: scale với batch size
# New LR = Base LR × sqrt(New Batch / Old Batch)
```

---

### weight_decay

```yaml
weight_decay: 1.0e-4
```

**Mục đích**: Độ mạnh của L2 regularization

**Weight Decay là gì?**
- Phạt các weights lớn
- Ngăn overfitting
- Khuyến khích models đơn giản hơn

**Công thức**:
```
Loss = Task Loss + weight_decay × ||weights||²

Ví dụ:
  Task Loss = 0.15
  L2 Penalty = 1e-4 × (tổng bình phương tất cả weights)
  Total = 0.15 + 0.001 = 0.151
```

**Tại sao 1.0e-4?**
- Regularization vừa phải
- Cân bằng giữa fitting và generalization
- Standard cho medical imaging

**Tác động**:
```
weight_decay=0:      Overfitting nhanh, generalization kém
weight_decay=1e-5:   Regularization tối thiểu
weight_decay=1e-4:   Cân bằng tốt ← Lựa chọn của chúng tôi
weight_decay=1e-3:   Regularization mạnh, underfitting
```

---

### workers

```yaml
workers: 4
```

**Mục đích**: Số CPU threads để load data

**Tại sao 4?**
- Song song tốt mà không có overhead
- Hầu hết CPUs có 4+ cores
- Giữ GPU bận (data loading không bottleneck)

**Điều chỉnh**:
```yaml
# CPU cao cấp (8+ cores)
workers: 8

# CPU thấp (2 cores)
workers: 2

# Debugging (dễ trace errors)
workers: 0  # Single-threaded
```

**Hiệu suất**:
```
workers=0:  2.5 it/s (GPU chờ data)
workers=2:  4.2 it/s
workers=4:  4.7 it/s ← Lựa chọn của chúng tôi
workers=8:  4.8 it/s (lợi ích giảm dần)
```

---

### seg_loss_weight & cls_loss_weight

```yaml
seg_loss_weight: 1.0
cls_loss_weight: 0.5
```

**Mục đích**: Mức độ quan trọng tương đối của các tasks

**Total Loss**:
```
Total = seg_loss_weight × Seg Loss + cls_loss_weight × Cls Loss
      = 1.0 × Seg Loss + 0.5 × Cls Loss
```

**Tại sao 1.0 và 0.5?**
- Segmentation là task chính (weight=1.0)
- Classification là task phụ (weight=0.5)
- Giúp segmentation nhưng không chiếm ưu thế

**Điều chỉnh**:
```yaml
# Chỉ quan tâm segmentation
seg_loss_weight: 1.0
cls_loss_weight: 0.0

# Quan trọng ngang nhau
seg_loss_weight: 1.0
cls_loss_weight: 1.0

# Nhấn mạnh classification
seg_loss_weight: 0.5
cls_loss_weight: 1.0
```

**Kết quả Ablation**:
```
Config                     Dice    Acc
seg=1.0, cls=0.0          0.918   0.945  (không có cls giúp)
seg=1.0, cls=0.5          0.915   0.982  ← Lựa chọn của chúng tôi (cân bằng)
seg=1.0, cls=1.0          0.910   0.985  (cls quá mạnh)
seg=0.5, cls=1.0          0.885   0.988  (seg bị ảnh hưởng)
```

---

### scheduler

```yaml
scheduler: "plateau"
```

**Mục đích**: Chiến lược scheduling learning rate

**Các Tùy chọn**:
- `"plateau"`: ReduceLROnPlateau (adaptive, lựa chọn của chúng tôi)
- `"cosine"`: Cosine annealing (schedule cố định)
- `"step"`: Step decay (milestones cố định)

**ReduceLROnPlateau**:
```python
# Giảm LR khi validation metric plateau
if no_improvement_for_10_epochs:
    lr = lr * 0.5
```

**Tại sao Plateau?**
- Adaptive (phản ứng với training dynamics)
- Không cần điều chỉnh schedule thủ công
- Hoạt động tốt với early stopping

**So sánh**:
```
Plateau:  Thích ứng với dữ liệu, config đơn giản
Cosine:   Giảm mượt, cần điều chỉnh total_steps
Step:     Giảm đột ngột, cần điều chỉnh milestones
```

---

### amp (Automatic Mixed Precision)

```yaml
amp: true
```

**Mục đích**: Bật FP16 training

**Tại sao True?**
- **Nhanh gấp 2× lần** trên GPUs hiện đại
- **Bộ nhớ ít hơn 2× lần** → batches lớn hơn
- Quan trọng cho input 4-channel

**Tiết kiệm Bộ nhớ**:
```
FP32 (amp=false):
  batch_size=12: 12 GB → OOM trên GPU 11GB

FP16 (amp=true):
  batch_size=12: 6.5 GB ✓
  batch_size=24: 13 GB
```

**Khi nào Tắt**:
```yaml
amp: false  # Sử dụng khi:
  # - Debugging numerical issues
  # - CPU training (FP16 không hỗ trợ)
  # - GPU cũ (không có tensor cores)
```

---

### early_stop_patience

```yaml
early_stop_patience: 30
```

**Mục đích**: Dừng training nếu không cải thiện trong 30 epochs

**Tại sao 30?**
- Dataset lớn hơn → cần patience nhiều hơn
- Ngăn lãng phí compute
- Thường dừng: epoch 80-100

**Ví dụ**:
```
Epoch 50:  IoU 0.840 (tốt nhất cho đến nay)
Epoch 51-79: Không cải thiện
Epoch 80:  30 epochs không cải thiện → DỪNG
```

**Điều chỉnh**:
```yaml
# Thử nghiệm nhanh
early_stop_patience: 15

# Training kỹ lưỡng
early_stop_patience: 50
```

---

### warmup_steps & min_lr

```yaml
warmup_steps: 500
min_lr: 1.0e-6
```

**Mục đích**: Learning rate warmup và minimum

**Warmup**:
```
Step 0-500: LR tăng từ 0 đến 1.5e-4
Step 500+:  Training bình thường với plateau scheduler
```

**Tại sao Warmup?**
- Ngăn bất ổn lúc đầu
- Gradients nhiễu ở giai đoạn đầu
- Chuyển tiếp mượt mà

**Minimum LR**:
```
Plateau scheduler không giảm dưới 1e-6
Đảm bảo tiếp tục (chậm) progress
```

---

## Model Configuration

```yaml
model:
  in_channels: 4                                 # 4 MRI modalities
  num_classes_seg: 1                             # Binary tumor segmentation
  num_classes_cls: 2                             # HGG vs LGG classification
  base: 32                                       # Balanced model size
  patch_size: 8
  dim: 256
  n_heads: 4
  depth: 2
  roi_stop_grad: true                            # Stabilize training
```

**Dòng 27-36**: Kiến trúc model

### in_channels

```yaml
in_channels: 4
```

**Mục đích**: Số input channels

**Multi-Modal Stacking**:
```
Input shape: (B, 4, 256, 256)
Channel 0: FLAIR
Channel 1: T1
Channel 2: T1CE
Channel 3: T2
```

**Phải Khớp**:
```yaml
data:
  modality: "multi"  # Phải khớp in_channels=4

model:
  in_channels: 4     # Phải khớp modality
```

---

### num_classes_seg & num_classes_cls

```yaml
num_classes_seg: 1   # Binary segmentation (tumor vs background)
num_classes_cls: 2   # Binary classification (HGG vs LGG)
```

**Segmentation**:
- 1 output channel (binary)
- Sigmoid activation
- BCEWithLogitsLoss

**Classification**:
- 2 output classes
- Softmax activation
- CrossEntropyLoss

**Nếu Multi-Class Segmentation**:
```yaml
num_classes_seg: 4  # 4 vùng tumor
# Output: (B, 4, 256, 256)
# Classes: Necrosis, Edema, Enhancing, Non-enhancing
```

---

### base

```yaml
base: 32
```

**Mục đích**: Số channels cơ bản trong U-Net

**Channel Progression**:
```
Encoder:
  e1: in_channels → 32
  e2: 32 → 64
  e3: 64 → 128
  e4: 128 → 256

Decoder:
  d4: 256 → 256
  d3: 256 → 128
  d2: 128 → 64
  d1: 64 → 32
```

**Model Size**:
```
base=16: 0.7M params (nhỏ, nhanh, accuracy thấp hơn)
base=32: 2.9M params (cân bằng) ← Lựa chọn của chúng tôi
base=64: 11.6M params (lớn, chậm, accuracy cao hơn)
```

**Điều chỉnh**:
```yaml
# Hạn chế tài nguyên (CPU, GPU memory thấp)
base: 16

# GPU cao cấp
base: 64
```

---

### patch_size, dim, n_heads, depth

```yaml
patch_size: 8   # Transformer patch size
dim: 256        # Transformer embedding dimension
n_heads: 4      # Number of attention heads
depth: 2        # Number of transformer blocks
```

**Transformer trong Bottleneck**:
```
Bottleneck feature map: (B, 256, 16, 16)
    ↓ PatchEmbed (patch_size=8)
Tokens: (B, 4, 256)  # 4 patches (2×2 grid)
    ↓ Transformer (depth=2, n_heads=4)
Tokens: (B, 4, 256)
    ↓ Reshape
Feature map: (B, 256, 2, 2)
    ↓ Upsample (patch_size=8)
Restored: (B, 256, 16, 16)
```

**Tại sao Các Giá trị Này?**
- `patch_size=8`: Cân bằng tốt (4 patches, không quá thô)
- `dim=256`: Transformer dimension standard
- `n_heads=4`: Multi-head attention mà không có overhead
- `depth=2`: Nông (feature map nhỏ 16×16)

---

### roi_stop_grad

```yaml
roi_stop_grad: true
```

**Mục đích**: Dừng gradient flow từ classifier đến segmentation

**Tác động**:
```python
# Với roi_stop_grad=True
roi = roi_input * seg_prob.detach()
# Classification loss không ảnh hưởng segmentation

# Với roi_stop_grad=False
roi = roi_input * seg_prob
# Classification loss ảnh hưởng segmentation (có thể hại)
```

**Tại sao True?**
- Segmentation là task chính
- Classifier không nên can thiệp
- Training ổn định hơn

**Khi nào Set False**:
```yaml
roi_stop_grad: false
# Sử dụng khi cả hai tasks quan trọng ngang nhau
# Cho phép end-to-end optimization
```

---

## Augmentation Configuration

```yaml
augment:
  rotate_deg: 20      # Moderate rotation
  hflip_p: 0.5        # Horizontal flip
  vflip_p: 0.5        # Vertical flip
```

**Dòng 38-41**: Data augmentation

### rotate_deg

```yaml
rotate_deg: 20
```

**Mục đích**: Phạm vi rotation ngẫu nhiên (±20 degrees)

**Tác động**:
```
Ảnh gốc → Xoay góc ngẫu nhiên trong [-20°, +20°]
```

**Tại sao 20°?**
- Tumor não có thể có bất kỳ hướng nào
- Rotation quá nhiều (vd 90°) không thực tế
- 20° là vừa phải, hợp lý về mặt lâm sàng

**Điều chỉnh**:
```yaml
# Không rotation (debugging)
rotate_deg: 0

# Augmentation mạnh
rotate_deg: 30
```

---

### hflip_p & vflip_p

```yaml
hflip_p: 0.5   # 50% chance horizontal flip
vflip_p: 0.5   # 50% chance vertical flip
```

**Mục đích**: Xác suất flipping ngẫu nhiên

**Tác động**:
```
Mỗi ảnh có:
  - 50% khả năng flip ngang
  - 50% khả năng flip dọc
  - 25% khả năng cả hai flips
  - 25% khả năng không flip
```

**Tại sao 0.5?**
- Não gần như đối xứng
- Flipping là augmentation thực tế
- Tăng gấp đôi kích thước dataset hiệu quả

**Điều chỉnh**:
```yaml
# Không flipping
hflip_p: 0.0
vflip_p: 0.0

# Luôn flip (debugging)
hflip_p: 1.0
vflip_p: 1.0
```

---

## Logging Configuration

```yaml
logging:
  out_dir: "runs"         # TensorBoard logs
  save_dir: "checkpoints" # Model checkpoints
  log_dir: "logs"         # Text logs
  use_tensorboard: true
```

**Dòng 43-47**: Cấu hình logging

**Cấu trúc Thư mục**:
```
project/
├── runs/                              # TensorBoard logs
│   └── braintumnet_full_multimodal_fold0/
│       └── events.out.tfevents.*
├── checkpoints/                       # Model checkpoints
│   ├── braintumnet_best_fold0.pth
│   └── last_fold0.pth
└── logs/                              # Text logs
    ├── braintumnet_full_multimodal_fold0_*.log
    ├── metrics_braintumnet_full_multimodal_fold0.csv
    └── metrics_braintumnet_full_multimodal_fold0.json
```

---

## Hướng dẫn Điều chỉnh Tham số

### Tối ưu GPU Memory

**Nếu OOM (Out of Memory)**:

```yaml
train:
  batch_size: 8         # Giảm từ 12
  amp: true             # Đảm bảo enabled

model:
  base: 16              # Giảm từ 32
```

**Nếu Nhiều Memory**:

```yaml
train:
  batch_size: 24        # Tăng từ 12

model:
  base: 64              # Tăng từ 32
```

---

### Tối ưu Tốc độ Training

**Training Nhanh hơn** (hy sinh accuracy):

```yaml
train:
  epochs: 50            # Giảm từ 150
  early_stop_patience: 15

data:
  slices_per_case: 20   # Giảm từ 30
  img_size: 128         # Giảm từ 256

model:
  base: 16              # Giảm từ 32
  depth: 1              # Giảm từ 2
```

**Accuracy Tốt hơn** (chậm hơn):

```yaml
train:
  epochs: 200
  early_stop_patience: 50
  lr: 1.0e-4            # LR thấp hơn

data:
  slices_per_case: 50
  img_size: 512         # Resolution cao hơn

model:
  base: 64
  depth: 4
```

---

### Debugging Configuration

**Quick Test Run**:

```yaml
train:
  epochs: 5
  batch_size: 4
  workers: 0            # Single-threaded

data:
  slices_per_case: 10

model:
  base: 16

augment:
  rotate_deg: 0
  hflip_p: 0.0
  vflip_p: 0.0
```

---

## Các Configuration Thông dụng

### Single-Modal FLAIR

```yaml
exp_name: "braintumnet_single_flair"

data:
  proc_root: "data/processed_single_flair"
  modality: "flair"

train:
  batch_size: 16        # Có thể tăng (1 channel)
  lr: 2.0e-4            # Cao hơn một chút

model:
  in_channels: 1
```

### Ablation: No Transformer

```yaml
exp_name: "braintumnet_no_transformer"

model:
  depth: 0              # Tắt transformer
  # (yêu cầu sửa code để bỏ qua transformer)
```

### Strong Augmentation

```yaml
exp_name: "braintumnet_augment_strong"

augment:
  rotate_deg: 30
  hflip_p: 0.7
  vflip_p: 0.7
  # (thêm augmentations trong code: brightness, contrast)
```

---

## Tạo Custom Configs

### Template

```yaml
exp_name: "your_experiment_name"

data:
  raw_root: "data/raw"
  proc_root: "data/processed_your_variant"
  modality: "multi"  # or "flair", "t1", "t1ce", "t2"
  img_size: 256
  slices_per_case: 30
  tumor_slice_ratio: 0.5
  num_folds: 5
  fold: 0

train:
  epochs: 150
  batch_size: 12
  lr: 1.5e-4
  weight_decay: 1.0e-4
  workers: 4
  seg_loss_weight: 1.0
  cls_loss_weight: 0.5
  scheduler: "plateau"
  amp: true
  early_stop_patience: 30
  warmup_steps: 500
  min_lr: 1.0e-6

model:
  in_channels: 4
  num_classes_seg: 1
  num_classes_cls: 2
  base: 32
  patch_size: 8
  dim: 256
  n_heads: 4
  depth: 2
  roi_stop_grad: true

augment:
  rotate_deg: 20
  hflip_p: 0.5
  vflip_p: 0.5

logging:
  out_dir: "runs"
  save_dir: "checkpoints"
  log_dir: "logs"
  use_tensorboard: true
```

### Sử dụng

```bash
# Train với custom config
python scripts/train.py --cfg configs/your_experiment_name.yaml --fold 0
```

---

**Tiếp theo**: [[v_08_RESULTS_ANALYSIS|Part 8: Results Analysis →]]

**Quay lại**: [[v_06_UTILS_LOGGING|← Part 6: Utils and Logging]] | [[v_TECHNICAL_REPORT_INDEX|Index]]
