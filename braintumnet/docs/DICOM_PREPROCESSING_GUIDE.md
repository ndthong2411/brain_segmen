# DICOM Preprocessing Guide - BrainTumNet

**Version**: 1.0
**Last Updated**: 2025-10-29

---

## 📋 Tổng Quan

Script `preprocess_dicom_to_multiclass.py` cho phép xử lý dữ liệu DICOM từ các nguồn khác nhau và chuyển đổi thành format PNG chuẩn cho training BrainTumNet.

### Tính Năng Chính

✅ **Hỗ trợ nhiều cấu trúc DICOM**
- BraTS-style với folder modality riêng biệt
- Single patient với multiple series
- Flat directory với tất cả DICOM files
- Auto-detect structure

✅ **Auto-detect modality**
- Tự động nhận diện FLAIR, T1, T1CE, T2 từ SeriesDescription
- Hỗ trợ segmentation masks (nếu có)

✅ **Xử lý robust**
- Rescale slope/intercept
- Percentile normalization (1-99%)
- Multi-format segmentation support

✅ **K-fold cross-validation**
- Patient-level splitting
- Đảm bảo không leak data giữa folds

---

## 🚀 Quick Start

### 1. Cài Đặt Dependencies

```bash
cd braintumnet
pip install -r requirements.txt
```

Đảm bảo `pydicom>=2.3.0` đã được cài đặt:
```bash
pip install pydicom
```

### 2. Chuẩn Bị DICOM Data

Có 3 cấu trúc DICOM được hỗ trợ:

#### **Cấu trúc 1: BraTS-style** (khuyến nghị)
```
dicom_data/
├── Patient001/
│   ├── FLAIR/
│   │   ├── slice001.dcm
│   │   ├── slice002.dcm
│   │   └── ...
│   ├── T1/
│   ├── T1CE/
│   ├── T2/
│   └── SEG/  (optional)
├── Patient002/
└── ...
```

#### **Cấu trúc 2: Patient-based**
```
dicom_data/
├── Patient001/
│   ├── Series001_FLAIR/
│   ├── Series002_T1/
│   ├── Series003_T1CE/
│   ├── Series004_T2/
│   └── Series005_SEG/
└── ...
```

#### **Cấu trúc 3: Flat directory**
```
dicom_data/
├── IMG00001.dcm
├── IMG00002.dcm
└── ...
```

### 3. Chạy Preprocessing

#### **Auto-detect (khuyến nghị)**
```bash
python scripts/preprocess_dicom_to_multiclass.py \
    --dicom_dir /path/to/dicom/data \
    --out_dir data/processed_multiclass_dicom \
    --structure auto \
    --img_size 256 \
    --num_folds 5
```

#### **Chỉ định cấu trúc cụ thể**
```bash
# BraTS-style
python scripts/preprocess_dicom_to_multiclass.py \
    --dicom_dir /path/to/dicom/data \
    --out_dir data/processed_multiclass_dicom \
    --structure brats

# Patient-based
python scripts/preprocess_dicom_to_multiclass.py \
    --dicom_dir /path/to/dicom/data \
    --out_dir data/processed_multiclass_dicom \
    --structure patient
```

#### **Test với ít patients**
```bash
python scripts/preprocess_dicom_to_multiclass.py \
    --dicom_dir /path/to/dicom/data \
    --out_dir data/processed_multiclass_dicom \
    --structure auto \
    --max_patients 5
```

---

## 📊 Output Structure

Sau khi preprocessing, bạn sẽ có:

```
data/processed_multiclass_dicom/
├── flair/              # PNG images (normalized to 0-255)
│   ├── Patient001_slice0001.png
│   ├── Patient001_slice0002.png
│   └── ...
├── t1/                 # PNG images
├── t1ce/               # PNG images
├── t2/                 # PNG images
├── seg/                # 3-class masks (0=bg, 1=TC, 2=ED)
│   ├── Patient001_slice0001.png
│   └── ...
├── all_slices.csv      # Metadata cho tất cả slices
├── train_fold0.csv     # Training set fold 0
├── val_fold0.csv       # Validation set fold 0
├── train_fold1.csv     # Training set fold 1
├── val_fold1.csv       # Validation set fold 1
├── ...
├── labels.csv          # Patient-level labels
├── mapping.csv         # Slice-to-patient mapping
└── class_mapping.json  # Class definitions
```

---

## 🔧 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dicom_dir` | (required) | Đường dẫn đến thư mục DICOM |
| `--out_dir` | `data/processed_multiclass_dicom` | Thư mục output |
| `--structure` | `auto` | Cấu trúc DICOM: auto, brats, patient, flat |
| `--img_size` | 256 | Kích thước ảnh output (square) |
| `--num_folds` | 5 | Số folds cho cross-validation |
| `--seed` | 42 | Random seed |
| `--max_patients` | None | Giới hạn số patients (testing) |

---

## 📝 DICOM Metadata Detection

### Modality Detection

Script tự động detect modality từ DICOM `SeriesDescription` tag:

| Modality | Keywords in SeriesDescription |
|----------|-------------------------------|
| **FLAIR** | "flair" |
| **T1** | "t1" (không có contrast) |
| **T1CE** | "t1" + ("ce", "gd", "contrast", "c+") |
| **T2** | "t2" |
| **SEG** | "seg", "label", "mask" |

**Ví dụ:**
- `"FLAIR_3D"` → FLAIR
- `"T1_pre_contrast"` → T1
- `"T1_post_gd"` → T1CE
- `"T2_SPACE"` → T2
- `"Segmentation_Mask"` → SEG

### Segmentation Format Support

Hỗ trợ nhiều format segmentation:

1. **BraTS-style**: `{0, 1, 2, 4}`
   - 0 = Background
   - 1 = NCR/NET → maps to TC (class 1)
   - 2 = Edema → maps to ED (class 2)
   - 4 = Enhancing Tumor → maps to TC (class 1)

2. **Binary**: `{0, 1}`
   - 0 = Background
   - 1 = Tumor → maps to TC (class 1)

3. **Multi-class**: `{0, 1, 2, ...}`
   - Clipped to `{0, 1, 2}`

---

## 🎯 3-Class Output Format

Output segmentation masks có 3 classes:

| Class | Value | Description | Color |
|-------|-------|-------------|-------|
| Background | 0 | Non-tumor tissue | Black |
| Tumor Core (TC) | 1 | Core tumor region | Red |
| Edema (ED) | 2 | Peritumoral edema | Green |

### BraTS Standard Regions

- **WT (Whole Tumor)** = TC + ED (classes 1 & 2)
- **TC (Tumor Core)** = Class 1 only
- **ED (Edema)** = Class 2 only

---

## 🔍 Data Validation

### Kiểm Tra Output

```bash
# Check số lượng files
ls data/processed_multiclass_dicom/flair/ | wc -l

# Check class distribution trong mask
python -c "
from PIL import Image
import numpy as np
mask = np.array(Image.open('data/processed_multiclass_dicom/seg/Patient001_slice0050.png'))
print('Classes:', np.unique(mask))
print('Pixel counts:', np.bincount(mask.flatten()))
"
```

Expected output:
```
Classes: [0 1 2]
Pixel counts: [60000, 3000, 2000]  # Example: bg, TC, ED
```

### Kiểm Tra Metadata

```bash
# View all_slices.csv
head -20 data/processed_multiclass_dicom/all_slices.csv

# View statistics
python -c "
import pandas as pd
df = pd.read_csv('data/processed_multiclass_dicom/all_slices.csv')
print('Total slices:', len(df))
print('Patients:', df['patient_id'].nunique())
print('\nLabel distribution:')
print(df['label'].value_counts())
print('\nTumor statistics:')
print(f\"WT: {df['has_wt'].sum()} ({df['has_wt'].mean()*100:.1f}%)\")
print(f\"TC: {df['has_tc'].sum()} ({df['has_tc'].mean()*100:.1f}%)\")
print(f\"ED: {df['has_ed'].sum()} ({df['has_ed'].mean()*100:.1f}%)\")
"
```

---

## ⚙️ Training với DICOM Data

Sau khi preprocessing xong, training hoàn toàn giống H5 format:

```bash
# Update config to point to DICOM processed data
# Edit configs/phase2_small.yaml:
# data:
#   proc_root: "data/processed_multiclass_dicom"

# Train
python scripts/train.py --cfg configs/phase2_small.yaml --fold 0

# Monitor
tensorboard --logdir runs/

# Evaluate
python scripts/evaluate.py \
    --cfg configs/phase2_small.yaml \
    --ckpt checkpoints/braintumnet_best_fold0.pth \
    --fold 0
```

---

## ❗ Troubleshooting

### 1. "No DICOM files found"

**Nguyên nhân:**
- Sai đường dẫn
- Files không có extension `.dcm`
- Files không phải DICOM format

**Giải pháp:**
```bash
# Kiểm tra files
ls /path/to/dicom/data

# Test đọc DICOM
python -c "import pydicom; ds = pydicom.dcmread('file.dcm'); print(ds)"

# Script tự động tìm files không có extension
# nên nếu vẫn không thấy, kiểm tra format
```

### 2. "Missing modality"

**Nguyên nhân:**
- Thiếu 1 trong 4 modalities (FLAIR, T1, T1CE, T2)
- SeriesDescription không chứa keyword

**Giải pháp:**
```bash
# Check SeriesDescription của tất cả files
python -c "
import pydicom
from pathlib import Path

for f in Path('/path/to/dicom/').rglob('*.dcm'):
    ds = pydicom.dcmread(str(f), stop_before_pixels=True)
    print(f.name, '→', ds.SeriesDescription)
"

# Nếu SeriesDescription không chuẩn, có thể:
# 1. Rename series trong DICOM viewer
# 2. Sửa detect_modality_from_series() function
```

### 3. "No common slices found"

**Nguyên nhân:**
- Các modalities có số slice khác nhau
- InstanceNumber không match

**Giải pháp:**
- Đảm bảo tất cả modalities có cùng số slices
- Check InstanceNumber hoặc SliceLocation

### 4. "Error reading DICOM"

**Nguyên nhân:**
- DICOM file bị corrupt
- Thiếu pixel data

**Giải pháp:**
```bash
# Test read individual file
python -c "
import pydicom
ds = pydicom.dcmread('problematic_file.dcm')
print('Shape:', ds.pixel_array.shape)
"
```

### 5. Empty masks (all zeros)

**Nguyên nhân:**
- Không có segmentation files
- Segmentation files không match với images

**Giải pháp:**
- Nếu không có segmentation, masks sẽ là all zeros (class 0 = background)
- Điều này OK cho inference, nhưng không thể train
- Cần có ít nhất một số cases có segmentation

---

## 💡 Best Practices

### 1. Kiểm Tra Data Trước Khi Preprocessing

```bash
# Count patients
ls /path/to/dicom/data | wc -l

# Check structure
tree -L 2 /path/to/dicom/data | head -50

# Test với ít patients trước
--max_patients 5
```

### 2. Verify Output Quality

```bash
# Visualize một vài samples
python scripts/visualize_batch.py \
    --cfg configs/phase2_small.yaml \
    --fold 0 \
    --n 8
```

### 3. Check Fold Splits

```bash
# Verify patient-level splitting
python -c "
import pandas as pd

for fold in range(5):
    train = pd.read_csv(f'data/processed_multiclass_dicom/train_fold{fold}.csv')
    val = pd.read_csv(f'data/processed_multiclass_dicom/val_fold{fold}.csv')

    train_pts = set(train['patient_id'])
    val_pts = set(val['patient_id'])

    assert len(train_pts & val_pts) == 0, f'Fold {fold}: Patient leak detected!'
    print(f'Fold {fold}: OK ({len(train_pts)} train, {len(val_pts)} val)')
"
```

---

## 📚 So Sánh: DICOM vs H5

| Aspect | H5 Format | DICOM Format |
|--------|-----------|--------------|
| **Speed** | Nhanh hơn (batch processing) | Chậm hơn (file-by-file) |
| **Standard** | Research only | Medical standard |
| **Metadata** | Limited | Rich (patient info, acquisition params) |
| **Size** | Compact | Lớn hơn |
| **Preprocessing** | ~10-15 phút | ~20-30 phút |
| **Compatibility** | Kaggle/BraTS | Hospital PACS systems |

**Khuyến nghị:**
- Research/Development: Dùng H5 (nhanh hơn)
- Production/Clinical: Dùng DICOM (standard)
- Both work với cùng training pipeline!

---

## 🔗 Related Documentation

- [QUICKSTART.md](quickstart/QUICKSTART.md) - Quick start guide
- [README.md](../README.md) - Main documentation
- [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) - Model architecture

---

## 📧 Support

Nếu gặp vấn đề:

1. Check [Troubleshooting section](#-troubleshooting)
2. Verify DICOM files với `pydicom`
3. Test với `--max_patients 1` first
4. Check console output và error messages

---

**Happy Preprocessing! 🧠📊**
