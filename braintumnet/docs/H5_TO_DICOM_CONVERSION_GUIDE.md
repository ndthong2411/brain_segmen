# H5 to DICOM Conversion Guide - BrainTumNet

**Version**: 1.0
**Last Updated**: 2025-10-29

---

## 📋 Tổng Quan

Script `convert_h5_to_dicom.py` cho phép chuyển đổi file H5 format (BraTS 2020) sang DICOM format chuẩn y tế.

### Lý Do Cần Chuyển Đổi

✅ **Clinical Integration**
- DICOM là format chuẩn trong bệnh viện
- Tương thích với PACS (Picture Archiving and Communication System)
- Có thể xem trên mọi DICOM viewer

✅ **Interoperability**
- Tích hợp với hệ thống y tế
- Chia sẻ dữ liệu giữa các tổ chức
- Standard metadata format

✅ **Research to Clinical Translation**
- Chuyển model từ research sang clinical use
- Test trên real clinical workflows
- Compliance với medical standards

---

## 🚀 Quick Start

### 1. Cài Đặt Dependencies

```bash
cd braintumnet
pip install pydicom

# Verify installation
python -c "import pydicom; print('pydicom version:', pydicom.__version__)"
```

### 2. Chuyển Đổi H5 sang DICOM

#### **Basic Usage**

```bash
python scripts/preprocessing/convert_h5_to_dicom.py \
    --h5_dir "E:\data\brats2020\h5_files" \
    --out_dir "E:\data\brats2020_dicom"
```

#### **With Custom Metadata**

```bash
python scripts/preprocessing/convert_h5_to_dicom.py \
    --h5_dir "E:\data\brats2020\h5_files" \
    --out_dir "E:\data\brats2020_dicom" \
    --patient_prefix "BraTS20_" \
    --study_description "Brain Tumor MRI Study 2020" \
    --institution "Medical Research Center"
```

#### **Test với Ít Files**

```bash
python scripts/preprocessing/convert_h5_to_dicom.py \
    --h5_dir "E:\data\brats2020\h5_files" \
    --out_dir "E:\data\test_dicom" \
    --max_files 10
```

---

## 📊 Output Structure

### DICOM Directory Organization

```
dicom_output/
├── Patient001_vol1/          # Một patient (volume)
│   ├── FLAIR/
│   │   ├── IM0001.dcm        # Slice 1
│   │   ├── IM0002.dcm        # Slice 2
│   │   └── ...
│   ├── T1/
│   │   ├── IM0001.dcm
│   │   └── ...
│   ├── T1CE/
│   │   ├── IM0001.dcm
│   │   └── ...
│   ├── T2/
│   │   ├── IM0001.dcm
│   │   └── ...
│   └── SEG/                  # Segmentation masks
│       ├── SEG0001.dcm
│       └── ...
├── Patient002_vol2/
├── Patient003_vol3/
└── ...
```

### DICOM Hierarchy

```
Study (Patient)
  └── Series (Modality: FLAIR, T1, T1CE, T2, SEG)
       └── Instance (Slice)
```

---

## 🔧 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--h5_dir` | (required) | Thư mục chứa H5 files |
| `--out_dir` | `data/dicom_output` | Thư mục output DICOM |
| `--patient_prefix` | `BraTS20_` | Prefix cho Patient ID |
| `--study_description` | `Brain Tumor MRI Study` | Mô tả study |
| `--institution` | `BrainTumNet Research` | Tên tổ chức |
| `--max_files` | None | Giới hạn số files (testing) |

---

## 📝 DICOM Metadata

### Generated Metadata

Script tự động tạo các DICOM tags chuẩn:

#### **Patient Level**
- `PatientID`: Patient001_vol1
- `PatientName`: Patient_Patient001_vol1
- `PatientBirthDate`: (empty - privacy)
- `PatientSex`: (empty - privacy)

#### **Study Level**
- `StudyInstanceUID`: Unique UID (same for all series)
- `StudyID`: Patient001
- `StudyDate`: Current date (YYYYMMDD)
- `StudyTime`: Current time (HHMMSS)
- `StudyDescription`: "Brain Tumor MRI Study"
- `AccessionNumber`: (empty)

#### **Series Level**
- `SeriesInstanceUID`: Unique UID (per modality)
- `SeriesNumber`:
  - 1 = FLAIR
  - 2 = T1
  - 3 = T1CE
  - 4 = T2
  - 5 = SEG
- `SeriesDescription`:
  - "FLAIR - Fluid Attenuated Inversion Recovery"
  - "T1 - Pre-Contrast"
  - "T1CE - Post-Contrast (Gd)"
  - "T2 - Weighted"
  - "Segmentation Mask"
- `Modality`: "MR" (hoặc "SEG" cho segmentation)

#### **Instance Level**
- `SOPInstanceUID`: Unique UID (per slice)
- `InstanceNumber`: Slice number (1, 2, 3, ...)
- `ImageType`: ['DERIVED', 'PRIMARY', 'AXIAL']

#### **Image Information**
- `Rows`, `Columns`: Image dimensions (e.g., 240x240)
- `BitsAllocated`: 16
- `BitsStored`: 16
- `PixelRepresentation`: 0 (unsigned)
- `PhotometricInterpretation`: MONOCHROME2
- `PixelSpacing`: [1.0, 1.0] mm
- `SliceThickness`: 1.0 mm
- `SliceLocation`: Varies by slice

#### **Rescale Parameters**
- `RescaleIntercept`: Original minimum value
- `RescaleSlope`: Scale factor
- Used to convert uint16 back to original float values

---

## 🔍 Verification

### 1. Check DICOM Files Created

```bash
# Count DICOM files
find E:\data\brats2020_dicom -name "*.dcm" | wc -l

# List structure
tree E:\data\brats2020_dicom -L 3
```

### 2. Read DICOM Metadata

```python
import pydicom

# Read a DICOM file
ds = pydicom.dcmread('dicom_output/Patient001_vol1/FLAIR/IM0001.dcm')

# Print metadata
print("Patient ID:", ds.PatientID)
print("Study Description:", ds.StudyDescription)
print("Series Description:", ds.SeriesDescription)
print("Image shape:", ds.pixel_array.shape)
print("Modality:", ds.Modality)
print("\nFull metadata:")
print(ds)
```

### 3. View in DICOM Viewer

**Recommended DICOM Viewers:**

1. **MicroDicom** (Windows) - Free
   - Download: http://www.microdicom.com
   - Open → Select patient folder

2. **RadiAnt** (Windows) - Free trial
   - Download: https://www.radiantviewer.com
   - Professional features

3. **Horos** (Mac) - Free
   - Download: https://horosproject.org
   - Full-featured medical viewer

4. **3D Slicer** (Cross-platform) - Free
   - Download: https://www.slicer.org
   - For 3D visualization and analysis

5. **OHIF Viewer** (Web-based) - Free
   - https://viewer.ohif.org
   - Upload DICOM files

### 4. Compare with Original H5

```python
import h5py
import pydicom
import numpy as np

# Load H5
with h5py.File('volume_1_slice_50.h5', 'r') as f:
    h5_image = f['image'][:, :, 0]  # FLAIR

# Load DICOM
ds = pydicom.dcmread('dicom_output/Patient001_vol1/FLAIR/IM0050.dcm')
dicom_image = ds.pixel_array.astype(np.float32)

# Apply rescale
dicom_image = dicom_image * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

# Compare
print("H5 shape:", h5_image.shape)
print("DICOM shape:", dicom_image.shape)
print("H5 range:", h5_image.min(), "-", h5_image.max())
print("DICOM range:", dicom_image.min(), "-", dicom_image.max())
```

---

## 🔄 Round-Trip Workflow

### Complete Pipeline: H5 → DICOM → PNG → Training

```bash
# Step 1: Convert H5 to DICOM
python scripts/preprocessing/convert_h5_to_dicom.py \
    --h5_dir "E:\data\brats2020\h5_files" \
    --out_dir "E:\data\brats2020_dicom"

# Step 2: Convert DICOM to PNG for training
python scripts/preprocess_dicom_to_multiclass.py \
    --dicom_dir "E:\data\brats2020_dicom" \
    --out_dir "data/processed_from_dicom" \
    --structure brats

# Step 3: Train model
python scripts/train.py --cfg configs/phases/phase2_small.yaml --fold 0

# Step 4: Inference (có thể xuất kết quả ra DICOM)
# TODO: Thêm script export predictions to DICOM
```

---

## 📊 Segmentation Encoding

### H5 Format (Input)
```
mask: (H, W, 3) numpy array
- Channel 0: Unknown (not used)
- Channel 1: Tumor Core (TC) - binary
- Channel 2: Edema (ED) - binary
```

### DICOM SEG Format (Output)
```
pixel_array: (H, W) uint16
- 0: Background
- 1: Tumor Core (from channel 1)
- 2: Edema (from channel 2)
```

### Decoding in DICOM Viewer

Trong DICOM viewer, bạn sẽ thấy:
- **Grayscale values**: 0, 1, 2
- **To visualize as colors**: Use LUT (Look-Up Table)
  - 0 → Black (Background)
  - 1 → Red (Tumor Core)
  - 2 → Green (Edema)

---

## ⚠️ Important Notes

### 1. Privacy & HIPAA

**Generated DICOM files không chứa thông tin nhận dạng thực:**
- PatientName: Generic (Patient_001, Patient_002, ...)
- PatientID: Generic
- PatientBirthDate: Empty
- PatientSex: Empty

**Nếu sử dụng với real patient data:**
- Phải anonymize properly
- Tuân thủ HIPAA/GDPR
- Use DICOM anonymization tools

### 2. UID Generation

**UIDs được tạo tự động và là unique:**
- StudyInstanceUID: Unique per patient
- SeriesInstanceUID: Unique per modality/series
- SOPInstanceUID: Unique per slice

**Regenerating DICOMs:**
- Mỗi lần chạy script sẽ tạo UIDs mới
- Không dùng để update existing studies trong PACS

### 3. Pixel Data Conversion

**H5 → DICOM conversion:**
- Original: float32 (any range)
- DICOM: uint16 (0-65535)
- Uses RescaleSlope & RescaleIntercept to preserve original values

**To get original values back:**
```python
original_value = pixel_value * RescaleSlope + RescaleIntercept
```

### 4. File Size

**DICOM files lớn hơn H5:**
- H5 file: ~10-50 KB per slice
- DICOM file: ~100-200 KB per slice
- Reason: Metadata + header overhead

**Example:**
- 57,195 H5 files (~1 GB) → ~11 GB DICOM files

---

## 🔧 Troubleshooting

### 1. "Module not found: pydicom"

```bash
pip install pydicom
```

### 2. "Error loading H5 file"

Check H5 file structure:
```python
import h5py

with h5py.File('file.h5', 'r') as f:
    print("Keys:", list(f.keys()))
    print("Image shape:", f['image'].shape)
    print("Mask shape:", f['mask'].shape)
```

Expected:
- Keys: ['image', 'mask']
- Image shape: (H, W, 4)
- Mask shape: (H, W, 3)

### 3. "DICOM files cannot be opened"

Verify DICOM format:
```python
import pydicom

ds = pydicom.dcmread('file.dcm')
print(ds)
```

### 4. "Wrong pixel values in DICOM"

Apply rescale:
```python
import pydicom

ds = pydicom.dcmread('file.dcm')
pixel_array = ds.pixel_array.astype(float)
pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
```

---

## 💡 Use Cases

### 1. Clinical Deployment

```bash
# Convert research data to clinical format
python scripts/preprocessing/convert_h5_to_dicom.py \
    --h5_dir "research_data/" \
    --out_dir "clinical_dicom/" \
    --institution "Your Hospital" \
    --study_description "Brain Tumor Analysis"

# Upload to PACS system (use DICOM send tools)
```

### 2. Data Sharing

```bash
# Create shareable DICOM datasets
python scripts/preprocessing/convert_h5_to_dicom.py \
    --h5_dir "data/" \
    --out_dir "shared_dicom/" \
    --patient_prefix "ANONYMIZED_"

# Compress and share
zip -r shared_dicom.zip shared_dicom/
```

### 3. Validation & Testing

```bash
# Test model on DICOM format
# 1. Convert H5 to DICOM
python scripts/preprocessing/convert_h5_to_dicom.py \
    --h5_dir "test_data/" \
    --out_dir "test_dicom/" \
    --max_files 10

# 2. Convert back to PNG
python scripts/preprocess_dicom_to_multiclass.py \
    --dicom_dir "test_dicom/" \
    --out_dir "test_png/"

# 3. Run inference
python scripts/predict.py \
    --cfg configs/phases/phase2_small.yaml \
    --ckpt checkpoints/best.pth \
    --img "test_png/flair/sample.png"
```

---

## 📚 Related Documentation

- [DICOM_PREPROCESSING_GUIDE.md](DICOM_PREPROCESSING_GUIDE.md) - DICOM to PNG
- [QUICKSTART.md](quickstart/QUICKSTART.md) - Training pipeline
- [README.md](../README.md) - Main documentation

---

## 🔗 External Resources

### DICOM Standards
- Official DICOM Standard: https://www.dicomstandard.org
- DICOM Library (pydicom): https://pydicom.github.io

### DICOM Viewers
- MicroDicom: http://www.microdicom.com
- RadiAnt: https://www.radiantviewer.com
- Horos (Mac): https://horosproject.org
- 3D Slicer: https://www.slicer.org

### DICOM Tools
- DCMTK (DICOM Toolkit): https://dicom.offis.de/dcmtk
- dcm4che (Java): https://www.dcm4che.org
- DICOM Anonymizer: https://www.dicomlibrary.com

---

## ✅ Summary

| Feature | Status | Notes |
|---------|--------|-------|
| H5 to DICOM conversion | ✅ | All 4 modalities + segmentation |
| DICOM metadata | ✅ | Standard tags, privacy-safe |
| UID generation | ✅ | Unique per study/series/instance |
| Pixel data preservation | ✅ | With rescale parameters |
| PACS compatibility | ✅ | Standard DICOM format |
| Round-trip support | ✅ | H5 → DICOM → PNG → Training |

---

**Happy Converting! 🔄🏥**
