# BraTS2020 Preprocessing Guide

## Overview

This guide explains how to preprocess the BraTS2020 dataset for training the BrainTumNet model.

---

## 🚨 IMPORTANT: Current Data Issue

**Your current dataset has ONLY 13 patients (2000 slices)**

This is **NOT enough** for good deep learning performance! You need:
- **Minimum recommended:** 50-100 patients (~8,000-15,000 slices)
- **BraTS2020 full dataset:** ~370 patients (~60,000 slices)
- **Current performance:** IoU stuck at 0.44 due to overfitting on small dataset
- **Expected with full data:** IoU 0.60-0.70

---

## Preprocessing Steps

### Step 1: Get BraTS2020 Dataset

Download the full BraTS2020 dataset from:
- Official website: http://braintumorsegmentation.org/
- Or Kaggle: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data

You should have:
```
data/raw/
├── meta_data.csv          (metadata for all slices)
├── BraTS20_001.h5         (patient 001 - 3D MRI)
├── BraTS20_002.h5         (patient 002)
├── BraTS20_003.h5
└── ... (370 .h5 files total)
```

### Step 2: Run Preprocessing Script

#### **Option A: Process Full Dataset (RECOMMENDED)**

```bash
# Single modality (T1CE - best tumor contrast)
python scripts/prepare_brats2020_h5.py \
  --h5_root data/raw \
  --meta_csv data/raw/meta_data.csv \
  --out data/processed_full

# Expected output:
# - 30,000-60,000 slices
# - 200-370 cases
# - ~3-5 GB disk space
```

#### **Option B: Multimodal (All 4 MRI Sequences)**

```bash
# Process FLAIR, T1, T1CE, T2 together (4-channel)
python scripts/prepare_brats2020_h5.py \
  --h5_root data/raw \
  --meta_csv data/raw/meta_data.csv \
  --out data/processed_multimodal \
  --multimodal

# Advantages:
# - Better performance (+5-10% Dice)
# - More information for model
# - Publication-quality results

# Disadvantages:
# - 4× larger files (12-20 GB)
# - Slower training
```

#### **Option C: Quick Test (Development)**

```bash
# Process only 1000 slices for quick testing
python scripts/prepare_brats2020_h5.py \
  --h5_root data/raw \
  --meta_csv data/raw/meta_data.csv \
  --out data/test \
  --max_slices 1000

# Use this ONLY for:
# - Testing code changes
# - Debugging preprocessing
# - Quick experiments
```

---

## What the Script Does

### 1. **Read HDF5 Files**
Each .h5 file contains:
- `image`: (240, 240, 155, 4) - 4 MRI modalities × 155 slices
  - Channel 0: FLAIR (good for edema)
  - Channel 1: T1 (anatomy)
  - Channel 2: T1CE (best for tumors) ✓
  - Channel 3: T2 (good for fluid)
- `mask`: (240, 240, 155, 3) - 3 tumor regions
  - ET: Enhancing Tumor
  - TC: Tumor Core
  - WT: Whole Tumor

### 2. **Normalize Intensities**
MRI scanners produce different intensity ranges. Normalization makes them comparable:

```python
# For each slice:
img = (img - min) / (max - min)  # Scale to [0, 1]
```

### 3. **Resize to 256×256**
- Pad to square if needed (240×240 already square)
- Resize with bilinear interpolation (smooth)
- Mask resized with nearest neighbor (sharp edges)

### 4. **Filter Empty Slices**
Skip slices with very little tumor:
- Default threshold: 0.001 (0.1% of pixels must be tumor)
- Configurable with `--min_tumor_ratio`

### 5. **Create Train/Val Splits**
- 5-fold cross-validation
- Stratified by HGG/LGG labels
- Patient-level splitting (no data leakage)

---

## Output Structure

```
data/processed/
├── images/                      # MRI slices
│   ├── vol001_slice050.png      # Single-modal (grayscale)
│   ├── vol001_slice051.npy      # Multi-modal (4-channel)
│   └── ...
│
├── masks/                       # Tumor masks
│   ├── vol001_slice050.png      # Binary: 0=bg, 255=tumor
│   └── ...
│
├── labels.csv                   # Case-level labels
│   case_id,label
│   vol001,1                     # 1=HGG, 0=LGG
│   vol002,0
│
├── mapping.csv                  # Slice→Case mapping
│   slice_id,case_id
│   vol001_slice050,vol001
│   vol001_slice051,vol001
│
└── Cross-validation splits:
    ├── split_train_fold0.txt    # Training slice IDs
    ├── split_val_fold0.txt      # Validation slice IDs
    ├── split_train_fold1.txt
    └── ... (10 files total)
```

---

## Preprocessing Output Example

When you run the script, you'll see detailed statistics:

```
======================================================================
BraTS2020 PREPROCESSING CONFIGURATION
======================================================================
Input directory:     data/raw
Metadata CSV:        data/raw/meta_data.csv
Output directory:    data/processed_full
Image size:          256×256
Min tumor ratio:     0.0010 (0.10%)
Mode:                Single-modal (T1CE only)
Slice limit:         None (processing ALL data ✓)
======================================================================

Found 57420 slices in metadata
✓ Processing ALL slices (full dataset mode)

Processing slices: 100%|████████████████| 57420/57420 [45:32<00:00, 21.01it/s]

======================================================================
PREPROCESSING SUMMARY
======================================================================
Total slices in metadata:    57420
✓ Successfully processed:    51234
✗ Skipped (no tumor):        3852
✗Skipped (errors):          12
✗ Skipped (file not found):  2322
Total skipped:               6186
----------------------------------------------------------------------
Unique cases:                369
Total tumor pixels:          1,234,567,890
Average tumor pixels/slice:  24,097
======================================================================

======================================================================
CROSS-VALIDATION SPLITS
======================================================================
Number of folds:       5
Total cases:           369
  • LGG (class 0):     76 cases
  • HGG (class 1):     293 cases
----------------------------------------------------------------------
Fold 0: 295 train cases (40987 slices) |  74 val cases (10247 slices)
Fold 1: 295 train cases (40982 slices) |  74 val cases (10252 slices)
Fold 2: 295 train cases (40995 slices) |  74 val cases (10239 slices)
Fold 3: 295 train cases (41001 slices) |  74 val cases (10233 slices)
Fold 4: 296 train cases (41025 slices) |  73 val cases (10209 slices)
======================================================================
```

---

## Common Issues

### ❌ "Only 13 cases processed"

**Problem:** You only have partial BraTS2020 data

**Solution:** Download full BraTS2020 dataset (~370 patients)

### ❌ "File not found errors"

**Problem:** meta_data.csv references files that don't exist

**Solution:**
1. Check paths in meta_data.csv match actual .h5 filenames
2. Ensure all .h5 files are in `--h5_root` directory

### ❌ "Processing too slow"

**Problem:** Processing 60,000 slices takes 30-60 minutes

**Solutions:**
- Use SSD instead of HDD
- Use `--max_slices 1000` for quick testing
- Run once, reuse processed data

### ❌ "All cases are LGG (class 0)"

**Problem:** Your subset only contains low-grade gliomas

**Solution:** Get full BraTS2020 with both HGG and LGG cases

---

## Advanced Options

### Adjust Tumor Filtering

```bash
# Keep only slices with >1% tumor pixels
python prepare_brats2020_h5.py ... --min_tumor_ratio 0.01

# Keep all slices (even blank ones)
python prepare_brats2020_h5.py ... --min_tumor_ratio 0
```

### Custom Image Size

```bash
# Use 512×512 (more detail, slower training)
python prepare_brats2020_h5.py ... --img_size 512

# Use 128×128 (faster, less detail)
python prepare_brats2020_h5.py ... --img_size 128
```

### Different Modality

```bash
# Use FLAIR instead of T1CE
python prepare_brats2020_h5.py ... --modality flair

# Options: flair, t1, t1ce (default), t2
```

---

## Next Steps

After preprocessing:

1. **Verify data:**
   ```bash
   # Check number of files
   ls data/processed_full/images/ | wc -l

   # Should see: 30,000+ files
   ```

2. **Train model:**
   ```bash
   python scripts/train.py --cfg configs/optimized.yaml --fold 0
   ```

3. **Monitor training:**
   ```bash
   tensorboard --logdir=runs
   ```

---

## Performance Expectations

| Dataset Size | Expected IoU | Expected Dice | Training Time |
|--------------|--------------|---------------|---------------|
| 13 cases (current) | 0.40-0.46 | 0.57-0.63 | 3 hours |
| 100 cases | 0.55-0.62 | 0.71-0.76 | 8 hours |
| 370 cases (full) | 0.60-0.70 | 0.75-0.82 | 24 hours |
| 370 cases (multimodal) | 0.65-0.75 | 0.79-0.86 | 36 hours |

---

## Questions?

- Check existing processed data: `ls -lh data/processed/`
- Run with `--help`: `python scripts/prepare_brats2020_h5.py --help`
- Test with small subset: Use `--max_slices 100`
