# Multi-Class Data Preprocessing Summary

**Date**: 2025-10-10
**Status**: ✅ Running
**Purpose**: Convert BraTS H5 data to 3-class PNG format

---

## What Was Done

### 1. Created Preprocessing Script
- **File**: `scripts/preprocess_h5_to_multiclass.py`
- **Input**: 57,195 H5 files from BraTS 2020 dataset
- **Output**: PNG images + 3-class masks

### 2. Label Mapping

#### Original H5 Format
- `image`: (240, 240, 4) - 4 modalities (FLAIR, T1, T1CE, T2)
- `mask`: (240, 240, 3) - 3 binary channels

#### Target 3-Class Format
```
Class 0: Background
Class 1: Tumor Core (TC) - from H5 channel 1
Class 2: Edema (ED) - from H5 channel 2
```

#### Evaluation Regions
```
Whole Tumor (WT) = TC + ED (classes 1 + 2)
Tumor Core (TC)  = class 1
Edema (ED)       = class 2
```

### 3. Processing Pipeline

1. **Load H5 file** → image (240×240×4) + mask (240×240×3)
2. **Convert 3-channel mask → 3-class single-channel mask**
   - Channel 2 (Edema) → class 2
   - Channel 1 (Tumor Core) → class 1 (priority over edema)
   - Background → class 0
3. **Normalize each modality** to [0, 255] using 1st-99th percentile
4. **Resize** to 256×256
   - Images: Bilinear interpolation
   - Masks: Nearest neighbor (preserve labels)
5. **Save as PNG** files
6. **Generate metadata** CSV with slice info

### 4. Output Structure

```
data/processed_multiclass/
├── flair/          # FLAIR modality images
│   ├── vol1_slice0.png
│   ├── vol1_slice1.png
│   └── ...
├── t1/             # T1 modality images
├── t1ce/           # T1CE modality images
├── t2/             # T2 modality images
├── seg/            # 3-class segmentation masks
│   ├── vol1_slice0.png  # Values: {0, 1, 2}
│   └── ...
├── all_slices.csv          # All slices metadata
├── train_fold0.csv         # Training set fold 0
├── val_fold0.csv           # Validation set fold 0
├── train_fold1.csv
├── val_fold1.csv
├── ...
└── class_mapping.json      # Label mapping documentation
```

### 5. K-Fold Splits

- **Number of folds**: 5
- **Split level**: Volume level (not slice level)
- **Purpose**: Prevent data leakage - all slices from same volume stay together

---

## Expected Results

### Dataset Statistics (Estimated)

Based on 57,195 H5 files:

```
Total slices: 57,195
Total volumes: ~370 volumes

Label distribution (estimated):
  Normal: ~28,000 slices (49%)
  WT:     ~29,000 slices (51%)
  TC:     ~22,000 slices (38%)
  ED:     ~20,000 slices (35%)

Per fold:
  Train: ~45,756 slices from ~296 volumes
  Val:   ~11,439 slices from ~74 volumes
```

### Metrics Comparison: Binary vs Multi-Class

#### Binary Segmentation (Old - `data/processed`)
```
Overall Dice: 0.91  ← INFLATED by 97% background
Overall IoU:  0.84
```
**Problem**: Background dominates metrics, making them unreliable.

#### Multi-Class Segmentation (New - `data/processed_multiclass`)
```
WT Dice: 0.88-0.90  ← Whole Tumor (honest metric)
WT IoU:  0.80-0.82

TC Dice: 0.82-0.85  ← Tumor Core (harder)
TC IoU:  0.70-0.74

ED Dice: 0.75-0.80  ← Edema (hardest)
ED IoU:  0.60-0.67

Mean Dice: 0.82-0.85  ← Average of WT, TC, ED
Mean IoU:  0.70-0.74
```
**Advantage**: Metrics computed ONLY on tumor regions, no background inflation.

---

## Training with Multi-Class Data

### Config Files Created

1. **`configs/multiclass.yaml`** - Baseline config
   - Batch size: 12 (RTX 3090 compatible)
   - Model: base=32, dim=256
   - Loss: Combined Dice + Focal
   - Expected: WT Dice 0.88-0.90

2. **`configs/multiclass_a100.yaml`** - A100 optimized
   - Batch size: 64 (A100 40GB)
   - Model: base=48, dim=384, depth=3
   - Class weights: [1.0, 2.0, 1.5] to emphasize tumor
   - Expected: WT Dice 0.90-0.92, TC Dice 0.85-0.88

### Training Commands

```bash
# 1. Baseline training (RTX 3090 / A100)
python scripts/train.py --cfg configs/multiclass.yaml --fold 0

# 2. A100 optimized training
python scripts/train.py --cfg configs/multiclass_a100.yaml --fold 0

# 3. 5-Fold Cross-Validation (if multiple GPUs)
for fold in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$fold python scripts/train.py \
    --cfg configs/multiclass_a100.yaml --fold $fold &
done
```

### Expected Training Time

**RTX 3090** (multiclass.yaml):
- Time per epoch: ~8-10 seconds
- Total time (250 epochs): ~35-40 minutes

**A100 40GB** (multiclass_a100.yaml):
- Time per epoch: ~2-3 seconds
- Total time (300 epochs): ~15-20 minutes

---

## Why Multi-Class is Better

### Problem with Binary Segmentation

1. **Background Dominance**: 97% background, 3% tumor
2. **Inflated Metrics**: Dice 0.91 looks good, but actually poor tumor segmentation
3. **Model Bias**: Model learns to predict background well, tumor boundaries poorly
4. **Misleading Results**: High scores don't reflect actual performance

### Solution with Multi-Class

1. **Separate Evaluation**: Metrics for WT, TC, ED computed independently
2. **No Background**: Background excluded from metric calculation
3. **Honest Assessment**: Lower but more accurate Dice scores
4. **Medical Standard**: Aligns with BraTS challenge evaluation protocol
5. **Better Analysis**: Can identify which tumor regions need improvement

### Example Scenario

**Binary Model**:
- Predicts 99% background correctly → Dice 0.91
- Predicts 50% tumor correctly → Hidden by background
- **Result**: Looks good, but fails on actual tumor

**Multi-Class Model**:
- WT Dice 0.85 → Directly shows tumor segmentation quality
- TC Dice 0.80 → Shows tumor core performance
- ED Dice 0.75 → Identifies edema as hardest region
- **Result**: Clear understanding of strengths/weaknesses

---

## Files Created

### Scripts
- `scripts/preprocess_h5_to_multiclass.py` - Main preprocessing script
- `scripts/check_h5_format.py` - H5 format inspection
- `scripts/check_h5_tumor.py` - Find H5 files with tumor

### Configs
- `configs/multiclass.yaml` - Baseline multi-class config
- `configs/multiclass_a100.yaml` - A100 optimized config

### Documentation
- `docs/25_10_10_MULTICLASS_PREPROCESSING.md` - Detailed preprocessing guide
- `docs/25_10_10_PREPROCESSING_SUMMARY.md` - This summary
- `docs/25_10_10_MULTICLASS_SEGMENTATION_GUIDE.md` - Multi-class implementation guide (created earlier)

---

## Next Steps

### 1. Wait for Preprocessing to Complete
- Monitor: Check `braintumnet/preprocess_log.txt`
- ETA: ~10-15 minutes from start
- Verify: Check `data/processed_multiclass/all_slices.csv`

### 2. Verify Output
```bash
# Check total slices
wc -l braintumnet/data/processed_multiclass/all_slices.csv

# Check file counts
ls -1 braintumnet/data/processed_multiclass/seg | wc -l
ls -1 braintumnet/data/processed_multiclass/flair | wc -l

# View statistics
head -20 braintumnet/preprocess_log.txt | tail -15
```

### 3. Visualize Samples (Optional)
```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load a sample with tumor
flair = np.array(Image.open('data/processed_multiclass/flair/vol100_slice100.png'))
seg = np.array(Image.open('data/processed_multiclass/seg/vol100_slice100.png'))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(flair, cmap='gray')
axes[0].set_title('FLAIR Image')
axes[1].imshow(seg, cmap='jet', vmin=0, vmax=2)
axes[1].set_title('3-Class Mask\n0=BG, 1=TC, 2=ED')
plt.savefig('sample_multiclass.png', dpi=150, bbox_inches='tight')
```

### 4. Start Training
```bash
# Train multi-class model
python scripts/train.py --cfg configs/multiclass_a100.yaml --fold 0
```

### 5. Compare Results

Train both binary and multi-class models, then compare:

| Metric | Binary | Multi-Class (WT) | Multi-Class (TC) | Multi-Class (ED) |
|--------|--------|------------------|------------------|------------------|
| Dice   | 0.91   | 0.90             | 0.86             | 0.80             |
| IoU    | 0.84   | 0.82             | 0.74             | 0.67             |

Binary score is **inflated**, multi-class scores are **honest**.

---

## Troubleshooting

### Issue: Preprocessing is slow
**Solution**: This is expected for 57k files. Use background process and wait.

### Issue: Out of memory during preprocessing
**Solution**: Script processes one file at a time, shouldn't cause OOM. Check system RAM.

### Issue: Some slices are missing
**Solution**: Script skips files with errors. Check `preprocess_log.txt` for error messages.

### Issue: Training fails with "num_classes mismatch"
**Solution**: Ensure config has `num_classes_seg: 3` under `model:` section.

---

## Summary

✅ **Completed**:
- Created H5 → 3-class PNG conversion script
- Configured multi-class training configs
- Started preprocessing on 57,195 H5 files
- Created comprehensive documentation

⏳ **In Progress**:
- Preprocessing 57k H5 files (~10-15 minutes total)

📋 **Next**:
- Verify preprocessed data
- Train multi-class model
- Compare binary vs multi-class results
- Report findings for paper

🎯 **Goal**:
- Honest tumor segmentation metrics without background inflation
- Better understanding of model performance on each tumor region
- Publication-quality results aligned with BraTS standards
