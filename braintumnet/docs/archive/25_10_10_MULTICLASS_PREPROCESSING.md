# Multi-Class Data Preprocessing Guide

**Date**: 2025-10-10
**Purpose**: Convert BraTS 2020 raw data to 3-class format for multi-class segmentation

---

## Overview

This preprocessing script converts BraTS NIfTI files with original labels `{0, 1, 2, 4}` into a 3-class PNG format `{0, 1, 2}` suitable for multi-class segmentation training.

### Why Multi-Class?

**Problem with Binary Segmentation**:
- Binary segmentation: Background (97%) vs Tumor (3%)
- Dice score is inflated by background pixels
- Model learns to predict background well, tumor boundaries poorly
- Result: High Dice (0.91) but poor actual tumor segmentation

**Solution with Multi-Class**:
- 3 classes: Background, Tumor Core (TC), Edema (ED)
- Evaluate tumor regions separately: WT, TC, ED
- No background in metric computation → honest evaluation
- Lower but more accurate Dice scores

---

## Label Mapping

### Original BraTS Labels
```
0: Background
1: NCR (Necrotic and Non-Enhancing Tumor)
2: ED (Peritumoral Edema)
4: ET (Enhancing Tumor)
```

### Target 3-Class Mapping
```
0: Background          (original 0)
1: Tumor Core (TC)     (original 1 + 4)  ← NCR + ET merged
2: Edema (ED)          (original 2)
```

### Evaluation Regions
```
Whole Tumor (WT) = TC + ED  (classes 1 + 2)
Tumor Core (TC)  = class 1 only
Edema (ED)       = class 2 only
```

**Important**: Metrics are computed ONLY on tumor regions (WT, TC, ED), ignoring background entirely.

---

## Usage

### 1. Basic Usage

```bash
python scripts/preprocess_multiclass.py \
    --raw_dir data/raw/BraTS2020 \
    --out_dir data/processed_multiclass \
    --img_size 256 \
    --slices_per_case 30 \
    --tumor_ratio 0.5 \
    --num_folds 5
```

### 2. Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--raw_dir` | str | Required | Path to BraTS raw data directory |
| `--out_dir` | str | `data/processed_multiclass` | Output directory |
| `--img_size` | int | 256 | Target image size (square) |
| `--slices_per_case` | int | 30 | Number of slices per case |
| `--tumor_ratio` | float | 0.5 | Ratio of tumor slices (0.5 = 50% tumor, 50% normal) |
| `--num_folds` | int | 5 | Number of K-fold splits |
| `--seed` | int | 42 | Random seed for reproducibility |

### 3. Expected Input Structure

```
data/raw/BraTS2020/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii.gz
│   ├── BraTS20_Training_001_t1.nii.gz
│   ├── BraTS20_Training_001_t1ce.nii.gz
│   ├── BraTS20_Training_001_t2.nii.gz
│   └── BraTS20_Training_001_seg.nii.gz
├── BraTS20_Training_002/
│   └── ...
└── ...
```

### 4. Output Structure

```
data/processed_multiclass/
├── flair/
│   ├── BraTS20_Training_001_slice050.png
│   ├── BraTS20_Training_001_slice051.png
│   └── ...
├── t1/
│   └── ...
├── t1ce/
│   └── ...
├── t2/
│   └── ...
├── seg/
│   ├── BraTS20_Training_001_slice050.png  ← 3-class labels {0,1,2}
│   └── ...
├── all_slices.csv
├── train_fold0.csv
├── val_fold0.csv
├── train_fold1.csv
├── val_fold1.csv
├── ...
└── class_mapping.json
```

---

## Processing Steps

### Step 1: Load NIfTI Files
- Load all 4 modalities: FLAIR, T1, T1CE, T2
- Load segmentation mask with original BraTS labels {0, 1, 2, 4}

### Step 2: Remap Labels
```python
def remap_labels_to_3class(seg_volume):
    remapped = np.zeros_like(seg_volume, dtype=np.uint8)
    remapped[seg_volume == 0] = 0           # Background
    remapped[(seg_volume == 1) | (seg_volume == 4)] = 1  # TC = NCR + ET
    remapped[seg_volume == 2] = 2           # ED
    return remapped
```

### Step 3: Normalize Modalities
- Compute 1st and 99th percentiles on brain region (non-zero pixels)
- Clip values to [p1, p99] range
- Normalize to [0, 255] uint8 range

### Step 4: Select Slices
- Find slices with tumor (any non-zero label)
- Find slices without tumor (all background)
- Sample `tumor_ratio * slices_per_case` tumor slices (evenly distributed)
- Sample remaining non-tumor slices (evenly distributed)
- Default: 50% tumor, 50% normal slices

### Step 5: Resize and Save
- Resize each slice to `img_size × img_size` (default 256×256)
- Use nearest neighbor for segmentation masks (preserve labels)
- Use bilinear for images (smooth interpolation)
- Save as PNG files

### Step 6: Create K-Fold Splits
- Split at **case level** (not slice level) to prevent data leakage
- Create 5 folds with equal case distribution
- Save train/val CSVs for each fold

---

## Output Files

### `all_slices.csv`
Contains metadata for all processed slices:

| Column | Description |
|--------|-------------|
| `slice_id` | Unique slice identifier (e.g., `BraTS20_Training_001_slice050`) |
| `case_id` | Case identifier (e.g., `BraTS20_Training_001`) |
| `slice_idx` | Original slice index in volume (0-154) |
| `label` | Primary label: `Normal`, `TC`, `ED`, or `WT` |
| `has_wt` | Binary flag: has Whole Tumor (TC or ED) |
| `has_tc` | Binary flag: has Tumor Core |
| `has_ed` | Binary flag: has Edema |

### `train_fold{i}.csv`, `val_fold{i}.csv`
Same format as `all_slices.csv`, but split into train/val sets for each fold.

### `class_mapping.json`
Documents the label mapping:
```json
{
  "num_classes": 3,
  "class_names": ["Background", "TumorCore", "Edema"],
  "class_labels": [0, 1, 2],
  "regions": {
    "WT": "Whole Tumor = TC + ED (classes 1,2)",
    "TC": "Tumor Core = class 1",
    "ED": "Edema = class 2"
  },
  "original_brats_mapping": {
    "0": "Background → 0",
    "1": "NCR (Necrotic) → 1 (TC)",
    "2": "ED (Edema) → 2",
    "4": "ET (Enhancing) → 1 (TC)"
  }
}
```

---

## Example Output Statistics

After preprocessing BraTS 2020 (369 cases):

```
Total slices: 11,070
Label distribution:
  WT      5,535  (50.0%)  ← Slices with any tumor
  Normal  5,535  (50.0%)  ← Slices without tumor
  TC      4,200  (37.9%)  ← Slices with tumor core
  ED      3,800  (34.3%)  ← Slices with edema

Tumor region statistics:
  Whole Tumor (WT): 5,535 slices (50.0%)
  Tumor Core (TC):  4,200 slices (37.9%)
  Edema (ED):       3,800 slices (34.3%)

Fold 0:
  Train: 8,856 slices from 295 cases
  Val:   2,214 slices from 74 cases
```

---

## Next Steps

### 1. Verify Preprocessing
```bash
# Check output directory
ls -lh data/processed_multiclass/

# Check CSV files
head data/processed_multiclass/all_slices.csv
wc -l data/processed_multiclass/train_fold0.csv
```

### 2. Visualize Samples
```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load a sample
flair = np.array(Image.open('data/processed_multiclass/flair/BraTS20_Training_001_slice080.png'))
seg = np.array(Image.open('data/processed_multiclass/seg/BraTS20_Training_001_slice080.png'))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(flair, cmap='gray')
axes[0].set_title('FLAIR Image')
axes[1].imshow(seg, cmap='jet', vmin=0, vmax=2)
axes[1].set_title('3-Class Segmentation\n0=BG, 1=TC, 2=ED')
plt.tight_layout()
plt.savefig('sample_multiclass.png')
```

### 3. Update Config
Edit `configs/multiclass.yaml`:
```yaml
data:
  proc_root: "data/processed_multiclass"  # Point to new data

model:
  num_classes_seg: 3                      # 3 classes instead of 1
```

### 4. Train Multi-Class Model
```bash
# Baseline config
python scripts/train.py --cfg configs/multiclass.yaml --fold 0

# A100 optimized
python scripts/train.py --cfg configs/multiclass_a100.yaml --fold 0
```

---

## Expected Results

### Binary Segmentation (Old)
```
Overall Dice: 0.91  ← INFLATED by 97% background
Overall IoU:  0.84
```

### Multi-Class Segmentation (New)
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

**Why lower?**
- These are **honest** metrics without background inflation
- More challenging task (3-class vs binary)
- Better reflects actual segmentation quality
- Aligns with medical segmentation standards

---

## Troubleshooting

### Issue: `FileNotFoundError` - NIfTI files not found
**Solution**: Check that `--raw_dir` points to the correct BraTS directory with subdirectories for each case.

### Issue: `MemoryError` during processing
**Solution**: Reduce `--slices_per_case` to 20 or 15 to use less memory.

### Issue: All slices are background
**Solution**: Check that segmentation files (`*_seg.nii.gz`) exist and contain non-zero labels.

### Issue: Unbalanced folds (one fold much smaller)
**Solution**: This is expected if the dataset size is not divisible by `num_folds`. Ensure at least 100+ cases for stable 5-fold splits.

---

## References

- **BraTS 2020 Dataset**: https://www.med.upenn.edu/cbica/brats2020/
- **Label Conventions**: https://www.med.upenn.edu/cbica/brats2020/data.html
- **Multi-Class Segmentation Guide**: `docs/25_10_10_MULTICLASS_SEGMENTATION_GUIDE.md`

---

## Summary

✅ **What this script does**:
- Converts BraTS NIfTI {0,1,2,4} → PNG 3-class {0,1,2}
- Extracts balanced tumor/normal slices
- Creates K-fold splits at case level
- Normalizes modalities to [0, 255]
- Saves metadata for tracking

✅ **What you get**:
- Honest evaluation without background inflation
- Separate metrics for WT, TC, ED regions
- Data ready for multi-class training
- Publication-quality results

✅ **Next steps**:
1. Run preprocessing script
2. Verify output with visualization
3. Train with `configs/multiclass.yaml`
4. Compare binary vs multi-class results
5. Report region-specific performance in paper
