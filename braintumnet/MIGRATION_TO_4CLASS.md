# Migration Guide: 3-Class → 4-Class Standard BraTS

## Overview

Your codebase has been **upgraded from 3-class to 4-class** standard BraTS segmentation.

### What Changed?

**Before (3-class):**
- Class 0: Background
- Class 1: Tumor Core (NCR + ET merged)
- Class 2: Edema

**After (4-class - STANDARD BraTS):**
- Class 0: Background
- Class 1: NCR/NET (Necrotic/Non-enhancing)
- Class 2: ED (Edema)
- Class 3: **ET (Enhancing Tumor)** ← NEW!

### Evaluation Regions

**Standard BraTS regions:**
- **ET** = class 3 only
- **TC** = classes 1 + 3 (NCR + ET)
- **WT** = classes 1 + 2 + 3 (all tumor)

---

## Files Modified

### 1. **Preprocessing** (`scripts/preprocess_nifti_to_multiclass.py`)
- ✅ Added `convert_brats_seg_to_4class()` function
- ✅ Mapping: BraTS label 4 (ET) → class 3
- ✅ Updated class_mapping.json to 4 classes

### 2. **Metrics** (`src/braintumnet/multiclass_metrics.py`)
- ✅ Added ET metrics computation
- ✅ Updated TC definition: classes 1 + 3 (instead of class 1 only)
- ✅ Updated WT definition: classes 1 + 2 + 3
- ✅ Added ET HD95 computation
- ✅ Supports both 3-class (legacy) and 4-class

### 3. **Trainer** (`src/braintumnet/engine/trainer.py`)
- ✅ Display ET, TC, WT metrics in console
- ✅ Log ET metrics to TensorBoard
- ✅ Log ET metrics to CSV files

### 4. **Config** (`configs/base.yaml`)
- ✅ `num_classes_seg: 3` → `4`
- ✅ Updated `focal_alpha` for 4 classes
- ✅ Updated `class_weights` for 4 classes

---

## Migration Steps

### Step 1: Re-preprocess Data (REQUIRED)

You **MUST** re-run preprocessing to generate 4-class masks:

```bash
cd braintumnet
python scripts/preprocess_nifti_to_multiclass.py \
    --data_root data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
    --out_dir data/processed_multiclass_4class \
    --img_size 256 \
    --num_folds 5
```

**Important**: Old 3-class data is **incompatible** with 4-class models!

### Step 2: Update Config

Already done in `configs/base.yaml`:
```yaml
model:
  num_classes_seg: 4  # Was 3

train:
  focal_alpha: [0.0, 0.3, 0.4, 0.3]  # 4 values
  class_weights: [1.0, 3.0, 4.0, 5.0]  # 4 values
```

### Step 3: Train New Models

```bash
# Train with 4-class
python scripts/train.py --model unetr --fold 0
```

**Note**: Models trained on 3-class data **cannot be used** with 4-class!

---

## Training Output Changes

### Before (3-class):
```
[Fold 0] Epoch 1/400 | Loss 0.9129 | WT 0.0731 | TC 0.0225 | ED 0.6739 | Mean 0.2565 | HD95 45.23
```

### After (4-class):
```
[Fold 0] Epoch 1/400 | Loss 0.9129 | ET 0.0521 | TC 0.0225 | WT 0.0731 | Mean 0.0492 | HD95 48.56
```

**Key Changes:**
- ✅ **ET metrics now displayed** (was merged into TC before)
- ✅ **TC is now NCR + ET** (not NCR + ET merged as before)
- ✅ **Mean is ET + TC + WT** (standard BraTS)
- ❌ ED no longer in mean (but still computed)

---

## TensorBoard Metrics

New metrics available:
- `val/ET_dice` ← NEW
- `val/ET_iou` ← NEW
- `val/ET_hd95` ← NEW
- `val/TC_dice` (definition changed)
- `val/TC_iou` (definition changed)
- `val/WT_dice`
- `val/WT_iou`
- `val/WT_hd95`

---

## Backward Compatibility

### Using 3-class (Legacy Mode)

If you want to keep using 3-class:

1. Set `num_classes_seg: 3` in config
2. Use old preprocessing data
3. Metrics will automatically use 3-class mode

The code **auto-detects** class count and adjusts metrics accordingly.

---

## Testing

Test script provided in `test_4class_brats.py`:

```bash
python test_4class_brats.py
```

Expected output:
```
4-class BraTS Test
==================
ET Dice: 0.XXXX
TC Dice: 0.XXXX  (NCR + ET)
WT Dice: 0.XXXX  (all tumor)
Mean Dice: 0.XXXX  (ET + TC + WT) / 3
```

---

## Common Issues

### Issue 1: Shape mismatch

**Error**: `Expected 3 channels, got 4`

**Solution**: Re-train model from scratch with `num_classes_seg: 4`

### Issue 2: Low ET Dice

**Normal**: ET is very small, expect low Dice initially (< 0.1)

**Solution**: Train longer, use higher `class_weights` for ET

### Issue 3: Old checkpoints don't work

**Expected**: 3-class checkpoints are incompatible with 4-class

**Solution**: Train new models or keep using 3-class mode

---

## Validation

To verify correct implementation:

1. **Check preprocessing output**:
   ```bash
   python -c "
   import numpy as np
   mask = np.load('data/processed_multiclass_4class/seg/BraTS20_001_00055.npy')
   print('Unique labels:', np.unique(mask))
   # Should print: [0 1 2 3]
   "
   ```

2. **Check model output shape**:
   ```python
   model = build_model(cfg)
   x = torch.randn(1, 4, 256, 256)
   y = model(x)[0]
   print(y.shape)  # Should be: torch.Size([1, 4, 256, 256])
   ```

3. **Check metrics**:
   ```bash
   python test_4class_brats.py
   ```

---

## Benefits of 4-Class

✅ **Standard BraTS compliance** - can compare with all papers
✅ **ET evaluation** - separate metrics for enhancing tumor
✅ **BraTS Challenge submission** - required for competition
✅ **Clinical relevance** - ET is important prognostic marker

---

## Next Steps

1. ✅ Re-preprocess data to 4-class
2. ✅ Train new models from scratch
3. ✅ Evaluate on validation set
4. ✅ Compare with BraTS leaderboard
5. ✅ Submit to BraTS Challenge (if applicable)

---

## Questions?

- Check `test_4class_brats.py` for examples
- Review `multiclass_metrics.py` for region definitions
- See `preprocess_nifti_to_multiclass.py` for label mapping

**Important**: This migration is **one-way** - you cannot use 4-class data with 3-class models, and vice versa.
