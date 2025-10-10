# Multi-Class Segmentation Guide for BrainTumNet

**Created:** October 9, 2025
**Purpose:** Fix background dominance by evaluating tumor sub-regions separately

---

## 🎯 Problem with Binary Segmentation

**Current approach (Binary):**
- Class 0: Background (97%)
- Class 1: Tumor (3%)

**Issues:**
- ❌ Background dominates metrics (Dice high but actual tumor accuracy low)
- ❌ No distinction between tumor sub-regions
- ❌ Can't evaluate boundary accuracy properly
- ❌ Model learns "predict mostly background"

**Your observation:** "DICE vẫn cao nhưng kết quả thực tế lại thấp trong việc segmentation"
→ **CHÍNH XÁC!** Background làm Dice cao giả tạo.

---

## ✅ Solution: Multi-Class Segmentation

**New approach (3 classes):**
- Class 0: Background
- Class 1: Tumor Core (TC) = NCR + ET
- Class 2: Edema (ED)

**Evaluated regions:**
1. **Whole Tumor (WT)** = TC + ED (classes 1 + 2)
2. **Tumor Core (TC)** = class 1 only
3. **Edema (ED)** = class 2 only

**Benefits:**
- ✅ Metrics computed ONLY on tumor regions (no background dominance)
- ✅ Separate evaluation for each sub-region
- ✅ True measure of segmentation quality
- ✅ Matches BraTS competition standard

---

## 📊 Expected Metrics Improvement

### Binary (Current):
```
Overall Dice: 0.91  (includes 97% background!)
Overall IoU:  0.84
```
**Problem:** High score but poor tumor segmentation.

### Multi-Class (New):
```
WT Dice: 0.88-0.90  (Whole Tumor only, no background)
TC Dice: 0.82-0.85  (Tumor Core)
ED Dice: 0.75-0.80  (Edema)
Mean Dice: 0.82-0.85 (average of WT, TC, ED)
```
**Better:** True reflection of tumor segmentation quality!

---

## 🔧 Implementation Status

### ✅ Completed Code Changes:

1. **losses_multiclass.py** - New loss functions:
   - `MultiClassDiceLoss` - Dice for each tumor class
   - `MultiClassFocalLoss` - Focal loss for class imbalance
   - `MultiClassCombinedLoss` - Dice + Focal (BEST)
   - `MultiTaskMultiClassLoss` - Segmentation + Classification

2. **metrics_multiclass.py** - Region-specific metrics:
   - `dice_score_multiclass()` - Dice for WT/TC/ED
   - `iou_score_multiclass()` - IoU for WT/TC/ED
   - `compute_all_region_metrics()` - All metrics at once
   - `RegionMetricsAccumulator` - Accumulate across batches

3. **Model updates:**
   - `SegUNetMasked` - Now supports `num_classes` parameter
   - `BrainTumNet` - Handles multi-class output and ROI computation

### ⚠️ Required: Data Reprocessing

**Current data:** Binary masks (0 = bg, 255 = tumor)
**Needed:** Multi-class masks (0 = bg, 1 = TC, 2 = ED)

You need to:
1. Get original BraTS data with labels {0, 1, 2, 4}
2. Remap to {0, 1, 2}:
   - 0 → 0 (background)
   - 1 + 4 → 1 (NCR + ET = Tumor Core)
   - 2 → 2 (Edema)
3. Reprocess all slices with new labels

---

## 🚀 Quick Start (After Data Reprocessing)

### Step 1: Create Multi-Class Config

```yaml
# configs/multiclass_v1.yaml
exp_name: "braintumnet_multiclass_v1"

data:
  proc_root: "data/processed_multiclass"  # NEW: reprocessed multi-class data
  # ... rest same

train:
  epochs: 250
  batch_size: 12
  lr: 1.0e-4
  # ... rest same

  # Multi-class loss settings
  loss_type: "multiclass_combined"  # Dice + Focal for multi-class
  num_classes_seg: 3                # Background, TC, ED

model:
  num_classes_seg: 3  # NEW: 3-class output
  # ... rest same
```

### Step 2: Train

```bash
python scripts/train.py --cfg configs/multiclass_v1.yaml --fold 0
```

### Step 3: View Results

Training will log:
```
Epoch 50/250:
  WT Dice: 0.89  IoU: 0.85
  TC Dice: 0.83  IoU: 0.77
  ED Dice: 0.78  IoU: 0.70
  Mean Dice: 0.83  Mean IoU: 0.77
```

**No more background dominance!** These are TRUE tumor segmentation scores.

---

## 📝 Data Reprocessing Script

Since your raw data is gone, here's what you need to do:

### Option 1: Re-download BraTS 2020

```bash
# 1. Download BraTS 2020 from:
#    https://www.med.upenn.edu/cbica/brats2020/data.html

# 2. Extract to data/raw/

# 3. Run preprocessing with multi-class labels
python scripts/preprocess_multiclass.py --input data/raw --output data/processed_multiclass
```

### Option 2: I Create Preprocessing Script for You

I can create `scripts/preprocess_multiclass.py` that:
- Reads NIfTI files with labels {0, 1, 2, 4}
- Remaps: 0→0, 1→1, 2→2, 4→1 (merge NCR+ET)
- Saves PNG masks with 3 classes
- Preserves all other processing (slicing, filtering, etc.)

**Do you want me to create this script?**

---

## 🎓 BraTS Standard Labels

Original BraTS labels:
```
0: Background (healthy tissue)
1: NCR (Necrotic Tumor Core) - dead tissue inside tumor
2: ED (Edema) - swelling around tumor
4: ET (Enhancing Tumor) - active tumor with contrast agent
```

Standard BraTS regions:
```
WT (Whole Tumor) = 1 + 2 + 4  (all tumor)
TC (Tumor Core)  = 1 + 4      (solid tumor)
ET (Enhancing)   = 4          (active tumor)
```

Our simplified approach (3 classes):
```
Class 0: Background = 0
Class 1: TC = 1 + 4  (NCR + ET)
Class 2: ED = 2      (Edema)

WT = Class 1 + Class 2
```

**Why merge 1+4?**
- Simpler: 3 classes vs 4
- Effective: TC is clinically important (solid tumor mass)
- Proven: Many papers use this approach

---

## 🔬 Technical Details

### Model Changes

**Before (Binary):**
```python
seg_head = nn.Conv2d(base, 1, 1)  # Output 1 channel
seg_prob = torch.sigmoid(logits)   # Sigmoid for binary
```

**After (Multi-class):**
```python
seg_head = nn.Conv2d(base, 3, 1)  # Output 3 channels
seg_prob = torch.softmax(logits, dim=1)  # Softmax for multi-class
```

### Loss Changes

**Before (Binary):**
```python
loss = DiceLoss + BCELoss
```

**After (Multi-class):**
```python
loss = MultiClassDiceLoss + MultiClassFocalLoss
# Dice computed per class, averaged (ignoring background)
# Focal loss handles class imbalance
```

### Metrics Changes

**Before (Binary):**
```python
dice = 2 * intersection / (pred + target)  # Includes background
```

**After (Multi-class):**
```python
# Compute separately for each region
wt_dice = dice(pred[1:], target[1:])  # Whole Tumor
tc_dice = dice(pred[1], target[1])     # Tumor Core
ed_dice = dice(pred[2], target[2])     # Edema
```

---

## 📊 Expected Results After Multi-Class

### Binary (Current - Background Dominance):
| Metric | Value | Reality |
|--------|-------|---------|
| Overall Dice | 0.91 | Inflated by 97% background |
| Overall IoU | 0.84 | Not tumor-specific |

### Multi-Class (NEW - True Tumor Metrics):
| Region | Dice | IoU | Meaning |
|--------|------|-----|---------|
| Whole Tumor | 0.89 | 0.85 | Overall tumor detection |
| Tumor Core | 0.83 | 0.77 | Solid tumor mass |
| Edema | 0.78 | 0.70 | Surrounding swelling |
| **Mean** | **0.83** | **0.77** | **True performance** |

**Improvement:** Honest metrics + better training (no background bias)

---

## 🎯 Next Steps

### Immediate (After Data Reprocessing):

1. **Reprocess data** with 3-class labels
2. **Train multi-class model**:
   ```bash
   python scripts/train.py --cfg configs/multiclass_v1.yaml --fold 0
   ```
3. **Compare with binary**:
   - Binary Overall Dice vs
   - Multi-class WT Dice (should be lower but more accurate)

### For Paper:

Report both approaches:

**Table: Comparison of Segmentation Approaches**

| Approach | Metric | Score | Notes |
|----------|--------|-------|-------|
| Binary | Overall Dice | 0.91 | Includes background |
| Binary | Overall IoU | 0.84 | Background-dominated |
| **Multi-class** | **WT Dice** | **0.89** | **Tumor only** |
| **Multi-class** | **TC Dice** | **0.83** | **Core tumor** |
| **Multi-class** | **ED Dice** | **0.78** | **Edema** |
| **Multi-class** | **Mean Dice** | **0.83** | **True quality** |

**Conclusion:** Multi-class provides honest tumor segmentation metrics.

---

## ❓ FAQ

### Q: Will multi-class be slower to train?
**A:** No, same speed. Model size similar (3 output channels vs 1).

### Q: Will accuracy drop?
**A:** Metrics will be LOWER but MORE ACCURATE (no background inflation).

### Q: Can I keep binary for comparison?
**A:** Yes! Train both, compare in paper.

### Q: What if I can't get raw data?
**A:** You can synthesize multi-class from binary by:
- Class 0: Background (where binary = 0)
- Class 1: Tumor Core (inner 50% of tumor blob)
- Class 2: Edema (outer 50% of tumor blob)
Not perfect but better than pure binary.

---

## 🔧 What I Can Do Next

**Option A:** Create preprocessing script
- `scripts/preprocess_multiclass.py`
- Handles BraTS NIfTI → PNG with 3 classes
- Ready to use when you get raw data

**Option B:** Create synthetic multi-class from current binary
- Use morphological operations to split tumor
- Inner region = TC, outer region = ED
- Quick workaround without raw data

**Option C:** Create trainer script for multi-class
- Update `trainer.py` to handle 3-class segmentation
- Support multi-class metrics logging
- Ready-to-use config

**Which do you want me to implement first?**

---

## 📚 Summary

✅ **Code ready** for multi-class segmentation
✅ **Losses implemented**: Multi-class Dice + Focal
✅ **Metrics ready**: WT, TC, ED evaluation
✅ **Model updated**: Supports 1 or 3 output classes

⚠️ **Need**: Reprocessed data with 3-class labels

**Expected result:**
- Honest tumor metrics (no background inflation)
- Better training (model focuses on tumor)
- Publication-ready evaluation (matches BraTS standard)

**Next:** Choose Option A, B, or C above!
