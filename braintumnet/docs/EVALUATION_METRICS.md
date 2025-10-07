# Evaluation Metrics Guide

## Overview

BrainTumNet now includes comprehensive evaluation metrics for medical image segmentation:

| Metric | Full Name | Range | Better | Purpose |
|--------|-----------|-------|--------|---------|
| **IoU** | Intersection over Union (Jaccard Index) | [0, 1] | Higher ↑ | Region overlap |
| **DSC** | Dice Similarity Coefficient | [0, 1] | Higher ↑ | Region overlap (F1) |
| **HD** | Hausdorff Distance | [0, ∞) | Lower ↓ | Boundary accuracy |
| **HD95** | 95th Percentile Hausdorff Distance | [0, ∞) | Lower ↓ | Robust boundary accuracy |

---

## Metric Definitions

### 1. IoU (Intersection over Union)

**Formula**: `IoU = |A ∩ B| / |A ∪ B|`

**What it measures**: Overlap between prediction and ground truth regions.

**Interpretation**:
- `1.0` = Perfect overlap
- `0.5` = Moderate overlap
- `0.0` = No overlap

**Example**:
- Prediction: 100 pixels
- Ground truth: 100 pixels
- Overlap: 80 pixels
- IoU = 80 / (100 + 100 - 80) = 80 / 120 = **0.667**

**Typical values for brain tumor segmentation**:
- Excellent: > 0.70
- Good: 0.60 - 0.70
- Fair: 0.50 - 0.60
- Poor: < 0.50

---

### 2. DSC (Dice Similarity Coefficient)

**Formula**: `DSC = 2 * |A ∩ B| / (|A| + |B|)`

**What it measures**: F1 score for segmentation (harmonic mean of precision and recall).

**Interpretation**:
- `1.0` = Perfect match
- `0.8` = Good match
- `0.5` = Moderate match
- `0.0` = No overlap

**Relationship to IoU**: `DSC = 2 * IoU / (1 + IoU)`

**Example** (same as IoU):
- DSC = 2 * 80 / (100 + 100) = 160 / 200 = **0.800**

**Typical values for brain tumor segmentation**:
- Excellent: > 0.80
- Good: 0.75 - 0.80
- Fair: 0.65 - 0.75
- Poor: < 0.65

**Note**: DSC is more commonly used in medical imaging literature than IoU.

---

### 3. HD (Hausdorff Distance)

**Formula**: `HD(A, B) = max(max_a min_b d(a,b), max_b min_a d(b,a))`

**What it measures**: Maximum distance from any point in prediction to nearest point in ground truth (and vice versa).

**Units**: Pixels (in your case, for 256×256 images)

**Interpretation**:
- `0` = Perfect boundary match
- `5` = 5-pixel maximum mismatch
- `∞` = Empty prediction or ground truth

**Sensitivity**: Very sensitive to outliers (single misclassified pixel can cause high HD).

**Example**:
- Prediction boundary: [..., point at (100, 50)]
- Ground truth boundary: nearest point at (100, 55)
- Distance: 5 pixels
- If this is the maximum distance: HD = **5.0 pixels**

**Typical values for 256×256 brain tumor images**:
- Excellent: < 5 pixels
- Good: 5-10 pixels
- Fair: 10-20 pixels
- Poor: > 20 pixels

---

### 4. HD95 (95th Percentile Hausdorff Distance)

**Formula**: 95th percentile of all point-to-nearest-point distances

**What it measures**: More robust version of HD that ignores the worst 5% of outliers.

**Why use HD95 instead of HD?**
- HD is extremely sensitive to a single outlier pixel
- HD95 provides a more stable and clinically meaningful measurement
- **Recommended** for medical image segmentation

**Interpretation**:
- `0` = Perfect boundary match
- `3` = 95% of boundary points within 3 pixels
- Lower is better

**Example**:
- 1000 boundary distances: [0.5, 0.8, 1.2, ..., 2.5, 45.0] ← outlier
- HD would be: 45.0 pixels (dominated by outlier)
- HD95 would be: ~2.5 pixels (ignores outliers)

**Typical values for 256×256 brain tumor images**:
- Excellent: < 3 pixels
- Good: 3-7 pixels
- Fair: 7-15 pixels
- Poor: > 15 pixels

---

## Usage

### During Training

Metrics are computed automatically during validation:

```bash
cd braintumnet
python scripts/train.py --config configs/full_dataset.yaml
```

**Output example**:
```
Epoch 50 - Val IoU: 0.6543, Dice: 0.7912
```

**Note**: HD/HD95 are NOT computed during training (too slow). Only IoU and Dice are tracked.

---

### Comprehensive Evaluation

After training, run full evaluation with all metrics:

```bash
# Evaluate single fold
python scripts/evaluate.py --cfg configs/full_dataset.yaml --fold 0

# Evaluate all folds
python scripts/evaluate.py --cfg configs/full_dataset.yaml --all_folds
```

**Output example**:
```
======================================================================
EVALUATION RESULTS - Fold 0
======================================================================

Segmentation Metrics:
  IoU (Jaccard):        0.6543
  Dice (F1):            0.7912
  Hausdorff Distance:   8.45 ± 3.21 pixels
  HD95 (95th percentile): 5.23 ± 1.87 pixels
  (HD computed on 4573 slices with tumor)

Classification Metrics:
  Accuracy:             0.8923
  F1 Score:             0.8756
  AUC-ROC:              0.9234
======================================================================
```

---

## Implementation Details

### Global vs Per-Slice Metrics

**IoU and Dice** (computed globally):
- Accumulate intersection and union across ALL slices
- Then compute final IoU/Dice
- This is the CORRECT way (avoids averaging issue)

**HD and HD95** (computed per-slice, then averaged):
- Computed separately for each slice with tumor
- Then averaged across all slices
- Reported with standard deviation

### Edge Cases Handled

1. **Empty prediction and empty ground truth**:
   - IoU = 1.0, Dice = 1.0 (perfect match)
   - HD = inf (no boundary)

2. **Empty prediction but tumor exists**:
   - IoU = 0.0, Dice = 0.0
   - HD = inf

3. **No tumor in ground truth**:
   - Slice is skipped for HD/HD95 computation

---

## Benchmark Results

### Expected Performance on BraTS2020

| Model | Dataset Size | IoU | Dice | HD95 |
|-------|--------------|-----|------|------|
| **Previous (plateau)** | 13 patients | 0.44 | 0.62 | ~20 |
| **Single-modal (T1CE)** | 369 patients | 0.60-0.70 | 0.75-0.82 | 5-8 |
| **Multi-modal (all 4)** | 369 patients | 0.65-0.75 | 0.79-0.85 | 3-6 |

### State-of-the-Art on BraTS2020

Top methods typically achieve:
- Dice: 0.85-0.92 (whole tumor)
- HD95: 2-5 pixels

Your target should be:
- Dice: > 0.75 (good performance)
- HD95: < 8 pixels (good boundary accuracy)

---

## Interpreting Results

### Good Performance Indicators

✓ **IoU > 0.65** and **Dice > 0.78**
✓ **HD95 < 6 pixels** (on 256×256 images)
✓ **HD95 std < 3** (consistent across slices)

### Signs of Problems

❌ **IoU < 0.50** and **Dice < 0.65**: Model underfitting or overfitting
❌ **HD95 > 15 pixels**: Poor boundary localization
❌ **HD95 std > 5**: Inconsistent predictions

### Troubleshooting

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Low IoU/Dice | Not enough data, overfitting | Use full dataset, stronger regularization |
| High HD/HD95 | Poor boundary precision | Add more edge-focused loss (e.g., boundary loss) |
| High IoU but high HD | Good region but rough boundaries | Increase image resolution or add post-processing |

---

## Code API

### Compute metrics programmatically:

```python
from braintumnet.metrics import compute_segmentation_metrics
import numpy as np

# Binary masks (0 or 1)
pred = np.array([[0, 1, 1], [1, 1, 0]])
target = np.array([[0, 1, 0], [1, 1, 1]])

# Compute all metrics
metrics = compute_segmentation_metrics(pred, target)

print(f"IoU: {metrics['iou']:.4f}")
print(f"Dice: {metrics['dice']:.4f}")
print(f"HD: {metrics['hd']:.2f}")
print(f"HD95: {metrics['hd95']:.2f}")
```

### Individual metrics:

```python
from braintumnet.metrics import (
    compute_iou,
    compute_dice_coefficient,
    compute_hausdorff_distance,
    compute_hausdorff_distance_95
)

iou = compute_iou(pred, target)
dice = compute_dice_coefficient(pred, target)
hd = compute_hausdorff_distance(pred, target)
hd95 = compute_hausdorff_distance_95(pred, target)
```

---

## Files Modified

1. **[src/braintumnet/metrics.py](../src/braintumnet/metrics.py)** - Added comprehensive metric functions
2. **[src/braintumnet/engine/evaluator.py](../src/braintumnet/engine/evaluator.py)** - Updated to compute HD/HD95
3. **[scripts/evaluate.py](../scripts/evaluate.py)** - Enhanced evaluation script

---

## References

1. **Dice Coefficient**: Dice, L. R. (1945). "Measures of the amount of ecologic association between species"
2. **IoU**: Jaccard, P. (1912). "The distribution of the flora in the alpine zone"
3. **Hausdorff Distance**: Hausdorff, F. (1914). "Grundzüge der Mengenlehre"
4. **Medical Image Segmentation Metrics**:
   - Taha & Hanbury (2015). "Metrics for evaluating 3D medical image segmentation"
   - BraTS Challenge papers (2017-2020)

---

**Created**: 2025-10-06
**Last Updated**: 2025-10-06
