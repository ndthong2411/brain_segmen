# Background Handling in Multi-class Segmentation

**Date**: 2025-10-14
**Issue**: Ensure background class doesn't bias loss computation
**Status**: ✅ FIXED AND VERIFIED

---

## Problem

In multi-class brain tumor segmentation:
- **Class 0**: Background (typically 95-99% of pixels)
- **Class 1**: Tumor Core (TC) - 0.5-2% of pixels
- **Class 2**: Edema (ED) - 1-3% of pixels

Without proper handling, the background class dominates the loss computation, causing the model to:
1. Focus on predicting background correctly
2. Ignore small tumor regions
3. Achieve high accuracy but poor IoU on tumor classes

**Goal**: Compute loss **only** on tumor classes (TC, ED), completely ignoring background.

---

## Solution: ignore_background=true

### Configuration

**File**: `configs/phase1_iou_focus.yaml`

```yaml
train:
  # Focal loss hyperparameters
  # When ignore_background=true, focal_alpha[0] MUST be 0.0 to ignore background
  focal_alpha: [0.0, 0.4, 0.1]           # [bg=0.0, TC, ED] - Emphasize TC, IGNORE bg
  focal_gamma: 3.0

  # Class weights for Dice/IoU (background weight ignored when ignore_background=true)
  class_weights: [1.0, 3.0, 2.0]         # [bg(ignored), TC, ED] - 3x weight on TC

  ignore_background: true                # Skip background in Dice/IoU, set bg focal_alpha=0
```

**Critical**:
- `focal_alpha[0] = 0.0` (not 0.5!) to give background pixels 0 weight
- `ignore_background: true` to skip background in Dice/IoU loops

---

## Implementation Details

### 1. MultiClassDiceLoss

**File**: `src/braintumnet/losses_multiclass.py:63`

```python
# Compute Dice for each class
dice_scores = []
start_idx = 1 if self.ignore_background else 0  # Skip class 0 if ignoring background

for c in range(start_idx, self.num_classes):  # Loop: c = 1, 2 (TC, ED only)
    pred_c = pred[:, c]
    target_c = target_one_hot[:, c]

    intersection = (pred_c * target_c).sum(dim=(1, 2))
    union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    dice_loss = 1.0 - dice

    # Apply class weight (only TC and ED, not background)
    class_weight = self.class_weights[c].to(dice_loss.device)  # c=1 → 3.0, c=2 → 2.0
    weighted_loss = dice_loss * class_weight
    dice_scores.append(weighted_loss.mean())

# Average across tumor classes only
total_loss = torch.stack(dice_scores).mean()
```

**Behavior**:
- When `ignore_background=True`: Loop runs for `c in [1, 2]` (TC, ED)
- Background (class 0) is **never computed**, saving computation
- `class_weights[1]` and `class_weights[2]` are used (3.0 and 2.0)
- Dice loss is computed **only** on tumor classes

### 2. MulticlassIoULoss

**File**: `src/braintumnet/losses_iou.py:70`

```python
# Same logic as Dice
iou_scores = []
start_idx = 1 if self.ignore_background else 0

for c in range(start_idx, self.num_classes):  # c = 1, 2 only
    pred_c = pred_probs[:, c]
    target_c = target_one_hot[:, c]

    intersection = (pred_c * target_c).sum(dim=(1, 2))
    union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2)) - intersection

    iou = (intersection + self.smooth) / (union + self.smooth)

    # Apply class weight
    class_weight = self.class_weights[c].to(iou.device)  # c=1 → 3.0, c=2 → 2.0
    weighted_iou = iou * class_weight
    iou_scores.append(weighted_iou.mean())

mean_iou = torch.stack(iou_scores).mean()
loss = 1.0 - mean_iou
```

**Behavior**:
- IoU computed **only** on TC and ED classes
- Background IoU is **never computed**
- This directly optimizes the target metric (tumor IoU)

### 3. MultiClassFocalLoss

**File**: `src/braintumnet/losses_multiclass.py:96-105`

```python
def __init__(self, num_classes=3, alpha=None, gamma=2.0, ignore_background=True):
    super().__init__()
    self.num_classes = num_classes
    self.gamma = gamma

    if alpha is None:
        # Default: equal weights, but 0 for background if ignored
        alpha = [0.0 if ignore_background else 1.0] + [1.0] * (num_classes - 1)
        # Example: alpha = [0.0, 1.0, 1.0] when ignore_background=True

    self.register_buffer('alpha', torch.tensor(alpha))
```

**Forward pass** (line 133):
```python
# Alpha weighting - ensure alpha tensor is on same device as target
alpha_t = self.alpha.to(target_flat.device)[target_flat]
# If target_flat = [0, 0, 0, 1, 1, 2, 2] (mixed background and tumor pixels)
# And alpha = [0.0, 0.4, 0.1]
# Then alpha_t = [0.0, 0.0, 0.0, 0.4, 0.4, 0.1, 0.1]
# Background pixels get weight 0.0 → contribute 0 to loss!

# Focal loss
loss = alpha_t * focal_weight * ce
return loss.mean()
```

**Behavior**:
- **All pixels are processed** (including background)
- But background pixels have `alpha[0] = 0.0`, so their loss contribution is **0**
- Only tumor pixels (TC, ED) contribute to the final loss
- Efficient: no branching, just multiplication by 0

### 4. BoundaryLoss

**File**: `src/braintumnet/losses_boundary.py:43`

```python
def __init__(self, theta0=3, theta=5, ignore_background=True):
    super().__init__()
    self.theta0 = theta0
    self.theta = theta
    self.ignore_background = ignore_background
```

**Forward pass** (line 109):
```python
# Compute boundary loss for each class
losses = []
start_idx = 1 if self.ignore_background else 0

for c in range(start_idx, C):  # c = 1, 2 only
    # Get class predictions and targets
    pred_c = pred_probs[:, c]
    target_c = (target == c).float()

    # Compute SDF and boundary loss
    # ... boundary computation ...

    losses.append(boundary_loss)

return torch.stack(losses).mean()
```

**Behavior**:
- Boundary loss computed **only** on TC and ED boundaries
- Background boundaries are **ignored**
- Focuses on precise tumor edge segmentation

---

## Verification Tests

### Test 1: Synthetic Data with 98.8% Background

```python
# Create batch: 98.8% background, 0.6% TC, 0.6% ED
seg_target = torch.zeros(2, 256, 256, dtype=torch.long).cuda()  # All background
seg_target[:, 100:120, 100:120] = 1  # Small TC region
seg_target[:, 130:150, 130:150] = 2  # Small ED region

loss, loss_dict = crit(seg_logits, seg_target, cls_logits, cls_target, None)
```

**Results**:
```
Batch composition:
  Class 0 (Background): 129472 pixels (98.8%)
  Class 1 (TC): 800 pixels (0.6%)
  Class 2 (ED): 800 pixels (0.6%)

Loss values:
  Total: 4.6736
  Dice: 2.4701
  Focal: 0.0023    ← Near zero despite 98.8% background!
  IoU: 0.9848
  Boundary: 0.0046
```

✅ **Focal loss is near 0** despite 98.8% background pixels!

### Test 2: Pure Background Image (100%)

```python
seg_target_bg = torch.zeros(2, 256, 256, dtype=torch.long).cuda()  # 100% background
loss_bg, loss_dict_bg = crit(seg_logits, seg_target_bg, cls_logits, cls_target, None)
```

**Results**:
```
Pure background loss: 4.7290
  Dice: 2.5000
  Focal: 0.0000    ← Exactly zero!
  IoU: 0.9999
  Boundary: 0.0000
```

✅ **Focal loss is exactly 0.0** with 100% background!

### Test 3: Actual BraTS Data

```python
# Load from data/processed_multiclass with real brain scans
dataset = SliceDataset(proc_root='data/processed_multiclass', ...)
batch = next(iter(loader))

# Forward pass
loss, loss_dict = criterion(seg, msk, cls, lab, aux)
```

**Results**:
```
Mask class distribution:
  Class 0 (Background): 260875 pixels (99.52%)
  Class 1 (TC): 1269 pixels (0.48%)

Loss values:
  Total: 8.8521
  Dice: 2.4811
  Focal: 0.0001    ← Near zero with 99.52% background!
  IoU: 0.9902
  Boundary: 0.0129
  Cls: 0.7082
  Aux: 4.0299
```

✅ **Focal loss = 0.0001** despite 99.52% background pixels!

---

## Key Insights

### 1. Why focal_alpha[0] MUST be 0.0

**Wrong** (baseline config):
```yaml
focal_alpha: [0.5, 0.3, 0.2]  # Background gets weight 0.5 ✗
ignore_background: true
```

With this config:
- Background pixels: 99% × 0.5 weight = 49.5% of focal loss
- TC pixels: 0.5% × 0.3 weight = 0.15% of focal loss
- ED pixels: 0.5% × 0.2 weight = 0.1% of focal loss
- **Background dominates focal loss!** ✗

**Correct** (Phase 1 config):
```yaml
focal_alpha: [0.0, 0.4, 0.1]  # Background gets weight 0.0 ✓
ignore_background: true
```

With this config:
- Background pixels: 99% × 0.0 weight = 0% of focal loss ✓
- TC pixels: 0.5% × 0.4 weight = 0.2% of focal loss
- ED pixels: 0.5% × 0.1 weight = 0.05% of focal loss
- **Only tumor classes contribute!** ✓

### 2. Class Weights Interpretation

```yaml
class_weights: [1.0, 3.0, 2.0]  # [bg(ignored), TC, ED]
```

- Index 0 (1.0): Background weight - **ignored** by Dice/IoU due to `start_idx=1`
- Index 1 (3.0): TC weight - **3× emphasis** on Tumor Core (the bottleneck)
- Index 2 (2.0): ED weight - **2× emphasis** on Edema

When looping `for c in range(1, 3)`:
- `class_weights[1]` = 3.0 is used for TC
- `class_weights[2]` = 2.0 is used for ED
- `class_weights[0]` = 1.0 is **never accessed**

### 3. Focal Alpha Interpretation

```yaml
focal_alpha: [0.0, 0.4, 0.1]  # [bg=0.0, TC, ED]
```

- Index 0 (0.0): Background weight - **multiplied** to all background pixels → **0 loss**
- Index 1 (0.4): TC weight - higher than ED to emphasize harder tumor core regions
- Index 2 (0.1): ED weight - lower because edema is easier to segment

Unlike Dice/IoU, focal loss processes **all pixels** but multiplies by alpha:
- `loss = alpha[target] * focal_weight * ce`
- Background pixels: `loss = 0.0 * focal_weight * ce = 0`
- TC pixels: `loss = 0.4 * focal_weight * ce`
- ED pixels: `loss = 0.1 * focal_weight * ce`

---

## Data Configuration

### Correct Data Path

```yaml
data:
  proc_root: "data/processed_multiclass"  # ✓ Multi-class data (0, 1, 2)
  modality: "multi"                        # FLAIR, T1, T1CE, T2
```

**NOT**:
```yaml
data:
  proc_root: "data/processed_full_multimodal"  # ✗ Different preprocessing
```

### Data Format

**Directory structure**:
```
data/processed_multiclass/
  flair/          # FLAIR modality images
  t1/             # T1 modality images
  t1ce/           # T1CE modality images
  t2/             # T2 modality images
  seg/            # Segmentation masks (PNG, values: 0, 1, 2)
  labels.csv      # Case-level labels (HGG vs LGG)
  mapping.csv     # Slice to case mapping
  train_fold{0-4}.csv
  val_fold{0-4}.csv
```

**Mask values**:
- `0`: Background (normal brain tissue)
- `1`: Tumor Core (TC) - enhancing tumor + necrosis
- `2`: Edema (ED) - peritumoral edema

**Mask format**: PNG images with integer values {0, 1, 2}

---

## Summary

### Configuration Checklist

- [x] `proc_root: "data/processed_multiclass"`
- [x] `focal_alpha: [0.0, 0.4, 0.1]` - Background = 0.0
- [x] `class_weights: [1.0, 3.0, 2.0]` - TC emphasized
- [x] `ignore_background: true`
- [x] All loss components tested

### Loss Computation

| Loss Component | Background Handling | Method |
|----------------|---------------------|--------|
| **Dice Loss** | Skipped | Loop starts at class 1 |
| **IoU Loss** | Skipped | Loop starts at class 1 |
| **Focal Loss** | Weight = 0.0 | Multiplied by alpha[0]=0.0 |
| **Boundary Loss** | Skipped | Loop starts at class 1 |

### Expected Behavior

With 99% background pixels:
- **Focal loss**: ~0.0 (background contributes 0)
- **Dice loss**: Computed on TC and ED only
- **IoU loss**: Computed on TC and ED only
- **Total loss**: Driven by tumor classes, **not biased by background** ✓

---

## Training Command

```bash
python scripts/train.py --cfg configs/phase1_iou_focus.yaml --fold 4
```

**Expected**:
- Training focuses on tumor regions (TC, ED)
- Background pixels don't bias loss
- IoU metric improves on tumor classes
- Model learns precise tumor boundaries

---

**Version**: Phase 1 - Background Handling
**Date**: 2025-10-14
**Status**: VERIFIED AND DOCUMENTED ✅
