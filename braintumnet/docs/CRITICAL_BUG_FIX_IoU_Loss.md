# CRITICAL BUG FIX - IoU Loss Negative Values

## 🐛 Bug Discovery

**Date:** 2025-10-15
**Severity:** HIGH
**Impact:** IoU Loss could be negative, causing incorrect gradient flow

### Symptoms

Training log showed:
```
⚠️  WARNING: Negative loss component detected!
  dice_l: 2.698639    ✅
  focal_l: 0.001221   ✅
  iou_l: -0.187913    ❌ NEGATIVE!
  boundary_l: 0.025049 ✅
```

## 🔍 Root Cause

### The Bug

In `src/braintumnet/losses_iou.py`, line 87-94 (old code):

```python
# ❌ WRONG: Weighting IoU before computing loss
iou = (intersection + self.smooth) / (union + self.smooth)
class_weight = self.class_weights[c].to(iou.device)
weighted_iou = iou * class_weight  # BUG HERE!
iou_scores.append(weighted_iou.mean())

mean_iou = torch.stack(iou_scores).mean()
loss = 1.0 - mean_iou  # Can be negative if mean_iou > 1.0
```

### Why It Causes Negative Loss

Given config:
```yaml
class_weights: [1.0, 4.0, 2.5]  # TC weight = 4.0
```

If Tumor Core has IoU = 0.3:
```
weighted_iou = 0.3 × 4.0 = 1.2  # Greater than 1.0!
mean_iou = (1.2 + other_classes) / 2 = potentially > 1.0
loss = 1.0 - 1.2 = -0.2  ❌ NEGATIVE!
```

### Mathematical Error

IoU is bounded [0, 1], so `1 - IoU` should give loss in [0, 1].

But multiplying IoU by class_weight **before** computing loss breaks this bound:
- IoU × weight can be > 1.0
- Therefore 1 - (IoU × weight) can be < 0.0 ❌

### Correct Approach

Weight should be applied to **LOSS**, not IoU:
```python
iou = (intersection + self.smooth) / (union + self.smooth)  # [0, 1]
iou_loss = 1.0 - iou  # [0, 1]
weighted_loss = iou_loss × class_weight  # [0, weight]
```

This ensures:
- IoU stays in [0, 1]
- Loss stays in [0, weight] (always positive)

## ✅ The Fix

### Changed Code

```python
# ✅ CORRECT: Weight the LOSS, not the IoU
for c in range(start_idx, self.num_classes):
    pred_c = pred_probs[:, c]
    target_c = target_one_hot[:, c]

    intersection = (pred_c * target_c).sum(dim=(1, 2))
    union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2)) - intersection

    iou = (intersection + self.smooth) / (union + self.smooth)  # [0, 1]

    # Compute loss FIRST
    iou_loss_c = 1.0 - iou  # [0, 1] - always positive

    # Then apply weight
    class_weight = self.class_weights[c].to(iou_loss_c.device)
    weighted_loss = iou_loss_c * class_weight  # [0, weight] - always positive
    iou_scores.append(weighted_loss.mean())

# Average across classes
loss = torch.stack(iou_scores).mean()  # Already a loss, no 1 - needed
return loss
```

### Key Changes

1. **Line 86:** Compute `iou_loss_c = 1.0 - iou` BEFORE weighting
2. **Line 90:** Apply weight to LOSS: `weighted_loss = iou_loss_c * class_weight`
3. **Line 94-96:** Return loss directly (no `1 - mean_iou`)

## 🧪 Verification

### Before Fix

```
IoU Loss = -0.187913  ❌
Total Loss = -2.4051  ❌
```

### After Fix

IoU Loss should always be ≥ 0:
```
IoU Loss = 0.5 to 2.5  ✅ (positive, weighted)
Total Loss = 1.0 to 6.0 ✅ (positive)
```

Range explanation:
- Each class loss ∈ [0, 1] (before weighting)
- With weights [1.0, 4.0, 2.5], weighted loss ∈ [0, 4.0] per class
- Mean across 2 classes: ∈ [0, 3.25]

### Expected Training Impact

**Before fix:**
- Negative loss → incorrect gradients
- Model still learning (because other components positive)
- But IoU component sending WRONG signals

**After fix:**
- All losses positive ✅
- Correct gradient flow ✅
- Better convergence expected ✅
- IoU should improve faster ✅

## 📊 Why Model Was Still Learning

Even with negative IoU loss, training continued because:

1. **Total loss was combination:**
   ```
   total = 1.0×dice + 1.0×focal + 2.0×iou + 0.5×boundary
         = 2.7    + 0.001   + 2.0×(-0.19) + 0.025
         = 2.7    + 0.001   - 0.38      + 0.025
         = 2.346 (still positive due to dice)
   ```

2. **Gradients still flowed:**
   - Dice loss: pushing in correct direction
   - Focal loss: pushing in correct direction
   - IoU loss: pushing in WRONG direction ❌
   - Boundary loss: pushing in correct direction

3. **Net effect:** Model learned, but suboptimally

## 🎯 Expected Improvements

After this fix:

### IoU Convergence
- **Before:** Conflicting gradients slow down IoU improvement
- **After:** All losses align, faster IoU convergence

### Training Stability
- **Before:** Loss could decrease while IoU decreases (contradiction)
- **After:** Loss ↓ ⟺ IoU ↑ (consistent)

### Final Performance
- **Before:** May plateau at 0.80-0.82 IoU
- **After:** Should reach 0.82-0.85 IoU (as designed)

### Convergence Speed
- **Before:** ~300-400 epochs to converge
- **After:** May converge in 200-300 epochs (faster)

## 🔄 Action Required

### For Ongoing Training

**If training is running:**
1. STOP current training
2. Update code (this fix already applied)
3. Restart from checkpoint:
   ```bash
   python scripts/train.py \
       --cfg configs/phase2_a100_80gb.yaml \
       --fold 0 \
       --resume checkpoints/braintumnet_*_latest.pth
   ```

**Why restart:**
- Previous checkpoints trained with bug
- But optimizer state might be corrupted
- Better to restart for clean gradients

### For New Training

Just use the fixed code - no action needed!

## 📝 Lessons Learned

### Principle: Weight the Loss, Not the Metric

**Wrong:**
```python
weighted_metric = metric × weight
loss = transform(weighted_metric)  # Can break bounds
```

**Right:**
```python
loss = transform(metric)  # Stays in proper bounds
weighted_loss = loss × weight  # Scale the loss
```

### Why This Matters

Metrics often have natural bounds:
- IoU ∈ [0, 1]
- Dice ∈ [0, 1]
- Accuracy ∈ [0, 1]

Loss transformations assume these bounds:
- `1 - IoU` assumes IoU ≤ 1
- `-log(prob)` assumes prob ∈ [0, 1]

Breaking bounds breaks the math!

### Class Weighting Best Practice

**For metrics → loss:**
1. Compute metric (bounded)
2. Transform to loss
3. Apply weights

**For direct losses:**
1. Compute loss per class
2. Apply weights
3. Average

## 🔍 How to Detect Similar Bugs

### 1. Add Assertions

```python
def forward(self, logits, target):
    loss = compute_loss(...)

    # Sanity checks
    assert loss >= 0, f"Loss should be non-negative, got {loss}"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    return loss
```

### 2. Monitor Loss Components

Already added in `losses_combined.py`:
```python
if dice_l < 0 or focal_l < 0 or iou_l < 0 or boundary_l < 0:
    print(f"⚠️  WARNING: Negative loss component detected!")
```

This caught the bug! ✅

### 3. Validate Math

For any custom loss:
1. Check bounds: loss ≥ 0?
2. Check limits: loss(perfect) = 0?
3. Check worst: loss(random) = expected value?

## ✅ Status

- [x] Bug identified
- [x] Root cause found
- [x] Fix implemented
- [x] Code updated in `losses_iou.py`
- [x] Similar issues checked (Dice Loss OK)
- [x] Documentation created
- [ ] Training restarted with fix
- [ ] Improvements verified

## 📚 References

- File: `src/braintumnet/losses_iou.py`
- Lines changed: 72-96
- Related: `losses_multiclass.py` (Dice Loss - was already correct)
- Debug tool: `losses_combined.py` (negative loss detection)

---

**Fix committed:** 2025-10-15
**Impact:** HIGH - Improves IoU convergence and training stability
**Action:** Restart training with updated code
