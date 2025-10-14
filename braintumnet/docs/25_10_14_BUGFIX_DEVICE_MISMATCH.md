# Bug Fix: Device Mismatch in Loss Functions

**Date**: 2025-10-14
**Status**: ✅ FIXED AND TESTED
**Issue**: RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)

---

## Problem Description

When running training with Phase 1 configuration:
```bash
python scripts/train.py --cfg configs/phase1_iou_focus.yaml --fold 4
```

**Error**:
```
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
  File "E:\thong\code\brain_segmen\braintumnet\src\braintumnet\losses_multiclass.py", line 132, in forward
    alpha_t = self.alpha[target_flat]
```

**Root Cause**:
- Loss functions use `register_buffer()` to store tensors like `alpha` and `class_weights`
- Registered buffers are created on CPU by default
- During training, input tensors (`logits`, `target`) are moved to GPU
- When indexing CPU buffer with GPU tensor, PyTorch throws device mismatch error

---

## Files Fixed

### 1. losses_multiclass.py

#### Fix 1: MultiClassFocalLoss - Line 132

**Before**:
```python
alpha_t = self.alpha[target_flat]
```

**After**:
```python
# Alpha weighting - ensure alpha tensor is on same device as target
alpha_t = self.alpha.to(target_flat.device)[target_flat]
```

**Also Fixed Line 123**:
```python
# Before
pt = probs_flat[torch.arange(len(target_flat)), target_flat]

# After
pt = probs_flat[torch.arange(len(target_flat), device=target_flat.device), target_flat]
```

#### Fix 2: MultiClassDiceLoss - Line 76

**Before**:
```python
# Apply class weight
weighted_loss = dice_loss * self.class_weights[c]
```

**After**:
```python
# Apply class weight - ensure same device
class_weight = self.class_weights[c].to(dice_loss.device)
weighted_loss = dice_loss * class_weight
```

---

### 2. losses_iou.py

#### Fix: MulticlassIoULoss - Line 86

**Before**:
```python
# Apply class weight
weighted_iou = iou * self.class_weights[c]
```

**After**:
```python
# Apply class weight - ensure same device
class_weight = self.class_weights[c].to(iou.device)
weighted_iou = iou * class_weight
```

---

### 3. losses_boundary.py

**Status**: ✅ No fix needed

Already handles device correctly:
```python
return torch.from_numpy(sdf).float().to(mask.device)  # Line 73
```

---

### 4. losses_combined.py

**Status**: ✅ No fix needed

Uses the fixed individual loss components, so inherits the fixes automatically.

---

## Testing Results

### Unit Tests - All Passed ✅

#### Test 1: MultiClassFocalLoss
```python
focal = MultiClassFocalLoss(num_classes=3, alpha=[0.5, 0.4, 0.1], gamma=3.0)
logits = torch.randn(2, 3, 64, 64).cuda()
target = torch.randint(0, 3, (2, 64, 64)).cuda()
loss = focal(logits, target)
# Result: Loss = 0.2544 ✅
```

#### Test 2: MultiClassDiceLoss
```python
dice = MultiClassDiceLoss(num_classes=3, class_weights=[1.0, 3.0, 2.0])
logits = torch.randn(2, 3, 64, 64).cuda()
target = torch.randint(0, 3, (2, 64, 64)).cuda()
loss = dice(logits, target)
# Result: Loss = 1.6657 ✅
```

#### Test 3: MulticlassIoULoss
```python
iou = MulticlassIoULoss(num_classes=3, class_weights=[1.0, 3.0, 2.0])
logits = torch.randn(2, 3, 64, 64).cuda()
target = torch.randint(0, 3, (2, 64, 64)).cuda()
loss = iou(logits, target)
# Result: Loss = 0.5062 ✅
```

#### Test 4: UltimateLoss (Combined)
```python
ultimate = UltimateLoss(
    num_classes=3,
    dice_weight=1.0, focal_weight=1.0,
    iou_weight=2.0, boundary_weight=0.5,
    focal_alpha=[0.5, 0.4, 0.1], focal_gamma=3.0,
    class_weights=[1.0, 3.0, 2.0]
)
logits = torch.randn(2, 3, 64, 64).cuda()
target = torch.randint(0, 3, (2, 64, 64)).cuda()
loss, loss_dict = ultimate(logits, target)

# Results:
# Total: 3.1072
#   Dice: 1.6688
#   Focal: 0.2567
#   IoU: 0.5006
#   Boundary: 0.3608
# ✅ ALL COMPONENTS WORKING
```

#### Test 5: UltimateMultiTaskLoss
```python
ultimate_mt = UltimateMultiTaskLoss(
    seg_loss_weight=1.0, cls_loss_weight=0.5,
    deep_supervision=True, aux_weight=0.3,
    num_classes=3,
    dice_weight=1.0, focal_weight=1.0,
    iou_weight=2.0, boundary_weight=0.5,
    focal_alpha=[0.5, 0.4, 0.1], focal_gamma=3.0,
    class_weights=[1.0, 3.0, 2.0]
)
seg_logits = torch.randn(2, 3, 256, 256).cuda()
seg_target = torch.randint(0, 3, (2, 256, 256)).cuda()
cls_logits = torch.randn(2, 1).cuda()
cls_target = torch.randint(0, 2, (2, 1)).float().cuda()
aux_outputs = None
loss, loss_dict = ultimate_mt(seg_logits, seg_target, cls_logits, cls_target, aux_outputs)

# Results:
# Total: 3.0999
# Loss dict keys: ['dice', 'focal', 'iou', 'boundary', 'total', 'cls', 'seg_weighted', 'cls_weighted']
# ✅ MULTITASK WORKING
```

#### Test 6: Full Pipeline with Config
```python
# Load phase1_iou_focus.yaml config
cfg = yaml.safe_load(open('configs/phase1_iou_focus.yaml'))
crit = create_loss_from_config(cfg)

# Forward pass
loss, loss_dict = crit(seg_logits, seg_target, cls_logits, cls_target, aux_outputs)

# Results:
# Total loss: 3.1004
#   Dice: 1.6656
#   Focal: 0.2562
#   IoU: 0.4992
#   Boundary: 0.3603
# ✅ CONFIG INTEGRATION WORKING

# Backward pass
loss.backward()
# ✅ GRADIENTS COMPUTED SUCCESSFULLY
```

---

## Summary of Changes

| File | Line | Issue | Fix |
|------|------|-------|-----|
| losses_multiclass.py | 123 | torch.arange on CPU | Added `device=target_flat.device` |
| losses_multiclass.py | 132 | self.alpha on CPU | Added `.to(target_flat.device)` |
| losses_multiclass.py | 76 | self.class_weights on CPU | Added `.to(dice_loss.device)` |
| losses_iou.py | 86 | self.class_weights on CPU | Added `.to(iou.device)` |

**Total lines changed**: 4 lines across 2 files

---

## Verification

### Before Fix:
```
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
```

### After Fix:
```
[12:21:24] [INFO] Using loss type: ultimate_multitask (Phase 1+ Ultimate Loss)
[12:21:24] [INFO]   Loss components: Dice + Focal + IoU + Boundary
[12:21:24] [INFO]   IoU weight: 2.0
[12:21:24] [INFO]   Boundary weight: 0.5

✅ All loss components compute successfully
✅ Forward pass works on GPU
✅ Backward pass computes gradients
✅ Training ready to start
```

---

## Training Command

Now ready to train:
```bash
python scripts/train.py --cfg configs/phase1_iou_focus.yaml --fold 4
```

**Expected behavior**:
- ✅ Loss components initialized on GPU
- ✅ Forward pass computes all 4 losses (Dice, Focal, IoU, Boundary)
- ✅ Backward pass computes gradients
- ✅ Training proceeds without device errors

---

## Lessons Learned

### Problem: register_buffer() tensors on CPU

When using `register_buffer()`:
```python
self.register_buffer('alpha', torch.tensor(alpha))
```

The buffer is created on CPU by default. During training with GPU tensors, indexing operations fail.

### Solution: Explicit device transfer

**Option 1**: Transfer buffer to tensor device during indexing
```python
alpha_t = self.alpha.to(target.device)[target]
```

**Option 2**: Transfer buffer element before computation
```python
class_weight = self.class_weights[c].to(loss.device)
```

### Best Practice

For loss functions that use buffers:
1. Always use `.to(tensor.device)` when indexing or computing with buffers
2. Test with CUDA tensors during unit testing
3. Verify forward + backward pass on GPU

---

## Impact

**Fixes Phase 1 Implementation**: All device mismatch issues resolved

**Testing Status**:
- ✅ Individual loss components
- ✅ Combined Ultimate loss
- ✅ Multitask loss with classification
- ✅ Config-based creation
- ✅ Forward + backward pass
- ✅ Gradient computation

**Training Status**: ✅ READY

---

**Version**: Phase 1 Bug Fix
**Date**: 2025-10-14
**Status**: VERIFIED AND TESTED ✅
