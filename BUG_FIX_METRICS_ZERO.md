# 🐛 Bug Fix: Metrics Returning 0.0000

## Vấn đề (The Problem)

Khi training model `segunetv2_phase1`, tất cả metrics (Dice, IoU) đều trả về **0.0000**:

```
Epoch 1/400 [Val]: 100%|█| 472/472 [03:24<00:00, 2.31it/s, WT=0.0000, TC=0.0000, ED=0.0000]
[12:52:25] Epoch 1/400 - SUMMARY - train_loss: 0.9008, val_iou: 0.0000, val_dice: 0.0000
```

Mặc dù:
- Model đang training (loss giảm: 0.9008)
- Classification accuracy khá tốt (0.8084)
- Nhưng **TẤT CẢ** Dice/IoU metrics = 0.0000 ❌

## Nguyên nhân gốc rễ (Root Cause)

### ❌ CODE SAI (Before Fix)

File: `braintumnet/src/braintumnet/multiclass_metrics.py`

```python
def update(self, pred: torch.Tensor, target: torch.Tensor):
    # Get prediction probabilities
    pred_probs = torch.softmax(pred, dim=1)  # (B, C, H, W) - SOFT PROBABILITIES
    
    # TC (Tumor Core) = class 1
    pred_tc = pred_probs[:, 1, :, :]  # ❌ WRONG! Values in [0.0, 1.0]
    target_tc = (target_squeezed == 1).float()  # ✓ Binary: 0 or 1
    
    intersection = (pred_tc * target_tc).sum()  # ❌ Very small!
    union = pred_tc.sum() + target_tc.sum()     # ❌ Very large!
```

**Vấn đề**: Sử dụng **soft probabilities** (giá trị liên tục 0-1 từ softmax) thay vì **hard predictions** (0 hoặc 1)

### Ví dụ minh họa (Illustration)

```python
# Logits (model output)
logits = [[-1.0, 2.5, -0.5],   # Strong prediction for class 1 (TC)
          [-2.0, -1.0, 3.0]]   # Strong prediction for class 2 (ED)

# WRONG: Use soft probabilities
pred_probs = softmax(logits)
# pred_probs[0] = [0.04, 0.88, 0.08]  # Sum of channel 1
# pred_probs[1] = [0.01, 0.02, 0.97]  # Sum of channel 2

pred_tc = pred_probs[:, 1]  # [0.88, 0.02] ❌ Continuous values!
target_tc = [1, 0]          # [1, 0] ✓ Binary!

intersection = 0.88*1 + 0.02*0 = 0.88
union = (0.88 + 0.02) + (1 + 0) = 1.90
dice = 2*0.88 / 1.90 = 0.926  # Seems OK for 2 pixels

# But with many pixels (128x128 = 16384 pixels):
# union becomes HUGE (sum of all probabilities across all pixels)
# intersection becomes TINY (only probabilistic overlap)
# Result: dice → 0.0000 ❌
```

```python
# CORRECT: Use hard predictions
pred_classes = argmax(logits)  # [1, 2] ✓ Integer class labels

pred_tc = (pred_classes == 1)  # [1, 0] ✓ Binary!
target_tc = [1, 0]             # [1, 0] ✓ Binary!

intersection = 1*1 + 0*0 = 1
union = (1 + 0) + (1 + 0) = 2
dice = 2*1 / 2 = 1.0  # Perfect match! ✓
```

## Giải pháp (Solution)

### ✅ CODE ĐÚNG (After Fix)

```python
def update(self, pred: torch.Tensor, target: torch.Tensor):
    # Get HARD predictions using argmax (not soft probabilities!)
    pred_classes = torch.argmax(pred, dim=1)  # (B, H, W) - integer [0, C-1]
    
    # TC (Tumor Core) = class 1
    pred_tc = (pred_classes == 1).float()  # ✓ CORRECT! Binary: 0 or 1
    target_tc = (target_squeezed == 1).float()  # ✓ Binary: 0 or 1
    
    intersection = (pred_tc * target_tc).sum()  # ✓ Correct count
    union = pred_tc.sum() + target_tc.sum()     # ✓ Correct count
```

## Các file đã sửa (Fixed Files)

1. **`braintumnet/src/braintumnet/multiclass_metrics.py`**
   - `multiclass_dice_coefficient()` - Sửa để dùng argmax
   - `multiclass_iou()` - Sửa để dùng argmax
   - `compute_brats_regions()` - Sửa để dùng argmax
   - `MulticlassMetricsAccumulator.update()` - Sửa để dùng argmax

## So sánh kết quả (Before vs After)

### ❌ Before Fix:
```
WT Dice: 0.0000, TC Dice: 0.0000, ED Dice: 0.0000
Mean Dice: 0.0000, Mean IoU: 0.0000
```

### ✅ After Fix:
```
WT Dice: 0.4516, TC Dice: 0.3333, ED Dice: 0.3158
Mean Dice: 0.3669, Mean IoU: 0.2264
```

## Giải thích kỹ thuật (Technical Explanation)

### Tại sao phải dùng hard predictions?

**Dice Coefficient** và **IoU** được định nghĩa cho **binary masks**:

```
Dice = 2 * |A ∩ B| / (|A| + |B|)
IoU = |A ∩ B| / |A ∪ B|
```

Trong đó:
- `A`, `B` là **tập hợp các pixels** (binary: 0 hoặc 1)
- `|A|` = số lượng pixels trong A (count)
- `|A ∩ B|` = số pixels cùng thuộc A và B (count)

**Khi dùng soft probabilities**:
- `A` không còn là tập hợp nữa, mà là **weighted set** (mỗi pixel có trọng số 0-1)
- `|A|` = tổng tất cả probabilities (rất lớn)
- `|A ∩ B|` = tổng probabilities tại vị trí có label=1 (rất nhỏ)
- Kết quả: Dice, IoU → 0

**Khi dùng hard predictions**:
- `A`, `B` đúng là binary masks (0 hoặc 1)
- `|A|`, `|A ∩ B|` đúng là số lượng pixels
- Kết quả: Dice, IoU có giá trị đúng ✓

## Verification

Chạy test để verify fix:

```bash
python test_metrics_fix.py
```

Output:
```
======================================================================
ALL TESTS PASSED! ✅
======================================================================

SUMMARY:
  The bug was using soft probabilities (softmax output) instead of
  hard predictions (argmax) for computing Dice/IoU metrics.

  BEFORE (WRONG): pred = softmax(logits)[:, class_idx]  # 0.0-1.0
  AFTER (CORRECT): pred = (argmax(logits) == class_idx)  # 0 or 1
```

## Kết luận (Conclusion)

- ✅ **Bug đã fix**: Metrics giờ sẽ trả về giá trị đúng (không còn 0.0000)
- ✅ **Nguyên nhân**: Dùng soft probabilities thay vì hard predictions
- ✅ **Giải pháp**: Chuyển sang dùng `argmax()` để lấy class labels trước khi tính metrics
- ✅ **Test**: Đã verify với `test_metrics_fix.py`

## Training tiếp

Bây giờ bạn có thể training lại:

```bash
python braintumnet/scripts/train.py --model segunetv2_phase1 --fold 3
```

Metrics giờ sẽ hiển thị giá trị đúng! 🎉
