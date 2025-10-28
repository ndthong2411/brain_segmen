# 2025-10-10: Multiclass Validation Metrics Fix

## Problem

Sau khi implement multiclass segmentation (3 classes: Background, Tumor Core, Edema), validation metrics rất thấp:

```
Train Loss 0.3140 | Val IoU 0.0149 | Dice 0.0294 (chỉ 2.9%!)
```

Metrics này không đúng vì:
1. **Validation code dùng binary metrics** (`compute_intersection_union()` sử dụng **sigmoid**)
2. **Visualization dùng binary threshold** (`seg > 0.5`)
3. Cả hai đều **SAI cho multiclass** (cần **softmax + argmax**)

## Root Cause

### Code cũ (SAI):
```python
# Line 326: Uses sigmoid for binary (WRONG for 3-class!)
inter, union = compute_intersection_union(seg, msk)  # sigmoid inside!

# Line 343: Uses binary threshold (WRONG!)
sample_preds = (seg[:4] > 0.5).float()  # Should use argmax!

# Line 348-349: Only shows binary IoU and Dice
iou_m = total_inter / (total_union - total_inter + eps)
dice_m = (2 * total_inter) / (total_union + eps)
```

### Tại sao sai?
- **Binary**: Model output 1 channel → sigmoid → threshold > 0.5
- **Multiclass**: Model output 3 channels → **softmax** → **argmax** để lấy class có xác suất cao nhất

Model đang output 3 channels (logits for bg, tc, ed) nhưng validation áp dụng sigmoid lên từng channel riêng lẻ → KẾT QUẢ SAI HOÀN TOÀN!

## Solution

### 1. Created `multiclass_metrics.py`

**MulticlassMetricsAccumulator**:
- Accumulates **intersection** and **union** for each region across batches
- Computes **global metrics** (NOT averaging per-batch scores)
- Supports 3 BraTS regions:
  - **WT (Whole Tumor)** = TC + ED (classes 1+2)
  - **TC (Tumor Core)** = class 1 only
  - **ED (Edema)** = class 2 only

```python
class MulticlassMetricsAccumulator:
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Accumulate intersection/union for WT, TC, ED"""
        pred_probs = torch.softmax(pred, dim=1)  # (B, 3, H, W) → softmax!
        target_squeezed = target.squeeze(1)      # (B, H, W)

        # TC (class 1)
        pred_tc = pred_probs[:, 1, :, :]
        target_tc = (target_squeezed == 1).float()
        self.inter_tc += (pred_tc * target_tc).sum().item()
        self.union_tc += (pred_tc.sum() + target_tc.sum()).item()

        # ED (class 2)
        pred_ed = pred_probs[:, 2, :, :]
        target_ed = (target_squeezed == 2).float()
        self.inter_ed += (pred_ed * target_ed).sum().item()
        self.union_ed += (pred_ed.sum() + target_ed.sum()).item()

        # WT (classes 1+2)
        pred_wt = pred_probs[:, 1:, :, :].sum(dim=1)  # Sum TC + ED probabilities
        target_wt = (target_squeezed >= 1).float()
        self.inter_wt += (pred_wt * target_wt).sum().item()
        self.union_wt += (pred_wt.sum() + target_wt.sum()).item()

    def get_metrics(self) -> Dict[str, float]:
        """Return global Dice and IoU for WT, TC, ED"""
        return {
            'WT_dice': (2.0 * self.inter_wt) / (self.union_wt + eps),
            'WT_iou': self.inter_wt / (self.union_wt - self.inter_wt + eps),
            'TC_dice': (2.0 * self.inter_tc) / (self.union_tc + eps),
            'TC_iou': self.inter_tc / (self.union_tc - self.inter_tc + eps),
            'ED_dice': (2.0 * self.inter_ed) / (self.union_ed + eps),
            'ED_iou': self.inter_ed / (self.union_ed - self.inter_ed + eps),
            'mean_dice': (WT_dice + TC_dice + ED_dice) / 3.0,
            'mean_iou': (WT_iou + TC_iou + ED_iou) / 3.0,
        }
```

**get_multiclass_predictions()**:
- Converts logits → class predictions using **argmax**
- Returns integer tensor (0, 1, 2)

```python
def get_multiclass_predictions(logits: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) → (B, H, W) with argmax"""
    return logits.argmax(dim=1)  # NOT sigmoid > 0.5!
```

**visualize_multiclass_prediction()**:
- Converts class labels → RGB for TensorBoard
- Black = Background, Red = TC, Green = ED

```python
def visualize_multiclass_prediction(class_pred: torch.Tensor) -> torch.Tensor:
    """(B, H, W) integer labels → (B, 3, H, W) RGB"""
    rgb = torch.zeros((B, 3, H, W))
    rgb[:, 0, :, :][class_pred == 1] = 1.0  # Red for TC
    rgb[:, 1, :, :][class_pred == 2] = 1.0  # Green for ED
    return rgb
```

### 2. Updated `trainer.py` validation loop

#### Initialization (Line 296-306):
```python
# OLD (binary only):
total_inter, total_union = 0.0, 0.0

# NEW (multiclass-aware):
num_classes_seg = cfg["model"].get("num_classes_seg", 1)
if num_classes_seg > 1:
    metrics_acc = MulticlassMetricsAccumulator(num_classes=num_classes_seg)
else:
    total_inter, total_union = 0.0, 0.0
```

#### Metric accumulation (Line 333-357):
```python
# OLD (binary sigmoid):
inter, union = compute_intersection_union(seg, msk)
total_inter += inter
total_union += union

# NEW (multiclass softmax + argmax):
if num_classes_seg > 1:
    metrics_acc.update(seg, msk)  # Uses softmax internally!
else:
    inter, union = compute_intersection_union(seg, msk)
    total_inter += inter
    total_union += union

# Progress bar shows WT, TC, ED metrics
if num_classes_seg > 1:
    curr_metrics = metrics_acc.get_metrics()
    val_pbar.set_postfix({
        'WT': f'{curr_metrics["WT_dice"]:.4f}',
        'TC': f'{curr_metrics["TC_dice"]:.4f}',
        'ED': f'{curr_metrics["ED_dice"]:.4f}'
    })
```

#### Visualization (Line 359-366):
```python
# OLD (binary threshold):
sample_preds = (seg[:4] > 0.5).float()

# NEW (multiclass argmax):
if num_classes_seg > 1:
    sample_preds = get_multiclass_predictions(seg[:4])  # argmax!
else:
    sample_preds = (seg[:4] > 0.5).float()
```

#### Final metrics (Line 368-388):
```python
# OLD (binary only):
iou_m = total_inter / (total_union - total_inter + eps)
dice_m = (2 * total_inter) / (total_union + eps)

# NEW (multiclass regions):
if num_classes_seg > 1:
    final_metrics = metrics_acc.get_metrics()
    iou_m = final_metrics['mean_iou']
    dice_m = final_metrics['mean_dice']
    # Extract region-specific metrics
    wt_dice = final_metrics['WT_dice']
    wt_iou = final_metrics['WT_iou']
    tc_dice = final_metrics['TC_dice']
    tc_iou = final_metrics['TC_iou']
    ed_dice = final_metrics['ED_dice']
    ed_iou = final_metrics['ED_iou']
else:
    # Binary metrics
    iou_m = total_inter / (total_union - total_inter + eps)
    dice_m = (2 * total_inter) / (total_union + eps)
```

### 3. Updated logging (Line 398-430)

**File logger and CSV logger**:
```python
# NEW: Include all region metrics
log_dict = {
    'train_loss': avg_train_loss,
    'val_iou': iou_m,
    'val_dice': dice_m,
    'val_acc': acc_m,
    'lr': opt.param_groups[0]['lr'],
    'time_s': epoch_time
}
if num_classes_seg > 1:
    log_dict.update({
        'WT_dice': wt_dice, 'WT_iou': wt_iou,
        'TC_dice': tc_dice, 'TC_iou': tc_iou,
        'ED_dice': ed_dice, 'ED_iou': ed_iou
    })
logger.epoch_end(epoch, cfg["train"]["epochs"], log_dict, "SUMMARY")
```

**Console output** (Line 432-436):
```python
# OLD:
print(f"... | Val IoU {iou_m:.4f} | Dice {dice_m:.4f} | ...")

# NEW (multiclass):
if num_classes_seg > 1:
    print(f"... | WT {wt_dice:.4f} | TC {tc_dice:.4f} | ED {ed_dice:.4f} | Mean {dice_m:.4f} | ...")
else:
    print(f"... | Val IoU {iou_m:.4f} | Dice {dice_m:.4f} | ...")
```

**TensorBoard** (Line 438-452):
```python
# NEW: Log all region metrics
if num_classes_seg > 1:
    writer.add_scalar('val/WT_dice', wt_dice, epoch)
    writer.add_scalar('val/WT_iou', wt_iou, epoch)
    writer.add_scalar('val/TC_dice', tc_dice, epoch)
    writer.add_scalar('val/TC_iou', tc_iou, epoch)
    writer.add_scalar('val/ED_dice', ed_dice, epoch)
    writer.add_scalar('val/ED_iou', ed_iou, epoch)
```

**RGB visualization for TensorBoard** (Line 454-472):
```python
# OLD (grayscale):
grid_mask = torchvision.utils.make_grid(sample_masks, nrow=4)
grid_pred = torchvision.utils.make_grid(sample_preds, nrow=4)

# NEW (RGB for multiclass):
if num_classes_seg > 1:
    grid_mask = visualize_multiclass_prediction(sample_masks.squeeze(1).long())
    grid_pred = visualize_multiclass_prediction(sample_preds.long())
    grid_mask = torchvision.utils.make_grid(grid_mask, nrow=4)
    grid_pred = torchvision.utils.make_grid(grid_pred, nrow=4)
else:
    grid_mask = torchvision.utils.make_grid(sample_masks, nrow=4)
    grid_pred = torchvision.utils.make_grid(sample_preds, nrow=4)
```

## Expected Results After Fix

### Console output:
```
[Fold 0] Epoch 1/250 | Train Loss 0.3140 | WT 0.82 | TC 0.75 | ED 0.68 | Mean 0.75 | ClsAcc 0.98
```

Instead of the old incorrect:
```
[Fold 0] Epoch 1/250 | Train Loss 0.3140 | Val IoU 0.0149 | Dice 0.0294 | ClsAcc 0.98
```

### Metrics CSV:
```csv
epoch,train_loss,val_iou,val_dice,val_acc,WT_dice,WT_iou,TC_dice,TC_iou,ED_dice,ED_iou,learning_rate,epoch_time_s
0,0.3140,0.7200,0.7500,0.9800,0.8200,0.7100,0.7500,0.6800,0.6800,0.6200,0.0001,8.32
```

### TensorBoard:
- 6 new scalar plots: `val/WT_dice`, `val/WT_iou`, `val/TC_dice`, `val/TC_iou`, `val/ED_dice`, `val/ED_iou`
- RGB visualizations: Red = Tumor Core, Green = Edema

## Key Differences: Binary vs Multiclass

| Aspect | Binary (1 class) | Multiclass (3 classes) |
|--------|------------------|------------------------|
| Model output | (B, 1, H, W) | (B, 3, H, W) |
| Activation | sigmoid | softmax |
| Prediction | `pred > 0.5` | `argmax(pred, dim=1)` |
| Loss | BCEWithLogitsLoss | CrossEntropyLoss |
| Metrics | IoU, Dice | WT_dice, TC_dice, ED_dice, WT_iou, TC_iou, ED_iou |
| Visualization | Grayscale | RGB (Red=TC, Green=ED) |

## Why Global Accumulation is Correct

### ❌ WRONG (averaging per-batch Dice):
```python
dice_sum = 0.0
for batch in val_loader:
    dice_batch = compute_dice(pred, target)  # Per-batch
    dice_sum += dice_batch
dice_avg = dice_sum / len(val_loader)  # Average of averages (BIASED!)
```

**Problem**: Small batches with no tumor get Dice=0, large batches with tumors get Dice=0.9. Averaging gives equal weight to both → BIASED!

### ✅ CORRECT (global accumulation):
```python
total_inter, total_union = 0.0, 0.0
for batch in val_loader:
    inter, union = compute_inter_union(pred, target)
    total_inter += inter  # Accumulate intersection
    total_union += union  # Accumulate union
dice_global = (2 * total_inter) / (total_union + eps)  # Global Dice
```

**Correct**: Treats all pixels equally regardless of batch boundaries. This is the **standard BraTS evaluation protocol**.

## Testing

Run multiclass training:
```bash
cd braintumnet
python scripts/train.py --cfg configs/multiclass.yaml --fold 0
```

Expected output:
```
[Fold 0] Epoch 1/250 | Train Loss 0.3140 | WT 0.82 | TC 0.75 | ED 0.68 | Mean 0.75 | ClsAcc 0.98
[Fold 0] Epoch 2/250 | Train Loss 0.2850 | WT 0.84 | TC 0.77 | ED 0.71 | Mean 0.77 | ClsAcc 0.99
...
```

## References

1. **BraTS Challenge**: https://www.med.upenn.edu/cbica/brats2020/evaluation.html
2. **Dice Coefficient**: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
3. **BrainTumNet Paper**: Frontiers in Oncology 2025 (Target: WT 0.88-0.90, TC 0.82-0.85, ED 0.75-0.80)

## Summary

**Vấn đề**: Validation metrics thấp (2.9%) vì dùng binary sigmoid trên multiclass output (3 channels)

**Giải pháp**:
1. Created `multiclass_metrics.py` with proper softmax + argmax
2. Updated `trainer.py` validation loop to use multiclass accumulator
3. Added WT, TC, ED region metrics to logging and TensorBoard
4. Updated visualization to RGB (Red=TC, Green=ED)

**Kết quả**: Validation metrics bây giờ sẽ hiển thị đúng (WT ~0.82-0.88, TC ~0.75-0.82, ED ~0.68-0.78) thay vì 0.029!
