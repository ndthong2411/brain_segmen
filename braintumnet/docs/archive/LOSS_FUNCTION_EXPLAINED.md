# Loss Function Architecture - Chi Tiết Đầy Đủ

## 🎯 Bạn Đang Dùng Loss Nào?

### Config Của Bạn
```yaml
# File: configs/phases/phase2_a100.yaml
train:
  loss_type: "ultimate_multitask"  # ← Đây là loss bạn đang dùng
```

### Mapping: Config → Code

```
Config (phase2_a100.yaml)
  loss_type: "ultimate_multitask"
        ↓
Trainer (trainer.py line 150)
  from losses_combined import create_loss_from_config
  crit = create_loss_from_config(cfg)
        ↓
Factory (losses/combined.py line 294)
  return UltimateMultiTaskLoss(...)
        ↓
Class (losses/combined.py line 137)
  UltimateMultiTaskLoss
    → Segmentation: UltimateLoss (line 168)
    → Classification: CrossEntropyLoss (line 171)
```

---

## 📦 Loss Components Hierarchy

### Level 1: UltimateMultiTaskLoss (Top Level)

**File:** `src/braintumnet/losses/combined.py`
**Class:** `UltimateMultiTaskLoss` (line 137)

```python
class UltimateMultiTaskLoss:
    def __init__(self, seg_loss_weight=1.0, cls_loss_weight=0.5, ...):
        self.seg_loss = UltimateLoss(...)        # Segmentation
        self.cls_loss = nn.CrossEntropyLoss()    # Classification

    def forward(self, seg_logits, seg_target, cls_logits, cls_target, aux_outputs):
        # 1. Segmentation loss (main + auxiliary)
        seg_loss, loss_dict = self.seg_loss(seg_logits, seg_target)

        # 2. Auxiliary segmentation losses (deep supervision)
        if aux_outputs:
            for aux in aux_outputs:
                aux_loss, _ = self.seg_loss(aux, seg_target)
                seg_loss += self.aux_weight * aux_loss

        # 3. Classification loss
        cls_loss = self.cls_loss(cls_logits, cls_target)

        # 4. Combine
        total = self.seg_loss_weight * seg_loss + \
                self.cls_loss_weight * cls_loss

        return total, loss_dict
```

**Trong Config:**
```yaml
seg_loss_weight: 1.0    # Weight cho segmentation
cls_loss_weight: 0.5    # Weight cho classification
aux_weight: 0.3         # Weight cho deep supervision
```

**Công Thức:**
```
Total Loss = 1.0 × SegmentationLoss + 0.5 × ClassificationLoss
           = 1.0 × (MainSeg + 0.3×Aux1 + 0.3×Aux2 + 0.3×Aux3) + 0.5 × ClsLoss
```

---

### Level 2: UltimateLoss (Segmentation Component)

**File:** `src/braintumnet/losses/combined.py`
**Class:** `UltimateLoss` (line 25)

```python
class UltimateLoss:
    def __init__(self, dice_weight=1.0, focal_weight=1.0,
                 iou_weight=2.0, boundary_weight=0.5, ...):

        # Import 4 loss modules
        from .losses_multiclass import MultiClassDiceLoss, MultiClassFocalLoss
        from .losses_iou import MulticlassIoULoss
        from .losses_boundary import BoundaryLoss

        self.dice_loss = MultiClassDiceLoss(...)
        self.focal_loss = MultiClassFocalLoss(...)
        self.iou_loss = MulticlassIoULoss(...)
        self.boundary_loss = BoundaryLoss(...)

    def forward(self, logits, target):
        # Compute each component
        dice_l = self.dice_loss(logits, target)
        focal_l = self.focal_loss(logits, target)
        iou_l = self.iou_loss(logits, target)        # ✅ Đã fix bug!
        boundary_l = self.boundary_loss(logits, target)

        # Weighted combination
        total = (self.dice_weight * dice_l +
                 self.focal_weight * focal_l +
                 self.iou_weight * iou_l +
                 self.boundary_weight * boundary_l)

        return total, loss_dict
```

**Trong Config:**
```yaml
dice_weight: 1.0
focal_weight: 1.0
iou_weight: 2.5         # Emphasize IoU (target metric)
boundary_weight: 0.6    # Precise boundaries
```

**Công Thức:**
```
SegLoss = 1.0×Dice + 1.0×Focal + 2.5×IoU + 0.6×Boundary
```

---

### Level 3: Individual Loss Components

#### 3.1 Dice Loss

**File:** `src/braintumnet/losses/multiclass.py`
**Class:** `MultiClassDiceLoss` (line 20)

```python
class MultiClassDiceLoss:
    def forward(self, logits, target):
        for c in [TC, ED]:  # Skip background
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice = (2 * intersection) / (union + ε)

            dice_loss = 1 - dice  # Convert to loss

            # Apply class weight
            weighted_loss = dice_loss * class_weight[c]
            losses.append(weighted_loss)

        return mean(losses)
```

**Config:**
```yaml
class_weights: [1.0, 4.0, 2.5]  # [bg(ignored), TC, ED]
```

**Công Thức:**
```
DiceLoss = (4.0 × DiceLoss_TC + 2.5 × DiceLoss_ED) / 2

Trong đó:
  DiceLoss_TC = 1 - (2×intersection_TC) / (pred_TC + target_TC)
  DiceLoss_ED = 1 - (2×intersection_ED) / (pred_ED + target_ED)
```

**Range:** [0, max(class_weights)] = [0, 4.0]

---

#### 3.2 Focal Loss

**File:** `src/braintumnet/losses/multiclass.py`
**Class:** `MultiClassFocalLoss` (line 86)

```python
class MultiClassFocalLoss:
    def forward(self, logits, target):
        pt = probability_of_true_class  # Confidence

        # Focal term: Focus on hard examples
        focal_weight = (1 - pt) ** gamma

        # Cross entropy
        ce = -log(pt)

        # Alpha weighting (class balance)
        loss = alpha[class] * focal_weight * ce

        return mean(loss)
```

**Config:**
```yaml
focal_alpha: [0.0, 0.5, 0.15]   # [bg=ignore, TC, ED]
focal_gamma: 3.0                # Focus on hard examples
```

**Công Thức:**
```
FocalLoss = Σ alpha[c] × (1 - p_t)^3.0 × (-log(p_t))

Alpha weights:
  - Background: 0.0 (IGNORED completely)
  - Tumor Core: 0.5 (emphasized)
  - Edema: 0.15
```

**Range:** [0, ∞) but typically [0, 5] in practice

---

#### 3.3 IoU Loss (✅ Bug Fixed!)

**File:** `src/braintumnet/losses/iou.py`
**Class:** `MulticlassIoULoss` (line 23)

```python
class MulticlassIoULoss:
    def forward(self, logits, target):
        for c in [TC, ED]:
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection

            iou = intersection / (union + ε)

            # ✅ FIX: Compute loss FIRST, then weight
            iou_loss = 1 - iou  # Loss in [0, 1]
            weighted_loss = iou_loss * class_weight[c]  # Always positive!

            losses.append(weighted_loss)

        return mean(losses)
```

**Config:**
```yaml
class_weights: [1.0, 4.0, 2.5]  # Same as Dice
```

**Công Thức:**
```
IoULoss = (4.0 × (1-IoU_TC) + 2.5 × (1-IoU_ED)) / 2

Range: [0, 4.0]
  - Perfect (IoU=1): loss = 0
  - Random (IoU=0): loss = (4.0 + 2.5)/2 = 3.25
```

**Bug đã fix:**
- OLD: `weighted_iou = iou × weight` → có thể > 1 → loss âm ❌
- NEW: `weighted_loss = (1-iou) × weight` → luôn dương ✅

---

#### 3.4 Boundary Loss

**File:** `src/braintumnet/losses/boundary.py`
**Class:** `BoundaryLoss` (line 23)

```python
class BoundaryLoss:
    def forward(self, logits, target):
        # Compute signed distance function (SDF)
        target_sdf = distance_transform(target)

        # Boundary region (near edges)
        boundary_region = (abs(target_sdf) <= threshold)

        # Weight pixels by proximity to boundary
        weight = exp(-abs(target_sdf))

        # Weighted L1 error
        error = abs(pred - target)
        loss = (weight * error * boundary_region).mean()

        return loss
```

**Config:**
- Không có config riêng, sử dụng default

**Công Thức:**
```
BoundaryLoss = mean(exp(-|SDF|) × |pred - target|)

Trong đó:
  - SDF = signed distance to boundary
  - Pixels gần boundary có weight cao hơn
```

**Range:** [0, ∞) but typically [0, 10]

---

## 📊 Complete Loss Formula

### Full Expansion

```
TOTAL_LOSS = SegLoss + 0.5 × ClsLoss

SegLoss = MainSegLoss + 0.3×Aux1 + 0.3×Aux2 + 0.3×Aux3

MainSegLoss = 1.0×Dice + 1.0×Focal + 2.5×IoU + 0.6×Boundary

Dice = (4.0×Dice_TC + 2.5×Dice_ED) / 2
Focal = (0.5×Focal_TC + 0.15×Focal_ED) / 2
IoU = (4.0×IoU_TC + 2.5×IoU_ED) / 2
Boundary = BoundaryLoss (no class weighting)

ClsLoss = CrossEntropy(pred_class, true_class)
```

### Numerical Example (Early Training)

Giả sử epoch đầu:
- TC IoU = 0.2, ED IoU = 0.3
- TC Dice = 0.25, ED Dice = 0.35
- Focal = 2.0
- Boundary = 3.5
- Cls correct (p=0.8)

```
Dice = (4.0×(1-0.25) + 2.5×(1-0.35)) / 2
     = (4.0×0.75 + 2.5×0.65) / 2
     = (3.0 + 1.625) / 2
     = 2.3125

IoU = (4.0×(1-0.2) + 2.5×(1-0.3)) / 2
    = (4.0×0.8 + 2.5×0.7) / 2
    = (3.2 + 1.75) / 2
    = 2.475

Focal = 2.0

Boundary = 3.5

MainSegLoss = 1.0×2.3125 + 1.0×2.0 + 2.5×2.475 + 0.6×3.5
            = 2.3125 + 2.0 + 6.1875 + 2.1
            = 12.6

SegLoss = 12.6 + 0.3×(aux losses, similar)
        ≈ 12.6 + 0.3×3×12 ≈ 23.4

ClsLoss = -log(0.8) ≈ 0.22

TOTAL = 23.4 + 0.5×0.22 = 23.51
```

### Expected Range

| Stage | Total Loss | IoU | Comment |
|-------|-----------|-----|---------|
| Random | 20-30 | 0.0-0.1 | Epoch 0-5 |
| Learning | 10-20 | 0.1-0.5 | Epoch 5-50 |
| Converging | 5-10 | 0.5-0.7 | Epoch 50-150 |
| Fine-tuning | 2-5 | 0.7-0.85 | Epoch 150-400 |
| Target | 1-3 | 0.85+ | Goal |

---

## 🔍 How to Verify What Loss You're Using

### 1. Check Training Log

```bash
grep "Using loss type" logs/*.log
```

Output:
```
Using loss type: ultimate_multitask (Phase 1+ Ultimate Loss)
  Loss components: Dice + Focal + IoU + Boundary
  IoU weight: 2.5
  Boundary weight: 0.6
```

### 2. Check TensorBoard

```bash
tensorboard --logdir=runs
```

Navigate to:
- **Scalars → train/loss_dice** ← Dice component
- **Scalars → train/loss_focal** ← Focal component
- **Scalars → train/loss_iou** ← IoU component (đã fix!)
- **Scalars → train/loss_boundary** ← Boundary component
- **Scalars → train/loss_total** ← Total loss

### 3. Check Code at Runtime

Add this to trainer.py:
```python
print(f"Loss type: {type(crit)}")
print(f"Loss config: {crit}")
```

Output:
```
Loss type: <class 'braintumnet.losses_combined.UltimateMultiTaskLoss'>
```

---

## 🎛️ How to Change Loss

### Option 1: Edit Config File

```yaml
# configs/phases/phase2_a100.yaml
train:
  # Change loss type
  loss_type: "ultimate_multitask"  # or "ultimate", "dice_focal", etc

  # Adjust weights
  dice_weight: 1.0      # ← Modify here
  focal_weight: 1.0
  iou_weight: 3.0       # ← Increase to emphasize IoU more
  boundary_weight: 0.8  # ← Increase for better boundaries

  # Adjust class balance
  class_weights: [1.0, 5.0, 3.0]  # ← Increase TC weight
  focal_alpha: [0.0, 0.6, 0.2]    # ← Adjust focal weighting
```

### Option 2: Create New Loss

1. **Define in losses/combined.py:**
```python
class MyCustomLoss(nn.Module):
    def __init__(self, ...):
        # Your implementation
        pass

    def forward(self, logits, target):
        # Your loss computation
        return loss, loss_dict
```

2. **Add to factory function:**
```python
def create_loss_from_config(cfg):
    loss_type = cfg['train']['loss_type']

    if loss_type == 'my_custom':
        return MyCustomLoss(...)
    # ... existing code
```

3. **Update config:**
```yaml
train:
  loss_type: "my_custom"
```

---

## 📈 Loss Monitoring Best Practices

### What to Watch

1. **Total Loss** → Should decrease steadily
2. **IoU Component** → Should decrease (IoU increasing)
3. **Individual Components** → All should be positive ✅
4. **Val IoU** → Main metric (most important!)

### Red Flags

❌ Total loss increasing → Learning rate too high
❌ Loss oscillating wildly → Batch size too small
❌ IoU component negative → BUG (đã fix!)
❌ Val IoU decreasing while train loss decreasing → Overfitting

### Good Signs

✅ Loss decreasing smoothly
✅ Val IoU increasing
✅ All components positive
✅ IoU component decreasing fastest (target metric working!)

---

## 🔧 Troubleshooting

### Problem: Loss is Negative

**Solution:** ✅ Already fixed in IoU Loss!

Check:
```bash
python scripts/verify_loss_fix.py
```

### Problem: Loss Not Decreasing

Try:
1. Lower learning rate: `lr: 5e-5 → 3e-5`
2. Increase batch size: `batch_size: 48 → 64`
3. Check data loading: `workers: 8 → 12`

### Problem: Val IoU Not Improving

Try:
1. Increase IoU weight: `iou_weight: 2.5 → 3.0`
2. Increase TC weight: `class_weights: [1.0, 4.0, 2.5] → [1.0, 5.0, 3.0]`
3. More training: `epochs: 400 → 500`

---

## 📚 File Reference

| File | Purpose |
|------|---------|
| **losses/combined.py** | UltimateLoss, UltimateMultiTaskLoss, factory |
| **losses/multiclass.py** | Dice, Focal losses |
| **losses/iou.py** | IoU, Tversky losses (✅ fixed!) |
| **losses/boundary.py** | Boundary, Hausdorff losses |
| **trainer.py** | Loss instantiation, training loop |
| **Config YAML** | Loss configuration |

---

## ✅ Summary

**You are using:**
```
UltimateMultiTaskLoss
├── Segmentation: UltimateLoss
│   ├── Dice Loss (class weighted)
│   ├── Focal Loss (class weighted)
│   ├── IoU Loss (class weighted) ✅ FIXED!
│   └── Boundary Loss
└── Classification: CrossEntropyLoss

Formula:
Total = 1.0×(Dice + Focal + 2.5×IoU + 0.6×Boundary) + 0.5×Classification
```

**Files involved:**
- Config: `configs/phases/phase2_a100.yaml`
- Factory: `losses/combined.py::create_loss_from_config()`
- Loss: `losses/combined.py::UltimateMultiTaskLoss`
- Components: `losses/multiclass.py`, `losses/iou.py`, `losses/boundary.py`
- Training: `trainer.py`

**All components are positive after fix!** ✅
