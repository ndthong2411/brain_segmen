# Fold 4 Training Analysis & Improvement Suggestions

**Training Log**: `braintumnet_multiclass_3class_fold4_20251011_012900.log`
**Duration**: 35h 53m (199 epochs, early stopped at epoch 199/250)
**Device**: CUDA GPU

---

## 📊 Performance Summary

### Best Results (Epoch 149)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **WT Dice** | **0.8476** | 0.88-0.90 | ⚠️ Below target (-0.032) |
| **TC Dice** | **0.8199** | 0.82-0.85 | ✅ At lower bound |
| **ED Dice** | **0.8561** | 0.75-0.80 | ✅✅ **Exceeds target!** |
| **Mean Dice** | **0.8412** | 0.82-0.85 | ✅ Good |
| **Mean IoU** | **0.7263** | ~0.70-0.75 | ✅ Good |
| **Val Accuracy** | **1.0000** | High | ✅ Perfect |

### Peak Performance by Metric

| Metric | Best Epoch | Best Value |
|--------|-----------|------------|
| **WT Dice** | 132 | **0.8487** |
| **WT IoU** | 132 | 0.7371 |
| **TC Dice** | 126 | **0.8222** |
| **TC IoU** | 126 | 0.6981 |
| **ED Dice** | 149 | **0.8561** ✅ |
| **ED IoU** | 149 | **0.7483** ✅ |
| **Val IoU** | 149 | **0.7263** |

### Training Progression

```
Epoch   | Train Loss | WT Dice | TC Dice | ED Dice | Mean Dice
--------|------------|---------|---------|---------|----------
1       | 1.3741     | 0.1064  | 0.3266  | 0.0279  | 0.1536
10      | 0.2637     | 0.7939  | 0.7574  | 0.8101  | 0.7871
50      | 0.1722     | 0.8328  | 0.8071  | 0.8406  | 0.8268
100     | 0.1292     | 0.8422  | 0.8149  | 0.8511  | 0.8361
132     | 0.1238     | 0.8487⭐ | 0.8188  | 0.8553  | 0.8410
149     | 0.1243     | 0.8476  | 0.8199  | 0.8561⭐ | 0.8412⭐
199     | 0.1038     | 0.8423  | 0.8164  | 0.8460  | 0.8349
```

---

## 🔍 Key Observations

### ✅ Strengths

1. **Edema (ED) Segmentation Excellence**
   - ED Dice: **0.8561** (target: 0.75-0.80)
   - **+0.056 above upper target** 🎉
   - ED IoU: 0.7483 (excellent overlap)

2. **Stable Training**
   - Train loss decreased steadily: 1.3741 → 0.1038
   - No significant overfitting (train/val gap is reasonable)
   - Val accuracy = 1.0 (classification task perfect)

3. **Good Overall Performance**
   - Mean Dice: 0.8412 (target: 0.82-0.85) ✅
   - Mean IoU: 0.7263 (solid overlap) ✅
   - TC Dice: 0.8199 (just below target lower bound)

4. **Efficient Training**
   - ~11 minutes per epoch
   - Early stopping at epoch 199 (saved 51 epochs)
   - 26 checkpoint improvements (good optimization)

### ⚠️ Areas for Improvement

1. **Whole Tumor (WT) Below Target**
   - Current: 0.8476 | Target: 0.88-0.90
   - **Gap: -0.032 to -0.052**
   - WT = TC + ED, so should be highest, but is lowest

2. **TC Slightly Below Target**
   - Current: 0.8199 | Target: 0.82-0.85
   - **Gap: -0.002 to -0.030**
   - TC is hardest (Tumor Core = NCR + ET)

3. **Early Plateau**
   - Best WT at epoch 132
   - Best TC at epoch 126
   - Best ED at epoch 149
   - Model plateaued after epoch 150

4. **Learning Rate Hit Zero**
   - LR at epoch 149: 0.00000
   - Cosine annealing reached min_lr too early
   - No exploration after epoch ~100

---

## 💡 Improvement Suggestions

### 🎯 Priority 1: Improve Whole Tumor (WT) Dice

#### Issue
WT Dice = 0.8476 (target: 0.88-0.90). Since WT = TC + ED, and ED is excellent (0.8561), the problem is likely **TC segmentation errors propagating to WT**.

#### Solutions

**A. Increase TC Class Weight**

```yaml
# configs/multiclass.yaml
train:
  # Current: [0.5, 0.3, 0.2] for [bg, tc, ed]
  focal_alpha: [0.5, 0.4, 0.1]  # Emphasize TC more, reduce ED

  # Add explicit class weights
  class_weights: [1.0, 2.5, 1.0]  # [bg, tc, ed] - 2.5× weight on TC
```

**Why**: TC is the hardest class and needs more focus during training.

**B. Increase Focal Gamma (Hard Example Mining)**

```yaml
train:
  focal_gamma: 3.0  # Increase from 2.0 → focus MORE on hard examples
```

**Why**: Focal loss with γ=3 focuses more on boundary pixels and difficult regions.

**C. Use Pure Focal Loss (Drop Dice)**

```yaml
train:
  loss_type: "multiclass_focal"  # Instead of "multiclass_dice_focal"
  focal_alpha: [0.3, 0.4, 0.3]   # More balanced
  focal_gamma: 3.0
```

**Why**: Dice loss might be causing the model to be too conservative. Focal loss is better for hard examples.

---

### 🎯 Priority 2: Extend Training with Better LR Schedule

#### Issue
Learning rate hit zero too early (epoch ~100), preventing further optimization.

#### Solutions

**A. Increase Warmup + Extend Decay**

```yaml
train:
  warmup_steps: 2000      # Increase from 1000
  min_lr: 5.0e-6          # Increase from 1e-6 (allows more fine-tuning)
  early_stop_patience: 75 # Increase from 50
```

**Why**: Longer warmup prevents early overfitting, higher min_lr allows continued learning.

**B. Use Step Decay Instead of Cosine**

```yaml
train:
  scheduler: "step"       # Change from "cosine"
  lr_decay_steps: [100, 150, 200]
  lr_decay_gamma: 0.5     # Multiply LR by 0.5 at each step
```

**Schedule**:
- Epoch 0-100: LR = 1e-4
- Epoch 100-150: LR = 5e-5
- Epoch 150-200: LR = 2.5e-5
- Epoch 200-250: LR = 1.25e-5

**Why**: Step decay maintains exploration capability longer.

**C. Add Restart (Cosine with Warm Restarts)**

```yaml
train:
  scheduler: "cosine_restart"
  restart_period: 50      # Restart every 50 epochs
  restart_mult: 1.5       # Each restart is 1.5× longer
```

**Why**: Restarts help escape local minima.

---

### 🎯 Priority 3: Architecture Modifications

#### A. Increase Model Capacity (More Parameters)

```yaml
# configs/multiclass.yaml
model:
  base: 48              # Increase from 32 → +125% params
  dim: 384              # Increase from 256 → +50% transformer capacity
  depth: 3              # Increase from 2 → +50% transformer depth
```

**New model size**: ~32M params (vs current 14.3M)

**Why**: Model might be underfitting (train loss still decreasing at epoch 199).

**Trade-off**: +30% training time, +50% memory

#### B. Increase Deep Supervision Weight

```python
# src/braintumnet/losses_multiclass.py (modify)
# Current: aux_weight = 0.3 for all auxiliary outputs

aux3_weight = 0.5  # 64×64 resolution
aux2_weight = 0.4  # 128×128 resolution
aux1_weight = 0.3  # 256×256 resolution

total_loss = main_loss + aux3_weight*aux3_loss + aux2_weight*aux2_loss + aux1_weight*aux1_loss
```

**Why**: Stronger supervision at intermediate scales improves boundary learning.

#### C. Add Dropout for Regularization

```python
# braintumnet/src/braintumnet/models/seg_unet.py
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.cbam = CBAM(out_ch)
        self.block = nn.Sequential(
            conv_bn_relu(out_ch + out_ch, out_ch),
            nn.Dropout2d(dropout),  # Add spatial dropout
            conv_bn_relu(out_ch, out_ch)
        )
```

**Config**:
```yaml
model:
  dropout: 0.1  # 10% dropout in decoder
```

**Why**: Prevents overfitting, improves generalization.

---

### 🎯 Priority 4: Data Augmentation

#### Current Augmentation (from config)
```yaml
augment:
  rotate_deg: 20
  hflip_p: 0.5
  vflip_p: 0.5
```

#### Enhanced Augmentation

```yaml
augment:
  rotate_deg: 30          # Increase from 20
  hflip_p: 0.5
  vflip_p: 0.5

  # Add new augmentations (requires code changes)
  elastic_transform: true
  elastic_alpha: 50
  elastic_sigma: 5

  brightness_range: [0.8, 1.2]
  contrast_range: [0.8, 1.2]

  gaussian_noise_std: 0.01
  gaussian_blur_sigma: [0.5, 1.5]
```

**Implementation**:
```python
# braintumnet/src/braintumnet/data/transforms.py
import torchvision.transforms as T

def augment_pair(img, mask, img_size, rotate_deg, hflip_p, vflip_p, train):
    # ... existing code ...

    if train:
        # Add brightness/contrast
        if random.random() > 0.5:
            img_t = T.ColorJitter(brightness=0.2, contrast=0.2)(img_t)

        # Add Gaussian noise
        if random.random() > 0.5:
            noise = torch.randn_like(img_t) * 0.01
            img_t = img_t + noise
```

**Why**: More augmentation → better generalization → higher validation Dice.

---

### 🎯 Priority 5: Ensemble Methods

Since you have 5 folds, use ensemble prediction:

```python
# scripts/ensemble_predict.py
def ensemble_predict(models, input):
    predictions = []
    for model in models:
        pred = model(input)
        predictions.append(pred)

    # Average softmax probabilities
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred

# Load all 5 fold models
models = [
    load_checkpoint('checkpoints/braintumnet_best_fold0.pth'),
    load_checkpoint('checkpoints/braintumnet_best_fold1.pth'),
    load_checkpoint('checkpoints/braintumnet_best_fold2.pth'),
    load_checkpoint('checkpoints/braintumnet_best_fold3.pth'),
    load_checkpoint('checkpoints/braintumnet_best_fold4.pth'),
]

# Predict with ensemble
pred = ensemble_predict(models, test_input)
```

**Expected gain**: +0.02 to +0.04 Dice improvement

---

## 📋 Recommended Action Plan

### Phase 1: Quick Wins (1-2 days)

1. **Adjust Loss Weights** ⭐ HIGHEST IMPACT
   ```yaml
   train:
     focal_alpha: [0.5, 0.4, 0.1]  # Emphasize TC
     focal_gamma: 3.0               # Focus on hard examples
     class_weights: [1.0, 2.5, 1.0] # 2.5× weight on TC
   ```

2. **Extend Learning Schedule**
   ```yaml
   train:
     min_lr: 5.0e-6              # Allow more fine-tuning
     early_stop_patience: 75     # More patience
     warmup_steps: 2000          # Longer warmup
   ```

3. **Retrain Fold 4**
   ```bash
   python scripts/train.py --cfg configs/multiclass_improved.yaml --fold 4
   ```

**Expected gain**: +0.01 to +0.03 WT Dice → **0.86-0.88**

---

### Phase 2: Architecture Improvements (3-5 days)

4. **Increase Model Capacity**
   ```yaml
   model:
     base: 48        # +125% params
     dim: 384        # +50% capacity
     depth: 3        # +50% depth
     dropout: 0.1    # Regularization
   ```

5. **Stronger Deep Supervision**
   - Modify `losses_multiclass.py`
   - Increase auxiliary loss weights to [0.5, 0.4, 0.3]

6. **Retrain All Folds**
   ```bash
   for fold in 0 1 2 3 4; do
       python scripts/train.py --cfg configs/multiclass_improved.yaml --fold $fold
   done
   ```

**Expected gain**: +0.02 to +0.04 WT Dice → **0.87-0.90** ✅

---

### Phase 3: Advanced Techniques (1 week)

7. **Enhanced Augmentation**
   - Implement elastic transform, brightness/contrast jitter
   - See code suggestions above

8. **Test-Time Augmentation (TTA)**
   ```python
   # Average predictions over 8 augmentations
   pred = (pred_orig + pred_hflip + pred_vflip + pred_rot90 +
           pred_rot180 + pred_rot270 + pred_hflip_rot90 + pred_vflip_rot90) / 8
   ```

9. **Ensemble All Folds**
   - Average predictions from all 5 fold models
   - Expected: +0.02-0.04 Dice

**Expected gain**: +0.03 to +0.05 WT Dice → **0.88-0.92** ✅✅

---

## 🎯 Quick Experiment Configs

### Config A: Focus on TC (Recommended First Try)

Create `configs/multiclass_focus_tc.yaml`:
```yaml
exp_name: "braintumnet_multiclass_focus_tc"

data:
  # ... same as multiclass.yaml ...

train:
  epochs: 250
  batch_size: 12
  lr: 1.0e-4
  weight_decay: 1.0e-4

  loss_type: "multiclass_dice_focal"
  focal_alpha: [0.5, 0.4, 0.1]    # ⭐ Emphasize TC
  focal_gamma: 3.0                 # ⭐ More hard example focus
  class_weights: [1.0, 2.5, 1.0]   # ⭐ 2.5× TC weight

  scheduler: "cosine"
  warmup_steps: 2000               # ⭐ Longer warmup
  min_lr: 5.0e-6                   # ⭐ Higher min LR
  early_stop_patience: 75          # ⭐ More patience

model:
  # ... same as multiclass.yaml ...
```

**Run**:
```bash
python scripts/train.py --cfg configs/multiclass_focus_tc.yaml --fold 4
```

---

### Config B: Larger Model

Create `configs/multiclass_large.yaml`:
```yaml
exp_name: "braintumnet_multiclass_large"

model:
  in_channels: 4
  num_classes_seg: 3
  num_classes_cls: 2
  base: 48                  # ⭐ Increase from 32
  patch_size: 8
  dim: 384                  # ⭐ Increase from 256
  n_heads: 6                # ⭐ Increase from 4
  depth: 3                  # ⭐ Increase from 2
  roi_stop_grad: true
  deep_supervision: true

train:
  batch_size: 8             # ⭐ Reduce due to larger model
  # ... rest same as Config A ...
```

**Run**:
```bash
python scripts/train.py --cfg configs/multiclass_large.yaml --fold 4
```

---

### Config C: Pure Focal Loss

Create `configs/multiclass_focal_only.yaml`:
```yaml
train:
  loss_type: "multiclass_focal"   # ⭐ Drop Dice loss
  focal_alpha: [0.3, 0.4, 0.3]    # ⭐ Balanced weights
  focal_gamma: 3.0
  # ... rest same as Config A ...
```

---

## 📊 Expected Results Summary

| Approach | Expected WT Dice | Expected TC Dice | Effort |
|----------|-----------------|-----------------|--------|
| **Current (Fold 4)** | 0.8476 | 0.8199 | - |
| **Config A (Focus TC)** | 0.86-0.88 | 0.83-0.85 | 1 day |
| **Config B (Larger Model)** | 0.87-0.89 | 0.84-0.86 | 3 days |
| **Config A+B Combined** | 0.88-0.90 ✅ | 0.85-0.87 ✅ | 5 days |
| **+ TTA + Ensemble** | 0.89-0.92 ✅✅ | 0.86-0.88 ✅ | 1 week |

---

## 🔬 Diagnostic Commands

### Check Other Folds Performance
```bash
# Check if other folds perform better
grep "Best Metrics" logs/braintumnet_multiclass_3class_fold*.log

# Compare WT Dice across folds
for fold in 0 1 2 3 4; do
    echo "Fold $fold:"
    grep "WT_dice:" logs/braintumnet_multiclass_3class_fold${fold}*.log | tail -1
done
```

### Visualize Training Curves
```bash
# Launch TensorBoard
tensorboard --logdir=runs

# View learning curves, loss trends, sample predictions
```

### Check Data Distribution
```python
# Check if fold 4 has harder cases
import pandas as pd
df = pd.read_csv('data/processed_multiclass/val_fold4.csv')
print(df['case_id'].value_counts())  # Check case distribution
```

---

## 🎓 Key Takeaways

### What's Working Well ✅
1. **Edema segmentation** (0.8561) exceeds target by 7%
2. **Training stability** - smooth convergence, no collapse
3. **Classification perfect** (val_acc = 1.0)
4. **Overall mean Dice** (0.8412) is competitive

### What Needs Improvement ⚠️
1. **Whole Tumor** (0.8476) is 3.5% below target lower bound
2. **Tumor Core** (0.8199) is marginally below target
3. **Learning rate** decayed too quickly (hit zero by epoch 100)
4. **Model capacity** might be insufficient (14.3M params)

### Top 3 Recommendations 🎯
1. **Adjust loss weights** (focal_alpha, class_weights) to emphasize TC
2. **Increase model capacity** (base=48, dim=384, depth=3)
3. **Extend training** (min_lr=5e-6, patience=75, warmup=2000)

---

## 📚 References

- Your comparison doc: [COMPARISON_PATCHBASED_VS_BRAINTUMNET.md](COMPARISON_PATCHBASED_VS_BRAINTUMNET.md)
- Config file: [configs/multiclass.yaml](../configs/multiclass.yaml)
- Loss implementation: [src/braintumnet/losses_multiclass.py](../src/braintumnet/losses_multiclass.py)
- BraTS 2020 benchmark: https://www.med.upenn.edu/cbica/brats2020/

---

**Status**: Fold 4 shows strong performance overall, with excellent ED segmentation. With targeted improvements (especially TC focus), expect to reach 0.88-0.90 WT Dice target. 🚀
