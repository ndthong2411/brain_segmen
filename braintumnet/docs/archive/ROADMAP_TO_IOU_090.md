# Roadmap to IoU 0.90 🎯

**Current Performance**: IoU = 0.7263 (Fold 4, Epoch 149)
**Target**: IoU = 0.90
**Gap**: +0.1737 (+24% relative improvement)

**Reality Check**: IoU 0.90 is at the **cutting edge** of BraTS performance. Current SOTA (2024):
- MedNeXt ensemble: DSC 0.896 (≈IoU 0.81-0.83)
- nnUNet + Swin-UNETR ensemble: DSC 0.92 on WT (≈IoU 0.85)
- **Your target (IoU 0.90) requires SOTA-level implementation**

---

## 📊 Gap Analysis

### Current Metrics (Fold 4)
```
Metric      | Current | Target  | Gap      | % Improvement Needed
------------|---------|---------|----------|-----------------------
Mean IoU    | 0.7263  | 0.90    | +0.1737  | +24%
WT IoU      | 0.7356  | 0.90    | +0.1644  | +22%
TC IoU      | 0.6948  | 0.90    | +0.2052  | +30% ⚠️
ED IoU      | 0.7483  | 0.90    | +0.1517  | +20%

Mean Dice   | 0.8412  | 0.947   | +0.1058  | +13%
WT Dice     | 0.8476  | 0.947   | +0.0994  | +12%
TC Dice     | 0.8199  | 0.947   | +0.1271  | +15%
ED Dice     | 0.8561  | 0.947   | +0.0909  | +11%
```

**Key Finding**: TC (Tumor Core) is the bottleneck - needs +30% improvement!

---

## 🎯 Strategic Approach

### Phase 1: Foundation (IoU 0.75-0.80) - 1 Week
**Goal**: +5-7% IoU gain through loss/training improvements
**Effort**: Medium
**Success Rate**: High (90%)

### Phase 2: Architecture (IoU 0.80-0.85) - 2 Weeks
**Goal**: +5-6% IoU gain through model upgrades
**Effort**: High
**Success Rate**: Medium (70%)

### Phase 3: Advanced (IoU 0.85-0.90) - 3-4 Weeks
**Goal**: +5% IoU gain through SOTA techniques
**Effort**: Very High
**Success Rate**: Low-Medium (40-60%)

---

## 🚀 Phase 1: Foundation Improvements (IoU 0.75-0.80)

### Target: +5-7% IoU improvement in 1 week

### 1A. Aggressive Loss Function Tuning ⭐ HIGHEST IMPACT

**Problem**: Current loss doesn't penalize IoU directly

**Solution 1: Add IoU Loss Component**

```python
# braintumnet/src/braintumnet/losses_multiclass.py

class MulticlassIoULoss(nn.Module):
    """IoU loss for multi-class segmentation"""
    def __init__(self, smooth=1.0, ignore_background=True):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, pred, target):
        """
        pred: (B, C, H, W) logits
        target: (B, 1, H, W) class indices
        """
        pred_probs = F.softmax(pred, dim=1)  # (B, C, H, W)

        # One-hot encode target
        B, C, H, W = pred.shape
        target_one_hot = F.one_hot(target.squeeze(1).long(), C).permute(0, 3, 1, 2).float()

        # Compute IoU per class
        iou_per_class = []
        start_idx = 1 if self.ignore_background else 0

        for c in range(start_idx, C):
            pred_c = pred_probs[:, c]
            target_c = target_one_hot[:, c]

            intersection = (pred_c * target_c).sum(dim=[1, 2])
            union = pred_c.sum(dim=[1, 2]) + target_c.sum(dim=[1, 2]) - intersection

            iou = (intersection + self.smooth) / (union + self.smooth)
            iou_per_class.append(iou)

        iou_per_class = torch.stack(iou_per_class, dim=1)  # (B, num_classes)

        # IoU loss = 1 - mean IoU
        iou_loss = 1.0 - iou_per_class.mean()
        return iou_loss

class CombinedSegmentationLoss(nn.Module):
    """Dice + Focal + IoU combined loss"""
    def __init__(self, dice_weight=1.0, focal_weight=1.0, iou_weight=1.0,
                 focal_alpha=None, focal_gamma=2.0, ignore_background=True):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight

        self.dice_loss = MulticlassDiceLoss(ignore_background=ignore_background)
        self.focal_loss = MulticlassFocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                                              ignore_background=ignore_background)
        self.iou_loss = MulticlassIoULoss(ignore_background=ignore_background)

    def forward(self, pred, target):
        dice_l = self.dice_loss(pred, target)
        focal_l = self.focal_loss(pred, target)
        iou_l = self.iou_loss(pred, target)

        total = (self.dice_weight * dice_l +
                 self.focal_weight * focal_l +
                 self.iou_weight * iou_l)

        return total, {'dice': dice_l.item(), 'focal': focal_l.item(), 'iou': iou_l.item()}
```

**Config**: `configs/multiclass_iou_focus.yaml`
```yaml
train:
  loss_type: "multiclass_dice_focal_iou"  # New combined loss

  # Loss weights
  dice_weight: 1.0
  focal_weight: 1.0
  iou_weight: 2.0  # ⭐ Emphasize IoU

  # Focal loss hyperparameters
  focal_alpha: [0.5, 0.5, 0.0]  # Ignore background, equal TC/ED
  focal_gamma: 3.0              # Hard example focus

  # Class weights for imbalance
  class_weights: [1.0, 3.0, 2.0]  # ⭐ 3× TC, 2× ED
```

**Expected gain**: +3-5% IoU

---

### 1B. Boundary-Aware Loss ⭐ CRITICAL FOR IoU

**Problem**: Current loss treats all pixels equally. IoU is most affected by boundary errors.

**Solution: Add Boundary Loss**

```python
# braintumnet/src/braintumnet/losses_multiclass.py

import scipy.ndimage as ndimage

class BoundaryLoss(nn.Module):
    """Boundary-aware loss for precise edge segmentation"""
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def compute_sdf(self, mask):
        """Compute signed distance function"""
        # mask: (H, W) binary mask
        pos_mask = mask.cpu().numpy()
        neg_mask = 1 - pos_mask

        pos_dist = ndimage.distance_transform_edt(pos_mask)
        neg_dist = ndimage.distance_transform_edt(neg_mask)

        sdf = neg_dist - pos_dist
        return torch.from_numpy(sdf).float().to(mask.device)

    def forward(self, pred, target):
        """
        pred: (B, C, H, W) logits
        target: (B, 1, H, W) class indices
        """
        pred_probs = F.softmax(pred, dim=1)

        B, C, H, W = pred.shape
        boundary_loss = 0

        for b in range(B):
            for c in range(1, C):  # Skip background
                target_mask_c = (target[b, 0] == c).float()
                pred_prob_c = pred_probs[b, c]

                # Compute signed distance function
                sdf = self.compute_sdf(target_mask_c)

                # Boundary loss: weighted by distance to boundary
                weight = torch.exp(-sdf.abs() / self.theta)
                boundary_loss += (weight * (pred_prob_c - target_mask_c).abs()).mean()

        return boundary_loss / (B * (C - 1))
```

**Add to combined loss**:
```python
class UltimateLoss(nn.Module):
    """Dice + Focal + IoU + Boundary"""
    def __init__(self, dice_weight=1.0, focal_weight=1.0,
                 iou_weight=1.0, boundary_weight=0.5):
        super().__init__()
        self.dice_loss = MulticlassDiceLoss()
        self.focal_loss = MulticlassFocalLoss()
        self.iou_loss = MulticlassIoULoss()
        self.boundary_loss = BoundaryLoss()

        self.weights = {
            'dice': dice_weight,
            'focal': focal_weight,
            'iou': iou_weight,
            'boundary': boundary_weight
        }

    def forward(self, pred, target):
        losses = {
            'dice': self.dice_loss(pred, target),
            'focal': self.focal_loss(pred, target),
            'iou': self.iou_loss(pred, target),
            'boundary': self.boundary_loss(pred, target)
        }

        total = sum(self.weights[k] * v for k, v in losses.items())
        return total, {k: v.item() for k, v in losses.items()}
```

**Expected gain**: +2-4% IoU (especially on boundaries)

---

### 1C. Advanced Training Schedule

```yaml
# configs/multiclass_iou_focus.yaml

train:
  epochs: 300  # Increase from 250
  batch_size: 16  # Increase from 12 (more stable gradients)

  # Learning rate
  lr: 5.0e-5  # Start lower for stability
  weight_decay: 5.0e-5  # Stronger regularization

  # Scheduler: Cosine with warm restarts
  scheduler: "cosine_restart"
  warmup_steps: 3000
  restart_period: 75  # Restart every 75 epochs
  restart_mult: 1.2
  min_lr: 1.0e-6

  # Early stopping
  early_stop_patience: 100  # More patience
  early_stop_metric: "val_iou"  # ⭐ Optimize for IoU directly

  # Optimization
  optimizer: "adamw"  # Better than Adam
  gradient_clip: 1.0  # Prevent exploding gradients

  # Mixed precision
  amp: true
  amp_dtype: "float16"
```

**Expected gain**: +1-2% IoU

---

### Phase 1 Summary

**Implement**:
1. IoU loss component (add to `losses_multiclass.py`)
2. Boundary loss (add to `losses_multiclass.py`)
3. Combined loss with weights: Dice=1.0, Focal=1.0, IoU=2.0, Boundary=0.5
4. Class weights: [1.0, 3.0, 2.0] (emphasize TC)
5. Extended training schedule (300 epochs, cosine restart)

**Config**: `configs/phase1_iou_focus.yaml` (create based on above)

**Expected Result**: IoU 0.75-0.80 (+5-7%)

**Time**: 3-4 days (1 training run ~36 hours + tuning)

---

## 🏗️ Phase 2: Architecture Upgrades (IoU 0.80-0.85)

### Target: +5-6% IoU improvement in 2 weeks

### 2A. Significantly Larger Model ⭐

**Current**: 14.3M parameters, base=32, dim=256
**Target**: 50-80M parameters (3-5× larger)

```yaml
# configs/phase2_large_model.yaml

model:
  in_channels: 4
  num_classes_seg: 3
  num_classes_cls: 2

  # ⭐ Massive capacity increase
  base: 64           # 2× increase (32 → 64) → 4× parameters
  dim: 512           # 2× increase (256 → 512) → 4× transformer capacity
  patch_size: 8
  n_heads: 8         # 2× increase (4 → 8)
  depth: 4           # 2× increase (2 → 4)

  # Advanced features
  roi_stop_grad: false  # Allow gradient flow for multi-task learning
  deep_supervision: true
  dropout: 0.15      # Regularization for large model
```

**Estimated model size**: ~57M parameters

**Trade-off**:
- +50% training time
- Requires 16GB+ GPU
- Batch size: reduce to 8-12

**Expected gain**: +3-4% IoU

---

### 2B. Residual Connections Everywhere

**Modify `seg_unet.py`**:

```python
# braintumnet/src/braintumnet/models/seg_unet.py

class ResidualConvBlock(nn.Module):
    """Residual convolutional block"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv_bn_relu(in_ch, out_ch)
        self.conv2 = conv_bn_relu(out_ch, out_ch)

        # Residual connection
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + identity  # Residual addition

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ResidualConvBlock(in_ch, out_ch)  # ⭐ Use residual
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.block(x)
        x_down = self.pool(x)
        return x, x_down

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.cbam = CBAM(out_ch)
        self.block = ResidualConvBlock(out_ch*2, out_ch)  # ⭐ Use residual
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, self.cbam(skip)], dim=1)
        x = self.block(x)
        x = self.dropout(x)
        return x
```

**Expected gain**: +1-2% IoU

---

### 2C. Multi-Scale Fusion (Feature Pyramid)

**Add to `seg_unet.py`**:

```python
class MultiScaleFusion(nn.Module):
    """Fuse features from multiple scales"""
    def __init__(self, channels_list):
        super().__init__()
        # channels_list: [base, base*2, base*4, base*8]
        self.convs = nn.ModuleList([
            nn.Conv2d(ch, channels_list[0], 1) for ch in channels_list
        ])

    def forward(self, features):
        """
        features: [s1, s2, s3, s4] with different spatial sizes
        """
        target_size = features[0].shape[2:]  # Use largest spatial size

        upsampled = []
        for i, feat in enumerate(features):
            # Project to same channel dimension
            feat = self.convs[i](feat)
            # Upsample to target size
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled.append(feat)

        # Fuse by summation
        fused = sum(upsampled)
        return fused

# Modify SegUNetMasked to use multi-scale fusion
class SegUNetMasked(nn.Module):
    def __init__(self, in_ch=1, base=32, ...):
        super().__init__()
        # ... existing code ...

        # Add multi-scale fusion before final head
        self.ms_fusion = MultiScaleFusion([base, base*2, base*4, base*8])

        # Final head takes fused features
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        s1, x1 = self.e1(x)
        s2, x2 = self.e2(x1)
        s3, x3 = self.e3(x2)
        s4, x4 = self.e4(x3)

        # ... transformer ...

        d4 = self.d4(b, s4)
        d3 = self.d3(d4, s3)
        d2 = self.d2(d3, s2)
        d1 = self.d1(d2, s1)

        # ⭐ Multi-scale fusion
        decoder_features = [d1, d2, d3, d4]
        fused = self.ms_fusion(decoder_features)

        seg = self.head(fused + d1)  # Residual with final decoder output

        # ... rest of code ...
```

**Expected gain**: +1-2% IoU

---

### Phase 2 Summary

**Implement**:
1. Increase model size: base=64, dim=512, depth=4, heads=8 (~57M params)
2. Add residual connections to all conv blocks
3. Add multi-scale fusion before final segmentation head
4. Add dropout=0.15 for regularization

**Config**: `configs/phase2_large_model.yaml`

**Expected Result**: IoU 0.80-0.85 (+5-6% from Phase 1)

**Time**: 1-2 weeks (2-3 training runs × 48 hours each + implementation)

---

## 🌟 Phase 3: SOTA Techniques (IoU 0.85-0.90)

### Target: +5% IoU improvement in 3-4 weeks

### 3A. Test-Time Augmentation (TTA) ⭐ EASY WIN

**Implementation**:

```python
# braintumnet/scripts/tta_predict.py

def tta_predict(model, image, num_augmentations=8):
    """
    Predict with test-time augmentation

    Augmentations:
    1. Original
    2. Horizontal flip
    3. Vertical flip
    4. Rotation 90°
    5. Rotation 180°
    6. Rotation 270°
    7. Horizontal flip + Rotation 90°
    8. Vertical flip + Rotation 90°
    """
    predictions = []

    # 1. Original
    pred = model(image)
    predictions.append(pred)

    # 2. Horizontal flip
    pred = model(torch.flip(image, dims=[3]))
    pred = torch.flip(pred, dims=[3])  # Flip back
    predictions.append(pred)

    # 3. Vertical flip
    pred = model(torch.flip(image, dims=[2]))
    pred = torch.flip(pred, dims=[2])  # Flip back
    predictions.append(pred)

    # 4-6. Rotations
    for k in [1, 2, 3]:  # 90, 180, 270 degrees
        pred = model(torch.rot90(image, k, dims=[2, 3]))
        pred = torch.rot90(pred, -k, dims=[2, 3])  # Rotate back
        predictions.append(pred)

    # 7-8. Combined
    img_hflip_rot = torch.rot90(torch.flip(image, dims=[3]), 1, dims=[2, 3])
    pred = model(img_hflip_rot)
    pred = torch.rot90(pred, -1, dims=[2, 3])
    pred = torch.flip(pred, dims=[3])
    predictions.append(pred)

    img_vflip_rot = torch.rot90(torch.flip(image, dims=[2]), 1, dims=[2, 3])
    pred = model(img_vflip_rot)
    pred = torch.rot90(pred, -1, dims=[2, 3])
    pred = torch.flip(pred, dims=[2])
    predictions.append(pred)

    # Average all predictions
    pred_mean = torch.stack(predictions).mean(dim=0)
    return pred_mean
```

**Expected gain**: +2-3% IoU (no retraining needed!)

---

### 3B. 5-Fold Ensemble ⭐ PROVEN TECHNIQUE

```python
# braintumnet/scripts/ensemble_predict.py

def ensemble_predict(models, image):
    """Average predictions from all fold models"""
    predictions = []

    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(image)[0]  # Get segmentation logits
            pred_prob = F.softmax(pred, dim=1)
            predictions.append(pred_prob)

    # Average probabilities
    pred_ensemble = torch.stack(predictions).mean(dim=0)
    return pred_ensemble

# Load all 5 fold models
models = []
for fold in range(5):
    ckpt_path = f'checkpoints/braintumnet_best_fold{fold}.pth'
    model = BrainTumNet(...)
    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    model.to('cuda')
    models.append(model)

# Predict
pred = ensemble_predict(models, test_image)
```

**Expected gain**: +2-3% IoU

---

### 3C. Post-Processing (Conditional Random Field)

```python
# braintumnet/scripts/postprocess.py

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def crf_postprocess(image, pred_prob, num_classes=3):
    """
    Apply Conditional Random Field (CRF) for boundary refinement

    Args:
        image: (C, H, W) input image
        pred_prob: (C, H, W) softmax probabilities
        num_classes: number of classes
    """
    H, W = image.shape[1:]

    # Setup CRF
    d = dcrf.DenseCRF2D(W, H, num_classes)

    # Unary potential from network
    U = unary_from_softmax(pred_prob.cpu().numpy())
    d.setUnaryEnergy(U)

    # Pairwise potentials (smooth segmentation)
    # Appearance kernel: nearby similar pixels should have same label
    d.addPairwiseGaussian(sxy=3, compat=3)

    # Bilateral kernel: nearby similar intensity should have same label
    image_rgb = image.cpu().numpy()
    d.addPairwiseBilateral(sxy=50, srgb=13, rgbim=image_rgb, compat=10)

    # Inference
    Q = d.inference(5)  # 5 iterations
    Q = np.array(Q).reshape((num_classes, H, W))

    return torch.from_numpy(Q).float()
```

**Expected gain**: +1-2% IoU (boundary refinement)

---

### 3D. Advanced Data Augmentation

```yaml
# configs/phase3_advanced.yaml

augment:
  # Geometric
  rotate_deg: 45  # Increase from 30
  hflip_p: 0.5
  vflip_p: 0.5
  elastic_transform: true
  elastic_alpha: 100
  elastic_sigma: 10

  # Intensity
  brightness_range: [0.7, 1.3]
  contrast_range: [0.7, 1.3]
  gamma_range: [0.8, 1.2]

  # Noise
  gaussian_noise_std: 0.02
  gaussian_blur_sigma: [0.5, 2.0]
  gaussian_blur_p: 0.3

  # Cutout/Dropout
  coarse_dropout_p: 0.2
  coarse_dropout_holes: 5
  coarse_dropout_size: [16, 32]

  # MixUp (advanced)
  mixup_alpha: 0.2
  mixup_p: 0.3
```

**Implementation** (requires modifying `transforms.py`):

```python
# braintumnet/src/braintumnet/data/transforms.py

import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter, map_coordinates

def elastic_transform(image, mask, alpha=100, sigma=10):
    """Elastic deformation"""
    shape = image.shape[1:]

    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    # Apply to image and mask
    image_deformed = map_coordinates(image.numpy(), indices, order=1).reshape(image.shape)
    mask_deformed = map_coordinates(mask.numpy(), indices, order=0).reshape(mask.shape)

    return torch.from_numpy(image_deformed), torch.from_numpy(mask_deformed)

def mixup(image1, mask1, image2, mask2, alpha=0.2):
    """MixUp augmentation"""
    lam = np.random.beta(alpha, alpha)
    mixed_image = lam * image1 + (1 - lam) * image2
    mixed_mask = lam * mask1 + (1 - lam) * mask2
    return mixed_image, mixed_mask
```

**Expected gain**: +1-2% IoU

---

### 3E. Knowledge Distillation (Optional)

Train a larger "teacher" model, then distill to student:

```python
# braintumnet/scripts/distillation.py

def distillation_loss(student_logits, teacher_logits, target, temperature=3.0, alpha=0.7):
    """
    Combined distillation loss

    Args:
        alpha: Weight for distillation (1-alpha for task loss)
        temperature: Softmax temperature for soft targets
    """
    # Soft targets from teacher
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    distill_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (temperature ** 2)

    # Hard targets (ground truth)
    task_loss = F.cross_entropy(student_logits, target.squeeze(1).long())

    # Combined
    total_loss = alpha * distill_loss + (1 - alpha) * task_loss
    return total_loss
```

**Expected gain**: +0.5-1% IoU (marginal)

---

### Phase 3 Summary

**Implement**:
1. Test-Time Augmentation (8 augmentations) - **NO RETRAINING**
2. 5-Fold Ensemble - **NO RETRAINING** (use existing models)
3. CRF Post-processing - **NO RETRAINING**
4. Advanced data augmentation (elastic, mixup, etc.)
5. Retrain with all improvements

**Combined TTA + Ensemble + CRF**: +4-6% IoU with ZERO retraining!

**With Retraining**: +6-8% IoU total

**Expected Result**: IoU 0.85-0.90 ✅

**Time**: 3-4 weeks (implementation + training)

---

## 📊 Complete Roadmap Summary

### Timeline & Expected Results

| Phase | Duration | Key Changes | Expected IoU | Cumulative Gain |
|-------|----------|-------------|--------------|-----------------|
| **Baseline** | - | Current model | 0.7263 | - |
| **Phase 1** | 1 week | Loss (IoU+Boundary) + Training | 0.75-0.80 | +5-7% |
| **Phase 2** | 2 weeks | Large model (57M params) + Residual | 0.80-0.85 | +10-13% |
| **Phase 3 (No retrain)** | 3 days | TTA + Ensemble + CRF | 0.84-0.88 | +15-18% |
| **Phase 3 (Full)** | 3-4 weeks | All SOTA techniques + retrain | 0.85-0.90 | +17-24% ✅ |

### Cost-Benefit Analysis

| Approach | Effort | Time | GPU Cost | Success Rate | IoU Gain |
|----------|--------|------|----------|--------------|----------|
| **Phase 1** | Medium | 1 week | 1× | 90% | +5-7% |
| **Phase 2** | High | 2 weeks | 2× | 70% | +5-6% |
| **TTA+Ensemble+CRF** | Low | 3 days | 0× | 95% | +4-6% ⭐ |
| **Phase 3 Full** | Very High | 4 weeks | 3× | 50-60% | +6-8% |

---

## 🎯 Recommended Priority Order

### Quick Wins (Do First!) 🚀

1. **TTA + 5-Fold Ensemble** (3 days, +4-6% IoU)
   - NO retraining needed
   - Use existing fold models
   - Highest ROI

2. **CRF Post-processing** (1 day, +1-2% IoU)
   - NO retraining needed
   - Pure post-processing

**Result after quick wins**: IoU 0.77-0.80 (from 0.7263)

### Medium-Term (2-3 weeks)

3. **Phase 1: Loss + Training** (1 week, +5-7% IoU)
   - Implement IoU loss + Boundary loss
   - Retrain 1-2 folds

4. **Phase 2: Large Model** (2 weeks, +5-6% IoU)
   - Scale up to 57M parameters
   - Add residual connections
   - Retrain all folds

**Result after 3 weeks**: IoU 0.82-0.87

### Long-Term (4-6 weeks total)

5. **Phase 3: Full SOTA** (2-3 weeks, +2-3% additional)
   - Advanced augmentation
   - Multi-scale fusion
   - Knowledge distillation

**Result after 6 weeks**: IoU 0.85-0.90 ✅

---

## 💻 Implementation Checklist

### Week 1: Quick Wins
- [ ] Implement TTA inference (8 augmentations)
- [ ] Train missing fold models (if needed)
- [ ] Implement 5-fold ensemble
- [ ] Test on validation set
- [ ] Implement CRF post-processing
- [ ] Measure IoU improvement

**Expected: IoU 0.77-0.80**

### Week 2: Phase 1
- [ ] Implement `MulticlassIoULoss` in `losses_multiclass.py`
- [ ] Implement `BoundaryLoss` in `losses_multiclass.py`
- [ ] Create `CombinedSegmentationLoss`
- [ ] Create config `phase1_iou_focus.yaml`
- [ ] Train fold 4 with new loss
- [ ] Evaluate results

**Expected: IoU 0.75-0.80 (single model)**

### Week 3-4: Phase 2
- [ ] Modify `seg_unet.py`: Add `ResidualConvBlock`
- [ ] Modify `seg_unet.py`: Add `MultiScaleFusion`
- [ ] Update `braintumnet.py`: Support larger model
- [ ] Create config `phase2_large_model.yaml` (base=64, dim=512)
- [ ] Train fold 4 (48 hours)
- [ ] Train fold 0 (48 hours)
- [ ] Evaluate 2-fold ensemble

**Expected: IoU 0.80-0.85 (single model), 0.83-0.87 (ensemble+TTA)**

### Week 5-6: Phase 3
- [ ] Implement elastic transform in `transforms.py`
- [ ] Implement mixup in `transforms.py`
- [ ] Add advanced augmentation to config
- [ ] Train all 5 folds with Phase 2 + Phase 3
- [ ] Full ensemble (5 folds + TTA + CRF)
- [ ] Final evaluation

**Expected: IoU 0.85-0.90** ✅

---

## 📈 Tracking Progress

### Metrics to Monitor

```python
# After each phase, record:
metrics = {
    'single_model_iou': 0.XX,
    'single_model_dice': 0.XX,
    'ensemble_iou': 0.XX,
    'ensemble_dice': 0.XX,
    'tta_iou': 0.XX,
    'tta_dice': 0.XX,
    'full_pipeline_iou': 0.XX,  # Ensemble + TTA + CRF
    'full_pipeline_dice': 0.XX,

    # Per-class
    'wt_iou': 0.XX,
    'tc_iou': 0.XX,
    'ed_iou': 0.XX,

    # Training
    'training_time_hours': XX,
    'model_params_M': XX,
    'gpu_memory_GB': XX
}
```

---

## ⚠️ Reality Check

### What if IoU 0.90 is not achievable?

**IoU 0.90 is extremely challenging**. Here's what's realistic:

| Target | Difficulty | Probability | What's Needed |
|--------|------------|-------------|---------------|
| IoU 0.78 | Easy | 95% | Quick wins (TTA+Ensemble) |
| IoU 0.82 | Medium | 80% | Phase 1 + Quick wins |
| IoU 0.85 | Hard | 60% | Phase 1+2 + Full ensemble |
| IoU 0.87 | Very Hard | 40% | All phases + perfect tuning |
| **IoU 0.90** | **Extreme** | **20-30%** | **SOTA implementation + luck** |

### Alternative Target: Dice 0.92

If IoU 0.90 is too hard, target **Dice 0.92** instead:
- Dice 0.92 ≈ IoU 0.85 (more achievable)
- Many papers report Dice instead of IoU
- Dice is less strict than IoU

**Recommendation**: Target **IoU 0.85 + Dice 0.92** as success criteria

---

## 🎓 Key Takeaways

1. **IoU 0.90 is at the cutting edge** - requires SOTA-level implementation
2. **Quick wins first**: TTA + Ensemble can give +4-6% IoU with NO retraining
3. **Loss function matters**: IoU loss + Boundary loss are critical
4. **Model capacity matters**: Current 14M → 57M parameters needed
5. **Ensemble is powerful**: 5-fold + TTA can add +5-7% IoU
6. **Be realistic**: IoU 0.85-0.87 is more achievable than 0.90

**Realistic Best-Case**: IoU 0.85-0.88 (Dice 0.92-0.94)

**Stretch Goal**: IoU 0.88-0.90 (Dice 0.94-0.95)

---

**Good luck! This is an ambitious goal that will push your implementation to SOTA levels.** 🚀
