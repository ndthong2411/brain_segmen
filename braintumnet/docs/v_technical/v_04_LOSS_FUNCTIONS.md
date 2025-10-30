# Phần 4: Loss Functions và Metrics

> **📐 Các Hàm Loss và Metrics Cho Multi-Class Segmentation**
>
> Tài liệu này giải thích chi tiết Dice Loss, Focal Loss, Combined Loss và Multi-Class Metrics.

---

## Mục Lục

1. [Tổng Quan Loss Functions](#1-tổng-quan-loss-functions)
2. [Dice Loss](#2-dice-loss)
3. [Focal Loss](#3-focal-loss)
4. [Combined Loss](#4-combined-loss)
5. [Deep Supervision Loss](#5-deep-supervision-loss)
6. [Multi-Class Metrics](#6-multi-class-metrics)
7. [Metrics Accumulation](#7-metrics-accumulation)

---

## 1. Tổng Quan Loss Functions

### Tại Sao Cần Multiple Losses?

**Segmentation khó vì**:
1. **Class imbalance**: Background >> Tumor
2. **Small objects**: Tumor nhỏ, dễ bị miss
3. **Boundary accuracy**: Cần sharp boundaries

**Giải pháp**: Kết hợp multiple losses

```
Combined Loss = α × Dice Loss + β × Focal Loss

Dice Loss:
- Optimize overlap (IoU-based)
- Handle class imbalance tốt
- Focus on entire region

Focal Loss:
- Focus on hard examples
- Down-weight easy pixels
- Better cho boundaries
```

### File Code Structure

```
src/braintumnet/
├── losses_multiclass.py        # Multi-class loss functions
│   ├── MultiClassDiceLoss
│   ├── MultiClassFocalLoss
│   └── MultiClassCombinedLoss
│
└── multiclass_metrics.py       # BraTS metrics
    ├── compute_dice_score
    ├── compute_hausdorff_distance
    └── MulticlassMetricsAccumulator
```

---

## 2. Dice Loss

### File Code

**File**: `src/braintumnet/losses_multiclass.py` (133 dòng)

### Dice Coefficient

**Công thức**:
```
Dice(P, G) = 2 × |P ∩ G| / (|P| + |G|)

P: Prediction set
G: Ground truth set
∩: Intersection
|·|: Cardinality (số elements)

Giá trị: [0, 1]
- 0: No overlap
- 1: Perfect overlap
```

**Ví dụ**:
```
Ground Truth:        Prediction:
┌────────┐          ┌────────┐
│  0000  │          │  0000  │
│  0110  │          │  0100  │
│  0110  │          │  0110  │
│  0000  │          │  0000  │
└────────┘          └────────┘

|G| = 4 pixels (label 1)
|P| = 3 pixels (predicted 1)
|P ∩ G| = 2 pixels (overlap)

Dice = 2×2 / (4+3) = 4/7 = 0.571
```

### Dice Loss Implementation

```python
class MultiClassDiceLoss(nn.Module):
    """
    Multi-class Dice Loss
    
    Compute Dice cho mỗi class, sau đó average
    
    Loss = 1 - Dice
    """
    def __init__(self, num_classes=3, smooth=1.0, 
                 weight=None, ignore_background=True):
        """
        Args:
            num_classes: Số classes (3 cho BraTS)
            smooth: Smoothing factor (tránh division by zero)
            weight: Class weights (None → uniform)
            ignore_background: Bỏ qua class 0 khi tính loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_background = ignore_background
        
        # Class weights
        if weight is None:
            self.weight = torch.ones(num_classes)
        else:
            self.weight = torch.tensor(weight)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - raw model outputs
            targets: (B, H, W) - class indices
        
        Returns:
            loss: scalar tensor
        """
        # Softmax để convert logits → probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(
            targets, num_classes=self.num_classes
        )  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Decide classes to compute
        if self.ignore_background:
            start_cls = 1
        else:
            start_cls = 0
        
        # Compute Dice per class
        dice_scores = []
        for cls in range(start_cls, self.num_classes):
            # Extract class probabilities và targets
            pred_cls = probs[:, cls, :, :]      # (B, H, W)
            target_cls = targets_one_hot[:, cls, :, :]  # (B, H, W)
            
            # Intersection và union
            intersection = (pred_cls * target_cls).sum(dim=[1, 2])  # (B,)
            pred_sum = pred_cls.sum(dim=[1, 2])       # (B,)
            target_sum = target_cls.sum(dim=[1, 2])   # (B,)
            
            # Dice per sample
            dice = (2.0 * intersection + self.smooth) / (
                pred_sum + target_sum + self.smooth
            )  # (B,)
            
            # Average over batch
            dice_scores.append(dice.mean())
        
        # Weighted average over classes
        dice_scores = torch.stack(dice_scores)  # (num_classes-start,)
        weights = self.weight[start_cls:].to(dice_scores.device)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Weighted mean Dice
        mean_dice = (dice_scores * weights).sum()
        
        # Loss = 1 - Dice
        loss = 1.0 - mean_dice
        
        return loss
```

### Tại Sao Smooth?

```python
# Không có smooth
intersection = 0
pred_sum = 0
target_sum = 0

dice = 2 × 0 / (0 + 0) = 0/0 = NaN  ❌

# Có smooth=1
dice = (2×0 + 1) / (0 + 0 + 1) = 1/1 = 1  ✓

→ Tránh division by zero
→ Slight bias towards higher Dice (acceptable)
```

### Multi-Class Dice Calculation

**Example batch**:
```python
# Batch of 4 samples, 3 classes, 4×4 spatial
logits = torch.randn(4, 3, 4, 4)
targets = torch.randint(0, 3, (4, 4, 4))

loss_fn = MultiClassDiceLoss(num_classes=3, ignore_background=True)
loss = loss_fn(logits, targets)

# Internal computation:
# 1. Softmax: logits → probs (B, 3, H, W)
# 2. One-hot: targets → (B, 3, H, W)
# 3. For class 1 (Tumor Core):
#      intersection_cls1 = (probs[:,1]*target[:,1]).sum()
#      dice_cls1 = 2*intersection / (pred_sum + target_sum)
# 4. For class 2 (Edema):
#      dice_cls2 = ...
# 5. Mean: (dice_cls1 + dice_cls2) / 2
# 6. Loss: 1 - mean_dice
```

---

## 3. Focal Loss

### Motivation

**Cross-Entropy problem**:
```
CE Loss = -log(p_correct)

Easy example:  p=0.9 → CE=0.105 (still contributes)
Hard example:  p=0.1 → CE=2.303 (high loss)

Problem: Easy examples dominate gradient!
```

**Focal Loss solution**:
```
FL = -(1-p)^γ × log(p)

Easy:  p=0.9 → (1-0.9)^2 = 0.01 → FL=0.001  (down-weighted!)
Hard:  p=0.1 → (1-0.1)^2 = 0.81 → FL=1.866  (emphasized)

γ=2 (typical): Focus on hard examples
```

### Implementation

```python
class MultiClassFocalLoss(nn.Module):
    """
    Multi-class Focal Loss
    
    FL = -α × (1-p_t)^γ × log(p_t)
    
    α: Class weights
    γ: Focusing parameter (2.0 default)
    p_t: Probability của correct class
    """
    def __init__(self, num_classes=3, alpha=None, gamma=2.0, 
                 ignore_background=True):
        """
        Args:
            num_classes: Số classes
            alpha: Class weights (None → uniform)
            gamma: Focusing parameter (2.0 recommended)
            ignore_background: Ignore class 0
        """
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ignore_background = ignore_background
        
        # Alpha weights
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            self.alpha = torch.tensor(alpha)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
        
        Returns:
            loss: scalar
        """
        # Softmax probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        
        # Get probabilities của correct class
        B, C, H, W = logits.shape
        targets_flat = targets.view(-1)  # (B×H×W,)
        probs_flat = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (B×H×W, C)
        
        # Extract p_t (probability của true class)
        p_t = probs_flat[range(len(targets_flat)), targets_flat]  # (B×H×W,)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # CE loss: -log(p_t)
        ce_loss = -torch.log(p_t + 1e-8)
        
        # Class weights
        alpha = self.alpha.to(logits.device)
        alpha_t = alpha[targets_flat]  # (B×H×W,)
        
        # Focal loss
        focal_loss = alpha_t * focal_weight * ce_loss  # (B×H×W,)
        
        # Optional: Ignore background
        if self.ignore_background:
            mask = (targets_flat != 0).float()
            focal_loss = focal_loss * mask
            loss = focal_loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = focal_loss.mean()
        
        return loss
```

### Focal Loss Example

```python
# Easy pixel (high confidence)
p_correct = 0.95
focal_weight = (1 - 0.95)**2 = 0.0025
ce = -log(0.95) = 0.051
focal_loss = 0.0025 × 0.051 = 0.000128

# Hard pixel (low confidence)
p_correct = 0.55
focal_weight = (1 - 0.55)**2 = 0.2025
ce = -log(0.55) = 0.598
focal_loss = 0.2025 × 0.598 = 0.121

→ Hard pixel có loss ~945× lớn hơn easy pixel!
→ Gradient focus vào hard examples
```

### Gamma Parameter Effect

```python
# γ = 0: Focal → CrossEntropy (no modulation)
# γ = 1: Mild focus
# γ = 2: Standard (recommended)
# γ = 5: Very aggressive focus

p_t = 0.8

γ=0: (1-0.8)^0 = 1.0     → No change
γ=1: (1-0.8)^1 = 0.2     → Moderate reduction
γ=2: (1-0.8)^2 = 0.04    → Strong reduction
γ=5: (1-0.8)^5 = 0.00032 → Very strong reduction

→ Higher γ: More focus on hard examples
→ γ=2: Good balance cho medical imaging
```

---

## 4. Combined Loss

### Rationale

**Tại sao kết hợp Dice + Focal?**

```
Dice Loss:
✓ Region-based optimization
✓ Handle class imbalance
✓ Optimize overlap directly
✗ Weak on boundaries
✗ Less sensitive to small errors

Focal Loss:
✓ Pixel-wise accuracy
✓ Focus on hard pixels
✓ Better boundaries
✗ Sensitive to class imbalance
✗ Can overfit easy classes

Combined:
✓✓ Best of both worlds!
```

### Implementation

```python
class MultiClassCombinedLoss(nn.Module):
    """
    Combined Dice + Focal Loss
    
    Loss = λ_dice × Dice + λ_focal × Focal
    
    Default: λ_dice=1.0, λ_focal=1.0
    """
    def __init__(
        self, 
        num_classes=3,
        dice_weight=1.0,
        focal_weight=1.0,
        class_weights=None,
        ignore_background=True,
        dice_smooth=1.0,
        focal_gamma=2.0
    ):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # Dice loss
        self.dice_loss = MultiClassDiceLoss(
            num_classes=num_classes,
            smooth=dice_smooth,
            weight=class_weights,
            ignore_background=ignore_background
        )
        
        # Focal loss
        self.focal_loss = MultiClassFocalLoss(
            num_classes=num_classes,
            alpha=class_weights,
            gamma=focal_gamma,
            ignore_background=ignore_background
        )
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
        
        Returns:
            loss: scalar
            loss_dict: {'dice': ..., 'focal': ..., 'total': ...}
        """
        # Compute individual losses
        dice_loss = self.dice_loss(logits, targets)
        focal_loss = self.focal_loss(logits, targets)
        
        # Weighted sum
        total_loss = (
            self.dice_weight * dice_loss + 
            self.focal_weight * focal_loss
        )
        
        # Return both total và breakdown
        loss_dict = {
            'dice': dice_loss.item(),
            'focal': focal_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict
```

### Hyperparameter Tuning

**Class weights**:
```python
# Từ data statistics (Phần 3):
# Background: 87.35% → weight = 0.344
# Tumor Core: 5.12% → weight = 5.865
# Edema: 7.53% → weight = 3.981

class_weights = [0.344, 5.865, 3.981]

loss_fn = MultiClassCombinedLoss(
    num_classes=3,
    class_weights=class_weights,
    ignore_background=True
)
```

**Loss weights tuning**:
```python
# Balanced (default)
dice_weight = 1.0
focal_weight = 1.0

# Emphasize overlap
dice_weight = 2.0
focal_weight = 1.0

# Emphasize boundaries
dice_weight = 1.0
focal_weight = 2.0

# Typical range: 0.5 - 2.0 cho mỗi weight
```

---

## 5. Deep Supervision Loss

### Motivation

**Problem**: Deep networks có gradient vanishing
**Solution**: Supervise ở intermediate layers

```
Output Layer:        Loss_main
    ↑
Decoder 1 (256×256): Loss_aux1
    ↑
Decoder 2 (128×128): Loss_aux2
    ↑
Decoder 3 (64×64):   Loss_aux3
    ↑
...

Total Loss = Loss_main 
             + 0.5 × Loss_aux1 
             + 0.3 × Loss_aux2 
             + 0.2 × Loss_aux3
```

### Implementation

```python
def compute_deep_supervision_loss(
    main_logits,      # (B, C, H, W) - final output
    aux_outputs,      # List of auxiliary outputs
    targets,          # (B, H, W) - ground truth
    loss_fn,          # Loss function
    aux_weights=None  # Weights cho auxiliary losses
):
    """
    Compute deep supervision loss
    
    Args:
        main_logits: Main segmentation output
        aux_outputs: [aux3, aux2, aux1] từ decoder
        targets: Ground truth labels
        loss_fn: MultiClassCombinedLoss instance
        aux_weights: [w3, w2, w1] weights (default: [0.2, 0.3, 0.5])
    
    Returns:
        total_loss: Weighted sum
        loss_dict: Breakdown của losses
    """
    if aux_weights is None:
        aux_weights = [0.2, 0.3, 0.5]  # Deeper → lower weight
    
    # Main loss
    main_loss, main_dict = loss_fn(main_logits, targets)
    
    total_loss = main_loss
    loss_dict = {'main': main_dict}
    
    # Auxiliary losses
    for i, (aux_out, weight) in enumerate(zip(aux_outputs, aux_weights)):
        # Downsample targets to match aux output size
        _, _, H, W = aux_out.shape
        targets_down = F.interpolate(
            targets.unsqueeze(1).float(),  # (B, 1, H_orig, W_orig)
            size=(H, W),
            mode='nearest'
        ).squeeze(1).long()  # (B, H, W)
        
        # Compute auxiliary loss
        aux_loss, aux_dict = loss_fn(aux_out, targets_down)
        
        # Add to total với weight
        total_loss = total_loss + weight * aux_loss
        loss_dict[f'aux{i}'] = aux_dict
    
    return total_loss, loss_dict
```

**Example usage**:
```python
# Model output với deep supervision
seg_logits, cls_logits, aux_outputs = model(images)

# aux_outputs = [aux3, aux2, aux1]
# aux3: (B, 3, 64, 64)
# aux2: (B, 3, 128, 128)
# aux1: (B, 3, 256, 256)

# Compute loss
total_loss, loss_dict = compute_deep_supervision_loss(
    main_logits=seg_logits,
    aux_outputs=aux_outputs,
    targets=masks,
    loss_fn=loss_fn,
    aux_weights=[0.2, 0.3, 0.5]
)

# loss_dict = {
#     'main': {'dice': 0.25, 'focal': 0.18, 'total': 0.43},
#     'aux0': {'dice': 0.32, 'focal': 0.24, 'total': 0.56},  # aux3 (deepest)
#     'aux1': {'dice': 0.28, 'focal': 0.21, 'total': 0.49},  # aux2
#     'aux2': {'dice': 0.26, 'focal': 0.19, 'total': 0.45}   # aux1 (shallowest)
# }

# total_loss = 0.43 (main) 
#              + 0.2×0.56 (aux3) 
#              + 0.3×0.49 (aux2) 
#              + 0.5×0.45 (aux1)
#            = 0.43 + 0.112 + 0.147 + 0.225
#            = 0.914
```

---

## 6. Multi-Class Metrics

### File Code

**File**: `src/braintumnet/multiclass_metrics.py` (215 dòng)

### Dice Score

```python
def compute_dice_score(pred, target, num_classes=3, 
                       ignore_background=True):
    """
    Compute Dice score per class
    
    Args:
        pred: (B, H, W) - predicted class indices
        target: (B, H, W) - ground truth indices
        num_classes: 3
        ignore_background: Bỏ qua class 0
    
    Returns:
        dice_scores: Dict {class_name: dice_value}
    """
    dice_scores = {}
    
    start_cls = 1 if ignore_background else 0
    
    for cls in range(start_cls, num_classes):
        # Binary masks
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        # Intersection và sums
        intersection = (pred_mask & target_mask).sum().float()
        pred_sum = pred_mask.sum().float()
        target_sum = target_mask.sum().float()
        
        # Dice
        if target_sum == 0:
            # No ground truth for this class
            dice = 1.0 if pred_sum == 0 else 0.0
        else:
            dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-8)
        
        dice_scores[f'class_{cls}'] = dice.item()
    
    return dice_scores
```

### BraTS Tumor Regions

**3 quan trọng regions**:

1. **WT (Whole Tumor)** = TC + ED = class 1 + class 2
2. **TC (Tumor Core)** = class 1 only
3. **ED (Edema)** = class 2 only

```python
def compute_brats_regions(pred, target):
    """
    Compute BraTS tumor regions từ 3-class predictions
    
    Args:
        pred: (B, H, W) - class indices {0, 1, 2}
        target: (B, H, W) - ground truth
    
    Returns:
        regions_pred: Dict of binary masks
        regions_target: Dict of binary masks
    """
    regions_pred = {}
    regions_target = {}
    
    # WT (Whole Tumor) = TC + ED
    regions_pred['WT'] = (pred == 1) | (pred == 2)
    regions_target['WT'] = (target == 1) | (target == 2)
    
    # TC (Tumor Core) = class 1
    regions_pred['TC'] = (pred == 1)
    regions_target['TC'] = (target == 1)
    
    # ED (Edema) = class 2
    regions_pred['ED'] = (pred == 2)
    regions_target['ED'] = (target == 2)
    
    return regions_pred, regions_target
```

### Hausdorff Distance

**Distance metric cho boundary accuracy**:

```python
def compute_hausdorff_distance(pred, target, percentile=95):
    """
    Compute 95th percentile Hausdorff Distance
    
    HD95 = 95th percentile of distances giữa boundaries
    
    Lower is better (mm)
    
    Args:
        pred: (H, W) binary mask
        target: (H, W) binary mask
        percentile: 95 (standard)
    
    Returns:
        hd95: scalar distance
    """
    from scipy.ndimage import distance_transform_edt
    
    # Convert to numpy
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # Edge detection
    pred_edge = pred_np ^ binary_erosion(pred_np)
    target_edge = target_np ^ binary_erosion(target_np)
    
    # Distance transforms
    pred_dist = distance_transform_edt(~pred_edge)
    target_dist = distance_transform_edt(~target_edge)
    
    # Hausdorff distances
    dist_pred_to_target = pred_dist[target_edge]
    dist_target_to_pred = target_dist[pred_edge]
    
    # Combine
    all_dists = np.concatenate([dist_pred_to_target, dist_target_to_pred])
    
    # Percentile
    if len(all_dists) == 0:
        return 0.0
    
    hd95 = np.percentile(all_dists, percentile)
    
    return hd95
```

**Ví dụ**:
```
Prediction boundary:   Ground truth boundary:
┌──────┐              ┌──────┐
│  *** │              │ **** │
│ *  * │              │*   * │
│  *** │              │ **** │
└──────┘              └──────┘

Distances from pred to GT:
- Top edge: 1 pixel off
- Left edge: 0 pixel
- Right edge: 0 pixel
- Bottom edge: 1 pixel off

Distances: [0, 0, 0, 1, 1, 1, 0, 0, ...]
95th percentile: 1.0 pixel

HD95 = 1.0 (good alignment)
```

---

## 7. Metrics Accumulation

### MetricsAccumulator Class

```python
class MulticlassMetricsAccumulator:
    """
    Accumulate metrics across batches
    
    Correct approach:
    1. Accumulate intersection và union cho mỗi class
    2. Compute final Dice = 2×Σintersection / (Σpred + Σtarget)
    
    Wrong approach:
    - Average Dice scores per batch → biased!
    """
    def __init__(self, num_classes=3, ignore_background=True):
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        
        # Accumulators
        self.reset()
    
    def reset(self):
        """Reset all accumulators"""
        self.intersection = {
            'WT': 0, 'TC': 0, 'ED': 0
        }
        self.pred_sum = {
            'WT': 0, 'TC': 0, 'ED': 0
        }
        self.target_sum = {
            'WT': 0, 'TC': 0, 'ED': 0
        }
        self.num_samples = 0
    
    def update(self, pred, target):
        """
        Update với 1 batch
        
        Args:
            pred: (B, H, W) - predicted classes
            target: (B, H, W) - ground truth
        """
        # Get BraTS regions
        regions_pred, regions_target = compute_brats_regions(pred, target)
        
        # Update accumulators cho mỗi region
        for region in ['WT', 'TC', 'ED']:
            pred_mask = regions_pred[region]
            target_mask = regions_target[region]
            
            # Intersection
            intersection = (pred_mask & target_mask).sum().item()
            self.intersection[region] += intersection
            
            # Sums
            self.pred_sum[region] += pred_mask.sum().item()
            self.target_sum[region] += target_mask.sum().item()
        
        self.num_samples += pred.shape[0]
    
    def compute(self):
        """
        Compute final metrics từ accumulated values
        
        Returns:
            metrics: Dict {region: dice_score}
        """
        metrics = {}
        
        for region in ['WT', 'TC', 'ED']:
            intersection = self.intersection[region]
            pred_sum = self.pred_sum[region]
            target_sum = self.target_sum[region]
            
            # Global Dice
            if target_sum == 0:
                dice = 1.0 if pred_sum == 0 else 0.0
            else:
                dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-8)
            
            metrics[region] = dice
        
        # Mean Dice
        metrics['Mean'] = np.mean([
            metrics['WT'], metrics['TC'], metrics['ED']
        ])
        
        return metrics
```

### Tại Sao Global Accumulation?

**❌ Wrong (averaging per-batch Dice)**:
```python
# Batch 1: 100 tumor pixels
dice_batch1 = 0.85

# Batch 2: 10 tumor pixels
dice_batch2 = 0.60

# Average
mean_dice = (0.85 + 0.60) / 2 = 0.725

Problem: Batch 2 có ít tumor nhưng contribute equally!
```

**✓ Correct (global accumulation)**:
```python
# Batch 1
intersection1 = 85
pred_sum1 = 100
target_sum1 = 100

# Batch 2
intersection2 = 6
pred_sum2 = 10
target_sum2 = 10

# Global Dice
total_intersection = 85 + 6 = 91
total_pred = 100 + 10 = 110
total_target = 100 + 10 = 110

dice = 2 × 91 / (110 + 110) = 0.827

→ More accurate! Batch 1 dominates correctly
```

### Usage Example

```python
# Training/Validation loop
accumulator = MulticlassMetricsAccumulator(num_classes=3)

for images, masks, labels in val_loader:
    # Forward
    seg_logits, cls_logits, aux = model(images)
    
    # Predictions
    seg_pred = seg_logits.argmax(dim=1)  # (B, H, W)
    
    # Update accumulator
    accumulator.update(seg_pred, masks)

# Compute final metrics
metrics = accumulator.compute()

print(f"WT Dice: {metrics['WT']:.4f}")
print(f"TC Dice: {metrics['TC']:.4f}")
print(f"ED Dice: {metrics['ED']:.4f}")
print(f"Mean Dice: {metrics['Mean']:.4f}")
```

**Output**:
```
WT Dice: 0.8856
TC Dice: 0.8234
ED Dice: 0.7612
Mean Dice: 0.8234
```

---

**[← Phần 3: Xử Lý Dữ Liệu](v_03_XU_LY_DU_LIEU.md)** | **[Phần 5: Training System →](v_05_TRAINING_SYSTEM.md)**
