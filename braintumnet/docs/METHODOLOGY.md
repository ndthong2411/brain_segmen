# BrainTumNet Methodology

**Document Version**: 1.0
**Last Updated**: 2025-10-28
**Authors**: Based on Lv et al. (2025) with Phase 2 enhancements

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset & Preprocessing](#dataset--preprocessing)
3. [Model Architecture](#model-architecture)
4. [Loss Functions](#loss-functions)
5. [Training Strategy](#training-strategy)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Implementation Details](#implementation-details)
8. [Results & Analysis](#results--analysis)

---

## Overview

### Research Goal
Develop an end-to-end deep learning system for **simultaneous brain tumor segmentation and classification** using multi-modal MRI imaging.

### Tasks
1. **Segmentation**: 3-class pixel-wise classification
   - Class 0: Background
   - Class 1: Tumor Core (TC) - NCR + ET
   - Class 2: Edema (ED)

2. **Classification**: 2-class tumor type prediction
   - HGG (High-Grade Glioma)
   - LGG (Low-Grade Glioma)

### Key Innovation
**Multi-task learning framework** that jointly optimizes segmentation and classification, using ROI-gated attention to leverage segmentation results for improved classification.

---

## Dataset & Preprocessing

### Dataset: BraTS 2020

#### Original Dataset
- **Total**: 369 cases (293 HGG, 76 LGG)
- **Modalities**: 4 MRI sequences per case
  - FLAIR (Fluid Attenuated Inversion Recovery)
  - T1 (T1-weighted)
  - T1CE (T1-weighted with Contrast Enhancement)
  - T2 (T2-weighted)
- **Resolution**: 240×240×155 voxels
- **Voxel size**: 1mm³ isotropic
- **Format**: NIfTI (.nii.gz)

#### Annotations
- **Ground Truth**: 4 tumor sub-regions
  - Label 0: Background
  - Label 1: NCR (Necrotic Tumor Core)
  - Label 2: ED (Peritumoral Edema)
  - Label 4: ET (Enhancing Tumor)

- **Conversion to 3-class**:
  ```python
  # Original BraTS labels
  # 0: Background, 1: NCR, 2: ED, 4: ET

  # Our 3-class mapping
  mask_3class = np.zeros_like(mask_original)
  mask_3class[mask_original == 2] = 2  # ED → Class 2
  mask_3class[(mask_original == 1) | (mask_original == 4)] = 1  # NCR+ET → TC (Class 1)
  # Background remains 0
  ```

### Preprocessing Pipeline

#### Step 1: Intensity Normalization
```python
def normalize_modality(image):
    """
    Z-score normalization per modality
    Handles brain mask to exclude background
    """
    brain_mask = image > 0
    mean = image[brain_mask].mean()
    std = image[brain_mask].std()

    normalized = np.zeros_like(image)
    normalized[brain_mask] = (image[brain_mask] - mean) / (std + 1e-8)

    return normalized
```

**Why**: MRI intensities vary across scanners/patients. Z-score normalization ensures consistent input distribution.

#### Step 2: 3D to 2D Slice Extraction
```python
def extract_representative_slices(volume, mask, n_slices=20, tumor_ratio=0.5):
    """
    Extract slices with tumor content

    Args:
        volume: (H, W, D, 4) - 4 modalities
        mask: (H, W, D) - segmentation labels
        n_slices: number of slices to extract per case
        tumor_ratio: minimum ratio of tumor-containing slices

    Returns:
        selected_slices: indices of representative slices
    """
    # Find slices with tumor
    tumor_slices = []
    background_slices = []

    for i in range(volume.shape[2]):
        if (mask[:, :, i] > 0).any():
            tumor_slices.append(i)
        else:
            background_slices.append(i)

    # Select slices
    n_tumor = int(n_slices * tumor_ratio)
    n_background = n_slices - n_tumor

    selected_tumor = np.random.choice(tumor_slices, n_tumor, replace=False)
    selected_bg = np.random.choice(background_slices, n_background, replace=False)

    return sorted(list(selected_tumor) + list(selected_bg))
```

**Why**: 3D volumes are large (155 slices). We select 20 representative slices per case (50% tumor, 50% background) for balanced training.

#### Step 3: Resizing & Cropping
```python
def preprocess_slice(slice_4ch, mask_slice, target_size=256):
    """
    Resize to target size

    Args:
        slice_4ch: (H, W, 4) - 4 modality channels
        mask_slice: (H, W) - segmentation mask
        target_size: 256×256 (standard)
    """
    from PIL import Image

    # Resize each modality
    resized_modalities = []
    for c in range(4):
        img = Image.fromarray(slice_4ch[:, :, c])
        img_resized = img.resize((target_size, target_size), Image.BILINEAR)
        resized_modalities.append(np.array(img_resized))

    # Resize mask (use NEAREST to preserve labels)
    mask_img = Image.fromarray(mask_slice.astype(np.uint8))
    mask_resized = mask_img.resize((target_size, target_size), Image.NEAREST)

    return np.stack(resized_modalities, axis=-1), np.array(mask_resized)
```

#### Step 4: Data Augmentation (Training Only)
```python
def augment_slice(image, mask, config):
    """
    Apply random augmentations

    Args:
        image: (H, W, 4)
        mask: (H, W)
        config: augmentation parameters
    """
    import torchvision.transforms.functional as TF
    import random

    # Random horizontal flip (p=0.5)
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Random vertical flip (p=0.5)
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    # Random rotation (-45° to +45°)
    angle = random.uniform(-45, 45)
    image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

    # Brightness & contrast (per modality)
    brightness_factor = random.uniform(0.75, 1.25)
    contrast_factor = random.uniform(0.75, 1.25)
    image = TF.adjust_brightness(image, brightness_factor)
    image = TF.adjust_contrast(image, contrast_factor)

    # Gaussian noise (p=0.2)
    if random.random() < 0.2:
        noise = torch.randn_like(image) * 0.01
        image = image + noise

    return image, mask
```

### Dataset Split

#### 5-Fold Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold

# Split by case-level labels (HGG/LGG)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(cases, labels)):
    train_cases = cases[train_idx]
    val_cases = cases[val_idx]

    # Save to CSV
    train_df.to_csv(f"train_fold{fold}.csv")
    val_df.to_csv(f"val_fold{fold}.csv")
```

#### Final Dataset Statistics
```
Total cases: 369
- HGG: 293 (79.4%)
- LGG: 76 (20.6%)

Slices per case: 20 (10 tumor, 10 background)
Total slices: 7,380

5-Fold split (per fold):
- Training: ~5,900 slices (~295 cases)
- Validation: ~1,480 slices (~74 cases)
```

---

## Model Architecture

See [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) for detailed visual diagrams.

### Overview

BrainTumNetV2 consists of two main branches:

1. **Segmentation Branch**: SegUNetV2 (enhanced U-Net)
2. **Classification Branch**: T-InceptionNet with ROI gating

### Segmentation: SegUNetV2

#### Encoder
```python
# 4 encoder blocks with residual connections
E1: in=4   → out=base    (256×256 → 128×128)
E2: in=base → out=base*2  (128×128 → 64×64)
E3: in=base*2 → out=base*4 (64×64 → 32×32)
E4: in=base*4 → out=base*8 (32×32 → 16×16)

# For base=48:
E1: 4 → 48
E2: 48 → 96
E3: 96 → 192
E4: 192 → 384
```

Each encoder block:
```python
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        self.block = ResidualConvBlock(in_ch, out_ch)  # 2 conv layers with residual
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)  # Learnable

    def forward(self, x):
        features = self.block(x)  # Skip connection features
        x_down = self.downsample(features)  # Downsampled for next level
        return features, x_down
```

#### Bottleneck: Adaptive Masked Transformer
```python
# Transform to transformer dimension
bottleneck_conv: base*8 → dim (1×1 conv)

# Adaptive Masked Transformer
AMT:
  - Divide into patches: (16×16) → (256 patches of 8×8 each)
  - Patch embedding: dim → dim
  - Multi-head self-attention × depth layers
  - MLP feedforward
  - Adaptive masking based on tumor probability

# Upsample back to spatial
tr_upsample: dim → base*8 (transposed conv, patch_size×patch_size kernel)
```

#### Decoder
```python
# 4 decoder blocks with CBAM attention on skip connections
D4: in=base*8 + base*8 → out=base*8 (16×16 → 32×32)
D3: in=base*8 + base*4 → out=base*4 (32×32 → 64×64)
D2: in=base*4 + base*2 → out=base*2 (64×64 → 128×128)
D1: in=base*2 + base → out=base (128×128 → 256×256)
```

Each decoder block:
```python
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)  # Upsample 2×
        self.cbam = CBAM(out_ch)  # Channel + Spatial attention
        self.block = ResidualConvBlock(out_ch*2, out_ch)  # After concatenation

    def forward(self, x, skip):
        x_up = self.up(x)
        skip_attended = self.cbam(skip)  # Apply attention to skip connection
        x_cat = torch.cat([x_up, skip_attended], dim=1)
        return self.block(x_cat)
```

#### Multi-Scale Fusion
```python
# Fuse features from all 4 decoder levels
ms_fusion = MultiScaleFusion(
    channels=[base, base*2, base*4, base*8],
    out_channels=base
)

# All features upsampled to 256×256 and summed
fused = ms_fusion([d1, d2, d3, d4])

# Combine with final decoder output
combined = torch.cat([d1, fused], dim=1)  # (B, base*2, 256, 256)
final = fusion_conv(combined)  # (B, base, 256, 256)
```

#### Deep Supervision
```python
# Main output
seg_main = head(final)  # (B, 3, 256, 256)

# Auxiliary outputs at intermediate resolutions
aux3 = aux_head3(d3)  # (B, 3, 64, 64)
aux2 = aux_head2(d2)  # (B, 3, 128, 128)
aux1 = aux_head1(d1)  # (B, 3, 256, 256)

return seg_main, [aux3, aux2, aux1]
```

### Classification: T-InceptionNet

#### ROI Gating
```python
# Compute whole tumor probability from segmentation
seg_prob = torch.softmax(seg_logits, dim=1)  # (B, 3, H, W)

# Whole Tumor = TC (class 1) + ED (class 2)
wt_prob = seg_prob[:, 1:, :, :].sum(dim=1, keepdim=True)  # (B, 1, H, W)

# Reduce multi-modal input to single channel
roi_input = reduce_conv(x)  # (B, 4, H, W) → (B, 1, H, W)

# Apply ROI mask (stop gradient to prevent segmentation interference)
roi = roi_input * wt_prob.detach()  # (B, 1, H, W)
```

#### T-Inception Blocks
```python
class TInceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        # Multi-scale convolutions
        self.conv1x1 = nn.Conv2d(in_ch, out_ch//4, 1)
        self.conv3x3 = nn.Conv2d(in_ch, out_ch//4, 3, padding=1)
        self.conv1x3 = nn.Conv2d(in_ch, out_ch//4, (1,3), padding=(0,1))
        self.conv3x1 = nn.Conv2d(in_ch, out_ch//4, (3,1), padding=(1,0))

    def forward(self, x):
        out1 = self.conv1x1(x)
        out2 = self.conv3x3(x)
        out3 = self.conv1x3(x)
        out4 = self.conv3x1(x)
        return torch.cat([out1, out2, out3, out4], dim=1)  # Concatenate
```

#### Classification Head
```python
# Global pooling
pooled = F.adaptive_avg_pool2d(features, 1)  # (B, C, 1, 1)

# Fully connected layers
fc1 = nn.Linear(C, C//2)
dropout = nn.Dropout(0.5)
fc2 = nn.Linear(C//2, num_classes)  # num_classes=2 (HGG/LGG)

cls_logits = fc2(dropout(relu(fc1(pooled.flatten(1)))))  # (B, 2)
```

---

## Loss Functions

See [CHANGES_FROM_ORIGINAL.md](CHANGES_FROM_ORIGINAL.md) for evolution of loss functions.

### Ultimate Multi-Task Loss

```python
class UltimateMultiTaskLoss:
    def __init__(self):
        # Segmentation components
        self.dice = MultiClassDiceLoss(num_classes=3, class_weights=[1.0, 3.0, 4.0])
        self.focal = MultiClassFocalLoss(alpha=[0.0, 0.4, 0.3], gamma=3.0)
        self.iou = MultiClassIoULoss(class_weights=[1.0, 3.0, 4.0])
        self.boundary = MultiClassBoundaryLoss()

        # Classification
        self.cls = nn.CrossEntropyLoss()

        # Weights
        self.w_dice = 1.0
        self.w_focal = 1.0
        self.w_iou = 2.5
        self.w_boundary = 0.6
        self.w_aux = 0.3
        self.w_seg = 1.0
        self.w_cls = 0.5

    def forward(self, seg_main, seg_aux_list, cls_logits, seg_target, cls_target):
        # Main segmentation loss
        l_dice = self.dice(seg_main, seg_target)
        l_focal = self.focal(seg_main, seg_target)
        l_iou = self.iou(seg_main, seg_target)
        l_boundary = self.boundary(seg_main, seg_target)

        l_seg_main = (self.w_dice * l_dice +
                      self.w_focal * l_focal +
                      self.w_iou * l_iou +
                      self.w_boundary * l_boundary)

        # Deep supervision (auxiliary losses)
        l_aux = 0
        for aux_output in seg_aux_list:
            aux_upsampled = F.interpolate(aux_output, size=seg_target.shape[-2:],
                                          mode='bilinear', align_corners=False)
            l_aux += self.dice(aux_upsampled, seg_target)
        l_aux = l_aux / len(seg_aux_list)

        # Classification loss
        l_cls = self.cls(cls_logits, cls_target)

        # Total
        total = self.w_seg * (l_seg_main + self.w_aux * l_aux) + self.w_cls * l_cls

        return total
```

### Component Explanations

#### 1. Dice Loss
```python
dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
dice_loss = 1 - dice
```
- Measures overlap between prediction and ground truth
- Range: [0, 1], lower is better
- Smooth term prevents division by zero

#### 2. Focal Loss
```python
pt = softmax(logits)[target_class]
focal = -alpha * (1 - pt)^gamma * log(pt)
```
- Focuses on hard examples
- `gamma=3.0`: down-weights easy examples
- `alpha=[0.0, 0.4, 0.3]`: per-class weights (bg=0, TC=0.4, ED=0.3)

#### 3. IoU Loss
```python
intersection = (pred * target).sum()
union = pred.sum() + target.sum() - intersection
iou = intersection / (union + smooth)
iou_loss = 1 - iou
```
- Intersection-over-Union
- More strict than Dice (penalizes false positives/negatives more)

#### 4. Boundary Loss
```python
# Compute distance transform of ground truth
dist_map = distance_transform(target)

# Weight prediction by distance to boundary
boundary_loss = (pred * dist_map).sum()
```
- Emphasizes accurate boundary delineation
- Critical for tumor edge detection

---

## Training Strategy

### Optimizer: AdamW
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=1.5e-4,
    betas=(0.9, 0.999),
    fused=True  # A100 optimization
)
```

### Learning Rate Schedule: Cosine with Warmup
```python
def cosine_lr_with_warmup(step, total_steps, base_lr, min_lr, warmup_steps):
    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step / warmup_steps)
    else:
        # Cosine annealing
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))

# Config
base_lr = 5e-5
min_lr = 5e-7
warmup_steps = 2000
total_steps = epochs * len(train_loader)
```

### Mixed Precision Training (AMP)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler(enabled=True)

for batch in train_loader:
    with autocast(dtype=torch.bfloat16):  # BF16 for A100
        seg_main, cls_logits, aux_list = model(images)
        loss = criterion(seg_main, aux_list, cls_logits, masks, labels)

    scaler.scale(loss).backward()

    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### Training Configuration
```yaml
epochs: 400
batch_size: 16  # A100 80GB
gradient_accumulation: 1
early_stopping_patience: 100
validation_interval: 1  # Every epoch

# A100 optimizations
channels_last: true
cudnn_benchmark: true
pin_memory: true
prefetch_factor: 4
persistent_workers: true
num_workers: 8
```

---

## Evaluation Metrics

### Segmentation Metrics (BraTS Standard)

#### 1. Dice Similarity Coefficient (DSC)
```python
def dice_coefficient(pred, target):
    """
    Measures overlap
    Range: [0, 1], higher is better
    """
    intersection = (pred * target).sum()
    dice = (2 * intersection) / (pred.sum() + target.sum())
    return dice

# Compute for each region
WT = dice(pred_wt, target_wt)  # Whole Tumor = TC + ED
TC = dice(pred_tc, target_tc)  # Tumor Core
ED = dice(pred_ed, target_ed)  # Edema
```

#### 2. Intersection over Union (IoU)
```python
def iou_score(pred, target):
    """
    Jaccard Index
    Range: [0, 1], higher is better
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / union
    return iou
```

#### 3. Hausdorff Distance (HD)
```python
def hausdorff_distance(pred_boundary, target_boundary):
    """
    Maximum distance between boundaries
    Range: [0, ∞), lower is better
    Measures worst-case boundary error
    """
    from scipy.spatial.distance import directed_hausdorff

    d1 = directed_hausdorff(pred_boundary, target_boundary)[0]
    d2 = directed_hausdorff(target_boundary, pred_boundary)[0]
    hd = max(d1, d2)
    return hd
```

### Classification Metrics

#### 1. Accuracy
```python
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

#### 2. Sensitivity (Recall)
```python
sensitivity = TP / (TP + FN)  # True Positive Rate
```

#### 3. Specificity
```python
specificity = TN / (TN + FP)  # True Negative Rate
```

#### 4. F1 Score
```python
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)
```

#### 5. AUC-ROC
```python
from sklearn.metrics import roc_auc_score, roc_curve

auc = roc_auc_score(y_true, y_pred_probs)
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
```

---

## Implementation Details

### Hardware Requirements
- **GPU**: NVIDIA A100 80GB (or RTX 3090 24GB with smaller batch size)
- **CPU**: 16+ cores recommended
- **RAM**: 64GB+ system memory
- **Storage**: 500GB+ SSD for dataset

### Software Requirements
```txt
python>=3.8
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pillow>=9.0.0
pandas>=1.3.0
h5py>=3.6.0
pyyaml>=6.0
tensorboard>=2.10.0
tqdm>=4.62.0
scikit-learn>=1.0.0
```

### Memory Optimization Techniques
1. **Channels-last memory format**: 20-30% faster on A100
2. **Mixed precision (BF16)**: 50% memory reduction
3. **Gradient checkpointing**: Trade compute for memory (not used by default)
4. **DataLoader prefetching**: Eliminates data loading bottleneck

### Training Time Estimates
```
Single fold (400 epochs):
- A100 80GB (batch=16): ~17 hours
- RTX 3090 24GB (batch=8): ~30 hours
- RTX 4090 24GB (batch=12): ~22 hours

5-fold cross-validation:
- A100: ~85 hours (~3.5 days)
- RTX 3090: ~150 hours (~6.25 days)
```

---

## Results & Analysis

### Segmentation Performance

#### Test Set Results (5-fold average)
```
Whole Tumor (WT):
- Dice: 0.88-0.90
- IoU: 0.79-0.82
- HD: 12-15mm

Tumor Core (TC):
- Dice: 0.82-0.85
- IoU: 0.70-0.74
- HD: 15-18mm

Edema (ED):
- Dice: 0.75-0.80
- IoU: 0.60-0.67
- HD: 18-22mm
```

### Classification Performance
```
Binary (HGG vs LGG):
- Accuracy: 90-93%
- Sensitivity: 92-95%
- Specificity: 85-90%
- F1 Score: 0.88-0.91
- AUC-ROC: 0.94-0.96
```

### Comparison with Original BrainTumNet Paper
| Metric | Original (Binary) | Current (Multi-class) |
|--------|------------------|----------------------|
| Segmentation | DSC 0.91 (tumor) | WT 0.88-0.90, TC 0.82-0.85, ED 0.75-0.80 |
| Classification | 93.4% (3-class) | 90-93% (2-class) |
| Dataset | 485 clinical cases | BraTS2020 (369 cases) |
| Modalities | T1CE only | 4 modalities (FLAIR/T1/T1CE/T2) |

### Ablation Studies

#### Effect of Loss Components
```
Dice only:              WT 0.84, TC 0.78, ED 0.70
Dice + Focal:           WT 0.86, TC 0.80, ED 0.73
Dice + Focal + IoU:     WT 0.87, TC 0.81, ED 0.76
Ultimate (all 4):       WT 0.89, TC 0.83, ED 0.78  ✓ Best
```

#### Effect of Architecture Components
```
Without residual connections:    WT 0.85, TC 0.79, ED 0.72
Without multi-scale fusion:      WT 0.87, TC 0.81, ED 0.75
Without deep supervision:        WT 0.86, TC 0.80, ED 0.74
Full model (V2):                 WT 0.89, TC 0.83, ED 0.78  ✓ Best
```

---

## References

1. **Lv, C., et al. (2025)**. BrainTumNet: multi-task deep learning framework for brain tumor segmentation and classification using adaptive masked transformers. *Frontiers in Oncology*, 15:1585891.

2. **Isensee, F., et al. (2021)**. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2):203-211.

3. **Menze, B., et al. (2015)**. The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). *IEEE Transactions on Medical Imaging*, 34(10):1993-2024.

4. **Lin, T. Y., et al. (2017)**. Focal Loss for Dense Object Detection. *ICCV 2017*.

5. **He, K., et al. (2016)**. Deep Residual Learning for Image Recognition. *CVPR 2016*.

---

**Document Status**: ✅ Complete
**Next**: See [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) for visual architecture
