# Part 10: Extension Guide

**Navigation**: [[TECHNICAL_REPORT_INDEX|← Back to Index]]

---

## Table of Contents

1. [Overview](#overview)
2. [Adding New Model Components](#adding-new-model-components)
3. [Adding New Loss Functions](#adding-new-loss-functions)
4. [Adding New Metrics](#adding-new-metrics)
5. [Adding New Data Augmentations](#adding-new-data-augmentations)
6. [Supporting New Datasets](#supporting-new-datasets)
7. [Implementing 3D Models](#implementing-3d-models)
8. [Multi-Task Extensions](#multi-task-extensions)
9. [Deployment and Production](#deployment-and-production)

---

## Overview

This guide shows you how to **extend BrainTumNet** with new features, components, and capabilities.

### Extension Philosophy

- **Modular**: Add components without breaking existing code
- **Configurable**: Control via YAML configs
- **Tested**: Verify each addition works
- **Documented**: Comment your code

---

## Adding New Model Components

### Example 1: Add Squeeze-and-Excitation (SE) Block

**Step 1**: Create new file `src/braintumnet/models/se_block.py`

```python
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Re-calibrates channel-wise features.

    Reference: Hu et al. "Squeeze-and-Excitation Networks" (CVPR 2018)
    """

    def __init__(self, in_channels, reduction=16):
        """
        Args:
            in_channels: Number of input channels
            reduction: Reduction ratio for bottleneck
        """
        super().__init__()

        # Squeeze: Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation: FC layers
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Recalibrated tensor (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Squeeze: (B, C, H, W) → (B, C, 1, 1) → (B, C)
        y = self.avg_pool(x).view(B, C)

        # Excitation: (B, C) → (B, C)
        y = self.fc(y)

        # Reshape and scale: (B, C) → (B, C, 1, 1) → (B, C, H, W)
        y = y.view(B, C, 1, 1)
        return x * y
```

**Step 2**: Integrate into U-Net

Modify `src/braintumnet/models/seg_unet.py`:

```python
from .se_block import SEBlock

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False):
        super().__init__()
        self.block = nn.Sequential(
            conv_bn_relu(in_ch, out_ch),
            conv_bn_relu(out_ch, out_ch)
        )
        self.pool = nn.MaxPool2d(2)

        # Add SE block
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        x = self.block(x)
        x = self.se(x)  # Apply SE before pooling
        x_down = self.pool(x)
        return x, x_down
```

**Step 3**: Add to config

```yaml
# In configs/se_experiment.yaml
model:
  use_se: true  # Enable SE blocks
```

**Step 4**: Update model builder

In `src/braintumnet/engine/trainer.py`:

```python
def build_model(cfg: Dict):
    mcfg = cfg["model"]
    use_se = mcfg.get("use_se", False)  # Get from config

    return BrainTumNet(
        in_ch=mcfg["in_channels"],
        num_cls=mcfg["num_classes_cls"],
        base=mcfg["base"],
        dim=mcfg["dim"],
        patch=mcfg["patch_size"],
        depth=mcfg["depth"],
        n_heads=mcfg["n_heads"],
        roi_stop_grad=mcfg["roi_stop_grad"],
        use_se=use_se  # Pass to model
    )
```

**Step 5**: Test

```bash
# Train with SE blocks
python train.py --cfg configs/se_experiment.yaml --fold 0

# Compare with baseline
# Expected: +0.5-1% Dice improvement
```

---

### Example 2: Add Residual Connections

**Modify encoder blocks** in `seg_unet.py`:

```python
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, residual=False):
        super().__init__()
        self.residual = residual

        self.block = nn.Sequential(
            conv_bn_relu(in_ch, out_ch),
            conv_bn_relu(out_ch, out_ch)
        )
        self.pool = nn.MaxPool2d(2)

        # Residual projection (if channels change)
        if residual and in_ch != out_ch:
            self.downsample = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        x = self.block(x)

        if self.residual:
            x = x + identity  # Residual connection

        x_down = self.pool(x)
        return x, x_down
```

**Config**:
```yaml
model:
  residual: true
```

---

## Adding New Loss Functions

### Example 1: Focal Loss

**Create** `src/braintumnet/losses/base.py` (add to existing):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)

    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    Args:
        alpha: Weighting factor [0, 1]
        gamma: Focusing parameter γ ≥ 0
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) - raw predictions
            targets: (B, 1, H, W) - binary targets (0 or 1)

        Returns:
            Focal loss (scalar)
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Compute focal weight
        # For positive samples (target=1): (1 - p)^γ
        # For negative samples (target=0): p^γ
        focal_weight = torch.where(
            targets == 1,
            (1 - probs).pow(self.gamma),
            probs.pow(self.gamma)
        )

        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Apply focal weight and alpha
        focal_loss = focal_weight * bce

        # Apply alpha weighting
        alpha_weight = torch.where(
            targets == 1,
            self.alpha,
            1 - self.alpha
        )
        focal_loss = alpha_weight * focal_loss

        return focal_loss.mean()


class DiceFocalLoss(nn.Module):
    """
    Combination of Dice Loss + Focal Loss.
    """

    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal = FocalLoss(alpha, gamma)

    def forward(self, logits, targets):
        # Dice loss (from existing code)
        dice_loss = dice_loss_with_logits(logits, targets)

        # Focal loss
        focal_loss = self.focal(logits, targets)

        # Combine
        total = self.dice_weight * dice_loss + self.focal_weight * focal_loss

        return total
```

**Update config**:

```yaml
# In configs/focal_experiment.yaml
train:
  seg_criterion: "DiceFocal"  # Use new loss
  focal_alpha: 0.25
  focal_gamma: 2.0
```

**Update trainer.py**:

```python
def train_one_fold(cfg, fold):
    # ...

    # Build loss
    seg_criterion_name = cfg["train"].get("seg_criterion", "DiceCE")

    if seg_criterion_name == "DiceCE":
        seg_criterion = DiceCELoss()
    elif seg_criterion_name == "DiceFocal":
        seg_criterion = DiceFocalLoss(
            alpha=cfg["train"].get("focal_alpha", 0.25),
            gamma=cfg["train"].get("focal_gamma", 2.0)
        )
    else:
        raise ValueError(f"Unknown criterion: {seg_criterion_name}")

    # ... rest of training
```

---

### Example 2: Boundary Loss

**For better boundary accuracy**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

class BoundaryLoss(nn.Module):
    """
    Boundary Loss for improved boundary delineation.

    Reference: Kervadec et al. "Boundary loss for highly unbalanced segmentation" (MIDL 2019)
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W)
            targets: (B, 1, H, W)

        Returns:
            Boundary loss
        """
        probs = torch.sigmoid(logits)

        # Compute distance transform (on CPU, numpy)
        B, C, H, W = targets.shape
        dist_maps = []

        for b in range(B):
            target_np = targets[b, 0].cpu().numpy()

            # Distance transform (distance to boundary)
            dist = distance_transform_edt(target_np) + distance_transform_edt(1 - target_np)
            dist_maps.append(dist)

        dist_maps = torch.from_numpy(np.stack(dist_maps)).unsqueeze(1).to(logits.device)

        # Boundary loss: integral of distance map weighted by prediction
        boundary_loss = (probs * dist_maps).mean()

        return boundary_loss
```

**Use**:
```python
class DiceBoundaryLoss(nn.Module):
    def __init__(self, dice_weight=1.0, boundary_weight=0.1):
        super().__init__()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.boundary = BoundaryLoss()

    def forward(self, logits, targets):
        dice = dice_loss_with_logits(logits, targets)
        boundary = self.boundary(logits, targets)
        return self.dice_weight * dice + self.boundary_weight * boundary
```

---

## Adding New Metrics

### Example: Sensitivity and Specificity

**Add to** `src/braintumnet/metrics/base.py`:

```python
def sensitivity_specificity(logits: torch.Tensor, target: torch.Tensor, eps=1e-6) -> Tuple[float, float]:
    """
    Compute Sensitivity (Recall) and Specificity.

    Sensitivity = TP / (TP + FN) - True Positive Rate
    Specificity = TN / (TN + FP) - True Negative Rate

    Args:
        logits: Model predictions (B, 1, H, W)
        target: Ground truth (B, 1, H, W)
        eps: Small epsilon for numerical stability

    Returns:
        (sensitivity, specificity)
    """
    pred = binarize(logits)

    # True Positives, False Negatives, True Negatives, False Positives
    tp = (pred * target).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    fp = (pred * (1 - target)).sum().item()

    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)

    return sensitivity, specificity
```

**Use in evaluator.py**:

```python
# In validation loop
from braintumnet.metrics import sensitivity_specificity

total_sens, total_spec = 0.0, 0.0
n = 0

for batch in val_loader:
    # ... forward pass ...

    sens, spec = sensitivity_specificity(seg, msk)
    total_sens += sens
    total_spec += spec
    n += 1

avg_sens = total_sens / n
avg_spec = total_spec / n

print(f"Sensitivity: {avg_sens:.4f}")
print(f"Specificity: {avg_spec:.4f}")
```

---

## Adding New Data Augmentations

### Example: Elastic Deformation

**Create** `src/braintumnet/data/augmentations.py`:

```python
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

def elastic_deformation(image, mask, alpha=30, sigma=5):
    """
    Elastic deformation for medical image augmentation.

    Args:
        image: (H, W) or (H, W, C) numpy array
        mask: (H, W) numpy array
        alpha: Deformation strength
        sigma: Smoothness of deformation

    Returns:
        Deformed (image, mask)
    """
    shape = image.shape[:2]

    # Random displacement fields
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    # Coordinate grid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).reshape(-1), (x + dx).reshape(-1)

    # Apply deformation
    if image.ndim == 2:
        image_def = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    else:
        channels = []
        for c in range(image.shape[2]):
            channels.append(map_coordinates(image[:, :, c], indices, order=1, mode='reflect').reshape(shape))
        image_def = np.stack(channels, axis=2)

    mask_def = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)

    return image_def, mask_def


def random_elastic_deformation(image, mask, p=0.5, alpha=30, sigma=5):
    """Apply elastic deformation with probability p."""
    if np.random.rand() < p:
        return elastic_deformation(image, mask, alpha, sigma)
    return image, mask
```

**Integrate into dataset** (`brats2020_dataset.py`):

```python
from .augmentations import random_elastic_deformation

class SliceDataset(Dataset):
    def __init__(self, ...):
        # ... existing code ...
        self.elastic_p = elastic_p  # Add to constructor

    def __getitem__(self, idx):
        # ... load image and mask ...

        if self.train:
            # Existing augmentations
            img, msk = augment_pair(img, msk, self.rotate_deg, self.hflip_p, self.vflip_p)

            # Add elastic deformation
            if self.elastic_p > 0:
                img_np = img.cpu().numpy().squeeze()
                msk_np = msk.cpu().numpy().squeeze()
                img_np, msk_np = random_elastic_deformation(img_np, msk_np, p=self.elastic_p)
                img = torch.from_numpy(img_np).unsqueeze(0).float()
                msk = torch.from_numpy(msk_np).unsqueeze(0).float()

        # ... rest of code ...
```

**Config**:
```yaml
augment:
  rotate_deg: 20
  hflip_p: 0.5
  vflip_p: 0.5
  elastic_p: 0.3  # 30% chance
  elastic_alpha: 30
  elastic_sigma: 5
```

---

## Supporting New Datasets

### Example: Support TCGA-GBM Dataset

**Step 1**: Create new preprocessing script

Create `scripts/prepare_tcga.py`:

```python
import os
import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image

def process_tcga(tcga_root, output_dir, img_size=256):
    """
    Process TCGA-GBM dataset.

    TCGA structure:
    tcga_root/
    ├── TCGA-02-0001/
    │   ├── t1.nii.gz
    │   ├── t2.nii.gz
    │   ├── flair.nii.gz
    │   └── seg.nii.gz
    ├── TCGA-02-0002/
    ...

    Args:
        tcga_root: Path to TCGA dataset
        output_dir: Where to save processed images
        img_size: Output image size
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    patients = sorted(os.listdir(tcga_root))
    labels = []  # For classification (if available)

    for patient_id in patients:
        patient_dir = os.path.join(tcga_root, patient_id)

        # Load NIfTI files
        t1 = nib.load(os.path.join(patient_dir, "t1.nii.gz")).get_fdata()
        t2 = nib.load(os.path.join(patient_dir, "t2.nii.gz")).get_fdata()
        flair = nib.load(os.path.join(patient_dir, "flair.nii.gz")).get_fdata()
        seg = nib.load(os.path.join(patient_dir, "seg.nii.gz")).get_fdata()

        # Process each slice
        for slice_idx in range(t1.shape[2]):
            t1_slice = t1[:, :, slice_idx]
            t2_slice = t2[:, :, slice_idx]
            flair_slice = flair[:, :, slice_idx]
            seg_slice = seg[:, :, slice_idx]

            # Skip empty slices
            if seg_slice.sum() == 0:
                continue

            # Normalize
            t1_slice = (t1_slice - t1_slice.min()) / (t1_slice.max() - t1_slice.min() + 1e-6)
            # ... same for t2, flair ...

            # Stack modalities
            img_stack = np.stack([flair_slice, t1_slice, t2_slice, np.zeros_like(t1_slice)], axis=2)

            # Resize
            img_pil = Image.fromarray((img_stack * 255).astype(np.uint8))
            img_resized = img_pil.resize((img_size, img_size), Image.BILINEAR)

            seg_pil = Image.fromarray((seg_slice * 255).astype(np.uint8))
            seg_resized = seg_pil.resize((img_size, img_size), Image.NEAREST)

            # Save
            filename = f"{patient_id}_slice_{slice_idx:03d}.png"
            img_resized.save(os.path.join(output_dir, "images", filename))
            seg_resized.save(os.path.join(output_dir, "masks", filename))

    print(f"Processed {len(patients)} patients")

if __name__ == "__main__":
    process_tcga("data/raw/TCGA-GBM", "data/processed_tcga", img_size=256)
```

**Step 2**: Update dataset class

Modify `brats2020_dataset.py` or create `tcga_dataset.py`:

```python
class TCGADataset(Dataset):
    """Same as SliceDataset but for TCGA."""
    # ... similar implementation ...
```

**Step 3**: Train on TCGA

```yaml
# configs/tcga_experiment.yaml
exp_name: "braintumnet_tcga"

data:
  proc_root: "data/processed_tcga"
  # ... rest similar ...
```

---

## Implementing 3D Models

### Example: 3D U-Net

**Create** `src/braintumnet/models/unet3d.py`:

```python
import torch
import torch.nn as nn

class Conv3DBlock(nn.Module):
    """3D Convolution block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder3DBlock(nn.Module):
    """3D Encoder block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = Conv3DBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.block(x)
        x_down = self.pool(x)
        return x, x_down


class Decoder3DBlock(nn.Module):
    """3D Decoder block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = Conv3DBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric segmentation.

    Input: (B, C, D, H, W) where D is depth (number of slices)
    Output: (B, 1, D, H, W)
    """

    def __init__(self, in_ch=4, base=32):
        super().__init__()

        # Encoder
        self.e1 = Encoder3DBlock(in_ch, base)
        self.e2 = Encoder3DBlock(base, base * 2)
        self.e3 = Encoder3DBlock(base * 2, base * 4)
        self.e4 = Encoder3DBlock(base * 4, base * 8)

        # Bottleneck
        self.bottleneck = Conv3DBlock(base * 8, base * 16)

        # Decoder
        self.d4 = Decoder3DBlock(base * 16, base * 8)
        self.d3 = Decoder3DBlock(base * 8, base * 4)
        self.d2 = Decoder3DBlock(base * 4, base * 2)
        self.d1 = Decoder3DBlock(base * 2, base)

        # Head
        self.head = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        # Encoder
        s1, x = self.e1(x)
        s2, x = self.e2(x)
        s3, x = self.e3(x)
        s4, x = self.e4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)

        # Head
        return self.head(x)
```

**Create 3D dataset**:

```python
class VolumetricDataset(Dataset):
    """
    Load entire 3D volumes instead of slices.

    Returns:
        image: (C, D, H, W) where D=num_slices
        mask: (1, D, H, W)
    """

    def __getitem__(self, idx):
        # Load all slices for one patient
        patient_id = self.patient_ids[idx]
        slices = []
        masks = []

        for slice_idx in range(155):  # BraTS has 155 slices
            img = load_image(patient_id, slice_idx)  # (C, H, W)
            msk = load_mask(patient_id, slice_idx)   # (1, H, W)
            slices.append(img)
            masks.append(msk)

        volume = torch.stack(slices, dim=1)  # (C, D, H, W)
        mask_volume = torch.stack(masks, dim=1)  # (1, D, H, W)

        return {"image": volume, "mask": mask_volume}
```

**Memory consideration**:
```
2D: (B, C, H, W) = (12, 4, 256, 256) = 12 MB
3D: (B, C, D, H, W) = (1, 4, 155, 256, 256) = 160 MB per sample!

Solution: Reduce batch size or use patch-based training
```

---

## Multi-Task Extensions

### Example: Add Tumor Sub-Region Segmentation

**Modify model output**:

```python
class BrainTumNetMultiRegion(nn.Module):
    """
    Multi-region segmentation:
    - Class 0: Background
    - Class 1: Necrotic core
    - Class 2: Edema
    - Class 3: Enhancing tumor
    """

    def __init__(self, in_ch=4, num_regions=4, num_cls=2, ...):
        super().__init__()
        self.seg = SegUNetMasked(in_ch=in_ch, base=base, ...)

        # Multi-class segmentation head
        self.seg_head = nn.Conv2d(base, num_regions, 1)

        # ... rest similar ...

    def forward(self, x):
        # Get features
        features = self.seg.encoder(x)

        # Multi-class segmentation
        seg_logits = self.seg_head(features)  # (B, 4, H, W)

        # Classification (same as before)
        cls_logits = self.cls_backbone(...)

        return seg_logits, cls_logits
```

**Update loss**:

```python
# Use CrossEntropyLoss for multi-class
seg_criterion = nn.CrossEntropyLoss()

# In training loop
seg_loss = seg_criterion(seg_logits, mask_multi_class)
# mask_multi_class shape: (B, H, W) with values in [0, 1, 2, 3]
```

---

## Deployment and Production

### Example 1: ONNX Export

**Export model**:

```python
import torch
import torch.onnx

# Load trained model
model = BrainTumNet(in_ch=4, ...)
load_ckpt(model, "checkpoints/best_model.pth")
model.eval()

# Dummy input
dummy_input = torch.randn(1, 4, 256, 256)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "braintumnet.onnx",
    input_names=["input"],
    output_names=["segmentation", "classification"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "segmentation": {0: "batch_size"},
        "classification": {0: "batch_size"}
    },
    opset_version=11
)

print("Model exported to braintumnet.onnx")
```

**Use ONNX for inference**:

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("braintumnet.onnx")

# Prepare input
img = load_and_preprocess_image("patient_001.png")
img_np = img.numpy()  # (1, 4, 256, 256)

# Run inference
outputs = session.run(None, {"input": img_np})
seg_logits, cls_logits = outputs

# Post-process
seg_prob = 1 / (1 + np.exp(-seg_logits))  # Sigmoid
seg_binary = (seg_prob > 0.5).astype(np.uint8)

print(f"Segmentation shape: {seg_binary.shape}")
print(f"Classification: {cls_logits.argmax()}")
```

---

### Example 2: TorchScript

**For deployment in C++**:

```python
import torch

# Load model
model = BrainTumNet(...)
load_ckpt(model, "checkpoints/best_model.pth")
model.eval()

# Trace model
example_input = torch.randn(1, 4, 256, 256)
traced_model = torch.jit.trace(model, example_input)

# Save
traced_model.save("braintumnet_traced.pt")

# Load in Python
loaded = torch.jit.load("braintumnet_traced.pt")
output = loaded(example_input)
```

**Use in C++**:

```cpp
#include <torch/script.h>
#include <iostream>

int main() {
    // Load model
    torch::jit::script::Module model = torch::jit::load("braintumnet_traced.pt");

    // Create input tensor
    auto input = torch::randn({1, 4, 256, 256});

    // Run inference
    auto outputs = model.forward({input}).toTuple();
    auto seg = outputs->elements()[0].toTensor();
    auto cls = outputs->elements()[1].toTensor();

    std::cout << "Inference complete!" << std::endl;
    return 0;
}
```

---

### Example 3: Flask Web API

**Create** `app.py`:

```python
from flask import Flask, request, jsonify
import torch
import numpy as np
from PIL import Image
import io
import base64

from braintumnet.models.braintumnet import BrainTumNet
from braintumnet.utils.io import load_ckpt
from braintumnet.data.transforms import resize_pad_to_square, to_tensor01

app = Flask(__name__)

# Load model once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BrainTumNet(in_ch=4, num_cls=2, base=32, dim=256, patch=8, depth=2, n_heads=4)
load_ckpt(model, "checkpoints/best_model.pth", map_location=device)
model.to(device)
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict on uploaded image.

    Request:
        {
            "image": "base64_encoded_image_data"
        }

    Response:
        {
            "segmentation": "base64_encoded_mask",
            "classification": "HGG" or "LGG",
            "confidence": 0.95
        }
    """
    # Get image from request
    data = request.json
    img_data = base64.b64decode(data["image"])
    img = Image.open(io.BytesIO(img_data)).convert("L")

    # Preprocess
    img_resized = resize_pad_to_square(img, 256, is_mask=False)
    img_tensor = to_tensor01(img_resized).unsqueeze(0).to(device)  # (1, 1, 256, 256)

    # Predict
    with torch.no_grad():
        seg_logits, cls_logits = model(img_tensor)
        seg_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
        cls_prob = torch.softmax(cls_logits, dim=1).squeeze().cpu().numpy()
        cls_pred = cls_prob.argmax()

    # Convert segmentation to image
    seg_binary = (seg_prob > 0.5).astype(np.uint8) * 255
    seg_img = Image.fromarray(seg_binary)

    # Encode segmentation as base64
    buffered = io.BytesIO()
    seg_img.save(buffered, format="PNG")
    seg_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Response
    response = {
        "segmentation": seg_base64,
        "classification": "HGG" if cls_pred == 0 else "LGG",
        "confidence": float(cls_prob[cls_pred])
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

**Run API**:

```bash
python app.py

# Test with curl
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_data..."}'
```

---

## Summary

This guide covered:

1. ✓ Adding model components (SE blocks, residual connections)
2. ✓ Creating new loss functions (Focal, Boundary)
3. ✓ Implementing new metrics (Sensitivity, Specificity)
4. ✓ Adding augmentations (Elastic deformation)
5. ✓ Supporting new datasets (TCGA)
6. ✓ Building 3D models (UNet3D)
7. ✓ Multi-task extensions (Multi-region segmentation)
8. ✓ Deployment (ONNX, TorchScript, Flask API)

**Key Principles**:
- Keep code modular
- Use config files for parameters
- Test each addition
- Document changes

**Next Steps**:
- Read papers for new techniques
- Experiment with ablations
- Benchmark performance
- Deploy to production

---

**Congratulations!** You now have complete documentation for BrainTumNet.

**Return to**: [[TECHNICAL_REPORT_INDEX|Index]]
