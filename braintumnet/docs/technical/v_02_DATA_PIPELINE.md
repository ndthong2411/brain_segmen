# Part 2: Data Pipeline Deep Dive (Phase 2 - Updated 2025-10-28)

> **📊 Complete guide from raw H5 files to PyTorch tensors ready for training**
>
> This document explains EVERY STEP of data processing with line-by-line code explanations for Phase 2 multi-class segmentation.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [File Analysis: preprocess_h5_to_multiclass.py](#2-file-analysis-preprocess_h5_to_multiclasspy)
3. [File Analysis: brats2020_dataset.py](#3-file-analysis-brats2020_datasetpy)
4. [File Analysis: transforms.py](#4-file-analysis-transformspy)
5. [Complete Data Flow](#5-complete-data-flow)
6. [Modification Guide](#6-modification-guide)
7. [Debugging Tips](#7-debugging-tips)

---

## 1. Pipeline Overview

### The Big Picture

The data pipeline has 3 main stages:

```
STAGE 1: Preprocessing (ONE-TIME)
Raw BraTS H5 → Processed PNG (multi-class)
[preprocess_h5_to_multiclass.py]

STAGE 2: Dataset Loading (EVERY EPOCH)
Processed files → PyTorch Dataset
[brats2020_dataset.py]

STAGE 3: Augmentation (EVERY BATCH)
Original data → Augmented variants
[transforms.py]
```

### Why This Design?

**Stage 1** - Preprocess once, use forever:
- ✅ H5 files are slow for repeated reading
- ✅ Convert to PNG format once
- ✅ Normalize and resize once
- ✅ Save disk space (only keep good slices)
- ✅ **NEW**: Convert 4-class BraTS → 3-class multi-class

**Stage 2** - Lazy loading:
- ✅ Don't load all 23GB into RAM
- ✅ Only load what's needed for current batch
- ✅ PyTorch DataLoader handles parallelization

**Stage 3** - On-the-fly augmentation:
- ✅ Create infinite variants (rotation, flip)
- ✅ Don't save augmented images (waste space)
- ✅ Different augmentation each epoch

---

## 2. File Analysis: preprocess_h5_to_multiclass.py

**Location**: `scripts/preprocess_h5_to_multiclass.py`
**Total Lines**: ~380
**Purpose**: Convert raw BraTS2020 H5 files to multi-class PNG format

### 2.1 Imports and Setup

```python
import os, sys
from pathlib import Path
import argparse
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import KFold
```

**Each import does**:
- `h5py`: Read H5 files (BraTS format)
- `PIL.Image`: Save PNG images
- `tqdm`: Progress bar
- `numpy`: Array operations
- `pandas`: CSV handling
- `KFold`: Cross-validation splits

### 2.2 Loading H5 Data

```python
def load_h5_data(h5_path):
    """Load H5 file and return image and mask.

    Args:
        h5_path: Path to H5 file

    Returns:
        image: (H, W, 4) numpy array - 4 modalities
        mask: (H, W, 3) numpy array - 3 binary channels
    """
    with h5py.File(h5_path, 'r') as f:
        image = f['image'][:]  # (240, 240, 4)
        mask = f['mask'][:]    # (240, 240, 3)
    return image, mask
```

**H5 Structure**:
- `image`: (240, 240, 4) - 4 MRI modalities [FLAIR, T1, T1CE, T2]
- `mask`: (240, 240, 3) - 3 binary channels [NCR, ED, ET]

### 2.3 Multi-Class Conversion ⭐ **NEW**

This is the **KEY FUNCTION** for Phase 2 multi-class segmentation:

```python
def convert_mask_to_3class(mask_3ch):
    """Convert 3-channel binary mask to 3-class single-channel mask.

    Args:
        mask_3ch: (H, W, 3) binary mask where each channel is 0 or 1

    Returns:
        mask_3class: (H, W) uint8 with values {0, 1, 2}
            0 = Background
            1 = Tumor Core (TC) - from channel 1
            2 = Edema (ED) - from channel 2
    """
    H, W, C = mask_3ch.shape
    mask_3class = np.zeros((H, W), dtype=np.uint8)

    # Priority: TC > ED > Background
    # Channel 2 = Edema → class 2
    mask_3class[mask_3ch[:, :, 2] > 0] = 2

    # Channel 1 = Tumor Core → class 1 (overwrites edema if overlapping)
    mask_3class[mask_3ch[:, :, 1] > 0] = 1

    # Channel 0 is ignored (not used in BraTS standard regions)
    # Background remains 0

    return mask_3class
```

**Mapping Explained**:

```
Original BraTS H5 Format (3 binary channels):
- Channel 0: Necrotic Core (NCR) - mostly ignored
- Channel 1: Tumor Core components → TC
- Channel 2: Peritumoral Edema → ED

BrainTumNet Phase 2 Format (3-class single channel):
- Class 0: Background (healthy brain)
- Class 1: Tumor Core (TC) - enhancing + necrotic
- Class 2: Edema (ED) - peritumoral swelling

Priority: TC (1) > ED (2) > Background (0)
If pixel is both TC and ED → TC wins (class 1)
```

**Example**:

```python
# Input: 3-channel binary mask
mask_3ch = np.array([
    [[0, 1, 0],  # Pixel has TC
     [0, 0, 1],  # Pixel has ED
     [0, 1, 1]], # Pixel has both TC and ED
])

# Output: 3-class mask
mask_3class = convert_mask_to_3class(mask_3ch)
# Result: [[1, 2, 1]]  # TC, ED, TC (priority)
```

**Why This Mapping?**

✅ **Aligns with BraTS evaluation**:
- WT (Whole Tumor) = TC + ED (classes 1, 2)
- TC (Tumor Core) = class 1
- ED (Edema) = class 2

✅ **Clinically meaningful**: Distinguishes active tumor from surrounding swelling

✅ **Balanced classes**: More stable training than 4-class

### 2.4 Image Normalization

```python
def normalize_image(image, modality_idx):
    """Normalize image to [0, 255] uint8.

    Args:
        image: (H, W) float array
        modality_idx: Modality index (0=FLAIR, 1=T1, 2=T1CE, 3=T2)

    Returns:
        normalized: (H, W) uint8 in [0, 255]
    """
    # Remove background
    brain_mask = image > 0

    if brain_mask.sum() == 0:
        return np.zeros_like(image, dtype=np.uint8)

    # Compute percentiles on brain region
    p1 = np.percentile(image[brain_mask], 1)
    p99 = np.percentile(image[brain_mask], 99)

    # Clip and normalize
    image_clipped = np.clip(image, p1, p99)
    image_norm = (image_clipped - p1) / (p99 - p1 + 1e-8)
    image_norm = (image_norm * 255).astype(np.uint8)

    return image_norm
```

**Why Percentile Normalization?**

```
Problem with min/max:
  Min=0, Max=10000 (outlier!)
  Normal tissue [100-500] → compressed to [0.01-0.05]

Solution with percentiles:
  P1=100, P99=500
  Normal tissue [100-500] → stretched to [0-255]
  Outliers clipped
```

**Step by step**:
1. Create brain mask (ignore background zeros)
2. Compute 1st and 99th percentiles (robust to outliers)
3. Clip values to [p1, p99]
4. Normalize to [0, 1]
5. Scale to [0, 255] uint8

### 2.5 Processing Single H5 File

```python
def process_h5_file(h5_path, out_dir, img_size=256):
    """Process a single H5 file and save PNG outputs.

    Args:
        h5_path: Path to H5 file
        out_dir: Output directory
        img_size: Target image size

    Returns:
        slice_info: Dict with metadata, or None if error
    """
    # Extract slice_id from filename (e.g., volume_1_slice_50.h5)
    fname = Path(h5_path).stem  # volume_1_slice_50

    # Load data
    try:
        image, mask_3ch = load_h5_data(h5_path)
    except Exception as e:
        print(f"Error loading {h5_path}: {e}")
        return None

    # Convert mask to 3-class ⭐
    mask_3class = convert_mask_to_3class(mask_3ch)

    # Extract volume and slice ID
    # volume_1_slice_50 → vol1_slice50
    parts = fname.split('_')
    vol_id = f"vol{parts[1]}"
    slice_id = f"slice{parts[3]}"
    output_id = f"{vol_id}_{slice_id}"

    # Save each modality
    modality_names = ['flair', 't1', 't1ce', 't2']
    for mod_idx, mod_name in enumerate(modality_names):
        mod_dir = out_dir / mod_name
        mod_dir.mkdir(parents=True, exist_ok=True)

        # Normalize and resize
        img_2d = normalize_image(image[:, :, mod_idx], mod_idx)
        img_resized = resize_array(img_2d, img_size, is_mask=False)

        # Save as PNG
        save_path = mod_dir / f"{output_id}.png"
        Image.fromarray(img_resized).save(save_path)

    # Save segmentation mask
    seg_dir = out_dir / "seg"
    seg_dir.mkdir(parents=True, exist_ok=True)

    mask_resized = resize_array(mask_3class, img_size, is_mask=True)
    save_path = seg_dir / f"{output_id}.png"
    Image.fromarray(mask_resized, mode='L').save(save_path)

    # Compute statistics
    has_tc = (mask_resized == 1).any()
    has_ed = (mask_resized == 2).any()
    has_wt = has_tc or has_ed

    # Determine primary label
    if has_tc and has_ed:
        label = "WT"  # Whole tumor
    elif has_tc:
        label = "TC"  # Tumor core only
    elif has_ed:
        label = "ED"  # Edema only
    else:
        label = "Normal"

    # Create metadata
    slice_info = {
        'slice_id': output_id,
        'volume_id': vol_id,
        'slice_idx': int(parts[3]),
        'label': label,
        'has_wt': int(has_wt),
        'has_tc': int(has_tc),
        'has_ed': int(has_ed),
    }

    return slice_info
```

**Output Structure**:

```
processed_multiclass/
├── flair/
│   ├── vol1_slice50.png  (256×256 uint8)
│   ├── vol1_slice51.png
│   └── ...
├── t1/
│   ├── vol1_slice50.png
│   └── ...
├── t1ce/
│   ├── vol1_slice50.png
│   └── ...
├── t2/
│   ├── vol1_slice50.png
│   └── ...
└── seg/
    ├── vol1_slice50.png  (256×256 with values {0,1,2})
    └── ...
```

### 2.6 K-Fold Split Creation

```python
def create_kfold_splits(all_slices_df, num_folds=5, seed=42):
    """Create K-fold splits at volume level.

    Args:
        all_slices_df: DataFrame with all slices
        num_folds: Number of folds
        seed: Random seed

    Returns:
        splits: List of (train_indices, val_indices) tuples
    """
    # Get unique volumes
    volume_ids = all_slices_df['volume_id'].unique()

    # Create K-fold splitter
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    splits = []
    for train_vols, val_vols in kf.split(volume_ids):
        train_vol_ids = volume_ids[train_vols]
        val_vol_ids = volume_ids[val_vols]

        # Get slice indices
        train_indices = all_slices_df[all_slices_df['volume_id'].isin(train_vol_ids)].index.tolist()
        val_indices = all_slices_df[all_slices_df['volume_id'].isin(val_vol_ids)].index.tolist()

        splits.append((train_indices, val_indices))

    return splits
```

**Why Split by Volume?**

❌ **Wrong** (split by slices):
```
Train: vol1_slice1, vol1_slice2, vol2_slice1
Val:   vol1_slice3, vol2_slice2
```
Problem: Slices from same volume in both train and val → **data leakage**!

✅ **Correct** (split by volumes):
```
Train: vol1_slice1, vol1_slice2, vol1_slice3
Val:   vol2_slice1, vol2_slice2, vol2_slice3
```
No leakage: Completely separate patients

---

## 3. File Analysis: brats2020_dataset.py

**Location**: `src/braintumnet/data/brats2020_dataset.py`
**Purpose**: PyTorch Dataset class for loading processed multi-class data

### 3.1 Dataset Class Definition

```python
class MultiClassSliceDataset(Dataset):
    """
    BraTS 2020 multi-class slice-based dataset.

    Loads preprocessed 2D slices with 3-class segmentation.
    Supports optional augmentation for training.
    """

    def __init__(
        self,
        proc_root: str,       # "data/processed_multiclass"
        split_csv: str,       # "train_fold0.csv"
        img_size: int=256,    # Image size
        rotate_deg: int=30,   # Augmentation: rotation range
        hflip_p: float=0.5,   # Augmentation: horizontal flip probability
        vflip_p: float=0.5,   # Augmentation: vertical flip probability
        train: bool=True      # Apply augmentation?
    ):
        self.proc_root = Path(proc_root)
        self.train = train
        self.img_size = img_size
        self.rotate_deg = rotate_deg
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p

        # Load split CSV
        df = pd.read_csv(split_csv)
        self.slice_ids = df['slice_id'].tolist()

        # Load case labels
        labels_csv = self.proc_root / "labels.csv"
        if labels_csv.exists():
            labels_df = pd.read_csv(labels_csv)
            self.case_labels = dict(zip(labels_df['case_id'], labels_df['grade']))
        else:
            self.case_labels = {}

        print(f"Loaded {len(self.slice_ids)} slices from {split_csv}")
```

### 3.2 Loading Multi-Modal Images

```python
def _load_multimodal_image(self, slice_id: str):
    """Load all 4 modalities for a slice.

    Args:
        slice_id: e.g., "vol1_slice50"

    Returns:
        image: (4, H, W) numpy array - [FLAIR, T1, T1CE, T2]
    """
    modalities = ['flair', 't1', 't1ce', 't2']
    channels = []

    for mod in modalities:
        img_path = self.proc_root / mod / f"{slice_id}.png"
        img = Image.open(img_path).convert('L')  # Grayscale
        img_arr = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
        channels.append(img_arr)

    # Stack to (4, H, W)
    image = np.stack(channels, axis=0)
    return image
```

### 3.3 Loading Multi-Class Mask

```python
def _load_multiclass_mask(self, slice_id: str):
    """Load 3-class segmentation mask.

    Args:
        slice_id: e.g., "vol1_slice50"

    Returns:
        mask: (H, W) numpy array with values {0, 1, 2}
    """
    mask_path = self.proc_root / "seg" / f"{slice_id}.png"
    mask = Image.open(mask_path).convert('L')
    mask_arr = np.array(mask, dtype=np.int64)  # int64 for CrossEntropyLoss
    return mask_arr
```

**Important**: Mask is `int64` (not float) for PyTorch `CrossEntropyLoss`!

### 3.4 Getting a Sample

```python
def __getitem__(self, idx):
    """
    Get a sample.

    Args:
        idx: Index in [0, len(dataset)-1]

    Returns:
        Dictionary with:
        - image: (4, H, W) tensor, float32
        - mask: (H, W) tensor, int64 with values {0,1,2}
        - label: scalar tensor, classification label
        - slice_id: string
        - case_id: string
    """
    slice_id = self.slice_ids[idx]

    # Load data
    image = self._load_multimodal_image(slice_id)  # (4, H, W)
    mask = self._load_multiclass_mask(slice_id)    # (H, W)

    # Apply augmentation if training
    if self.train:
        image, mask = augment_multimodal_pair(
            image, mask,
            self.rotate_deg,
            self.hflip_p,
            self.vflip_p
        )

    # Convert to tensors
    image_tensor = torch.from_numpy(image).float()  # (4, H, W)
    mask_tensor = torch.from_numpy(mask).long()     # (H, W) int64

    # Get classification label
    case_id = slice_id.split('_')[0]  # "vol1_slice50" → "vol1"
    label = self.case_labels.get(case_id, 0)  # HGG=0, LGG=1

    return {
        "image": image_tensor,
        "mask": mask_tensor,
        "label": torch.tensor(label, dtype=torch.long),
        "slice_id": slice_id,
        "case_id": case_id
    }
```

**Key Points**:
- Image: `float32` in [0, 1] range
- Mask: `int64` (long) with class indices {0, 1, 2}
- Multi-modal: All 4 modalities stacked in channel dimension

---

## 4. File Analysis: transforms.py

**Location**: `src/braintumnet/data/transforms.py`

### 4.1 Augmentation for Multi-Modal Data

```python
def augment_multimodal_pair(image, mask, rotate_deg=30, hflip_p=0.5, vflip_p=0.5):
    """
    Apply same augmentations to multi-modal image and mask.

    Args:
        image: (C, H, W) numpy array - C modalities
        mask: (H, W) numpy array - integer class labels
        rotate_deg: Rotation range ±degrees
        hflip_p: Horizontal flip probability
        vflip_p: Vertical flip probability

    Returns:
        image_aug: (C, H, W) augmented image
        mask_aug: (H, W) augmented mask
    """
    import random
    import torch
    import torchvision.transforms.functional as TF

    # Convert to PIL for augmentation
    # Each modality separately
    C, H, W = image.shape
    image_pils = [Image.fromarray((image[c] * 255).astype(np.uint8)) for c in range(C)]
    mask_pil = Image.fromarray(mask.astype(np.uint8))

    # Random rotation
    angle = random.uniform(-rotate_deg, rotate_deg)
    image_pils = [TF.rotate(img, angle, fill=0) for img in image_pils]
    mask_pil = TF.rotate(mask_pil, angle, fill=0, interpolation=Image.NEAREST)

    # Random horizontal flip
    if random.random() < hflip_p:
        image_pils = [TF.hflip(img) for img in image_pils]
        mask_pil = TF.hflip(mask_pil)

    # Random vertical flip
    if random.random() < vflip_p:
        image_pils = [TF.vflip(img) for img in image_pils]
        mask_pil = TF.vflip(mask_pil)

    # Convert back to numpy
    image_aug = np.stack([np.array(img, dtype=np.float32) / 255.0 for img in image_pils], axis=0)
    mask_aug = np.array(mask_pil, dtype=np.int64)

    return image_aug, mask_aug
```

**CRITICAL**: Same transformation for ALL modalities and mask!

```
Original:
FLAIR: Brain at 0°
T1CE:  Brain at 0°
Mask:  Tumor at 0°

After rotation +15°:
FLAIR: Brain at +15°  ← Same rotation
T1CE:  Brain at +15°  ← Same rotation
Mask:  Tumor at +15°  ← Same rotation (matches!)
```

---

## 5. Complete Data Flow

### 5.1 From Raw to Batch (Step by Step)

```
1. RAW DATA (one-time setup)
   File: volume_1_slice_50.h5
   Content: image (240×240×4), mask (240×240×3 binary)
   Size: ~2 MB

        ↓ [preprocess_h5_to_multiclass.py]

2. PREPROCESSED (saved to disk)
   Files:
     - flair/vol1_slice50.png (256×256 uint8)
     - t1/vol1_slice50.png
     - t1ce/vol1_slice50.png
     - t2/vol1_slice50.png
     - seg/vol1_slice50.png (256×256 with {0,1,2})
   Size: ~200 KB total

        ↓ [MultiClassSliceDataset.__init__]

3. DATASET READY
   slice_ids = ["vol1_slice50", "vol1_slice51", ...]
   Length: 45,756 slices (train fold 0)

        ↓ [MultiClassSliceDataset.__getitem__(idx)]

4. LOAD ONE SAMPLE
   image = load 4 PNGs, stack → (4, 256, 256)
   mask = load 1 PNG → (256, 256) with {0,1,2}

        ↓ [augment_multimodal_pair() if training]

5. AUGMENTED SAMPLE
   image: rotate 15°, flip horizontally
   mask: rotate 15°, flip horizontally (same transforms!)

        ↓ [to_tensor()]

6. TENSOR SAMPLE
   image_tensor: (4, 256, 256) float32
   mask_tensor: (256, 256) int64
   label: tensor(0) or tensor(1)

        ↓ [DataLoader collate]

7. BATCH
   images: (16, 4, 256, 256)  # batch_size=16
   masks: (16, 256, 256)      # int64 for CrossEntropyLoss
   labels: (16,)

        ↓ [model.forward()]

8. PREDICTIONS
   seg_logits: (16, 3, 256, 256)  # 3 class logits
   cls_logits: (16, 2)             # HGG/LGG logits
   aux_outputs: [(16,3,H/4,W/4), (16,3,H/2,W/2), (16,3,H,W)]  # Deep supervision
```

### 5.2 Memory Flow

```
Disk → RAM → GPU

Disk:
- All 57,195 slices saved as PNG
- Total: ~11 GB (4 modalities + 1 mask per slice)

RAM (during training):
- DataLoader loads batch_size × num_workers slices
- Example: 16 × 4 = 64 slices in RAM
- ~50 MB (negligible)

GPU:
- One batch: 16 slices
- Images: 16 × 4 × 256 × 256 × 4 bytes = 16 MB
- Masks: 16 × 1 × 256 × 256 × 8 bytes = 8 MB
- Model: ~87M parameters × 4 bytes = 348 MB
- Activations: ~2 GB (in forward pass)
- Gradients: ~348 MB
- Optimizer state: ~696 MB (Adam)
- Total: ~3.4 GB per batch (Phase 2 Large)
```

**GPU Memory for Different Batch Sizes**:
```
batch_size=8:  ~2.0 GB  ✅ RTX 3060 (12GB)
batch_size=12: ~2.8 GB  ✅ RTX 3090 (24GB)
batch_size=16: ~3.4 GB  ✅ A100 (80GB)
batch_size=32: ~6.2 GB  ✅ A100 (80GB) with gradient checkpointing
```

---

## 6. Modification Guide

### 6.1 Change Number of Classes

**Current**: 3 classes (background, TC, ED)
**Want**: 4 classes (background, NCR, ED, ET)

**Step 1**: Modify `convert_mask_to_3class()`:

```python
def convert_mask_to_4class(mask_3ch):
    """4-class: bg=0, NCR=1, ED=2, ET=3"""
    H, W, C = mask_3ch.shape
    mask_4class = np.zeros((H, W), dtype=np.uint8)

    # Channel 2 (Edema) → class 2
    mask_4class[mask_3ch[:, :, 2] > 0] = 2

    # Channel 0 (NCR) → class 1
    mask_4class[mask_3ch[:, :, 0] > 0] = 1

    # Channel 1 (ET) → class 3
    mask_4class[mask_3ch[:, :, 1] > 0] = 3

    return mask_4class
```

**Step 2**: Update config:

```yaml
model:
  num_classes_seg: 4  # Changed from 3
```

**Step 3**: Update loss class weights:

```yaml
train:
  class_weights: [1.0, 3.0, 2.5, 4.0]  # [bg, NCR, ED, ET]
```

### 6.2 Add New Augmentation: Elastic Deformation

```python
from scipy.ndimage import gaussian_filter, map_coordinates

def elastic_deform(image, mask, alpha=30, sigma=5):
    """Apply elastic deformation to multi-modal image and mask."""
    C, H, W = image.shape

    # Random displacement field
    dx = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma) * alpha

    # Grid
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # Apply to each modality
    image_deformed = np.zeros_like(image)
    for c in range(C):
        image_deformed[c] = map_coordinates(image[c], indices, order=1).reshape(H, W)

    # Apply to mask (order=0 for nearest neighbor)
    mask_deformed = map_coordinates(mask, indices, order=0).reshape(H, W)

    return image_deformed, mask_deformed
```

**Add to augmentation**:

```python
def augment_multimodal_pair(image, mask, ...):
    # ... existing augmentations ...

    # Elastic deformation (20% chance)
    if random.random() < 0.2:
        image, mask = elastic_deform(image, mask)

    return image, mask
```

### 6.3 Change Normalization Strategy

**Current**: Percentile-based [p1, p99]
**Want**: Z-score normalization

```python
def normalize_zscore(image, modality_idx):
    """Z-score normalization: (x - mean) / std"""
    brain_mask = image > 0

    if brain_mask.sum() == 0:
        return np.zeros_like(image, dtype=np.float32)

    mean = image[brain_mask].mean()
    std = image[brain_mask].std()

    normalized = np.zeros_like(image, dtype=np.float32)
    normalized[brain_mask] = (image[brain_mask] - mean) / (std + 1e-8)

    # Clip to reasonable range
    normalized = np.clip(normalized, -5, 5)

    # Scale to [0, 1] for PNG saving
    normalized = (normalized + 5) / 10  # [-5, 5] → [0, 1]

    return (normalized * 255).astype(np.uint8)
```

---

## 7. Debugging Tips

### 7.1 Visualize Preprocessing Output

```python
import matplotlib.pyplot as plt

# Load one slice
proc_root = Path("data/processed_multiclass")
slice_id = "vol1_slice50"

# Load all modalities
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

modalities = ['flair', 't1', 't1ce', 't2']
for i, mod in enumerate(modalities):
    img = Image.open(proc_root / mod / f"{slice_id}.png")
    axes[i//3, i%3].imshow(img, cmap='gray')
    axes[i//3, i%3].set_title(f"{mod.upper()}")
    axes[i//3, i%3].axis('off')

# Load mask
mask = Image.open(proc_root / "seg" / f"{slice_id}.png")
mask_arr = np.array(mask)

# Visualize mask with colors
axes[1, 1].imshow(mask_arr, cmap='jet', vmin=0, vmax=2)
axes[1, 1].set_title("Segmentation (0=bg, 1=TC, 2=ED)")
axes[1, 1].axis('off')

# Stats
axes[1, 2].axis('off')
stats_text = f"Mask Statistics:\n"
stats_text += f"Background: {(mask_arr == 0).sum()} pixels\n"
stats_text += f"TC (class 1): {(mask_arr == 1).sum()} pixels\n"
stats_text += f"ED (class 2): {(mask_arr == 2).sum()} pixels\n"
axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12)

plt.tight_layout()
plt.savefig(f"{slice_id}_visualization.png", dpi=150, bbox_inches='tight')
print(f"Saved visualization to {slice_id}_visualization.png")
```

### 7.2 Check Class Distribution

```python
import pandas as pd
from collections import Counter

# Load all slices info
df = pd.read_csv("data/processed_multiclass/all_slices.csv")

# Count labels
label_counts = Counter(df['label'])
print("\nLabel distribution:")
for label, count in label_counts.items():
    print(f"  {label}: {count} slices ({count/len(df)*100:.1f}%)")

# Check class balance per fold
for fold in range(5):
    train_df = pd.read_csv(f"data/processed_multiclass/train_fold{fold}.csv")
    val_df = pd.read_csv(f"data/processed_multiclass/val_fold{fold}.csv")

    print(f"\nFold {fold}:")
    print(f"  Train: {len(train_df)} slices")
    print(f"  Val:   {len(val_df)} slices")
    print(f"  Ratio: {len(train_df)/len(val_df):.2f}")
```

### 7.3 Test DataLoader Speed

```python
from torch.utils.data import DataLoader
from braintumnet.data.brats2020_dataset import MultiClassSliceDataset
import time

# Create dataset
dataset = MultiClassSliceDataset(
    proc_root="data/processed_multiclass",
    split_csv="data/processed_multiclass/train_fold0.csv",
    train=True
)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Time 10 batches
start = time.time()
for i, batch in enumerate(loader):
    if i >= 10:
        break
end = time.time()

print(f"Average time per batch: {(end-start)/10:.3f} seconds")
print(f"Expected time per epoch: {(end-start)/10 * len(loader) / 60:.1f} minutes")
```

**Expected results**:
- Good: <0.5 seconds per batch
- Acceptable: 0.5-1 second per batch
- Slow: >1 second per batch (check num_workers, disk speed)

---

## Summary

### What We Learned

✅ **Stage 1**: Preprocessing (`preprocess_h5_to_multiclass.py`)
   - Read H5 → Normalize → Convert to 3-class → Save PNG
   - **Key**: `convert_mask_to_3class()` for multi-class segmentation
   - Create metadata (all_slices.csv, labels.csv)
   - Generate K-fold splits at volume level

✅ **Stage 2**: Dataset (`brats2020_dataset.py`)
   - PyTorch Dataset class for multi-class
   - Lazy loading (load on demand)
   - Load 4 modalities + 3-class mask
   - Return int64 mask for CrossEntropyLoss

✅ **Stage 3**: Augmentation (`transforms.py`)
   - Rotation, flipping for multi-modal
   - Apply SAME transforms to all modalities and mask
   - On-the-fly during training

### Key Takeaways

1. **Multi-class conversion is crucial**: 3-channel binary → 3-class single channel
2. **Lazy loading**: Don't load all 11GB into RAM at once
3. **On-the-fly augmentation**: Infinite data variants
4. **Volume-level splits**: Prevent data leakage
5. **Mask dtype matters**: int64 (long) for CrossEntropyLoss

### Next Steps

Now you understand the complete data pipeline for Phase 2!

👉 **Next**: [Part 3 - Model Architecture](v_03_MODEL_ARCHITECTURE.md)

Learn how SegUNetV2 processes multi-class data!

---

[[v_TECHNICAL_REPORT_INDEX|← Back to Index]] | [[v_01_PROJECT_OVERVIEW|← Previous: Overview]] | [[v_03_MODEL_ARCHITECTURE|Next: Architecture →]]
