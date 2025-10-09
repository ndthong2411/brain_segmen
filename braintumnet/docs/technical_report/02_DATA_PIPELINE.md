# Part 2: Data Pipeline Deep Dive

> **📊 Complete guide from raw HDF5 files to training-ready PyTorch tensors**
>
> This part explains EVERY step of data processing with line-by-line code explanation.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [File Analysis: prepare_brats2020_h5.py](#2-file-analysis-prepare_brats2020_h5py)
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
STAGE 1: Preprocessing (ONE TIME)
Raw BraTS HDF5 → Preprocessed PNG/NPY
[prepare_brats2020_h5.py]

STAGE 2: Dataset Loading (EVERY EPOCH)
Preprocessed files → PyTorch Dataset
[brats2020_dataset.py]

STAGE 3: Augmentation (EVERY BATCH)
Original data → Augmented variants
[transforms.py]
```

### Why This Design?

**Stage 1** - Preprocess once, use forever:
- ✅ HDF5 is slow to read repeatedly
- ✅ Convert to fast PNG/NPY format once
- ✅ Normalize and resize once
- ✅ Save disk space (only keep good slices)

**Stage 2** - Lazy loading:
- ✅ Don't load all 23GB into RAM
- ✅ Load only what's needed for current batch
- ✅ PyTorch DataLoader handles parallelization

**Stage 3** - On-the-fly augmentation:
- ✅ Create infinite variations (rotation, flip)
- ✅ Don't store augmented images (waste space)
- ✅ Different augmentation every epoch

---

## 2. File Analysis: prepare_brats2020_h5.py

**Location**: `scripts/prepare_brats2020_h5.py`
**Total Lines**: 416 lines
**Purpose**: Convert raw BraTS2020 HDF5 files to preprocessed format

### 2.1 Imports and Setup

```python
import os, argparse, csv, h5py
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
```

**What each import does**:
- `h5py`: Read HDF5 files (BraTS format)
- `PIL.Image`: Save PNG images
- `tqdm`: Progress bar
- `numpy`: Array operations
- `csv`: Read/write CSV metadata

### 2.2 Helper Function: `_rescale01()`

**Purpose**: Normalize image to [0, 1] range

```python
def _rescale01(arr: np.ndarray) -> np.ndarray:
    """
    Rescale array to [0, 1] range, ignoring background (zeros).

    Args:
        arr: Input array (e.g., 240x240 MRI slice)

    Returns:
        Normalized array in [0, 1]
    """
    arr = arr.astype(np.float32)

    # Only consider non-zero pixels (brain tissue, not background)
    nz = arr > 0

    if nz.sum() > 0:
        # Get min/max from brain tissue only
        a = arr[nz]
        lo, hi = a.min(), a.max()
    else:
        # All zeros (shouldn't happen, but handle it)
        lo, hi = arr.min(), arr.max()

    # Avoid division by zero
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)

    # Normalize
    out = (arr - lo) / (hi - lo)

    # Handle NaN/Inf (shouldn't happen, but be safe)
    out[~np.isfinite(out)] = 0

    return out
```

**Why ignore background?**
- MRI background is always 0 (no signal)
- We want to normalize brain tissue intensity
- Example: Brain tissue ranges [100, 1000] → normalize to [0, 1]
- Background stays 0

**Example**:
```python
# Input: MRI slice with background
arr = np.array([[0, 0, 0],
                [0, 100, 200],
                [0, 150, 250]])

# Output after _rescale01():
# [[0.0, 0.0, 0.0],
#  [0.0, 0.0, 0.667],
#  [0.0, 0.333, 1.0]]
# Brain tissue [100-250] mapped to [0-1], background stays 0
```

### 2.3 Helper Function: `_save_png01()`

```python
def _save_png01(x: np.ndarray, path: str):
    """
    Save normalized [0, 1] array as PNG [0, 255].

    Args:
        x: Array in [0, 1] range
        path: Output PNG path
    """
    # Scale to [0, 255]
    x = (x * 255.0).clip(0, 255).astype(np.uint8)

    # Save using PIL
    Image.fromarray(x).save(path)
```

**Why clip?**
- Sometimes floating point errors cause values slightly outside [0, 1]
- `clip(0, 255)` ensures safe range

### 2.4 Main Function: `process_h5_brats2020()` - Part 1: Setup

```python
def process_h5_brats2020(
    h5_root: str,              # "data/raw" - where HDF5 files are
    meta_csv: str,             # "data/raw/meta_data.csv" - labels
    out_root: str,             # "data/processed_full_multimodal" - output
    img_size: int=256,         # Resize to this size
    modality_idx: int=2,       # 0=FLAIR, 1=T1, 2=T1CE, 3=T2
    max_slices: int=None,      # Limit for testing (None = all)
    multimodal: bool=False,    # Save all 4 channels?
    min_tumor_ratio: float=0.001  # Skip slices with tiny tumors
):
```

**Parameters explained**:

1. **h5_root**: Where your raw HDF5 files are
   ```
   data/raw/
   ├── BraTS2020_Training_001_slice_100.h5
   ├── BraTS2020_Training_001_slice_101.h5
   └── ...
   ```

2. **meta_csv**: CSV with metadata
   ```csv
   slice_path,target,volume,slice
   BraTS2020_Training_001_slice_100.h5,0,vol1,100
   ```
   - `target`: 0=HGG, 1=LGG
   - `volume`: patient ID
   - `slice`: slice number

3. **multimodal**:
   - `False`: Save only T1CE as PNG (1 channel)
   - `True`: Save all 4 modalities as NPY (4 channels)

4. **min_tumor_ratio**: Filter criterion
   - Slices with <0.1% tumor pixels are skipped
   - Saves disk space, removes mostly-empty slices

**Setup code**:

```python
# Create output directories
os.makedirs(os.path.join(out_root, "images"), exist_ok=True)
os.makedirs(os.path.join(out_root, "masks"), exist_ok=True)

# Prepare output CSV files
labels_path = os.path.join(out_root, "labels.csv")
mapping_path = os.path.join(out_root, "mapping.csv")

# Read metadata CSV
slice_info = []
with open(meta_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        slice_info.append(row)
        if max_slices and len(slice_info) >= max_slices:
            break  # For testing, process only N slices
```

### 2.5 Main Function - Part 2: Processing Loop

```python
# Track statistics
processed = 0
skipped_no_tumor = 0
skipped_error = 0

# Process each slice
for info in tqdm(slice_info, desc="Processing slices"):
    # 1. Build path to HDF5 file
    h5_filename = os.path.basename(info['slice_path'])
    h5_path = os.path.join(h5_root, h5_filename)

    # 2. Check if file exists
    if not os.path.exists(h5_path):
        skipped += 1
        continue

    # 3. Extract metadata
    volume_id = info['volume']  # e.g., "vol1"
    slice_idx = info['slice']   # e.g., "100"
    label = int(info['target']) # 0 or 1
```

**Read HDF5 file**:

```python
    try:
        with h5py.File(h5_path, 'r') as f:
            # HDF5 structure:
            # - 'image': shape (H, W, 4) - 4 MRI modalities
            # - 'mask': shape (H, W, 3) - 3 tumor regions

            if 'image' in f and 'mask' in f:
                image = np.array(f['image'])  # (240, 240, 4)
                mask = np.array(f['mask'])    # (240, 240, 3)
            else:
                skipped_error += 1
                continue
```

**What's in the HDF5?**

`image` has 4 channels:
```
Channel 0: FLAIR  - shows edema
Channel 1: T1     - anatomical structure
Channel 2: T1CE   - enhancing tumor (BEST for detection)
Channel 3: T2     - overall tumor
```

`mask` has 3 channels:
```
Channel 0: Necrotic/non-enhancing tumor core
Channel 1: Peritumoral edema
Channel 2: Enhancing tumor
```

### 2.6 Main Function - Part 3: Modality Selection

```python
    # Select single modality OR keep all
    if multimodal:
        # Keep all 4 channels
        img_data = image  # (240, 240, 4)
    else:
        # Extract one channel
        img_data = image[:, :, modality_idx]  # (240, 240)
```

**Why T1CE (index 2) is default?**
- T1CE shows contrast enhancement
- Tumors have leaky blood vessels → contrast agent accumulates
- Brightest signal = active tumor
- Most informative for tumor detection

### 2.7 Main Function - Part 4: Normalization

```python
    # Normalize each modality independently
    if multimodal:
        # Normalize each of 4 channels
        for ch in range(4):
            img_data[:, :, ch] = _rescale01(img_data[:, :, ch])
    else:
        # Normalize single channel
        img_data = _rescale01(img_data)
```

**Why normalize each channel separately?**
- Different MRI sequences have different intensity ranges
- FLAIR might be [0, 500], T1CE might be [0, 2000]
- Normalizing to [0, 1] makes them comparable

### 2.8 Main Function - Part 5: Mask Processing

```python
    # Combine 3 tumor regions into 1 binary mask
    if len(mask.shape) == 3 and mask.shape[2] > 1:
        # Whole Tumor = union of all regions
        wt_mask = (mask > 0).any(axis=2).astype(np.float32)
    else:
        wt_mask = (mask > 0).astype(np.float32)
```

**Why combine masks?**
- Original: 3 separate regions (necrotic, edema, enhancing)
- Our task: binary (tumor vs background)
- Simpler, more stable training

**Example**:
```python
# Original mask (3 channels):
# Channel 0: [[0, 1, 1],    # Necrotic
#             [0, 1, 0]]
# Channel 1: [[1, 1, 0],    # Edema
#             [1, 1, 0]]
# Channel 2: [[0, 0, 1],    # Enhancing
#             [0, 0, 1]]

# Combined (1 channel):
# [[1, 1, 1],  # Any tumor region
#  [1, 1, 1]]
```

### 2.9 Main Function - Part 6: Quality Filtering

```python
    # Calculate tumor ratio
    total_pixels = wt_mask.shape[0] * wt_mask.shape[1]
    tumor_pixels = wt_mask.sum()
    tumor_ratio = tumor_pixels / total_pixels

    # Skip if too little tumor
    if tumor_ratio < min_tumor_ratio:
        skipped_no_tumor += 1
        continue  # Don't save this slice
```

**Why filter?**

Out of ~155 slices per patient:
- ~50 slices have NO tumor (top/bottom of brain)
- ~30 slices have very tiny tumor (<0.1%)
- ~75 slices have substantial tumor (>0.1%)

We keep only the 75 good slices to:
- ✅ Save disk space (75 instead of 155 per patient)
- ✅ Balance dataset (more tumor examples)
- ✅ Speed up training (fewer useless slices)

### 2.10 Main Function - Part 7: Resize with Padding

```python
    # Get current size
    h, w = img_data.shape[:2]  # (240, 240) typically

    # Pad to square
    s = max(h, w)
    pad_h = s - h
    pad_w = s - w

    # Symmetric padding
    pad_h_before = pad_h // 2
    pad_h_after = pad_h - pad_h_before
    pad_w_before = pad_w // 2
    pad_w_after = pad_w - pad_w_before
```

**Why pad first, then resize?**

Without padding:
```
(240, 200) → resize to (256, 256)
Result: Image is stretched! Brain looks squished.
```

With padding:
```
(240, 200) → pad to (240, 240) → resize to (256, 256)
Result: Correct aspect ratio, brain looks normal.
```

**Padding code**:

```python
    if multimodal:
        # Pad 4-channel image
        img_padded = np.pad(
            img_data,
            ((pad_h_before, pad_h_after),
             (pad_w_before, pad_w_after),
             (0, 0)),  # Don't pad channel dimension
            mode='constant',
            constant_values=0
        )
    else:
        # Pad 1-channel image
        img_padded = np.pad(
            img_data,
            ((pad_h_before, pad_h_after),
             (pad_w_before, pad_w_after)),
            mode='constant',
            constant_values=0
        )

    # Pad mask
    mask_padded = np.pad(
        wt_mask,
        ((pad_h_before, pad_h_after),
         (pad_w_before, pad_w_after)),
        mode='constant',
        constant_values=0
    )
```

**Resize using cv2 or PIL**:

```python
    import cv2

    # Resize image
    if multimodal:
        img_resized = cv2.resize(
            img_padded,
            (img_size, img_size),
            interpolation=cv2.INTER_LINEAR  # Smooth interpolation
        )
    else:
        img_resized = cv2.resize(
            img_padded,
            (img_size, img_size),
            interpolation=cv2.INTER_LINEAR
        )

    # Resize mask
    mask_resized = cv2.resize(
        mask_padded,
        (img_size, img_size),
        interpolation=cv2.INTER_NEAREST  # No smoothing for binary mask
    )
```

**Why INTER_LINEAR for image but INTER_NEAREST for mask?**

Image (continuous values):
```
[0.0, 0.5, 1.0] → resize → [0.0, 0.25, 0.5, 0.75, 1.0]
Smooth interpolation is good
```

Mask (binary 0/1):
```
[0, 0, 1, 1] → resize with LINEAR → [0, 0, 0.5, 0.75, 1.0, 1.0]
Bad! We get 0.5, 0.75 (not binary anymore)

[0, 0, 1, 1] → resize with NEAREST → [0, 0, 0, 1, 1, 1]
Good! Stays binary
```

### 2.11 Main Function - Part 8: Save Files

```python
    # Create slice ID
    slice_id = f"{volume_id}_slice{int(slice_idx):03d}"
    # Example: "vol1_slice100"

    # Save image
    if multimodal:
        # Save as NumPy array (.npy)
        img_path = os.path.join(out_root, "images", f"{slice_id}.npy")
        np.save(img_path, img_resized.astype(np.float32))
    else:
        # Save as PNG
        img_path = os.path.join(out_root, "images", f"{slice_id}.png")
        _save_png01(img_resized, img_path)

    # Save mask (always PNG)
    mask_path = os.path.join(out_root, "masks", f"{slice_id}.png")
    _save_png01(mask_resized, mask_path)

    processed += 1
```

**File formats**:

Single-modal:
```
images/vol1_slice100.png  (grayscale PNG, 65 KB)
masks/vol1_slice100.png   (binary PNG, 20 KB)
```

Multi-modal:
```
images/vol1_slice100.npy  (4-channel NPY, 1 MB)
masks/vol1_slice100.png   (binary PNG, 20 KB)
```

**Why NPY for multi-modal?**
- PNG only supports 1 or 3 channels
- NPY can store 4 channels with exact float values
- Fast to load with `np.load()`

### 2.12 Main Function - Part 9: Metadata Files

After processing all slices:

```python
# Write labels.csv
with open(labels_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case_id", "label"])
    for volume_id, label in case_labels.items():
        writer.writerow([volume_id, label])

# Write mapping.csv
with open(mapping_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["slice_id", "case_id"])
    for slice_id, case_id in slice_mapping:
        writer.writerow([slice_id, case_id])
```

**Generated files**:

`labels.csv`:
```csv
case_id,label
vol1,0
vol2,1
vol3,0
...
```

`mapping.csv`:
```csv
slice_id,case_id
vol1_slice100,vol1
vol1_slice101,vol1
vol2_slice050,vol2
...
```

### 2.13 Cross-Validation Splits

```python
def make_folds(proc_root: str, num_folds: int=5):
    """Create stratified K-fold splits"""

    # Read labels
    case_label = {}
    with open(os.path.join(proc_root, "labels.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_label[row["case_id"]] = int(row["label"])

    # Separate by class
    cases_hgg = [c for c, l in case_label.items() if l == 0]
    cases_lgg = [c for c, l in case_label.items() if l == 1]

    # Shuffle with fixed seed (reproducibility)
    random.seed(42)
    random.shuffle(cases_hgg)
    random.shuffle(cases_lgg)

    # Distribute across folds (round-robin)
    folds = [[] for _ in range(num_folds)]
    for i, case in enumerate(cases_hgg):
        folds[i % num_folds].append(case)
    for i, case in enumerate(cases_lgg):
        folds[i % num_folds].append(case)
```

**Why stratified?**

Not stratified:
```
Fold 0: All HGG (260 cases)
Fold 1: All LGG (109 cases)
Fold 2-4: Empty

Bad! Validation has only one class.
```

Stratified:
```
Fold 0: 52 HGG + 22 LGG (same ratio as full dataset)
Fold 1: 52 HGG + 22 LGG
Fold 2: 52 HGG + 22 LGG
...

Good! Each fold has both classes in correct proportion.
```

**Writing split files**:

```python
    # Read slice-to-case mapping
    case_slices = {}
    with open(os.path.join(proc_root, "mapping.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_slices.setdefault(row["case_id"], []).append(row["slice_id"])

    # Write splits
    for k in range(num_folds):
        val_cases = set(folds[k])
        train_cases = set(case_label.keys()) - val_cases

        # Collect slice IDs
        train_slices = []
        for case in train_cases:
            train_slices.extend(case_slices[case])

        val_slices = []
        for case in val_cases:
            val_slices.extend(case_slices[case])

        # Write files
        with open(os.path.join(proc_root, f"split_train_fold{k}.txt"), "w") as f:
            f.write("\n".join(train_slices))

        with open(os.path.join(proc_root, f"split_val_fold{k}.txt"), "w") as f:
            f.write("\n".join(val_slices))
```

**Result** (5 folds × 2 files):
```
split_train_fold0.txt  (18,102 slice IDs)
split_val_fold0.txt    (4,573 slice IDs)
split_train_fold1.txt
split_val_fold1.txt
...
split_train_fold4.txt
split_val_fold4.txt
```

---

## 3. File Analysis: brats2020_dataset.py

**Location**: `src/braintumnet/data/brats2020_dataset.py`
**Total Lines**: 99
**Purpose**: PyTorch Dataset class for loading preprocessed data

### 3.1 What is a PyTorch Dataset?

PyTorch needs to know:
1. **How many samples** do you have? → `__len__()`
2. **How to get sample i**? → `__getitem__(i)`

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __len__(self):
        return 1000  # We have 1000 samples

    def __getitem__(self, idx):
        # Load and return sample at index idx
        image = load_image(idx)
        label = load_label(idx)
        return image, label
```

Then PyTorch can:
```python
from torch.utils.data import DataLoader

dataset = MyDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for images, labels in loader:
    # images: (32, C, H, W) - batch of 32
    # labels: (32,) - batch of 32
    model(images)
```

### 3.2 Class Definition

```python
class SliceDataset(Dataset):
    """
    BraTS 2020 slice-based dataset.

    Loads preprocessed 2D slices with optional augmentation.
    Supports both single-modal (PNG) and multi-modal (NPY) formats.
    """

    def __init__(
        self,
        proc_root: str,       # "data/processed_full_multimodal"
        split_file: str,      # "split_train_fold0.txt"
        img_size: int=256,    # Image size (should match preprocessing)
        rotate_deg: int=30,   # Augmentation: rotation range
        hflip_p: float=0.5,   # Augmentation: horizontal flip probability
        vflip_p: float=0.5,   # Augmentation: vertical flip probability
        train: bool=True,     # Apply augmentation?
        in_channels: int=1    # 1 (single-modal) or 4 (multi-modal)
    ):
```

### 3.3 Initialization - Part 1: Load Slice IDs

```python
    self.proc_root = proc_root
    self.train = train
    self.img_size = img_size
    self.rotate_deg = rotate_deg
    self.hflip_p = hflip_p
    self.vflip_p = vflip_p
    self.in_channels = in_channels

    # Load slice IDs from split file
    with open(split_file, "r") as f:
        self.slice_ids = [x.strip() for x in f if x.strip()]

    # Example split_train_fold0.txt:
    # vol1_slice100
    # vol1_slice101
    # vol2_slice050
    # ...

    print(f"Loaded {len(self.slice_ids)} slices from {split_file}")
```

### 3.4 Initialization - Part 2: Load Labels

```python
    # Load case-level labels (HGG=0, LGG=1)
    self.case_label = {}
    labels_csv = os.path.join(proc_root, "labels.csv")

    if os.path.exists(labels_csv):
        with open(labels_csv) as f:
            next(f)  # Skip header
            for line in f:
                if "," in line:
                    case_id, label = line.strip().split(",")
                    self.case_label[case_id] = int(label)

    # Result: {"vol1": 0, "vol2": 1, "vol3": 0, ...}
```

### 3.5 Initialization - Part 3: Load Mapping

```python
    # Load slice-to-case mapping
    self.slice_case = {}
    mapping_csv = os.path.join(proc_root, "mapping.csv")

    if os.path.exists(mapping_csv):
        with open(mapping_csv) as f:
            next(f)  # Skip header
            for line in f:
                if "," in line:
                    slice_id, case_id = line.strip().split(",")
                    self.slice_case[slice_id] = case_id

    # Result: {"vol1_slice100": "vol1", "vol1_slice101": "vol1", ...}
```

**Why do we need mapping?**

Each slice needs a label, but labels are per-patient:
- Slice `vol1_slice100` → Patient `vol1` → Label `0` (HGG)
- Slice `vol1_slice101` → Patient `vol1` → Label `0` (HGG)
- Slice `vol2_slice050` → Patient `vol2` → Label `1` (LGG)

### 3.6 Getting Dataset Length

```python
def __len__(self):
    """Return number of slices in this split"""
    return len(self.slice_ids)
```

Simple! Just count how many slice IDs we loaded.

**Example**:
```python
dataset = SliceDataset("data/processed", "split_train_fold0.txt")
print(len(dataset))  # 18102 (for fold 0 training set)
```

### 3.7 Loading an Image

```python
def _load_image(self, slice_id: str):
    """
    Load image, auto-detecting format (NPY or PNG).

    Returns:
        PIL Image (single-modal) or NumPy array (multi-modal)
    """
    # Try multi-modal first
    npy_path = os.path.join(self.proc_root, "images", f"{slice_id}.npy")
    if os.path.exists(npy_path):
        # Multi-modal: Load 4-channel NPY
        img_array = np.load(npy_path)  # (256, 256, 4) float32
        return img_array

    # Fall back to single-modal PNG
    png_path = os.path.join(self.proc_root, "images", f"{slice_id}.png")
    if os.path.exists(png_path):
        # Single-modal: Load grayscale PNG
        return Image.open(png_path).convert("L")

    # File not found
    raise FileNotFoundError(f"Neither {npy_path} nor {png_path} found")
```

**Why auto-detect?**
- Same code works for single-modal and multi-modal
- Dataset automatically adapts based on what files exist

### 3.8 Loading a Mask

```python
def _load_mask(self, slice_id: str) -> Image.Image:
    """Load binary segmentation mask (always PNG)"""
    mask_path = os.path.join(self.proc_root, "masks", f"{slice_id}.png")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)

    return Image.open(mask_path).convert("L")  # Grayscale
```

**Masks are always PNG** (even for multi-modal) because:
- Mask is always single-channel (binary)
- PNG is smaller than NPY for binary data

### 3.9 Getting a Sample - Complete Function

```python
def __getitem__(self, idx):
    """
    Get one sample (image, mask, label).

    Args:
        idx: Index in [0, len(dataset)-1]

    Returns:
        Dictionary with:
        - image: (C, H, W) tensor, C=1 or 4
        - mask: (1, H, W) tensor, binary
        - label: scalar tensor, 0 or 1
        - slice_id: string
        - case_id: string
    """
    # 1. Get slice ID
    slice_id = self.slice_ids[idx]  # e.g., "vol1_slice100"

    # 2. Load image
    img = self._load_image(slice_id)

    # 3. Load mask
    msk = self._load_mask(slice_id)

    # 4. Process based on format
    if isinstance(img, np.ndarray):
        # Multi-modal path (NPY file)
        # img shape: (H, W, 4)

        # For multi-modal, augmentation was already applied during preprocessing
        # or we skip augmentation to save time
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        # Result: (4, H, W)

        # Process mask
        msk_array = np.asarray(msk).astype(np.float32)
        if msk_array.max() > 1.0:
            msk_array /= 255.0  # Normalize to [0, 1]
        msk_tensor = torch.from_numpy(msk_array > 0.5).float().unsqueeze(0)
        # Result: (1, H, W)

    else:
        # Single-modal path (PNG file)
        # img is PIL Image

        # Apply augmentation
        from .transforms import augment_pair
        img_tensor, msk_tensor = augment_pair(
            img, msk,
            self.img_size,
            self.rotate_deg,
            self.hflip_p,
            self.vflip_p,
            self.train  # Only augment if training
        )

    # 5. Get label
    case_id = self.slice_case.get(slice_id, slice_id.split("_")[0])
    label = self.case_label.get(case_id, 0)

    # 6. Return everything
    return {
        "image": img_tensor,    # (C, H, W) float32
        "mask": msk_tensor,     # (1, H, W) float32
        "label": torch.tensor(label, dtype=torch.long),  # scalar
        "slice_id": slice_id,   # string (for debugging)
        "case_id": case_id      # string (for debugging)
    }
```

**Example usage**:

```python
dataset = SliceDataset("data/processed", "split_train_fold0.txt", train=True)

sample = dataset[0]  # Get first sample

print(sample["image"].shape)   # torch.Size([4, 256, 256]) for multi-modal
print(sample["mask"].shape)    # torch.Size([1, 256, 256])
print(sample["label"])         # tensor(0) or tensor(1)
print(sample["slice_id"])      # "vol1_slice100"
```

---

## 4. File Analysis: transforms.py

**Location**: `src/braintumnet/data/transforms.py`
**Total Lines**: 42
**Purpose**: Data augmentation functions

### 4.1 Why Augmentation?

**Problem**: Limited data
- We have ~22,000 training slices
- Deep learning works best with millions of samples

**Solution**: Create variations
- Rotate image → same tumor, different orientation
- Flip image → mirror version
- Each epoch sees different variations → model learns better

**Result**: Effectively infinite training data!

### 4.2 Function: `resize_pad_to_square()`

```python
def resize_pad_to_square(
    img: Image.Image,   # PIL Image
    size: int,          # Target size (256)
    is_mask: bool=False # Use nearest-neighbor for masks?
) -> Image.Image:
    """
    Pad image to square, then resize.

    Why pad first?
    - Preserves aspect ratio
    - No stretching/distortion

    Args:
        img: PIL Image (any size)
        size: Output size
        is_mask: If True, use NEAREST interpolation (for binary masks)

    Returns:
        PIL Image (size × size)
    """
    w, h = img.size  # Get current size

    s = max(w, h)  # Target size before resize

    # Calculate padding needed
    pad_left = (s - w) // 2
    pad_top = (s - h) // 2
    pad_right = s - w - pad_left
    pad_bottom = s - h - pad_top

    # Pad with zeros (black)
    import torchvision.transforms.functional as TF
    img = TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

    # Now img is square (s × s)

    # Resize to final size
    if is_mask:
        # For masks: nearest-neighbor (keeps binary values)
        img = img.resize((size, size), Image.NEAREST)
    else:
        # For images: bilinear (smooth)
        img = img.resize((size, size), Image.BILINEAR)

    return img
```

**Example**:

```python
# Input: 200×240 image
img = Image.new('L', (200, 240))

# Pad to square
img_padded = resize_pad_to_square(img, 256, is_mask=False)

# Result: 256×256
# Padding added: left=20, right=20, top=0, bottom=0
```

### 4.3 Function: `to_tensor01()`

```python
def to_tensor01(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to PyTorch tensor in [0, 1] range.

    Args:
        img: PIL Image (grayscale)

    Returns:
        torch.Tensor of shape (1, H, W) in [0, 1]
    """
    # Convert to NumPy
    arr = np.asarray(img).astype(np.float32)

    # Normalize to [0, 1] if needed
    if arr.max() > 1.0:
        arr /= 255.0

    # Convert to tensor and add channel dimension
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
```

**Why [0, 1] range?**
- Neural networks work better with small values
- Easier to apply normalization later
- Standard practice in computer vision

### 4.4 Function: `augment_pair()` - Main Augmentation

```python
def augment_pair(
    img: Image.Image,       # Input image
    msk: Image.Image,       # Input mask
    img_size: int,          # Target size
    rotate_deg: int=30,     # Rotation range ±degrees
    hflip_p: float=0.5,     # Horizontal flip probability
    vflip_p: float=0.5,     # Vertical flip probability
    train: bool=True        # Apply augmentation?
):
    """
    Apply same augmentations to both image and mask.

    IMPORTANT: Augmentations must be IDENTICAL for image and mask!
    If we rotate image 15°, we MUST rotate mask 15° too.

    Returns:
        img_tensor: (1, H, W) in [0, 1]
        msk_tensor: (1, H, W) in [0, 1]
    """
    import torchvision.transforms.functional as TF

    # Step 1: Resize (always)
    img = resize_pad_to_square(img, img_size, is_mask=False)
    msk = resize_pad_to_square(msk, img_size, is_mask=True)

    if train:
        # Step 2: Random rotation
        angle = random.uniform(-rotate_deg, rotate_deg)
        # Example: angle = 15.3°

        img = TF.rotate(img, angle)
        msk = TF.rotate(msk, angle)  # SAME angle!

        # Step 3: Random horizontal flip
        if random.random() < hflip_p:
            img = TF.hflip(img)
            msk = TF.hflip(msk)

        # Step 4: Random vertical flip
        if random.random() < vflip_p:
            img = TF.vflip(img)
            msk = TF.vflip(msk)

    # Step 5: Convert to tensors
    img_tensor = to_tensor01(img)

    # For mask, ensure binary
    msk_arr = np.asarray(msk).astype(np.float32)
    if msk_arr.max() > 1.0:
        msk_arr /= 255.0
    msk_tensor = torch.from_numpy(msk_arr > 0.5).float().unsqueeze(0)

    return img_tensor, msk_tensor
```

**Augmentation examples**:

Original:
```
Image: Brain with tumor on right
Mask: Tumor region on right
```

After rotation (+20°):
```
Image: Brain rotated 20° clockwise
Mask: Tumor rotated 20° clockwise (matches!)
```

After horizontal flip:
```
Image: Brain mirrored (tumor now on left)
Mask: Tumor also on left (matches!)
```

**Why these augmentations?**

✅ **Rotation**: Tumor can appear at any angle
✅ **Horizontal flip**: Left/right hemisphere symmetry
✅ **Vertical flip**: Sometimes useful for axial slices

❌ **Color jitter**: No! MRI intensity has medical meaning
❌ **Crop**: No! We need full brain context

---

## 5. Complete Data Flow

### 5.1 From Raw to Batch (Step-by-Step)

```
1. RAW DATA (one time setup)
   File: BraTS2020_Training_001_slice_100.h5
   Content: image (240×240×4), mask (240×240×3)
   Size: ~2 MB

        ↓ [prepare_brats2020_h5.py]

2. PREPROCESSED (saved to disk)
   Files: vol1_slice100.npy, vol1_slice100.png
   Content: image (256×256×4), mask (256×256)
   Size: 1 MB + 65 KB

        ↓ [SliceDataset.__init__]

3. DATASET READY
   slice_ids = ["vol1_slice100", "vol1_slice101", ...]
   Length: 18,102 slices (for fold 0 training)

        ↓ [SliceDataset.__getitem__(idx)]

4. LOAD ONE SAMPLE
   img = np.load("vol1_slice100.npy")  # (256, 256, 4)
   msk = Image.open("vol1_slice100.png")  # (256, 256)

        ↓ [augment_pair() if training]

5. AUGMENTED SAMPLE
   img: rotated 15°, flipped horizontally
   msk: rotated 15°, flipped horizontally (same transforms!)

        ↓ [to_tensor()]

6. TENSOR SAMPLE
   img_tensor: (4, 256, 256) float32
   msk_tensor: (1, 256, 256) float32
   label: tensor(0)

        ↓ [DataLoader collate]

7. BATCH
   images: (12, 4, 256, 256)  # batch_size=12
   masks: (12, 1, 256, 256)
   labels: (12,)

        ↓ [model.forward()]

8. PREDICTIONS
   seg_logits: (12, 1, 256, 256)
   cls_logits: (12, 2)
```

### 5.2 Memory Flow

```
Disk → RAM → GPU

Disk:
- All 22,677 slices stored
- Total: ~23 GB

RAM (during training):
- DataLoader loads batch_size × num_workers slices
- Example: 12 × 4 = 48 slices in RAM
- ~48 MB (negligible)

GPU:
- One batch: 12 slices
- Images: 12 × 4 × 256 × 256 × 4 bytes = 12 MB
- Masks: 12 × 1 × 256 × 256 × 4 bytes = 3 MB
- Model: ~14M parameters × 4 bytes = 56 MB
- Activations: ~500 MB (during forward pass)
- Gradients: ~56 MB
- Optimizer state: ~112 MB
- Total: ~740 MB per batch
```

**Why GPU memory is important?**

If batch_size too large:
```
batch_size=12: ~740 MB ✅
batch_size=32: ~1.9 GB ✅
batch_size=64: ~3.8 GB ⚠️ (might overflow on 4GB GPU)
```

---

## 6. Modification Guide

### 6.1 Add New Augmentation: Gaussian Blur

**File**: `transforms.py`

```python
# Add import
import cv2

# In augment_pair(), after rotation:
if train and random.random() < 0.3:  # 30% chance
    # Convert to numpy
    img_np = np.array(img)

    # Apply Gaussian blur
    img_np = cv2.GaussianBlur(img_np, (5, 5), 1.0)

    # Convert back
    img = Image.fromarray(img_np.astype(np.uint8))
```

### 6.2 Add New Augmentation: Elastic Deformation

```python
# Add to transforms.py
from scipy.ndimage import gaussian_filter, map_coordinates

def elastic_deform(image, mask, alpha=30, sigma=5):
    """Apply elastic deformation"""
    shape = image.shape

    # Random displacement fields
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    # Coordinate meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # Apply deformation
    image_deformed = map_coordinates(image, indices, order=1).reshape(shape)
    mask_deformed = map_coordinates(mask, indices, order=0).reshape(shape)

    return image_deformed, mask_deformed

# Use in augment_pair():
if train and random.random() < 0.2:  # 20% chance
    img_np = np.array(img)
    msk_np = np.array(msk)
    img_np, msk_np = elastic_deform(img_np, msk_np)
    img = Image.fromarray(img_np.astype(np.uint8))
    msk = Image.fromarray(msk_np.astype(np.uint8))
```

### 6.3 Change Normalization: Z-Score Instead of [0,1]

**File**: `prepare_brats2020_h5.py`

Replace `_rescale01()`:

```python
def _zscore_normalize(arr: np.ndarray) -> np.ndarray:
    """Z-score normalization: (x - mean) / std"""
    arr = arr.astype(np.float32)
    nz = arr > 0

    if nz.sum() > 0:
        mean = arr[nz].mean()
        std = arr[nz].std()
        arr = (arr - mean) / (std + 1e-6)

    return arr
```

**Then update Dataset** to NOT do [0,1] clipping:

```python
# In brats2020_dataset.py, remove this line:
# arr /= 255.0

# Keep values as is (z-scored)
```

### 6.4 Use Different Modality

**Preprocess only FLAIR** (index 0):

```bash
python scripts/prepare_brats2020_h5.py \
  --modality_idx 0 \
  --out data/processed_flair
```

**Config**:
```yaml
data:
  proc_root: "data/processed_flair"

model:
  in_channels: 1  # Single modality
```

### 6.5 Filter Slices Differently

**Keep ALL slices** (including empty ones):

```python
# In prepare_brats2020_h5.py, comment out:
# if tumor_ratio < min_tumor_ratio:
#     skipped_no_tumor += 1
#     continue

# Now all slices are saved
```

**Or via command line**:
```bash
python scripts/prepare_brats2020_h5.py --min_tumor_ratio 0.0
```

---

## 7. Debugging Tips

### 7.1 Test Preprocessing

```bash
# Process only 100 slices for testing
python scripts/prepare_brats2020_h5.py \
  --max_slices 100 \
  --out data/processed_test
```

### 7.2 Visualize Dataset

```python
from braintumnet.data.brats2020_dataset import SliceDataset
import matplotlib.pyplot as plt

# Load dataset
ds = SliceDataset(
    "data/processed_full_multimodal",
    "data/processed_full_multimodal/split_train_fold0.txt",
    train=True
)

# Get sample
sample = ds[0]

# Visualize
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# Show all 4 modalities
for i in range(4):
    axes[i].imshow(sample['image'][i], cmap='gray')
    axes[i].set_title(f"Channel {i}")
    axes[i].axis('off')

# Show mask
axes[4].imshow(sample['mask'][0], cmap='hot')
axes[4].set_title("Mask")
axes[4].axis('off')

plt.suptitle(f"Slice: {sample['slice_id']}, Label: {sample['label']}")
plt.show()
```

### 7.3 Check Augmentation

```python
# Load same sample multiple times (should be different due to augmentation)
for i in range(4):
    sample = ds[0]  # Same index
    plt.subplot(2, 2, i+1)
    plt.imshow(sample['image'][0], cmap='gray')
    plt.title(f"Variation {i+1}")
plt.show()
```

### 7.4 Verify Data Loading Speed

```python
from torch.utils.data import DataLoader
import time

ds = SliceDataset("data/processed_full_multimodal", "split_train_fold0.txt")
loader = DataLoader(ds, batch_size=12, num_workers=4, shuffle=True)

start = time.time()
for i, batch in enumerate(loader):
    if i >= 10:
        break  # Time first 10 batches
end = time.time()

print(f"Average time per batch: {(end-start)/10:.3f} seconds")
# Should be <1 second for good performance
```

---

## Summary

### What We Learned

✅ **Stage 1**: Preprocessing (`prepare_brats2020_h5.py`)
   - Read HDF5 → Normalize → Resize → Save PNG/NPY
   - Create metadata (labels.csv, mapping.csv)
   - Generate cross-validation splits

✅ **Stage 2**: Dataset (`brats2020_dataset.py`)
   - PyTorch Dataset class
   - Lazy loading (load on demand)
   - Auto-detect single/multi-modal

✅ **Stage 3**: Augmentation (`transforms.py`)
   - Rotation, flipping
   - Applied on-the-fly during training
   - Same transforms for image and mask

### Key Takeaways

1. **Preprocess once, use forever**: Disk space vs computation time tradeoff
2. **Lazy loading**: Don't load all 23GB into RAM at once
3. **On-the-fly augmentation**: Infinite data variations
4. **Stratified splits**: Ensure balanced train/val sets
5. **Multi-modal flexibility**: Same code works for 1 or 4 channels

### Next Steps

Now you understand how data flows from raw files to PyTorch batches!

👉 **Next**: [[03_MODEL_ARCHITECTURE|Part 3 - Model Architecture]]

Learn how the neural network processes these batches!

---

[[TECHNICAL_REPORT_INDEX|← Back to Index]] | [[03_MODEL_ARCHITECTURE|Next: Model Architecture →]]
