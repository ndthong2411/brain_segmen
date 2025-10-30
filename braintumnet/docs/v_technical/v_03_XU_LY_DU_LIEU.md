# Phần 3: Xử Lý Dữ Liệu Chi Tiết

> **📊 Pipeline Xử Lý Dữ Liệu BraTS 2020 - Từ H5 đến PNG Multi-Class**
>
> Tài liệu này giải thích chi tiết quá trình preprocessing, augmentation, và data loading.

---

## Mục Lục

1. [Tổng Quan Pipeline](#1-tổng-quan-pipeline)
2. [BraTS 2020 Dataset](#2-brats-2020-dataset)
3. [Preprocessing H5 to PNG](#3-preprocessing-h5-to-png)
4. [Multi-Class Label Processing](#4-multi-class-label-processing)
5. [Data Augmentation](#5-data-augmentation)
6. [Dataset và DataLoader](#6-dataset-và-dataloader)
7. [Fold Splitting](#7-fold-splitting)
8. [Data Statistics](#8-data-statistics)

---

## 1. Tổng Quan Pipeline

### Luồng Xử Lý Hoàn Chỉnh

```
BraTS 2020 Raw Data
    │
    ├── BraTS20_Training_001/
    │   ├── *_flair.nii.gz  (FLAIR modality)
    │   ├── *_t1.nii.gz     (T1 modality)
    │   ├── *_t1ce.nii.gz   (T1CE modality)
    │   ├── *_t2.nii.gz     (T2 modality)
    │   └── *_seg.nii.gz    (Segmentation labels)
    │
    ↓ Step 1: Convert NIfTI → H5
    │
BraTS H5 Files (57,195 files)
    │ Each file: {flair, t1, t1ce, t2, seg} (256×256 slices)
    │
    ↓ Step 2: Preprocess H5 → PNG Multi-Class
    │
Processed PNG Dataset
    │
    ├── flair/
    │   └── BraTS20_Training_001_0050.png (grayscale, 256×256)
    ├── t1/
    ├── t1ce/
    ├── t2/
    ├── seg/
    │   └── BraTS20_Training_001_0050.png (3-class: 0,1,2)
    │
    ├── all_slices.csv       (metadata)
    ├── labels.csv           (HGG/LGG labels)
    ├── mapping.csv          (patient→grade mapping)
    │
    └── Folds (5-fold CV):
        ├── train_fold0.csv
        ├── val_fold0.csv
        ├── train_fold1.csv
        ├── val_fold1.csv
        ...
    │
    ↓ Step 3: Data Loading
    │
PyTorch Dataset
    │ __getitem__: Load 4 modalities + seg
    │ Apply augmentations
    │ Normalize
    │
    ↓ Step 4: DataLoader
    │
Training Batches
    │ Shape: (B, 4, 256, 256) inputs
    │        (B, 256, 256) multi-class labels
```

---

## 2. BraTS 2020 Dataset

### Dataset Structure

**Tổng quan**:
- **369 patients** (cases)
- **259 HGG** (High-Grade Glioma - ác tính cao)
- **110 LGG** (Low-Grade Glioma - ác tính thấp)
- Mỗi patient: ~155 slices
- **Tổng: 57,195 slices** sau preprocessing

### MRI Modalities

**4 loại MRI sequences**:

1. **FLAIR (Fluid-Attenuated Inversion Recovery)**
   - Highlight edema (phù nề)
   - Nổi bật vùng tín hiệu cao
   - Best cho detecting whole tumor

2. **T1 (T1-weighted)**
   - Anatomical structure
   - Contrast thấp cho tumor
   - Baseline reference

3. **T1CE (T1-weighted with Contrast Enhancement)**
   - Contrast agent injection
   - Highlight active tumor core
   - Best cho detecting enhancing tumor

4. **T2 (T2-weighted)**
   - Highlight fluid/edema
   - High signal cho cysts
   - Complement FLAIR

### Segmentation Labels (BraTS Annotation)

**Original BraTS labels** (4 classes):
```
Label 0: Background
Label 1: Necrotic/Non-enhancing tumor core (NCR/NET)
Label 2: Edema (ED)
Label 4: Enhancing tumor (ET)
```

**BraTS Tumor Regions** (composite):
```
Whole Tumor (WT) = Label 1 + Label 2 + Label 4
Tumor Core (TC)  = Label 1 + Label 4
Enhancing Tumor (ET) = Label 4
```

**Our 3-class mapping**:
```
Class 0: Background (BraTS label 0)
Class 1: Tumor Core (BraTS labels 1 + 4)  # NCR/NET + ET
Class 2: Edema (BraTS label 2)
```

**Tại sao 3-class thay vì 4-class?**
- NCR/NET (label 1) và ET (label 4) đều là tumor core
- Kết hợp chúng → robust hơn
- Easier to learn (less class imbalance)
- Clinically meaningful: {Background, Tumor, Edema}

---

## 3. Preprocessing H5 to PNG

### File Code

**File**: `scripts/preprocess_h5_to_multiclass.py` (240 dòng)

### Main Function

```python
def preprocess_h5_to_multiclass(
    h5_dir: str,
    out_dir: str,
    img_size: int = 256,
    num_folds: int = 5,
    seed: int = 42
):
    """
    Convert BraTS H5 files to PNG multi-class format
    
    Args:
        h5_dir: Directory chứa H5 files
        out_dir: Output directory
        img_size: Target image size (256)
        num_folds: Number of CV folds (5)
        seed: Random seed
    """
    # Create output directories
    modalities = ['flair', 't1', 't1ce', 't2', 'seg']
    for mod in modalities:
        os.makedirs(os.path.join(out_dir, mod), exist_ok=True)
    
    # Process all H5 files
    all_records = []
    for h5_file in tqdm(glob.glob(os.path.join(h5_dir, '*.h5'))):
        records = process_single_h5(h5_file, out_dir, img_size)
        all_records.extend(records)
    
    # Save metadata
    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(out_dir, 'all_slices.csv'), index=False)
    
    # Create fold splits
    create_fold_splits(df, out_dir, num_folds, seed)
    
    print(f"✓ Preprocessed {len(all_records)} slices")
    print(f"✓ Created {num_folds}-fold splits")
```

### Processing Single H5 File

```python
def process_single_h5(h5_path, out_dir, img_size):
    """
    Xử lý 1 file H5 (1 patient)
    
    Input: BraTS20_Training_001.h5
    Output: Multiple PNG files (1 per slice)
    
    Returns: List of records với metadata
    """
    records = []
    filename = os.path.basename(h5_path).replace('.h5', '')
    
    with h5py.File(h5_path, 'r') as hf:
        # Load all modalities
        flair = hf['flair'][:]  # (H, W, num_slices)
        t1 = hf['t1'][:]
        t1ce = hf['t1ce'][:]
        t2 = hf['t2'][:]
        seg = hf['seg'][:]      # Original BraTS labels
        
        num_slices = flair.shape[2]
        
        for slice_idx in range(num_slices):
            # Extract slice
            flair_slice = flair[:, :, slice_idx]
            t1_slice = t1[:, :, slice_idx]
            t1ce_slice = t1ce[:, :, slice_idx]
            t2_slice = t2[:, :, slice_idx]
            seg_slice = seg[:, :, slice_idx]
            
            # Skip empty slices (no tumor, all background)
            if seg_slice.max() == 0:
                continue
            
            # Resize to target size
            flair_resized = cv2.resize(flair_slice, (img_size, img_size))
            t1_resized = cv2.resize(t1_slice, (img_size, img_size))
            t1ce_resized = cv2.resize(t1ce_slice, (img_size, img_size))
            t2_resized = cv2.resize(t2_slice, (img_size, img_size))
            seg_resized = cv2.resize(
                seg_slice, (img_size, img_size), 
                interpolation=cv2.INTER_NEAREST  # Nearest cho labels!
            )
            
            # Convert segmentation to 3-class
            seg_multiclass = convert_to_multiclass(seg_resized)
            
            # Normalize modalities to [0, 255]
            flair_norm = normalize_to_uint8(flair_resized)
            t1_norm = normalize_to_uint8(t1_resized)
            t1ce_norm = normalize_to_uint8(t1ce_resized)
            t2_norm = normalize_to_uint8(t2_resized)
            
            # Save as PNG
            slice_name = f"{filename}_{slice_idx:04d}.png"
            cv2.imwrite(
                os.path.join(out_dir, 'flair', slice_name), flair_norm
            )
            cv2.imwrite(
                os.path.join(out_dir, 't1', slice_name), t1_norm
            )
            cv2.imwrite(
                os.path.join(out_dir, 't1ce', slice_name), t1ce_norm
            )
            cv2.imwrite(
                os.path.join(out_dir, 't2', slice_name), t2_norm
            )
            cv2.imwrite(
                os.path.join(out_dir, 'seg', slice_name), seg_multiclass
            )
            
            # Record metadata
            records.append({
                'patient_id': filename,
                'slice_idx': slice_idx,
                'slice_name': slice_name,
                'has_tumor': 1
            })
    
    return records
```

### Normalization to uint8

```python
def normalize_to_uint8(img):
    """
    Normalize MRI slice to [0, 255] uint8
    
    Steps:
    1. Clip outliers (percentile-based)
    2. Min-max normalize to [0, 1]
    3. Scale to [0, 255]
    """
    # Clip outliers
    p1, p99 = np.percentile(img, [1, 99])
    img_clipped = np.clip(img, p1, p99)
    
    # Min-max normalize
    img_min = img_clipped.min()
    img_max = img_clipped.max()
    
    if img_max - img_min < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    
    img_norm = (img_clipped - img_min) / (img_max - img_min)
    
    # Scale to [0, 255]
    img_uint8 = (img_norm * 255).astype(np.uint8)
    
    return img_uint8
```

**Tại sao clip percentiles?**
```
MRI intensities có outliers do noise/artifacts:

Original histogram:
|                      *    (outlier)
|                  ******** (main distribution)
|                *********
|              ***********
|____________***___________*___
0          500    1000   5000

Sau clip p1-p99:
|              
|                  ******** 
|                *********
|              ***********
|____________***___________
0          500    1000

→ Better contrast trong main distribution
→ Không bị ảnh hưởng bởi extreme values
```

---

## 4. Multi-Class Label Processing

### Label Conversion

```python
def convert_to_multiclass(seg):
    """
    Convert BraTS labels (0,1,2,4) → 3-class (0,1,2)
    
    Mapping:
    BraTS 0 → Class 0 (Background)
    BraTS 1 → Class 1 (Tumor Core - NCR/NET)
    BraTS 2 → Class 2 (Edema)
    BraTS 4 → Class 1 (Tumor Core - ET)
    
    Args:
        seg: (H, W) array với values in {0, 1, 2, 4}
    
    Returns:
        seg_multiclass: (H, W) array với values in {0, 1, 2}
    """
    seg_multiclass = np.zeros_like(seg, dtype=np.uint8)
    
    # Background (0 → 0)
    seg_multiclass[seg == 0] = 0
    
    # Tumor Core: NCR/NET (1) và ET (4) → class 1
    seg_multiclass[seg == 1] = 1
    seg_multiclass[seg == 4] = 1
    
    # Edema (2 → 2)
    seg_multiclass[seg == 2] = 2
    
    return seg_multiclass
```

**Ví dụ conversion**:
```python
# BraTS original
seg_brats = np.array([
    [0, 0, 2, 2],
    [0, 1, 4, 2],
    [0, 1, 1, 0],
    [0, 0, 0, 0]
])

# After conversion
seg_multiclass = convert_to_multiclass(seg_brats)
# array([
#     [0, 0, 2, 2],  # Edema → class 2
#     [0, 1, 1, 2],  # NCR/NET → 1, ET → 1
#     [0, 1, 1, 0],  
#     [0, 0, 0, 0]
# ])
```

### Label Statistics

```python
def compute_label_statistics(out_dir):
    """
    Tính thống kê label distribution
    """
    seg_dir = os.path.join(out_dir, 'seg')
    
    class_counts = {0: 0, 1: 0, 2: 0}
    total_pixels = 0
    
    for seg_file in tqdm(os.listdir(seg_dir)):
        seg = cv2.imread(
            os.path.join(seg_dir, seg_file), 
            cv2.IMREAD_GRAYSCALE
        )
        
        for cls in [0, 1, 2]:
            class_counts[cls] += (seg == cls).sum()
        
        total_pixels += seg.size
    
    # Compute percentages
    for cls in [0, 1, 2]:
        pct = 100 * class_counts[cls] / total_pixels
        print(f"Class {cls}: {pct:.2f}%")
    
    # Compute class weights (inverse frequency)
    weights = []
    for cls in [0, 1, 2]:
        freq = class_counts[cls] / total_pixels
        weight = 1.0 / (freq + 1e-6)
        weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum() * len(weights)
    
    print(f"Class weights: {weights}")
    return weights
```

**Example output**:
```
Class 0 (Background): 87.35%
Class 1 (Tumor Core): 5.12%
Class 2 (Edema):      7.53%

Class weights: [0.344, 5.865, 3.981]
→ Background weight nhỏ (nhiều pixels)
→ Tumor Core weight cao (ít pixels)
→ Edema weight trung bình
```

---

## 5. Data Augmentation

### File Code

**File**: `src/braintumnet/data/transforms.py` (108 dòng)

### Augmentation Pipeline

```python
class BraTSAugmentation:
    """
    Data augmentation cho BraTS multi-modal MRI
    
    Augmentations:
    1. Random horizontal flip (p=0.5)
    2. Random vertical flip (p=0.5)
    3. Random rotation (±15 degrees)
    4. Random scaling (0.9-1.1)
    5. Elastic deformation
    6. Intensity shift/scale (per modality)
    7. Gaussian noise
    """
    def __init__(self, mode='train'):
        self.mode = mode
        
        if mode == 'train':
            self.aug = A.Compose([
                # Geometric augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomScale(scale_limit=0.1, p=0.3),
                
                # Elastic deformation (medical imaging specific)
                A.ElasticTransform(
                    alpha=50, sigma=5, alpha_affine=5, p=0.3
                ),
                
                # Intensity augmentations (applied to image only)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
            ], 
            additional_targets={
                't1': 'image',
                't1ce': 'image',
                't2': 'image'
            })
        else:
            self.aug = None
    
    def __call__(self, flair, t1, t1ce, t2, seg):
        """
        Apply augmentation to multi-modal input + segmentation
        
        Args:
            flair, t1, t1ce, t2: (H, W) grayscale images
            seg: (H, W) segmentation mask
        
        Returns:
            Augmented (flair, t1, t1ce, t2, seg)
        """
        if self.mode != 'train' or self.aug is None:
            return flair, t1, t1ce, t2, seg
        
        # Apply augmentation
        augmented = self.aug(
            image=flair,      # Primary image
            mask=seg,         # Segmentation
            t1=t1,            # Additional modality
            t1ce=t1ce,
            t2=t2
        )
        
        return (
            augmented['image'],
            augmented['t1'],
            augmented['t1ce'],
            augmented['t2'],
            augmented['mask']
        )
```

### Ví Dụ Augmentation

**Original**:
```
FLAIR:        T1:           Seg:
┌──────┐    ┌──────┐    ┌──────┐
│  ███ │    │  ██  │    │  012 │
│  ███ │    │  ██  │    │  012 │
│  ███ │    │  ██  │    │  012 │
└──────┘    └──────┘    └──────┘
```

**Sau HorizontalFlip**:
```
FLAIR:        T1:           Seg:
┌──────┐    ┌──────┐    ┌──────┐
│ ███  │    │  ██  │    │ 210  │
│ ███  │    │  ██  │    │ 210  │
│ ███  │    │  ██  │    │ 210  │
└──────┘    └──────┘    └──────┘
```

**Sau Rotation (+10°)**:
```
FLAIR:        T1:           Seg:
┌──────┐    ┌──────┐    ┌──────┐
│ ███  │    │ ██   │    │ 012  │
│  ███ │    │  ██  │    │  012 │
│   ███│    │   ██ │    │   012│
└──────┘    └──────┘    └──────┘
```

### Tại Sao Các Augmentations Này?

**Geometric augmentations**:
- **Flip**: Brain anatomy có symmetry
- **Rotation**: ±15° simulate different slice angles
- **Scaling**: Simulate different zoom levels

**Elastic deformation**:
- Medical imaging specific
- Simulate natural tissue deformations
- Robust to inter-patient variations

**Intensity augmentations**:
- MRI intensities vary giữa scanners
- Brightness/Contrast: Scanner variations
- Gaussian noise: Scanner noise artifacts

**KHÔNG dùng augmentations sau** (tại sao?):
- ❌ **CutOut/CoarseDropout**: Có thể xóa tumor
- ❌ **Color jitter**: MRI là grayscale
- ❌ **Strong rotations (>30°)**: Không realistic cho brain slices
- ❌ **GridDistortion**: Quá aggressive cho medical

---

## 6. Dataset và DataLoader

### File Code

**File**: `src/braintumnet/data/dataset.py` (152 dòng)

### BraTSDataset Class

```python
class BraTSDataset(Dataset):
    """
    PyTorch Dataset cho BraTS multi-class segmentation
    
    Returns:
        image: (4, 256, 256) float32 - normalized multi-modal input
        mask: (256, 256) long - multi-class labels {0, 1, 2}
        label: int - HGG (0) or LGG (1)
    """
    def __init__(
        self, 
        data_dir: str,
        fold: int,
        mode: str = 'train',
        img_size: int = 256,
        transform=None
    ):
        """
        Args:
            data_dir: Root directory (processed_multiclass/)
            fold: Fold number (0-4)
            mode: 'train' or 'val'
            img_size: Image size (256)
            transform: Augmentation transform
        """
        self.data_dir = data_dir
        self.mode = mode
        self.img_size = img_size
        self.transform = transform
        
        # Load fold CSV
        csv_file = f"{mode}_fold{fold}.csv"
        csv_path = os.path.join(data_dir, csv_file)
        self.df = pd.read_csv(csv_path)
        
        # Load labels (HGG/LGG)
        labels_csv = os.path.join(data_dir, 'labels.csv')
        self.labels_df = pd.read_csv(labels_csv)
        
        print(f"Loaded {len(self.df)} {mode} samples for fold {fold}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Load 1 sample: multi-modal images + segmentation + classification label
        """
        row = self.df.iloc[idx]
        slice_name = row['slice_name']
        patient_id = row['patient_id']
        
        # Load 4 modalities
        flair = self._load_image('flair', slice_name)
        t1 = self._load_image('t1', slice_name)
        t1ce = self._load_image('t1ce', slice_name)
        t2 = self._load_image('t2', slice_name)
        
        # Load segmentation
        seg = self._load_seg(slice_name)
        
        # Apply augmentation
        if self.transform:
            flair, t1, t1ce, t2, seg = self.transform(
                flair, t1, t1ce, t2, seg
            )
        
        # Stack modalities
        image = np.stack([flair, t1, t1ce, t2], axis=0)  # (4, H, W)
        
        # Normalize each modality
        image = self._normalize(image)
        
        # Get classification label
        label = self._get_classification_label(patient_id)
        
        # Convert to tensors
        image = torch.from_numpy(image).float()  # (4, 256, 256)
        seg = torch.from_numpy(seg).long()       # (256, 256)
        label = torch.tensor(label).long()       # scalar
        
        return image, seg, label
    
    def _load_image(self, modality, slice_name):
        """Load single modality image"""
        img_path = os.path.join(self.data_dir, modality, slice_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        return img  # (H, W) uint8
    
    def _load_seg(self, slice_name):
        """Load segmentation mask"""
        seg_path = os.path.join(self.data_dir, 'seg', slice_name)
        seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        
        # Verify values in {0, 1, 2}
        assert set(np.unique(seg)).issubset({0, 1, 2}), \
            f"Invalid seg values: {np.unique(seg)}"
        
        return seg  # (H, W) uint8
    
    def _normalize(self, image):
        """
        Normalize multi-modal input
        
        Per-modality Z-score normalization:
        x_norm = (x - mean) / (std + eps)
        """
        normalized = np.zeros_like(image, dtype=np.float32)
        
        for i in range(4):  # 4 modalities
            mod = image[i]
            mean = mod.mean()
            std = mod.std()
            normalized[i] = (mod - mean) / (std + 1e-6)
        
        return normalized
    
    def _get_classification_label(self, patient_id):
        """
        Get HGG/LGG label for patient
        
        Returns:
            0: HGG (High-Grade Glioma)
            1: LGG (Low-Grade Glioma)
        """
        row = self.labels_df[
            self.labels_df['patient_id'] == patient_id
        ]
        
        if len(row) == 0:
            raise ValueError(f"Patient {patient_id} not found in labels")
        
        grade = row.iloc[0]['grade']
        
        if grade == 'HGG':
            return 0
        elif grade == 'LGG':
            return 1
        else:
            raise ValueError(f"Invalid grade: {grade}")
```

### DataLoader Creation

```python
def create_dataloaders(
    data_dir: str,
    fold: int,
    batch_size: int = 12,
    num_workers: int = 4,
    img_size: int = 256
):
    """
    Create train và val dataloaders
    
    Returns:
        train_loader, val_loader
    """
    # Augmentation
    train_transform = BraTSAugmentation(mode='train')
    val_transform = None
    
    # Datasets
    train_dataset = BraTSDataset(
        data_dir=data_dir,
        fold=fold,
        mode='train',
        img_size=img_size,
        transform=train_transform
    )
    
    val_dataset = BraTSDataset(
        data_dir=data_dir,
        fold=fold,
        mode='val',
        img_size=img_size,
        transform=val_transform
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Ensure consistent batch size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
```

**Ví dụ sử dụng**:
```python
train_loader, val_loader = create_dataloaders(
    data_dir='data/processed_multiclass',
    fold=0,
    batch_size=12,
    num_workers=4
)

# Iterate
for images, masks, labels in train_loader:
    print(images.shape)  # (12, 4, 256, 256)
    print(masks.shape)   # (12, 256, 256)
    print(labels.shape)  # (12,)
    break
```

---

## 7. Fold Splitting

### Stratified K-Fold

```python
def create_fold_splits(df, out_dir, num_folds=5, seed=42):
    """
    Create stratified K-fold splits
    
    Stratification:
    - By patient_id (not slice-level)
    - By HGG/LGG grade
    
    Ensures:
    - No patient leakage giữa train/val
    - Balanced HGG/LGG distribution trong mỗi fold
    """
    # Load labels
    labels_csv = os.path.join(out_dir, 'labels.csv')
    labels_df = pd.read_csv(labels_csv)
    
    # Get unique patients
    patients = df['patient_id'].unique()
    
    # Get labels for patients
    patient_labels = []
    for pid in patients:
        grade = labels_df[labels_df['patient_id'] == pid].iloc[0]['grade']
        patient_labels.append(1 if grade == 'HGG' else 0)
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(patients, patient_labels)):
        # Get patient IDs
        train_patients = patients[train_idx]
        val_patients = patients[val_idx]
        
        # Get slices for patients
        train_df = df[df['patient_id'].isin(train_patients)]
        val_df = df[df['patient_id'].isin(val_patients)]
        
        # Save CSVs
        train_df.to_csv(
            os.path.join(out_dir, f'train_fold{fold}.csv'), 
            index=False
        )
        val_df.to_csv(
            os.path.join(out_dir, f'val_fold{fold}.csv'), 
            index=False
        )
        
        print(f"Fold {fold}:")
        print(f"  Train: {len(train_patients)} patients, {len(train_df)} slices")
        print(f"  Val:   {len(val_patients)} patients, {len(val_df)} slices")
```

**Example output**:
```
Fold 0:
  Train: 295 patients, 45,756 slices
  Val:   74 patients, 11,439 slices

Fold 1:
  Train: 295 patients, 45,812 slices
  Val:   74 patients, 11,383 slices

...
```

**Tại sao stratified by patient?**
```
❌ Wrong (slice-level split):
Train: Patient A slices 1-100
Val:   Patient A slices 101-155
→ Data leakage! Model thấy same patient trong train

✓ Correct (patient-level split):
Train: Patients A, B, C, ...
Val:   Patients X, Y, Z, ...
→ No leakage, realistic evaluation
```

---

## 8. Data Statistics

### Dataset Summary

```python
def print_dataset_statistics(data_dir):
    """In thống kê tổng quan"""
    all_csv = os.path.join(data_dir, 'all_slices.csv')
    labels_csv = os.path.join(data_dir, 'labels.csv')
    
    df = pd.read_csv(all_csv)
    labels_df = pd.read_csv(labels_csv)
    
    print("="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    # Total counts
    print(f"Total patients: {df['patient_id'].nunique()}")
    print(f"Total slices:   {len(df)}")
    
    # HGG/LGG distribution
    hgg_count = (labels_df['grade'] == 'HGG').sum()
    lgg_count = (labels_df['grade'] == 'LGG').sum()
    print(f"\nGrade distribution:")
    print(f"  HGG: {hgg_count} ({100*hgg_count/len(labels_df):.1f}%)")
    print(f"  LGG: {lgg_count} ({100*lgg_count/len(labels_df):.1f}%)")
    
    # Slices per patient
    slices_per_patient = df.groupby('patient_id').size()
    print(f"\nSlices per patient:")
    print(f"  Mean: {slices_per_patient.mean():.1f}")
    print(f"  Std:  {slices_per_patient.std():.1f}")
    print(f"  Min:  {slices_per_patient.min()}")
    print(f"  Max:  {slices_per_patient.max()}")
    
    # Class distribution (pixel-level)
    compute_label_statistics(data_dir)
```

**Example output**:
```
==================================================
DATASET STATISTICS
==================================================
Total patients: 369
Total slices:   57,195

Grade distribution:
  HGG: 259 (70.2%)
  LGG: 110 (29.8%)

Slices per patient:
  Mean: 155.0
  Std:  8.5
  Min:  135
  Max:  170

Class distribution (pixel-level):
  Class 0 (Background): 87.35%
  Class 1 (Tumor Core): 5.12%
  Class 2 (Edema):      7.53%

Class weights: [0.344, 5.865, 3.981]
```

---

**[← Phần 2B: Kiến Trúc (Phần 2)](v_02_KIEN_TRUC_MODEL_PHAN2.md)** | **[Phần 4: Loss Functions →](v_04_LOSS_FUNCTIONS.md)**
