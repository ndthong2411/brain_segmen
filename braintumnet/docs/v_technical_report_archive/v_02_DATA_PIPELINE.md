# Part 2: Data Pipeline Deep Dive

> **📊 Hướng dẫn đầy đủ từ raw HDF5 files đến PyTorch tensors sẵn sàng cho training**
>
> Part này giải thích TỪNG BƯỚC của data processing với giải thích code từng dòng.

---

## Mục lục

1. [Tổng quan Pipeline](#1-tổng-quan-pipeline)
2. [Phân tích File: prepare_brats2020_h5.py](#2-phân-tích-file-prepare_brats2020_h5py)
3. [Phân tích File: brats2020_dataset.py](#3-phân-tích-file-brats2020_datasetpy)
4. [Phân tích File: transforms.py](#4-phân-tích-file-transformspy)
5. [Complete Data Flow](#5-complete-data-flow)
6. [Modification Guide](#6-modification-guide)
7. [Debugging Tips](#7-debugging-tips)

---

## 1. Tổng quan Pipeline

### Bức tranh Toàn cảnh

Data pipeline có 3 giai đoạn chính:

```
GIAI ĐOẠN 1: Preprocessing (MỘT LẦN)
Raw BraTS HDF5 → Preprocessed PNG/NPY
[prepare_brats2020_h5.py]

GIAI ĐOẠN 2: Dataset Loading (MỖI EPOCH)
Preprocessed files → PyTorch Dataset
[brats2020_dataset.py]

GIAI ĐOẠN 3: Augmentation (MỖI BATCH)
Original data → Augmented variants
[transforms.py]
```

### Tại sao Thiết kế Này?

**Giai đoạn 1** - Preprocess một lần, sử dụng mãi mãi:
- ✅ HDF5 chậm khi đọc lặp lại
- ✅ Convert sang PNG/NPY format nhanh một lần
- ✅ Normalize và resize một lần
- ✅ Tiết kiệm disk space (chỉ giữ slices tốt)

**Giai đoạn 2** - Lazy loading:
- ✅ Không load tất cả 23GB vào RAM
- ✅ Chỉ load những gì cần cho batch hiện tại
- ✅ PyTorch DataLoader xử lý parallelization

**Giai đoạn 3** - On-the-fly augmentation:
- ✅ Tạo vô hạn biến thể (rotation, flip)
- ✅ Không lưu augmented images (lãng phí space)
- ✅ Augmentation khác nhau mỗi epoch

---

## 2. Phân tích File: prepare_brats2020_h5.py

**Location**: `scripts/prepare_brats2020_h5.py`
**Tổng số Dòng**: 416 dòng
**Mục đích**: Convert raw BraTS2020 HDF5 files sang preprocessed format

### 2.1 Imports và Setup

```python
import os, argparse, csv, h5py
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
```

**Mỗi import làm gì**:
- `h5py`: Đọc HDF5 files (BraTS format)
- `PIL.Image`: Lưu PNG images
- `tqdm`: Progress bar
- `numpy`: Array operations
- `csv`: Đọc/viết CSV metadata

### 2.2 Helper Function: `_rescale01()`

**Mục đích**: Normalize image về [0, 1] range

```python
def _rescale01(arr: np.ndarray) -> np.ndarray:
    """
    Rescale array về [0, 1] range, bỏ qua background (zeros).

    Args:
        arr: Input array (vd 240x240 MRI slice)

    Returns:
        Normalized array trong [0, 1]
    """
    arr = arr.astype(np.float32)

    # Chỉ xét non-zero pixels (brain tissue, không phải background)
    nz = arr > 0

    if nz.sum() > 0:
        # Lấy min/max từ brain tissue only
        a = arr[nz]
        lo, hi = a.min(), a.max()
    else:
        # Tất cả zeros (không nên xảy ra, nhưng xử lý nó)
        lo, hi = arr.min(), arr.max()

    # Tránh division by zero
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)

    # Normalize
    out = (arr - lo) / (hi - lo)

    # Xử lý NaN/Inf (không nên xảy ra, nhưng an toàn)
    out[~np.isfinite(out)] = 0

    return out
```

**Tại sao bỏ qua background?**
- MRI background luôn là 0 (không có signal)
- Chúng ta muốn normalize brain tissue intensity
- Ví dụ: Brain tissue ranges [100, 1000] → normalize về [0, 1]
- Background giữ nguyên 0

**Ví dụ**:
```python
# Input: MRI slice với background
arr = np.array([[0, 0, 0],
                [0, 100, 200],
                [0, 150, 250]])

# Output sau _rescale01():
# [[0.0, 0.0, 0.0],
#  [0.0, 0.0, 0.667],
#  [0.0, 0.333, 1.0]]
# Brain tissue [100-250] map tới [0-1], background giữ nguyên 0
```

### 2.3 Helper Function: `_save_png01()`

```python
def _save_png01(x: np.ndarray, path: str):
    """
    Lưu normalized [0, 1] array dưới dạng PNG [0, 255].

    Args:
        x: Array trong [0, 1] range
        path: Output PNG path
    """
    # Scale tới [0, 255]
    x = (x * 255.0).clip(0, 255).astype(np.uint8)

    # Lưu bằng PIL
    Image.fromarray(x).save(path)
```

**Tại sao clip?**
- Đôi khi floating point errors gây ra giá trị hơi ra ngoài [0, 1]
- `clip(0, 255)` đảm bảo safe range

### 2.4 Main Function: `process_h5_brats2020()` - Phần 1: Setup

```python
def process_h5_brats2020(
    h5_root: str,              # "data/raw" - nơi HDF5 files nằm
    meta_csv: str,             # "data/raw/meta_data.csv" - labels
    out_root: str,             # "data/processed_full_multimodal" - output
    img_size: int=256,         # Resize về kích thước này
    modality_idx: int=2,       # 0=FLAIR, 1=T1, 2=T1CE, 3=T2
    max_slices: int=None,      # Giới hạn để test (None = tất cả)
    multimodal: bool=False,    # Lưu cả 4 channels?
    min_tumor_ratio: float=0.001  # Skip slices với tumors nhỏ
):
```

**Parameters giải thích**:

1. **h5_root**: Nơi raw HDF5 files của bạn
   ```
   data/raw/
   ├── BraTS2020_Training_001_slice_100.h5
   ├── BraTS2020_Training_001_slice_101.h5
   └── ...
   ```

2. **meta_csv**: CSV với metadata
   ```csv
   slice_path,target,volume,slice
   BraTS2020_Training_001_slice_100.h5,0,vol1,100
   ```
   - `target`: 0=HGG, 1=LGG
   - `volume`: patient ID
   - `slice`: slice number

3. **multimodal**:
   - `False`: Chỉ lưu T1CE dưới dạng PNG (1 channel)
   - `True`: Lưu cả 4 modalities dưới dạng NPY (4 channels)

4. **min_tumor_ratio**: Tiêu chí filter
   - Slices với <0.1% tumor pixels bị skip
   - Tiết kiệm disk space, loại bỏ slices gần như rỗng

**Setup code**:

```python
# Tạo output directories
os.makedirs(os.path.join(out_root, "images"), exist_ok=True)
os.makedirs(os.path.join(out_root, "masks"), exist_ok=True)

# Chuẩn bị output CSV files
labels_path = os.path.join(out_root, "labels.csv")
mapping_path = os.path.join(out_root, "mapping.csv")

# Đọc metadata CSV
slice_info = []
with open(meta_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        slice_info.append(row)
        if max_slices and len(slice_info) >= max_slices:
            break  # Để test, chỉ process N slices
```

### 2.5 Main Function - Phần 2: Processing Loop

```python
# Theo dõi thống kê
processed = 0
skipped_no_tumor = 0
skipped_error = 0

# Xử lý từng slice
for info in tqdm(slice_info, desc="Processing slices"):
    # 1. Build path tới HDF5 file
    h5_filename = os.path.basename(info['slice_path'])
    h5_path = os.path.join(h5_root, h5_filename)

    # 2. Kiểm tra file tồn tại
    if not os.path.exists(h5_path):
        skipped += 1
        continue

    # 3. Trích xuất metadata
    volume_id = info['volume']  # vd "vol1"
    slice_idx = info['slice']   # vd "100"
    label = int(info['target']) # 0 hoặc 1
```

**Đọc HDF5 file**:

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

**HDF5 chứa gì?**

`image` có 4 channels:
```
Channel 0: FLAIR  - hiển thị edema
Channel 1: T1     - cấu trúc giải phẫu
Channel 2: T1CE   - enhancing tumor (TỐT NHẤT cho detection)
Channel 3: T2     - overall tumor
```

`mask` có 3 channels:
```
Channel 0: Necrotic/non-enhancing tumor core
Channel 1: Peritumoral edema
Channel 2: Enhancing tumor
```

### 2.6 Main Function - Phần 3: Modality Selection

```python
    # Chọn single modality HOẶC giữ tất cả
    if multimodal:
        # Giữ cả 4 channels
        img_data = image  # (240, 240, 4)
    else:
        # Trích xuất một channel
        img_data = image[:, :, modality_idx]  # (240, 240)
```

**Tại sao T1CE (index 2) là mặc định?**
- T1CE hiển thị contrast enhancement
- Tumors có mạch máu rò rỉ → chất tương phản tích tụ
- Signal sáng nhất = tumor hoạt động
- Thông tin nhất cho tumor detection

### 2.7 Main Function - Phần 4: Normalization

```python
    # Normalize mỗi modality độc lập
    if multimodal:
        # Normalize mỗi channel trong 4 channels
        for ch in range(4):
            img_data[:, :, ch] = _rescale01(img_data[:, :, ch])
    else:
        # Normalize single channel
        img_data = _rescale01(img_data)
```

**Tại sao normalize mỗi channel riêng biệt?**
- Các MRI sequences khác nhau có intensity ranges khác nhau
- FLAIR có thể là [0, 500], T1CE có thể là [0, 2000]
- Normalizing về [0, 1] làm chúng có thể so sánh

### 2.8 Main Function - Phần 5: Mask Processing

```python
    # Kết hợp 3 tumor regions thành 1 binary mask
    if len(mask.shape) == 3 and mask.shape[2] > 1:
        # Whole Tumor = union của tất cả regions
        wt_mask = (mask > 0).any(axis=2).astype(np.float32)
    else:
        wt_mask = (mask > 0).astype(np.float32)
```

**Tại sao kết hợp masks?**
- Gốc: 3 regions riêng biệt (necrotic, edema, enhancing)
- Task của chúng ta: binary (tumor vs background)
- Đơn giản hơn, training ổn định hơn

**Ví dụ**:
```python
# Original mask (3 channels):
# Channel 0: [[0, 1, 1],    # Necrotic
#             [0, 1, 0]]
# Channel 1: [[1, 1, 0],    # Edema
#             [1, 1, 0]]
# Channel 2: [[0, 0, 1],    # Enhancing
#             [0, 0, 1]]

# Combined (1 channel):
# [[1, 1, 1],  # Bất kỳ tumor region nào
#  [1, 1, 1]]
```

### 2.9 Main Function - Phần 6: Quality Filtering

```python
    # Tính tumor ratio
    total_pixels = wt_mask.shape[0] * wt_mask.shape[1]
    tumor_pixels = wt_mask.sum()
    tumor_ratio = tumor_pixels / total_pixels

    # Skip nếu quá ít tumor
    if tumor_ratio < min_tumor_ratio:
        skipped_no_tumor += 1
        continue  # Không lưu slice này
```

**Tại sao filter?**

Trong ~155 slices mỗi patient:
- ~50 slices KHÔNG có tumor (đầu/cuối não)
- ~30 slices có tumor rất nhỏ (<0.1%)
- ~75 slices có tumor đáng kể (>0.1%)

Chúng ta chỉ giữ 75 slices tốt để:
- ✅ Tiết kiệm disk space (75 thay vì 155 mỗi patient)
- ✅ Cân bằng dataset (nhiều tumor examples hơn)
- ✅ Tăng tốc training (ít slices vô dụng hơn)

### 2.10 Main Function - Phần 7: Resize với Padding

```python
    # Lấy kích thước hiện tại
    h, w = img_data.shape[:2]  # (240, 240) thường

    # Pad về square
    s = max(h, w)
    pad_h = s - h
    pad_w = s - w

    # Symmetric padding
    pad_h_before = pad_h // 2
    pad_h_after = pad_h - pad_h_before
    pad_w_before = pad_w // 2
    pad_w_after = pad_w - pad_w_before
```

**Tại sao pad trước, rồi mới resize?**

Không padding:
```
(240, 200) → resize tới (256, 256)
Kết quả: Image bị kéo dãn! Não trông bị méo.
```

Có padding:
```
(240, 200) → pad tới (240, 240) → resize tới (256, 256)
Kết quả: Aspect ratio đúng, não trông bình thường.
```

**Padding code**:

```python
    if multimodal:
        # Pad 4-channel image
        img_padded = np.pad(
            img_data,
            ((pad_h_before, pad_h_after),
             (pad_w_before, pad_w_after),
             (0, 0)),  # Không pad channel dimension
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

**Resize sử dụng cv2 hoặc PIL**:

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
        interpolation=cv2.INTER_NEAREST  # Không smoothing cho binary mask
    )
```

**Tại sao INTER_LINEAR cho image nhưng INTER_NEAREST cho mask?**

Image (continuous values):
```
[0.0, 0.5, 1.0] → resize → [0.0, 0.25, 0.5, 0.75, 1.0]
Smooth interpolation là tốt
```

Mask (binary 0/1):
```
[0, 0, 1, 1] → resize với LINEAR → [0, 0, 0.5, 0.75, 1.0, 1.0]
Tệ! Chúng ta có 0.5, 0.75 (không còn binary)

[0, 0, 1, 1] → resize với NEAREST → [0, 0, 0, 1, 1, 1]
Tốt! Vẫn binary
```

### 2.11 Main Function - Phần 8: Save Files

```python
    # Tạo slice ID
    slice_id = f"{volume_id}_slice{int(slice_idx):03d}"
    # Ví dụ: "vol1_slice100"

    # Lưu image
    if multimodal:
        # Lưu dưới dạng NumPy array (.npy)
        img_path = os.path.join(out_root, "images", f"{slice_id}.npy")
        np.save(img_path, img_resized.astype(np.float32))
    else:
        # Lưu dưới dạng PNG
        img_path = os.path.join(out_root, "images", f"{slice_id}.png")
        _save_png01(img_resized, img_path)

    # Lưu mask (luôn PNG)
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

**Tại sao NPY cho multi-modal?**
- PNG chỉ hỗ trợ 1 hoặc 3 channels
- NPY có thể lưu 4 channels với exact float values
- Nhanh để load với `np.load()`

### 2.12 Main Function - Phần 9: Metadata Files

Sau khi xử lý tất cả slices:

```python
# Viết labels.csv
with open(labels_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case_id", "label"])
    for volume_id, label in case_labels.items():
        writer.writerow([volume_id, label])

# Viết mapping.csv
with open(mapping_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["slice_id", "case_id"])
    for slice_id, case_id in slice_mapping:
        writer.writerow([slice_id, case_id])
```

**Files được tạo**:

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
    """Tạo stratified K-fold splits"""

    # Đọc labels
    case_label = {}
    with open(os.path.join(proc_root, "labels.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_label[row["case_id"]] = int(row["label"])

    # Phân tách theo class
    cases_hgg = [c for c, l in case_label.items() if l == 0]
    cases_lgg = [c for c, l in case_label.items() if l == 1]

    # Shuffle với fixed seed (reproducibility)
    random.seed(42)
    random.shuffle(cases_hgg)
    random.shuffle(cases_lgg)

    # Phân phối qua các folds (round-robin)
    folds = [[] for _ in range(num_folds)]
    for i, case in enumerate(cases_hgg):
        folds[i % num_folds].append(case)
    for i, case in enumerate(cases_lgg):
        folds[i % num_folds].append(case)
```

**Tại sao stratified?**

Không stratified:
```
Fold 0: Tất cả HGG (260 cases)
Fold 1: Tất cả LGG (109 cases)
Fold 2-4: Rỗng

Tệ! Validation chỉ có một class.
```

Stratified:
```
Fold 0: 52 HGG + 22 LGG (tỷ lệ giống full dataset)
Fold 1: 52 HGG + 22 LGG
Fold 2: 52 HGG + 22 LGG
...

Tốt! Mỗi fold có cả hai classes với tỷ lệ đúng.
```

**Viết split files**:

```python
    # Đọc slice-to-case mapping
    case_slices = {}
    with open(os.path.join(proc_root, "mapping.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_slices.setdefault(row["case_id"], []).append(row["slice_id"])

    # Viết splits
    for k in range(num_folds):
        val_cases = set(folds[k])
        train_cases = set(case_label.keys()) - val_cases

        # Thu thập slice IDs
        train_slices = []
        for case in train_cases:
            train_slices.extend(case_slices[case])

        val_slices = []
        for case in val_cases:
            val_slices.extend(case_slices[case])

        # Viết files
        with open(os.path.join(proc_root, f"split_train_fold{k}.txt"), "w") as f:
            f.write("\n".join(train_slices))

        with open(os.path.join(proc_root, f"split_val_fold{k}.txt"), "w") as f:
            f.write("\n".join(val_slices))
```

**Kết quả** (5 folds × 2 files):
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

## 3. Phân tích File: brats2020_dataset.py

**Location**: `src/braintumnet/data/brats2020_dataset.py`
**Tổng số Dòng**: 99
**Mục đích**: PyTorch Dataset class để load preprocessed data

### 3.1 PyTorch Dataset là gì?

PyTorch cần biết:
1. **Có bao nhiêu samples**? → `__len__()`
2. **Cách lấy sample i**? → `__getitem__(i)`

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __len__(self):
        return 1000  # Chúng ta có 1000 samples

    def __getitem__(self, idx):
        # Load và return sample tại index idx
        image = load_image(idx)
        label = load_label(idx)
        return image, label
```

Sau đó PyTorch có thể:
```python
from torch.utils.data import DataLoader

dataset = MyDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for images, labels in loader:
    # images: (32, C, H, W) - batch của 32
    # labels: (32,) - batch của 32
    model(images)
```

### 3.2 Class Definition

```python
class SliceDataset(Dataset):
    """
    BraTS 2020 slice-based dataset.

    Load preprocessed 2D slices với optional augmentation.
    Hỗ trợ cả single-modal (PNG) và multi-modal (NPY) formats.
    """

    def __init__(
        self,
        proc_root: str,       # "data/processed_full_multimodal"
        split_file: str,      # "split_train_fold0.txt"
        img_size: int=256,    # Image size (nên khớp preprocessing)
        rotate_deg: int=30,   # Augmentation: rotation range
        hflip_p: float=0.5,   # Augmentation: horizontal flip probability
        vflip_p: float=0.5,   # Augmentation: vertical flip probability
        train: bool=True,     # Áp dụng augmentation?
        in_channels: int=1    # 1 (single-modal) hoặc 4 (multi-modal)
    ):
```

### 3.3 Initialization - Phần 1: Load Slice IDs

```python
    self.proc_root = proc_root
    self.train = train
    self.img_size = img_size
    self.rotate_deg = rotate_deg
    self.hflip_p = hflip_p
    self.vflip_p = vflip_p
    self.in_channels = in_channels

    # Load slice IDs từ split file
    with open(split_file, "r") as f:
        self.slice_ids = [x.strip() for x in f if x.strip()]

    # Ví dụ split_train_fold0.txt:
    # vol1_slice100
    # vol1_slice101
    # vol2_slice050
    # ...

    print(f"Loaded {len(self.slice_ids)} slices from {split_file}")
```

### 3.4 Initialization - Phần 2: Load Labels

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

    # Kết quả: {"vol1": 0, "vol2": 1, "vol3": 0, ...}
```

### 3.5 Initialization - Phần 3: Load Mapping

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

    # Kết quả: {"vol1_slice100": "vol1", "vol1_slice101": "vol1", ...}
```

**Tại sao cần mapping?**

Mỗi slice cần một label, nhưng labels là per-patient:
- Slice `vol1_slice100` → Patient `vol1` → Label `0` (HGG)
- Slice `vol1_slice101` → Patient `vol1` → Label `0` (HGG)
- Slice `vol2_slice050` → Patient `vol2` → Label `1` (LGG)

### 3.6 Getting Dataset Length

```python
def __len__(self):
    """Return số slices trong split này"""
    return len(self.slice_ids)
```

Đơn giản! Chỉ đếm có bao nhiêu slice IDs chúng ta đã load.

**Ví dụ**:
```python
dataset = SliceDataset("data/processed", "split_train_fold0.txt")
print(len(dataset))  # 18102 (cho fold 0 training set)
```

### 3.7 Loading một Image

```python
def _load_image(self, slice_id: str):
    """
    Load image, tự động detect format (NPY hoặc PNG).

    Returns:
        PIL Image (single-modal) hoặc NumPy array (multi-modal)
    """
    # Thử multi-modal trước
    npy_path = os.path.join(self.proc_root, "images", f"{slice_id}.npy")
    if os.path.exists(npy_path):
        # Multi-modal: Load 4-channel NPY
        img_array = np.load(npy_path)  # (256, 256, 4) float32
        return img_array

    # Fall back sang single-modal PNG
    png_path = os.path.join(self.proc_root, "images", f"{slice_id}.png")
    if os.path.exists(png_path):
        # Single-modal: Load grayscale PNG
        return Image.open(png_path).convert("L")

    # File không tìm thấy
    raise FileNotFoundError(f"Neither {npy_path} nor {png_path} found")
```

**Tại sao auto-detect?**
- Cùng code hoạt động cho single-modal và multi-modal
- Dataset tự động thích ứng dựa trên files tồn tại

### 3.8 Loading một Mask

```python
def _load_mask(self, slice_id: str) -> Image.Image:
    """Load binary segmentation mask (luôn PNG)"""
    mask_path = os.path.join(self.proc_root, "masks", f"{slice_id}.png")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)

    return Image.open(mask_path).convert("L")  # Grayscale
```

**Masks luôn là PNG** (ngay cả multi-modal) vì:
- Mask luôn single-channel (binary)
- PNG nhỏ hơn NPY cho binary data

### 3.9 Getting một Sample - Complete Function

```python
def __getitem__(self, idx):
    """
    Lấy một sample (image, mask, label).

    Args:
        idx: Index trong [0, len(dataset)-1]

    Returns:
        Dictionary với:
        - image: (C, H, W) tensor, C=1 hoặc 4
        - mask: (1, H, W) tensor, binary
        - label: scalar tensor, 0 hoặc 1
        - slice_id: string
        - case_id: string
    """
    # 1. Lấy slice ID
    slice_id = self.slice_ids[idx]  # vd "vol1_slice100"

    # 2. Load image
    img = self._load_image(slice_id)

    # 3. Load mask
    msk = self._load_mask(slice_id)

    # 4. Xử lý dựa trên format
    if isinstance(img, np.ndarray):
        # Multi-modal path (NPY file)
        # img shape: (H, W, 4)

        # Với multi-modal, augmentation đã được áp dụng trong preprocessing
        # hoặc chúng ta skip augmentation để tiết kiệm thời gian
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        # Kết quả: (4, H, W)

        # Xử lý mask
        msk_array = np.asarray(msk).astype(np.float32)
        if msk_array.max() > 1.0:
            msk_array /= 255.0  # Normalize về [0, 1]
        msk_tensor = torch.from_numpy(msk_array > 0.5).float().unsqueeze(0)
        # Kết quả: (1, H, W)

    else:
        # Single-modal path (PNG file)
        # img là PIL Image

        # Áp dụng augmentation
        from .transforms import augment_pair
        img_tensor, msk_tensor = augment_pair(
            img, msk,
            self.img_size,
            self.rotate_deg,
            self.hflip_p,
            self.vflip_p,
            self.train  # Chỉ augment nếu training
        )

    # 5. Lấy label
    case_id = self.slice_case.get(slice_id, slice_id.split("_")[0])
    label = self.case_label.get(case_id, 0)

    # 6. Return mọi thứ
    return {
        "image": img_tensor,    # (C, H, W) float32
        "mask": msk_tensor,     # (1, H, W) float32
        "label": torch.tensor(label, dtype=torch.long),  # scalar
        "slice_id": slice_id,   # string (cho debugging)
        "case_id": case_id      # string (cho debugging)
    }
```

**Ví dụ sử dụng**:

```python
dataset = SliceDataset("data/processed", "split_train_fold0.txt", train=True)

sample = dataset[0]  # Lấy sample đầu tiên

print(sample["image"].shape)   # torch.Size([4, 256, 256]) cho multi-modal
print(sample["mask"].shape)    # torch.Size([1, 256, 256])
print(sample["label"])         # tensor(0) hoặc tensor(1)
print(sample["slice_id"])      # "vol1_slice100"
```

---

## 4. Phân tích File: transforms.py

**Location**: `src/braintumnet/data/transforms.py`
**Tổng số Dòng**: 42
**Mục đích**: Data augmentation functions

### 4.1 Tại sao Augmentation?

**Vấn đề**: Dữ liệu hạn chế
- Chúng ta có ~22,000 training slices
- Deep learning hoạt động tốt nhất với hàng triệu samples

**Giải pháp**: Tạo biến thể
- Rotate image → cùng tumor, hướng khác
- Flip image → phiên bản mirror
- Mỗi epoch thấy biến thể khác → model học tốt hơn

**Kết quả**: Effectively vô hạn training data!

### 4.2 Function: `resize_pad_to_square()`

```python
def resize_pad_to_square(
    img: Image.Image,   # PIL Image
    size: int,          # Target size (256)
    is_mask: bool=False # Sử dụng nearest-neighbor cho masks?
) -> Image.Image:
    """
    Pad image về square, sau đó resize.

    Tại sao pad trước?
    - Giữ aspect ratio
    - Không stretching/distortion

    Args:
        img: PIL Image (bất kỳ size nào)
        size: Output size
        is_mask: Nếu True, sử dụng NEAREST interpolation (cho binary masks)

    Returns:
        PIL Image (size × size)
    """
    w, h = img.size  # Lấy kích thước hiện tại

    s = max(w, h)  # Target size trước resize

    # Tính padding cần thiết
    pad_left = (s - w) // 2
    pad_top = (s - h) // 2
    pad_right = s - w - pad_left
    pad_bottom = s - h - pad_top

    # Pad với zeros (black)
    import torchvision.transforms.functional as TF
    img = TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

    # Bây giờ img là square (s × s)

    # Resize về final size
    if is_mask:
        # Cho masks: nearest-neighbor (giữ binary values)
        img = img.resize((size, size), Image.NEAREST)
    else:
        # Cho images: bilinear (smooth)
        img = img.resize((size, size), Image.BILINEAR)

    return img
```

**Ví dụ**:

```python
# Input: 200×240 image
img = Image.new('L', (200, 240))

# Pad về square
img_padded = resize_pad_to_square(img, 256, is_mask=False)

# Kết quả: 256×256
# Padding added: left=20, right=20, top=0, bottom=0
```

### 4.3 Function: `to_tensor01()`

```python
def to_tensor01(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image sang PyTorch tensor trong [0, 1] range.

    Args:
        img: PIL Image (grayscale)

    Returns:
        torch.Tensor của shape (1, H, W) trong [0, 1]
    """
    # Convert sang NumPy
    arr = np.asarray(img).astype(np.float32)

    # Normalize về [0, 1] nếu cần
    if arr.max() > 1.0:
        arr /= 255.0

    # Convert sang tensor và thêm channel dimension
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
```

**Tại sao [0, 1] range?**
- Neural networks hoạt động tốt hơn với giá trị nhỏ
- Dễ áp dụng normalization sau
- Thực hành chuẩn trong computer vision

### 4.4 Function: `augment_pair()` - Main Augmentation

```python
def augment_pair(
    img: Image.Image,       # Input image
    msk: Image.Image,       # Input mask
    img_size: int,          # Target size
    rotate_deg: int=30,     # Rotation range ±degrees
    hflip_p: float=0.5,     # Horizontal flip probability
    vflip_p: float=0.5,     # Vertical flip probability
    train: bool=True        # Áp dụng augmentation?
):
    """
    Áp dụng cùng augmentations cho cả image và mask.

    QUAN TRỌNG: Augmentations phải GIỐNG NHAU cho image và mask!
    Nếu chúng ta rotate image 15°, PHẢI rotate mask 15° cũng.

    Returns:
        img_tensor: (1, H, W) trong [0, 1]
        msk_tensor: (1, H, W) trong [0, 1]
    """
    import torchvision.transforms.functional as TF

    # Bước 1: Resize (luôn)
    img = resize_pad_to_square(img, img_size, is_mask=False)
    msk = resize_pad_to_square(msk, img_size, is_mask=True)

    if train:
        # Bước 2: Random rotation
        angle = random.uniform(-rotate_deg, rotate_deg)
        # Ví dụ: angle = 15.3°

        img = TF.rotate(img, angle)
        msk = TF.rotate(msk, angle)  # CÙNG angle!

        # Bước 3: Random horizontal flip
        if random.random() < hflip_p:
            img = TF.hflip(img)
            msk = TF.hflip(msk)

        # Bước 4: Random vertical flip
        if random.random() < vflip_p:
            img = TF.vflip(img)
            msk = TF.vflip(msk)

    # Bước 5: Convert sang tensors
    img_tensor = to_tensor01(img)

    # Cho mask, đảm bảo binary
    msk_arr = np.asarray(msk).astype(np.float32)
    if msk_arr.max() > 1.0:
        msk_arr /= 255.0
    msk_tensor = torch.from_numpy(msk_arr > 0.5).float().unsqueeze(0)

    return img_tensor, msk_tensor
```

**Ví dụ augmentation**:

Gốc:
```
Image: Não với tumor bên phải
Mask: Tumor region bên phải
```

Sau rotation (+20°):
```
Image: Não xoay 20° theo chiều kim đồng hồ
Mask: Tumor xoay 20° theo chiều kim đồng hồ (khớp!)
```

Sau horizontal flip:
```
Image: Não mirror (tumor bây giờ bên trái)
Mask: Tumor cũng bên trái (khớp!)
```

**Tại sao các augmentations này?**

✅ **Rotation**: Tumor có thể xuất hiện ở bất kỳ góc nào
✅ **Horizontal flip**: Tính đối xứng bán cầu trái/phải
✅ **Vertical flip**: Đôi khi hữu ích cho axial slices

❌ **Color jitter**: Không! MRI intensity có ý nghĩa y tế
❌ **Crop**: Không! Chúng ta cần full brain context

---

## 5. Complete Data Flow

### 5.1 Từ Raw đến Batch (Từng Bước)

```
1. RAW DATA (một lần setup)
   File: BraTS2020_Training_001_slice_100.h5
   Content: image (240×240×4), mask (240×240×3)
   Size: ~2 MB

        ↓ [prepare_brats2020_h5.py]

2. PREPROCESSED (lưu vào disk)
   Files: vol1_slice100.npy, vol1_slice100.png
   Content: image (256×256×4), mask (256×256)
   Size: 1 MB + 65 KB

        ↓ [SliceDataset.__init__]

3. DATASET READY
   slice_ids = ["vol1_slice100", "vol1_slice101", ...]
   Length: 18,102 slices (cho fold 0 training)

        ↓ [SliceDataset.__getitem__(idx)]

4. LOAD MỘT SAMPLE
   img = np.load("vol1_slice100.npy")  # (256, 256, 4)
   msk = Image.open("vol1_slice100.png")  # (256, 256)

        ↓ [augment_pair() nếu training]

5. AUGMENTED SAMPLE
   img: xoay 15°, flipped horizontally
   msk: xoay 15°, flipped horizontally (cùng transforms!)

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
- Tất cả 22,677 slices được lưu
- Tổng: ~23 GB

RAM (trong training):
- DataLoader load batch_size × num_workers slices
- Ví dụ: 12 × 4 = 48 slices trong RAM
- ~48 MB (không đáng kể)

GPU:
- Một batch: 12 slices
- Images: 12 × 4 × 256 × 256 × 4 bytes = 12 MB
- Masks: 12 × 1 × 256 × 256 × 4 bytes = 3 MB
- Model: ~14M parameters × 4 bytes = 56 MB
- Activations: ~500 MB (trong forward pass)
- Gradients: ~56 MB
- Optimizer state: ~112 MB
- Tổng: ~740 MB mỗi batch
```

**Tại sao GPU memory quan trọng?**

Nếu batch_size quá lớn:
```
batch_size=12: ~740 MB ✅
batch_size=32: ~1.9 GB ✅
batch_size=64: ~3.8 GB ⚠️ (có thể overflow trên GPU 4GB)
```

---

## 6. Modification Guide

### 6.1 Thêm Augmentation Mới: Gaussian Blur

**File**: `transforms.py`

```python
# Thêm import
import cv2

# Trong augment_pair(), sau rotation:
if train and random.random() < 0.3:  # 30% chance
    # Convert sang numpy
    img_np = np.array(img)

    # Áp dụng Gaussian blur
    img_np = cv2.GaussianBlur(img_np, (5, 5), 1.0)

    # Convert lại
    img = Image.fromarray(img_np.astype(np.uint8))
```

### 6.2 Thêm Augmentation Mới: Elastic Deformation

```python
# Thêm vào transforms.py
from scipy.ndimage import gaussian_filter, map_coordinates

def elastic_deform(image, mask, alpha=30, sigma=5):
    """Áp dụng elastic deformation"""
    shape = image.shape

    # Random displacement fields
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    # Coordinate meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # Áp dụng deformation
    image_deformed = map_coordinates(image, indices, order=1).reshape(shape)
    mask_deformed = map_coordinates(mask, indices, order=0).reshape(shape)

    return image_deformed, mask_deformed

# Sử dụng trong augment_pair():
if train and random.random() < 0.2:  # 20% chance
    img_np = np.array(img)
    msk_np = np.array(msk)
    img_np, msk_np = elastic_deform(img_np, msk_np)
    img = Image.fromarray(img_np.astype(np.uint8))
    msk = Image.fromarray(msk_np.astype(np.uint8))
```

### 6.3 Thay đổi Normalization: Z-Score thay vì [0,1]

**File**: `prepare_brats2020_h5.py`

Thay thế `_rescale01()`:

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

**Sau đó update Dataset** để KHÔNG làm [0,1] clipping:

```python
# Trong brats2020_dataset.py, xóa dòng này:
# arr /= 255.0

# Giữ giá trị như là (z-scored)
```

### 6.4 Sử dụng Modality Khác

**Preprocess chỉ FLAIR** (index 0):

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

### 6.5 Filter Slices Khác

**Giữ TẤT CẢ slices** (bao gồm cả rỗng):

```python
# Trong prepare_brats2020_h5.py, comment out:
# if tumor_ratio < min_tumor_ratio:
#     skipped_no_tumor += 1
#     continue

# Bây giờ tất cả slices được lưu
```

**Hoặc qua command line**:
```bash
python scripts/prepare_brats2020_h5.py --min_tumor_ratio 0.0
```

---

## 7. Debugging Tips

### 7.1 Test Preprocessing

```bash
# Chỉ xử lý 100 slices để test
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

# Lấy sample
sample = ds[0]

# Visualize
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# Hiển thị tất cả 4 modalities
for i in range(4):
    axes[i].imshow(sample['image'][i], cmap='gray')
    axes[i].set_title(f"Channel {i}")
    axes[i].axis('off')

# Hiển thị mask
axes[4].imshow(sample['mask'][0], cmap='hot')
axes[4].set_title("Mask")
axes[4].axis('off')

plt.suptitle(f"Slice: {sample['slice_id']}, Label: {sample['label']}")
plt.show()
```

### 7.3 Kiểm tra Augmentation

```python
# Load cùng sample nhiều lần (nên khác nhau do augmentation)
for i in range(4):
    sample = ds[0]  # Cùng index
    plt.subplot(2, 2, i+1)
    plt.imshow(sample['image'][0], cmap='gray')
    plt.title(f"Variation {i+1}")
plt.show()
```

### 7.4 Xác minh Tốc độ Loading Data

```python
from torch.utils.data import DataLoader
import time

ds = SliceDataset("data/processed_full_multimodal", "split_train_fold0.txt")
loader = DataLoader(ds, batch_size=12, num_workers=4, shuffle=True)

start = time.time()
for i, batch in enumerate(loader):
    if i >= 10:
        break  # Time 10 batches đầu
end = time.time()

print(f"Average time per batch: {(end-start)/10:.3f} seconds")
# Nên <1 giây cho hiệu suất tốt
```

---

## Tóm tắt

### Chúng ta Đã học gì

✅ **Giai đoạn 1**: Preprocessing (`prepare_brats2020_h5.py`)
   - Đọc HDF5 → Normalize → Resize → Lưu PNG/NPY
   - Tạo metadata (labels.csv, mapping.csv)
   - Generate cross-validation splits

✅ **Giai đoạn 2**: Dataset (`brats2020_dataset.py`)
   - PyTorch Dataset class
   - Lazy loading (load on demand)
   - Auto-detect single/multi-modal

✅ **Giai đoạn 3**: Augmentation (`transforms.py`)
   - Rotation, flipping
   - Áp dụng on-the-fly trong training
   - Cùng transforms cho image và mask

### Điểm chính Rút ra

1. **Preprocess một lần, sử dụng mãi mãi**: Tradeoff giữa disk space và computation time
2. **Lazy loading**: Không load tất cả 23GB vào RAM cùng lúc
3. **On-the-fly augmentation**: Vô hạn biến thể dữ liệu
4. **Stratified splits**: Đảm bảo train/val sets cân bằng
5. **Multi-modal flexibility**: Cùng code hoạt động cho 1 hoặc 4 channels

### Các Bước Tiếp theo

Bây giờ bạn hiểu data flow từ raw files đến PyTorch batches!

👉 **Tiếp theo**: [[v_03_MODEL_ARCHITECTURE|Part 3 - Model Architecture]]

Tìm hiểu neural network xử lý các batches này như thế nào!

---

[[v_TECHNICAL_REPORT_INDEX|← Quay lại Index]] | [[v_03_MODEL_ARCHITECTURE|Tiếp theo: Model Architecture →]]
