# Phần 10: Hướng Dẫn Mở Rộng

**Điều hướng**: [[v_TECHNICAL_REPORT_INDEX|← Về Index]]

---

## Mục Lục

1. [Tổng Quan](#tổng-quan)
2. [Thêm Component Model Mới](#thêm-component-model-mới)
3. [Thêm Loss Function Mới](#thêm-loss-function-mới)
4. [Thêm Metric Mới](#thêm-metric-mới)
5. [Thêm Augmentation Dữ Liệu Mới](#thêm-augmentation-dữ-liệu-mới)
6. [Hỗ Trợ Dataset Mới](#hỗ-trợ-dataset-mới)
7. [Implement Model 3D](#implement-model-3d)
8. [Mở Rộng Multi-Task](#mở-rộng-multi-task)
9. [Triển Khai và Production](#triển-khai-và-production)

---

## Tổng Quan

Hướng dẫn này chỉ cho bạn cách **mở rộng BrainTumNet** với tính năng, component và khả năng mới.

### Triết Lý Mở Rộng

- **Modular**: Thêm component mà không phá code hiện có
- **Có thể cấu hình**: Điều khiển qua config YAML
- **Đã test**: Xác minh mỗi phần thêm vào hoạt động
- **Có tài liệu**: Comment code của bạn

---

## Thêm Component Model Mới

### Ví dụ 1: Thêm Squeeze-and-Excitation (SE) Block

**Bước 1**: Tạo file mới `src/braintumnet/models/se_block.py`

```python
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Hiệu chỉnh lại feature theo kênh.

    Tham khảo: Hu et al. "Squeeze-and-Excitation Networks" (CVPR 2018)
    """

    def __init__(self, in_channels, reduction=16):
        """
        Args:
            in_channels: Số kênh đầu vào
            reduction: Tỷ lệ giảm cho bottleneck
        """
        super().__init__()

        # Squeeze: Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation: FC layer
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Tensor đầu vào (B, C, H, W)

        Returns:
            Tensor đã hiệu chỉnh lại (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Squeeze: (B, C, H, W) → (B, C, 1, 1) → (B, C)
        y = self.avg_pool(x).view(B, C)

        # Excitation: (B, C) → (B, C)
        y = self.fc(y)

        # Reshape và scale: (B, C) → (B, C, 1, 1) → (B, C, H, W)
        y = y.view(B, C, 1, 1)
        return x * y
```

**Bước 2**: Tích hợp vào U-Net

Sửa `src/braintumnet/models/seg_unet.py`:

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

        # Thêm SE block
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        x = self.block(x)
        x = self.se(x)  # Áp dụng SE trước pooling
        x_down = self.pool(x)
        return x, x_down
```

**Bước 3**: Thêm vào config

```yaml
# Trong configs/se_experiment.yaml
model:
  use_se: true  # Bật SE block
```

**Bước 4**: Cập nhật model builder

Trong `src/braintumnet/engine/trainer.py`:

```python
def build_model(cfg: Dict):
    mcfg = cfg["model"]
    use_se = mcfg.get("use_se", False)  # Lấy từ config

    return BrainTumNet(
        in_ch=mcfg["in_channels"],
        num_cls=mcfg["num_classes_cls"],
        base=mcfg["base"],
        dim=mcfg["dim"],
        patch=mcfg["patch_size"],
        depth=mcfg["depth"],
        n_heads=mcfg["n_heads"],
        roi_stop_grad=mcfg["roi_stop_grad"],
        use_se=use_se  # Truyền vào model
    )
```

**Bước 5**: Test

```bash
# Train với SE block
python train.py --cfg configs/se_experiment.yaml --fold 0

# So sánh với baseline
# Mong đợi: +0.5-1% cải thiện Dice
```

---

### Ví dụ 2: Thêm Residual Connection

**Sửa encoder block** trong `seg_unet.py`:

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

        # Projection residual (nếu kênh thay đổi)
        if residual and in_ch != out_ch:
            self.downsample = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        x = self.block(x)

        if self.residual:
            x = x + identity  # Kết nối residual

        x_down = self.pool(x)
        return x, x_down
```

**Config**:
```yaml
model:
  residual: true
```

---

## Thêm Loss Function Mới

### Ví dụ 1: Focal Loss

**Tạo** `src/braintumnet/losses.py` (thêm vào file hiện có):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss để giải quyết class imbalance.

    Tham khảo: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)

    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    Args:
        alpha: Hệ số trọng số [0, 1]
        gamma: Tham số focusing γ ≥ 0
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) - dự đoán thô
            targets: (B, 1, H, W) - target nhị phân (0 hoặc 1)

        Returns:
            Focal loss (scalar)
        """
        # Chuyển logit sang xác suất
        probs = torch.sigmoid(logits)

        # Tính trọng số focal
        # Cho mẫu dương (target=1): (1 - p)^γ
        # Cho mẫu âm (target=0): p^γ
        focal_weight = torch.where(
            targets == 1,
            (1 - probs).pow(self.gamma),
            probs.pow(self.gamma)
        )

        # Tính BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Áp dụng trọng số focal và alpha
        focal_loss = focal_weight * bce

        # Áp dụng trọng số alpha
        alpha_weight = torch.where(
            targets == 1,
            self.alpha,
            1 - self.alpha
        )
        focal_loss = alpha_weight * focal_loss

        return focal_loss.mean()


class DiceFocalLoss(nn.Module):
    """
    Kết hợp Dice Loss + Focal Loss.
    """

    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal = FocalLoss(alpha, gamma)

    def forward(self, logits, targets):
        # Dice loss (từ code hiện có)
        dice_loss = dice_loss_with_logits(logits, targets)

        # Focal loss
        focal_loss = self.focal(logits, targets)

        # Kết hợp
        total = self.dice_weight * dice_loss + self.focal_weight * focal_loss

        return total
```

**Cập nhật config**:

```yaml
# Trong configs/focal_experiment.yaml
train:
  seg_criterion: "DiceFocal"  # Dùng loss mới
  focal_alpha: 0.25
  focal_gamma: 2.0
```

**Cập nhật trainer.py**:

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
        raise ValueError(f"Criterion không rõ: {seg_criterion_name}")

    # ... phần còn lại của training
```

---

### Ví dụ 2: Boundary Loss

**Để độ chính xác ranh giới tốt hơn**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np

class BoundaryLoss(nn.Module):
    """
    Boundary Loss để phân định ranh giới tốt hơn.

    Tham khảo: Kervadec et al. "Boundary loss for highly unbalanced segmentation" (MIDL 2019)
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

        # Tính distance transform (trên CPU, numpy)
        B, C, H, W = targets.shape
        dist_maps = []

        for b in range(B):
            target_np = targets[b, 0].cpu().numpy()

            # Distance transform (khoảng cách đến ranh giới)
            dist = distance_transform_edt(target_np) + distance_transform_edt(1 - target_np)
            dist_maps.append(dist)

        dist_maps = torch.from_numpy(np.stack(dist_maps)).unsqueeze(1).to(logits.device)

        # Boundary loss: tích phân distance map trọng số bởi dự đoán
        boundary_loss = (probs * dist_maps).mean()

        return boundary_loss
```

**Dùng**:
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

## Thêm Metric Mới

### Ví dụ: Sensitivity và Specificity

**Thêm vào** `src/braintumnet/metrics.py`:

```python
def sensitivity_specificity(logits: torch.Tensor, target: torch.Tensor, eps=1e-6) -> Tuple[float, float]:
    """
    Tính Sensitivity (Recall) và Specificity.

    Sensitivity = TP / (TP + FN) - True Positive Rate
    Specificity = TN / (TN + FP) - True Negative Rate

    Args:
        logits: Dự đoán model (B, 1, H, W)
        target: Ground truth (B, 1, H, W)
        eps: Epsilon nhỏ cho ổn định số học

    Returns:
        (sensitivity, specificity)
    """
    pred = binarize(logits)

    # True Positive, False Negative, True Negative, False Positive
    tp = (pred * target).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    fp = (pred * (1 - target)).sum().item()

    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)

    return sensitivity, specificity
```

**Dùng trong evaluator.py**:

```python
# Trong validation loop
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

## Thêm Augmentation Dữ Liệu Mới

### Ví dụ: Biến Dạng Đàn Hồi

**Tạo** `src/braintumnet/data/augmentations.py`:

```python
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

def elastic_deformation(image, mask, alpha=30, sigma=5):
    """
    Biến dạng đàn hồi cho augmentation ảnh y tế.

    Args:
        image: (H, W) hoặc (H, W, C) numpy array
        mask: (H, W) numpy array
        alpha: Độ mạnh biến dạng
        sigma: Độ mịn biến dạng

    Returns:
        (image, mask) đã biến dạng
    """
    shape = image.shape[:2]

    # Trường dịch chuyển ngẫu nhiên
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    # Lưới tọa độ
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).reshape(-1), (x + dx).reshape(-1)

    # Áp dụng biến dạng
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
    """Áp dụng biến dạng đàn hồi với xác suất p."""
    if np.random.rand() < p:
        return elastic_deformation(image, mask, alpha, sigma)
    return image, mask
```

**Tích hợp vào dataset** (`brats2020_dataset.py`):

```python
from .augmentations import random_elastic_deformation

class SliceDataset(Dataset):
    def __init__(self, ...):
        # ... code hiện có ...
        self.elastic_p = elastic_p  # Thêm vào constructor

    def __getitem__(self, idx):
        # ... load ảnh và mask ...

        if self.train:
            # Augmentation hiện có
            img, msk = augment_pair(img, msk, self.rotate_deg, self.hflip_p, self.vflip_p)

            # Thêm biến dạng đàn hồi
            if self.elastic_p > 0:
                img_np = img.cpu().numpy().squeeze()
                msk_np = msk.cpu().numpy().squeeze()
                img_np, msk_np = random_elastic_deformation(img_np, msk_np, p=self.elastic_p)
                img = torch.from_numpy(img_np).unsqueeze(0).float()
                msk = torch.from_numpy(msk_np).unsqueeze(0).float()

        # ... phần còn lại ...
```

**Config**:
```yaml
augment:
  rotate_deg: 20
  hflip_p: 0.5
  vflip_p: 0.5
  elastic_p: 0.3  # 30% cơ hội
  elastic_alpha: 30
  elastic_sigma: 5
```

---

## Hỗ Trợ Dataset Mới

### Ví dụ: Hỗ trợ Dataset TCGA-GBM

**Bước 1**: Tạo script tiền xử lý mới

Tạo `scripts/prepare_tcga.py`:

```python
import os
import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image

def process_tcga(tcga_root, output_dir, img_size=256):
    """
    Xử lý dataset TCGA-GBM.

    Cấu trúc TCGA:
    tcga_root/
    ├── TCGA-02-0001/
    │   ├── t1.nii.gz
    │   ├── t2.nii.gz
    │   ├── flair.nii.gz
    │   └── seg.nii.gz
    ├── TCGA-02-0002/
    ...

    Args:
        tcga_root: Đường dẫn đến dataset TCGA
        output_dir: Nơi lưu ảnh đã xử lý
        img_size: Kích thước ảnh output
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    patients = sorted(os.listdir(tcga_root))
    labels = []  # Cho classification (nếu có)

    for patient_id in patients:
        patient_dir = os.path.join(tcga_root, patient_id)

        # Load file NIfTI
        t1 = nib.load(os.path.join(patient_dir, "t1.nii.gz")).get_fdata()
        t2 = nib.load(os.path.join(patient_dir, "t2.nii.gz")).get_fdata()
        flair = nib.load(os.path.join(patient_dir, "flair.nii.gz")).get_fdata()
        seg = nib.load(os.path.join(patient_dir, "seg.nii.gz")).get_fdata()

        # Xử lý mỗi lát cắt
        for slice_idx in range(t1.shape[2]):
            t1_slice = t1[:, :, slice_idx]
            t2_slice = t2[:, :, slice_idx]
            flair_slice = flair[:, :, slice_idx]
            seg_slice = seg[:, :, slice_idx]

            # Bỏ qua lát trống
            if seg_slice.sum() == 0:
                continue

            # Chuẩn hóa
            t1_slice = (t1_slice - t1_slice.min()) / (t1_slice.max() - t1_slice.min() + 1e-6)
            # ... tương tự cho t2, flair ...

            # Stack modality
            img_stack = np.stack([flair_slice, t1_slice, t2_slice, np.zeros_like(t1_slice)], axis=2)

            # Resize
            img_pil = Image.fromarray((img_stack * 255).astype(np.uint8))
            img_resized = img_pil.resize((img_size, img_size), Image.BILINEAR)

            seg_pil = Image.fromarray((seg_slice * 255).astype(np.uint8))
            seg_resized = seg_pil.resize((img_size, img_size), Image.NEAREST)

            # Lưu
            filename = f"{patient_id}_slice_{slice_idx:03d}.png"
            img_resized.save(os.path.join(output_dir, "images", filename))
            seg_resized.save(os.path.join(output_dir, "masks", filename))

    print(f"Đã xử lý {len(patients)} bệnh nhân")

if __name__ == "__main__":
    process_tcga("data/raw/TCGA-GBM", "data/processed_tcga", img_size=256)
```

**Bước 2**: Cập nhật dataset class

Sửa `brats2020_dataset.py` hoặc tạo `tcga_dataset.py`:

```python
class TCGADataset(Dataset):
    """Giống như SliceDataset nhưng cho TCGA."""
    # ... implementation tương tự ...
```

**Bước 3**: Train trên TCGA

```yaml
# configs/tcga_experiment.yaml
exp_name: "braintumnet_tcga"

data:
  proc_root: "data/processed_tcga"
  # ... phần còn lại tương tự ...
```

---

## Implement Model 3D

### Ví dụ: 3D U-Net

**Tạo** `src/braintumnet/models/unet3d.py`:

```python
import torch
import torch.nn as nn

class Conv3DBlock(nn.Module):
    """Block convolution 3D."""

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
    """Block encoder 3D."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = Conv3DBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.block(x)
        x_down = self.pool(x)
        return x, x_down


class Decoder3DBlock(nn.Module):
    """Block decoder 3D."""

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
    3D U-Net cho segmentation volumetric.

    Input: (B, C, D, H, W) trong đó D là depth (số lát cắt)
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

**Tạo dataset 3D**:

```python
class VolumetricDataset(Dataset):
    """
    Load toàn bộ volume 3D thay vì lát cắt.

    Returns:
        image: (C, D, H, W) trong đó D=num_slices
        mask: (1, D, H, W)
    """

    def __getitem__(self, idx):
        # Load tất cả lát cho một bệnh nhân
        patient_id = self.patient_ids[idx]
        slices = []
        masks = []

        for slice_idx in range(155):  # BraTS có 155 lát
            img = load_image(patient_id, slice_idx)  # (C, H, W)
            msk = load_mask(patient_id, slice_idx)   # (1, H, W)
            slices.append(img)
            masks.append(msk)

        volume = torch.stack(slices, dim=1)  # (C, D, H, W)
        mask_volume = torch.stack(masks, dim=1)  # (1, D, H, W)

        return {"image": volume, "mask": mask_volume}
```

**Cân nhắc bộ nhớ**:
```
2D: (B, C, H, W) = (12, 4, 256, 256) = 12 MB
3D: (B, C, D, H, W) = (1, 4, 155, 256, 256) = 160 MB mỗi mẫu!

Giải pháp: Giảm batch size hoặc dùng training patch-based
```

---

## Mở Rộng Multi-Task

### Ví dụ: Thêm Segmentation Vùng Con Của U

**Sửa model output**:

```python
class BrainTumNetMultiRegion(nn.Module):
    """
    Segmentation nhiều vùng:
    - Class 0: Background
    - Class 1: Lõi hoại tử
    - Class 2: Phù nề
    - Class 3: U tăng cường
    """

    def __init__(self, in_ch=4, num_regions=4, num_cls=2, ...):
        super().__init__()
        self.seg = SegUNetMasked(in_ch=in_ch, base=base, ...)

        # Head segmentation multi-class
        self.seg_head = nn.Conv2d(base, num_regions, 1)

        # ... phần còn lại tương tự ...

    def forward(self, x):
        # Lấy feature
        features = self.seg.encoder(x)

        # Segmentation multi-class
        seg_logits = self.seg_head(features)  # (B, 4, H, W)

        # Classification (giống như trước)
        cls_logits = self.cls_backbone(...)

        return seg_logits, cls_logits
```

**Cập nhật loss**:

```python
# Dùng CrossEntropyLoss cho multi-class
seg_criterion = nn.CrossEntropyLoss()

# Trong vòng lặp training
seg_loss = seg_criterion(seg_logits, mask_multi_class)
# mask_multi_class shape: (B, H, W) với giá trị trong [0, 1, 2, 3]
```

---

## Triển Khai và Production

### Ví dụ 1: Export ONNX

**Export model**:

```python
import torch
import torch.onnx

# Load model đã train
model = BrainTumNet(in_ch=4, ...)
load_ckpt(model, "checkpoints/best_model.pth")
model.eval()

# Input dummy
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

print("Model đã export sang braintumnet.onnx")
```

**Dùng ONNX cho inference**:

```python
import onnxruntime as ort
import numpy as np

# Load model ONNX
session = ort.InferenceSession("braintumnet.onnx")

# Chuẩn bị input
img = load_and_preprocess_image("patient_001.png")
img_np = img.numpy()  # (1, 4, 256, 256)

# Chạy inference
outputs = session.run(None, {"input": img_np})
seg_logits, cls_logits = outputs

# Post-process
seg_prob = 1 / (1 + np.exp(-seg_logits))  # Sigmoid
seg_binary = (seg_prob > 0.5).astype(np.uint8)

print(f"Shape segmentation: {seg_binary.shape}")
print(f"Classification: {cls_logits.argmax()}")
```

---

### Ví dụ 2: TorchScript

**Để triển khai trong C++**:

```python
import torch

# Load model
model = BrainTumNet(...)
load_ckpt(model, "checkpoints/best_model.pth")
model.eval()

# Trace model
example_input = torch.randn(1, 4, 256, 256)
traced_model = torch.jit.trace(model, example_input)

# Lưu
traced_model.save("braintumnet_traced.pt")

# Load trong Python
loaded = torch.jit.load("braintumnet_traced.pt")
output = loaded(example_input)
```

**Dùng trong C++**:

```cpp
#include <torch/script.h>
#include <iostream>

int main() {
    // Load model
    torch::jit::script::Module model = torch::jit::load("braintumnet_traced.pt");

    // Tạo input tensor
    auto input = torch::randn({1, 4, 256, 256});

    // Chạy inference
    auto outputs = model.forward({input}).toTuple();
    auto seg = outputs->elements()[0].toTensor();
    auto cls = outputs->elements()[1].toTensor();

    std::cout << "Inference hoàn thành!" << std::endl;
    return 0;
}
```

---

### Ví dụ 3: Flask Web API

**Tạo** `app.py`:

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

# Load model một lần khi khởi động
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BrainTumNet(in_ch=4, num_cls=2, base=32, dim=256, patch=8, depth=2, n_heads=4)
load_ckpt(model, "checkpoints/best_model.pth", map_location=device)
model.to(device)
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    """
    Dự đoán trên ảnh upload.

    Request:
        {
            "image": "dữ_liệu_ảnh_encode_base64"
        }

    Response:
        {
            "segmentation": "mask_encode_base64",
            "classification": "HGG" hoặc "LGG",
            "confidence": 0.95
        }
    """
    # Lấy ảnh từ request
    data = request.json
    img_data = base64.b64decode(data["image"])
    img = Image.open(io.BytesIO(img_data)).convert("L")

    # Tiền xử lý
    img_resized = resize_pad_to_square(img, 256, is_mask=False)
    img_tensor = to_tensor01(img_resized).unsqueeze(0).to(device)  # (1, 1, 256, 256)

    # Dự đoán
    with torch.no_grad():
        seg_logits, cls_logits = model(img_tensor)
        seg_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
        cls_prob = torch.softmax(cls_logits, dim=1).squeeze().cpu().numpy()
        cls_pred = cls_prob.argmax()

    # Chuyển segmentation sang ảnh
    seg_binary = (seg_prob > 0.5).astype(np.uint8) * 255
    seg_img = Image.fromarray(seg_binary)

    # Encode segmentation dưới dạng base64
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

**Chạy API**:

```bash
python app.py

# Test với curl
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "dữ_liệu_base64_encode..."}'
```

---

## Tóm Tắt

Hướng dẫn này đã đề cập:

1. ✓ Thêm component model (SE block, residual connection)
2. ✓ Tạo loss function mới (Focal, Boundary)
3. ✓ Implement metric mới (Sensitivity, Specificity)
4. ✓ Thêm augmentation (Biến dạng đàn hồi)
5. ✓ Hỗ trợ dataset mới (TCGA)
6. ✓ Xây dựng model 3D (UNet3D)
7. ✓ Mở rộng multi-task (Segmentation nhiều vùng)
8. ✓ Triển khai (ONNX, TorchScript, Flask API)

**Nguyên Tắc Chính**:
- Giữ code modular
- Dùng file config cho tham số
- Test mỗi phần thêm vào
- Tài liệu hóa thay đổi

**Bước Tiếp Theo**:
- Đọc paper cho kỹ thuật mới
- Thử nghiệm với ablation
- Benchmark hiệu suất
- Triển khai vào production

---

**Chúc mừng!** Bạn giờ có tài liệu hoàn chỉnh cho BrainTumNet.

**Quay lại**: [[v_TECHNICAL_REPORT_INDEX|Index]]
