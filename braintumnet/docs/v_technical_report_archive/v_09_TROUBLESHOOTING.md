# Phần 9: Hướng Dẫn Xử Lý Lỗi

**Điều hướng**: [[v_TECHNICAL_REPORT_INDEX|← Về Index]]

---

## Mục Lục

1. [Lỗi Thường Gặp](#lỗi-thường-gặp)
2. [Vấn Đề Cài Đặt](#vấn-đề-cài-đặt)
3. [Vấn Đề Tiền Xử Lý Dữ Liệu](#vấn-đề-tiền-xử-lý-dữ-liệu)
4. [Vấn Đề Training](#vấn-đề-training)
5. [Vấn Đề Bộ Nhớ](#vấn-đề-bộ-nhớ)
6. [Vấn Đề Hiệu Suất](#vấn-đề-hiệu-suất)
7. [Vấn Đề Inference](#vấn-đề-inference)
8. [Chiến Lược Debug](#chiến-lược-debug)

---

## Lỗi Thường Gặp

### Lỗi: "CUDA out of memory"

**Triệu chứng**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB
(GPU 0; 11.00 GiB total capacity; 9.12 GiB already allocated; ...)
```

**Nguyên nhân**:
1. Batch size quá lớn
2. Model quá lớn
3. Độ phân giải ảnh quá cao
4. Rò rỉ bộ nhớ (không giải phóng tensor)

**Giải pháp**:

**Giải pháp 1**: Giảm batch size
```yaml
# Trong file config
train:
  batch_size: 8  # Giảm từ 12
```

**Giải pháp 2**: Bật mixed precision
```yaml
train:
  amp: true  # FP16 dùng ít hơn 50% bộ nhớ
```

**Giải pháp 3**: Giảm kích thước model
```yaml
model:
  base: 16  # Giảm từ 32
```

**Giải pháp 4**: Giảm kích thước ảnh
```yaml
data:
  img_size: 128  # Giảm từ 256
```

**Giải pháp 5**: Xóa cache GPU
```python
# Thêm vào vòng lặp training
import torch
torch.cuda.empty_cache()
```

**Giải pháp 6**: Dùng gradient accumulation
```python
# Trong trainer.py, sửa vòng lặp training
accumulation_steps = 4  # Batch size hiệu quả = 12 × 4 = 48

for i, batch in enumerate(train_loader):
    loss = compute_loss(...)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

### Lỗi: "Checkpoint fold mismatch"

**Triệu chứng**:
```
ValueError: Fold mismatch! Checkpoint is for fold 0,
but you're trying to resume fold 1.
```

**Nguyên nhân**: Resume training với checkpoint sai

**Giải pháp**:
```bash
# Kiểm tra fold của checkpoint
python -c "import torch; print(torch.load('checkpoints/last_fold0.pth')['fold'])"
# Output: 0

# Dùng checkpoint đúng
python train.py --cfg configs/default.yaml --fold 0 --resume checkpoints/last_fold0.pth
```

---

### Lỗi: "File not found: split_train_fold0.txt"

**Triệu chứng**:
```
FileNotFoundError: [Errno 2] No such file or directory:
'data/processed_full_multimodal/split_train_fold0.txt'
```

**Nguyên nhân**: Quên chạy tiền xử lý

**Giải pháp**:
```bash
# Chạy tiền xử lý trước
python scripts/prepare_brats2020_h5.py \
    --h5 data/raw/brats2020_training.h5 \
    --out data/processed_full_multimodal \
    --modality multi \
    --img_size 256 \
    --slices_per_case 30 \
    --num_folds 5

# Sau đó train
python train.py --cfg configs/full_dataset_multimodal.yaml --fold 0
```

---

### Lỗi: "NaN loss during training"

**Triệu chứng**:
```
Epoch 5/100 | Train Loss nan | Val IoU nan | Dice nan
```

**Nguyên nhân**:
1. Learning rate quá cao
2. Gradient explosion
3. Không ổn định số học
4. Dữ liệu xấu (giá trị inf/nan)

**Giải pháp**:

**Giải pháp 1**: Giảm learning rate
```yaml
train:
  lr: 5.0e-5  # Giảm từ 1.5e-4
```

**Giải pháp 2**: Thêm gradient clipping
```python
# Trong trainer.py
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Giải pháp 3**: Kiểm tra dữ liệu
```python
# Thêm vào dataloader
import torch
import numpy as np

def check_batch(batch):
    img = batch["image"]
    msk = batch["mask"]

    if torch.isnan(img).any():
        print("NaN trong ảnh!")
    if torch.isinf(img).any():
        print("Inf trong ảnh!")
    if img.max() > 10 or img.min() < -10:
        print(f"Giá trị đáng ngờ: min={img.min()}, max={img.max()}")

# Dùng trong vòng lặp training
for batch in train_loader:
    check_batch(batch)
    ...
```

**Giải pháp 4**: Bật phát hiện anomaly
```python
# Ở đầu train.py
import torch
torch.autograd.set_detect_anomaly(True)
```

---

### Lỗi: "Expected 4D tensor, got 3D"

**Triệu chứng**:
```
RuntimeError: Expected 4D (unbatched 3D) or 5D (batched 4D) input to conv2d,
but got input of size: [256, 256]
```

**Nguyên nhân**: Thiếu batch dimension hoặc channel dimension

**Giải pháp**:
```python
# Shape đúng:
# Training: (B, C, H, W) = (12, 4, 256, 256)
# Inference: (1, C, H, W) = (1, 4, 256, 256)

# Thêm batch dimension
img_tensor = img_tensor.unsqueeze(0)  # (C, H, W) → (1, C, H, W)

# Thêm channel dimension
img_tensor = img_tensor.unsqueeze(0)  # (H, W) → (1, H, W)
```

---

## Vấn Đề Cài Đặt

### Lỗi: "No module named 'torch'"

**Triệu chứng**:
```
ModuleNotFoundError: No module named 'torch'
```

**Giải pháp**:
```bash
# Cài PyTorch (kiểm tra https://pytorch.org cho hệ thống của bạn)

# Chỉ CPU
pip install torch torchvision

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### Lỗi: "CUDA not available"

**Triệu chứng**:
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Chẩn đoán**:
```bash
# Kiểm tra NVIDIA driver
nvidia-smi

# Kiểm tra phiên bản CUDA của PyTorch
python -c "import torch; print(torch.version.cuda)"

# Kiểm tra nếu PyTorch được build với CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**Giải pháp**:

**Giải pháp 1**: Cài PyTorch có CUDA
```bash
# Cài lại với phiên bản CUDA đúng
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Giải pháp 2**: Cập nhật driver NVIDIA
```bash
# Ubuntu/Linux
sudo ubuntu-drivers autoinstall
sudo reboot

# Windows
# Tải từ https://www.nvidia.com/Download/index.aspx
```

**Giải pháp 3**: Dùng CPU (chậm hơn)
```yaml
# Trong training, model tự động dùng CPU nếu CUDA không có
# Mong đợi chậm hơn 10-20×
```

---

## Vấn Đề Tiền Xử Lý Dữ Liệu

### Lỗi: "HDF5 file corrupt"

**Triệu chứng**:
```
OSError: Unable to open file (file signature not found)
```

**Giải pháp**:
```bash
# Tải lại file HDF5
# Kiểm tra tính toàn vẹn file
md5sum data/raw/brats2020_training.h5

# Nếu lỗi, tải lại từ BraTS
```

---

### Lỗi: "Preprocessing crashes midway"

**Triệu chứng**:
```
Processing patient 234/369...
Killed
```

**Nguyên nhân**: Hết RAM

**Giải pháp**:
```bash
# Xử lý theo batch nhỏ hơn
python scripts/prepare_brats2020_h5.py \
    --h5 data/raw/brats2020_training.h5 \
    --out data/processed_full_multimodal \
    --modality multi \
    --start_idx 0 \
    --end_idx 100  # Xử lý 100 bệnh nhân mỗi lần

# Sau đó tiếp tục
python scripts/prepare_brats2020_h5.py \
    ... \
    --start_idx 100 \
    --end_idx 200

# (Cần sửa code để hỗ trợ --start_idx/--end_idx)
```

---

### Cảnh báo: "Imbalanced dataset"

**Triệu chứng**:
```
WARNING: Tumor slices: 2340 (21%), Non-tumor: 8730 (79%)
```

**Giải pháp**: Đã được xử lý bởi `tumor_slice_ratio` trong config
```yaml
data:
  tumor_slice_ratio: 0.5  # Cân bằng thành 50-50
```

---

## Vấn Đề Training

### Lỗi: "Training stuck at low accuracy"

**Triệu chứng**:
```
Epoch 50/100 | Dice 0.65 | Không cải thiện
```

**Chẩn đoán**:
```python
# Kiểm tra learning rate
print(f"LR hiện tại: {optimizer.param_groups[0]['lr']}")

# Kiểm tra nếu model cập nhật
before = model.seg.e1.block[0].weight.clone()
# Train một batch
after = model.seg.e1.block[0].weight
print(f"Weight đã thay đổi: {not torch.equal(before, after)}")
```

**Giải pháp**:

**Giải pháp 1**: Tăng learning rate
```yaml
train:
  lr: 3.0e-4  # Tăng từ 1.5e-4
```

**Giải pháp 2**: Kiểm tra nếu có layer bị freeze
```python
# Tất cả tham số nên trainable
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"Bị freeze: {name}")
        param.requires_grad = True
```

**Giải pháp 3**: Khởi tạo lại model
```python
# Đôi khi khởi tạo xấu
model = build_model(cfg).to(device)
# Train lại từ đầu
```

---

### Lỗi: "Validation worse than training"

**Triệu chứng**:
```
Train Dice: 0.95 | Val Dice: 0.75 (khoảng cách 15%)
```

**Nguyên nhân**: Overfitting

**Giải pháp**:

**Giải pháp 1**: Tăng weight decay
```yaml
train:
  weight_decay: 5.0e-4  # Tăng từ 1.0e-4
```

**Giải pháp 2**: Thêm dropout
```python
# Trong định nghĩa model
self.dropout = nn.Dropout(0.3)

# Trong forward pass
x = self.dropout(x)
```

**Giải pháp 3**: Augmentation nhiều hơn
```yaml
augment:
  rotate_deg: 30  # Tăng từ 20
  hflip_p: 0.7    # Tăng từ 0.5
```

**Giải pháp 4**: Dùng early stopping (đã có trong config)
```yaml
train:
  early_stop_patience: 20  # Giảm từ 30
```

---

### Lỗi: "Loss oscillating wildly"

**Triệu chứng**:
```
Epoch 10: loss=0.25
Epoch 11: loss=0.45  ← Nhảy lớn
Epoch 12: loss=0.22
Epoch 13: loss=0.52  ← Nhảy nữa
```

**Nguyên nhân**: Learning rate quá cao

**Giải pháp**:
```yaml
train:
  lr: 5.0e-5  # Giảm từ 1.5e-4
```

---

## Vấn Đề Bộ Nhớ

### Lỗi: "System RAM full"

**Triệu chứng**:
```
MemoryError: Unable to allocate array
```

**Nguyên nhân**: Data loading dùng quá nhiều RAM

**Giải pháp**:

**Giải pháp 1**: Giảm num_workers
```yaml
train:
  workers: 2  # Giảm từ 4
```

**Giải pháp 2**: Dùng lazy loading (đã implement trong Dataset)

**Giải pháp 3**: Giảm batch size
```yaml
train:
  batch_size: 8  # Giảm từ 12
```

---

### Lỗi: "Disk space full"

**Triệu chứng**:
```
OSError: [Errno 28] No space left on device
```

**Giải pháp**:

**Giải pháp 1**: Dọn dẹp checkpoint cũ
```bash
# Chỉ giữ checkpoint tốt nhất
cd checkpoints
rm last_fold*.pth  # Xóa checkpoint trung gian
```

**Giải pháp 2**: Giảm logging TensorBoard
```yaml
logging:
  use_tensorboard: false  # Tắt TensorBoard
```

**Giải pháp 3**: Dùng ổ đĩa ngoài
```yaml
logging:
  out_dir: "/mnt/external/runs"
  save_dir: "/mnt/external/checkpoints"
```

---

## Vấn Đề Hiệu Suất

### Vấn đề: "Training too slow"

**Triệu chứng**:
```
2.5 it/s (mong đợi 4-5 it/s)
```

**Chẩn đoán**:
```python
import time

# Đo thời gian data loading
start = time.time()
for i, batch in enumerate(train_loader):
    if i == 100:
        break
data_time = time.time() - start
print(f"Data loading: {data_time/100:.3f}s mỗi batch")

# Đo thời gian forward pass
start = time.time()
with torch.no_grad():
    for i, batch in enumerate(train_loader):
        img = batch["image"].to(device)
        model(img)
        if i == 100:
            break
forward_time = time.time() - start
print(f"Forward pass: {forward_time/100:.3f}s mỗi batch")
```

**Giải pháp**:

**Giải pháp 1**: Bật mixed precision (nếu chưa)
```yaml
train:
  amp: true  # Tăng tốc 2×
```

**Giải pháp 2**: Tăng num_workers (nếu data loading chậm)
```yaml
train:
  workers: 8  # Tăng từ 4 (nếu CPU có đủ core)
```

**Giải pháp 3**: Pin memory
```python
# Trong dataloader
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # Thêm này để transfer GPU nhanh hơn
)
```

**Giải pháp 4**: Dùng GPU nhanh hơn
```
RTX 3050:  2.5 it/s
RTX 3060:  4.7 it/s  ← Hiện tại
RTX 3090:  9.2 it/s
RTX 4090: 14.5 it/s
```

---

### Vấn đề: "Poor segmentation accuracy"

**Triệu chứng**:
```
Dice < 0.80 (mong đợi > 0.90)
```

**Chẩn đoán**:
```python
# Kiểm tra dataset
print(f"Số mẫu train: {len(train_ds)}")
print(f"Số mẫu val: {len(val_ds)}")

# Kiểm tra phân phối dữ liệu
labels = []
for i in range(len(train_ds)):
    labels.append(train_ds[i]["label"])
print(f"Phân phối lớp: {np.bincount(labels)}")

# Visualize dự đoán
import matplotlib.pyplot as plt
img, msk = val_ds[0]["image"], val_ds[0]["mask"]
with torch.no_grad():
    pred = model(img.unsqueeze(0).to(device))
    pred = torch.sigmoid(pred[0]).cpu().squeeze()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img.squeeze(), cmap='gray')
axes[0].set_title('Input')
axes[1].imshow(msk.squeeze(), cmap='gray')
axes[1].set_title('Ground Truth')
axes[2].imshow(pred, cmap='hot')
axes[2].set_title('Prediction')
plt.show()
```

**Giải pháp**:

**Giải pháp 1**: Train lâu hơn
```yaml
train:
  epochs: 200  # Tăng từ 150
  early_stop_patience: 50
```

**Giải pháp 2**: Dùng multi-modal (nếu đang dùng single-modal)
```yaml
data:
  modality: "multi"  # Dùng cả 4 modality
model:
  in_channels: 4
```

**Giải pháp 3**: Điều chỉnh weight loss
```yaml
train:
  seg_loss_weight: 1.5  # Nhấn mạnh segmentation
  cls_loss_weight: 0.3
```

---

## Vấn Đề Inference

### Lỗi: "Model outputs wrong shape"

**Triệu chứng**:
```
Mong đợi (1, 1, 256, 256), nhận được (1, 2, 256, 256)
```

**Nguyên nhân**: Model trả về cả seg và cls

**Giải pháp**:
```python
# Unpack cả hai output
seg_logits, cls_logits = model(img)

# Chỉ dùng segmentation
seg_prob = torch.sigmoid(seg_logits)
```

---

### Lỗi: "Prediction all zeros"

**Triệu chứng**:
```
Mask dự đoán trống (toàn số 0)
```

**Chẩn đoán**:
```python
# Kiểm tra khoảng output model
seg_logits, _ = model(img)
print(f"Khoảng logit: [{seg_logits.min():.2f}, {seg_logits.max():.2f}]")

seg_prob = torch.sigmoid(seg_logits)
print(f"Khoảng prob: [{seg_prob.min():.4f}, {seg_prob.max():.4f}]")

# Nếu prob max < 0.5, không có pixel nào được dự đoán
```

**Giải pháp**:

**Giải pháp 1**: Giảm threshold
```python
threshold = 0.3  # Giảm từ 0.5
seg_binary = (seg_prob > threshold).float()
```

**Giải pháp 2**: Kiểm tra nếu model load đúng
```python
# Xác minh checkpoint
checkpoint = torch.load(ckpt_path)
print(f"Key checkpoint: {checkpoint.keys()}")

# Load và xác minh
load_ckpt(model, ckpt_path)
print("Model đã load thành công")
```

**Giải pháp 3**: Kiểm tra chuẩn hóa input
```python
# Input nên trong khoảng [0, 1]
print(f"Khoảng input: [{img.min():.2f}, {img.max():.2f}]")

# Nếu không, chuẩn hóa
img = (img - img.min()) / (img.max() - img.min() + 1e-6)
```

---

## Chiến Lược Debug

### Chiến lược 1: Overfit trên Một Batch

**Mục đích**: Xác minh model có thể học

```python
# Trong script training
single_batch = next(iter(train_loader))

for epoch in range(100):
    optimizer.zero_grad()
    seg, cls = model(single_batch["image"].to(device))
    loss, _, _ = criterion(seg, single_batch["mask"].to(device),
                          cls, single_batch["label"].to(device))
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

# Mong đợi: Loss nên về gần 0
# Nếu không: Vấn đề với model/optimizer
```

---

### Chiến lược 2: Visualize Feature Trung Gian

**Mục đích**: Hiểu model học gì

```python
import matplotlib.pyplot as plt

# Hook để capture activation
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Đăng ký hook
model.seg.e1.block.register_forward_hook(get_activation('e1'))
model.seg.e2.block.register_forward_hook(get_activation('e2'))
model.seg.e3.block.register_forward_hook(get_activation('e3'))

# Forward pass
with torch.no_grad():
    model(img.unsqueeze(0).to(device))

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(activations['e1'][0, 0].cpu(), cmap='viridis')
axes[0].set_title('Encoder 1 (32 kênh)')
axes[1].imshow(activations['e2'][0, 0].cpu(), cmap='viridis')
axes[1].set_title('Encoder 2 (64 kênh)')
axes[2].imshow(activations['e3'][0, 0].cpu(), cmap='viridis')
axes[2].set_title('Encoder 3 (128 kênh)')
plt.show()
```

---

### Chiến lược 3: So Sánh Với Baseline

**Mục đích**: Cô lập vấn đề

```python
# Train U-Net đơn giản (không có attention, không có transformer)
from braintumnet.models.seg_unet import SegUNetMasked

simple_unet = SegUNetMasked(in_ch=4, base=32, dim=256,
                             patch=8, depth=0, n_heads=4)  # depth=0 tắt transformer

# Train và so sánh
# Nếu U-Net đơn giản hoạt động nhưng model đầy đủ không → vấn đề ở transformer/attention
# Nếu U-Net đơn giản cũng thất bại → vấn đề ở dữ liệu/training
```

---

### Chiến lược 4: Kiểm Tra Gradient

**Mục đích**: Xác minh backprop hoạt động

```python
# Sau loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm == 0:
            print(f"Gradient bằng 0: {name}")
        elif grad_norm > 100:
            print(f"Gradient explosion: {name} (norm={grad_norm:.2f})")
```

---

### Chiến lược 5: Log Mọi Thứ

**Mục đích**: Theo dõi chi tiết training

```python
# Trong vòng lặp training
import wandb  # Hoặc TensorBoard

wandb.init(project="braintumnet-debug")

for epoch in range(epochs):
    for i, batch in enumerate(train_loader):
        # ... code training ...

        # Log chi tiết
        wandb.log({
            'loss': loss.item(),
            'seg_loss': l_seg.item(),
            'cls_loss': l_cls.item(),
            'lr': optimizer.param_groups[0]['lr'],
            'grad_norm': sum(p.grad.norm().item() for p in model.parameters()),
            'weight_norm': sum(p.norm().item() for p in model.parameters()),
        })

        if i % 100 == 0:
            # Log ảnh
            wandb.log({
                'input': wandb.Image(batch["image"][0]),
                'mask': wandb.Image(batch["mask"][0]),
                'pred': wandb.Image(torch.sigmoid(seg[0]) > 0.5),
            })
```

---

## Tham Khảo Nhanh

### Checklist Cho Vấn Đề Training

- [ ] CUDA available? (`torch.cuda.is_available()`)
- [ ] Dữ liệu đã tiền xử lý? (kiểm tra `data/processed_*/`)
- [ ] Config đúng? (kiểm tra `exp_name`, `in_channels`, etc.)
- [ ] Đủ GPU memory? (giảm `batch_size` nếu OOM)
- [ ] Mixed precision bật? (`amp: true`)
- [ ] Learning rate hợp lý? (1e-5 đến 1e-3)
- [ ] Model đang cập nhật? (kiểm tra gradient)
- [ ] Data loading nhanh? (kiểm tra `num_workers`)
- [ ] Augmentation hợp lý? (không quá cực đoan)
- [ ] Checkpoint đang lưu? (kiểm tra `checkpoints/`)

### Điều Chỉnh Tham Số Thường Gặp

| Vấn đề | Tham số | Hướng |
|--------|---------|-------|
| OOM | `batch_size` | ↓ Giảm |
| OOM | `base` | ↓ Giảm |
| OOM | `img_size` | ↓ Giảm |
| Training chậm | `amp` | Bật |
| Training chậm | `workers` | ↑ Tăng |
| Overfitting | `weight_decay` | ↑ Tăng |
| Overfitting | `rotate_deg` | ↑ Tăng |
| Underfitting | `epochs` | ↑ Tăng |
| Underfitting | `base` | ↑ Tăng |
| Không ổn định | `lr` | ↓ Giảm |
| Không ổn định | Gradient clip | Thêm |

---

**Tiếp theo**: [[v_10_EXTENSION_GUIDE|Phần 10: Hướng Dẫn Mở Rộng →]]

**Quay lại**: [[v_08_RESULTS_ANALYSIS|← Phần 8: Phân Tích Kết Quả]] | [[v_TECHNICAL_REPORT_INDEX|Index]]
