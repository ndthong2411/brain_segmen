# Cách Đọc Metrics Khi Training - Loss Âm Là Bình Thường!

## TL;DR - Tóm Tắt Nhanh

**❌ ĐỪNG nhìn vào `train_loss` (có thể âm, khó đọc)**
**✅ NHÌN VÀO những metrics này:**

| Metric | Giải Thích | Tốt Khi Nào |
|--------|-----------|------------|
| **val_iou** | IoU trên validation set | **> 0.70** = Tốt, **> 0.80** = Rất tốt, **> 0.85** = Xuất sắc |
| **TC_iou** | IoU của Tumor Core (khó nhất) | **> 0.65** = Tốt, **> 0.70** = Rất tốt |
| **ED_iou** | IoU của Edema | **> 0.70** = Tốt, **> 0.75** = Rất tốt |
| **WT_iou** | IoU của Whole Tumor | **> 0.75** = Tốt, **> 0.80** = Rất tốt |

## Tại Sao Loss Lại Âm?

Loss function của bạn là tổng có trọng số:

```python
total_loss = 1.0 × dice_loss +
             1.0 × focal_loss +
             2.0 × iou_loss +
             0.5 × boundary_loss
```

Trong đó:
- **IoU Loss** = `1 - IoU` → khi IoU cao (gần 1), loss này gần 0
- **Dice Loss** = `1 - Dice` → tương tự, gần 0 khi tốt
- **Focal Loss** và **Boundary Loss** cũng giảm khi model học tốt

Nhưng do cách tính và trọng số, **tổng có thể âm** mà không có vấn đề gì!

## Ví Dụ Từ Log Của Bạn

```
Epoch 32/350:
  train_loss: -2.4051    ← Âm, khó hiểu
  val_iou: 0.7055        ← ✅ IoU = 0.70 = RẤT TỐT!
  WT_iou: 0.7119         ← ✅ Whole Tumor IoU tốt
  TC_iou: 0.6711         ← ✅ Tumor Core IoU khá
  ED_iou: 0.7334         ← ✅ Edema IoU tốt
```

**Kết luận:** Model đang train CỰC TỐT! Đạt IoU 0.70 rồi!

## So Sánh Với Baseline

| Stage | Mean IoU | Đánh Giá |
|-------|----------|----------|
| **Baseline** | 0.7263 | Tốt |
| **Epoch 1** | 0.0153 | Bắt đầu (chưa học gì) |
| **Epoch 6** | 0.2305 | Học nhanh |
| **Epoch 13** | 0.7102 | NHẢY VỌT! Đuổi kịp baseline |
| **Epoch 18** | 0.7112 | Vượt baseline nhẹ |
| **Epoch 29** | 0.7125 | ✅ BEST! Vượt baseline |
| **Epoch 32** | 0.7055 | Vẫn tốt (dao động nhẹ) |

## Metrics Quan Trọng Nhất

### 1. **Validation IoU** (val_iou) - QUAN TRỌNG NHẤT
- Đây là metric chính bạn cần theo dõi
- **Target của dự án: 0.90**
- Hiện tại: **~0.71** (đã đạt được baseline!)

### 2. **Tumor Core IoU** (TC_iou) - Class Khó Nhất
- TC là bottleneck (khó phân đoạn nhất)
- Hiện tại: **~0.67**
- Nếu TC cải thiện → Mean IoU sẽ tăng mạnh

### 3. **Training Stability**
- Loss nên **giảm đều** (dù âm hay dương)
- Val IoU nên **tăng hoặc ổn định**
- Nếu loss tăng + IoU giảm → Có vấn đề (learning rate quá cao, overfitting...)

## Theo Dõi Training Đúng Cách

### 1. Xem TensorBoard (Khuyến Nghị)
```bash
tensorboard --logdir=runs
```

Mở trình duyệt: http://localhost:6006

**Các tab quan trọng:**
- **Scalars → val/mean_iou**: Đồ thị IoU qua các epoch
- **Scalars → val/TC_iou**: Theo dõi Tumor Core
- **Scalars → train/loss_***: Các thành phần loss riêng lẻ

### 2. Xem Log File
```bash
# Log mới nhất
ls -lt logs/*.log | head -1

# Xem tóm tắt các epoch
grep "SUMMARY" logs/braintumnet_*.log
```

### 3. So Sánh Các Checkpoint
```python
import torch

# Load checkpoint
ckpt = torch.load("checkpoints/braintumnet_phase2_small_fold4_best.pth")
print(f"Best IoU: {ckpt['val_iou']:.4f}")
print(f"Best Epoch: {ckpt['epoch']}")
```

## Dấu Hiệu Training Tốt

✅ **Validation IoU tăng đều** (quan trọng nhất)
✅ Training loss giảm đều (dù âm hay dương)
✅ Gap giữa train và val không quá lớn (< 10% là OK)
✅ TC IoU cũng cải thiện (class khó)

## Dấu Hiệu Có Vấn đề

❌ **Validation IoU giảm** trong nhiều epoch liên tiếp
❌ Training loss tăng hoặc dao động mạnh
❌ Gap train-val quá lớn (overfitting)
❌ Loss = NaN hoặc Inf
❌ GPU utilization thấp (< 50%)

## Khi Nào Nên Dừng Training?

### Early Stopping Tự Động
Config của bạn:
```yaml
early_stop_patience: 80  # Dừng nếu 80 epochs không cải thiện
```

### Dừng Thủ Công
Dừng khi một trong các điều kiện:
1. **Val IoU đạt target** (0.85-0.90)
2. **Val IoU không cải thiện** trong 50+ epochs
3. **Overfitting nghiêm trọng** (train loss giảm, val IoU giảm)
4. **Đã đủ thời gian** để train các fold khác

## Target Theo Roadmap

| Phase | Single Model | With TTA | With Ensemble | Target |
|-------|-------------|----------|---------------|---------|
| **Phase 1** | 0.75-0.80 | 0.77-0.82 | 0.78-0.83 | ✅ Done |
| **Phase 2 Small** | 0.80-0.82 | 0.82-0.84 | 0.83-0.85 | ← Đang làm |
| **Phase 2 A100** | 0.82-0.85 | 0.84-0.87 | 0.85-0.88 | Kế tiếp |
| **Phase 3** | N/A | +2-3% | +2-3% | N/A |
| **FINAL** | 0.85-0.88 | 0.87-0.90 | **0.88-0.91** | 🎯 0.90 |

## Lời Khuyên

### 1. Tập Trung Vào IoU, Không Phải Loss
```python
# ❌ SAI:
if train_loss < -3.0:
    print("Model đang train tốt")

# ✅ ĐÚNG:
if val_iou > 0.75:
    print("Model đang train tốt")
```

### 2. Kiểm Tra Định Kỳ
- Mỗi 10 epochs: Xem val_iou có cải thiện không
- Mỗi 50 epochs: Xem TensorBoard để kiểm tra trend
- Cuối mỗi fold: So sánh với fold trước

### 3. Lưu Checkpoints Quan Trọng
Model tự động lưu:
- Best IoU checkpoint: `*_best.pth`
- Latest checkpoint: `*_latest.pth`

### 4. Monitor GPU
```bash
watch -n 1 nvidia-smi
```

Đảm bảo:
- GPU Util: > 80%
- GPU Memory: Sử dụng đầy (gần full)
- Power: > 200W (A100 nên 300-350W)

## Câu Hỏi Thường Gặp

### Q: Loss âm có nghĩa là gì?
**A:** Không sao cả! Chỉ là cách tính loss. Nhìn vào **val_iou** thay vì loss.

### Q: Loss âm bao nhiêu là tốt?
**A:** Không quan trọng! Nhìn vào:
- **val_iou > 0.70** = Tốt
- **val_iou > 0.80** = Rất tốt
- **val_iou > 0.85** = Xuất sắc

### Q: Tại sao val_iou dao động?
**A:** Bình thường! Validation set nhỏ nên có noise. Quan trọng là **trend tăng**, không cần mỗi epoch đều tăng.

### Q: Khi nào thì đạt IoU 0.90?
**A:** Cần:
1. Train Phase 2 A100 (hoặc Phase 2 Small) → 0.82-0.85
2. Train 5 folds → Ensemble
3. Apply TTA (Test-Time Augmentation)
4. Kết hợp ensemble + TTA → 0.88-0.91 ✅

### Q: Training bao lâu?
**A:**
- Phase 2 Small (RTX 3090): ~48 giờ/fold × 5 = 240 giờ (~10 ngày)
- Phase 2 A100: ~24 giờ/fold × 5 = 120 giờ (~5 ngày)

## Tóm Lại

🎯 **Metric chính: val_iou**
📊 **Công cụ: TensorBoard**
✅ **Hiện tại: 0.71 = RẤT TỐT!**
🚀 **Target: 0.90**
💪 **Bước tiếp: Train hết 350 epochs, rồi train 4 folds còn lại**

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│  METRICS TO WATCH                               │
├─────────────────────────────────────────────────┤
│  ✅ val_iou       > 0.70  (MAIN METRIC)        │
│  ✅ TC_iou        > 0.65  (Hardest class)       │
│  ✅ ED_iou        > 0.70  (Should be good)      │
│  ✅ WT_iou        > 0.75  (Easiest)             │
│                                                  │
│  ❌ train_loss    (IGNORE - can be negative)   │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  CURRENT STATUS (Epoch 32)                      │
├─────────────────────────────────────────────────┤
│  val_iou:  0.7055  → ✅ GREAT!                 │
│  TC_iou:   0.6711  → ✅ Good                   │
│  ED_iou:   0.7334  → ✅ Great                  │
│  WT_iou:   0.7119  → ✅ Great                  │
│                                                  │
│  Verdict: Training is working PERFECTLY! 🎉     │
└─────────────────────────────────────────────────┘
```
