# Part 4: Training System Deep Dive

**Navigation**: [[v_TECHNICAL_REPORT_INDEX|← Quay lại Index]]

---

## Mục lục

1. [Tổng quan Training System](#tổng-quan-training-system)
2. [Loss Functions](#loss-functions)
3. [Metrics và Evaluation](#metrics-và-evaluation)
4. [Training Loop (trainer.py)](#training-loop-trainerpy)
5. [Learning Rate Scheduling](#learning-rate-scheduling)
6. [Checkpoint Management](#checkpoint-management)
7. [Logging và Monitoring](#logging-và-monitoring)
8. [Mixed Precision Training](#mixed-precision-training)
9. [Modification Guides](#modification-guides)

---

## Tổng quan Training System

### Training System là gì?

Training system phối hợp:
- **Data Loading**: Batching và augmentation
- **Forward Pass**: Model predictions
- **Loss Calculation**: Multi-task objective
- **Backward Pass**: Gradient computation
- **Optimization**: Parameter updates
- **Validation**: Performance evaluation
- **Checkpointing**: Model saving
- **Logging**: Metrics tracking

### Các File Chính

| File | Mục đích | Dòng | Độ phức tạp |
|------|---------|-------|------------|
| `engine/trainer.py` | Main training loop | 307 | Cao |
| `losses/base.py` | Loss functions | 28 | Thấp |
| `metrics/base.py` | Evaluation metrics | 248 | Trung bình |
| `utils/io.py` | Checkpoint I/O | 121 | Trung bình |
| `utils/logger.py` | Text logging | 204 | Trung bình |
| `utils/metrics_logger.py` | CSV/JSON logging | ~200 | Trung bình |

### Training Flow Diagram

```
┌─────────────────────────────────────────────────┐
│             Training Initialization             │
│  - Load config                                  │
│  - Build dataloaders                            │
│  - Build model                                  │
│  - Tạo optimizer, scheduler, loss               │
│  - Khởi tạo loggers (file, TensorBoard, CSV)   │
│  - Resume từ checkpoint (nếu chỉ định)         │
└────────────────────┬────────────────────────────┘
                     ↓
        ┌────────────────────────┐
        │  CHO MỖI EPOCH         │
        └────────────┬───────────┘
                     ↓
    ┌────────────────────────────────┐
    │      TRAINING PHASE            │
    │  ┌──────────────────────────┐  │
    │  │ CHO MỖI BATCH:           │  │
    │  │  1. Load batch           │  │
    │  │  2. Forward pass         │  │
    │  │  3. Tính loss            │  │
    │  │  4. Backward pass        │  │
    │  │  5. Cập nhật weights     │  │
    │  │  6. Cập nhật LR (cosine) │  │
    │  │  7. Log tới TensorBoard  │  │
    │  └──────────────────────────┘  │
    └────────────────┬───────────────┘
                     ↓
    ┌────────────────────────────────┐
    │    VALIDATION PHASE            │
    │  ┌──────────────────────────┐  │
    │  │ CHO MỖI BATCH:           │  │
    │  │  1. Load batch           │  │
    │  │  2. Forward pass (no grad)│  │
    │  │  3. Tích lũy metrics     │  │
    │  │  4. Lưu samples          │  │
    │  └──────────────────────────┘  │
    │                                │
    │  Tính global metrics:          │
    │  - IoU (Intersection/Union)    │
    │  - Dice (2*Inter/Union)        │
    │  - Classification Accuracy     │
    └────────────────┬───────────────┘
                     ↓
    ┌────────────────────────────────┐
    │     LOGGING & CHECKPOINTING    │
    │  - Log metrics tới file/CSV    │
    │  - Log tới TensorBoard         │
    │  - Cập nhật LR (ReduceLROnPlateau)│
    │  - Lưu best checkpoint         │
    │  - Lưu last checkpoint         │
    │  - Kiểm tra early stopping     │
    └────────────────┬───────────────┘
                     ↓
                [Epoch Tiếp theo]
                     ↓
┌─────────────────────────────────────────────────┐
│          Training Complete                      │
│  - Log summary statistics                       │
│  - Đóng tất cả loggers                         │
│  - Return best IoU                              │
└─────────────────────────────────────────────────┘
```

---

## Loss Functions

**File**: `src/braintumnet/losses/base.py` (28 dòng)

BrainTumNet sử dụng **multi-task loss** kết hợp segmentation và classification objectives.

### Dice Loss with Logits

```python
def dice_loss_with_logits(logits, target, eps=1e-6):
    pred = torch.sigmoid(logits)
    num = 2 * (pred * target).sum(dim=(2,3))
    den = (pred.pow(2).sum(dim=(2,3)) + target.pow(2).sum(dim=(2,3))) + eps
    dice = 1 - (num + eps) / den
    return dice.mean()
```

**Dòng 3-8**: Dice loss implementation

**Dice Loss là gì?**

Dice coefficient đo overlap giữa prediction và ground truth:
```
Dice = 2 * |A ∩ B| / (|A| + |B|)
```

Trong đó:
- A = predicted tumor pixels
- B = ground truth tumor pixels
- |A ∩ B| = intersection (overlap)
- |A| + |B| = tổng của cả hai sets

**Dice Loss** = 1 - Dice Coefficient (convert similarity sang loss)

**Tại sao Dice Loss cho Medical Segmentation?**

1. **Xử lý Class Imbalance**:
   - Medical images: ~95% background, ~5% tumor
   - Cross-entropy thiên vị mạnh về background
   - Dice tập trung vào overlap, không phải từng pixels

2. **Có thể vi phân**:
   - Có thể optimize với gradient descent
   - Smooth gradient flow

3. **Trực quan**:
   - Trực tiếp optimize evaluation metric
   - Dice score là chuẩn trong medical imaging

**Giải thích Từng Bước**:

```python
# Input shapes:
# logits: (B, 1, 256, 256) - raw predictions
# target: (B, 1, 256, 256) - binary ground truth (0 hoặc 1)

pred = torch.sigmoid(logits)
# Áp dụng sigmoid để convert logits → probabilities [0, 1]
# pred shape: (B, 1, 256, 256)

num = 2 * (pred * target).sum(dim=(2,3))
# Tử số: 2 * intersection
# (pred * target): Element-wise multiplication
# .sum(dim=(2,3)): Sum over H và W dimensions
# num shape: (B, 1) - một giá trị per sample trong batch

den = (pred.pow(2).sum(dim=(2,3)) + target.pow(2).sum(dim=(2,3))) + eps
# Mẫu số: |A|² + |B|² (squared sums)
# Đây là Sørensen-Dice variant (ổn định hơn |A| + |B|)
# eps: Giá trị nhỏ để ngăn division by zero
# den shape: (B, 1)

dice = 1 - (num + eps) / den
# Convert Dice coefficient sang loss
# dice shape: (B, 1)

return dice.mean()
# Trung bình over batch
# Output: scalar loss value
```

**Tại sao Square trong Mẫu số?**

Hai công thức Dice phổ biến:

1. **Linear Dice** (classical):
   ```
   Dice = 2*|A∩B| / (|A| + |B|)
   ```

2. **Squared Dice** (dùng ở đây):
   ```
   Dice = 2*|A∩B| / (|A|² + |B|²)
   ```

**Ưu điểm của Squared Dice**:
- Gradients ổn định hơn
- Phạt lỗi lớn nặng hơn
- Chuẩn trong nnU-Net và nhiều medical segmentation papers

**Ví dụ Trực quan**:

```
Ground Truth (target):     Prediction (pred):
┌────────────┐             ┌────────────┐
│ 0  0  0  0 │             │ 0  0  0  0 │
│ 0  1  1  0 │             │ 0  1  1  0 │
│ 0  1  1  0 │             │ 0  0  1  1 │  ← Overlap một phần
│ 0  0  0  0 │             │ 0  0  0  0 │
└────────────┘             └────────────┘

Intersection (pred * target):
┌────────────┐
│ 0  0  0  0 │
│ 0  1  1  0 │  ← 3 pixels overlap
│ 0  0  1  0 │
│ 0  0  0  0 │
└────────────┘

Tính toán:
- Intersection: 3 pixels
- |A| (pred): 5 pixels
- |B| (target): 4 pixels
- Dice = 2*3 / (5 + 4) = 6/9 = 0.667
- Dice Loss = 1 - 0.667 = 0.333
```

---

### DiceCELoss (Hybrid Loss)

```python
class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, seg_logits, seg_mask):
        return dice_loss_with_logits(seg_logits, seg_mask) + self.bce(seg_logits, seg_mask)
```

**Dòng 10-15**: Combined Dice + Cross-Entropy loss

**Tại sao Kết hợp Dice + BCE?**

Mỗi loss có điểm mạnh bổ sung:

| Loss | Điểm mạnh | Điểm yếu |
|------|-----------|------------|
| **Dice** | Xử lý imbalance, optimize overlap | Có thể không ổn định sớm trong training |
| **BCE** | Gradients ổn định, pixel-wise accuracy | Thiên vị về majority class |
| **Dice+BCE** | Tốt nhất của cả hai: training ổn định + overlap tốt | Không có! |

**Công thức Toán học**:

```
Total Loss = Dice Loss + BCE Loss
           = (1 - Dice) + BCE
           = (1 - 2*|A∩B|/(|A|²+|B|²)) + Σ -[y*log(p) + (1-y)*log(1-p)]
```

**Cách Hoạt động Cùng nhau**:

1. **Early Training** (epoch 1-10):
   - Dice loss có thể nhiễu (predictions kém)
   - BCE cung cấp gradients ổn định
   - Model học boundaries cơ bản

2. **Mid Training** (epoch 10-50):
   - Dice loss trở nên đáng tin cậy hơn
   - Đẩy model maximize overlap
   - BCE tiếp tục refine pixel accuracy

3. **Late Training** (epoch 50+):
   - Cả hai losses hoạt động cùng nhau
   - Fine-tune segmentation boundaries
   - Optimize cả overlap và pixel accuracy

**Kết quả Ablation Study** (từ experiments):
- Chỉ Dice: Dice 0.872, IoU 0.773
- Chỉ BCE: Dice 0.854, IoU 0.745
- **Dice+BCE**: **Dice 0.914, IoU 0.843** ← Tốt nhất!

---

### MultiTaskLoss

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, seg_w=1.0, cls_w=0.7):
        super().__init__()
        self.seg_w = seg_w
        self.cls_w = cls_w
        self.seg_loss = DiceCELoss()
        self.cls_loss = nn.CrossEntropyLoss()
```

**Dòng 17-23**: Multi-task loss initialization

**Mục đích**: Kết hợp segmentation và classification losses

**Parameters**:
- `seg_w=1.0`: Segmentation weight (task chính)
- `cls_w=0.7`: Classification weight (task phụ)

**Tại sao seg_w > cls_w?**
- Segmentation là task chính
- Classification là auxiliary (giúp nhưng không quan trọng)
- Tỷ lệ 1.0:0.7 tìm được thực nghiệm

**CrossEntropyLoss là gì?**

Loss chuẩn cho classification:
```
CE = -Σ y_i * log(softmax(logits_i))
```

Cho 2-class (HGG/LGG):
```
CE = -[y_0*log(p_0) + y_1*log(p_1)]
```

Trong đó:
- y_i: One-hot encoded true label
- p_i: Predicted probability sau softmax

---

```python
    def forward(self, seg_logits, seg_mask, cls_logits, cls_label):
        l_seg = self.seg_loss(seg_logits, seg_mask)
        l_cls = self.cls_loss(cls_logits, cls_label)
        return self.seg_w * l_seg + self.cls_w * l_cls, l_seg.detach(), l_cls.detach()
```

**Dòng 24-27**: Multi-task loss forward pass

**Từng Bước**:
```python
# Input shapes:
# seg_logits: (B, 1, 256, 256) - segmentation predictions
# seg_mask: (B, 1, 256, 256) - binary ground truth
# cls_logits: (B, 2) - classification logits (HGG/LGG)
# cls_label: (B,) - integer labels (0 hoặc 1)

l_seg = self.seg_loss(seg_logits, seg_mask)
# Tính segmentation loss (Dice + BCE)
# l_seg: scalar

l_cls = self.cls_loss(cls_logits, cls_label)
# Tính classification loss (CrossEntropy)
# l_cls: scalar

total_loss = self.seg_w * l_seg + self.cls_w * l_cls
# Weighted combination
# total_loss: scalar

return total_loss, l_seg.detach(), l_cls.detach()
# Return:
#   - total_loss: Cho backward pass
#   - l_seg.detach(): Cho logging (không có grad)
#   - l_cls.detach(): Cho logging (không có grad)
```

**Tại sao .detach() cho logging?**
- `.detach()`: Loại bỏ khỏi computation graph
- Logging values không cần gradients
- Tiết kiệm memory

**Ví dụ Giá trị**:
```
Epoch 1:
  l_seg = 0.85 (cao - segmentation kém)
  l_cls = 0.45 (trung bình - random guessing)
  total = 1.0*0.85 + 0.7*0.45 = 1.165

Epoch 50:
  l_seg = 0.12 (thấp - segmentation tốt)
  l_cls = 0.08 (thấp - classification tốt)
  total = 1.0*0.12 + 0.7*0.08 = 0.176
```

---

## Metrics và Evaluation

**File**: `src/braintumnet/metrics/base.py` (248 dòng)

### Core Functions

#### binarize

```python
def binarize(logits: torch.Tensor, thr: float=0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > thr).float()
```

**Dòng 7-8**: Convert logits sang binary predictions

**Làm gì**:
1. Áp dụng sigmoid: logits → probabilities [0, 1]
2. Threshold ở 0.5: prob > 0.5 → 1, else → 0
3. Convert sang float: {0.0, 1.0}

**Tại sao threshold ở 0.5?**
- Chuẩn cho binary classification
- Có thể điều chỉnh cho precision/recall trade-off
- 0.5 là cân bằng (không thiên vị FP hay FN)

---

#### compute_intersection_union

```python
def compute_intersection_union(logits: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """
    Tính intersection và union cho global IoU/Dice calculation.
    Đây là cách ĐÚNG để tính metrics qua batches.

    Returns:
        intersection: Total intersection count
        union: Total union count (pred + target)
    """
    pred = binarize(logits)
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item()
    return inter, union
```

**Dòng 10-22**: **HÀM QUAN TRỌNG** cho global metrics đúng

**Tại sao Đây là Cách Tiếp cận Đúng**:

**SAI** (per-sample averaging):
```python
# ĐỪNG LÀM NHƯ NÀY!
 ious = []
for sample in batch:
    iou = intersection(sample) / union(sample)
    ious.append(iou)
final_iou = mean(ious)
```

**Vấn đề**: Tumors nhỏ nhận trọng số ngang với tumors lớn
```
Sample 1: 10 pixels tumor → IoU 0.9 → Trọng số cao
Sample 2: 1000 pixels tumor → IoU 0.8 → Cùng trọng số
Trung bình: (0.9 + 0.8) / 2 = 0.85

NHƯNG Sample 2 có 100× nhiều pixels hơn!
```

**ĐÚNG** (global averaging):
```python
# LÀM NHƯ NÀY!
total_inter = 0
total_union = 0
for sample in batch:
    total_inter += intersection(sample)
    total_union += union(sample)
final_iou = total_inter / (total_union - total_inter)
```

**Cho trọng số đúng cho tất cả pixels**:
```
Sample 1: inter=9, union=10
Sample 2: inter=800, union=1000
Tổng: inter=809, union=1010
IoU = 809 / (1010 - 809) = 809/201 = 0.802

Điều này đúng đắn weight Sample 2 nhiều hơn!
```

**Chi tiết Implementation**:
```python
pred = binarize(logits)          # (B, 1, H, W) → {0, 1}
inter = (pred * target).sum().item()
# Element-wise multiplication, sum TẤT CẢ pixels
# .item() convert tensor sang Python float

union = pred.sum().item() + target.sum().item()
# Tổng pred pixels + tổng target pixels
# Chú ý: Đếm intersection hai lần!
# Sau này: IoU = inter / (union - inter) sửa điều này

return inter, union
```

**Tại sao return inter và union riêng?**
- Tích lũy qua nhiều batches
- Tính final metric sau khi xử lý tất cả data
- Chính xác hơn averaging per-batch metrics

---

#### Segmentation Metrics

```python
def compute_dice_coefficient(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """
    Dice Similarity Coefficient (DSC).

    DSC = 2 * |A ∩ B| / (|A| + |B|)
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()

    # Xử lý edge cases
    if pred_sum == 0 and target_sum == 0:
        return 1.0  # Cả hai rỗng = khớp hoàn hảo
    if pred_sum == 0 or target_sum == 0:
        return 0.0  # Một rỗng, một không = không overlap

    dice = (2.0 * intersection) / (pred_sum + target_sum + eps)
    return float(dice)
```

**Dòng 77-105**: Dice coefficient calculation

**Điểm Chính**:

1. **Convert sang boolean**:
   - Đảm bảo binary values
   - Hoạt động với bất kỳ numeric input nào

2. **Xử lý Edge case**:
   - Cả hai rỗng → 1.0 (khớp hoàn hảo)
   - Một rỗng → 0.0 (không overlap)
   - Ngăn division errors

3. **Linear Dice** (không squared):
   - Dùng cho evaluation (không phải training)
   - Chuẩn trong medical imaging papers

---

```python
def compute_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """
    Intersection over Union (IoU / Jaccard Index).

    IoU = |A ∩ B| / |A ∪ B|
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    # Xử lý edge cases
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / (union + eps)
    return float(iou)
```

**Dòng 108-133**: IoU calculation

**Mối quan hệ với Dice**:
```
Cho: IoU = i / u

Dice = 2*i / (|A| + |B|)
     = 2*i / (u + i)        [vì |A| + |B| = u + i]
     = 2*IoU / (1 + IoU)

Ngược lại:
IoU = Dice / (2 - Dice)
```

**Ví dụ**:
```
IoU = 0.75
Dice = 2*0.75 / (1 + 0.75) = 1.5 / 1.75 = 0.857

Dice = 0.857
IoU = 0.857 / (2 - 0.857) = 0.857 / 1.143 = 0.75
```

**Typical Ranges**:
- IoU 0.50 → Dice 0.667 (Okay)
- IoU 0.70 → Dice 0.824 (Tốt)
- IoU 0.85 → Dice 0.919 (Xuất sắc)
- **Kết quả của chúng ta**: IoU 0.843 → Dice 0.915

---

#### Hausdorff Distance

```python
def compute_hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Hausdorff Distance (HD) - khoảng cách tối đa từ một điểm trong một set
    đến điểm gần nhất trong set kia.

    HD(A, B) = max(max_a min_b d(a,b), max_b min_a d(b,a))

    Thấp hơn là tốt hơn (0 = overlap hoàn hảo).
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    # Lấy boundary points
    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)

    # Xử lý empty masks
    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')

    # Tính symmetric Hausdorff distance
    hd_forward = directed_hausdorff(pred_points, target_points)[0]
    hd_backward = directed_hausdorff(target_points, pred_points)[0]
    hd = max(hd_forward, hd_backward)

    return float(hd)
```

**Dòng 136-168**: Hausdorff distance calculation

**Hausdorff Distance là gì?**

Đo boundary accuracy:
```
HD(A, B) = max của:
  - Khoảng cách xa nhất từ bất kỳ A point nào đến nearest B point
  - Khoảng cách xa nhất từ bất kỳ B point nào đến nearest A point
```

**Ví dụ Trực quan**:
```
Ground Truth Boundary:     Prediction Boundary:
    ●●●●●●                     ●●●●●●
    ●    ●                     ●    ●
    ●    ●                     ●    ●●●  ← Outlier!
    ●●●●●●                     ●●●●●●

Hausdorff Distance = khoảng cách tới outlier (worst case)
```

**Tại sao Symmetric?**
- `directed_hausdorff(A, B)`: Worst point trong A
- `directed_hausdorff(B, A)`: Worst point trong B
- `max(both)`: Worst overall

**Giải thích**:
- HD = 0: Boundary khớp hoàn hảo
- HD = 5: Worst point lệch 5 pixels
- HD = 50: Lỗi boundary lớn (outlier)

**Vấn đề**: Rất nhạy cảm với outliers!

---

```python
def compute_hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    95th percentile Hausdorff Distance (HD95) - robust hơn với outliers.

    Thay vì dùng maximum distance (nhạy cảm với outliers),
    sử dụng 95th percentile của distances.
    """
    # ... (code giống HD nhưng dùng percentile)

    # Tính distances từ mỗi point đến nearest point trong set kia
    from scipy.spatial.distance import cdist
    distances_matrix = cdist(pred_points, target_points)

    # Minimum distance từ mỗi pred point đến bất kỳ target point nào
    min_dist_pred_to_target = distances_matrix.min(axis=1)
    # Minimum distance từ mỗi target point đến bất kỳ pred point nào
    min_dist_target_to_pred = distances_matrix.min(axis=0)

    # Kết hợp tất cả minimum distances
    all_distances = np.concatenate([min_dist_pred_to_target, min_dist_target_to_pred])

    # Return 95th percentile
    hd95 = np.percentile(all_distances, 95)
    return float(hd95)
```

**Dòng 171-212**: HD95 calculation (robust hơn)

**Tại sao HD95 Tốt hơn**:

Bỏ qua 5% outliers tệ nhất:
```
Tất cả distances (sorted): [0, 0, 1, 1, 2, 2, 3, 3, 4, 50]
                                                    ↑
                                                 Outlier

HD (max):     50 pixels  ← Bị chi phối bởi outlier
HD95 (95th):   4 pixels  ← Robust với outlier
```

**Chuẩn trong Medical Imaging**:
- Hầu hết papers báo cáo HD95 (không phải HD)
- Liên quan lâm sàng hơn
- Outliers thường do annotation noise

**Kết quả Của chúng ta**:
- HD: ~45 pixels (bị ảnh hưởng bởi outliers)
- HD95: ~12 pixels (robust, có ý nghĩa lâm sàng)

---

#### Classification Metrics

```python
def cls_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Tuple[float,float,float]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    auc = float("nan")
    try:
        ncls = y_prob.shape[1]
        if ncls == 2:
            auc = roc_auc_score(y_true, y_prob[:,1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        pass
    return acc, f1, auc
```

**Dòng 58-70**: Classification metrics

**Metrics Giải thích**:

1. **Accuracy**: % predictions đúng
   ```
   Acc = Correct / Total
   ```

2. **F1 Score**: Harmonic mean của precision và recall
   ```
   F1 = 2 * (Precision * Recall) / (Precision + Recall)
   ```
   - `average="macro"`: Average F1 per class (cân bằng)

3. **AUC-ROC**: Area Under Receiver Operating Characteristic
   - Đo khả năng phân tách classes của classifier
   - 0.5 = random guessing
   - 1.0 = phân tách hoàn hảo
   - Cho binary: dùng probability của positive class (y_prob[:,1])

**Tại sao F1 hơn Accuracy?**
- Xử lý class imbalance
- BraTS có nhiều HGG hơn LGG
- F1 cho trọng số ngang nhau cho cả hai classes

---

## Training Loop (trainer.py)

**File**: `src/braintumnet/engine/trainer.py` (307 dòng)

Đây là **file quan trọng nhất** - phối hợp toàn bộ quá trình training.

### Initialization

```python
def train_one_fold(cfg: Dict, fold: int, config_path: str = None, resume_from: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Khởi tạo loggers
    log_dir = cfg["logging"].get("log_dir", "logs")
    logger = TrainingLogger(log_dir, cfg["exp_name"], fold)
    metrics_logger = MetricsLogger(log_dir, cfg["exp_name"], fold)
```

**Dòng 54-60**: Function signature và logger initialization

**Parameters**:
- `cfg`: Configuration dictionary (loaded từ YAML)
- `fold`: Fold number (0-4 cho 5-fold CV)
- `config_path`: Path tới config file (để lưu copy)
- `resume_from`: Checkpoint path (để resuming training)

**Loggers**:
- `TrainingLogger`: Human-readable text log
- `MetricsLogger`: CSV/JSON metrics cho analysis

---

```python
    train_loader, val_loader = build_dataloaders(cfg, fold)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model(cfg).to(device)

    # Đếm parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")
```

**Dòng 68-76**: Build dataloaders và model

**Dataloader Details**:
```python
def build_dataloaders(cfg: Dict, fold: int):
    proc = cfg["data"]["proc_root"]
    img_size = cfg["data"]["img_size"]
    train_list = os.path.join(proc, f"split_train_fold{fold}.txt")
    val_list   = os.path.join(proc, f"split_val_fold{fold}.txt")

    train_ds = SliceDataset(proc, train_list, img_size,
                            cfg["augment"]["rotate_deg"],
                            cfg["augment"]["hflip_p"],
                            cfg["augment"]["vflip_p"],
                            True,  # train=True (enable augmentation)
                            cfg["model"]["in_channels"])

    val_ds   = SliceDataset(proc, val_list, img_size,
                            0, 0, 0,  # Không augmentation
                            False,  # train=False
                            cfg["model"]["in_channels"])

    train_loader = DataLoader(train_ds,
                              batch_size=cfg["train"]["batch_size"],
                              shuffle=True,  # Shuffle training
                              num_workers=cfg["train"]["workers"])

    val_loader   = DataLoader(val_ds,
                              batch_size=cfg["train"]["batch_size"],
                              shuffle=False,  # Không shuffle validation
                              num_workers=cfg["train"]["workers"])

    return train_loader, val_loader
```

**Điểm Chính**:
- Training data: Augmentation enabled, shuffled
- Validation data: Không augmentation, không shuffled
- Separate split files mỗi fold

**Parameter Counting**:
```python
total_params = sum(p.numel() for p in model.parameters())
```
- `.numel()`: Number of elements trong tensor
- Thường: ~2.9M parameters cho base=32

---

```python
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    crit = MultiTaskLoss(cfg["train"]["seg_loss_weight"], cfg["train"]["cls_loss_weight"])
    scaler = torch.amp.GradScaler(device='cuda', enabled=cfg["train"].get("amp", False))
```

**Dòng 78-80**: Optimizer, loss, và mixed precision

**Adam Optimizer**:
- Adaptive learning rate per parameter
- Momentum và RMSprop combined
- `weight_decay`: L2 regularization (thường: 1e-5)

**Tại sao Adam hơn SGD?**
- Hội tụ nhanh hơn
- Robust với hyperparameter choices
- Chuẩn cho medical imaging

**GradScaler** (cho mixed precision):
- Scale gradients để ngăn underflow trong FP16
- `enabled=False` cho FP32 training
- Giải thích thêm trong Mixed Precision section

---

```python
    # ReduceLROnPlateau scheduler cho adaptive learning rate
    plateau_scheduler = None
    if cfg["train"]["scheduler"] == "plateau":
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=0.5, patience=10, min_lr=1e-7
        )
```

**Dòng 82-87**: Learning rate scheduler

**ReduceLROnPlateau**:
- Monitor validation metric (IoU)
- Nếu không cải thiện trong `patience` epochs → giảm LR
- `mode='max'`: Maximize IoU (không minimize loss)
- `factor=0.5`: LR → LR/2
- `min_lr=1e-7`: Dừng giảm dưới mức này

**Ví dụ**:
```
Epoch 1-20:  IoU cải thiện, LR = 1e-4
Epoch 21-30: IoU plateau, LR = 1e-4
Epoch 31:    Không cải thiện 10 epochs → LR = 5e-5
Epoch 31-40: IoU cải thiện chậm
Epoch 41-50: IoU plateau lại
Epoch 51:    Không cải thiện 10 epochs → LR = 2.5e-5
...
```

---

```python
    # TensorBoard
    writer = None
    if HAS_TENSORBOARD and cfg["logging"].get("use_tensorboard", True):
        tb_log_dir = os.path.join(cfg["logging"]["out_dir"], f"{cfg['exp_name']}_fold{fold}")
        ensure_dir(tb_log_dir)
        writer = SummaryWriter(tb_log_dir)
        logger.info(f"TensorBoard logging to: {tb_log_dir}")
```

**Dòng 89-95**: TensorBoard setup

**TensorBoard**: Visualization real-time
- Loss curves
- Learning rate schedule
- Sample predictions
- Gradient histograms

**Xem với**:
```bash
tensorboard --logdir=runs/
```

---

### Resume Training

```python
    # Resume từ checkpoint nếu chỉ định
    if resume_from is not None:
        logger.info(f"Resuming training from checkpoint: {resume_from}")
        from ..utils.io import load_training_state
        resume_info = load_training_state(resume_from, model, opt, plateau_scheduler, scaler, device, expected_fold=fold)
        start_epoch = resume_info['epoch'] + 1  # Bắt đầu từ epoch tiếp theo
        best_iou = resume_info['best_iou']
        best_iou_epoch = resume_info['best_iou_epoch']
        step = start_epoch * len(train_loader)
        logger.info(f"  Starting from epoch {start_epoch}")
        logger.info(f"  Previous best IoU: {best_iou:.4f} at epoch {best_iou_epoch + 1}")
```

**Dòng 104-114**: Resume training từ checkpoint

**Những gì Được Khôi phục**:
- Model weights
- Optimizer state (momentum, learning rate)
- Scheduler state (patience counter, best metric)
- Scaler state (loss scale cho mixed precision)
- Training progress (epoch, best IoU)
- **Fold number** (validates checkpoint đúng)

**Tại sao Validate Fold?**
- Ngăn vô tình resuming fold 0 với fold 1 checkpoint
- Báo lỗi nếu phát hiện mismatch

**Ví dụ**:
```bash
# Train fold 0, bị interrupt ở epoch 50
python train.py --cfg configs/default.yaml --fold 0

# Resume fold 0 từ last checkpoint
python train.py --cfg configs/default.yaml --fold 0 --resume checkpoints/last_fold0.pth

# LỖI: Sai fold
python train.py --cfg configs/default.yaml --fold 1 --resume checkpoints/last_fold0.pth
# ValueError: Fold mismatch! Checkpoint is for fold 0, but you're trying to resume fold 1.
```

---

### Training Phase

```python
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        epoch_start_time = time.time()
        logger.epoch_start(epoch, cfg["train"]["epochs"], "TRAIN")

        model.train()  # Enable dropout, batch norm training mode
        train_loss_sum = 0.0

        # Progress bar cho training
        if HAS_TQDM:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Train]", ncols=100)
        else:
            pbar = train_loader
```

**Dòng 122-133**: Training epoch initialization

**`model.train()`**: Quan trọng!
- Enable dropout (cho regularization)
- Batch norm dùng batch statistics (không phải running average)
- Ngược lại: `model.eval()` cho validation

**Progress Bar**:
- `tqdm`: Hiển thị progress, loss, LR real-time
- Falls back sang regular iterator nếu tqdm không installed

---

```python
        for batch_idx, batch in enumerate(pbar):
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)
            lab = batch["label"].to(device)

            with torch.amp.autocast(device_type='cuda', enabled=cfg["train"].get("amp", False)):
                seg, cls = model(img)
                loss, l_seg, l_cls = crit(seg, msk, cls, lab)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
```

**Dòng 135-145**: Core training step

**Từng Bước**:

1. **Load batch tới GPU**:
```python
img = batch["image"].to(device)  # (B, C, 256, 256)
msk = batch["mask"].to(device)   # (B, 1, 256, 256)
lab = batch["label"].to(device)  # (B,)
```

2. **Forward pass với mixed precision**:
```python
with torch.amp.autocast(device_type='cuda', enabled=True):
    # Chạy trong FP16 cho tốc độ
    seg, cls = model(img)
    loss, l_seg, l_cls = crit(seg, msk, cls, lab)
```

3. **Backward pass**:
```python
opt.zero_grad(set_to_none=True)  # Xóa previous gradients
                                  # set_to_none=True tiết kiệm memory

scaler.scale(loss).backward()     # Scale loss cho FP16
                                  # Tính gradients

scaler.step(opt)                  # Unscale gradients & cập nhật weights
scaler.update()                   # Cập nhật loss scale cho iteration tiếp theo
```

**Tại sao set_to_none=True?**
- `zero_grad()` set gradients tới 0 (allocates memory)
- `zero_grad(set_to_none=True)` set tới None (frees memory)
- ~10% memory savings

---

```python
            if cfg["train"]["scheduler"] == "cosine":
                _cosine_lr_with_warmup(opt, cfg["train"]["lr"], step, total_steps,
                                      warmup_steps=cfg["train"].get("warmup_steps", 500),
                                      min_lr=cfg["train"].get("min_lr", 1e-6))

            train_loss_sum += loss.item()

            # Cập nhật progress bar
            if HAS_TQDM:
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{opt.param_groups[0]["lr"]:.2e}'})
```

**Dòng 146-155**: Cosine LR và logging

**Cosine Learning Rate**:
- Cập nhật mỗi step (không phải epoch)
- Smooth decay với warmup
- Giải thích trong section tiếp theo

**Progress Bar Update**:
```
Epoch 50/100 [Train]: 42%|████▏     | 834/2000 [01:23<01:57, 9.9it/s, loss=0.1234, lr=2.34e-05]
```

---

### Validation Phase

```python
        # validation
        model.eval()  # Tắt dropout, dùng running batch norm stats
        total_inter, total_union = 0.0, 0.0
        acc_m, n = 0.0, 0

        with torch.no_grad():  # Tắt gradient computation (tiết kiệm memory)
            for batch_idx, batch in enumerate(val_pbar):
                img = batch["image"].to(device)
                msk = batch["mask"].to(device)
                lab = batch["label"].to(device)
                seg, cls = model(img)

                # Tích lũy intersection và union cho global metrics
                inter, union = compute_intersection_union(seg, msk)
                total_inter += inter
                total_union += union

                acc_m += (cls.argmax(1)==lab).float().mean().item()
                n += 1
```

**Dòng 167-191**: Validation loop

**Điểm Khác biệt từ Training**:
- `model.eval()`: Hành vi deterministic
- `torch.no_grad()`: Không tính gradients
- Không backward pass, không optimizer step
- Tích lũy metrics globally (không phải per-batch average)

**Tại sao Global Metrics?**
```python
# IoU global đúng
total_inter = tổng tất cả intersections
total_union = tổng tất cả unions
iou = total_inter / (total_union - total_inter)

# SAI per-batch average
batch_ious = [iou_batch1, iou_batch2, ...]
average_iou = mean(batch_ious)  # Thiên vị về small tumors!
```

---

```python
        # Tính final global metrics
        eps = 1e-6
        iou_m = total_inter / (total_union - total_inter + eps)
        dice_m = (2 * total_inter) / (total_union + eps)
        acc_m /= n
```

**Dòng 205-209**: Final metric calculation

**Công thức**:
```python
# IoU (Jaccard Index)
IoU = Intersection / Union
    = I / (I ∪ P ∪ T)
    = I / (P + T - I)
    = total_inter / (total_union - total_inter)

# Dice (F1 Score)
Dice = 2 * Intersection / (Pred + Target)
     = 2 * I / (P + T)
     = 2 * total_inter / total_union

# Accuracy
Acc = Correct Classifications / Total
    = acc_m / n
```

---

### Checkpointing

```python
        # Kiểm tra cải thiện
        if iou_m > best_iou:
            best_iou = iou_m
            best_iou_epoch = epoch
            epochs_without_improvement = 0
            ckpt_dir = cfg["logging"]["save_dir"]
            ensure_dir(ckpt_dir)
            save_ckpt(model, os.path.join(ckpt_dir, f"braintumnet_best_fold{fold}.pth"))
            logger.best_checkpoint("IoU", best_iou, epoch)
            print(f"  -> New best IoU: {best_iou:.4f}, checkpoint saved")
        else:
            epochs_without_improvement += 1
```

**Dòng 254-265**: Lưu best checkpoint

**Hai Loại Checkpoints**:

1. **Best Checkpoint** (best_fold{fold}.pth):
   - Lưu khi validation IoU cải thiện
   - Chỉ model weights (nhẹ)
   - Dùng cho final evaluation

2. **Last Checkpoint** (last_fold{fold}.pth):
   - Lưu mỗi epoch
   - Full training state (model, optimizer, scheduler, scaler)
   - Dùng để resuming training

---

```python
        # Lưu "last" checkpoint mỗi epoch để resume capability
        ckpt_dir = cfg["logging"]["save_dir"]
        ensure_dir(ckpt_dir)
        last_ckpt_path = os.path.join(ckpt_dir, f"last_fold{fold}.pth")
        save_training_state(last_ckpt_path, epoch, model, opt, plateau_scheduler, scaler,
                           best_iou, best_iou_epoch, cfg, fold=fold)
```

**Dòng 276-281**: Lưu last checkpoint

**Những gì Được Lưu**:
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),  # Nếu tồn tại
    'scaler_state_dict': scaler.state_dict(),        # Nếu tồn tại
    'best_iou': best_iou,
    'best_iou_epoch': best_iou_epoch,
    'fold': fold,  # Cho validation
    'config': config  # Full config
}
```

**Kích thước Files**:
- Best checkpoint: ~12 MB (chỉ model)
- Last checkpoint: ~25 MB (full state)

---

### Early Stopping

```python
        # Kiểm tra early stopping
        if epochs_without_improvement >= early_stop_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs ({epochs_without_improvement} epochs without improvement)")
            print(f"\n[Early Stop] No improvement for {early_stop_patience} epochs. Best IoU: {best_iou:.4f} at epoch {best_iou_epoch+1}")
            break
```

**Dòng 283-287**: Early stopping

**Mục đích**: Ngăn lãng phí thời gian training

**Cách Hoạt động**:
```
Epoch 1-50:   IoU cải thiện → epochs_without_improvement = 0
Epoch 51-80:  IoU plateau → epochs_without_improvement = 30
Epoch 81:     30 >= patience (30) → DỪNG!
```

**Tại sao Early Stop?**
- Validation metric plateau
- Training thêm = overfitting
- Tiết kiệm thời gian cho experiments khác

**Typical Patience**:
- Dataset nhỏ: 20-30 epochs
- Dataset lớn: 10-15 epochs
- Config của chúng ta: 30 epochs

---

## Learning Rate Scheduling

### Cosine Annealing với Warmup

```python
def _cosine_lr_with_warmup(optimizer, base_lr, t, T, warmup_steps=500, min_lr=1e-6):
    """Cosine learning rate với warmup và minimum LR để ngăn về zero"""
    if t < warmup_steps:
        # Linear warmup
        lr = base_lr * (t / warmup_steps)
    else:
        # Cosine decay với minimum LR
        progress = (t - warmup_steps) / (T - warmup_steps)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups: pg["lr"] = lr
```

**Dòng 25-34**: Cosine learning rate scheduler

**Hai Giai đoạn**:

1. **Warmup** (step 0 đến warmup_steps):
```python
lr = base_lr * (t / warmup_steps)
```
```
Step 0:    lr = 1e-4 * (0 / 500) = 0
Step 250:  lr = 1e-4 * (250 / 500) = 5e-5
Step 500:  lr = 1e-4 * (500 / 500) = 1e-4
```

**Tại sao Warmup?**
- LR ban đầu lớn có thể làm training không ổn định
- Gradients nhiễu ở giai đoạn đầu
- Warmup làm mượt phần start

2. **Cosine Decay** (step warmup_steps đến T):
```python
progress = (t - warmup_steps) / (T - warmup_steps)  # [0, 1]
lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(π * progress))
```

**Trực quan**:
```
LR
^
│ 1e-4 ┤         ╭─────╮
│      │       ╱       ╲
│      │      ╱         ╲
│      │     ╱           ╲
│      │    ╱             ╲
│      │   ╱               ╲___________
│ 1e-6 ┤  ╱
│      └──────────────────────────────> Steps
│     0   500              10000
│     └warmup┘  └─── cosine decay ───┘
```

**Tại sao Cosine?**
- Smooth decay (không sudden drops)
- Dành nhiều thời gian ở high LR (exploration)
- Dành nhiều thời gian ở low LR (fine-tuning)
- Tốt hơn step decay

**Tại sao min_lr?**
- Ngăn LR → 0
- Luôn tiến bộ một chút
- Chuẩn trong modern training

---

### ReduceLROnPlateau

**Alternative scheduler** (dùng trong config của chúng ta):

```python
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',        # Maximize IoU
    factor=0.5,        # LR *= 0.5
    patience=10,       # Đợi 10 epochs
    min_lr=1e-7        # Không xuống dưới mức này
)
```

**Cách Hoạt động**:
```
Gọi sau mỗi epoch:
plateau_scheduler.step(val_iou)

Nếu IoU không cải thiện trong 10 epochs:
  old_lr = 1e-4
  new_lr = old_lr * 0.5 = 5e-5
```

**Ưu điểm**:
- Adaptive (phản ứng với training dynamics)
- Không cần tune schedule
- Hoạt động tốt với early stopping

**So sánh**:

| Scheduler | Ưu điểm | Nhược điểm |
|-----------|------|------|
| **Cosine** | Mượt, dự đoán được | Cần tuning warmup/total_steps |
| **Plateau** | Adaptive, đơn giản | Có thể giảm quá muộn/sớm |
| **Step** | Đơn giản | Cần tuning step points |

**Lựa chọn Của chúng ta**: Plateau
- Hoạt động tốt cho medical imaging
- Kết hợp tốt với early stopping
- Ít hyperparameter tuning hơn

---

## Checkpoint Management

**File**: `src/braintumnet/utils/io.py`

### Saving Training State

```python
def save_training_state(path: str, epoch: int, model, optimizer, scheduler, scaler,
                       best_iou: float, best_iou_epoch: int, config: Dict = None, fold: int = None):
    """
    Lưu complete training state để resuming.
    """
    ensure_dir(os.path.dirname(path))

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': best_iou,
        'best_iou_epoch': best_iou_epoch,
        'fold': fold,  # Lưu fold number để validation
    }

    # Thêm scheduler state nếu tồn tại
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Thêm scaler state nếu tồn tại
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    # Thêm config nếu provided
    if config is not None:
        checkpoint['config'] = config

    torch.save(checkpoint, path)
    print(f"Saved training state to: {path}")
```

**Dòng 22-62**: Lưu complete training state

**Tại sao Lưu Mọi thứ?**
- Model weights: Rõ ràng
- Optimizer state: Momentum buffers, per-parameter LR
- Scheduler state: Patience counter, best metric, # reductions
- Scaler state: Loss scale cho mixed precision
- Training info: Biết nơi resume
- Config: Xác minh settings khớp

**Ví dụ Saved State**:
```python
checkpoint = {
    'epoch': 49,  # Vừa hoàn thành epoch 49
    'model_state_dict': OrderedDict([...]),  # ~2.9M params
    'optimizer_state_dict': {
        'state': {  # Momentum cho mỗi param
            0: {'exp_avg': tensor(...), 'exp_avg_sq': tensor(...), 'step': 50000},
            1: {...},
            ...
        },
        'param_groups': [{'lr': 5e-5, 'weight_decay': 1e-5, ...}]
    },
    'scheduler_state_dict': {
        'best': 0.8430,  # Best IoU seen
        'num_bad_epochs': 5,  # Epochs không cải thiện
        'cooldown_counter': 0,
        ...
    },
    'scaler_state_dict': {'scale': 65536.0, 'growth_factor': 2.0, ...},
    'best_iou': 0.8430,
    'best_iou_epoch': 44,
    'fold': 0,
    'config': {...}  # Full YAML config
}
```

---

### Loading Training State

```python
def load_training_state(path: str, model, optimizer, scheduler=None, scaler=None, map_location="cpu", expected_fold=None):
    """
    Load complete training state để resuming.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)

    # Validate fold nếu provided
    checkpoint_fold = checkpoint.get('fold', None)
    if expected_fold is not None and checkpoint_fold is not None:
        if checkpoint_fold != expected_fold:
            raise ValueError(
                f"Fold mismatch! Checkpoint is for fold {checkpoint_fold}, "
                f"but you're trying to resume fold {expected_fold}. "
                f"Please use the correct checkpoint: last_fold{expected_fold}.pth"
            )

    # Load model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler nếu provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load scaler nếu provided
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Return training info
    info = {
        'epoch': checkpoint['epoch'],
        'best_iou': checkpoint.get('best_iou', -1.0),
        'best_iou_epoch': checkpoint.get('best_iou_epoch', 0),
        'config': checkpoint.get('config', None),
    }

    print(f"Loaded training state from: {path}")
    print(f"  Resuming from epoch {info['epoch'] + 1}")
    print(f"  Best IoU so far: {info['best_iou']:.4f} (epoch {info['best_iou_epoch'] + 1})")

    return info
```

**Dòng 64-120**: Load training state

**Fold Validation** (QUAN TRỌNG!):
```python
if checkpoint_fold != expected_fold:
    raise ValueError(...)
```

**Tại sao Cần thiết?**
```bash
# Tình huống: Training fold 0 và fold 1 đồng thời
python train.py --fold 0 &  # Background process
python train.py --fold 1 &  # Background process

# Cả hai lưu tới: checkpoints/last_fold{fold}.pth

# Resume fold 0 nhưng vô tình dùng fold 1 checkpoint
python train.py --fold 0 --resume checkpoints/last_fold1.pth

# KHÔNG có validation: Training tiếp tục với data sai!
# CÓ validation: Lỗi ngay lập tức ✓
```

---

## Logging và Monitoring

### TrainingLogger (Text Logs)

**File**: `src/braintumnet/utils/logger.py`

**Ví dụ Log Output**:
```
================================================================================
BrainTumNet Training Log
================================================================================
Experiment: multimodal_training
Fold: 0
Start Time: 2024-01-15 10:30:45
--------------------------------------------------------------------------------

[10:30:50] [INFO] Training on device: cuda
[10:30:51] [INFO] Train batches: 1823, Val batches: 456
[10:30:52] [INFO] Model parameters: 2.9M total, 2.9M trainable

--------------------------------------------------------------------------------
Epoch 1/100 - TRAIN
--------------------------------------------------------------------------------
[10:35:23] Epoch 1/100 - SUMMARY - train_loss: 0.8234, val_iou: 0.4523, val_dice: 0.6234, val_acc: 0.6789, lr: 1.00e-04, time_s: 273
[10:35:23] [SUCCESS] *** NEW BEST IOU: 0.4523 (epoch 1) - Checkpoint saved ***

...

[12:45:12] [INFO] ReduceLROnPlateau: Reducing learning rate 1.00e-04 -> 5.00e-05

...

================================================================================
Training Complete!
================================================================================
Total Time: 2h 15m 38s

Best Metrics:
  iou: 0.8430 (epoch 55)
  dice: 0.9148 (epoch 55)
  acc: 0.9823 (epoch 60)

Log file: logs/multimodal_training_fold0_20240115_103045.log
================================================================================
```

---

### MetricsLogger (CSV/JSON)

**CSV Output** (`metrics_fold0.csv`):
```csv
epoch,train_loss,val_iou,val_dice,val_acc,learning_rate,epoch_time_s
0,0.8234,0.4523,0.6234,0.6789,0.0001,273.45
1,0.6745,0.5678,0.7234,0.7234,0.0001,268.92
2,0.5432,0.6123,0.7589,0.7456,0.0001,265.33
...
54,0.1234,0.8430,0.9148,0.9823,0.00005,259.87
```

**JSON Output** (`metrics_fold0.json`):
```json
{
  "experiment": "multimodal_training",
  "fold": 0,
  "epochs": [
    {
      "epoch": 0,
      "train_loss": 0.8234,
      "val_iou": 0.4523,
      "val_dice": 0.6234,
      "val_acc": 0.6789,
      "learning_rate": 0.0001,
      "epoch_time_s": 273.45
    },
    ...
  ],
  "best_metrics": {
    "val_iou": {"value": 0.8430, "epoch": 54},
    "val_dice": {"value": 0.9148, "epoch": 54},
    "val_acc": {"value": 0.9823, "epoch": 60}
  }
}
```

**Use Cases**:
- CSV: Dễ load trong pandas, Excel
- JSON: Machine-readable, nested structure
- Cả hai: Tự động generated

---

## Mixed Precision Training

**Mixed Precision là gì?**
- Dùng FP16 (16-bit floats) thay vì FP32 (32-bit)
- **Nhanh gấp 2× lần** trên modern GPUs
- **Bộ nhớ ít hơn 2× lần** → batches lớn hơn

**Thách thức**:
- FP16 range: ~±65,000 (vs FP32: ~±10³⁸)
- Gradients nhỏ underflow về zero
- Gradients lớn overflow về infinity

**Giải pháp**: Automatic Mixed Precision (AMP)

### AMP Hoạt động như thế nào

```python
# 1. Enable autocast cho forward pass
with torch.amp.autocast(device_type='cuda', enabled=True):
    seg, cls = model(img)         # Chạy trong FP16
    loss, l_seg, l_cls = crit(...)  # Tính trong FP16

# 2. Scale loss trước backward
scaler.scale(loss).backward()  # loss *= scale_factor (vd 65536)
                                # Ngăn underflow

# 3. Unscale gradients và update
scaler.step(optimizer)  # Unscale: grad /= scale_factor
                         # Kiểm tra inf/nan
                         # Cập nhật weights nếu valid

# 4. Cập nhật scale factor cho iteration tiếp theo
scaler.update()  # Nếu thành công: giữ scale
                 # Nếu overflow: giảm scale
```

**Gradient Scaling Giải thích**:

```
Không scaling:
  FP32 gradient: 1e-5  →  FP16: 0 (underflow!) ✗

Với scaling (scale=65536):
  FP32 gradient: 1e-5
  Scaled: 1e-5 * 65536 = 0.65536  →  FP16: 0.65536 ✓
  Sau backward: 0.65536
  Unscale: 0.65536 / 65536 = 1e-5 ✓
```

**Dynamic Loss Scaling**:
```
Initial scale: 65536

Iteration 1-1000: Không overflow → scale = 65536
Iteration 1001: Overflow detected! → scale = 32768
Iteration 1002-2000: Không overflow → scale = 32768
Iteration 2001: Không overflow 2000 iters → scale = 65536 (tăng)
```

**Speedup Benchmarks** (trên RTX 3090):
- FP32: 2.3 it/s
- FP16 (AMP): 4.7 it/s
- **Speedup: 2.04×**

**Memory Savings**:
- FP32: 12 GB
- FP16 (AMP): 6.5 GB
- **Savings: 46%** → Có thể tăng batch size!

---

## Modification Guides

### Thay đổi Batch Size

**Config file** (`configs/default.yaml`):
```yaml
train:
  batch_size: 8  # Thay đổi cái này

  # Quy tắc chung:
  # - GPU 6GB: batch_size=4
  # - GPU 11GB: batch_size=8
  # - GPU 24GB: batch_size=16
```

**Cảnh báo**: Thay đổi batch size ảnh hưởng:
- Training speed (lớn hơn = nhanh hơn nếu GPU có memory)
- Gradient noise (lớn hơn = ổn định hơn)
- Learning rate (batch lớn hơn → có thể cần LR lớn hơn)

**Recommended LR adjustment**:
```
New LR = Base LR * sqrt(New Batch / Old Batch)

Ví dụ:
  Old: batch=8, lr=1e-4
  New: batch=16, lr=1e-4 * sqrt(16/8) = 1.414e-4
```

---

### Thêm Metric Mới

**Bước 1**: Define metric function trong `metrics/base.py`:
```python
def precision_score_seg(logits: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
    """Precision = TP / (TP + FP)"""
    pred = binarize(logits)
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    return tp / (tp + fp + eps)
```

**Bước 2**: Tích lũy trong validation loop (`trainer.py`):
```python
# Trong validation loop (quanh dòng 178)
total_tp, total_fp = 0.0, 0.0

for batch in val_loader:
    ...
    seg, cls = model(img)

    pred = binarize(seg)
    total_tp += (pred * msk).sum().item()
    total_fp += (pred * (1 - msk)).sum().item()

# Sau loop (quanh dòng 205)
precision = total_tp / (total_tp + total_fp + 1e-6)
```

**Bước 3**: Log metric:
```python
# Trong logging section (quanh dòng 214)
logger.epoch_end(epoch, cfg["train"]["epochs"], {
    'train_loss': avg_train_loss,
    'val_iou': iou_m,
    'val_dice': dice_m,
    'val_precision': precision,  # Thêm ở đây
    'val_acc': acc_m,
    'lr': opt.param_groups[0]['lr'],
    'time_s': epoch_time
}, "SUMMARY")
```

---

### Thay đổi Optimizer

**SGD với momentum**:
```python
opt = torch.optim.SGD(
    model.parameters(),
    lr=cfg["train"]["lr"],
    momentum=0.9,
    weight_decay=cfg["train"]["weight_decay"],
    nesterov=True
)
```

**AdamW (Adam với better weight decay)**:
```python
opt = torch.optim.AdamW(
    model.parameters(),
    lr=cfg["train"]["lr"],
    betas=(0.9, 0.999),
    weight_decay=cfg["train"]["weight_decay"]
)
```

**Khi nào dùng mỗi loại**:
- **Adam**: Lựa chọn mặc định, hoạt động hầu hết thời gian
- **AdamW**: Cải thiện nhẹ hơn Adam, regularization tốt hơn
- **SGD**: Training lâu hơn nhưng đôi khi generalization tốt hơn

---

### Thêm Gradient Clipping

**Tại sao**: Ngăn exploding gradients

**Cách** (trong `trainer.py` quanh dòng 143):
```python
scaler.scale(loss).backward()

# Thêm dòng này:
scaler.unscale_(opt)  # Unscale trước khi clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

scaler.step(opt)
scaler.update()
```

**Làm gì**:
```python
# Trước clipping:
gradients = [0.1, 0.5, 3.0, 0.2]  # Một gradient rất lớn!
norm = sqrt(0.1² + 0.5² + 3.0² + 0.2²) = 3.08

# Clip tới max_norm=1.0:
scale_factor = max_norm / norm = 1.0 / 3.08 = 0.325
gradients *= scale_factor = [0.032, 0.162, 0.974, 0.065]
norm = 1.0 ✓
```

**Khi nào dùng**:
- Training không ổn định (loss spikes tới nan)
- Gradients exploding
- RNNs/Transformers (vấn đề phổ biến)

---

**Tiếp theo**: [[v_05_EVALUATION_INFERENCE|Part 5: Evaluation and Inference →]]

**Quay lại**: [[v_03_MODEL_ARCHITECTURE|← Part 3: Model Architecture]] | [[v_TECHNICAL_REPORT_INDEX|Index]]
