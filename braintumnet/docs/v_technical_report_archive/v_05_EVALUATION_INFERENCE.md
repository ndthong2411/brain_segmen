# Phần 5: Đánh giá và Dự đoán

**Điều hướng**: [[v_TECHNICAL_REPORT_INDEX|← Quay lại Mục lục]]

---

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [Evaluator (evaluator.py)](#evaluator-evaluatorpy)
3. [Script dự đoán (predict.py)](#script-dự-đoán-predictpy)
4. [Test-Time Augmentation (TTA)](#test-time-augmentation-tta)
5. [Ensemble Predictions](#ensemble-predictions)
6. [Batch Inference](#batch-inference)
7. [Ví dụ sử dụng thực tế](#ví-dụ-sử-dụng-thực-tế)
8. [Hướng dẫn chỉnh sửa](#hướng-dẫn-chỉnh-sửa)

---

## Tổng quan

### Đánh giá vs Dự đoán (Inference)

**Đánh giá (Evaluation)**: Đo lường hiệu suất mô hình trên dữ liệu validation/test có ground truth
- Input: Ảnh + Nhãn
- Output: Các chỉ số (Dice, IoU, Accuracy, v.v.)
- Mục đích: Định lượng chất lượng mô hình

**Dự đoán (Inference)**: Áp dụng mô hình lên dữ liệu mới (không có ground truth)
- Input: Chỉ có ảnh
- Output: Dự đoán (masks, classes)
- Mục đích: Sử dụng lâm sàng, triển khai

### Các file quan trọng

| File | Mục đích | Dòng | Use Case |
|------|---------|-------|----------|
| `engine/evaluator.py` | Đánh giá toàn diện | 112 | Phân tích tập validation |
| `scripts/predict.py` | Dự đoán ảnh đơn | 107 | Triển khai lâm sàng |

---

## Evaluator (evaluator.py)

**File**: `src/braintumnet/engine/evaluator.py` (112 dòng)

Script này tính toán **tất cả các chỉ số** trên một fold validation: IoU, Dice, HD, HD95, Accuracy, F1, AUC.

### Giải thích code đầy đủ

#### Khởi tạo

```python
def evaluate(cfg: Dict, fold: int, ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = cfg["data"]["proc_root"]
    val_list = os.path.join(proc, f"split_val_fold{fold}.txt")
    ds = SliceDataset(proc, val_list, cfg["data"]["img_size"], train=False, in_channels=cfg["model"]["in_channels"])
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["workers"])
```

**Dòng 11-16**: Thiết lập dataloader

**Các điểm quan trọng**:
- `train=False`: Không augmentation (đánh giá phải deterministic)
- `shuffle=False`: Xử lý theo thứ tự (có thể tái tạo)
- Sử dụng cùng batch size với training (để hiệu quả)

---

```python
    model = BrainTumNet(in_ch=cfg["model"]["in_channels"], num_cls=cfg["model"]["num_classes_cls"],
                        base=cfg["model"]["base"], dim=cfg["model"]["dim"], patch=cfg["model"]["patch_size"],
                        depth=cfg["model"]["depth"], n_heads=cfg["model"]["n_heads"],
                        roi_stop_grad=cfg["model"]["roi_stop_grad"]).to(device)
    load_ckpt(model, ckpt_path, map_location=device)
    model.eval()
```

**Dòng 17-22**: Tải mô hình

**`model.eval()`**: Rất quan trọng cho đánh giá!
- Vô hiệu hóa dropout (deterministic)
- BatchNorm sử dụng running statistics (không phải batch stats)
- Đảm bảo khả năng tái tạo

**Tại sao dùng `load_ckpt` (không phải `load_training_state`)?**
- `load_ckpt`: Chỉ tải trọng số mô hình (nhẹ)
- `load_training_state`: Tải optimizer, scheduler, v.v. (không cần thiết cho eval)

---

#### Bộ tích lũy chỉ số (Metric Accumulators)

```python
    # Classification metrics
    y_true, y_pred, y_prob = [], [], []

    # Segmentation metrics (global)
    total_inter, total_union = 0.0, 0.0

    # Per-slice metrics for HD and HD95 (accumulated)
    hd_scores = []
    hd95_scores = []
```

**Dòng 24-32**: Khởi tạo các bộ tích lũy chỉ số

**Tại sao có các chiến lược khác nhau?**

1. **Classification**: Thu thập tất cả dự đoán, tính chỉ số một lần
   - Các hàm scikit-learn cần tất cả dữ liệu cùng lúc
   - Ví dụ: AUC-ROC cần tất cả xác suất

2. **Segmentation (IoU/Dice)**: Tích lũy intersection/union toàn cục
   - Tính trung bình toàn cục đúng (giải thích ở Phần 4)
   - Tiết kiệm bộ nhớ

3. **Hausdorff Distances**: Tính theo từng slice, tính trung bình sau
   - HD là chỉ số theo từng ảnh (không tổng hợp được như intersection)
   - Lưu tất cả giá trị để tính mean và std

---

#### Vòng lặp đánh giá

```python
    import torch.nn.functional as F
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"Evaluating Fold {fold}"):
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)
            lab = batch["label"].cpu().numpy()
            seg, cls = model(img)
```

**Dòng 34-40**: Forward pass

**`torch.no_grad()`**: Cần thiết cho đánh giá!
- Vô hiệu hóa tính toán gradient
- Tiết kiệm bộ nhớ (~50% giảm)
- Inference nhanh hơn

**Progress Bar**:
```
Evaluating Fold 0: 100%|████████████| 456/456 [02:34<00:00,  2.95it/s]
```

---

#### Chỉ số phân loại (Classification Metrics)

```python
            # Classification
            prob = F.softmax(cls, dim=1).cpu().numpy()
            y_true.extend(lab.tolist())
            y_pred.extend(prob.argmax(1).tolist())
            y_prob.extend(prob.tolist())
```

**Dòng 42-46**: Thu thập dự đoán phân loại

**Từng bước**:
```python
# Input: cls shape (B, 2) - raw logits

prob = F.softmax(cls, dim=1)
# Chuyển logits → xác suất
# prob shape: (B, 2), giá trị cộng lại bằng 1.0 mỗi hàng

prob = prob.cpu().numpy()
# Chuyển sang CPU và convert sang numpy

y_true.extend(lab.tolist())
# Nhãn ground truth: [0, 1, 0, 1, ...] (HGG/LGG)

y_pred.extend(prob.argmax(1).tolist())
# Nhãn dự đoán: argmax trên các classes
# Ví dụ: prob=[0.8, 0.2] → argmax=0 (HGG)

y_prob.extend(prob.tolist())
# Tất cả xác suất cho tính toán AUC
# Ví dụ: [[0.8, 0.2], [0.3, 0.7], ...]
```

**Tại sao lưu mọi thứ?**
- Không thể tính AUC dần dần
- Cần tất cả xác suất cùng lúc
- Bộ nhớ: ~1MB cho 10k mẫu

---

#### Chỉ số phân đoạn (Segmentation Metrics) - Toàn cục

```python
            # Segmentation (accumulate global metrics)
            inter, union = compute_intersection_union(seg, msk)
            total_inter += inter
            total_union += union
```

**Dòng 48-51**: Tích lũy chỉ số phân đoạn toàn cục

**Chỉ số toàn cục đúng**:
```python
# Batch 1: 100 slices
inter1, union1 = compute_intersection_union(seg1, msk1)
total_inter = 45000  # pixels
total_union = 50000  # pixels

# Batch 2: 100 slices
inter2, union2 = compute_intersection_union(seg2, msk2)
total_inter += 38000  # Bây giờ là 83000
total_union += 45000  # Bây giờ là 95000

# ... tất cả batches ...

# Chỉ số toàn cục cuối cùng:
iou = total_inter / (total_union - total_inter)
    = 83000 / (95000 - 83000)
    = 83000 / 12000
    = 0.691

dice = 2 * total_inter / total_union
     = 2 * 83000 / 95000
     = 0.874
```

---

#### Hausdorff Distance (Theo từng slice)

```python
            # Per-slice HD and HD95 (on CPU)
            pred_masks = binarize(seg).cpu().numpy()
            target_masks = msk.cpu().numpy()

            for pred_slice, target_slice in zip(pred_masks, target_masks):
                # Only compute HD/HD95 if there's tumor in ground truth
                if target_slice.sum() > 0:
                    metrics = compute_segmentation_metrics(
                        pred_slice.squeeze(),
                        target_slice.squeeze(),
                        compute_hd=True,
                        compute_hd95=True
                    )
                    if not np.isinf(metrics['hd']) and not np.isnan(metrics['hd']):
                        hd_scores.append(metrics['hd'])
                    if not np.isinf(metrics['hd95']) and not np.isnan(metrics['hd95']):
                        hd95_scores.append(metrics['hd95'])
```

**Dòng 53-69**: Tính Hausdorff distances theo từng slice

**Tại sao theo từng slice?**
- HD đo khoảng cách biên (chỉ số không gian)
- Không thể tổng hợp qua các ảnh như intersection/union
- Phải tính trên từng ảnh riêng lẻ

**Tại sao `if target_slice.sum() > 0`?**
- Bỏ qua các slice không có khối u
- HD không xác định nếu ground truth rỗng
- Ngăn chia cho 0 / giá trị inf

**Tại sao lọc inf/nan?**
```python
if not np.isinf(metrics['hd']) and not np.isnan(metrics['hd']):
```
- Các trường hợp edge: Dự đoán rỗng, lỗi số học
- Chỉ bao gồm giá trị HD hợp lệ trong trung bình
- Đảm bảo thống kê vững

**Ví dụ**:
```python
# Batch của 8 slices
pred_masks shape: (8, 1, 256, 256)
target_masks shape: (8, 1, 256, 256)

for i in range(8):
    pred = pred_masks[i].squeeze()   # (256, 256)
    target = target_masks[i].squeeze()  # (256, 256)

    if target.sum() > 0:  # Có khối u
        hd = compute_hausdorff_distance(pred, target)
        # Ví dụ: hd = 12.3 pixels
        hd_scores.append(12.3)

# Sau tất cả batches:
hd_scores = [12.3, 8.7, 15.2, ..., 10.9]  # 3456 giá trị
hd_mean = np.mean(hd_scores)  # 11.8 pixels
hd_std = np.std(hd_scores)    # 4.2 pixels
```

---

#### Tính toán chỉ số cuối cùng

```python
    # Compute classification metrics
    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
    acc, f1, auc = cls_metrics(y_true, y_pred, y_prob)

    # Compute segmentation metrics
    eps = 1e-6
    iou = total_inter / (total_union - total_inter + eps)
    dice = (2 * total_inter) / (total_union + eps)

    # Average HD and HD95
    hd_mean = np.mean(hd_scores) if len(hd_scores) > 0 else float('nan')
    hd95_mean = np.mean(hd95_scores) if len(hd95_scores) > 0 else float('nan')
    hd_std = np.std(hd_scores) if len(hd_scores) > 0 else float('nan')
    hd95_std = np.std(hd95_scores) if len(hd95_scores) > 0 else float('nan')
```

**Dòng 71-84**: Tính chỉ số cuối cùng

**Chỉ số phân loại**:
```python
acc, f1, auc = cls_metrics(y_true, y_pred, y_prob)
```
- Gọi các hàm scikit-learn
- Trả về accuracy, macro F1, AUC-ROC

**Chỉ số phân đoạn**:
```python
iou = total_inter / (total_union - total_inter + eps)
dice = (2 * total_inter) / (total_union + eps)
```
- Chỉ số toàn cục (tính trung bình đúng)
- `eps` ngăn chia cho 0

**Thống kê Hausdorff**:
```python
hd_mean = np.mean(hd_scores)
hd_std = np.std(hd_scores)
```
- Mean và standard deviation
- Báo cáo độ bất định trong độ chính xác biên

---

#### Kết quả đầu ra

```python
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS - Fold {fold}")
    print("=" * 70)
    print("\nSegmentation Metrics:")
    print(f"  IoU (Jaccard):        {iou:.4f}")
    print(f"  Dice (F1):            {dice:.4f}")
    print(f"  Hausdorff Distance:   {hd_mean:.2f} ± {hd_std:.2f} pixels")
    print(f"  HD95 (95th percentile): {hd95_mean:.2f} ± {hd95_std:.2f} pixels")
    print(f"  (HD computed on {len(hd_scores)} slices with tumor)")
    print("\nClassification Metrics:")
    print(f"  Accuracy:             {acc:.4f}")
    print(f"  F1 Score:             {f1:.4f}")
    print(f"  AUC-ROC:              {auc:.4f}")
    print("=" * 70 + "\n")
```

**Dòng 86-99**: In kết quả

**Ví dụ đầu ra**:
```
======================================================================
EVALUATION RESULTS - Fold 0
======================================================================

Segmentation Metrics:
  IoU (Jaccard):        0.8430
  Dice (F1):            0.9148
  Hausdorff Distance:   45.23 ± 18.76 pixels
  HD95 (95th percentile): 12.34 ± 5.67 pixels
  (HD computed on 3456 slices with tumor)

Classification Metrics:
  Accuracy:             0.9823
  F1 Score:             0.9812
  AUC-ROC:              0.9956
======================================================================
```

**Giải thích**:
- **IoU 0.843**: Độ chồng lấp xuất sắc (>0.7 là tốt cho medical imaging)
- **Dice 0.915**: Tương ứng với IoU 0.843
- **HD 45px**: Lỗi biên trường hợp xấu nhất (nhạy cảm với outliers)
- **HD95 12px**: Lỗi biên vững (bỏ qua 5% outliers)
- **Acc 0.982**: Độ chính xác phân loại 98.2%
- **AUC 0.996**: Phân tách HGG/LGG gần như hoàn hảo

---

### Cách sử dụng

```bash
# Đánh giá fold 0 với checkpoint tốt nhất
python -m braintumnet.engine.evaluator \
    --cfg configs/full_dataset_multimodal.yaml \
    --fold 0 \
    --ckpt checkpoints/braintumnet_best_fold0.pth
```

**Tích hợp trong Training**:
```python
# Sau khi training hoàn tất
from braintumnet.engine.evaluator import evaluate

results = evaluate(cfg, fold=0, ckpt_path="checkpoints/braintumnet_best_fold0.pth")
print(f"Final Dice: {results['dice']:.4f}")
```

---

## Script dự đoán (predict.py)

**File**: `scripts/predict.py` (107 dòng)

Script này thực hiện **inference trên một ảnh đơn** và trực quan hóa kết quả.

### Giải thích code đầy đủ

#### Dự đoán ảnh đơn

```python
def predict_single(model, img_path, img_size=256, device="cuda"):
    """Predict segmentation and classification for a single image."""
    # Load image
    img = Image.open(img_path).convert("L")
    img_resized = resize_pad_to_square(img, img_size, is_mask=False)
    img_tensor = to_tensor01(img_resized).unsqueeze(0).to(device)  # (1,1,H,W)

    # Predict
    model.eval()
    with torch.no_grad():
        seg_logits, cls_logits = model(img_tensor)
        seg_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()  # (H,W)
        cls_prob = torch.softmax(cls_logits, dim=1).squeeze().cpu().numpy()  # (num_classes,)
        cls_pred = cls_prob.argmax()

    return seg_prob, cls_pred, cls_prob
```

**Dòng 15-30**: Dự đoán trên ảnh đơn

**Xử lý từng bước**:

1. **Tải ảnh**:
```python
img = Image.open(img_path).convert("L")
# Chế độ "L" = grayscale (8-bit pixels)
# Ví dụ: FLAIR MRI slice
```

2. **Resize và Pad**:
```python
img_resized = resize_pad_to_square(img, img_size=256, is_mask=False)
# Xử lý kích thước input tùy ý
# Pad thành hình vuông (giữ tỷ lệ khung hình)
# is_mask=False → nội suy cho ảnh (không phải nearest neighbor)
```

**Ví dụ**:
```
Gốc: 240×240 → Resize thành 256×256 (pad 8px mỗi bên)
Gốc: 512×384 → Resize thành 256×192, pad thành 256×256
```

3. **Chuyển sang Tensor**:
```python
img_tensor = to_tensor01(img_resized).unsqueeze(0).to(device)
# to_tensor01: PIL Image → torch.Tensor, chuẩn hóa về [0, 1]
# unsqueeze(0): (1, 256, 256) → (1, 1, 256, 256) - thêm batch dim
# .to(device): Chuyển sang GPU
```

4. **Inference**:
```python
model.eval()
with torch.no_grad():
    seg_logits, cls_logits = model(img_tensor)
```
- `model.eval()`: Vô hiệu hóa dropout, dùng running batch norm stats
- `torch.no_grad()`: Không tính gradient (nhanh hơn, ít bộ nhớ hơn)

5. **Post-Processing**:
```python
seg_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
# (1, 1, 256, 256) → (256, 256)
# sigmoid: logits → xác suất [0, 1]
# squeeze: Bỏ batch và channel dims
# cpu().numpy(): Tensor → numpy array

cls_prob = torch.softmax(cls_logits, dim=1).squeeze().cpu().numpy()
# (1, 2) → (2,)
# softmax: logits → xác suất [0, 1], cộng lại bằng 1
# Ví dụ: [0.85, 0.15] - 85% HGG, 15% LGG

cls_pred = cls_prob.argmax()
# Lấy class dự đoán
# Ví dụ: argmax([0.85, 0.15]) = 0 (HGG)
```

**Giá trị trả về**:
- `seg_prob`: (256, 256) float array, giá trị [0, 1]
- `cls_pred`: Integer (0 hoặc 1)
- `cls_prob`: (2,) float array, xác suất

---

#### Trực quan hóa

```python
def visualize_prediction(img_path, seg_prob, cls_pred, cls_prob, save_path=None):
    """Visualize input image, predicted segmentation, and classification."""
    img = Image.open(img_path).convert("L")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Segmentation mask
    axes[1].imshow(seg_prob, cmap='hot')
    axes[1].set_title("Predicted Tumor Mask")
    axes[1].axis('off')

    # Binary segmentation
    seg_binary = (seg_prob > 0.5).astype(np.uint8)
    axes[2].imshow(img, cmap='gray')
    axes[2].imshow(seg_binary, cmap='Reds', alpha=0.4)
    axes[2].set_title(f"Overlay | Class: {cls_pred} ({cls_prob[cls_pred]:.2f})")
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved prediction to: {save_path}")
    else:
        plt.show()

    plt.close()
```

**Dòng 32-63**: Hàm trực quan hóa

**Trực quan hóa ba panel**:

1. **Ảnh gốc** (Trái):
```python
axes[0].imshow(img, cmap='gray')
```
- Hiển thị MRI slice input
- Colormap grayscale

2. **Mask dự đoán** (Giữa):
```python
axes[1].imshow(seg_prob, cmap='hot')
```
- Hiển thị heatmap xác suất
- Colormap 'hot': đen (0.0) → đỏ → vàng → trắng (1.0)
- Trực quan hóa độ tin cậy của mô hình

3. **Overlay** (Phải):
```python
seg_binary = (seg_prob > 0.5).astype(np.uint8)
axes[2].imshow(img, cmap='gray')  # Base layer
axes[2].imshow(seg_binary, cmap='Reds', alpha=0.4)  # Overlay
axes[2].set_title(f"Overlay | Class: {cls_pred} ({cls_prob[cls_pred]:.2f})")
```
- Binary mask overlay trên ảnh gốc
- `alpha=0.4`: Độ trong suốt 40%
- Hiển thị vị trí khối u + phân loại

**Ví dụ đầu ra**:
```
┌────────────────┬────────────────┬─────────────────────────────┐
│ Input Image    │ Predicted Mask │ Overlay | Class: 0 (0.85)  │
│                │                │                             │
│   [MRI slice]  │  [Heatmap]     │  [MRI + Red overlay]        │
│                │                │                             │
└────────────────┴────────────────┴─────────────────────────────┘
```

---

#### Hàm main

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="Path to config YAML")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    ap.add_argument("--img", type=str, required=True, help="Path to input image")
    ap.add_argument("--out", type=str, default=None, help="Output visualization path")
    args = ap.parse_args()

    # Load config
    cfg = load_yaml(args.cfg)

    # Build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = cfg["model"]
    model = BrainTumNet(
        in_ch=mcfg["in_channels"],
        num_cls=mcfg["num_classes_cls"],
        base=mcfg["base"],
        dim=mcfg["dim"],
        patch=mcfg["patch_size"],
        depth=mcfg["depth"],
        n_heads=mcfg["n_heads"],
        roi_stop_grad=mcfg["roi_stop_grad"]
    ).to(device)

    # Load checkpoint
    load_ckpt(model, args.ckpt, map_location=device)
    print(f"Loaded checkpoint: {args.ckpt}")

    # Predict
    seg_prob, cls_pred, cls_prob = predict_single(
        model, args.img, cfg["data"]["img_size"], device
    )

    print(f"Classification: {'HGG' if cls_pred == 0 else 'LGG'} (confidence: {cls_prob[cls_pred]:.4f})")
    print(f"Segmentation: mean={seg_prob.mean():.4f}, max={seg_prob.max():.4f}")

    # Visualize
    visualize_prediction(args.img, seg_prob, cls_pred, cls_prob, args.out)
```

**Dòng 65-103**: Giao diện dòng lệnh

**Cách sử dụng**:
```bash
python scripts/predict.py \
    --cfg configs/full_dataset_multimodal.yaml \
    --ckpt checkpoints/braintumnet_best_fold0.pth \
    --img data/processed_full_multimodal/images/BraTS20_001_0000_slice_075.png \
    --out predictions/result.png
```

**Đầu ra**:
```
Loaded checkpoint: checkpoints/braintumnet_best_fold0.pth
Classification: HGG (confidence: 0.8523)
Segmentation: mean=0.1234, max=0.9876
Saved prediction to: predictions/result.png
```

---

## Test-Time Augmentation (TTA)

**TTA là gì?**
- Áp dụng augmentation trong quá trình inference (không chỉ training)
- Tính trung bình dự đoán từ nhiều phiên bản augmented
- Cải thiện độ vững và độ chính xác

**TTA hoạt động như thế nào**:
```
Ảnh gốc
    ↓
┌───┴───┬───────┬───────┬───────┐
│       │       │       │       │
│ Không │ Flip  │ Rot   │ Flip  │
│ Aug   │ H     │ 90°   │ + Rot │
│       │       │       │       │
└───┬───┴───┬───┴───┬───┴───┬───┘
    ↓       ↓       ↓       ↓
  Pred1   Pred2   Pred3   Pred4
    │       │       │       │
    └───────┴───┬───┴───────┘
                ↓
         Trung bình dự đoán
                ↓
          Dự đoán cuối cùng
```

### Triển khai

**Thêm vào `predict.py`**:
```python
def predict_with_tta(model, img_path, img_size=256, device="cuda"):
    """
    Predict with Test-Time Augmentation.
    Augmentations: original, hflip, vflip, hflip+vflip
    """
    # Load image
    img = Image.open(img_path).convert("L")
    img_resized = resize_pad_to_square(img, img_size, is_mask=False)
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    model.eval()
    predictions = []

    with torch.no_grad():
        # Augmentation 1: Original
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
        seg, cls = model(img_tensor)
        predictions.append(torch.sigmoid(seg).cpu())

        # Augmentation 2: Horizontal flip
        img_hflip = torch.flip(img_tensor, [3])  # Flip width
        seg, cls = model(img_hflip)
        seg_hflip_back = torch.flip(torch.sigmoid(seg), [3])  # Flip back
        predictions.append(seg_hflip_back.cpu())

        # Augmentation 3: Vertical flip
        img_vflip = torch.flip(img_tensor, [2])  # Flip height
        seg, cls = model(img_vflip)
        seg_vflip_back = torch.flip(torch.sigmoid(seg), [2])  # Flip back
        predictions.append(seg_vflip_back.cpu())

        # Augmentation 4: Both flips
        img_hvflip = torch.flip(img_tensor, [2, 3])
        seg, cls = model(img_hvflip)
        seg_hvflip_back = torch.flip(torch.sigmoid(seg), [2, 3])
        predictions.append(seg_hvflip_back.cpu())

    # Average all predictions
    seg_prob = torch.stack(predictions).mean(dim=0).squeeze().numpy()

    # Classification from original (no augmentation for cls)
    with torch.no_grad():
        _, cls = model(img_tensor)
        cls_prob = torch.softmax(cls, dim=1).squeeze().cpu().numpy()
        cls_pred = cls_prob.argmax()

    return seg_prob, cls_pred, cls_prob
```

**Cách sử dụng**:
```python
# Thay thế predict_single bằng predict_with_tta
seg_prob, cls_pred, cls_prob = predict_with_tta(model, args.img, cfg["data"]["img_size"], device)
```

**Cải thiện dự kiến**:
- Dice: +0.5-1.5% (ví dụ: 0.915 → 0.925)
- HD95: -5-10% (biên tốt hơn)
- Chi phí: Chậm hơn 4× (4 forward passes)

---

## Ensemble Predictions

**Ensemble là gì?**
- Kết hợp dự đoán từ nhiều mô hình
- Các mô hình được train trên các folds khác nhau hoặc với các seeds khác nhau
- Giảm phương sai, cải thiện độ vững

### Triển khai

```python
def predict_ensemble(model_paths, cfg, img_path, device="cuda"):
    """
    Predict using ensemble of models.

    Args:
        model_paths: List of checkpoint paths
        cfg: Config dict
        img_path: Input image path
        device: 'cuda' or 'cpu'

    Returns:
        seg_prob, cls_pred, cls_prob (averaged)
    """
    # Load image once
    img = Image.open(img_path).convert("L")
    img_resized = resize_pad_to_square(img, cfg["data"]["img_size"], is_mask=False)
    img_tensor = to_tensor01(img_resized).unsqueeze(0).to(device)

    seg_probs = []
    cls_probs = []

    for ckpt_path in model_paths:
        # Build model
        mcfg = cfg["model"]
        model = BrainTumNet(
            in_ch=mcfg["in_channels"],
            num_cls=mcfg["num_classes_cls"],
            base=mcfg["base"],
            dim=mcfg["dim"],
            patch=mcfg["patch_size"],
            depth=mcfg["depth"],
            n_heads=mcfg["n_heads"],
            roi_stop_grad=mcfg["roi_stop_grad"]
        ).to(device)

        # Load checkpoint
        load_ckpt(model, ckpt_path, map_location=device)
        model.eval()

        # Predict
        with torch.no_grad():
            seg, cls = model(img_tensor)
            seg_probs.append(torch.sigmoid(seg).cpu().numpy())
            cls_probs.append(torch.softmax(cls, dim=1).cpu().numpy())

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Average predictions
    seg_prob = np.mean(seg_probs, axis=0).squeeze()
    cls_prob = np.mean(cls_probs, axis=0).squeeze()
    cls_pred = cls_prob.argmax()

    return seg_prob, cls_pred, cls_prob
```

**Cách sử dụng**:
```python
# Ensemble tất cả 5 folds
model_paths = [
    "checkpoints/braintumnet_best_fold0.pth",
    "checkpoints/braintumnet_best_fold1.pth",
    "checkpoints/braintumnet_best_fold2.pth",
    "checkpoints/braintumnet_best_fold3.pth",
    "checkpoints/braintumnet_best_fold4.pth",
]

seg_prob, cls_pred, cls_prob = predict_ensemble(model_paths, cfg, img_path, device)
```

**Cải thiện dự kiến**:
- Dice: +1-3% (ví dụ: 0.915 → 0.935)
- Vững hơn với outliers
- Chi phí: Chậm hơn 5× (5 mô hình)

**Kết hợp TTA + Ensemble**:
```python
# Mỗi mô hình với TTA, sau đó ensemble
# Chất lượng tốt nhất, nhưng chậm hơn 20× (5 mô hình × 4 augmentations)
```

---

## Batch Inference

Để xử lý nhiều ảnh hiệu quả:

```python
def predict_batch(model, image_paths, img_size=256, batch_size=16, device="cuda"):
    """
    Predict on batch of images.

    Args:
        model: BrainTumNet model
        image_paths: List of image paths
        img_size: Resize dimension
        batch_size: Batch size for inference
        device: 'cuda' or 'cpu'

    Returns:
        seg_probs: List of (H, W) arrays
        cls_preds: List of class predictions
        cls_probs: List of (num_classes,) probability arrays
    """
    model.eval()

    seg_probs = []
    cls_preds = []
    cls_probs_all = []

    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]

        # Load batch
        images = []
        for img_path in batch_paths:
            img = Image.open(img_path).convert("L")
            img_resized = resize_pad_to_square(img, img_size, is_mask=False)
            img_tensor = to_tensor01(img_resized)
            images.append(img_tensor)

        # Stack to batch
        batch_tensor = torch.stack(images).to(device)  # (B, 1, H, W)

        # Predict
        with torch.no_grad():
            seg, cls = model(batch_tensor)
            seg_prob = torch.sigmoid(seg).cpu().numpy()
            cls_prob = torch.softmax(cls, dim=1).cpu().numpy()

        # Collect results
        for j in range(len(batch_paths)):
            seg_probs.append(seg_prob[j].squeeze())
            cls_probs_all.append(cls_prob[j])
            cls_preds.append(cls_prob[j].argmax())

    return seg_probs, cls_preds, cls_probs_all
```

**Cách sử dụng**:
```python
# Xử lý 1000 ảnh
image_paths = [...]  # List của 1000 paths

seg_probs, cls_preds, cls_probs = predict_batch(
    model, image_paths, batch_size=16, device="cuda"
)

# Lưu kết quả
for i, (seg, cls) in enumerate(zip(seg_probs, cls_preds)):
    np.save(f"predictions/seg_{i:04d}.npy", seg)
    print(f"Image {i}: Class {cls}")
```

**Hiệu suất**:
- Batch size 1: 10 ảnh/giây
- Batch size 16: 45 ảnh/giây
- **Tăng tốc 4.5×!**

---

## Ví dụ sử dụng thực tế

### Ví dụ 1: Đánh giá tất cả các Folds

```python
import yaml
from braintumnet.engine.evaluator import evaluate

# Load config
with open("configs/full_dataset_multimodal.yaml") as f:
    cfg = yaml.safe_load(f)

# Evaluate all 5 folds
results_all = []
for fold in range(5):
    print(f"\n{'='*70}")
    print(f"Evaluating Fold {fold}")
    print(f"{'='*70}\n")

    ckpt_path = f"checkpoints/braintumnet_best_fold{fold}.pth"
    results = evaluate(cfg, fold, ckpt_path)
    results_all.append(results)

# Aggregate results
import numpy as np
dice_mean = np.mean([r['dice'] for r in results_all])
dice_std = np.std([r['dice'] for r in results_all])
iou_mean = np.mean([r['iou'] for r in results_all])
iou_std = np.std([r['iou'] for r in results_all])

print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS (5 Folds)")
print("="*70)
print(f"Dice:  {dice_mean:.4f} ± {dice_std:.4f}")
print(f"IoU:   {iou_mean:.4f} ± {iou_std:.4f}")
print("="*70)
```

**Đầu ra**:
```
======================================================================
CROSS-VALIDATION RESULTS (5 Folds)
======================================================================
Dice:  0.9148 ± 0.0023
IoU:   0.8430 ± 0.0031
======================================================================
```

---

### Ví dụ 2: Script triển khai lâm sàng

```python
#!/usr/bin/env python
"""
Script triển khai lâm sàng cho BrainTumNet.
Xử lý MRI bệnh nhân và tạo báo cáo.
"""

import sys
from pathlib import Path
import torch
import yaml
from PIL import Image
import numpy as np

# Add to path
sys.path.append(str(Path(__file__).parent / "src"))

from braintumnet.models.braintumnet import BrainTumNet
from braintumnet.utils.io import load_ckpt
from scripts.predict import predict_with_tta

def process_patient(patient_dir, model, cfg, device):
    """
    Xử lý tất cả slices cho một bệnh nhân.

    Returns:
        report: Dictionary với các phát hiện lâm sàng
    """
    patient_dir = Path(patient_dir)
    slices = sorted(patient_dir.glob("*.png"))

    results = []
    for slice_path in slices:
        seg_prob, cls_pred, cls_prob = predict_with_tta(
            model, str(slice_path), cfg["data"]["img_size"], device
        )
        results.append({
            'slice': slice_path.name,
            'tumor_volume': seg_prob.sum(),  # Pixel count
            'tumor_fraction': (seg_prob > 0.5).mean(),
            'max_confidence': seg_prob.max(),
            'grade': 'HGG' if cls_pred == 0 else 'LGG',
            'grade_confidence': cls_prob[cls_pred]
        })

    # Tạo báo cáo
    total_tumor_volume = sum(r['tumor_volume'] for r in results)
    dominant_grade = max(set(r['grade'] for r in results), key=lambda g: sum(1 for r in results if r['grade'] == g))
    avg_confidence = np.mean([r['grade_confidence'] for r in results])

    report = {
        'patient_id': patient_dir.name,
        'num_slices': len(results),
        'total_tumor_volume': total_tumor_volume,
        'dominant_grade': dominant_grade,
        'avg_confidence': avg_confidence,
        'slice_details': results
    }

    return report

def main():
    # Load config and model
    cfg = yaml.safe_load(open("configs/full_dataset_multimodal.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BrainTumNet(...).to(device)
    load_ckpt(model, "checkpoints/best_model.pth", device)

    # Process patient
    report = process_patient("data/patients/BraTS20_001", model, cfg, device)

    # In báo cáo
    print("\n" + "="*70)
    print(f"BÁO CÁO LÂM SÀNG: {report['patient_id']}")
    print("="*70)
    print(f"Độ mức độ chính: {report['dominant_grade']} ({report['avg_confidence']:.1%} độ tin cậy)")
    print(f"Thể tích khối u: {report['total_tumor_volume']:.0f} voxels")
    print(f"Slices đã phân tích: {report['num_slices']}")
    print("="*70)

if __name__ == "__main__":
    main()
```

---

## Hướng dẫn chỉnh sửa

### Thêm Rotation TTA

```python
def predict_with_rotation_tta(model, img_path, img_size=256, device="cuda"):
    """TTA với rotations: 0°, 90°, 180°, 270°"""
    # Load image
    img = Image.open(img_path).convert("L")
    img_resized = resize_pad_to_square(img, img_size, is_mask=False)
    img_tensor = to_tensor01(img_resized).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    predictions = []

    with torch.no_grad():
        for k in range(4):  # 0°, 90°, 180°, 270°
            # Rotate
            img_rot = torch.rot90(img_tensor, k=k, dims=[2, 3])

            # Predict
            seg, _ = model(img_rot)
            seg_prob = torch.sigmoid(seg)

            # Rotate back
            seg_back = torch.rot90(seg_prob, k=-k, dims=[2, 3])
            predictions.append(seg_back.cpu())

    # Average
    seg_prob = torch.stack(predictions).mean(dim=0).squeeze().numpy()

    return seg_prob
```

---

### Lưu dự đoán dưới dạng NIfTI

```python
import nibabel as nib

def save_as_nifti(seg_prob, output_path, threshold=0.5):
    """
    Lưu segmentation dưới dạng NIfTI file (định dạng ảnh y tế).

    Args:
        seg_prob: (H, W) hoặc (D, H, W) array
        output_path: Output .nii.gz path
        threshold: Ngưỡng nhị phân hóa
    """
    # Binarize
    seg_binary = (seg_prob > threshold).astype(np.uint8)

    # Tạo NIfTI image
    nifti_img = nib.Nifti1Image(seg_binary, affine=np.eye(4))

    # Lưu
    nib.save(nifti_img, output_path)
    print(f"Đã lưu NIfTI vào: {output_path}")
```

**Cách sử dụng**:
```python
seg_prob, _, _ = predict_single(model, img_path, device=device)
save_as_nifti(seg_prob, "predictions/patient001_seg.nii.gz")
```

---

**Tiếp theo**: [[v_06_UTILS_LOGGING|Phần 6: Utils và Logging →]]

**Quay lại**: [[v_04_TRAINING_SYSTEM|← Phần 4: Hệ thống Training]] | [[v_TECHNICAL_REPORT_INDEX|Mục lục]]
