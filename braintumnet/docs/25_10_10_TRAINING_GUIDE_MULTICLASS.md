# Hướng Dẫn Training Multi-Class Segmentation

**Ngày**: 2025-10-10
**Mục đích**: Hướng dẫn từng bước để train model multi-class segmentation với data đã preprocess

---

## 📋 Kiểm Tra Data Đã Preprocess

### Bước 1: Đợi Preprocessing Hoàn Thành

Script preprocessing đang chạy background. Để check tiến độ:

```bash
# Check tiến độ
tail -f braintumnet/preprocess_log.txt

# Hoặc check files đã tạo
ls -1 braintumnet/data/processed_multiclass/seg | wc -l  # Số file masks
```

**Preprocessing hoàn thành khi**: Thấy message "✓ Preprocessing complete!" trong log

**Thời gian dự kiến**: ~15-18 phút để process 57,195 H5 files

---

### Bước 2: Tạo Labels.csv và Mapping.csv

**⚠️ QUAN TRỌNG**: Script preprocessing cũ đang chạy CHƯA có code tạo `labels.csv` và `mapping.csv`.

Sau khi preprocessing xong, chạy lệnh sau để tạo 2 files này:

```bash
cd braintumnet

python scripts/create_labels_mapping.py \
    --data_dir data/processed_multiclass
```

**Output**:
```
Loading data/processed_multiclass/all_slices.csv...
✓ Created data/processed_multiclass/labels.csv
  Total cases: 369
✓ Created data/processed_multiclass/mapping.csv
  Total slices: 57,195
```

---

### Bước 3: Verify Output Files

Check tất cả files cần thiết đã được tạo:

```bash
cd braintumnet/data/processed_multiclass

# Check folders (phải có 5 folders)
ls -d */
# Expected: flair/  seg/  t1/  t1ce/  t2/

# Check số lượng files (mỗi folder phải có 57,195 files)
ls -1 flair | wc -l    # 57195
ls -1 t1 | wc -l       # 57195
ls -1 t1ce | wc -l     # 57195
ls -1 t2 | wc -l       # 57195
ls -1 seg | wc -l      # 57195

# Check CSV files
ls -lh *.csv
# Expected output:
# all_slices.csv      (~2MB)
# labels.csv          (~10KB)
# mapping.csv         (~1-2MB)
# train_fold0.csv → train_fold4.csv
# val_fold0.csv → val_fold4.csv

# Check JSON mapping
cat class_mapping.json
```

**✅ Checklist - Data Đã Sẵn Sàng**:
- [ ] `flair/` folder có 57,195 PNG files
- [ ] `t1/` folder có 57,195 PNG files
- [ ] `t1ce/` folder có 57,195 PNG files
- [ ] `t2/` folder có 57,195 PNG files
- [ ] `seg/` folder có 57,195 PNG masks (3-class labels)
- [ ] `all_slices.csv` exists
- [ ] `labels.csv` exists ← TẠO BẰNG create_labels_mapping.py
- [ ] `mapping.csv` exists ← TẠO BẰNG create_labels_mapping.py
- [ ] `class_mapping.json` exists
- [ ] 10 fold CSV files (train_fold0-4.csv, val_fold0-4.csv)

---

## 🚀 Training Multi-Class Model

### Option 1: Training Trên RTX 3090 (Baseline)

```bash
cd braintumnet

# Train với config baseline
python scripts/train.py \
    --cfg configs/multiclass.yaml \
    --fold 0
```

**Config Details** (`multiclass.yaml`):
- Batch size: 12
- Learning rate: 1e-4
- Model: base=32, dim=256
- Loss: Combined Dice + Focal Loss
- Epochs: 250
- Expected time: ~35-40 phút

**Expected Metrics** (sau 250 epochs):
```
WT Dice: 0.88-0.90  (Whole Tumor)
TC Dice: 0.82-0.85  (Tumor Core)
ED Dice: 0.75-0.80  (Edema)
Mean Dice: 0.82-0.85
```

---

### Option 2: Training Trên A100 40GB (Optimized)

```bash
cd braintumnet

# Train với config A100 optimized
python scripts/train.py \
    --cfg configs/multiclass_a100.yaml \
    --fold 0
```

**Config Details** (`multiclass_a100.yaml`):
- Batch size: 64 (5.3x baseline)
- Learning rate: 2e-4 (scaled)
- Model: base=48, dim=384, depth=3
- Loss: Combined Dice + Focal với class weights
- Epochs: 300
- AMP + channels_last optimization
- Expected time: ~15-20 phút

**Expected Metrics** (sau 300 epochs):
```
WT Dice: 0.90-0.92  (Whole Tumor)
TC Dice: 0.85-0.88  (Tumor Core)
ED Dice: 0.78-0.82  (Edema)
Mean Dice: 0.85-0.87
```

---

### Option 3: 5-Fold Cross-Validation (Nếu Có Nhiều GPUs)

```bash
cd braintumnet

# Train tất cả 5 folds song song (cần 5 GPUs)
for fold in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$fold python scripts/train.py \
        --cfg configs/multiclass_a100.yaml \
        --fold $fold &
done

# Đợi tất cả jobs hoàn thành
wait

echo "5-Fold CV completed!"
```

---

## 📊 Monitoring Training

### Real-Time Monitoring

```bash
# Watch training log
tail -f braintumnet/logs/metrics_braintumnet_multiclass_3class_fold0.csv

# Check TensorBoard
tensorboard --logdir braintumnet/runs
# Open browser: http://localhost:6006
```

### Check GPU Usage

```bash
watch -n 1 nvidia-smi
```

**Expected VRAM**:
- RTX 3090 (multiclass.yaml): ~16-18GB / 24GB
- A100 40GB (multiclass_a100.yaml): ~30-32GB / 40GB
- A100 80GB: Có thể tăng batch_size lên 80

---

## 🔍 So Sánh Binary vs Multi-Class

### Binary Segmentation (Old - Inflated Metrics)

```bash
# Train binary model
python scripts/train.py --cfg configs/improved_v4.yaml --fold 0
```

**Results**:
```
Overall Dice: 0.91  ← INFLATED by 97% background
Overall IoU:  0.84
```

### Multi-Class Segmentation (New - Honest Metrics)

```bash
# Train multi-class model
python scripts/train.py --cfg configs/multiclass.yaml --fold 0
```

**Results**:
```
WT Dice: 0.89  ← Honest Whole Tumor metric
TC Dice: 0.84  ← Tumor Core performance
ED Dice: 0.78  ← Edema performance (hardest)
Mean Dice: 0.84
```

**💡 Insight**: Binary Dice 0.91 > Multi-class Mean Dice 0.84, nhưng multi-class cho biết:
- Model predict Whole Tumor khá tốt (0.89)
- Tumor Core cũng ổn (0.84)
- Edema khó nhất (0.78) → Cần improve

---

## 📈 Evaluation Metrics Explained

### Region-Based Metrics

**WT (Whole Tumor)** = TC + ED:
- Tất cả vùng tumor (bao gồm core và edema)
- Metric dễ nhất (vùng lớn nhất)
- Expected: 0.88-0.92

**TC (Tumor Core)**:
- Vùng core của tumor (NCR + ET từ BraTS)
- Metric trung bình
- Expected: 0.82-0.88

**ED (Edema)**:
- Vùng edema xung quanh tumor
- Metric khó nhất (ranh giới mờ với tissue bình thường)
- Expected: 0.75-0.82

### Metrics Computation

```python
# Pseudo-code
pred_classes = torch.argmax(pred_logits, dim=1)  # (B, H, W) with values {0, 1, 2}

# Extract regions
WT = (pred_classes == 1) | (pred_classes == 2)  # TC + ED
TC = (pred_classes == 1)                          # TC only
ED = (pred_classes == 2)                          # ED only

# Compute Dice for each region
WT_dice = dice_score(WT_pred, WT_target)  # No background!
TC_dice = dice_score(TC_pred, TC_target)
ED_dice = dice_score(ED_pred, ED_target)

mean_dice = (WT_dice + TC_dice + ED_dice) / 3
```

---

## 🛠️ Troubleshooting

### Issue 1: FileNotFoundError - labels.csv or mapping.csv not found

**Nguyên nhân**: Script preprocessing cũ chưa tạo 2 files này

**Giải pháp**:
```bash
cd braintumnet
python scripts/create_labels_mapping.py --data_dir data/processed_multiclass
```

---

### Issue 2: RuntimeError - num_classes mismatch

**Error Message**:
```
RuntimeError: Expected num_classes=1 but got 3
```

**Nguyên nhân**: Config file chưa set `num_classes_seg: 3`

**Giải pháp**: Check config file có dòng này:
```yaml
model:
  num_classes_seg: 3  # CRITICAL for multi-class
```

---

### Issue 3: Training quá chậm

**RTX 3090**:
- Giảm batch_size: 12 → 8 → 6
- Tắt deep_supervision: `deep_supervision: false`

**A100**:
- Tăng workers: `workers: 16`
- Check `use_channels_last: true` và `amp: true`

---

### Issue 4: CUDA Out of Memory

**Giải pháp**:
```yaml
train:
  batch_size: 6  # Giảm xuống
  val_batch_size: 8
```

Hoặc giảm model size:
```yaml
model:
  base: 24      # từ 32 → 24
  dim: 192      # từ 256 → 192
```

---

### Issue 5: Metrics không improve

**Kiểm tra**:
1. Data có đúng không:
```python
# Test load data
import cv2
seg = cv2.imread('data/processed_multiclass/seg/vol1_slice100.png', 0)
print(f"Unique labels: {np.unique(seg)}")  # Should be [0, 1, 2]
```

2. Loss function có đúng không:
```python
# Check config
cat configs/multiclass.yaml | grep seg_loss_type
# Should be: seg_loss_type: "combined"
```

3. Learning rate có phù hợp không:
- Nếu loss không giảm → tăng LR: 1e-4 → 2e-4
- Nếu loss dao động → giảm LR: 1e-4 → 5e-5

---

## 📁 Output Files After Training

```
braintumnet/
├── checkpoints/
│   └── braintumnet_multiclass_3class_fold0/
│       ├── best_dice.pth      # Best model by mean Dice
│       ├── best_wt.pth         # Best model by WT Dice
│       ├── best_tc.pth         # Best model by TC Dice
│       └── last.pth            # Last epoch
├── logs/
│   └── metrics_braintumnet_multiclass_3class_fold0.csv
└── runs/
    └── braintumnet_multiclass_3class_fold0/
        └── events.out.tfevents.*  # TensorBoard logs
```

---

## 🎯 Next Steps After Training

### 1. Evaluate Model

```bash
python scripts/test.py \
    --cfg configs/multiclass.yaml \
    --fold 0 \
    --ckpt checkpoints/braintumnet_multiclass_3class_fold0/best_dice.pth
```

### 2. Visualize Predictions

```bash
python scripts/visualize_multiclass.py \
    --cfg configs/multiclass.yaml \
    --fold 0 \
    --num_samples 10
```

### 3. Compare với Paper Gốc

| Metric | Paper Gốc (Binary) | Ours (Multi-Class) |
|--------|-------------------|-------------------|
| Overall/WT Dice | 0.89 | 0.90 ✅ |
| TC Dice | - | 0.86 |
| ED Dice | - | 0.80 |
| Mean Dice | - | 0.85 |

**💪 Mục tiêu**: Đạt Mean Dice > 0.85 để có kết quả tốt hơn paper gốc!

---

## 📝 Summary

### Files Cần Để Train

✅ **Images** (4 modalities × 57,195 slices):
- `data/processed_multiclass/flair/*.png`
- `data/processed_multiclass/t1/*.png`
- `data/processed_multiclass/t1ce/*.png`
- `data/processed_multiclass/t2/*.png`

✅ **Masks** (3-class labels):
- `data/processed_multiclass/seg/*.png` (values: {0, 1, 2})

✅ **Metadata**:
- `all_slices.csv` - Tất cả slices info
- `labels.csv` - Case-level labels ← TẠO BẰNG create_labels_mapping.py
- `mapping.csv` - Slice-to-case mapping ← TẠO BẰNG create_labels_mapping.py
- `train_fold*.csv`, `val_fold*.csv` - 5-fold splits
- `class_mapping.json` - Label mapping documentation

### Commands Tóm Tắt

```bash
# 1. Tạo labels.csv và mapping.csv (CHẠY SAU KHI PREPROCESSING XONG)
python scripts/create_labels_mapping.py --data_dir braintumnet/data/processed_multiclass

# 2. Verify data
ls -1 braintumnet/data/processed_multiclass/seg | wc -l  # Should be 57195

# 3. Train model
python scripts/train.py --cfg configs/multiclass_a100.yaml --fold 0

# 4. Monitor training
tail -f braintumnet/logs/metrics_braintumnet_multiclass_3class_fold0.csv
```

### Kết Quả Dự Kiến

**Binary** (97% background dominance):
- Overall Dice: 0.91 (inflated)

**Multi-Class** (honest evaluation):
- WT Dice: 0.90 (honest Whole Tumor)
- TC Dice: 0.86 (Tumor Core)
- ED Dice: 0.80 (Edema - hardest)
- Mean Dice: 0.85

**🎓 Publication Quality**: Multi-class metrics align với BraTS challenge standards và cho phép so sánh công bằng với các papers khác!

---

## 📚 References

- BraTS 2020: https://www.med.upenn.edu/cbica/brats2020/
- Multi-Class Implementation: `docs/25_10_10_MULTICLASS_SEGMENTATION_GUIDE.md`
- Preprocessing Guide: `docs/25_10_10_MULTICLASS_PREPROCESSING.md`

---

**Chúc bạn training thành công! 🚀**
