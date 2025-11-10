# Validation Guide

## Cách chạy validation đơn giản

Tất cả các lệnh validation giờ chỉ cần truyền đường dẫn checkpoint vào!

### 1. Evaluation (Single Fold)

Đánh giá một checkpoint duy nhất:

```bash
python braintumnet/scripts/evaluate.py checkpoints/braintumnet_best_fold0.pth
python braintumnet/scripts/evaluate.py checkpoints/braintumnet_best_fold4.pth
```

**Output:**
- Tự động detect fold từ tên file (fold0, fold4, etc.)
- Load config từ `configs/base.yaml`
- In ra metrics: IoU, Dice, HD, HD95, Accuracy, F1, AUC-ROC

---

### 2. Test-Time Augmentation (TTA)

TTA áp dụng 8 augmentations và average predictions để cải thiện accuracy.

**Expected gain: +2-3% IoU** (không cần train lại!)

```bash
python braintumnet/scripts/tta_inference.py checkpoints/braintumnet_best_fold0.pth
python braintumnet/scripts/tta_inference.py checkpoints/braintumnet_best_fold4.pth
```

**Output mặc định:** `results/tta_fold{N}.csv`

**Custom output path:**
```bash
python braintumnet/scripts/tta_inference.py checkpoints/braintumnet_best_fold0.pth --output my_results.csv
```

**Lưu ý:** TTA chậm hơn ~8x so với inference thông thường (8 augmentations)

---

### 3. 5-Fold Ensemble

Ensemble tất cả các fold models để cải thiện generalization.

**Expected gain: +3-4% IoU** (dùng các fold models có sẵn!)

```bash
python braintumnet/scripts/ensemble_inference.py "checkpoints/braintumnet_best_fold*.pth"
```

**Với TTA (VERY SLOW nhưng tốt nhất):**
```bash
python braintumnet/scripts/ensemble_inference.py "checkpoints/braintumnet_best_fold*.pth" --use_tta
```

**Expected gain với TTA:** +5-7% IoU (ensemble + TTA cộng lại)

**Output mặc định:** `results/ensemble_results.csv`

**Custom output:**
```bash
python braintumnet/scripts/ensemble_inference.py "checkpoints/braintumnet_best_fold*.pth" --output my_ensemble.csv
```

**Lưu ý:**
- Glob pattern phải đặt trong dấu ngoặc kép `"..."`
- Với TTA enabled, mất ~40x thời gian (5 models × 8 augmentations)

---

## So sánh các phương pháp

| Method | Command | Speed | Expected Gain |
|--------|---------|-------|---------------|
| **Single Model** | `evaluate.py checkpoint.pth` | 1x | Baseline |
| **TTA** | `tta_inference.py checkpoint.pth` | 8x slower | +2-3% IoU |
| **Ensemble (5 folds)** | `ensemble_inference.py "fold*.pth"` | 5x slower | +3-4% IoU |
| **Ensemble + TTA** | `ensemble_inference.py "fold*.pth" --use_tta` | 40x slower | +5-7% IoU |

---

## Workflow đầy đủ

### Sau khi train xong:

```bash
# 1. Train model (ví dụ fold 4)
python braintumnet/scripts/train.py --model nnunet --fold 4

# 2. Evaluate checkpoint vừa train
python braintumnet/scripts/evaluate.py checkpoints/braintumnet_best_fold4.pth

# 3. (Optional) Thử TTA để cải thiện kết quả
python braintumnet/scripts/tta_inference.py checkpoints/braintumnet_best_fold4.pth

# 4. (Optional) Sau khi train cả 5 folds, chạy ensemble
python braintumnet/scripts/ensemble_inference.py "checkpoints/braintumnet_best_fold*.pth"

# 5. (Optional) Best performance - Ensemble + TTA (VERY SLOW!)
python braintumnet/scripts/ensemble_inference.py "checkpoints/braintumnet_best_fold*.pth" --use_tta
```

---

## Lưu ý quan trọng

1. **Checkpoint naming convention:**
   - Scripts tự động detect fold từ tên file
   - Pattern: `*fold{N}.pth` (ví dụ: `braintumnet_best_fold0.pth`, `last_fold4.pth`)

2. **Config loading:**
   - Tất cả scripts đều load config từ `configs/base.yaml`
   - Không cần chỉ định config path

3. **Data paths:**
   - Data root được load từ config: `cfg['data']['proc_root']`
   - Validation split file: `{data_root}/val_fold{N}.csv`

4. **Device:**
   - Default: CUDA (nếu available)
   - Override: `--device cpu` hoặc `--device cuda:1`

---

## Troubleshooting

**Q: Script báo "Checkpoint not found"?**
A: Kiểm tra đường dẫn checkpoint có đúng không. Dùng đường dẫn tương đối hoặc tuyệt đối đều được.

**Q: Script báo "Config not found"?**
A: Đảm bảo file `configs/base.yaml` tồn tại.

**Q: Ensemble không tìm thấy checkpoints?**
A: Đặt glob pattern trong dấu ngoặc kép: `"checkpoints/braintumnet_best_fold*.pth"`

**Q: TTA quá chậm?**
A: Bình thường! TTA chạy 8 augmentations nên chậm hơn ~8x. Chỉ dùng khi cần accuracy cao nhất.

**Q: Ensemble + TTA quá chậm?**
A: Cực kỳ chậm! Chỉ dùng cho final evaluation hoặc khi submit competition.

---

## Backup scripts

Các script gốc đã được backup:
- `scripts/evaluate_backup.py`
- `scripts/tta_inference_backup.py`
- `scripts/ensemble_inference_backup.py`

Nếu cần quay lại version cũ, rename files này.
