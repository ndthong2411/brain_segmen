# 03 · Training & Loss — Điều khiển vòng lặp học

Tài liệu này giải thích rõ cách BrainTumNet training, các loại loss, scheduler, mixed precision, và cách đọc log. Bám theo để cấu hình hoặc debug phiên train mới.

---

## 1. Pipeline training tổng quát

- Entry point: `braintumnet/scripts/train.py`
  - Hợp config (`base.yaml` + `models/*.yaml` + optional `hardware_*.yaml`).
  - In summary cấu hình và gọi `train_one_fold(cfg, fold)`.
- Trainer chính: `braintumnet/src/braintumnet/engine/trainer.py`
  - Xây DataLoader via `dataset_factory`.
  - Build model (`models/__init__.py`).
  - Thiết lập optimizer, scheduler, loss, AMP, TensorBoard.
  - Vòng lặp epoch: train → val → checkpoint.

> 🧭 Logging: dùng `TrainingLogger` (ghi file) + `SummaryWriter` (TensorBoard). Log mặc định nằm trong `logs/` và `runs/`.

---

## 2. Loss system

### 2.1 MultiTaskLoss (loss chuẩn cũ)

- File: `braintumnet/src/braintumnet/losses/base.py > MultiTaskLoss`.
- Thành phần:
  - `seg_loss`: tùy `loss_type` (Dice-CE, Dice-Focal, Multiclass…) với auto mapping.
  - `cls_loss`: CrossEntropy cho branch phân loại (HGG/LGG).
  - Optional `BoundaryLoss` nếu `boundary_w > 0`.
- 2025-11 update: nếu `num_classes_seg > 1` và bạn chọn loss nhị phân (`dice_ce`, `dice_focal`), lớp sẽ tự chuyển sang biến thể đa lớp (`MulticlassDiceCELoss`, `MulticlassDiceFocalLoss`). Trainer log sẽ báo “auto-adjusted” để bạn biết.
- Tham số điều khiển trong `cfg["train"]`:
  - `seg_loss_weight`, `cls_loss_weight`, `boundary_loss_weight`.
  - `focal_alpha`, `focal_gamma`, `class_weights`, `ignore_background`.

### 2.2 UltimateLoss / UltimateMultiTaskLoss

- File: `braintumnet/src/braintumnet/losses/combined.py`.
- Kích hoạt khi `loss_type` là `"ultimate"` hoặc `"ultimate_multitask"`.
- Bao gồm Dice + Focal + IoU + Boundary, cộng thêm deep supervision (aux) và classification nếu bật.
- Phù hợp khi bạn muốn tối đa hóa IoU (cấu hình `dice_weight`, `focal_weight`, `iou_weight`, `boundary_weight`).

---

## 3. Optimizer & Scheduler

- Optimizer mặc định: `AdamW` (config `train.optimizer`). Có tùy chọn `optimizer_fused` (True khi chạy A100 với torch>=2.0).
- Learning rate:
  - `train.lr` (mặc định `5e-5`).
  - Gradient Accumulation: `train.grad_accum_steps`.
  - Cosine scheduler với warmup (mặc định `warmup_steps=2000`, `min_lr`).
  - Một số config hỗ trợ `plateau` hoặc `onecycle`.
- Gradient clipping: `train.grad_clip_norm` (1.0).

> 🔍 Khi thay `grad_accum_steps`, nhớ tính lại LR hiệu dụng (LR không tự chia).

---

## 4. Mixed Precision & AMP

- Điều khiển bằng `train.amp` (True/False) và `train.amp_dtype` (`float16`/`bfloat16`).
- Trainer dùng `torch.amp.autocast` và `torch.amp.GradScaler` (chỉ cần cho float16).
- Loss đã được cập nhật để tính toán bất biến với độ chính xác (casting về float32 khi cần để tránh NaN).

> 🧊 Nếu gặp NaN khi training A100: kiểm tra `amp_dtype` → nên dùng `bfloat16` nếu có.

---

## 5. Deep supervision & aux loss

- Các model có thể trả `(seg, cls, aux_outputs)` hoặc `(seg, None, aux)`.
- Trainer tự resize aux segmentation về kích thước gốc bằng `F.interpolate` và dùng cùng `seg_loss` để tính auxiliary loss (trọng số `train.aux_loss_weights` hoặc mặc định `[0.5, 0.25, 0.125]`).
- Với `ultimate_multitask`, deep supervision do loss đảm trách; trong `MultiTaskLoss` chỉ áp dụng cho mô hình cũ.

---

## 6. Logging & checkpoints

- TensorBoard: thư mục `runs/{exp_name}_fold{fold}`.
- Training logger: `logs/{exp_name}_fold{fold}.log` (ghi loss, lr, thời gian, dice theo lớp).
- Checkpoints:
  - `checkpoints/best_fold{fold}.pth` (lưu khi IoU tốt hơn).
  - `checkpoints/last_fold{fold}.pth` (mỗi epoch).
- Hàm `save_training_state` lưu cả optimizer, scheduler, scaler để resume.

> ⏱️ Resume: `python scripts/train.py --model ... --fold ... --resume` (mặc định tự tìm `last_foldX.pth`).

---

## 7. Kiểm tra trước khi train dài

1. Chạy notebook `brain_seg.ipynb` phần “Optional Quick Training” (1 epoch) để đảm bảo loss giảm và không lỗi data.
2. Kiểm tra log: nếu thấy “Using loss type: multiclass_dice_focal(auto)” nhưng bạn muốn loss khác → chỉnh `train.loss_type` lại hoặc cung cấp `class_weights` phù hợp.
3. Theo dõi GPU/CPU: `train.workers`, `prefetch_factor`, `pin_memory` có thể cần điều chỉnh tùy môi trường.

---

## Next steps

- Để đánh giá và suy ra kết quả, đọc `04_evaluation_inference.md`.
- Khi triển khai trên máy chủ hoặc gỡ lỗi, xem `05_operations_troubleshooting.md`.
