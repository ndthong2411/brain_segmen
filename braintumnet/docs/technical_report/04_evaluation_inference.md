# 04 · Evaluation & Inference — Đo lường và xuất dự đoán

Tài liệu này đưa bạn qua các bước kiểm tra chất lượng mô hình, lưu checkpoint, xuất kết quả, và các công cụ notebook kèm theo.

---

## 1. Metrics chính

- **Dice & IoU**: tính cho 3 vùng BraTS (WT, TC, ED) trong `braintumnet/src/braintumnet/multiclass_metrics.py`.
- **Mean Dice/IoU**: trung bình ba vùng, dùng để xếp hạng checkpoint.
- **ClsAcc**: accuracy phân loại HGG/LGG (chỉ có khi model trả logits classification).
- **Logging**: trong mỗi epoch, trainer in `WT_dice`, `TC_dice`, `ED_dice`, `val_iou`, `val_dice`, `val_acc`.
- Auxiliary: `MetricsLogger` lưu chi tiết metrics từng epoch trong `logs/metrics_*` (CSV).

> 🧮 Số liệu validation được tính trong chế độ `torch.inference_mode()` nên không ảnh hưởng tốc độ training quá nhiều.

---

## 2. Checkpoint & Resume

- `checkpoints/best_fold{fold}.pth`: lưu state dict tốt nhất theo IoU.
- `checkpoints/last_fold{fold}.pth`: lưu full state (model + optimizer + scaler).
- Resume training:
  ```bash
  python braintumnet/scripts/train.py --model swin_unetr --fold 3 --resume
  ```
  Nếu không cung cấp path, script tìm `checkpoints/last_fold{fold}.pth`.
- Tự tạo checkpoint tên khác? `train.py` có thể sửa `cfg["logging"]["exp_name"]` để isolate.

---

## 3. Evaluation script

- File: `braintumnet/scripts/evaluate.py`.
- Sử dụng checkpoints đã train để tính lại metrics trên fold validation (hoặc bộ mới).
- Ví dụ:
  ```bash
  python braintumnet/scripts/evaluate.py \
      --model nnunet \
      --ckpt checkpoints/braintumnet_best_fold3.pth \
      --fold 3
  ```
- Script sử dụng cùng `build_model` và `dataset_factory` nên luôn đồng nhất pipeline.

---

## 4. Inference & TTA

- `scripts/predict.py`: inference cơ bản cho một thư mục slice PNG.
  - Tham số chính: `--input_dir`, `--output_dir`, `--ckpt`.
  - Chỉ áp dụng cho mô hình segmentation (logits classification bị bỏ qua).
- `scripts/tta_inference.py`: áp dụng Test-Time Augmentation (flip, rotate).
  - Kết hợp nhiều augmentation → trung bình xác suất.
  - Hữu ích khi cần tăng điểm chút ít mà không retrain.
- `scripts/ensemble_inference.py`: support ensemble nhiều checkpoint.

> 🎯 Output inference mặc định là mask 0/1/2. Có thể hậu xử lý (morphology) nếu muốn cải thiện submission.

---

## 5. Notebook hỗ trợ

- `notebooks/brain_seg.ipynb`:
  - Cell visualization: hiển thị dự đoán vs ground truth (dựa trên sample loader).
  - Cell logging: liệt kê log gần nhất, checkpoint path.
- `notebooks/dataset_eda.ipynb`: dùng lại để so sánh mask GT và pred (khi bạn ghi đè input).

---

## 6. Quản lý log & TensorBoard

- Chạy TensorBoard:
  ```bash
  tensorboard --logdir runs --port 6006
  ```
- Mỗi experiment → `runs/{exp_name}_fold{fold}`.
- Lưu ý: khi resume với exp_name giống nhau, log sẽ nối tiếp. Để tách run mới, đổi `cfg["logging"]["exp_name"]` trước khi train lại.

---

## 7. Checklist trước khi đánh giá cuối

1. Đảm bảo `loss_type` và `ignore_background` khớp với cách tính métric mong muốn.
2. Xem log epoch cuối để chắc chắn không NaN và metrics tăng đều (nếu tụt mạnh → overfitting hoặc sai config).
3. Nếu ensemble: xác nhận kích thước output giống nhau, và bạn đang dùng cùng fold/tiền xử lý.
4. Ghi chú lại command, config, checkpoint vào `docs/` hoặc `logs/` để dễ tái lập.

---

## Next steps

- Sẵn sàng chiến fight? Nếu gặp lỗi runtime, chuyển sang `05_operations_troubleshooting.md`.
- Muốn có bảng tham chiếu lệnh và module? Xem `06_appendix_references.md`.
