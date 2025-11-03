# 05 · Operations & Troubleshooting — Vận hành ổn định, xử lý sự cố

Tài liệu này tập trung vào những vấn đề “đời thường” khi chạy BrainTumNet: cấu hình phần cứng, lỗi thường gặp, và cách debug nhanh.

---

## 1. Môi trường & phụ thuộc

- Python 3.10+ khuyến nghị (Torch 2.x).
- Conda env mẫu (giả định đã có CUDA phù hợp):
  ```bash
  conda create -n brain_seg python=3.10
  conda activate brain_seg
  pip install -r requirements.txt  # nếu bạn duy trì file này
  ```
- Gói quan trọng:
  - `torch`, `torchvision`, `torchaudio` (phiên bản tương thích GPU).
  - `monai`, `einops` (cho Swin-UNETR, UNETR).
  - `tqdm`, `tensorboard`, `pandas`, `lmdb`, `nibabel`.
- Cấu hình Windows/PowerShell: dùng `python -c "import sys; sys.path.append('braintumnet\\src'); ..."` khi test nhanh để tránh lỗi path.

---

## 2. Quản lý đường dẫn & config

- Luôn chạy script từ thư mục gốc repo (`E:\thong\code\brain_segmen` trong ví dụ).
- Notebook đã có helper `resolve_path` để chuyển relative → absolute. Nếu port sang nơi khác, cập nhật helper này để tránh “FileNotFoundError” giả.
- Khi thêm model mới, nhớ tạo file config tương ứng trong `configs/models/` và cập nhật README nếu cần.

---

## 3. Lỗi phổ biến & cách xử lý

| Triệu chứng | Nguyên nhân gốc | Cách khắc phục |
| ----------- | --------------- | -------------- |
| `SyntaxError` tại trainer | Merge code bị thêm `else` thừa | Chạy `python -m compileall braintumnet/src/braintumnet` sau khi chỉnh để phát hiện sớm. |
| NaN ở epoch đầu | AMP float16 + loss chưa ổn định | Loss đã cast float32, nhưng nếu vẫn gặp: đổi `amp_dtype` sang `bfloat16` hoặc giảm `lr`. |
| Dice thấp dù loss thấp | Loss nhị phân áp dụng cho multi-class | Giờ đã tự động chuyển sang loss đa lớp; kiểm tra log “auto-adjusted” để xác nhận. |
| `ModuleNotFoundError: einops` | Swin-UNETR cần `einops` | `pip install einops`. |
| Dataset không tìm thấy CSV | Chưa preprocess hoặc sai path | Rerun preprocessing, kiểm tra `cfg['data']` và helper `resolve_path`. |
| GPU idle, CPU 100% | DataLoader không đủ worker | Tăng `train.workers`, `prefetch_factor`, bật `pin_memory`. |
| TensorBoard không hiện | Log path sai | Kiểm tra `cfg['logging']`; đảm bảo `runs/` tồn tại và quyền ghi OK. |

---

## 4. Tips vận hành

- **Checkpoint hygiene:** định kỳ dọn `checkpoints/` để tránh nhầm, đặt tên bao gồm ngày/giờ nếu thử nghiệm nhiều.
- **Log rotation:** `TrainingLogger` ghi file dài. Khi train lâu, chép log sang `logs/archive/` để dễ quản lý.
- **Warm-up dài:** `warmup_steps` mặc định 2000. Nếu dataset nhỏ, cân nhắc giảm còn 500 để lr tăng nhanh hơn.
- **Validation song song:** Có thể đặt `train.val_interval = 2` để giảm tần suất validation khi GPU yếu.
- **A100 optimization:** bật `train.optimizer_fused = True`, `train.channels_last = True`, `train.pin_memory = True` (đã mặc định trong `base.yaml`).

---

## 5. Quy trình debug nhanh

1. **Check config merge:** in `cfg` ở `train.py` (đã làm sẵn) để chắc chắn giá trị như mong muốn.
2. **Kiểm tra sample batch:** trong notebook, load `next(iter(train_loader))`, kiểm tra shape `(batch, 4, 256, 256)` và mask `(batch, 1, 256, 256)`.
3. **Gradient check:** nếu loss đứng yên, bật `torch.autograd.set_detect_anomaly(True)` tạm thời để phát hiện ops bất thường (chỉ khi debug).
4. **Profiling DataLoader:** chạy `scripts/benchmark_dataloader.py` để biết bottleneck I/O.
5. **Hạn chế bug merge:** sau mỗi chỉnh sửa lớn, chạy `python -m compileall` như ở trên để bảo đảm không lỗi cú pháp.

---

## Next steps

- Muốn có bảng tra lệnh và đường dẫn quan trọng? xem `06_appendix_references.md`.
- Đã hiểu toàn hệ thống? Quay lại `README.md` của technical report để xem lại flow tổng quan.
