# 01 · Data Pipeline — Từ NIfTI tới LMDB sẵn sàng train

Mục tiêu: hiểu và kiểm tra nhanh toàn bộ luồng dữ liệu của BrainTumNet. Khi bạn đọc xong, bạn có thể tái tạo dữ liệu đầu vào cho bất kỳ mô hình nào trong repo.

---

## 1. Nguồn dữ liệu

- **BraTS 2020** (MRI 4 modality: FLAIR, T1, T1CE, T2) + segmentation `seg`.
- Cấu trúc gốc: mỗi ca bệnh là một thư mục `.nii`.
- Đường dẫn gốc được cấu hình trong `configs/base.yaml > data.raw_root`.

> 📁 Ví dụ: `braintumnet/data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`

---

## 2. Bước 1 – Chuyển NIfTI → PNG đa lớp

- Script: `python braintumnet/scripts/preprocess_nifti_to_multiclass.py`
- Đầu ra: `processed_multiclass` gồm 5 thư mục (`flair/`, `t1/`, `t1ce/`, `t2/`, `seg/`) + CSV cho các fold.
- Các tham số chính:
  - `--img_size`: resize + pad ảnh vuông (mặc định 256).
  - `--slices_per_case`: chọn số lát cắt “nhiều khối u” (giảm tỷ lệ nền).
  - `--num_folds`: tạo train/val split (mặc định 5).
- Mapping nhãn:
  - 0 → nền.
  - 1 → Tumor Core (gộp label 1 + 4).
  - 2 → Edema (label 2).

> ☑️ Kiểm tra nhanh: mở một PNG trong `seg/` (sử dụng viewer) và đảm bảo chỉ chứa giá trị 0/1/2.

---

## 3. Bước 2 – Chuyển PNG → LMDB (tăng tốc I/O)

- Script: `python braintumnet/scripts/convert_to_lmdb.py`
- Output: thư mục LMDB gồm `data.mdb`, `lock.mdb`, `meta.json`, `*.csv` copy từ processed.
- Ưu điểm:
  - Giảm overhead đọc file rời khi huấn luyện (đặc biệt trên A100).
  - Thông tin meta (số slice, danh sách ID) được lưu trong LMDB để dataset đọc nhanh.
- Tham số quan trọng:
  - `--map_size`: dung lượng tối đa của LMDB (GB). Mặc định 50 GB.

> 🔁 Có thể chạy lại script bất cứ lúc nào. Nếu thay đổi số fold hay cấu trúc, nhớ xóa LMDB cũ để tránh “lỗi nhầm metadata”.

---

## 4. Dataset Factory & lựa chọn backend

- Tất cả loader đi qua `braintumnet/src/braintumnet/data/dataset_factory.py`.
- Hai backend:
  - `"png"` → `SliceDataset` (dùng khi bạn chưa convert LMDB).
  - `"lmdb"` → `LMDBDataset` (khuyến nghị cho training dài).
- `cfg["data"]["backend"]` điều khiển backend (mặc định `lmdb` trong base).
- Hàm `get_data_root(cfg)` tự trả đúng đường dẫn (`proc_root` hoặc `lmdb_root`).

> 🧪 Test nhanh trong notebook: dùng cell `create_dataset` (xem `notebooks/brain_seg.ipynb`) để xác nhận số lượng mẫu và cấu trúc batch.

---

## 5. Kiểm tra chất lượng & EDA

Những công cụ có sẵn:

- `notebooks/dataset_eda.ipynb`: thống kê phân phối lát cắt, tỷ lệ lớp.
- `scripts/benchmark_dataloader.py`: đo tốc độ DataLoader với nhiều worker/batch size.
- Notebook `brain_seg.ipynb`:
  - Cell “Class Distribution Snapshot”: quét 500 lát để thấy tỷ lệ pixel 0/1/2.
  - Cell visualization: hiển thị 4 modality + mask để kiểm tra alignment.

> 🛡️ Nếu Dice/Iou đầu epoch gần 0: kiểm tra lại mask bằng visualization – thường là do pipeline bị lỗi mapping.

---

## 6. Thường gặp & xử lý

| Sự cố | Nguyên nhân | Cách xử lý |
| ----- | ---------- | ---------- |
| Thiếu `train_fold*.csv` khi notebook đọc | Chưa chạy preprocessing hoặc đường dẫn tương đối | Bật `RUN_PREPROCESS=True` hoặc cấu hình lại path bằng helper `resolve_path`. |
| LMDB đọc chậm | Chưa bật `pin_memory`, `prefetch_factor` thấp | Điều chỉnh trong `configs/base.yaml > train`. |
| Memory không đủ khi convert | `--map_size` quá nhỏ | Tăng map size, hoặc chạy trên ổ có nhiều trống. |

---

## Next steps

- Đã chuẩn bị xong dữ liệu? Đi tiếp `02_models.md` để chọn backbone.
- Nếu bạn muốn reprocess dữ liệu với tham số mới, lưu lại script call trong `docs/` hoặc một README riêng để lần sau tái sử dụng dễ dàng.
