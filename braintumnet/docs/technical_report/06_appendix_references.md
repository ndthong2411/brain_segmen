# 06 · Appendix & Quick References

Một trang duy nhất để tra cứu nhanh các lệnh, đường dẫn và bảng tham số chính của BrainTumNet.

---

## 1. Cấu trúc thư mục quan trọng

| Path | Nội dung |
| ---- | -------- |
| `braintumnet/configs/` | `base.yaml`, `models/*.yaml`, hardware overrides. |
| `braintumnet/scripts/` | CLI cho preprocess, train, evaluate, inference. |
| `braintumnet/src/braintumnet/` | Code chính (models, data, engine, losses, utils). |
| `braintumnet/docs/technical/` | Tài liệu legacy (phiên bản trước). |
| `braintumnet/docs/technical_report/` | Bộ báo cáo mới (tài liệu bạn đang đọc). |
| `braintumnet/notebooks/` | Notebook hỗ trợ EDA, training pipeline. |
| `braintumnet/data/` | Raw, processed PNG, LMDB (không commit). |
| `checkpoints/`, `logs/`, `runs/` | Kết quả training. |

---

## 2. Lệnh CLI thường dùng

```bash
# 1. Preprocess BraTS (NIfTI -> PNG + CSV)
python braintumnet/scripts/preprocess_nifti_to_multiclass.py \
    --nifti_dir <RAW_DIR> \
    --out_dir braintumnet/data/processed_multiclass \
    --img_size 256 \
    --slices_per_case 30 \
    --num_folds 5

# 2. Convert PNG -> LMDB
python braintumnet/scripts/convert_to_lmdb.py \
    --png_root braintumnet/data/processed_multiclass \
    --lmdb_root braintumnet/data/lmdb_processed_multiclass \
    --map_size 50

# 3. Train (ví dụ Swin-UNETR, fold 0, config A100)
python braintumnet/scripts/train.py --model swin_unetr --cfg a100 --fold 0

# 4. Resume training
python braintumnet/scripts/train.py --model swin_unetr --fold 0 --resume

# 5. Evaluate checkpoint
python braintumnet/scripts/evaluate.py \
    --model swin_unetr \
    --ckpt checkpoints/braintumnet_best_fold0.pth \
    --fold 0

# 6. Inference (không TTA)
python braintumnet/scripts/predict.py \
    --model swin_unetr \
    --ckpt checkpoints/braintumnet_best_fold0.pth \
    --input_dir <DIR_WITH_PNG_SLICES> \
    --output_dir outputs/predictions
```

---

## 3. Bảng tham số loss & training

| Config key | Ý nghĩa | Gợi ý |
| ---------- | ------- | ----- |
| `train.loss_type` | Loại loss (dice_ce, dice_focal, multiclass_dice_focal, ultimate…) | Với `num_classes_seg > 1`, loss nhị phân tự động nâng cấp. |
| `train.seg_loss_weight`, `cls_loss_weight` | Tỉ trọng seg/cls | K-dữ: seg=1.0, cls=0.5; nếu seg-only, đặt cls=0.0. |
| `train.focal_alpha` | Alpha cho focal (list hoặc scalar) | Multiclass: cung cấp list `[bg, class1, class2]`. |
| `train.class_weights` | Class weights cho CE hoặc Dice multi-class | Giúp cân bằng TC/ED. |
| `train.lr`, `train.weight_decay` | Learning rate & regularization | `5e-5` + `1.5e-4` là baseline an toàn. |
| `train.grad_accum_steps` | Gradient accumulation | Tăng khi batch size lớn gây thiếu VRAM. |
| `train.scheduler` | `cosine`, `plateau`, `onecycle` | Cosine + warmup 2000 là mặc định. |
| `train.amp`, `train.amp_dtype` | Mixed precision | A100: nên dùng `amp=True`, `amp_dtype=bfloat16`. |
| `train.channels_last` | Memory format optimize | Kích hoạt khi GPU hỗ trợ (A100). |

---

## 4. Bảng tham khảo model configs

| Model | Config file | Tham số đáng chú ý |
| ----- | ----------- | ------------------ |
| SegUNetV2 | `configs/models/segunetv2.yaml` | `base`, `dim`, `patch_size`, `depth`, `n_heads`, `dropout`. |
| BrainTumNetV2 | `braintumnet_v2.py` (dùng chung với SegUNetV2) | `roi_stop_grad`, `deep_supervision`, `multi_scale_fusion`. |
| Swin-UNETR | `configs/models/swin_unetr.yaml` | `feature_size`, `use_checkpoint`. |
| nnU-Net | `configs/models/nnunet.yaml` | `base`, `deep_supervision`. |
| TransUNet | `configs/models/transunet.yaml` | `embed_dim`, `depth`, `num_heads`, `base`. |
| UNETR | `configs/models/unetr.yaml` | `hidden_size`, `feature_size`, `num_heads`. |
| LG-UNETR | `configs/models/lg_unetr.yaml` | `base`, `embed_dim`, `depth`, `num_heads`. |

---

## 5. Notebook quan trọng

| Notebook | Mục đích |
| -------- | -------- |
| `notebooks/brain_seg.ipynb` | Pipeline tất cả bước (config, dataset, quick train, inference). |
| `notebooks/dataset_eda.ipynb` | Phân tích dataset, tỷ lệ lớp, visualization. |
| `main.ipynb` (nếu có) | Notebook lịch sử (không duy trì thường xuyên). |

---

## 6. Liên hệ giữa tài liệu

- Bộ báo cáo cũ (`docs/technical/v_*`) chứa lịch sử chi tiết và quyết định thiết kế 2024–2025.
- Bộ mới (`docs/technical_report/*`) là bản rút gọn dễ đọc, thích hợp cho onboarding và vận hành hàng ngày.
- Khi cần đào sâu (ví dụ lý do thay đổi loss ngày 2025-10-15), hãy xem `docs/technical/UPDATE_SUMMARY_2025_10_15.md`.

---

## Kết thúc

Bạn đã có tất cả tài liệu cần thiết: quay lại `README.md` của technical report nếu muốn lặp lại luồng đọc, hoặc mở trực tiếp file bạn cần tra cứu khi làm việc.
