# 02 · Model Zoo — Chọn kiến trúc nào cho BrainTumNet?

Mục tiêu: nắm hết các mô hình trong `braintumnet/src/braintumnet/models/`, hiểu điểm mạnh và cách bật/tắt chúng qua config.

---

## 1. Tổng quan

- Hàm factory: `braintumnet/src/braintumnet/models/__init__.py > build_model(cfg)`
- `cfg["model"]["model_type"]` quyết định kiến trúc.
- Bộ tham số chung: `in_channels` (4), `num_classes_seg` (3), `num_classes_cls` (2 — classification HGG/LGG).

| Model type | File | Khi nào dùng | Ghi chú |
| ---------- | ---- | ------------ | ------- |
| `segunetv2`/`v2` | `braintumnet_v2.py`, `seg_unet_v2.py` | Baseline đa nhiệm (seg + cls). | Có backbone transformer nhẹ, hỗ trợ deep supervision. |
| `swin_unetr` | `swin_unetr_wrapper.py` | Transfer từ MONAI Swin-UNETR 2D. | Seg only, cần thư viện `einops`, `monai`. |
| `nnunet` | `nnunet_wrapper.py` | nnU-Net style 5 level. | Deep supervision mặc định. |
| `unetr` | `unetr_wrapper.py` | MONAI UNETR 2D. | Seg only. |
| `transunet` | `transunet_wrapper.py` | ResNet + Transformer hybrid. | Hữu ích khi muốn so sánh backbone CNN vs hybrid. |
| `lg_unetr` | `lg_unetr_wrapper.py` | Local-Global UNETR (song song CNN + Transformer). | Nhiều tham số hơn; cần GPU mạnh. |

---

## 2. BrainTumNetV2 (SegUNetV2 + classification)

- File chính: `braintumnet_v2.py`, `seg_unet_v2.py`.
- Thành phần:
  - Encoder conv residual + InstanceNorm, LeakyReLU.
  - Bottleneck transformer `AdaptiveMaskedTransformer` (patch attention).
  - Decoder có CBAM attention, multi-scale fusion (tùy chọn).
  - Deep supervision: 3 head phụ (64x64, 128x128, 256x256).
  - Classification branch: mask input bằng whole-tumor probability rồi đưa qua `TInceptionNet`.
- Các flag cấu hình (`configs/models/segunetv2.yaml`):
  - `base`, `dim`, `patch_size`, `depth`, `n_heads`, `dropout`.
  - `deep_supervision`, `multi_scale_fusion`, `roi_stop_grad`.

> ✅ Khi cần multi-task (seg + HGG/LGG), đây là mô hình duy nhất sẵn sàng production trong repo.

---

## 3. Swin-UNETR Wrapper

- Địa chỉ: `models/swin_unetr_wrapper.py`.
- Dựa trên `monai.networks.nets.SwinUNETR` (2D adaptation).
- Điểm chú ý:
  - `img_size` đã bị MONAI deprecate; wrapper xử lý try/except để tương thích 1.3+.
  - Trả về `(seg_logits, None)` → trainer biết đây là model segmentation-only.
  - Cần cài `einops` (nếu thiếu sẽ báo lỗi ngay khi instantiate).
- Config (`configs/models/swin_unetr.yaml`):
  - `feature_size`, `img_size`, `use_checkpoint`.

---

## 4. nnU-Net Wrapper

- File: `models/nnunet_wrapper.py`.
- Thiết kế:
  - Encoder 5 tầng, residual block kiểu nnU-Net (InstanceNorm, LeakyReLU 0.01).
  - Downsample bằng conv stride 2 (học được).
  - Deep supervision nhiều mức (32, 64, 128).
  - Trả về `(seg, None, aux)` nếu `deep_supervision=True`.
- Cần loss hỗ trợ multi-class (đã auto chuyển thông qua `MultiTaskLoss`).

---

## 5. TransUNet, UNETR, LG-UNETR

- **TransUNet:** `transunet_wrapper.py`
  - Encoder ResNet, decoder transformer.
  - Config: `embed_dim`, `depth`, `num_heads`, `base`.
  - Dùng cho nghiên cứu hybrid CNN + ViT.
- **UNETR:** `unetr_wrapper.py`
  - MONAI UNETR (2D). `hidden_size`, `feature_size`, `num_heads` cấu hình.
- **LG-UNETR:** `lg_unetr_wrapper.py`
  - Local-global fusion, nhiều modules hơn; chú ý footprint bộ nhớ.
  - Config: `base`, `num_levels`, `embed_dim`, `depth`, `num_heads`.

> ⚙️ Tất cả các wrapper “segmentation only” luôn trả `(seg_logits, None)` hoặc `(seg_logits, None, aux)` để tương thích với trainer.

---

## 6. Tips chọn model

| Use case | Gợi ý | Lý do |
| -------- | ----- | ----- |
| Training nhanh, baseline | `segunetv2` (phase2_small) | 26M params, loss đa nhiệm mạnh, có doc chi tiết. |
| Tối đa hóa Dice/IoU | `swin_unetr` hoặc `nnunet` | Pull từ cộng đồng BraTS, hiệu quả cao đa lớp. |
| Nghiên cứu kiến trúc mới | `transunet`, `lg_unetr` | Dễ chỉnh ViT depth/head. |
| Thiếu RAM/GPU | Giảm `base`, `dim`, tắt `deep_supervision`; cân nhắc dùng `segunetv2` thu gọn. |

---

## 7. Kết nối với config

- `scripts/train.py` tự merge:
  1. `configs/base.yaml`
  2. `configs/models/{model}.yaml`
  3. Option `configs/hardware_{cfg}.yaml` (ví dụ `--cfg a100`).
- Bạn có thể tạo file riêng (ví dụ `configs/models/nnunet_light.yaml`) rồi gọi: `python scripts/train.py --model nnunet_light` (đồng thời thêm case vào `train.py` nếu muốn).

---

## Next steps

Chọn xong mô hình → sang `03_training_and_loss.md` để nắm loss, lịch học, mixed precision, và cách logger quản lý các mô hình khác nhau.
