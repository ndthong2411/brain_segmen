# BrainTumNet Phase 2 - Technical Documentation

Tài liệu kỹ thuật chi tiết về **SegUNetV2** và các cải tiến Phase 2.

---

## 📚 Files

1. **[v2_00_INDEX.md](v2_00_INDEX.md)** - Chỉ mục và hướng dẫn
2. **[v2_01_PHASE2_OVERVIEW.md](v2_01_PHASE2_OVERVIEW.md)** - Tổng quan Phase 2
3. **[v2_02_SEGUNETV2_ARCHITECTURE.md](v2_02_SEGUNETV2_ARCHITECTURE.md)** - Kiến trúc chi tiết
4. **[v2_03_PHASE2_FEATURES.md](v2_03_PHASE2_FEATURES.md)** - Tính năng Phase 2
5. **[v2_04_TRAINING_CONFIG.md](v2_04_TRAINING_CONFIG.md)** - Cấu hình training
6. **[v2_06_UPGRADE_REASONING.md](v2_06_UPGRADE_REASONING.md)** - ⭐ **Tại sao thay đổi**

---

## 🎯 Bắt Đầu

**Đọc theo thứ tự**:
1. Bắt đầu với [INDEX](v2_00_INDEX.md)
2. Đọc [Phase 2 Overview](v2_01_PHASE2_OVERVIEW.md) để hiểu tổng quan
3. **QUAN TRỌNG**: Đọc [Upgrade Reasoning](v2_06_UPGRADE_REASONING.md) để hiểu **TẠI SAO** thay đổi
4. Xem [Architecture](v2_02_SEGUNETV2_ARCHITECTURE.md) để hiểu implementation
5. Tham khảo [Training Config](v2_04_TRAINING_CONFIG.md) khi cần train

---

## 📊 Tóm Tắt Nhanh

### Phase 2 Improvements (7 core + 3 optional)

**7 Core Improvements** (always on):
1. InstanceNorm → Medical imaging standard
2. LeakyReLU → No dying neurons
3. Residual blocks → Deeper training
4. Strided conv → Learned downsampling
5. Multi-scale fusion → Multi-resolution features
6. Deep supervision → Better gradient flow
7. Dropout → Prevent overfitting

**3 Optional Features**:
8. Multi-Scale Transformer (+1.5-2.5% Dice, expensive)
9. Attention Gates (+1-2% Dice, medium cost)
10. Boundary Refinement (+2-3% Dice, cheap)

### Model Configs

| Config | Params | GPU | Target IoU |
|--------|--------|-----|-----------|
| V1 Baseline | 14M | RTX 3090 | 0.84 |
| Phase 2 Small | 37M | RTX 3090 | 0.80-0.82 |
| Phase 2 Large | 87M | A100 80GB | 0.85-0.90 |

---

## 🔑 Key Files

### Implementation
- `src/braintumnet/models/seg_unet_v2.py` (478 dòng)
- `src/braintumnet/models/braintumnet_v2.py` (170 dòng)
- `src/braintumnet/models/multiscale_transformer.py` (243 dòng)

### Configuration
- `configs/phases/phase2_small.yaml` - RTX 3090
- `configs/phases/phase2_a100.yaml` - A100 80GB
- `configs/models/segunetv2_phase2.yaml` - Model config

---

## 💡 File Quan Trọng Nhất

**[v2_06_UPGRADE_REASONING.md](v2_06_UPGRADE_REASONING.md)**

Giải thích **TẠI SAO** mỗi thay đổi:
- Vấn đề V1 gặp phải
- Giải pháp V2 đưa ra
- Lý do chọn giải pháp này
- Trade-offs và evidence

**Đọc file này để hiểu reasoning đằng sau mọi quyết định!**

---

## 📖 Liên Kết

### Tài Liệu Khác

- [Technical Docs](../technical/) - Tài liệu toàn bộ project
- [v_03a_SEGUNETV2_ARCHITECTURE.md](../technical/v_03a_SEGUNETV2_ARCHITECTURE.md) - V2 overview (shorter)

### External Resources

- nnU-Net paper: Medical imaging best practices
- ResNet paper: Residual connections
- Feature Pyramid Networks: Multi-scale fusion

---

## ❓ FAQ

**Q: Tôi nên đọc cả technical docs gốc không?**
- A: Có! Folder này chỉ về SegUNetV2. Technical docs gốc giải thích toàn bộ project.

**Q: V1 vs V2 - dùng cái nào?**
- A: V2 cho multi-class, V1 cho binary (nếu đủ tốt)

**Q: Tại sao tách ra folder riêng?**
- A: Tài liệu V2 rất chi tiết (20+ trang). Tách ra để dễ quản lý và không làm lộn xộn technical docs gốc.

---

## 📞 Contact

Nếu có câu hỏi:
1. Đọc [Upgrade Reasoning](v2_06_UPGRADE_REASONING.md) trước
2. Kiểm tra [Troubleshooting](../technical/v_09_TROUBLESHOOTING.md)
3. Xem code comments trong implementation files

---

**Bắt đầu đọc**: [INDEX →](v2_00_INDEX.md)

---

*Tài liệu được tạo: 2025-01-14*
*Total: ~15,000 dòng tài liệu về SegUNetV2 và Phase 2*
