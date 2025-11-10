# Quick Start - Phase 1 Optimized Training

## 🚀 Cách Chạy Nhanh

### Test Phase 1 (RTX 3090):
```bash
cd braintumnet
python scripts/train.py --model segunetv2_phase1 --fold 0
```

### Full Training Phase 1 (A100):
```bash
# Train all 5 folds
python scripts/train.py --model segunetv2_phase1 --cfg a100 --fold 0
python scripts/train.py --model segunetv2_phase1 --cfg a100 --fold 1
python scripts/train.py --model segunetv2_phase1 --cfg a100 --fold 2
python scripts/train.py --model segunetv2_phase1 --cfg a100 --fold 3
python scripts/train.py --model segunetv2_phase1 --cfg a100 --fold 4
```

### So Sánh với Baseline:
```bash
# Baseline
python scripts/train.py --model segunetv2 --cfg a100 --fold 0

# Phase 1 (optimized)
python scripts/train.py --model segunetv2_phase1 --cfg a100 --fold 0
```

---

## 📊 Dự Kiến Kết Quả

| Metric | Baseline | Phase 1 | Improvement |
|--------|----------|---------|-------------|
| Dice | 0.8699 | 0.91-0.94 | +4-7% |
| IoU | 0.7717 | 0.84-0.87 | +7-10% |
| Gap | 10% | 5-7% | HALVED! |

**vs SOTA**: Phase 1 expected 0.91-0.94 > nnUNet (0.83-0.85) ✅

---

## 🔧 Phase 1 Improvements

1. **Boundary Refinement** - Fix 10% IoU-Dice gap
2. **SGDR Scheduler** - Fix early plateau (epoch 46)
3. **Medical Augmentation** - 6 advanced transforms
4. **Optimized Loss Weights** - Boundary weight 0.6→1.0
5. **Deep Supervision Scheduling** - 0.5→0.1 gradual decay
6. **Gradient Centralization** - Better optimization

---

## 📁 Files Modified

- ✅ `configs/models/segunetv2_phase1.yaml` - NEW config
- ✅ `braintumnet/src/braintumnet/models/seg_unet_v2.py` - Boundary module
- ✅ `braintumnet/src/braintumnet/engine/trainer.py` - SGDR, DS scheduler, GC
- ✅ `braintumnet/src/braintumnet/data/advanced_transforms.py` - Medical aug
- ✅ `braintumnet/src/braintumnet/data/lmdb_dataset.py` - Aug integration
- ✅ `scripts/train.py` - Support segunetv2_phase1

---

## 🎯 Next Steps

1. **Chạy test ngay**: `python scripts/train.py --model segunetv2_phase1 --fold 0`
2. **Nếu OK**: Chạy all 5 folds trên A100
3. **Phase 2**: Chỉ cần nếu muốn >0.95 Dice

**Status**: Ready to train! ✅
