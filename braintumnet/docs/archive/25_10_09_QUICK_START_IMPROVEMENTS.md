# 🚀 Quick Start: Deep Supervision + Boundary Loss

**TL;DR**: Chạy ngay 3 lệnh này để test improvements:

```bash
# 1. Test implementation (2 phút)
cd braintumnet
python scripts/test_improvements.py

# 2. Quick train (30 phút, 5 epochs để verify)
python scripts/train.py --config configs/improved_v1_deep_supervision.yaml --fold 0 --epochs 5

# 3. Full train (nếu test OK)
python scripts/train.py --config configs/improved_v1_deep_supervision.yaml --fold 0
```

---

## 📊 What Changed?

### Deep Supervision ✅
- Thêm 3 auxiliary segmentation heads ở decoder levels
- Expected: +2-3% IoU, +1-2% Dice
- Chi phí: Minimal (chỉ thêm 3 Conv layers)

### Boundary Loss ✅
- Penalize boundary errors mạnh hơn
- Expected: +3-5% IoU (TARGET CHÍNH)
- Chi phí: ~5-10ms/batch (có caching)

### Training Schedule ✅
- 250 epochs (từ 150)
- Cosine annealing (từ ReduceLROnPlateau)
- LR 1e-4 (từ 1.5e-4) - match paper

---

## 📁 Files Changed

```
src/braintumnet/models/
├── seg_unet.py          ✅ +deep_supervision parameter, auxiliary heads
├── braintumnet.py       ✅ Handle deep supervision outputs

src/braintumnet/
├── losses/base.py            ✅ +BoundaryLoss class, +boundary_w parameter

src/braintumnet/engine/
├── trainer.py           ✅ Compute auxiliary losses, +boundary loss

configs/
├── improved_v1_deep_supervision.yaml    ✅ Config v1 (deep supervision only)
└── improved_v2_boundary_loss.yaml       ✅ Config v2 (+ boundary loss)

scripts/
└── test_improvements.py                 ✅ Test script

docs/
├── IMPROVEMENTS_CHANGELOG.md            ✅ Chi tiết đầy đủ
└── QUICK_START_IMPROVEMENTS.md          ✅ This file
```

---

## 🧪 Testing

### Step 1: Verify Implementation

```bash
python scripts/test_improvements.py
```

**Expected output**:
```
TEST 1: Deep Supervision                              ✅ PASSED
TEST 2: Deep Supervision Disabled                     ✅ PASSED
TEST 3: Boundary Loss                                 ✅ PASSED
TEST 4: MultiTaskLoss with Boundary                   ✅ PASSED
TEST 5: Auxiliary Loss Computation                    ✅ PASSED

🎉 ALL TESTS PASSED!
```

**Nếu có test FAILED**: Check error message và fix trước khi train

---

## 🏋️ Training

### Option 1: Deep Supervision Only (Week 1)

```bash
# Quick test (5 epochs, ~30 phút)
python scripts/train.py \
    --config configs/improved_v1_deep_supervision.yaml \
    --fold 0 \
    --epochs 5

# Full training (250 epochs, ~2-3 ngày)
python scripts/train.py \
    --config configs/improved_v1_deep_supervision.yaml \
    --fold 0
```

**Target**: Dice 0.92+, IoU 0.86+

### Option 2: Deep Supervision + Boundary Loss (Week 2)

```bash
# After v1 shows good results
python scripts/train.py \
    --config configs/improved_v2_boundary_loss.yaml \
    --fold 0
```

**Target**: Dice 0.925+, IoU 0.88+

---

## 📈 Monitoring

### TensorBoard

```bash
tensorboard --logdir runs/
# Open: http://localhost:6006
```

**Key metrics to watch**:
- `val/iou` - PRIMARY METRIC (should increase)
- `val/dice` - Should also increase
- `train/loss_total` - Should decrease smoothly

### Logs

```bash
# Real-time monitoring
tail -f logs/braintumnet_improved_v1_deep_supervision_fold0.log

# Check best results
grep "Best IoU" logs/braintumnet_improved_v1_deep_supervision_fold0.log
```

---

## 🎯 Expected Results Timeline

```
Baseline:    Dice 0.909, IoU 0.835
   ↓ (Week 1: Deep Supervision)
Improved v1: Dice 0.920, IoU 0.860 (+2.5% IoU)
   ↓ (Week 2: + Boundary Loss)
Improved v2: Dice 0.925, IoU 0.880 (+4.5% IoU)
   ↓ (Week 3: + TTA)
Final:       Dice 0.930, IoU 0.900 (+6.5% IoU)
```

---

## 🐛 Troubleshooting

### "CUDA out of memory"
```yaml
# Edit config:
train:
  batch_size: 8  # Giảm từ 12
```

### "Loss is NaN"
```yaml
# Edit config:
train:
  boundary_loss_weight: 0.1  # Giảm từ 0.2
```

### "No improvement after 100 epochs"
- Check TensorBoard curves
- Có thể đã converge sớm (good thing!)
- Hoặc LR quá thấp (tăng lên 2e-4)

---

## 📚 Full Documentation

Xem file `IMPROVEMENTS_CHANGELOG.md` cho:
- ✅ Chi tiết implementation
- ✅ Code explanations
- ✅ Hyperparameter tuning
- ✅ Debugging guide
- ✅ References

---

## ✅ Checklist

**Before Training**:
- [ ] Run `python scripts/test_improvements.py` → All tests pass
- [ ] Check GPU memory: `nvidia-smi` → >6GB free
- [ ] Config file reviewed

**During Training (First Hour)**:
- [ ] TensorBoard running
- [ ] Check train/loss decreasing
- [ ] Check no NaN/Inf values
- [ ] GPU utilization >80%

**After 50 Epochs**:
- [ ] Val Dice > 0.90
- [ ] Val IoU > 0.83 (should be improving)
- [ ] No early stopping triggered

---

## 🚨 Decision Points

### After 5 Epochs Test:
- ✅ Loss decreasing normally → Continue
- ❌ Loss NaN/Inf → Reduce boundary_loss_weight
- ❌ OOM error → Reduce batch_size

### After 50 Epochs:
- ✅ IoU > 0.85 → Great! Continue to 250
- ⚠️ IoU < 0.84 → May need tuning (check logs)
- ❌ IoU same as baseline → Something wrong, debug

### After 250 Epochs (v1):
- ✅ IoU > 0.86 → Move to v2 (boundary loss)
- ⚠️ IoU 0.84-0.86 → Acceptable, still try v2
- ❌ IoU < 0.84 → Debug before v2

---

## 🎓 Next Steps

1. **Week 1**: Train v1 (deep supervision)
2. **Week 2**: Train v2 (+ boundary loss)
3. **Week 3**: Implement Test-Time Augmentation
4. **Week 4**: Full 5-fold + Ensemble

**Paper target**: Dice 0.93-0.94, IoU 0.91-0.92

---

**Bắt đầu ngay**: `python scripts/test_improvements.py` ✨
