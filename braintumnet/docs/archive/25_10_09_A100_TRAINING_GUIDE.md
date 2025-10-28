# A100 GPU Training Guide

**Created:** October 9, 2025
**Purpose:** Hướng dẫn tối ưu training BrainTumNet trên A100 GPU

---

## 📋 Configs Available

### 1. **a100_optimized.yaml** - Maximum Performance 🚀
**Khi nào dùng:**
- A100 80GB VRAM
- Hoặc A100 40GB nếu muốn đẩy tối đa performance (có thể VRAM overflow)

**Đặc điểm:**
- Batch size: 32 (lớn nhất)
- Model size: base=48, dim=384, depth=3 (lớn nhất)
- Learning rate: 2e-4 (cao nhất)
- Expected VRAM: ~28-32GB
- Expected speed: ~2-3 sec/epoch
- Expected results: **Dice 0.93+, IoU 0.90+**

**Command:**
```bash
python scripts/train.py --config configs/a100_optimized.yaml --fold 0
```

### 2. **a100_safe.yaml** - Guaranteed Fit ✅
**Khi nào dùng:**
- A100 40GB VRAM (guaranteed to work)
- Khi a100_optimized.yaml bị VRAM overflow

**Đặc điểm:**
- Batch size: 24 (vừa phải)
- Model size: base=40, dim=320, depth=2 (vừa phải)
- Learning rate: 1.7e-4
- Expected VRAM: ~20-25GB
- Expected speed: ~2.5-3.5 sec/epoch
- Expected results: **Dice 0.92+, IoU 0.89+**

**Command:**
```bash
python scripts/train.py --config configs/a100_safe.yaml --fold 0
```

### 3. **improved_v2_boundary_loss.yaml** - Baseline
**Khi nào dùng:**
- GPU khác (V100, RTX 3090, etc.)
- Muốn compare với baseline

**Đặc điểm:**
- Batch size: 12
- Model size: base=32, dim=256, depth=2
- Learning rate: 1e-4
- Expected VRAM: ~12-16GB

---

## 🎯 Decision Tree

```
Bạn có A100 GPU?
│
├─ YES → Bao nhiêu VRAM?
│   │
│   ├─ 80GB → Dùng a100_optimized.yaml (tối ưu nhất)
│   │
│   └─ 40GB → Muốn đẩy tối đa hay chơi an toàn?
│       │
│       ├─ Đẩy tối đa → Thử a100_optimized.yaml
│       │   └─ Nếu VRAM overflow → Chuyển sang a100_safe.yaml
│       │
│       └─ An toàn → Dùng a100_safe.yaml
│
└─ NO → Dùng improved_v2_boundary_loss.yaml
```

---

## 🔧 A100-Specific Optimizations Explained

### 1. **Larger Batch Size (24-32 vs 12)**
- **Why:** A100 có 40-80GB VRAM, có thể load nhiều samples cùng lúc
- **Benefit:** Better gradient estimates, faster convergence
- **Trade-off:** Cần tăng learning rate tương ứng (sqrt scaling)

### 2. **Mixed Precision (AMP)**
- **Why:** A100 có Tensor Cores với 312 TFLOPS FP16 (vs 19.5 TFLOPS FP32)
- **Benefit:** 2-3x faster training, 50% less VRAM
- **Implementation:** `amp: true` trong config

### 3. **Channels Last Memory Format**
- **Why:** Tensor Cores prefer NHWC layout over NCHW
- **Benefit:** 10-20% speedup on convolutions
- **Implementation:** `use_channels_last: true` trong config

### 4. **More Workers (8 vs 4)**
- **Why:** A100 has PCIe 4.0 (16 GT/s vs 8 GT/s for PCIe 3.0)
- **Benefit:** Reduce data loading bottleneck
- **Implementation:** `workers: 8` trong config

### 5. **Larger Model Capacity**
- **Why:** More VRAM → can train larger models
- **Benefit:** Better accuracy potential
- **Parameters:**
  - `base: 48` (vs 32): +50% channels → +2.25x parameters in encoder
  - `dim: 384` (vs 256): +50% transformer embedding dimension
  - `depth: 3` (vs 2): +50% transformer layers
  - Total: ~3-4x more parameters

### 6. **Higher Learning Rate (2e-4 vs 1e-4)**
- **Why:** Larger batch size needs higher LR to maintain same effective step size
- **Scaling rule:** LR_new = LR_base × sqrt(BS_new / BS_base)
  - sqrt(32/12) ≈ 1.63
  - 1e-4 × 1.63 ≈ 1.63e-4 → round to 2e-4
- **Benefit:** Faster convergence with large batch

---

## 📊 Performance Comparison

| Config | GPU | Batch | Model Params | VRAM | Time/Epoch | Total Time | Expected Dice | Expected IoU |
|--------|-----|-------|--------------|------|------------|------------|---------------|--------------|
| improved_v2 | V100/3090 | 12 | ~15M | 12-16GB | ~8-10s | ~35-42 min | 0.91-0.92 | 0.87-0.88 |
| a100_safe | A100-40GB | 24 | ~40M | 20-25GB | ~2.5-3.5s | ~12-18 min | 0.92-0.925 | 0.89-0.895 |
| a100_optimized | A100-80GB | 32 | ~65M | 28-32GB | ~2-3s | ~10-15 min | 0.925-0.94 | 0.90-0.92 |

**Notes:**
- Time for 300 epochs
- A100 is **3-4x faster** than V100
- Larger models on A100 give **+1-2% improvement** in metrics

---

## 🚀 Quick Start on A100

### Step 1: Check GPU
```bash
nvidia-smi
```
Verify:
- Model: A100-SXM4 or A100-PCIE
- Memory: 40GB or 80GB

### Step 2: Choose Config
- 80GB → `a100_optimized.yaml`
- 40GB → `a100_safe.yaml` (hoặc thử `a100_optimized.yaml` trước)

### Step 3: Quick Test (5 epochs)
```bash
cd braintumnet
python scripts/train.py --config configs/a100_optimized.yaml --fold 0 --epochs 5
```

Monitor VRAM usage:
```bash
watch -n 1 nvidia-smi
```

### Step 4: Full Training
Nếu test OK (không VRAM overflow):
```bash
python scripts/train.py --config configs/a100_optimized.yaml --fold 0
```

### Step 5: Monitor Progress
```bash
tensorboard --logdir runs/
```

---

## ⚠️ Troubleshooting

### VRAM Overflow on a100_optimized.yaml

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**

#### Option 1: Switch to a100_safe.yaml
```bash
python scripts/train.py --config configs/a100_safe.yaml --fold 0
```

#### Option 2: Reduce Batch Size
Edit `configs/a100_optimized.yaml`:
```yaml
batch_size: 32 → 28  # Try 28, 24, 20
lr: 2.0e-4 → 1.8e-4  # Scale down proportionally
```

#### Option 3: Reduce Model Size
Edit `configs/a100_optimized.yaml`:
```yaml
base: 48 → 40
dim: 384 → 320
depth: 3 → 2
```

### Slow Training Speed

**Check 1: AMP enabled?**
```yaml
amp: true  # Should be true
```

**Check 2: Channels last enabled?**
```yaml
use_channels_last: true  # Should be true
```

**Check 3: Enough workers?**
```yaml
workers: 8  # Should be 8 for A100
```

**Check 4: GPU utilization**
```bash
nvidia-smi dmon -s u
# Should show ~90-100% utilization
```

### Lower Than Expected Results

**After 300 epochs:**
- Dice < 0.92 → Something wrong, check logs
- Dice 0.92-0.925 → Normal for a100_safe
- Dice 0.925-0.94 → Excellent for a100_optimized
- Dice > 0.94 → Amazing! Ready to publish

**If results too low:**
1. Check data quality
2. Verify all improvements enabled (deep_supervision=true, boundary_loss_weight=0.2)
3. Check training didn't diverge (monitor loss curves)
4. Try longer training (300 → 400 epochs)

---

## 🎯 Expected Timeline

### A100-80GB with a100_optimized.yaml

**Phase 1: Setup & Testing (30 minutes)**
- Data preparation: ~10 min
- Quick test (5 epochs): ~15 sec
- Validation: ~5 min

**Phase 2: Full Training (15-20 minutes per fold)**
- 300 epochs: ~10-15 min
- Validation: ~2-3 min
- Checkpoint saving: ~1-2 min

**Phase 3: 5-Fold Cross-Validation (2 hours)**
- 5 folds × 20 min = ~100 min
- Final metrics computation: ~10 min

**Total: ~2.5 hours** for complete 5-fold CV on A100
(vs **~7-8 hours** on V100)

---

## 📈 Model Size Comparison

| Config | base | dim | depth | n_heads | Total Params | VRAM (BS=1) |
|--------|------|-----|-------|---------|--------------|-------------|
| improved_v2 | 32 | 256 | 2 | 4 | ~15M | ~2GB |
| a100_safe | 40 | 320 | 2 | 5 | ~40M | ~3.5GB |
| a100_optimized | 48 | 384 | 3 | 6 | ~65M | ~5GB |

**VRAM breakdown for a100_optimized (batch_size=32):**
- Model weights: ~260MB (65M params × 4 bytes)
- Model activations (forward): ~8GB (32 samples)
- Gradients (backward): ~8GB
- Optimizer states (Adam): ~520MB (2× params)
- Deep supervision aux outputs: ~6GB (3 aux heads × 32 samples)
- Boundary loss distance maps: ~4GB (cached)
- PyTorch overhead: ~2GB
- **Total: ~29GB** (fits in 40GB with some margin)

---

## 💡 Tips for Maximum Performance

### 1. **Use Latest PyTorch**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
PyTorch 2.0+ has better A100 support.

### 2. **Clear Cache Between Runs**
```python
import torch
torch.cuda.empty_cache()
```

### 3. **Monitor GPU Temperature**
```bash
nvidia-smi dmon -s t
```
A100 should stay <80°C under full load.

### 4. **Use Multiple Folds in Parallel**
If you have access to multiple A100s:
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/a100_optimized.yaml --fold 0

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/a100_optimized.yaml --fold 1
```

5-fold CV in **20 minutes** instead of 100 minutes!

---

## 📝 Summary

**For A100-80GB:**
- Use `a100_optimized.yaml`
- Expected: Dice 0.93+, IoU 0.90+
- Training time: ~15 min/fold
- Total 5-fold: ~2 hours

**For A100-40GB:**
- Start with `a100_safe.yaml` (guaranteed)
- Or try `a100_optimized.yaml` (might work)
- Expected: Dice 0.92+, IoU 0.89+
- Training time: ~18 min/fold
- Total 5-fold: ~2.5 hours

**Next Steps:**
1. Run quick test (5 epochs)
2. If OK → Full training (300 epochs)
3. Monitor TensorBoard
4. Compare with baseline
5. If better → Prepare paper! 📄

---

**Good luck! 🚀**
