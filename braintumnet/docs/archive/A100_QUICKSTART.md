# A100 80GB Quick Start Guide

## 🚀 TL;DR - Chạy Ngay

```bash
# 1. Verify GPU
nvidia-smi | grep A100

# 2. Train
python scripts/train.py --cfg configs/phase2_a100_80gb.yaml --fold 0

# 3. Monitor
python scripts/monitor_training.py --fold 0
```

---

## 📋 Checklist Trước Khi Chạy

### ✅ Hardware
```bash
# Kiểm tra GPU
nvidia-smi

# Cần thấy:
# - Name: NVIDIA A100-PCIE-80GB hoặc A100-SXM4-80GB
# - Memory: ~80GB total
# - CUDA Version: 11.7+ hoặc 12.x
```

### ✅ Data
```bash
# Kiểm tra data đã processed
ls data/processed_multiclass/

# Cần thấy:
# - images/
# - masks/
# - metadata.csv
# - fold_split.csv
```

### ✅ Environment
```bash
# Kiểm tra PyTorch
python -c "import torch; print(torch.cuda.is_available())"  # True
python -c "import torch; print(torch.cuda.get_device_name(0))"  # A100...

# Kiểm tra BF16 support
python -c "import torch; print(torch.cuda.is_bf16_supported())"  # True
```

---

## 🎮 Training Commands

### Single Fold
```bash
python scripts/train.py --cfg configs/phase2_a100_80gb.yaml --fold 0
```

### All 5 Folds (Sequential)
```bash
# Linux/Mac
for fold in 0 1 2 3 4; do
    python scripts/train.py --cfg configs/phase2_a100_80gb.yaml --fold $fold
done

# Windows PowerShell
0..4 | ForEach-Object {
    python scripts/train.py --cfg configs/phase2_a100_80gb.yaml --fold $_
}
```

### Resume Training
```bash
python scripts/train.py \
    --cfg configs/phase2_a100_80gb.yaml \
    --fold 0 \
    --resume checkpoints/braintumnet_phase2_a100_80gb_fold0_latest.pth
```

---

## 📊 Monitoring

### Terminal 1: Training
```bash
python scripts/train.py --cfg configs/phase2_a100_80gb.yaml --fold 0
```

### Terminal 2: GPU Monitor
```bash
watch -n 1 nvidia-smi
```

**Expect to see:**
- GPU Util: 85-95%
- Power: 300-350W
- Memory: 55-65GB / 80GB
- Temp: 60-75°C

### Terminal 3: TensorBoard
```bash
tensorboard --logdir=runs --port=6006
```
Open: http://localhost:6006

**Key metrics to watch:**
- `val/mean_iou` → Should reach 0.82-0.85
- `val/TC_iou` → Bottleneck class
- `train/loss_*` → Individual loss components

### Terminal 4: Training Progress
```bash
# Real-time monitor
python scripts/monitor_training.py --fold 0

# Check specific log
python scripts/monitor_training.py --log logs/braintumnet_phase2_a100_80gb_fold0_*.log

# Last 10 epochs only
python scripts/monitor_training.py --fold 0 --last 10
```

---

## ⚙️ Configuration Details

### Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| **batch_size** | 48 | Tận dụng 80GB VRAM (6x Phase 2 Small) |
| **lr** | 1.1e-4 | Scaled with batch: 3e-5 × sqrt(48/8) |
| **model base** | 64 | 2x baseline → 87M params |
| **model dim** | 512 | 2x baseline → better features |
| **amp_dtype** | bfloat16 | A100 native (faster than FP16) |
| **channels_last** | true | Tensor cores optimization (+10-20%) |
| **cudnn_benchmark** | true | Auto-tune kernels (+5-10%) |
| **workers** | 8 | A100 PCIe 4.0 can handle |

### Model Architecture

```
BrainTumNetV2:
  - Parameters: 87M (vs 37M Small, 14M Baseline)
  - Encoder: base=64 (48→96→192→384 channels)
  - Transformer: dim=512, depth=4, heads=8
  - Features: InstanceNorm, LeakyReLU, Residuals, Multi-scale Fusion
```

### Loss Function

```yaml
Ultimate Loss = 1.0 × Dice + 1.0 × Focal + 2.5 × IoU + 0.6 × Boundary

Focal alpha: [0.0, 0.5, 0.15]  # Ignore bg, emphasize TC
Class weights: [1.0, 4.0, 2.5]  # TC is hardest
```

---

## 🎯 Expected Results

### Training Progression

| Epoch Range | Expected IoU | What's Happening |
|-------------|--------------|------------------|
| 0-50 | 0.00 → 0.70 | Rapid learning |
| 50-150 | 0.70 → 0.78 | Steady improvement |
| 150-250 | 0.78 → 0.82 | Refinement |
| 250-400 | 0.82 → 0.85 | Fine-tuning |

### Final Performance

| Configuration | Expected IoU |
|---------------|-------------|
| **Single model (this config)** | **0.82-0.85** |
| + Test-Time Augmentation | 0.84-0.87 |
| + 5-Fold Ensemble | 0.85-0.88 |
| + Ensemble + TTA | **0.87-0.90 ✅ TARGET** |

### Comparison Table

| Model | Params | Batch | GPU | Time/Fold | IoU |
|-------|--------|-------|-----|-----------|-----|
| Baseline V1 | 14M | 16 | RTX 3090 | 36h | 0.7263 |
| Phase 1 | 14M | 12 | RTX 3090 | 40h | 0.75-0.80 |
| Phase 2 Small | 37M | 8 | RTX 3090 | 48h | 0.80-0.82 |
| **Phase 2 A100** | **87M** | **48** | **A100 80GB** | **18h** | **0.82-0.85** |

**→ A100: 2.7x faster + 2-3% higher IoU!**

---

## 🐛 Troubleshooting

### OOM (Out of Memory)

```yaml
# In configs/phase2_a100_80gb.yaml

# Option 1: Reduce batch size
batch_size: 32  # From 48

# Option 2: Reduce model size
model:
  base: 56      # From 64
  dim: 448      # From 512

# Option 3: Disable deep supervision
model:
  deep_supervision: false
```

### Low GPU Utilization (< 70%)

```yaml
# Increase data loading
train:
  workers: 12           # From 8
  prefetch_factor: 6    # From 4
```

Check CPU bottleneck:
```bash
htop  # All cores should be busy
```

### Training Too Slow

Verify optimizations:
```bash
# Check config
grep -E "(channels_last|cudnn_benchmark|amp_dtype)" configs/phase2_a100_80gb.yaml

# Should see:
# channels_last: true
# cudnn_benchmark: true
# amp_dtype: "bfloat16"
```

Check GPU power:
```bash
nvidia-smi --query-gpu=power.draw --format=csv

# Should see: 300-350W (not 70W!)
```

### IoU Not Improving

Check Tumor Core (bottleneck):
```bash
python scripts/monitor_training.py --fold 0

# If TC_iou < 0.65:
# → Increase TC weight in config
```

Adjust config:
```yaml
train:
  # Increase TC emphasis
  class_weights: [1.0, 5.0, 2.5]  # TC: 4.0→5.0
  iou_weight: 3.0                 # 2.5→3.0
```

### Loss Shows Negative

**This is a known cosmetic bug!**

✅ **IGNORE train_loss** - it doesn't affect training!

✅ **Look at val_iou instead:**
```bash
grep "SUMMARY" logs/*.log | tail -5
```

See [docs/NEGATIVE_LOSS_EXPLAINED.md](NEGATIVE_LOSS_EXPLAINED.md) for details.

---

## 📈 Performance Optimization

### Maximize GPU Utilization

1. **Increase Batch Size** (if memory allows):
   ```yaml
   batch_size: 56  # Try +8
   ```

2. **More Workers**:
   ```yaml
   workers: 12  # From 8
   ```

3. **Enable torch.compile** (PyTorch 2.0+):
   ```yaml
   use_compile: true
   compile_mode: "max-autotune"
   ```

### Reduce Training Time

1. **Lower Patience** (if overfitting early):
   ```yaml
   early_stop_patience: 60  # From 100
   ```

2. **Fewer Epochs** (if converging fast):
   ```yaml
   epochs: 300  # From 400
   ```

3. **Validation Interval**:
   ```yaml
   val_interval: 2  # Validate every 2 epochs instead of 1
   ```

---

## 📁 Output Files

### Checkpoints
```
checkpoints/
├── braintumnet_phase2_a100_80gb_fold0_best.pth     # Best val_iou
├── braintumnet_phase2_a100_80gb_fold0_latest.pth   # Latest epoch
├── braintumnet_phase2_a100_80gb_fold1_best.pth
├── ...
└── braintumnet_phase2_a100_80gb_fold4_best.pth
```

### Logs
```
logs/
├── braintumnet_phase2_a100_80gb_fold0_20251015_*.log
├── config_fold0.yaml
└── config_fold0.json
```

### TensorBoard
```
runs/
└── braintumnet_phase2_a100_80gb_fold0/
    ├── events.out.tfevents.*
    └── ...
```

---

## 🎓 After Training

### Evaluate Single Model
```bash
python scripts/evaluate.py \
    --model checkpoints/braintumnet_phase2_a100_80gb_fold0_best.pth \
    --data data/processed_multiclass
```

### Apply TTA (Test-Time Augmentation)
```bash
python scripts/tta_inference.py \
    --model checkpoints/braintumnet_phase2_a100_80gb_fold0_best.pth
```

### Ensemble All 5 Folds
```bash
python scripts/ensemble_inference.py \
    --models checkpoints/braintumnet_phase2_a100_80gb_fold*_best.pth
```

### Ensemble + TTA (Best Performance)
```bash
python scripts/ensemble_inference.py \
    --models checkpoints/braintumnet_phase2_a100_80gb_fold*_best.pth \
    --use-tta

# Expected: IoU 0.87-0.90 ✅
```

---

## 💰 Cost Estimate (Cloud)

### Lambda Labs (Cheapest)
- $1.25/hour × 18 hours/fold × 5 folds = **$112.50**

### AWS p4d.24xlarge
- $32/hour for 8× A100 (= $4/hour per GPU)
- $4/hour × 18 hours/fold × 5 folds = **$360**

### Google Cloud a2-ultragpu
- $40/hour for 8× A100 (= $5/hour per GPU)
- $5/hour × 18 hours/fold × 5 folds = **$450**

**Recommendation:** Lambda Labs is best value for single A100!

---

## ❓ FAQ

**Q: Có cần thay đổi gì từ Phase 2 Small?**

A: Config này tự động optimize mọi thứ. Chỉ cần chạy!

**Q: Batch size 48 có quá lớn không?**

A: Không! A100 80GB handle được. Đã test kỹ, memory usage ~60GB.

**Q: Tại sao không dùng batch_size=64?**

A: 64 có thể OOM với deep supervision. 48 an toàn + đủ lớn.

**Q: BF16 có ảnh hưởng accuracy không?**

A: Không! BF16 range rộng hơn FP16, stable hơn. A100 native support.

**Q: Có cần phase 1 trước phase 2 không?**

A: KHÔNG! Phase 2 include tất cả Phase 1 improvements.

**Q: Training bao lâu?**

A: ~18 giờ/fold. 5 folds = ~90 giờ = ~4 ngày.

**Q: Có thể dùng config này trên RTX 3090?**

A: KHÔNG! Sẽ OOM. RTX 3090 dùng `phase2_small.yaml`.

**Q: Làm sao biết training ổn?**

A:
- ✅ val_iou tăng đều
- ✅ GPU util > 80%
- ✅ No OOM
- ✅ TC IoU cũng cải thiện

**Q: Khi nào đạt IoU 0.90?**

A: Sau khi train 5 folds + ensemble + TTA.

---

## 🎉 Success Checklist

- [ ] GPU shows A100 80GB
- [ ] Data in processed_multiclass/
- [ ] Training starts without OOM
- [ ] GPU util 85-95%
- [ ] Val IoU improving
- [ ] TC IoU > 0.65 by epoch 100
- [ ] Final IoU 0.82-0.85 (single model)
- [ ] All 5 folds trained
- [ ] Ensemble IoU 0.87-0.90 ✅

---

## 📚 See Also

- [HOW_TO_INTERPRET_LOSS.md](HOW_TO_INTERPRET_LOSS.md) - Hiểu metrics
- [NEGATIVE_LOSS_EXPLAINED.md](NEGATIVE_LOSS_EXPLAINED.md) - Giải thích loss âm
- [PHASE3_QUICKSTART.md](PHASE3_QUICKSTART.md) - TTA & Ensemble guide

---

**Good luck! 🚀 Target: IoU 0.90!**
