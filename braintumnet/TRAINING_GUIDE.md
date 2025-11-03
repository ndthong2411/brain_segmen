# BrainTumNet Training Guide - SOTA Models

Complete guide for training multiple SOTA models for brain tumor segmentation.

## 🚀 Quick Start

### Train on Default Hardware (RTX 3090 / Local)

```bash
# Swin-UNETR (recommended, best performance)
python scripts/train.py --model swin_unetr --fold 0

# nnU-Net (lightweight, efficient)
python scripts/train.py --model nnunet --fold 0

# UNETR (pure transformer)
python scripts/train.py --model unetr --fold 0

# SegUNetV2 (baseline with classification)
python scripts/train.py --model segunetv2 --fold 0
```

### Train on A100 Server

```bash
# Swin-UNETR on A100 (optimized for speed)
python scripts/train.py --model swin_unetr --cfg a100 --fold 0

# nnU-Net on A100
python scripts/train.py --model nnunet --cfg a100 --fold 0

# Any model on A100
python scripts/train.py --model <model_name> --cfg a100 --fold <0-4>
```

### Resume Training

```bash
# Auto-detect checkpoint
python scripts/train.py --model swin_unetr --fold 0 --resume

# Specify checkpoint path
python scripts/train.py --model swin_unetr --fold 0 --resume checkpoints/swin_unetr_fold0_last.pth
```

---

## 📊 Available Models

| Model | Parameters | Memory (A100) | Expected Dice | Best For |
|-------|-----------|---------------|---------------|----------|
| **Swin-UNETR** | 27M | ~20GB (bs=16) | 0.89-0.92 | **Best performance** |
| **nnU-Net** | 7M | ~12GB (bs=16) | 0.88-0.91 | Lightweight, proven |
| **UNETR** | 88M | ~28GB (bs=12) | 0.86-0.89 | Research baseline |
| **SegUNetV2** | 67M | ~24GB (bs=16) | 0.85-0.88 | Multi-task (seg+cls) |

---

## ⚙️ Configuration System

### Automatic Config Loading

Configs are automatically merged in this priority:

1. **base.yaml** - Common settings for all models
2. **models/{model}.yaml** - Model-specific parameters
3. **hardware_{hardware}.yaml** - Hardware optimizations (optional)

### Example: Swin-UNETR on A100

```bash
python scripts/train.py --model swin_unetr --cfg a100 --fold 0
```

**What happens:**
1. Loads `configs/base.yaml` (common settings)
2. Merges `configs/models/swin_unetr.yaml` (model params)
3. Merges `configs/hardware_a100.yaml` (A100 optimizations)
4. Sets batch_size=16, bfloat16, fused optimizer, etc.

### Hardware Configs

#### Default (3090 / Local)
- Batch size: 12-14 (model-dependent)
- AMP: float16
- Workers: 12
- No special optimizations

#### A100 (`--cfg a100`)
- Batch size: 16 (12 for UNETR)
- AMP: **bfloat16** (better on A100)
- Workers: 16
- Fused optimizer: True
- Channels last: True
- Prefetch factor: 8

---

## 📁 Config File Structure

```
configs/
├── base.yaml                    # Common settings
├── hardware_a100.yaml           # A100 optimizations
└── models/
    ├── swin_unetr.yaml         # Swin-UNETR params
    ├── nnunet.yaml             # nnU-Net params
    ├── unetr.yaml              # UNETR params
    └── segunetv2.yaml          # SegUNetV2 params
```

**No need to specify config files manually!** Just use `--model` and `--cfg` flags.

---

## 🎯 Training Examples

### Scenario 1: Train All Models on A100

```bash
# Train 5 folds of Swin-UNETR (best model)
for fold in 0 1 2 3 4; do
    python scripts/train.py --model swin_unetr --cfg a100 --fold $fold
done

# Train nnU-Net for comparison
for fold in 0 1 2 3 4; do
    python scripts/train.py --model nnunet --cfg a100 --fold $fold
done
```

### Scenario 2: Parallel Training on Multiple GPUs

```bash
# GPU 0: Swin-UNETR fold 0
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --model swin_unetr --cfg a100 --fold 0 &

# GPU 1: nnU-Net fold 0
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --model nnunet --cfg a100 --fold 0 &

wait
```

### Scenario 3: Quick Experiment on Local Machine

```bash
# Test Swin-UNETR with smaller batch
python scripts/train.py --model swin_unetr --fold 0
```

---

## 📈 Expected Training Time

### On A100 (batch=16, LMDB backend)

| Model | Time/Epoch | Total (400 epochs) |
|-------|-----------|-------------------|
| Swin-UNETR | ~3-4 min | ~20-25 hours |
| nnU-Net | ~2-3 min | ~15-20 hours |
| UNETR | ~4-5 min | ~25-30 hours |
| SegUNetV2 | ~3-4 min | ~20-25 hours |

### On RTX 3090 (batch=12-14, LMDB backend)

| Model | Time/Epoch | Total (400 epochs) |
|-------|-----------|-------------------|
| Swin-UNETR | ~5-6 min | ~35-40 hours |
| nnU-Net | ~3-4 min | ~20-25 hours |
| UNETR | ~6-8 min | ~40-50 hours |
| SegUNetV2 | ~5-6 min | ~35-40 hours |

---

## 🔧 Advanced Usage

### Custom Config Override

You can still use old-style config files:

```bash
python scripts/train.py --cfg configs/custom_config.yaml --fold 0
```

### Check Config Before Training

```bash
python scripts/test_config_system.py
```

This will show exactly what settings will be used for each model+hardware combo.

### Model Testing

```bash
# Test all models with dummy data
python scripts/test_models.py
```

Expected output:
```
✅ SEGUNETV2 TEST PASSED!
✅ SWIN_UNETR TEST PASSED!
✅ NNUNET TEST PASSED!
✅ UNETR TEST PASSED!
```

---

## 📊 Results Tracking

### Checkpoints

Saved automatically to `checkpoints/`:
- `{model}_fold{fold}_best.pth` - Best IoU checkpoint
- `{model}_fold{fold}_last.pth` - Latest checkpoint (for resume)

### Logs

- **Console logs**: `logs/{model}_fold{fold}.log`
- **TensorBoard**: `runs/{model}_fold{fold}/`

View TensorBoard:
```bash
tensorboard --logdir runs/
```

---

## 🎓 Training Tips

### 1. Start with Swin-UNETR
It has the best expected performance and is easy to train.

```bash
python scripts/train.py --model swin_unetr --cfg a100 --fold 0
```

### 2. Use A100 if Available
A100 optimizations (bfloat16, fused optimizer) provide 20-30% speedup.

### 3. Train Multiple Folds
For robust results, train all 5 folds and ensemble:

```bash
for fold in 0 1 2 3 4; do
    python scripts/train.py --model swin_unetr --cfg a100 --fold $fold
done
```

### 4. Monitor Training
Check TensorBoard for:
- Training/validation loss curves
- Dice/IoU metrics
- Learning rate schedule

### 5. Early Stopping
Training will automatically stop if no improvement for 100 epochs.

---

## 🐛 Troubleshooting

### CUDA Out of Memory

**Solution 1**: Reduce batch size
- Edit `configs/models/{model}.yaml`
- Or use default hardware (smaller batch)

**Solution 2**: Use gradient checkpointing (Swin-UNETR/UNETR)
- Already enabled by default (`use_checkpoint: true`)

### Missing Dependencies

```bash
pip install einops  # Required for MONAI transformers
pip install monai   # Required for Swin-UNETR and UNETR
```

### LMDB Data Not Found

Make sure LMDB database exists:
```bash
ls braintumnet/data/lmdb_processed_multiclass_full/
```

If not, convert from PNG:
```bash
python scripts/convert_to_lmdb.py \
    --input_dir braintumnet/data/processed_multiclass_full \
    --output_dir braintumnet/data/lmdb_processed_multiclass_full
```

---

## 📝 Summary

### Simple Commands

```bash
# Default hardware (3090)
python scripts/train.py --model swin_unetr --fold 0

# A100 server
python scripts/train.py --model swin_unetr --cfg a100 --fold 0

# Resume
python scripts/train.py --model swin_unetr --fold 0 --resume
```

### No Config Files Needed!
The system auto-selects the right config based on:
- `--model`: Which architecture
- `--cfg`: Which hardware (default or a100)
- `--fold`: Which fold (0-4)

**That's it! Happy training! 🚀**
