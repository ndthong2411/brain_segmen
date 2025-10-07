# BrainTumNet - Quick Start Cheatsheet

**Your System:** Python 3.10.18 | RTX 3090 (24GB) | Windows

---

## 🚀 One-Page Quick Reference

### Setup (10 minutes)
```bash
# 1. Create environment
python -m venv .venv
.\.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 3090
```

---

### Data Preparation (45-90 minutes)

**Download BraTS2020:**
- Kaggle: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data
- Extract to: `braintumnet/data/raw/`

**Preprocess (choose one):**

```bash
# OPTION A: Full dataset (RECOMMENDED)
python scripts/prepare_brats2020_h5.py \
  --h5_root data/raw \
  --meta_csv data/raw/meta_data.csv \
  --out data/processed_full

# OPTION B: Multimodal (BEST performance)
python scripts/prepare_brats2020_h5.py \
  --h5_root data/raw \
  --meta_csv data/raw/meta_data.csv \
  --out data/processed_multimodal \
  --multimodal

# OPTION C: Quick test (code testing only)
python scripts/prepare_brats2020_h5.py \
  --h5_root data/raw \
  --meta_csv data/raw/meta_data.csv \
  --out data/test \
  --max_slices 1000
```

**Update config:**
```yaml
# Edit configs/optimized.yaml
data:
  proc_root: "data/processed_full"  # Match your preprocessing!
```

---

### Training (10-40 hours)

```bash
# Single fold
python scripts/train.py --cfg configs/optimized.yaml --fold 0

# All 5 folds (for publication)
for /L %i in (0,1,4) do python scripts/train.py --cfg configs/optimized.yaml --fold %i
```

---

### Monitoring (during training)

**Console output:**
```
Epoch 42/200 [Train]: 100%|████| 2562/2562
  loss: 0.6330  lr: 1.5e-4
Epoch 42/200 [Val]:   100%|████| 640/640
  iou: 0.5823  dice: 0.7364
→ New best IoU: 0.5823, checkpoint saved!
```

**TensorBoard (separate terminal):**
```bash
tensorboard --logdir=runs
# Open: http://localhost:6006
```

---

### Evaluation (5 minutes)

```bash
python scripts/evaluate.py \
  --cfg configs/optimized.yaml \
  --fold 0 \
  --ckpt checkpoints/braintumnet_best_fold0.pth
```

**Expected results:**
- IoU: 0.60-0.70
- Dice: 0.75-0.82
- Accuracy: 95-98%

---

### Visualization (10 minutes)

```bash
# Visualize predictions
python scripts/visualize_batch.py \
  --cfg configs/optimized.yaml \
  --fold 0 \
  --ckpt checkpoints/braintumnet_best_fold0.pth \
  --output results.png

# Single prediction
python scripts/predict.py \
  --ckpt checkpoints/braintumnet_best_fold0.pth \
  --image data/processed_full/images/vol185_slice75.png \
  --output prediction.png
```

---

## 📊 Expected Performance

| Dataset | Cases | Slices | IoU | Dice | Time |
|---------|-------|--------|-----|------|------|
| Current (13 cases) | 13 | 2,000 | 0.44 | 0.63 | 3 hrs |
| Full (369 cases) | 369 | 51,000 | 0.69 | 0.82 | 38 hrs |
| Multimodal | 369 | 51,000 | 0.73 | 0.86 | 45 hrs |

---

## 🔧 Common Commands

```bash
# Check GPU usage
nvidia-smi

# View training log
type logs\braintumnet_optimized_fold0.log | more

# Count processed files
dir /b data\processed_full\images | find /c ".png"

# Check checkpoint
python -c "import torch; print(torch.load('checkpoints/braintumnet_best_fold0.pth')['epoch'])"

# Compare runs
python scripts/compare_runs.py \
  --run_dirs runs/exp1 runs/exp2 \
  --output comparison.png
```

---

## 📁 File Locations

```
Important files after training:

checkpoints/
  └── braintumnet_best_fold0.pth      # Best model (use this!)

logs/
  ├── braintumnet_optimized_fold0.log # Detailed training log
  ├── metrics_fold0.csv               # All metrics (Excel-ready)
  └── best_metrics.json               # Best values

runs/
  └── braintumnet_optimized_fold0/    # TensorBoard data
```

---

## ⚠️ Troubleshooting

**Out of GPU memory:**
```yaml
# configs/optimized.yaml
train:
  batch_size: 8  # Reduce from 16
```

**Training too slow:**
```yaml
# configs/optimized.yaml
train:
  workers: 2  # Reduce from 4
```

**TensorBoard not showing:**
```bash
rd /s /q runs\.cache
tensorboard --logdir=runs --reload_interval=5
```

---

## 📚 Documentation

- **Complete Guide:** [docs/COMPLETE_WORKFLOW.md](docs/COMPLETE_WORKFLOW.md)
- **Workflow Diagram:** [docs/WORKFLOW_DIAGRAM.md](docs/WORKFLOW_DIAGRAM.md)
- **Preprocessing Guide:** [docs/PREPROCESSING_GUIDE.md](docs/PREPROCESSING_GUIDE.md)
- **Bug Fixes:** [docs/BUG_FIXES.md](docs/BUG_FIXES.md)

---

## ✅ Minimal Working Example (Full Pipeline)

```bash
# Complete workflow in one script

# 1. Setup
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Preprocess (assumes you have BraTS2020 in data/raw/)
python scripts/prepare_brats2020_h5.py \
  --h5_root data/raw \
  --meta_csv data/raw/meta_data.csv \
  --out data/processed_full

# 3. Train
python scripts/train.py --cfg configs/optimized.yaml --fold 0

# 4. Evaluate
python scripts/evaluate.py \
  --cfg configs/optimized.yaml \
  --fold 0 \
  --ckpt checkpoints/braintumnet_best_fold0.pth

# 5. Visualize
python scripts/visualize_batch.py \
  --cfg configs/optimized.yaml \
  --fold 0 \
  --ckpt checkpoints/braintumnet_best_fold0.pth \
  --output results.png

# Done! 🎉
```

---

**Total time:** ~40-50 hours (mostly training)
**Disk space:** ~15 GB
**Expected Dice score:** 0.75-0.82
**Publication ready:** ✅ Workshop/Conference
