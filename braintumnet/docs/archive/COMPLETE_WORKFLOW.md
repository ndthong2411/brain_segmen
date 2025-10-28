# BrainTumNet: Complete Workflow from Start to End

**Your System:** Python 3.10.18 | RTX 3090 (24GB) | Windows

This guide shows **every command** you need to run from project setup to final results.

---

## 📋 Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Data Preparation](#3-data-preparation)
4. [Training](#4-training)
5. [Monitoring](#5-monitoring)
6. [Evaluation](#6-evaluation)
7. [Inference & Visualization](#7-inference--visualization)
8. [Results Analysis](#8-results-analysis)

---

## 1. Prerequisites

### Check Your System
```bash
# Check Python version (need 3.8+)
python --version
# Output: Python 3.10.18 ✓

# Check GPU
nvidia-smi
# Output: RTX 3090, 24GB ✓

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
# Should output: True
```

### Download BraTS2020 Dataset

**Option A: Kaggle (Recommended - HDF5 format)**
1. Go to: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data
2. Download `archive.zip` (~8 GB)
3. Extract to: `E:\thong\code\brain_segmen\braintumnet\data\raw\`

**Option B: Official (NIfTI format)**
1. Register at: http://braintumorsegmentation.org/
2. Download BraTS2020 Training Data
3. Extract to: `E:\thong\code\brain_segmen\braintumnet\data\raw\`

**You should have:**
```
braintumnet/data/raw/
├── meta_data.csv          (metadata for slices)
├── BraTS20_001.h5         (or .nii.gz files)
├── BraTS20_002.h5
└── ... (370 .h5 files for full dataset)
```

---

## 2. Environment Setup

### Step 2.1: Create Virtual Environment
```bash
# Navigate to project
cd E:\thong\code\brain_segmen\braintumnet

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# You should see (.venv) in your prompt
```

### Step 2.2: Install Dependencies
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# This installs:
# - torch>=2.1 (PyTorch with CUDA)
# - torchvision>=0.16
# - numpy, pillow, pyyaml
# - h5py (for HDF5 files)
# - tqdm (progress bars)
# - matplotlib (visualization)
# - tensorboard (monitoring)
# ... and more

# Expected time: 5-10 minutes
```

### Step 2.3: Verify Installation
```bash
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Expected output:
# PyTorch: 2.1.0+cu121 (or similar)
# CUDA: True

# Test GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 3090
```

---

## 3. Data Preparation

### Step 3.1: Preprocess Dataset

**⚠️ IMPORTANT DECISION:**

**Option A: Full Dataset (RECOMMENDED for real training)**
```bash
# Process ALL 370 patients (~60,000 slices)
# Time: 30-60 minutes | Disk space: ~3-5 GB

python scripts/prepare_brats2020_h5.py \
  --h5_root data/raw \
  --meta_csv data/raw/meta_data.csv \
  --out data/processed_full

# Expected output:
# ======================================================================
# BraTS2020 PREPROCESSING CONFIGURATION
# ======================================================================
# Input directory:     data/raw
# Output directory:    data/processed_full
# Mode:                Single-modal (T1CE only)
# Slice limit:         None (processing ALL data ✓)
# ======================================================================
#
# Found 57420 slices in metadata
# ✓ Processing ALL slices (full dataset mode)
#
# Processing slices: 100%|████████████| 57420/57420 [45:32<00:00]
#
# ======================================================================
# PREPROCESSING SUMMARY
# ======================================================================
# ✓ Successfully processed:    51234
# ✗ Skipped (no tumor):        3852
# ✗ Skipped (errors):          12
# Unique cases:                369
# ======================================================================
```

**Option B: Quick Test (for testing code only)**
```bash
# Process only 1000 slices for quick testing
# Time: 2-3 minutes | Disk space: ~50 MB

python scripts/prepare_brats2020_h5.py \
  --h5_root data/raw \
  --meta_csv data/raw/meta_data.csv \
  --out data/test \
  --max_slices 1000

# ⚠️ WARNING: This is ONLY for code testing!
# Results will NOT be publishable (IoU ~0.40-0.50)
```

**Option C: Multimodal (best performance)**
```bash
# Process all 4 MRI sequences (FLAIR, T1, T1CE, T2)
# Time: 60-90 minutes | Disk space: ~12-20 GB

python scripts/prepare_brats2020_h5.py \
  --h5_root data/raw \
  --meta_csv data/raw/meta_data.csv \
  --out data/processed_multimodal \
  --multimodal

# Expected: +5-10% better Dice score
# Best for publication-quality results
```

### Step 3.2: Verify Preprocessed Data
```bash
# Check number of files
dir data\processed_full\images | find /c ".png"
# Should show: 30,000-60,000 files

# Check data structure
tree /F data\processed_full | more

# Expected structure:
# data/processed_full/
# ├── images/                  (30,000+ PNG files)
# ├── masks/                   (30,000+ PNG files)
# ├── labels.csv               (200-370 cases)
# ├── mapping.csv              (30,000+ entries)
# ├── split_train_fold0.txt
# ├── split_val_fold0.txt
# └── ... (10 split files total)
```

---

## 4. Training

### Step 4.1: Choose Configuration

**Option A: Optimized Config (RECOMMENDED)**
```bash
# Best settings for your RTX 3090
# - ReduceLROnPlateau scheduler
# - Early stopping (25 epochs patience)
# - Good regularization
# - Expected IoU: 0.60-0.70

# Edit config if needed:
notepad configs\optimized.yaml

# Key settings:
# - batch_size: 16 (perfect for 24GB GPU)
# - lr: 2.0e-4
# - scheduler: "plateau"
# - early_stop_patience: 25
```

**Option B: Default Config (alternative)**
```bash
# Longer training, cosine scheduler
notepad configs\default.yaml

# Key settings:
# - epochs: 250
# - scheduler: "cosine"
# - No early stopping
```

**Option C: Quick Test Config**
```bash
# Only 3 epochs for testing
notepad configs\quick_test.yaml

# Use ONLY to verify code works!
```

### Step 4.2: Update Config for Your Data

**Edit the config file:**
```bash
notepad configs\optimized.yaml
```

**Change this line:**
```yaml
data:
  proc_root: "data/processed_full"  # Match your preprocessing output!
  # Use "data/processed_multimodal" if you ran multimodal preprocessing
```

### Step 4.3: Start Training

**Single Fold Training:**
```bash
# Train on fold 0
python scripts/train.py --cfg configs/optimized.yaml --fold 0

# Expected output:
# ======================================================================
# BrainTumNet Training Log
# ======================================================================
# Experiment: braintumnet_optimized
# Fold: 0
# Start Time: 2025-10-06 14:30:00
# ======================================================================
#
# [14:30:00] [INFO] Training on device: cuda
# [14:30:00] [INFO] Train batches: 2562, Val batches: 640
# [14:30:00] [INFO] Model parameters: 14.3M total, 14.3M trainable
# [14:30:00] [INFO] TensorBoard logging to: runs/braintumnet_optimized_fold0
# [14:30:00] [INFO] Starting training for 200 epochs...
#
# Epoch 1/200 [Train]: 100%|████| 2562/2562 [12:34<00:00]
#   loss: 1.4970  lr: 2.0e-4
# Epoch 1/200 [Val]:   100%|████| 640/640 [2:15<00:00]
#   iou: 0.1523  dice: 0.2634
# → New best IoU: 0.1523, checkpoint saved!
#
# Epoch 2/200 [Train]: 100%|████| 2562/2562 [12:31<00:00]
#   loss: 1.2108  lr: 2.0e-4
# Epoch 2/200 [Val]:   100%|████| 640/640 [2:14<00:00]
#   iou: 0.2347  dice: 0.3808
# → New best IoU: 0.2347, checkpoint saved!
# ...

# Training time per epoch: ~15 minutes (12 min train + 3 min val)
# Total time:
#   - With early stopping: 10-20 hours (80-120 epochs)
#   - Full 200 epochs: ~50 hours
```

**All Folds Training (5-fold cross-validation):**
```bash
# Train all 5 folds sequentially
for /L %i in (0,1,4) do python scripts/train.py --cfg configs/optimized.yaml --fold %i

# Total time: 50-100 hours
# Best for final publication results
```

### Step 4.4: Training Outputs

After training starts, you'll see these files:

```
braintumnet/
├── checkpoints/
│   └── braintumnet_best_fold0.pth    # Best model weights
│
├── logs/
│   ├── braintumnet_optimized_fold0_20251006.log    # Detailed log
│   ├── metrics_braintumnet_optimized_fold0.csv     # Metrics CSV
│   └── best_metrics.json                           # Best values
│
└── runs/
    └── braintumnet_optimized_fold0/
        └── events.out.tfevents...    # TensorBoard data
```

---

## 5. Monitoring

### Step 5.1: Real-time Console Output

You'll see live progress in your terminal:
```
Epoch 42/200 [Train]: 100%|█████████████████| 2562/2562
  loss: 0.6330  lr: 1.5e-4

Epoch 42/200 [Val]:   100%|█████████████████| 640/640
  iou: 0.5823  dice: 0.7364

→ New best IoU: 0.5823, checkpoint saved!
```

### Step 5.2: TensorBoard (Real-time Graphs)

**Open new terminal, activate environment:**
```bash
cd E:\thong\code\brain_segmen\braintumnet
.\.venv\Scripts\activate

# Start TensorBoard
tensorboard --logdir=runs

# Expected output:
# Serving TensorBoard on localhost; to expose to the network, use --bind_all
# TensorBoard 2.13.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

**Open browser:**
```
http://localhost:6006
```

**You'll see:**
- **Scalars Tab:**
  - train/loss_total (should decrease)
  - train/loss_seg, train/loss_cls
  - train/lr (learning rate schedule)
  - val/iou, val/dice, val/cls_acc (should increase)
  - epoch/train_loss

- **Images Tab:**
  - Sample predictions every 10 epochs
  - Input MRI | Ground truth | Prediction

- **Graphs Tab:**
  - Model architecture visualization

### Step 5.3: Visualization Script (Optional)

**Live plot from metrics CSV:**
```bash
python scripts/visualize_training.py \
  --run_dir runs/braintumnet_optimized_fold0 \
  --output training_progress.png \
  --auto_refresh

# Opens matplotlib window with:
# - Loss curve
# - Learning rate
# - IoU/Dice curves
# - Accuracy curve
# Auto-refreshes every 5 seconds
```

### Step 5.4: Check Logs

```bash
# View detailed log
type logs\braintumnet_optimized_fold0_20251006.log | more

# View metrics CSV
type logs\metrics_braintumnet_optimized_fold0.csv
```

---

## 6. Evaluation

### Step 6.1: Evaluate Best Model

```bash
# Evaluate on validation set (fold 0)
python scripts/evaluate.py \
  --cfg configs/optimized.yaml \
  --fold 0 \
  --ckpt checkpoints/braintumnet_best_fold0.pth

# Expected output:
# ======================================================================
# Evaluation Results
# ======================================================================
# Checkpoint: checkpoints/braintumnet_best_fold0.pth
# Fold: 0
# Dataset: Validation (10,247 slices)
# ======================================================================
#
# Segmentation Metrics:
#   IoU:  0.6523
#   Dice: 0.7891
#
# Classification Metrics:
#   Accuracy: 98.45%
#   Precision: 98.67%
#   Recall: 98.23%
#   F1-Score: 98.45%
#
# Per-class Accuracy:
#   LGG (class 0): 97.83%
#   HGG (class 1): 98.71%
# ======================================================================
```

### Step 6.2: Evaluate All Folds

```bash
# If you trained all 5 folds
for /L %i in (0,1,4) do (
  python scripts/evaluate.py --cfg configs/optimized.yaml --fold %i --ckpt checkpoints/braintumnet_best_fold%i.pth
)

# Average the results for final publication numbers
```

---

## 7. Inference & Visualization

### Step 7.1: Run Inference on New Data

```bash
# Predict on a single slice
python scripts/predict.py \
  --ckpt checkpoints/braintumnet_best_fold0.pth \
  --image data/processed_full/images/vol185_slice75.png \
  --output prediction.png

# Creates visualization:
# [Input MRI | Prediction | Overlay]
```

### Step 7.2: Batch Prediction

```bash
# Predict on entire validation set
python scripts/predict.py \
  --ckpt checkpoints/braintumnet_best_fold0.pth \
  --image_dir data/processed_full/images \
  --split data/processed_full/split_val_fold0.txt \
  --output_dir predictions/fold0 \
  --save_masks

# Creates:
# predictions/fold0/
#   ├── vol185_slice75_pred.png
#   ├── vol185_slice76_pred.png
#   └── ...
```

### Step 7.3: Visualize Batch

```bash
# Visualize random samples from validation set
python scripts/visualize_batch.py \
  --cfg configs/optimized.yaml \
  --fold 0 \
  --ckpt checkpoints/braintumnet_best_fold0.pth \
  --num_samples 16 \
  --output validation_samples.png

# Creates 4×4 grid showing:
# Row 1: Input images
# Row 2: Ground truth masks
# Row 3: Predicted masks
# Row 4: Overlays
```

---

## 8. Results Analysis

### Step 8.1: Compare Multiple Runs

```bash
# Compare different experiments
python scripts/compare_runs.py \
  --run_dirs runs/braintumnet_default_fold0 runs/braintumnet_optimized_fold0 \
  --run_names "Default" "Optimized" \
  --output comparison.png

# Creates comparison plot with:
# - Loss curves (both runs)
# - IoU curves (both runs)
# - Summary table
```

### Step 8.2: Check Best Metrics

```bash
# View best metrics JSON
type logs\best_metrics.json

# Example output:
# {
#   "train_loss": [0.0853, 187],
#   "val_iou": [0.6889, 142],
#   "val_dice": [0.8154, 142],
#   "val_acc": [0.9845, 139],
#   "iou": [0.6889, 142]
# }
# Format: [value, epoch_achieved]
```

### Step 8.3: Generate Report

Create summary document:
```markdown
# BrainTumNet Training Results

## Configuration
- Dataset: BraTS2020 (369 cases, 51,234 slices)
- Model: BrainTumNet (14.3M params)
- Training: Optimized config, RTX 3090, 24GB VRAM

## Results (Fold 0)
| Metric | Value | Epoch |
|--------|-------|-------|
| Best IoU | 0.6889 | 142 |
| Best Dice | 0.8154 | 142 |
| Best Accuracy | 98.45% | 139 |
| Final Loss | 0.0853 | 187 |

## Training Time
- Total epochs: 150 (early stopped at 167)
- Time per epoch: ~15 minutes
- Total time: ~38 hours

## Publication Quality
✅ Workshop: Dice > 0.70 (achieved 0.8154)
✅ Conference: Dice > 0.80 (achieved 0.8154)
⚠️ Top-tier: Dice > 0.85 (close, need multimodal)
```

---

## 📊 Expected Timeline

### Full Workflow (Single Fold)

| Step | Time | Disk Space |
|------|------|------------|
| 1. Setup environment | 10 min | 2 GB |
| 2. Download BraTS2020 | 30 min | 8 GB |
| 3. Preprocess data | 45 min | 5 GB |
| 4. Training (with early stop) | 10-20 hrs | 500 MB |
| 5. Evaluation | 5 min | - |
| 6. Visualization | 10 min | 100 MB |
| **Total** | **12-22 hours** | **~15 GB** |

### Full Workflow (All 5 Folds)

| Step | Time |
|------|------|
| Preprocessing | 45 min (once) |
| Training (5 folds) | 50-100 hrs |
| Evaluation | 25 min |
| **Total** | **~55-105 hours** |

---

## 🎯 Expected Performance

### With Current Data (13 cases):
```
IoU:  0.44-0.46 ❌ Too low
Dice: 0.61-0.63 ❌ Too low
Reason: Not enough data (overfitting)
```

### With Full Dataset (369 cases):

**Single Modality (T1CE):**
```
IoU:  0.60-0.70 ✅
Dice: 0.75-0.82 ✅
Acc:  95-98% ✅
Publishable: Workshop/Conference
```

**Multimodal (FLAIR + T1 + T1CE + T2):**
```
IoU:  0.65-0.75 ✅✅
Dice: 0.79-0.86 ✅✅
Acc:  96-99% ✅✅
Publishable: Top-tier Conference
```

---

## 🔧 Troubleshooting

### Out of GPU Memory
```bash
# Reduce batch size in config
# configs/optimized.yaml:
train:
  batch_size: 8  # Reduced from 16
```

### Training too slow
```bash
# Check GPU usage
nvidia-smi

# Reduce workers if CPU bottleneck
# configs/optimized.yaml:
train:
  workers: 2  # Reduced from 4
```

### TensorBoard not showing graphs
```bash
# Clear cache
rd /s /q runs\.cache

# Restart TensorBoard
tensorboard --logdir=runs --reload_interval=5
```

### Early stopping too aggressive
```bash
# Increase patience
# configs/optimized.yaml:
train:
  early_stop_patience: 40  # Increased from 25
```

---

## 📝 Next Steps After Training

1. **Analyze results:**
   - Compare with baselines
   - Check failure cases
   - Visualize predictions

2. **Improve performance:**
   - Try multimodal training
   - Ensemble multiple folds
   - Fine-tune hyperparameters

3. **Prepare for publication:**
   - Train on all 5 folds
   - Calculate mean ± std metrics
   - Create publication-quality figures

4. **Deploy model:**
   - Export to ONNX
   - Create inference API
   - Build web interface

---

## 🚀 Quick Commands Summary

```bash
# COMPLETE WORKFLOW
# =================

# 1. Setup
cd E:\thong\code\brain_segmen\braintumnet
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Preprocess (get BraTS2020 first!)
python scripts/prepare_brats2020_h5.py \
  --h5_root data/raw \
  --meta_csv data/raw/meta_data.csv \
  --out data/processed_full

# 3. Train
python scripts/train.py --cfg configs/optimized.yaml --fold 0

# 4. Monitor (separate terminal)
tensorboard --logdir=runs

# 5. Evaluate
python scripts/evaluate.py \
  --cfg configs/optimized.yaml \
  --fold 0 \
  --ckpt checkpoints/braintumnet_best_fold0.pth

# 6. Visualize
python scripts/visualize_batch.py \
  --cfg configs/optimized.yaml \
  --fold 0 \
  --ckpt checkpoints/braintumnet_best_fold0.pth \
  --output results.png
```

---

**That's it! You now have the complete workflow from start to finish. Good luck with your brain tumor segmentation project! 🧠✨**
