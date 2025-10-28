# BrainTumNet - Quick Start Guide

## Directory Structure

**Everything is self-contained in this `braintumnet/` folder.**

```
braintumnet/
├── configs/
│   ├── default.yaml              # Full training (250 epochs)
│   └── quick_test.yaml           # Quick test (3 epochs)
├── scripts/
│   ├── prepare_brats2020_h5.py   # HDF5 → PNG preprocessing
│   ├── train.py                  # Training
│   ├── evaluate.py               # Evaluation
│   ├── predict.py                # Inference
│   └── visualize_batch.py        # Visualization
├── src/braintumnet/              # Core package
├── data/
│   ├── raw/                      # Original data (optional)
│   └── processed/                # Preprocessed PNG slices
├── checkpoints/                  # Trained models
├── runs/                         # TensorBoard logs
├── docs/                         # Documentation
└── verify_setup.py               # Setup verification
```

---

## 1. Installation

### Prerequisites
- Python 3.8+
- CUDA GPU (recommended)

### Install Dependencies
```bash
cd braintumnet
pip install -r requirements.txt
```

### Verify Setup
```bash
python verify_setup.py
```

Expected output: `ALL CHECKS PASSED`

---

## 2. Prepare Data

### Option A: HDF5 Format (Kaggle BraTS2020)
```bash
python scripts/prepare_brats2020_h5.py \
  --h5_root "/path/to/BraTS2020/content/data" \
  --meta_csv "/path/to/meta_data.csv" \
  --out data/processed \
  --modality t1ce
```

### Option B: NIfTI Format (Official BraTS2020)
```bash
python scripts/prepare_brats2020.py \
  --raw "/path/to/BraTS2020/TrainingData" \
  --out data/processed
```

**Output:**
- `data/processed/images/` - PNG slices
- `data/processed/masks/` - Segmentation masks
- `data/processed/labels.csv` - Case labels
- `data/processed/split_*_fold*.txt` - 5-fold splits

---

## 3. Train Model

### Quick Test (3 epochs, ~2 minutes)
```bash
python scripts/train.py \
  --cfg configs/quick_test.yaml \
  --fold 0
```

### Full Training (250 epochs, ~6-8 hours)
```bash
python scripts/train.py \
  --cfg configs/default.yaml \
  --fold 0
```

**Monitor Training:**

Option 1: TensorBoard (web browser)
```bash
tensorboard --logdir runs/
```
Open browser: http://localhost:6006

Option 2: Live Plots (GUI window)
```bash
python scripts/visualize_training.py --logdir runs
```

Option 3: Progress bars (built-in)
- Automatically shown during training with real-time metrics

**Outputs:**
- `checkpoints/braintumnet_best_fold0.pth` - Best model
- `runs/braintumnet_*/` - TensorBoard logs

---

## 4. Evaluate Model

```bash
python scripts/evaluate.py \
  --cfg configs/quick_test.yaml \
  --ckpt checkpoints/braintumnet_best_fold0.pth \
  --fold 0
```

**Metrics:**
- Classification: Accuracy, F1, AUC
- Segmentation: IoU, Dice

---

## 5. Make Predictions

```bash
python scripts/predict.py \
  --cfg configs/quick_test.yaml \
  --ckpt checkpoints/braintumnet_best_fold0.pth \
  --img data/processed/images/sample.png \
  --out prediction.png
```

**Output:**
- `prediction.png` - 3-panel visualization
- Console: Classification + confidence

---

## 6. Visualize Training Progress

### Live Training Visualization
```bash
# Real-time plots (auto-refreshing)
python scripts/visualize_training.py --logdir runs
```

### Compare Multiple Runs
```bash
# Compare different experiments
python scripts/compare_runs.py --logdir runs
```

### Save Visualization Snapshot
```bash
# Save current progress to file
python scripts/visualize_training.py --logdir runs --save progress.png
```

**See:** `docs/VISUALIZATION_GUIDE.md` for complete visualization documentation

---

## 7. Cross-Validation (All 5 Folds)

```bash
# Train all folds
for fold in {0..4}; do
  python scripts/train.py --cfg configs/default.yaml --fold $fold
done

# Evaluate all folds
for fold in {0..4}; do
  python scripts/evaluate.py \
    --cfg configs/default.yaml \
    --ckpt checkpoints/braintumnet_best_fold${fold}.pth \
    --fold $fold
done

# Compare all folds
python scripts/compare_runs.py --logdir runs --save all_folds_comparison.png
```

---

## Common Commands

### View Configuration
```bash
cat configs/default.yaml
```

### Check GPU
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Monitor Training (Live)
```bash
tensorboard --logdir runs/ --port 6006
```

### List Checkpoints
```bash
ls -lh checkpoints/
```

### Visualize Batch
```bash
python scripts/visualize_batch.py \
  --cfg configs/quick_test.yaml \
  --fold 0 \
  --n 8
```

---

## Troubleshooting

### "CUDA out of memory"
**Solution:** Reduce batch size in config
```yaml
train:
  batch_size: 8  # Reduce from 16
```

### "No module named braintumnet"
**Solution:** Ensure you're in the braintumnet/ directory
```bash
cd braintumnet
python verify_setup.py
```

### "FileNotFoundError: labels.csv"
**Solution:** Run data preprocessing first
```bash
python scripts/prepare_brats2020_h5.py ...
```

### Poor performance after training
**Solutions:**
1. Train for more epochs (250 instead of 3)
2. Use full model size (configs/default.yaml)
3. Check data quality and augmentation
4. Ensure balanced train/val splits

---

## Configuration Tips

### Customize Training
Edit `configs/default.yaml`:
```yaml
train:
  epochs: 250           # Number of epochs
  batch_size: 16        # Adjust based on GPU memory
  lr: 1.0e-4           # Learning rate
  scheduler: "cosine"   # LR scheduling
  amp: true            # Mixed precision (faster)
```

### Customize Model
```yaml
model:
  base: 32             # Base channels (16=small, 32=default, 64=large)
  dim: 256            # Transformer dimension
  depth: 2            # Transformer depth
```

### Customize Augmentation
```yaml
augment:
  rotate_deg: 30      # Rotation range
  hflip_p: 0.5        # Horizontal flip probability
  vflip_p: 0.5        # Vertical flip probability
```

---

## Expected Performance

### Quick Test (3 epochs)
- Training time: ~2 minutes
- IoU: 0.35-0.45
- Classification: 85-100%

### Full Training (250 epochs)
- Training time: ~6-8 hours per fold
- IoU: 0.65-0.75
- Dice: 0.75-0.85
- Classification: 90-95%

---

## Next Steps

1. **Read Full Documentation**: `README.md`
2. **Check Test Results**: `docs/TEST_RESULTS.md`
3. **Review Architecture**: See README Model Architecture section
4. **Experiment**: Try different configs, augmentations, model sizes

---

## Support

- **Documentation**: `README.md`, `docs/`
- **Issues**: Check `docs/TEST_RESULTS.md` for known issues
- **Verification**: Run `python verify_setup.py`

---

**Happy Training! 🧠🎯**
