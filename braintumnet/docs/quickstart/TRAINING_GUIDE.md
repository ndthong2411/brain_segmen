# Complete Training Guide

Step-by-step guide to train BrainTumNet with full visualization.

---

## Prerequisites

✅ Setup verified (run `python verify_setup.py`)
✅ Data preprocessed (2000+ slices in `data/processed/`)
✅ Dependencies installed (`pip install -r requirements.txt`)

---

## Training Workflow

### Step 1: Choose Configuration

**Quick Test (3 epochs, ~2 minutes)**
```bash
CONFIG=configs/quick_test.yaml
```

**Full Training (250 epochs, ~6-8 hours)**
```bash
CONFIG=configs/default.yaml
```

### Step 2: Start Training with Visualization

**Option A: Training Only**
```bash
python scripts/train.py --cfg $CONFIG --fold 0
```

You'll see:
- ✅ Progress bars for training and validation
- ✅ Real-time loss and metrics
- ✅ Time estimates
- ✅ TensorBoard logs saved to `runs/`

**Option B: Training + Live Monitoring (Recommended)**

Open 2 terminals:

**Terminal 1: Train**
```bash
cd braintumnet
python scripts/train.py --cfg configs/default.yaml --fold 0
```

**Terminal 2: Monitor**
```bash
cd braintumnet
python scripts/visualize_training.py --logdir runs
```

**Option C: Full Monitoring Suite (3 terminals)**

**Terminal 1: Train**
```bash
cd braintumnet
python scripts/train.py --cfg configs/default.yaml --fold 0
```

**Terminal 2: TensorBoard**
```bash
cd braintumnet
tensorboard --logdir runs/
# Open http://localhost:6006 in browser
```

**Terminal 3: Live Plots**
```bash
cd braintumnet
python scripts/visualize_training.py --logdir runs
```

---

## What You'll See

### Console Output

```
Training on device: cuda
Train batches: 400, Val batches: 100
TensorBoard logging to: runs/braintumnet_brats2020_fold0

Epoch 1/250 [Train]: 100%|████████████| 400/400 [02:15<00:00, loss=0.6234, lr=1.00e-04]
Epoch 1/250 [Val]:   100%|████████████| 100/100 [00:30<00:00, iou=0.2134, dice=0.3456]
[Fold 0] Epoch 1/250 | Train Loss 0.6234 | Val IoU 0.2134 | Dice 0.3456 | ClsAcc 0.7500
  -> New best IoU: 0.2134, checkpoint saved

Epoch 2/250 [Train]: 100%|████████████| 400/400 [02:12<00:00, loss=0.5891, lr=9.99e-05]
Epoch 2/250 [Val]:   100%|████████████| 100/100 [00:29<00:00, iou=0.2567, dice=0.3892]
[Fold 0] Epoch 2/250 | Train Loss 0.5891 | Val IoU 0.2567 | Dice 0.3892 | ClsAcc 0.7800
  -> New best IoU: 0.2567, checkpoint saved
```

### Live Plots (visualize_training.py)

4 auto-updating graphs:
1. **Training Loss** - Total, Segmentation, Classification
2. **Learning Rate** - Cosine schedule decay
3. **Segmentation Metrics** - IoU and Dice scores
4. **Classification Accuracy** - Tumor grade prediction

### TensorBoard (localhost:6006)

**Scalars Tab:**
- Training curves (loss, LR)
- Validation metrics (IoU, Dice, Accuracy)
- Smoothed and raw data

**Images Tab (every 10 epochs):**
- Input MRI slices
- Ground truth masks
- Model predictions

**Graphs Tab:**
- Full model architecture
- Layer connections

---

## Training Progress Expectations

### Quick Test (3 epochs)
```
Epoch 1/3: IoU ~0.20-0.30 (initialization)
Epoch 2/3: IoU ~0.30-0.40 (rapid learning)
Epoch 3/3: IoU ~0.35-0.45 (early convergence)
Classification: 85-100% (small dataset, may overfit)
```

### Full Training (250 epochs)

**Phase 1: Epochs 1-50 (Warm-up)**
- IoU: 0.2 → 0.5
- Dice: 0.3 → 0.6
- Learning basic tumor patterns
- High variance in metrics

**Phase 2: Epochs 51-150 (Learning)**
- IoU: 0.5 → 0.7
- Dice: 0.6 → 0.8
- Refining segmentation boundaries
- Metrics stabilizing

**Phase 3: Epochs 151-250 (Convergence)**
- IoU: 0.7 → 0.75
- Dice: 0.8 → 0.85
- Fine-tuning
- Metrics plateau

**Expected Final Performance:**
- IoU: 0.65-0.75
- Dice: 0.75-0.85
- Classification Accuracy: 90-95%

---

## During Training

### Monitor Progress

**Check Current Metrics (console):**
- Look at latest epoch summary
- Best IoU is tracked automatically
- Checkpoints saved when IoU improves

**View Detailed Plots:**
```bash
# Live plots
python scripts/visualize_training.py --logdir runs

# Or TensorBoard
tensorboard --logdir runs/
```

### Save Snapshots

```bash
# While training is running (different terminal)
python scripts/visualize_training.py --logdir runs --save snapshot_epoch_100.png
```

### Check Files

```bash
# View latest checkpoint
ls -lh checkpoints/

# View TensorBoard logs
ls runs/*/events.out.tfevents.*

# Monitor GPU usage (if CUDA)
nvidia-smi
```

---

## After Training

### Step 1: Evaluate Model

```bash
python scripts/evaluate.py \
  --cfg configs/default.yaml \
  --ckpt checkpoints/braintumnet_best_fold0.pth \
  --fold 0
```

**Output:**
```
Evaluation Results:
  Segmentation IoU:  0.7234
  Segmentation Dice: 0.8123
  Classification Acc: 0.9150
  Classification F1:  0.9087
  Classification AUC: 0.9523
```

### Step 2: Visualize Final Results

```bash
# Save final training visualization
python scripts/visualize_training.py --logdir runs --save final_training_curves.png

# View in TensorBoard
tensorboard --logdir runs/
```

### Step 3: Test Predictions

```bash
# Pick a test image
python scripts/predict.py \
  --cfg configs/default.yaml \
  --ckpt checkpoints/braintumnet_best_fold0.pth \
  --img data/processed/images/BraTS20_Training_001_slice_080.png \
  --out prediction_sample.png
```

**Output:**
- `prediction_sample.png` - 3-panel visualization
- Console: Classification result and confidence

---

## Cross-Validation (All 5 Folds)

### Train All Folds

```bash
for fold in {0..4}; do
  echo "=== Training Fold $fold ==="
  python scripts/train.py --cfg configs/default.yaml --fold $fold
done
```

**Time:** ~30-40 hours total (6-8 hours per fold)

### Evaluate All Folds

```bash
for fold in {0..4}; do
  echo "=== Evaluating Fold $fold ==="
  python scripts/evaluate.py \
    --cfg configs/default.yaml \
    --ckpt checkpoints/braintumnet_best_fold${fold}.pth \
    --fold $fold
done
```

### Compare Folds

```bash
python scripts/compare_runs.py --logdir runs --save all_folds_comparison.png
```

**Output:**
```
================================================================================
SUMMARY TABLE
================================================================================
Run Name                                   Best IoU  Best Dice   Best Acc
--------------------------------------------------------------------------------
braintumnet_brats2020_fold0                  0.7245     0.8123     0.9250
braintumnet_brats2020_fold1                  0.7189     0.8067     0.9180
braintumnet_brats2020_fold2                  0.7301     0.8156     0.9210
braintumnet_brats2020_fold3                  0.7098     0.8034     0.9190
braintumnet_brats2020_fold4                  0.7267     0.8145     0.9230
--------------------------------------------------------------------------------
Mean ± Std                                   0.7220±    0.8105±    0.9212±
                                             0.0076     0.0049     0.0027
================================================================================
```

---

## Customizing Training

### Adjust Hyperparameters

Edit `configs/default.yaml`:

```yaml
train:
  epochs: 250              # More epochs for better convergence
  batch_size: 16          # Reduce if GPU memory limited
  lr: 1.0e-4             # Learning rate
  scheduler: "cosine"     # LR decay schedule
  amp: true              # Mixed precision (faster, less memory)

model:
  base: 32               # Model size (16=small, 32=default, 64=large)
  dim: 256              # Transformer dimension
  depth: 2              # Transformer layers

augment:
  rotate_deg: 30        # Data augmentation strength
  hflip_p: 0.5
  vflip_p: 0.5
```

### Create Custom Config

```bash
# Copy default config
cp configs/default.yaml configs/my_experiment.yaml

# Edit your config
nano configs/my_experiment.yaml

# Train with your config
python scripts/train.py --cfg configs/my_experiment.yaml --fold 0
```

---

## Troubleshooting

### Training is slow
- Enable AMP: `amp: true` in config
- Reduce batch size if using CPU
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

### Out of memory (CUDA)
- Reduce `batch_size` in config (try 8 or 4)
- Enable `amp: true` for mixed precision
- Use smaller model: `base: 16`

### Poor performance after training
- Train for more epochs (250 instead of 3)
- Check data quality: `python scripts/visualize_batch.py --cfg configs/default.yaml --fold 0 --n 8`
- Verify balanced splits (HGG vs LGG ratio)
- Try different augmentation settings

### Progress bars not showing
- Install tqdm: `pip install tqdm`
- Check Windows terminal compatibility

### Visualization not updating
- Ensure training is running
- Check log directory: `ls runs/`
- Increase refresh interval: `--refresh 10`

---

## Best Practices

### 1. Always Monitor Training
```bash
# Start monitoring before training completes
python scripts/visualize_training.py --logdir runs &
python scripts/train.py --cfg configs/default.yaml --fold 0
```

### 2. Save Periodic Snapshots
```bash
# In a separate terminal
while true; do
  python scripts/visualize_training.py --logdir runs --save "snapshots/$(date +%Y%m%d_%H%M%S).png"
  sleep 1800  # Every 30 minutes
done
```

### 3. Keep Training Logs
```bash
# Save console output
python scripts/train.py --cfg configs/default.yaml --fold 0 2>&1 | tee training.log
```

### 4. Organize Experiments
```bash
# Create experiment directory
mkdir -p experiments/exp_001

# Run experiment
python scripts/train.py --cfg configs/my_config.yaml --fold 0

# Archive results
cp configs/my_config.yaml experiments/exp_001/
cp checkpoints/braintumnet_best_fold0.pth experiments/exp_001/
python scripts/visualize_training.py --logdir runs --save experiments/exp_001/training_curves.png
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Train (quick test) | `python scripts/train.py --cfg configs/quick_test.yaml --fold 0` |
| Train (full) | `python scripts/train.py --cfg configs/default.yaml --fold 0` |
| Live plots | `python scripts/visualize_training.py --logdir runs` |
| TensorBoard | `tensorboard --logdir runs/` |
| Evaluate | `python scripts/evaluate.py --cfg configs/default.yaml --ckpt checkpoints/xxx.pth --fold 0` |
| Predict | `python scripts/predict.py --cfg configs/default.yaml --ckpt checkpoints/xxx.pth --img image.png --out pred.png` |
| Compare runs | `python scripts/compare_runs.py --logdir runs` |

---

## Next Steps

After successful training:
1. ✅ Evaluate on all folds
2. ✅ Compare fold performance
3. ✅ Test on holdout cases
4. ✅ Document results
5. ✅ Consider ensemble predictions
6. ✅ Deploy for inference

**See also:**
- `QUICKSTART.md` - Quick commands
- `docs/VISUALIZATION_GUIDE.md` - Complete visualization docs
- `VISUALIZATION_QUICKREF.md` - Visualization quick reference
- `README.md` - Full project documentation

---

**Happy Training! 🧠🚀**
