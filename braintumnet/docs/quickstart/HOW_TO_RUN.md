# How to Run BrainTumNet - Complete Guide

**Everything you need to know to run BrainTumNet with full visualization.**

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Verify Setup
```bash
cd braintumnet
python verify_setup.py
```

**Expected output:** `ALL CHECKS PASSED ✅`

### Step 2: Run Quick Test with Visualization

**Terminal 1: Start Training**
```bash
python scripts/train.py --cfg configs/quick_test.yaml --fold 0
```

**Terminal 2: Watch Live Plots**
```bash
python scripts/visualize_training.py --logdir runs
```

**Done!** In ~2 minutes you'll see:
- Progress bars with real-time metrics
- Auto-updating plots (4 graphs)
- Model saved to `checkpoints/`

---

## 📊 What You'll See

### Console (Terminal 1)
```
Training on device: cuda
Train batches: 400, Val batches: 100
TensorBoard logging to: runs/braintumnet_quick_test_fold0

Epoch 1/3 [Train]: 100%|████████████| 400/400 [02:15, loss=0.6234, lr=1.00e-04]
Epoch 1/3 [Val]:   100%|████████████| 100/100 [00:30, iou=0.2134, dice=0.3456]
[Fold 0] Epoch 1/3 | Train Loss 0.6234 | Val IoU 0.2134 | Dice 0.3456 | ClsAcc 0.7500
  -> New best IoU: 0.2134, checkpoint saved

Epoch 2/3 [Train]: 100%|████████████| 400/400 [02:12, loss=0.5891, lr=9.99e-05]
Epoch 2/3 [Val]:   100%|████████████| 100/100 [00:29, iou=0.2567, dice=0.3892]
[Fold 0] Epoch 2/3 | Train Loss 0.5891 | Val IoU 0.2567 | Dice 0.3892 | ClsAcc 0.7800
  -> New best IoU: 0.2567, checkpoint saved

Epoch 3/3 [Train]: 100%|████████████| 400/400 [02:11, loss=0.5523, lr=9.96e-05]
Epoch 3/3 [Val]:   100%|████████████| 100/100 [00:28, iou=0.4572, dice=0.6234]
[Fold 0] Epoch 3/3 | Train Loss 0.5523 | Val IoU 0.4572 | Dice 0.6234 | ClsAcc 0.8900
  -> New best IoU: 0.4572, checkpoint saved
```

### Live Plots (Terminal 2)
```
4-panel window with auto-updating graphs:
┌─────────────────────┬─────────────────────┐
│ Training Loss       │ Learning Rate       │
│ (decreasing curve)  │ (cosine decay)      │
├─────────────────────┼─────────────────────┤
│ IoU & Dice          │ Classification Acc  │
│ (increasing curves) │ (improving)         │
└─────────────────────┴─────────────────────┘

Updates every 5 seconds automatically!
```

---

## 🎯 All Running Options

### Option 1: Training Only (Minimal)
```bash
python scripts/train.py --cfg configs/quick_test.yaml --fold 0
```
- Progress bars in console
- TensorBoard logs saved
- Best model saved

### Option 2: Training + Live Plots
```bash
# Terminal 1
python scripts/train.py --cfg configs/quick_test.yaml --fold 0

# Terminal 2
python scripts/visualize_training.py --logdir runs
```
- Everything from Option 1
- Plus: Auto-updating 4-panel plots

### Option 3: Training + TensorBoard
```bash
# Terminal 1
python scripts/train.py --cfg configs/quick_test.yaml --fold 0

# Terminal 2
tensorboard --logdir runs/
# Open http://localhost:6006 in browser
```
- Everything from Option 1
- Plus: Interactive web dashboard

### Option 4: Full Monitoring (Recommended)
```bash
# Terminal 1: Train
python scripts/train.py --cfg configs/quick_test.yaml --fold 0

# Terminal 2: Live Plots
python scripts/visualize_training.py --logdir runs

# Terminal 3: TensorBoard
tensorboard --logdir runs/
# Open http://localhost:6006 in browser
```
- All features enabled
- Maximum visibility
- Best for debugging

---

## 🔧 Configuration Options

### Quick Test (3 epochs, ~2 minutes)
```bash
python scripts/train.py --cfg configs/quick_test.yaml --fold 0
```
- Good for: Testing setup, quick experiments
- Performance: IoU ~0.35-0.45
- Time: ~2 minutes

### Full Training (250 epochs, ~6-8 hours)
```bash
python scripts/train.py --cfg configs/default.yaml --fold 0
```
- Good for: Production model, publication results
- Performance: IoU ~0.65-0.75, Dice ~0.75-0.85
- Time: ~6-8 hours

### Custom Configuration
```bash
# Create your config
cp configs/default.yaml configs/my_config.yaml

# Edit as needed
nano configs/my_config.yaml

# Train with it
python scripts/train.py --cfg configs/my_config.yaml --fold 0
```

---

## 📈 After Training

### Step 1: Evaluate Model
```bash
python scripts/evaluate.py \
  --cfg configs/quick_test.yaml \
  --ckpt checkpoints/braintumnet_best_fold0.pth \
  --fold 0
```

**Output:**
```
Evaluation Results:
  Segmentation IoU:  0.4572
  Segmentation Dice: 0.6234
  Classification Acc: 0.8900
  Classification F1:  0.8756
  Classification AUC: 0.9123
```

### Step 2: Make Predictions
```bash
python scripts/predict.py \
  --cfg configs/quick_test.yaml \
  --ckpt checkpoints/braintumnet_best_fold0.pth \
  --img data/processed/images/BraTS20_Training_001_slice_080.png \
  --out prediction.png
```

**Output:**
- `prediction.png` - 3-panel visualization
- Console: Classification result

### Step 3: Visualize Training History
```bash
# Save final plot
python scripts/visualize_training.py --logdir runs --save final_results.png

# Or view in TensorBoard
tensorboard --logdir runs/
```

---

## 🎓 Cross-Validation (All 5 Folds)

### Train All Folds
```bash
for fold in {0..4}; do
  echo "=== Training Fold $fold ==="
  python scripts/train.py --cfg configs/default.yaml --fold $fold
done
```

### Compare All Folds
```bash
python scripts/compare_runs.py --logdir runs --save all_folds.png
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
================================================================================
```

---

## 💡 Pro Tips

### 1. Monitor While Training
```bash
# Start both at once
python scripts/visualize_training.py --logdir runs &
python scripts/train.py --cfg configs/default.yaml --fold 0
```

### 2. Save Snapshots Periodically
```bash
# In a separate terminal, run every 30 minutes
while true; do
  python scripts/visualize_training.py --logdir runs --save "snapshots/$(date +%H%M).png"
  sleep 1800
done
```

### 3. Check GPU Usage
```bash
# If using CUDA
nvidia-smi
watch -n 1 nvidia-smi  # Update every second
```

### 4. Save Training Logs
```bash
python scripts/train.py --cfg configs/default.yaml --fold 0 2>&1 | tee training.log
```

---

## 🔍 Troubleshooting

### "CUDA out of memory"
**Solution:** Reduce batch size in config
```yaml
train:
  batch_size: 8  # Or 4
```

### "No module named braintumnet"
**Solution:** Ensure you're in braintumnet/ directory
```bash
cd braintumnet
python verify_setup.py
```

### Training is slow (CPU)
**Solution:** Enable AMP in config (if GPU available)
```yaml
train:
  amp: true
```

### Plots not updating
**Solution:** Increase refresh interval
```bash
python scripts/visualize_training.py --logdir runs --refresh 10
```

### Poor performance after quick test
**Solution:** This is normal! Quick test is only 3 epochs.
```bash
# Train full model for better results
python scripts/train.py --cfg configs/default.yaml --fold 0
```

---

## 📁 Output Files

After training, you'll have:

```
braintumnet/
├── checkpoints/
│   └── braintumnet_best_fold0.pth         # Best model (13.7 MB)
│
├── runs/
│   └── braintumnet_quick_test_fold0/
│       └── events.out.tfevents.*          # TensorBoard logs
│
└── (optional visualizations)
    ├── test_visualization.png             # Training curves
    ├── comparison.png                     # Multi-run comparison
    └── prediction.png                     # Sample prediction
```

---

## 🎯 Common Workflows

### Workflow 1: Quick Experiment
```bash
# 1. Modify config
cp configs/default.yaml configs/experiment_001.yaml
nano configs/experiment_001.yaml

# 2. Train
python scripts/train.py --cfg configs/experiment_001.yaml --fold 0

# 3. Visualize
python scripts/visualize_training.py --logdir runs --save exp001.png

# 4. Evaluate
python scripts/evaluate.py \
  --cfg configs/experiment_001.yaml \
  --ckpt checkpoints/braintumnet_best_fold0.pth \
  --fold 0
```

### Workflow 2: Full Production Run
```bash
# 1. Train all folds with monitoring
for fold in {0..4}; do
  python scripts/train.py --cfg configs/default.yaml --fold $fold &
  tensorboard --logdir runs/ --port $((6006 + fold)) &
done

# 2. Wait for completion (6-8 hours per fold)

# 3. Evaluate all folds
for fold in {0..4}; do
  python scripts/evaluate.py \
    --cfg configs/default.yaml \
    --ckpt checkpoints/braintumnet_best_fold${fold}.pth \
    --fold $fold
done

# 4. Compare results
python scripts/compare_runs.py --logdir runs --save final_comparison.png
```

### Workflow 3: Hyperparameter Tuning
```bash
# Create configs
for lr in 1e-3 1e-4 1e-5; do
  sed "s/lr: 1.0e-4/lr: $lr/" configs/default.yaml > configs/lr_${lr}.yaml
done

# Train all
for config in configs/lr_*.yaml; do
  python scripts/train.py --cfg $config --fold 0
done

# Compare
python scripts/compare_runs.py --logdir runs --save lr_comparison.png
```

---

## 📚 Documentation Quick Links

- **First time?** → Read `QUICKSTART.md`
- **Want details?** → Read `TRAINING_GUIDE.md`
- **Need commands?** → Read `VISUALIZATION_QUICKREF.md`
- **Want examples?** → Read `docs/VISUALIZATION_GUIDE.md`
- **See all docs?** → Read `DOCS_INDEX.md`

---

## ✅ Checklist

Before running:
- [ ] Installed dependencies: `pip install -r requirements.txt`
- [ ] Verified setup: `python verify_setup.py`
- [ ] Preprocessed data: Check `data/processed/images/`
- [ ] Configured experiment: Edit `configs/` if needed

During training:
- [ ] Monitoring with live plots or TensorBoard
- [ ] Checking progress bars for metrics
- [ ] Saving periodic snapshots (optional)

After training:
- [ ] Evaluated model: `scripts/evaluate.py`
- [ ] Tested predictions: `scripts/predict.py`
- [ ] Visualized results: `scripts/visualize_training.py`
- [ ] Compared runs (if multiple): `scripts/compare_runs.py`

---

## 🎉 Summary

**To run BrainTumNet:**

1. **Verify:** `python verify_setup.py`
2. **Train:** `python scripts/train.py --cfg configs/quick_test.yaml --fold 0`
3. **Monitor:** `python scripts/visualize_training.py --logdir runs`
4. **Evaluate:** `python scripts/evaluate.py --cfg configs/quick_test.yaml --ckpt checkpoints/xxx.pth --fold 0`

**That's it!** 🚀

---

**Need help?** See troubleshooting above or check `DOCS_INDEX.md` for guides.
