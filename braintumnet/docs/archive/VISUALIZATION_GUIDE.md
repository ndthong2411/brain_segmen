# BrainTumNet - Visualization Guide

Complete guide to visualizing training progress, results, and predictions.

---

## Table of Contents

1. [Real-time Training Monitoring](#real-time-training-monitoring)
2. [Live Training Plots](#live-training-plots)
3. [TensorBoard Visualization](#tensorboard-visualization)
4. [Comparing Multiple Runs](#comparing-multiple-runs)
5. [Visualizing Predictions](#visualizing-predictions)
6. [Progress Bars During Training](#progress-bars-during-training)
7. [Custom Visualizations](#custom-visualizations)

---

## Real-time Training Monitoring

### Live Plot Updates

Monitor training progress with automatically updating plots:

```bash
cd braintumnet
python scripts/visualize_training.py --logdir runs
```

**Features:**
- 4 synchronized plots (Loss, LR, IoU/Dice, Accuracy)
- Auto-refreshes every 5 seconds (configurable)
- Shows current and best values
- Works while training is running

**Options:**
```bash
# Specify refresh interval (default: 5 seconds)
python scripts/visualize_training.py --logdir runs --refresh 10

# Monitor specific run
python scripts/visualize_training.py --logdir runs --run braintumnet_brats2020_fold0

# Save snapshot instead of live view
python scripts/visualize_training.py --logdir runs --save training_snapshot.png
```

**Example Workflow:**
```bash
# Terminal 1: Start training
python scripts/train.py --cfg configs/default.yaml --fold 0

# Terminal 2: Monitor live
python scripts/visualize_training.py --logdir runs
```

---

## Live Training Plots

The training script now includes **progress bars** and **live metrics** during training:

```bash
python scripts/train.py --cfg configs/quick_test.yaml --fold 0
```

**You'll see:**
```
Epoch 1/3 [Train]: 100%|████████| 400/400 [02:15<00:00, loss: 0.4521, lr: 1.00e-04]
Epoch 1/3 [Val]:   100%|████████| 100/100 [00:30<00:00, iou: 0.3245, dice: 0.4123]
[Fold 0] Epoch 1/3 | Train Loss 0.4521 | Val IoU 0.3245 | Dice 0.4123 | ClsAcc 0.8500
```

**Features:**
- Real-time loss and learning rate display
- Validation metrics updated live
- Estimated time remaining
- Clean, formatted output

---

## TensorBoard Visualization

### Launch TensorBoard

```bash
cd braintumnet
tensorboard --logdir runs/
```

Then open your browser to: **http://localhost:6006**

### What You'll See

**1. Scalars Tab:**
- `train/loss_total` - Total training loss
- `train/loss_seg` - Segmentation loss component
- `train/loss_cls` - Classification loss component
- `train/lr` - Learning rate schedule
- `val/iou` - Validation IoU score
- `val/dice` - Validation Dice score
- `val/cls_acc` - Classification accuracy
- `epoch/train_loss` - Epoch-averaged training loss

**2. Images Tab (every 10 epochs):**
- `samples/input` - Input MRI slices
- `samples/ground_truth` - Ground truth masks
- `samples/prediction` - Model predictions

**3. Graphs Tab:**
- Full model architecture visualization

### TensorBoard Tips

```bash
# Custom port
tensorboard --logdir runs/ --port 6007

# Compare multiple runs
tensorboard --logdir runs/ --bind_all

# Load specific run only
tensorboard --logdir runs/braintumnet_brats2020_fold0/
```

---

## Comparing Multiple Runs

Compare different experiments, hyperparameters, or folds side-by-side:

```bash
cd braintumnet
python scripts/compare_runs.py --logdir runs
```

**Output:**
- Side-by-side plots of all metrics
- Summary table with best scores
- Star markers on best epochs

**Compare specific runs:**
```bash
python scripts/compare_runs.py --logdir runs --runs \
  braintumnet_brats2020_fold0 \
  braintumnet_brats2020_fold1 \
  braintumnet_quick_test_fold0
```

**Save comparison:**
```bash
python scripts/compare_runs.py --logdir runs --save comparison_all_folds.png
```

**Example Output:**
```
================================================================================
SUMMARY TABLE
================================================================================
Run Name                                   Best IoU  Best Dice   Best Acc
--------------------------------------------------------------------------------
braintumnet_brats2020_fold0                  0.7245     0.8123     0.9250
braintumnet_brats2020_fold1                  0.7189     0.8067     0.9180
braintumnet_quick_test_fold0                 0.4572     0.6234     0.8900
================================================================================
```

---

## Visualizing Predictions

### Visualize Training Batch

View a batch of samples from your dataset:

```bash
cd braintumnet
python scripts/visualize_batch.py --cfg configs/quick_test.yaml --fold 0 --n 8
```

**Output:** Grid showing 8 samples with input images and masks

**Options:**
```bash
# More samples
python scripts/visualize_batch.py --cfg configs/default.yaml --fold 0 --n 16

# Save to file
python scripts/visualize_batch.py --cfg configs/default.yaml --fold 0 --n 8 --save batch_samples.png
```

### Single Image Prediction

Make and visualize prediction on a single image:

```bash
cd braintumnet
python scripts/predict.py \
  --cfg configs/quick_test.yaml \
  --ckpt checkpoints/braintumnet_best_fold0.pth \
  --img data/processed/images/BraTS20_Training_001_slice_080.png \
  --out prediction.png
```

**Output:** 3-panel visualization:
- Left: Input image
- Center: Model prediction
- Right: Overlay (input + prediction)

**Console output:**
```
Classification: HGG (High-Grade Glioma)
Confidence: 0.9234
Tumor coverage: 12.3%
```

### Batch Prediction Visualization

Process multiple images at once:

```bash
# Find some test images
ls data/processed/images/ | head -5

# Predict on each
for img in data/processed/images/BraTS20_Training_001_slice_*.png; do
  python scripts/predict.py \
    --cfg configs/quick_test.yaml \
    --ckpt checkpoints/braintumnet_best_fold0.pth \
    --img "$img" \
    --out "predictions/$(basename $img)"
done
```

---

## Progress Bars During Training

The enhanced trainer automatically shows progress bars (requires `tqdm`):

### Training Progress
```
Epoch 1/250 [Train]: 100%|████████████| 400/400 [02:15<00:00, 2.95it/s, loss=0.4521, lr=1.00e-04]
```

### Validation Progress
```
Epoch 1/250 [Val]: 100%|████████████| 100/100 [00:30<00:00, 3.33it/s, iou=0.3245, dice=0.4123]
```

### Disable Progress Bars

If you prefer simple output (e.g., for logging to file):

```bash
# Uninstall tqdm (not recommended)
pip uninstall tqdm

# Or redirect to file
python scripts/train.py --cfg configs/default.yaml --fold 0 > train.log 2>&1
```

---

## Custom Visualizations

### Creating Your Own Plots

Load TensorBoard logs programmatically:

```python
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# Load event file
ea = event_accumulator.EventAccumulator('runs/braintumnet_brats2020_fold0/events.out.tfevents.xxx')
ea.Reload()

# Extract metrics
iou_events = ea.Scalars('val/iou')
epochs = [e.step for e in iou_events]
iou_values = [e.value for e in iou_events]

# Custom plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, iou_values, 'b-o', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('IoU Score')
plt.title('Custom Validation IoU Plot')
plt.grid(True)
plt.savefig('custom_iou.png')
```

### Exporting Metrics to CSV

```python
import csv
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('runs/braintumnet_brats2020_fold0/events.out.tfevents.xxx')
ea.Reload()

# Export to CSV
with open('metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'iou', 'dice', 'accuracy'])

    iou_events = ea.Scalars('val/iou')
    dice_events = ea.Scalars('val/dice')
    acc_events = ea.Scalars('val/cls_acc')

    for iou, dice, acc in zip(iou_events, dice_events, acc_events):
        writer.writerow([iou.step, iou.value, dice.value, acc.value])
```

---

## Visualization Best Practices

### 1. **Monitor While Training**
```bash
# Terminal 1
python scripts/train.py --cfg configs/default.yaml --fold 0

# Terminal 2
tensorboard --logdir runs/

# Terminal 3
python scripts/visualize_training.py --logdir runs
```

### 2. **Save Snapshots Periodically**
```bash
# Add to training script or cron job
python scripts/visualize_training.py --logdir runs --save snapshots/epoch_$(date +%Y%m%d_%H%M%S).png
```

### 3. **Compare Experiments**
After trying different hyperparameters:
```bash
python scripts/compare_runs.py --logdir runs --save final_comparison.png
```

### 4. **Archive Results**
```bash
# Create results directory
mkdir -p results/experiment_001

# Copy important visualizations
cp test_visualization.png results/experiment_001/
cp comparison.png results/experiment_001/
cp checkpoints/braintumnet_best_fold0.pth results/experiment_001/

# Export TensorBoard data
tensorboard --logdir runs --export_as_csv results/experiment_001/metrics.csv
```

---

## Troubleshooting

### "No module named tensorboard"
```bash
pip install tensorboard
```

### "No module named tqdm"
```bash
pip install tqdm
```

### TensorBoard not showing data
- Ensure training has started and logged at least one step
- Check that `use_tensorboard: true` in your config
- Verify log directory path: `ls runs/`

### Live plot not updating
- Increase refresh interval: `--refresh 10`
- Check that training is still running
- Ensure event files are being written: `ls runs/*/events.out.tfevents.*`

### Plots look compressed
- Increase figure size in script
- Save to file with higher DPI: modify `dpi=300` in script

---

## Quick Reference

| Task | Command |
|------|---------|
| Live training plot | `python scripts/visualize_training.py --logdir runs` |
| TensorBoard | `tensorboard --logdir runs/` |
| Compare runs | `python scripts/compare_runs.py --logdir runs` |
| Visualize batch | `python scripts/visualize_batch.py --cfg configs/quick_test.yaml --fold 0 --n 8` |
| Single prediction | `python scripts/predict.py --cfg configs/quick_test.yaml --ckpt checkpoints/xxx.pth --img path.png --out pred.png` |
| Save snapshot | `python scripts/visualize_training.py --logdir runs --save snapshot.png` |

---

## Examples Gallery

### Full Training Monitoring Setup

```bash
# Terminal 1: Train model
cd braintumnet
python scripts/train.py --cfg configs/default.yaml --fold 0

# Terminal 2: TensorBoard (browser: http://localhost:6006)
cd braintumnet
tensorboard --logdir runs/

# Terminal 3: Live plot (GUI window)
cd braintumnet
python scripts/visualize_training.py --logdir runs

# Terminal 4: Periodic snapshots (every 5 minutes)
cd braintumnet
while true; do
  python scripts/visualize_training.py --logdir runs --save "snapshots/$(date +%Y%m%d_%H%M%S).png"
  sleep 300
done
```

### Cross-Validation Comparison

```bash
# Train all 5 folds
for fold in {0..4}; do
  python scripts/train.py --cfg configs/default.yaml --fold $fold
done

# Compare all folds
python scripts/compare_runs.py --logdir runs --runs \
  braintumnet_brats2020_fold0 \
  braintumnet_brats2020_fold1 \
  braintumnet_brats2020_fold2 \
  braintumnet_brats2020_fold3 \
  braintumnet_brats2020_fold4 \
  --save cross_validation_comparison.png
```

---

**Happy Visualizing! 📊📈**
