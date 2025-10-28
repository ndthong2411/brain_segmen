# Training Visualization - Quick Reference

## 🎯 Quick Start

### While Training

**Terminal 1: Train**
```bash
python scripts/train.py --cfg configs/default.yaml --fold 0
```

**Terminal 2: Monitor (choose one)**
```bash
# Option A: TensorBoard (browser)
tensorboard --logdir runs/

# Option B: Live plots (GUI window)
python scripts/visualize_training.py --logdir runs

# Option C: Both!
tensorboard --logdir runs/ &
python scripts/visualize_training.py --logdir runs
```

---

## 📊 Visualization Options

| What | Command | Output |
|------|---------|--------|
| **Live plots (GUI)** | `python scripts/visualize_training.py --logdir runs` | Auto-updating window |
| **TensorBoard** | `tensorboard --logdir runs/` | Browser: localhost:6006 |
| **Save snapshot** | `python scripts/visualize_training.py --logdir runs --save out.png` | PNG file |
| **Compare runs** | `python scripts/compare_runs.py --logdir runs` | Side-by-side comparison |
| **Progress bars** | Built-in during training | Console output |

---

## 🔍 What You'll See

### Live Plots (4 panels)
```
┌─────────────────┬─────────────────┐
│ Training Loss   │ Learning Rate   │
│ (Total/Seg/Cls) │ (Schedule)      │
├─────────────────┼─────────────────┤
│ Val IoU & Dice  │ Classification  │
│ (Segmentation)  │ Accuracy        │
└─────────────────┴─────────────────┘
```

### Progress Bars
```
Epoch 1/250 [Train]: 100%|████| 400/400 [02:15, loss: 0.4521, lr: 1.00e-04]
Epoch 1/250 [Val]:   100%|████| 100/100 [00:30, iou: 0.3245, dice: 0.4123]
```

### TensorBoard Tabs
- **Scalars**: Loss curves, metrics, LR schedule
- **Images**: Sample predictions (every 10 epochs)
- **Graphs**: Model architecture

---

## 📈 Common Tasks

### Monitor Current Training
```bash
# Auto-refreshing plots (updates every 5s)
python scripts/visualize_training.py --logdir runs
```

### Compare Experiments
```bash
# After training multiple configurations
python scripts/compare_runs.py --logdir runs --save comparison.png
```

### Save Progress Snapshot
```bash
# Take snapshot while training
python scripts/visualize_training.py --logdir runs --save snapshot_$(date +%H%M).png
```

### Monitor Specific Run
```bash
# If you have multiple runs
python scripts/visualize_training.py --logdir runs --run braintumnet_brats2020_fold0
```

### Change Refresh Rate
```bash
# Update every 10 seconds instead of 5
python scripts/visualize_training.py --logdir runs --refresh 10
```

---

## 🎨 Visualization Features

### Built-in (Automatic)
✅ Progress bars during training
✅ Real-time loss/metrics in console
✅ TensorBoard logging every 10 steps
✅ Sample predictions logged every 10 epochs

### Interactive
✅ Live updating plots (GUI)
✅ TensorBoard web interface
✅ Multi-run comparison
✅ Snapshot export

---

## 💡 Pro Tips

1. **Start monitoring BEFORE training completes**
   ```bash
   python scripts/visualize_training.py --logdir runs &
   python scripts/train.py --cfg configs/default.yaml --fold 0
   ```

2. **Save snapshots periodically**
   ```bash
   # Every 5 minutes
   while true; do
     python scripts/visualize_training.py --logdir runs --save "snap_$(date +%H%M).png"
     sleep 300
   done
   ```

3. **Compare all folds after cross-validation**
   ```bash
   python scripts/compare_runs.py --logdir runs --save all_folds.png
   ```

4. **Use TensorBoard for detailed analysis**
   ```bash
   tensorboard --logdir runs/ --port 6006
   # Then explore Scalars, Images, and Graphs tabs
   ```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| No data showing | Wait for training to log (first 10 steps) |
| Plot not updating | Increase refresh: `--refresh 10` |
| "Module not found" | `pip install tensorboard matplotlib tqdm` |
| TensorBoard empty | Check: `ls runs/*/events.out.tfevents.*` |

---

## 📚 Full Documentation

For complete guide with examples and customization:
```bash
cat docs/VISUALIZATION_GUIDE.md
```

---

## 🚀 Example Workflow

**Complete training monitoring setup:**

```bash
cd braintumnet

# Terminal 1: Train
python scripts/train.py --cfg configs/default.yaml --fold 0

# Terminal 2: TensorBoard (http://localhost:6006)
tensorboard --logdir runs/

# Terminal 3: Live plots
python scripts/visualize_training.py --logdir runs

# After training completes:
python scripts/compare_runs.py --logdir runs --save final_results.png
```

**Expected Progress:**
- Epoch 1-50: IoU ~0.3-0.5 (learning patterns)
- Epoch 51-150: IoU ~0.5-0.7 (refining)
- Epoch 151-250: IoU ~0.7-0.75 (converging)

---

**Quick help:** `python scripts/visualize_training.py --help`
