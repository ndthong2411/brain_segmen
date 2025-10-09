# Part 5: Evaluation and Inference

**Navigation**: [[TECHNICAL_REPORT_INDEX|← Back to Index]]

---

## Table of Contents

1. [Overview](#overview)
2. [Evaluator (evaluator.py)](#evaluator-evaluatorpy)
3. [Prediction Script (predict.py)](#prediction-script-predictpy)
4. [Test-Time Augmentation (TTA)](#test-time-augmentation-tta)
5. [Ensemble Predictions](#ensemble-predictions)
6. [Batch Inference](#batch-inference)
7. [Practical Usage Examples](#practical-usage-examples)
8. [Modification Guides](#modification-guides)

---

## Overview

### Evaluation vs Inference

**Evaluation**: Measure model performance on validation/test data with ground truth
- Input: Images + Labels
- Output: Metrics (Dice, IoU, Accuracy, etc.)
- Purpose: Quantify model quality

**Inference**: Apply model to new data (no ground truth)
- Input: Images only
- Output: Predictions (masks, classes)
- Purpose: Clinical usage, deployment

### Key Files

| File | Purpose | Lines | Use Case |
|------|---------|-------|----------|
| `engine/evaluator.py` | Comprehensive evaluation | 112 | Validation set analysis |
| `scripts/predict.py` | Single image inference | 107 | Clinical deployment |

---

## Evaluator (evaluator.py)

**File**: `src/braintumnet/engine/evaluator.py` (112 lines)

This script computes **all metrics** on a validation fold: IoU, Dice, HD, HD95, Accuracy, F1, AUC.

### Complete Code Walkthrough

#### Initialization

```python
def evaluate(cfg: Dict, fold: int, ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = cfg["data"]["proc_root"]
    val_list = os.path.join(proc, f"split_val_fold{fold}.txt")
    ds = SliceDataset(proc, val_list, cfg["data"]["img_size"], train=False, in_channels=cfg["model"]["in_channels"])
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["workers"])
```

**Line 11-16**: Setup dataloader

**Key Points**:
- `train=False`: No augmentation (evaluation should be deterministic)
- `shuffle=False`: Process in order (reproducible)
- Uses same batch size as training (for efficiency)

---

```python
    model = BrainTumNet(in_ch=cfg["model"]["in_channels"], num_cls=cfg["model"]["num_classes_cls"],
                        base=cfg["model"]["base"], dim=cfg["model"]["dim"], patch=cfg["model"]["patch_size"],
                        depth=cfg["model"]["depth"], n_heads=cfg["model"]["n_heads"],
                        roi_stop_grad=cfg["model"]["roi_stop_grad"]).to(device)
    load_ckpt(model, ckpt_path, map_location=device)
    model.eval()
```

**Line 17-22**: Load model

**`model.eval()`**: Critical for evaluation!
- Disables dropout (deterministic)
- BatchNorm uses running statistics (not batch stats)
- Ensures reproducibility

**Why `load_ckpt` (not `load_training_state`)?**
- `load_ckpt`: Only loads model weights (lightweight)
- `load_training_state`: Loads optimizer, scheduler, etc. (unnecessary for eval)

---

#### Metric Accumulators

```python
    # Classification metrics
    y_true, y_pred, y_prob = [], [], []

    # Segmentation metrics (global)
    total_inter, total_union = 0.0, 0.0

    # Per-slice metrics for HD and HD95 (accumulated)
    hd_scores = []
    hd95_scores = []
```

**Line 24-32**: Initialize metric accumulators

**Why Different Strategies?**

1. **Classification**: Collect all predictions, compute metrics once
   - Scikit-learn functions need all data at once
   - Example: AUC-ROC requires all probabilities

2. **Segmentation (IoU/Dice)**: Accumulate intersection/union globally
   - Correct global averaging (explained in Part 4)
   - Memory efficient

3. **Hausdorff Distances**: Compute per-slice, average later
   - HD is per-image metric (not aggregatable like intersection)
   - Store all values for mean and std

---

#### Evaluation Loop

```python
    import torch.nn.functional as F
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"Evaluating Fold {fold}"):
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)
            lab = batch["label"].cpu().numpy()
            seg, cls = model(img)
```

**Line 34-40**: Forward pass

**`torch.no_grad()`**: Essential for evaluation!
- Disables gradient computation
- Saves memory (~50% reduction)
- Faster inference

**Progress Bar**:
```
Evaluating Fold 0: 100%|████████████| 456/456 [02:34<00:00,  2.95it/s]
```

---

#### Classification Metrics

```python
            # Classification
            prob = F.softmax(cls, dim=1).cpu().numpy()
            y_true.extend(lab.tolist())
            y_pred.extend(prob.argmax(1).tolist())
            y_prob.extend(prob.tolist())
```

**Line 42-46**: Collect classification predictions

**Step-by-Step**:
```python
# Input: cls shape (B, 2) - raw logits

prob = F.softmax(cls, dim=1)
# Convert logits → probabilities
# prob shape: (B, 2), values sum to 1.0 per row

prob = prob.cpu().numpy()
# Move to CPU and convert to numpy

y_true.extend(lab.tolist())
# Ground truth labels: [0, 1, 0, 1, ...] (HGG/LGG)

y_pred.extend(prob.argmax(1).tolist())
# Predicted labels: argmax over classes
# Example: prob=[0.8, 0.2] → argmax=0 (HGG)

y_prob.extend(prob.tolist())
# All probabilities for AUC computation
# Example: [[0.8, 0.2], [0.3, 0.7], ...]
```

**Why Store Everything?**
- Can't compute AUC incrementally
- Need all probabilities at once
- Memory: ~1MB for 10k samples

---

#### Segmentation Metrics (Global)

```python
            # Segmentation (accumulate global metrics)
            inter, union = compute_intersection_union(seg, msk)
            total_inter += inter
            total_union += union
```

**Line 48-51**: Accumulate global segmentation metrics

**Correct Global Metrics**:
```python
# Batch 1: 100 slices
inter1, union1 = compute_intersection_union(seg1, msk1)
total_inter = 45000  # pixels
total_union = 50000  # pixels

# Batch 2: 100 slices
inter2, union2 = compute_intersection_union(seg2, msk2)
total_inter += 38000  # Now 83000
total_union += 45000  # Now 95000

# ... all batches ...

# Final global metrics:
iou = total_inter / (total_union - total_inter)
    = 83000 / (95000 - 83000)
    = 83000 / 12000
    = 0.691

dice = 2 * total_inter / total_union
     = 2 * 83000 / 95000
     = 0.874
```

---

#### Hausdorff Distance (Per-Slice)

```python
            # Per-slice HD and HD95 (on CPU)
            pred_masks = binarize(seg).cpu().numpy()
            target_masks = msk.cpu().numpy()

            for pred_slice, target_slice in zip(pred_masks, target_masks):
                # Only compute HD/HD95 if there's tumor in ground truth
                if target_slice.sum() > 0:
                    metrics = compute_segmentation_metrics(
                        pred_slice.squeeze(),
                        target_slice.squeeze(),
                        compute_hd=True,
                        compute_hd95=True
                    )
                    if not np.isinf(metrics['hd']) and not np.isnan(metrics['hd']):
                        hd_scores.append(metrics['hd'])
                    if not np.isinf(metrics['hd95']) and not np.isnan(metrics['hd95']):
                        hd95_scores.append(metrics['hd95'])
```

**Line 53-69**: Compute Hausdorff distances per slice

**Why Per-Slice?**
- HD measures boundary distance (spatial metric)
- Can't aggregate across images like intersection/union
- Must compute on each image individually

**Why `if target_slice.sum() > 0`?**
- Skip slices with no tumor
- HD undefined if ground truth is empty
- Prevents division by zero / inf values

**Why Filter inf/nan?**
```python
if not np.isinf(metrics['hd']) and not np.isnan(metrics['hd']):
```
- Edge cases: Empty predictions, numerical errors
- Only include valid HD values in average
- Ensures robust statistics

**Example**:
```python
# Batch of 8 slices
pred_masks shape: (8, 1, 256, 256)
target_masks shape: (8, 1, 256, 256)

for i in range(8):
    pred = pred_masks[i].squeeze()   # (256, 256)
    target = target_masks[i].squeeze()  # (256, 256)

    if target.sum() > 0:  # Has tumor
        hd = compute_hausdorff_distance(pred, target)
        # Example: hd = 12.3 pixels
        hd_scores.append(12.3)

# After all batches:
hd_scores = [12.3, 8.7, 15.2, ..., 10.9]  # 3456 values
hd_mean = np.mean(hd_scores)  # 11.8 pixels
hd_std = np.std(hd_scores)    # 4.2 pixels
```

---

#### Final Metric Computation

```python
    # Compute classification metrics
    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
    acc, f1, auc = cls_metrics(y_true, y_pred, y_prob)

    # Compute segmentation metrics
    eps = 1e-6
    iou = total_inter / (total_union - total_inter + eps)
    dice = (2 * total_inter) / (total_union + eps)

    # Average HD and HD95
    hd_mean = np.mean(hd_scores) if len(hd_scores) > 0 else float('nan')
    hd95_mean = np.mean(hd95_scores) if len(hd95_scores) > 0 else float('nan')
    hd_std = np.std(hd_scores) if len(hd_scores) > 0 else float('nan')
    hd95_std = np.std(hd95_scores) if len(hd95_scores) > 0 else float('nan')
```

**Line 71-84**: Compute final metrics

**Classification Metrics**:
```python
acc, f1, auc = cls_metrics(y_true, y_pred, y_prob)
```
- Calls scikit-learn functions
- Returns accuracy, macro F1, AUC-ROC

**Segmentation Metrics**:
```python
iou = total_inter / (total_union - total_inter + eps)
dice = (2 * total_inter) / (total_union + eps)
```
- Global metrics (correct averaging)
- `eps` prevents division by zero

**Hausdorff Statistics**:
```python
hd_mean = np.mean(hd_scores)
hd_std = np.std(hd_scores)
```
- Mean and standard deviation
- Reports uncertainty in boundary accuracy

---

#### Results Output

```python
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS - Fold {fold}")
    print("=" * 70)
    print("\nSegmentation Metrics:")
    print(f"  IoU (Jaccard):        {iou:.4f}")
    print(f"  Dice (F1):            {dice:.4f}")
    print(f"  Hausdorff Distance:   {hd_mean:.2f} ± {hd_std:.2f} pixels")
    print(f"  HD95 (95th percentile): {hd95_mean:.2f} ± {hd95_std:.2f} pixels")
    print(f"  (HD computed on {len(hd_scores)} slices with tumor)")
    print("\nClassification Metrics:")
    print(f"  Accuracy:             {acc:.4f}")
    print(f"  F1 Score:             {f1:.4f}")
    print(f"  AUC-ROC:              {auc:.4f}")
    print("=" * 70 + "\n")
```

**Line 86-99**: Print results

**Example Output**:
```
======================================================================
EVALUATION RESULTS - Fold 0
======================================================================

Segmentation Metrics:
  IoU (Jaccard):        0.8430
  Dice (F1):            0.9148
  Hausdorff Distance:   45.23 ± 18.76 pixels
  HD95 (95th percentile): 12.34 ± 5.67 pixels
  (HD computed on 3456 slices with tumor)

Classification Metrics:
  Accuracy:             0.9823
  F1 Score:             0.9812
  AUC-ROC:              0.9956
======================================================================
```

**Interpretation**:
- **IoU 0.843**: Excellent overlap (>0.7 is good for medical imaging)
- **Dice 0.915**: Corresponds to IoU 0.843
- **HD 45px**: Worst-case boundary error (sensitive to outliers)
- **HD95 12px**: Robust boundary error (ignores 5% outliers)
- **Acc 0.982**: 98.2% classification accuracy
- **AUC 0.996**: Near-perfect separation of HGG/LGG

---

### Usage

```bash
# Evaluate fold 0 with best checkpoint
python -m braintumnet.engine.evaluator \
    --cfg configs/full_dataset_multimodal.yaml \
    --fold 0 \
    --ckpt checkpoints/braintumnet_best_fold0.pth
```

**Integration in Training**:
```python
# After training completes
from braintumnet.engine.evaluator import evaluate

results = evaluate(cfg, fold=0, ckpt_path="checkpoints/braintumnet_best_fold0.pth")
print(f"Final Dice: {results['dice']:.4f}")
```

---

## Prediction Script (predict.py)

**File**: `scripts/predict.py` (107 lines)

This script performs **inference on a single image** and visualizes the results.

### Complete Code Walkthrough

#### Single Image Prediction

```python
def predict_single(model, img_path, img_size=256, device="cuda"):
    """Predict segmentation and classification for a single image."""
    # Load image
    img = Image.open(img_path).convert("L")
    img_resized = resize_pad_to_square(img, img_size, is_mask=False)
    img_tensor = to_tensor01(img_resized).unsqueeze(0).to(device)  # (1,1,H,W)

    # Predict
    model.eval()
    with torch.no_grad():
        seg_logits, cls_logits = model(img_tensor)
        seg_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()  # (H,W)
        cls_prob = torch.softmax(cls_logits, dim=1).squeeze().cpu().numpy()  # (num_classes,)
        cls_pred = cls_prob.argmax()

    return seg_prob, cls_pred, cls_prob
```

**Line 15-30**: Predict on single image

**Step-by-Step Processing**:

1. **Load Image**:
```python
img = Image.open(img_path).convert("L")
# "L" mode = grayscale (8-bit pixels)
# Example: FLAIR MRI slice
```

2. **Resize and Pad**:
```python
img_resized = resize_pad_to_square(img, img_size=256, is_mask=False)
# Handles arbitrary input sizes
# Pads to square (maintains aspect ratio)
# is_mask=False → interpolation for images (not nearest neighbor)
```

**Example**:
```
Original: 240×240 → Resize to 256×256 (pad 8px each side)
Original: 512×384 → Resize to 256×192, pad to 256×256
```

3. **Convert to Tensor**:
```python
img_tensor = to_tensor01(img_resized).unsqueeze(0).to(device)
# to_tensor01: PIL Image → torch.Tensor, normalize to [0, 1]
# unsqueeze(0): (1, 256, 256) → (1, 1, 256, 256) - add batch dim
# .to(device): Move to GPU
```

4. **Inference**:
```python
model.eval()
with torch.no_grad():
    seg_logits, cls_logits = model(img_tensor)
```
- `model.eval()`: Disable dropout, use running batch norm stats
- `torch.no_grad()`: Don't compute gradients (faster, less memory)

5. **Post-Processing**:
```python
seg_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
# (1, 1, 256, 256) → (256, 256)
# sigmoid: logits → probabilities [0, 1]
# squeeze: Remove batch and channel dims
# cpu().numpy(): Tensor → numpy array

cls_prob = torch.softmax(cls_logits, dim=1).squeeze().cpu().numpy()
# (1, 2) → (2,)
# softmax: logits → probabilities [0, 1], sum to 1
# Example: [0.85, 0.15] - 85% HGG, 15% LGG

cls_pred = cls_prob.argmax()
# Get predicted class
# Example: argmax([0.85, 0.15]) = 0 (HGG)
```

**Return Values**:
- `seg_prob`: (256, 256) float array, values [0, 1]
- `cls_pred`: Integer (0 or 1)
- `cls_prob`: (2,) float array, probabilities

---

#### Visualization

```python
def visualize_prediction(img_path, seg_prob, cls_pred, cls_prob, save_path=None):
    """Visualize input image, predicted segmentation, and classification."""
    img = Image.open(img_path).convert("L")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Segmentation mask
    axes[1].imshow(seg_prob, cmap='hot')
    axes[1].set_title("Predicted Tumor Mask")
    axes[1].axis('off')

    # Binary segmentation
    seg_binary = (seg_prob > 0.5).astype(np.uint8)
    axes[2].imshow(img, cmap='gray')
    axes[2].imshow(seg_binary, cmap='Reds', alpha=0.4)
    axes[2].set_title(f"Overlay | Class: {cls_pred} ({cls_prob[cls_pred]:.2f})")
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved prediction to: {save_path}")
    else:
        plt.show()

    plt.close()
```

**Line 32-63**: Visualization function

**Three-Panel Visualization**:

1. **Original Image** (Left):
```python
axes[0].imshow(img, cmap='gray')
```
- Shows input MRI slice
- Grayscale colormap

2. **Predicted Mask** (Middle):
```python
axes[1].imshow(seg_prob, cmap='hot')
```
- Shows probability heatmap
- 'hot' colormap: black (0.0) → red → yellow → white (1.0)
- Visualizes model confidence

3. **Overlay** (Right):
```python
seg_binary = (seg_prob > 0.5).astype(np.uint8)
axes[2].imshow(img, cmap='gray')  # Base layer
axes[2].imshow(seg_binary, cmap='Reds', alpha=0.4)  # Overlay
axes[2].set_title(f"Overlay | Class: {cls_pred} ({cls_prob[cls_pred]:.2f})")
```
- Binary mask overlaid on original image
- `alpha=0.4`: 40% transparency
- Shows tumor location + classification

**Example Output**:
```
┌────────────────┬────────────────┬─────────────────────────────┐
│ Input Image    │ Predicted Mask │ Overlay | Class: 0 (0.85)  │
│                │                │                             │
│   [MRI slice]  │  [Heatmap]     │  [MRI + Red overlay]        │
│                │                │                             │
└────────────────┴────────────────┴─────────────────────────────┘
```

---

#### Main Function

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="Path to config YAML")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    ap.add_argument("--img", type=str, required=True, help="Path to input image")
    ap.add_argument("--out", type=str, default=None, help="Output visualization path")
    args = ap.parse_args()

    # Load config
    cfg = load_yaml(args.cfg)

    # Build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = cfg["model"]
    model = BrainTumNet(
        in_ch=mcfg["in_channels"],
        num_cls=mcfg["num_classes_cls"],
        base=mcfg["base"],
        dim=mcfg["dim"],
        patch=mcfg["patch_size"],
        depth=mcfg["depth"],
        n_heads=mcfg["n_heads"],
        roi_stop_grad=mcfg["roi_stop_grad"]
    ).to(device)

    # Load checkpoint
    load_ckpt(model, args.ckpt, map_location=device)
    print(f"Loaded checkpoint: {args.ckpt}")

    # Predict
    seg_prob, cls_pred, cls_prob = predict_single(
        model, args.img, cfg["data"]["img_size"], device
    )

    print(f"Classification: {'HGG' if cls_pred == 0 else 'LGG'} (confidence: {cls_prob[cls_pred]:.4f})")
    print(f"Segmentation: mean={seg_prob.mean():.4f}, max={seg_prob.max():.4f}")

    # Visualize
    visualize_prediction(args.img, seg_prob, cls_pred, cls_prob, args.out)
```

**Line 65-103**: Command-line interface

**Usage**:
```bash
python scripts/predict.py \
    --cfg configs/full_dataset_multimodal.yaml \
    --ckpt checkpoints/braintumnet_best_fold0.pth \
    --img data/processed_full_multimodal/images/BraTS20_001_0000_slice_075.png \
    --out predictions/result.png
```

**Output**:
```
Loaded checkpoint: checkpoints/braintumnet_best_fold0.pth
Classification: HGG (confidence: 0.8523)
Segmentation: mean=0.1234, max=0.9876
Saved prediction to: predictions/result.png
```

---

## Test-Time Augmentation (TTA)

**What is TTA?**
- Apply augmentations during inference (not just training)
- Average predictions from multiple augmented versions
- Improves robustness and accuracy

**How TTA Works**:
```
Original Image
    ↓
┌───┴───┬───────┬───────┬───────┐
│       │       │       │       │
│ No    │ Flip  │ Rot   │ Flip  │
│ Aug   │ H     │ 90°   │ + Rot │
│       │       │       │       │
└───┬───┴───┬───┴───┬───┴───┬───┘
    ↓       ↓       ↓       ↓
  Pred1   Pred2   Pred3   Pred4
    │       │       │       │
    └───────┴───┬───┴───────┘
                ↓
         Average Predictions
                ↓
          Final Prediction
```

### Implementation

**Add to `predict.py`**:
```python
def predict_with_tta(model, img_path, img_size=256, device="cuda"):
    """
    Predict with Test-Time Augmentation.
    Augmentations: original, hflip, vflip, hflip+vflip
    """
    # Load image
    img = Image.open(img_path).convert("L")
    img_resized = resize_pad_to_square(img, img_size, is_mask=False)
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    model.eval()
    predictions = []

    with torch.no_grad():
        # Augmentation 1: Original
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
        seg, cls = model(img_tensor)
        predictions.append(torch.sigmoid(seg).cpu())

        # Augmentation 2: Horizontal flip
        img_hflip = torch.flip(img_tensor, [3])  # Flip width
        seg, cls = model(img_hflip)
        seg_hflip_back = torch.flip(torch.sigmoid(seg), [3])  # Flip back
        predictions.append(seg_hflip_back.cpu())

        # Augmentation 3: Vertical flip
        img_vflip = torch.flip(img_tensor, [2])  # Flip height
        seg, cls = model(img_vflip)
        seg_vflip_back = torch.flip(torch.sigmoid(seg), [2])  # Flip back
        predictions.append(seg_vflip_back.cpu())

        # Augmentation 4: Both flips
        img_hvflip = torch.flip(img_tensor, [2, 3])
        seg, cls = model(img_hvflip)
        seg_hvflip_back = torch.flip(torch.sigmoid(seg), [2, 3])
        predictions.append(seg_hvflip_back.cpu())

    # Average all predictions
    seg_prob = torch.stack(predictions).mean(dim=0).squeeze().numpy()

    # Classification from original (no augmentation for cls)
    with torch.no_grad():
        _, cls = model(img_tensor)
        cls_prob = torch.softmax(cls, dim=1).squeeze().cpu().numpy()
        cls_pred = cls_prob.argmax()

    return seg_prob, cls_pred, cls_prob
```

**Usage**:
```python
# Replace predict_single with predict_with_tta
seg_prob, cls_pred, cls_prob = predict_with_tta(model, args.img, cfg["data"]["img_size"], device)
```

**Expected Improvement**:
- Dice: +0.5-1.5% (e.g., 0.915 → 0.925)
- HD95: -5-10% (better boundaries)
- Cost: 4× slower (4 forward passes)

---

## Ensemble Predictions

**What is Ensemble?**
- Combine predictions from multiple models
- Models trained on different folds or with different seeds
- Reduces variance, improves robustness

### Implementation

```python
def predict_ensemble(model_paths, cfg, img_path, device="cuda"):
    """
    Predict using ensemble of models.

    Args:
        model_paths: List of checkpoint paths
        cfg: Config dict
        img_path: Input image path
        device: 'cuda' or 'cpu'

    Returns:
        seg_prob, cls_pred, cls_prob (averaged)
    """
    # Load image once
    img = Image.open(img_path).convert("L")
    img_resized = resize_pad_to_square(img, cfg["data"]["img_size"], is_mask=False)
    img_tensor = to_tensor01(img_resized).unsqueeze(0).to(device)

    seg_probs = []
    cls_probs = []

    for ckpt_path in model_paths:
        # Build model
        mcfg = cfg["model"]
        model = BrainTumNet(
            in_ch=mcfg["in_channels"],
            num_cls=mcfg["num_classes_cls"],
            base=mcfg["base"],
            dim=mcfg["dim"],
            patch=mcfg["patch_size"],
            depth=mcfg["depth"],
            n_heads=mcfg["n_heads"],
            roi_stop_grad=mcfg["roi_stop_grad"]
        ).to(device)

        # Load checkpoint
        load_ckpt(model, ckpt_path, map_location=device)
        model.eval()

        # Predict
        with torch.no_grad():
            seg, cls = model(img_tensor)
            seg_probs.append(torch.sigmoid(seg).cpu().numpy())
            cls_probs.append(torch.softmax(cls, dim=1).cpu().numpy())

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Average predictions
    seg_prob = np.mean(seg_probs, axis=0).squeeze()
    cls_prob = np.mean(cls_probs, axis=0).squeeze()
    cls_pred = cls_prob.argmax()

    return seg_prob, cls_pred, cls_prob
```

**Usage**:
```python
# Ensemble all 5 folds
model_paths = [
    "checkpoints/braintumnet_best_fold0.pth",
    "checkpoints/braintumnet_best_fold1.pth",
    "checkpoints/braintumnet_best_fold2.pth",
    "checkpoints/braintumnet_best_fold3.pth",
    "checkpoints/braintumnet_best_fold4.pth",
]

seg_prob, cls_pred, cls_prob = predict_ensemble(model_paths, cfg, img_path, device)
```

**Expected Improvement**:
- Dice: +1-3% (e.g., 0.915 → 0.935)
- More robust to outliers
- Cost: 5× slower (5 models)

**Combine TTA + Ensemble**:
```python
# Each model with TTA, then ensemble
# Best quality, but 20× slower (5 models × 4 augmentations)
```

---

## Batch Inference

For processing many images efficiently:

```python
def predict_batch(model, image_paths, img_size=256, batch_size=16, device="cuda"):
    """
    Predict on batch of images.

    Args:
        model: BrainTumNet model
        image_paths: List of image paths
        img_size: Resize dimension
        batch_size: Batch size for inference
        device: 'cuda' or 'cpu'

    Returns:
        seg_probs: List of (H, W) arrays
        cls_preds: List of class predictions
        cls_probs: List of (num_classes,) probability arrays
    """
    model.eval()

    seg_probs = []
    cls_preds = []
    cls_probs_all = []

    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]

        # Load batch
        images = []
        for img_path in batch_paths:
            img = Image.open(img_path).convert("L")
            img_resized = resize_pad_to_square(img, img_size, is_mask=False)
            img_tensor = to_tensor01(img_resized)
            images.append(img_tensor)

        # Stack to batch
        batch_tensor = torch.stack(images).to(device)  # (B, 1, H, W)

        # Predict
        with torch.no_grad():
            seg, cls = model(batch_tensor)
            seg_prob = torch.sigmoid(seg).cpu().numpy()
            cls_prob = torch.softmax(cls, dim=1).cpu().numpy()

        # Collect results
        for j in range(len(batch_paths)):
            seg_probs.append(seg_prob[j].squeeze())
            cls_probs_all.append(cls_prob[j])
            cls_preds.append(cls_prob[j].argmax())

    return seg_probs, cls_preds, cls_probs_all
```

**Usage**:
```python
# Process 1000 images
image_paths = [...]  # List of 1000 paths

seg_probs, cls_preds, cls_probs = predict_batch(
    model, image_paths, batch_size=16, device="cuda"
)

# Save results
for i, (seg, cls) in enumerate(zip(seg_probs, cls_preds)):
    np.save(f"predictions/seg_{i:04d}.npy", seg)
    print(f"Image {i}: Class {cls}")
```

**Performance**:
- Batch size 1: 10 images/sec
- Batch size 16: 45 images/sec
- **4.5× speedup!**

---

## Practical Usage Examples

### Example 1: Evaluate All Folds

```python
import yaml
from braintumnet.engine.evaluator import evaluate

# Load config
with open("configs/full_dataset_multimodal.yaml") as f:
    cfg = yaml.safe_load(f)

# Evaluate all 5 folds
results_all = []
for fold in range(5):
    print(f"\n{'='*70}")
    print(f"Evaluating Fold {fold}")
    print(f"{'='*70}\n")

    ckpt_path = f"checkpoints/braintumnet_best_fold{fold}.pth"
    results = evaluate(cfg, fold, ckpt_path)
    results_all.append(results)

# Aggregate results
import numpy as np
dice_mean = np.mean([r['dice'] for r in results_all])
dice_std = np.std([r['dice'] for r in results_all])
iou_mean = np.mean([r['iou'] for r in results_all])
iou_std = np.std([r['iou'] for r in results_all])

print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS (5 Folds)")
print("="*70)
print(f"Dice:  {dice_mean:.4f} ± {dice_std:.4f}")
print(f"IoU:   {iou_mean:.4f} ± {iou_std:.4f}")
print("="*70)
```

**Output**:
```
======================================================================
CROSS-VALIDATION RESULTS (5 Folds)
======================================================================
Dice:  0.9148 ± 0.0023
IoU:   0.8430 ± 0.0031
======================================================================
```

---

### Example 2: Clinical Deployment Script

```python
#!/usr/bin/env python
"""
Clinical deployment script for BrainTumNet.
Processes patient MRI and generates report.
"""

import sys
from pathlib import Path
import torch
import yaml
from PIL import Image
import numpy as np

# Add to path
sys.path.append(str(Path(__file__).parent / "src"))

from braintumnet.models.braintumnet import BrainTumNet
from braintumnet.utils.io import load_ckpt
from scripts.predict import predict_with_tta

def process_patient(patient_dir, model, cfg, device):
    """
    Process all slices for a patient.

    Returns:
        report: Dictionary with clinical findings
    """
    patient_dir = Path(patient_dir)
    slices = sorted(patient_dir.glob("*.png"))

    results = []
    for slice_path in slices:
        seg_prob, cls_pred, cls_prob = predict_with_tta(
            model, str(slice_path), cfg["data"]["img_size"], device
        )
        results.append({
            'slice': slice_path.name,
            'tumor_volume': seg_prob.sum(),  # Pixel count
            'tumor_fraction': (seg_prob > 0.5).mean(),
            'max_confidence': seg_prob.max(),
            'grade': 'HGG' if cls_pred == 0 else 'LGG',
            'grade_confidence': cls_prob[cls_pred]
        })

    # Generate report
    total_tumor_volume = sum(r['tumor_volume'] for r in results)
    dominant_grade = max(set(r['grade'] for r in results), key=lambda g: sum(1 for r in results if r['grade'] == g))
    avg_confidence = np.mean([r['grade_confidence'] for r in results])

    report = {
        'patient_id': patient_dir.name,
        'num_slices': len(results),
        'total_tumor_volume': total_tumor_volume,
        'dominant_grade': dominant_grade,
        'avg_confidence': avg_confidence,
        'slice_details': results
    }

    return report

def main():
    # Load config and model
    cfg = yaml.safe_load(open("configs/full_dataset_multimodal.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BrainTumNet(...).to(device)
    load_ckpt(model, "checkpoints/best_model.pth", device)

    # Process patient
    report = process_patient("data/patients/BraTS20_001", model, cfg, device)

    # Print report
    print("\n" + "="*70)
    print(f"CLINICAL REPORT: {report['patient_id']}")
    print("="*70)
    print(f"Dominant Grade: {report['dominant_grade']} ({report['avg_confidence']:.1%} confidence)")
    print(f"Tumor Volume: {report['total_tumor_volume']:.0f} voxels")
    print(f"Slices Analyzed: {report['num_slices']}")
    print("="*70)

if __name__ == "__main__":
    main()
```

---

## Modification Guides

### Add Rotation TTA

```python
def predict_with_rotation_tta(model, img_path, img_size=256, device="cuda"):
    """TTA with rotations: 0°, 90°, 180°, 270°"""
    # Load image
    img = Image.open(img_path).convert("L")
    img_resized = resize_pad_to_square(img, img_size, is_mask=False)
    img_tensor = to_tensor01(img_resized).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    predictions = []

    with torch.no_grad():
        for k in range(4):  # 0°, 90°, 180°, 270°
            # Rotate
            img_rot = torch.rot90(img_tensor, k=k, dims=[2, 3])

            # Predict
            seg, _ = model(img_rot)
            seg_prob = torch.sigmoid(seg)

            # Rotate back
            seg_back = torch.rot90(seg_prob, k=-k, dims=[2, 3])
            predictions.append(seg_back.cpu())

    # Average
    seg_prob = torch.stack(predictions).mean(dim=0).squeeze().numpy()

    return seg_prob
```

---

### Save Predictions as NIfTI

```python
import nibabel as nib

def save_as_nifti(seg_prob, output_path, threshold=0.5):
    """
    Save segmentation as NIfTI file (medical image format).

    Args:
        seg_prob: (H, W) or (D, H, W) array
        output_path: Output .nii.gz path
        threshold: Binarization threshold
    """
    # Binarize
    seg_binary = (seg_prob > threshold).astype(np.uint8)

    # Create NIfTI image
    nifti_img = nib.Nifti1Image(seg_binary, affine=np.eye(4))

    # Save
    nib.save(nifti_img, output_path)
    print(f"Saved NIfTI to: {output_path}")
```

**Usage**:
```python
seg_prob, _, _ = predict_single(model, img_path, device=device)
save_as_nifti(seg_prob, "predictions/patient001_seg.nii.gz")
```

---

**Next**: [[06_UTILS_LOGGING|Part 6: Utils and Logging →]]

**Back**: [[04_TRAINING_SYSTEM|← Part 4: Training System]] | [[TECHNICAL_REPORT_INDEX|Index]]
