# Phần 7: Inference và Deployment

> **🚀 Sử Dụng Model - Inference, Prediction, Visualization**
>
> Tài liệu này hướng dẫn cách sử dụng trained model để predict trên data mới.

---

## Mục Lục

1. [Single Image Inference](#1-single-image-inference)
2. [Batch Inference](#2-batch-inference)
3. [Post-Processing](#3-post-processing)
4. [Visualization](#4-visualization)
5. [Model Export](#5-model-export)
6. [Production Deployment](#6-production-deployment)

---

## 1. Single Image Inference

### Load Trained Model

```python
import torch
from src.braintumnet.models.braintumnet_v2 import BrainTumNetV2

def load_trained_model(checkpoint_path, device='cuda'):
    """
    Load trained model từ checkpoint
    
    Args:
        checkpoint_path: Path to .pth file
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Loaded model in eval mode
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model (config từ checkpoint hoặc hardcode)
    model = BrainTumNetV2(
        in_ch=4,
        num_cls=2,
        base=48,
        dim=384,
        patch=8,
        depth=4,
        n_heads=8,
        num_classes_seg=3,
        dropout=0.0,  # No dropout trong inference
        deep_supervision=False  # Chỉ cần main output
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device và eval mode
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best Dice: {checkpoint.get('best_val_dice', 'N/A'):.4f}")
    
    return model
```

### Predict Single Sample

```python
def predict_single(
    model, 
    flair_path, 
    t1_path, 
    t1ce_path, 
    t2_path,
    device='cuda'
):
    """
    Predict segmentation cho 1 slice
    
    Args:
        model: Trained BrainTumNet
        *_path: Paths to 4 modality images
        device: Device
    
    Returns:
        seg_pred: (H, W) numpy array với class indices {0, 1, 2}
        cls_pred: Classification (0=HGG, 1=LGG)
        seg_prob: (3, H, W) class probabilities
    """
    import cv2
    import numpy as np
    
    # Load images
    flair = cv2.imread(flair_path, cv2.IMREAD_GRAYSCALE)
    t1 = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
    t1ce = cv2.imread(t1ce_path, cv2.IMREAD_GRAYSCALE)
    t2 = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)
    
    # Stack modalities
    image = np.stack([flair, t1, t1ce, t2], axis=0)  # (4, H, W)
    
    # Normalize (z-score per modality)
    image_norm = np.zeros_like(image, dtype=np.float32)
    for i in range(4):
        mean = image[i].mean()
        std = image[i].std()
        image_norm[i] = (image[i] - mean) / (std + 1e-6)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_norm).unsqueeze(0)  # (1, 4, H, W)
    image_tensor = image_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        seg_logits, cls_logits = model(image_tensor)
    
    # Segmentation prediction
    seg_prob = torch.softmax(seg_logits, dim=1)  # (1, 3, H, W)
    seg_pred = seg_logits.argmax(dim=1)  # (1, H, W)
    
    # Classification prediction
    cls_pred = cls_logits.argmax(dim=1)  # (1,)
    
    # Convert to numpy
    seg_pred = seg_pred.cpu().numpy()[0]  # (H, W)
    seg_prob = seg_prob.cpu().numpy()[0]  # (3, H, W)
    cls_pred = cls_pred.cpu().item()
    
    return seg_pred, cls_pred, seg_prob
```

### Example Usage

```python
# Load model
model = load_trained_model('checkpoints/braintumnet_best_fold0.pth')

# Predict
seg_pred, cls_pred, seg_prob = predict_single(
    model,
    flair_path='data/processed_multiclass/flair/BraTS20_Training_001_0050.png',
    t1_path='data/processed_multiclass/t1/BraTS20_Training_001_0050.png',
    t1ce_path='data/processed_multiclass/t1ce/BraTS20_Training_001_0050.png',
    t2_path='data/processed_multiclass/t2/BraTS20_Training_001_0050.png'
)

print(f"Segmentation shape: {seg_pred.shape}")  # (256, 256)
print(f"Unique classes: {np.unique(seg_pred)}")  # [0 1 2]
print(f"Classification: {'HGG' if cls_pred == 0 else 'LGG'}")

# Probabilities
print(f"Background prob: {seg_prob[0].mean():.4f}")
print(f"Tumor Core prob: {seg_prob[1].mean():.4f}")
print(f"Edema prob: {seg_prob[2].mean():.4f}")
```

---

## 2. Batch Inference

### Predict Entire Dataset

```python
def predict_dataset(
    model,
    data_dir,
    fold,
    mode='val',
    batch_size=16,
    device='cuda',
    save_dir='predictions'
):
    """
    Predict toàn bộ validation/test set
    
    Args:
        model: Trained model
        data_dir: Data directory
        fold: Fold number
        mode: 'val' or 'test'
        batch_size: Batch size for inference
        save_dir: Directory to save predictions
    
    Returns:
        results: Dict với metrics và predictions
    """
    from src.braintumnet.data.dataset import BraTSDataset
    from torch.utils.data import DataLoader
    from src.braintumnet.multiclass_metrics import MulticlassMetricsAccumulator
    
    # Create dataset
    dataset = BraTSDataset(
        data_dir=data_dir,
        fold=fold,
        mode=mode,
        transform=None  # No augmentation
    )
    
    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Metrics accumulator
    accumulator = MulticlassMetricsAccumulator(num_classes=3)
    
    # Storage
    all_predictions = []
    all_probabilities = []
    all_cls_predictions = []
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Inference loop
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks, labels) in enumerate(tqdm(loader)):
            # Move to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            seg_logits, cls_logits = model(images)
            
            # Predictions
            seg_prob = torch.softmax(seg_logits, dim=1)
            seg_pred = seg_logits.argmax(dim=1)
            cls_pred = cls_logits.argmax(dim=1)
            
            # Update metrics
            accumulator.update(seg_pred, masks)
            
            # Store predictions
            all_predictions.append(seg_pred.cpu().numpy())
            all_probabilities.append(seg_prob.cpu().numpy())
            all_cls_predictions.append(cls_pred.cpu().numpy())
    
    # Compute final metrics
    metrics = accumulator.compute()
    
    # Concatenate predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_probabilities = np.concatenate(all_probabilities, axis=0)
    all_cls_predictions = np.concatenate(all_cls_predictions, axis=0)
    
    # Save predictions
    np.save(f"{save_dir}/predictions_fold{fold}.npy", all_predictions)
    np.save(f"{save_dir}/probabilities_fold{fold}.npy", all_probabilities)
    np.save(f"{save_dir}/cls_predictions_fold{fold}.npy", all_cls_predictions)
    
    # Save metrics
    with open(f"{save_dir}/metrics_fold{fold}.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Predictions saved to {save_dir}")
    print(f"  WT Dice: {metrics['WT']:.4f}")
    print(f"  TC Dice: {metrics['TC']:.4f}")
    print(f"  ED Dice: {metrics['ED']:.4f}")
    print(f"  Mean Dice: {metrics['Mean']:.4f}")
    
    return {
        'metrics': metrics,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'cls_predictions': all_cls_predictions
    }
```

### Test Time Augmentation (TTA)

```python
def predict_with_tta(model, image, device='cuda'):
    """
    Test-time augmentation để improve predictions
    
    Augmentations:
    1. Original
    2. Horizontal flip
    3. Vertical flip
    4. Horizontal + Vertical flip
    
    Returns:
        seg_pred: Averaged predictions
    """
    # Original
    with torch.no_grad():
        seg_logits1, _ = model(image)
        seg_prob1 = torch.softmax(seg_logits1, dim=1)
    
    # Horizontal flip
    image_hflip = torch.flip(image, dims=[3])
    with torch.no_grad():
        seg_logits2, _ = model(image_hflip)
        seg_prob2 = torch.softmax(seg_logits2, dim=1)
        seg_prob2 = torch.flip(seg_prob2, dims=[3])
    
    # Vertical flip
    image_vflip = torch.flip(image, dims=[2])
    with torch.no_grad():
        seg_logits3, _ = model(image_vflip)
        seg_prob3 = torch.softmax(seg_logits3, dim=1)
        seg_prob3 = torch.flip(seg_prob3, dims=[2])
    
    # Both flips
    image_hvflip = torch.flip(image, dims=[2, 3])
    with torch.no_grad():
        seg_logits4, _ = model(image_hvflip)
        seg_prob4 = torch.softmax(seg_logits4, dim=1)
        seg_prob4 = torch.flip(seg_prob4, dims=[2, 3])
    
    # Average probabilities
    seg_prob_avg = (seg_prob1 + seg_prob2 + seg_prob3 + seg_prob4) / 4.0
    
    # Final prediction
    seg_pred = seg_prob_avg.argmax(dim=1)
    
    return seg_pred, seg_prob_avg
```

---

## 3. Post-Processing

### Remove Small Components

```python
def remove_small_components(mask, min_size=50):
    """
    Remove small connected components
    
    Args:
        mask: (H, W) segmentation mask
        min_size: Minimum component size (pixels)
    
    Returns:
        mask_cleaned: Cleaned mask
    """
    from scipy.ndimage import label, sum as ndi_sum
    
    mask_cleaned = np.zeros_like(mask)
    
    for cls in [1, 2]:  # TC và ED
        # Binary mask for class
        binary_mask = (mask == cls).astype(np.uint8)
        
        # Connected components
        labeled, num_components = label(binary_mask)
        
        # Keep only large components
        for comp_id in range(1, num_components + 1):
            comp_mask = (labeled == comp_id)
            comp_size = comp_mask.sum()
            
            if comp_size >= min_size:
                mask_cleaned[comp_mask] = cls
    
    return mask_cleaned
```

### Fill Holes

```python
def fill_holes(mask):
    """
    Fill holes trong segmentation masks
    
    Args:
        mask: (H, W) segmentation
    
    Returns:
        mask_filled: Mask với holes filled
    """
    from scipy.ndimage import binary_fill_holes
    
    mask_filled = np.zeros_like(mask)
    
    for cls in [1, 2]:
        binary_mask = (mask == cls)
        filled = binary_fill_holes(binary_mask)
        mask_filled[filled] = cls
    
    return mask_filled
```

### Morphological Operations

```python
def morphological_closing(mask, kernel_size=3):
    """
    Morphological closing: Dilation → Erosion
    
    Smooths boundaries và fills small gaps
    """
    import cv2
    
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    
    mask_closed = np.zeros_like(mask)
    
    for cls in [1, 2]:
        binary_mask = (mask == cls).astype(np.uint8)
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        mask_closed[closed > 0] = cls
    
    return mask_closed
```

---

## 4. Visualization

### Overlay Segmentation

```python
def visualize_segmentation(
    flair, 
    seg_pred, 
    seg_gt=None,
    alpha=0.5,
    save_path=None
):
    """
    Visualize segmentation overlay trên FLAIR image
    
    Args:
        flair: (H, W) FLAIR image
        seg_pred: (H, W) predicted segmentation
        seg_gt: (H, W) ground truth (optional)
        alpha: Overlay transparency
        save_path: Path to save image
    """
    import matplotlib.pyplot as plt
    
    # Color map
    colors = {
        0: [0, 0, 0],       # Background: Black
        1: [255, 0, 0],     # Tumor Core: Red
        2: [0, 255, 0]      # Edema: Green
    }
    
    # Create RGB overlay
    overlay = np.zeros((*seg_pred.shape, 3), dtype=np.uint8)
    for cls, color in colors.items():
        overlay[seg_pred == cls] = color
    
    # Convert FLAIR to RGB
    flair_rgb = cv2.cvtColor(flair, cv2.COLOR_GRAY2RGB)
    
    # Blend
    blended = cv2.addWeighted(flair_rgb, 1-alpha, overlay, alpha, 0)
    
    # Plot
    if seg_gt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # FLAIR
        axes[0].imshow(flair, cmap='gray')
        axes[0].set_title('FLAIR')
        axes[0].axis('off')
        
        # Ground Truth
        gt_overlay = np.zeros((*seg_gt.shape, 3), dtype=np.uint8)
        for cls, color in colors.items():
            gt_overlay[seg_gt == cls] = color
        gt_blended = cv2.addWeighted(flair_rgb, 1-alpha, gt_overlay, alpha, 0)
        axes[1].imshow(gt_blended)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(blended)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(flair, cmap='gray')
        axes[0].set_title('FLAIR')
        axes[0].axis('off')
        
        axes[1].imshow(blended)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()
```

### 3D Volume Visualization

```python
def visualize_3d_volume(predictions_3d, spacing=(1, 1, 1)):
    """
    Visualize 3D segmentation volume
    
    Args:
        predictions_3d: (D, H, W) 3D segmentation
        spacing: Voxel spacing
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 4))
    
    # Axial view
    ax1 = fig.add_subplot(131)
    ax1.imshow(predictions_3d[predictions_3d.shape[0]//2], cmap='jet')
    ax1.set_title('Axial')
    ax1.axis('off')
    
    # Coronal view
    ax2 = fig.add_subplot(132)
    ax2.imshow(predictions_3d[:, predictions_3d.shape[1]//2], cmap='jet')
    ax2.set_title('Coronal')
    ax2.axis('off')
    
    # Sagittal view
    ax3 = fig.add_subplot(133)
    ax3.imshow(predictions_3d[:, :, predictions_3d.shape[2]//2], cmap='jet')
    ax3.set_title('Sagittal')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()
```

---

## 5. Model Export

### Export to ONNX

```python
def export_to_onnx(model, onnx_path, input_shape=(1, 4, 256, 256)):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: Trained model
        onnx_path: Output .onnx file path
        input_shape: Input tensor shape
    """
    import torch.onnx
    
    # Dummy input
    dummy_input = torch.randn(*input_shape).cuda()
    
    # Export
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['seg_output', 'cls_output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'seg_output': {0: 'batch_size'},
            'cls_output': {0: 'batch_size'}
        },
        opset_version=13
    )
    
    print(f"✓ Exported to {onnx_path}")
```

### TorchScript

```python
def export_to_torchscript(model, script_path):
    """
    Export to TorchScript (optimized for production)
    """
    model.eval()
    
    # Trace model
    example_input = torch.randn(1, 4, 256, 256).cuda()
    traced_model = torch.jit.trace(model, example_input)
    
    # Save
    traced_model.save(script_path)
    
    print(f"✓ Exported to {script_path}")
```

---

## 6. Production Deployment

### Flask API Example

```python
from flask import Flask, request, jsonify
import torch
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model globally
model = load_trained_model('checkpoints/best_model.pth')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint để predict
    
    Expected input: 4 images (FLAIR, T1, T1CE, T2) as multipart/form-data
    
    Returns: JSON với segmentation và classification
    """
    try:
        # Load images từ request
        flair = Image.open(io.BytesIO(request.files['flair'].read()))
        t1 = Image.open(io.BytesIO(request.files['t1'].read()))
        t1ce = Image.open(io.BytesIO(request.files['t1ce'].read()))
        t2 = Image.open(io.BytesIO(request.files['t2'].read()))
        
        # Convert to numpy
        flair_np = np.array(flair)
        t1_np = np.array(t1)
        t1ce_np = np.array(t1ce)
        t2_np = np.array(t2)
        
        # Preprocess
        image = np.stack([flair_np, t1_np, t1ce_np, t2_np], axis=0)
        image_norm = preprocess(image)
        
        # Predict
        image_tensor = torch.from_numpy(image_norm).unsqueeze(0).cuda()
        
        with torch.no_grad():
            seg_logits, cls_logits = model(image_tensor)
        
        # Post-process
        seg_pred = seg_logits.argmax(dim=1).cpu().numpy()[0]
        cls_pred = cls_logits.argmax(dim=1).cpu().item()
        
        # Response
        return jsonify({
            'segmentation': seg_pred.tolist(),
            'classification': 'HGG' if cls_pred == 0 else 'LGG',
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Docker Deployment

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model và code
COPY checkpoints/best_model.pth checkpoints/
COPY src/ src/
COPY app.py .

# Expose port
EXPOSE 5000

# Run
CMD ["python", "app.py"]
```

---

**[← Phần 6: Configuration](v_06_CONFIGURATION.md)** | **[Phần 8: Troubleshooting →](v_08_TROUBLESHOOTING.md)**
