import os, argparse, sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from braintumnet.utils.io import load_yaml, load_ckpt
from braintumnet.models.braintumnet import BrainTumNet
from braintumnet.data.transforms import resize_pad_to_square, to_tensor01

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

if __name__ == "__main__":
    main()
