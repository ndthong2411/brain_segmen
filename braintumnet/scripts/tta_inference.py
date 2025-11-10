"""
Test-Time Augmentation (TTA) Inference

Applies 8 augmentations and averages predictions for improved accuracy.

Expected gain: +2-3% IoU (NO retraining needed!)

Usage:
    python braintumnet/scripts/tta_inference.py checkpoints/braintumnet_best_fold0.pth
    python braintumnet/scripts/tta_inference.py checkpoints/braintumnet_best_fold4.pth

Author: BrainTumNet Phase 3
Date: 2025-10-14
"""

import argparse
import torch
import torch.nn.functional as F
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from braintumnet.models.braintumnet import BrainTumNet
from braintumnet.models.braintumnet_v2 import BrainTumNetV2
from braintumnet.data.brats2020_dataset import SliceDataset
from torch.utils.data import DataLoader


def tta_predict_single(model, image, device='cuda'):
    """
    Apply Test-Time Augmentation to a single image

    8 augmentations:
    1. Original
    2. Horizontal flip
    3. Vertical flip
    4. Rotate 90°
    5. Rotate 180°
    6. Rotate 270°
    7. Horizontal flip + Rotate 90°
    8. Vertical flip + Rotate 90°

    Args:
        model: Trained model
        image: (1, C, H, W) input tensor
        device: device to run on

    Returns:
        pred_prob: (1, num_classes, H, W) averaged softmax probabilities
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        image = image.to(device)

        # 1. Original
        output = model(image)
        seg = output[0] if isinstance(output, tuple) else output
        predictions.append(F.softmax(seg, dim=1))

        # 2. Horizontal flip
        img_hflip = torch.flip(image, dims=[3])
        output = model(img_hflip)
        seg = output[0] if isinstance(output, tuple) else output
        seg = torch.flip(seg, dims=[3])  # Flip back
        predictions.append(F.softmax(seg, dim=1))

        # 3. Vertical flip
        img_vflip = torch.flip(image, dims=[2])
        output = model(img_vflip)
        seg = output[0] if isinstance(output, tuple) else output
        seg = torch.flip(seg, dims=[2])  # Flip back
        predictions.append(F.softmax(seg, dim=1))

        # 4-6. Rotations (90°, 180°, 270°)
        for k in [1, 2, 3]:
            img_rot = torch.rot90(image, k, dims=[2, 3])
            output = model(img_rot)
            seg = output[0] if isinstance(output, tuple) else output
            seg = torch.rot90(seg, -k, dims=[2, 3])  # Rotate back
            predictions.append(F.softmax(seg, dim=1))

        # 7. Horizontal flip + Rotate 90°
        img_hflip_rot = torch.rot90(torch.flip(image, dims=[3]), 1, dims=[2, 3])
        output = model(img_hflip_rot)
        seg = output[0] if isinstance(output, tuple) else output
        seg = torch.rot90(seg, -1, dims=[2, 3])
        seg = torch.flip(seg, dims=[3])
        predictions.append(F.softmax(seg, dim=1))

        # 8. Vertical flip + Rotate 90°
        img_vflip_rot = torch.rot90(torch.flip(image, dims=[2]), 1, dims=[2, 3])
        output = model(img_vflip_rot)
        seg = output[0] if isinstance(output, tuple) else output
        seg = torch.rot90(seg, -1, dims=[2, 3])
        seg = torch.flip(seg, dims=[2])
        predictions.append(F.softmax(seg, dim=1))

    # Average all predictions
    pred_mean = torch.stack(predictions).mean(dim=0)
    return pred_mean


def compute_iou(pred, target, num_classes=3, ignore_background=True):
    """Compute IoU for each class"""
    ious = []
    start_idx = 1 if ignore_background else 0

    for c in range(start_idx, num_classes):
        pred_c = (pred == c)
        target_c = (target == c)

        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        ious.append(iou)

    return ious


def main():
    parser = argparse.ArgumentParser(
        description='TTA Inference - just pass checkpoint path',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python braintumnet/scripts/tta_inference.py checkpoints/braintumnet_best_fold0.pth
  python braintumnet/scripts/tta_inference.py checkpoints/braintumnet_best_fold4.pth
        """
    )
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (default: results/tta_fold{N}.csv)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()

    # Extract fold number from checkpoint path
    import re
    match = re.search(r'fold(\d+)', args.checkpoint)
    fold = int(match.group(1)) if match else 0

    # Set default output path if not specified
    if args.output is None:
        args.output = f'results/tta_fold{fold}.csv'

    print(f"\nDetected fold: {fold}")

    # Load base config
    ROOT = Path(__file__).resolve().parents[1]
    cfg_path = ROOT / "configs" / "base.yaml"
    print(f"Loading config from: {cfg_path}")
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Override fold
    cfg['data']['fold'] = fold

    # Get data_root from config
    data_root = cfg['data']['proc_root']

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Determine model type
    model_type = cfg['model'].get('model_type', 'v1')

    if model_type == 'v2':
        model = BrainTumNetV2(
            in_ch=cfg['model']['in_channels'],
            num_cls=cfg['model']['num_classes_cls'],
            base=cfg['model']['base'],
            dim=cfg['model']['dim'],
            patch=cfg['model']['patch_size'],
            depth=cfg['model']['depth'],
            n_heads=cfg['model']['n_heads'],
            num_classes_seg=cfg['model']['num_classes_seg'],
            dropout=cfg['model'].get('dropout', 0.0),
            roi_stop_grad=cfg['model'].get('roi_stop_grad', True),
            deep_supervision=cfg['model'].get('deep_supervision', True),
            multi_scale_fusion=cfg['model'].get('multi_scale_fusion', False)
        )
    else:
        model = BrainTumNet(
            in_ch=cfg['model']['in_channels'],
            num_cls=cfg['model']['num_classes_cls'],
            base=cfg['model']['base'],
            dim=cfg['model']['dim'],
            patch=cfg['model']['patch_size'],
            depth=cfg['model']['depth'],
            n_heads=cfg['model']['n_heads'],
            roi_stop_grad=cfg['model'].get('roi_stop_grad', True),
            deep_supervision=cfg['model'].get('deep_supervision', True),
            num_classes_seg=cfg['model']['num_classes_seg']
        )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully ({model_type.upper()})")

    # Load validation dataset
    split_file = os.path.join(data_root, f'val_fold{fold}.csv')
    print(f"Loading validation data from: {split_file}")

    dataset = SliceDataset(
        proc_root=data_root,
        split_file=split_file,
        img_size=cfg['data']['img_size'],
        train=False,
        in_channels=cfg['model']['in_channels']
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Validation set: {len(dataset)} samples")
    print(f"\nStarting TTA inference (8 augmentations per sample)...")
    print(f"This will take ~8x longer than normal inference")

    # Run inference with TTA
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="TTA Inference")):
            img = batch['image']
            msk = batch['mask']

            # TTA prediction
            pred_prob = tta_predict_single(model, img, device)
            pred_class = pred_prob.argmax(dim=1).cpu()

            # Compute IoU
            target = msk.squeeze(1).cpu()  # Remove channel dim if present
            ious = compute_iou(pred_class[0], target[0], num_classes=3, ignore_background=True)

            results.append({
                'slice_id': batch['slice_id'][0],
                'tc_iou': ious[0],
                'ed_iou': ious[1],
                'mean_iou': np.mean(ious)
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)

    # Print summary
    print(f"\n{'='*70}")
    print("TTA Inference Results")
    print(f"{'='*70}")
    print(f"TC IoU:   {df['tc_iou'].mean():.4f} ± {df['tc_iou'].std():.4f}")
    print(f"ED IoU:   {df['ed_iou'].mean():.4f} ± {df['ed_iou'].std():.4f}")
    print(f"Mean IoU: {df['mean_iou'].mean():.4f} ± {df['mean_iou'].std():.4f}")
    print(f"\nResults saved to: {args.output}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
