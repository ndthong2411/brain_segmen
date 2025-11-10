"""
5-Fold Ensemble Inference

Averages predictions from all 5 fold models for improved generalization.

Expected gain: +3-4% IoU (use existing fold models!)

Usage:
    python braintumnet/scripts/ensemble_inference.py "checkpoints/braintumnet_best_fold*.pth"
    python braintumnet/scripts/ensemble_inference.py "checkpoints/braintumnet_best_fold*.pth" --use_tta

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
import glob

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from braintumnet.models.braintumnet import BrainTumNet
from braintumnet.models.braintumnet_v2 import BrainTumNetV2
from braintumnet.data.brats2020_dataset import SliceDataset
from torch.utils.data import DataLoader
# Import TTA functions from tta_inference.py in same directory
import importlib.util
spec = importlib.util.spec_from_file_location("tta_inference",
                                                os.path.join(os.path.dirname(__file__), "tta_inference.py"))
tta_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tta_module)
tta_predict_single = tta_module.tta_predict_single
compute_iou = tta_module.compute_iou


def load_models(checkpoint_paths, cfg, device='cuda'):
    """Load all fold models"""
    models = []
    model_type = cfg['model'].get('model_type', 'v1')

    print(f"\nLoading {len(checkpoint_paths)} models ({model_type.upper()}):")

    for i, ckpt_path in enumerate(checkpoint_paths):
        print(f"  [{i+1}/{len(checkpoint_paths)}] {Path(ckpt_path).name}")

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

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        models.append(model)

    print(f"All {len(models)} models loaded successfully!")
    return models


def ensemble_predict(models, image, device='cuda', use_tta=False):
    """
    Average predictions from all models

    Args:
        models: List of trained models
        image: (1, C, H, W) input tensor
        device: device
        use_tta: if True, apply TTA to each model (slow but best performance)

    Returns:
        pred_prob: (1, num_classes, H, W) averaged probabilities
    """
    predictions = []

    with torch.no_grad():
        for model in models:
            if use_tta:
                # TTA for each model (SLOW but best performance)
                pred_prob = tta_predict_single(model, image, device)
            else:
                # Standard inference
                image_dev = image.to(device)
                output = model(image_dev)
                seg = output[0] if isinstance(output, tuple) else output
                pred_prob = F.softmax(seg, dim=1)

            predictions.append(pred_prob)

    # Average across all models
    pred_ensemble = torch.stack(predictions).mean(dim=0)
    return pred_ensemble


def main():
    parser = argparse.ArgumentParser(
        description='5-Fold Ensemble Inference - just pass checkpoint glob pattern',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python braintumnet/scripts/ensemble_inference.py "checkpoints/braintumnet_best_fold*.pth"
  python braintumnet/scripts/ensemble_inference.py "checkpoints/braintumnet_best_fold*.pth" --use_tta
        """
    )
    parser.add_argument('checkpoints', type=str,
                       help='Checkpoint glob pattern (e.g., "checkpoints/braintumnet_best_fold*.pth")')
    parser.add_argument('--output', type=str, default='results/ensemble_results.csv',
                       help='Output CSV file (default: results/ensemble_results.csv)')
    parser.add_argument('--use_tta', action='store_true',
                       help='Use TTA for each model (SLOW but +2-3%% IoU)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()

    # Load base config
    ROOT = Path(__file__).resolve().parents[1]
    cfg_path = ROOT / "configs" / "base.yaml"
    print(f"Loading config from: {cfg_path}")
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Get data_root from config
    data_root = cfg['data']['proc_root']

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Find checkpoint files
    checkpoint_paths = sorted(glob.glob(args.checkpoints))
    if len(checkpoint_paths) == 0:
        raise ValueError(f"No checkpoints found matching: {args.checkpoints}")

    print(f"Found {len(checkpoint_paths)} checkpoint(s)")

    # Load all models
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    models = load_models(checkpoint_paths, cfg, device)

    # Load validation dataset (use fold 0 as default for ensemble)
    fold = 0
    split_file = os.path.join(data_root, f'val_fold{fold}.csv')
    print(f"\nUsing validation fold {fold}: {split_file}")

    dataset = SliceDataset(
        proc_root=data_root,
        split_file=split_file,
        img_size=cfg['data']['img_size'],
        train=False,
        in_channels=cfg['model']['in_channels']
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Validation set: {len(dataset)} samples")

    if args.use_tta:
        print(f"\n⚠️  TTA enabled: This will take ~{len(models)}×8 = {len(models)*8}x longer!")
        print(f"   Expected gain: +{len(models)+2}-{len(models)+4}% IoU vs single model")
    else:
        print(f"\nStandard ensemble (no TTA)")
        print(f"Expected gain: +{len(models)-1}-{len(models)+1}% IoU vs single model")

    print(f"\nStarting ensemble inference...")

    # Run inference
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Ensemble Inference")):
            img = batch['image']
            msk = batch['mask']

            # Ensemble prediction
            pred_prob = ensemble_predict(models, img, device, use_tta=args.use_tta)
            pred_class = pred_prob.argmax(dim=1).cpu()

            # Compute IoU
            target = msk.squeeze(1).cpu()
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
    method_name = f"{len(models)}-Fold Ensemble" + (" + TTA" if args.use_tta else "")

    print(f"\n{'='*70}")
    print(f"{method_name} Results")
    print(f"{'='*70}")
    print(f"Number of models: {len(models)}")
    print(f"TTA enabled: {args.use_tta}")
    print(f"\nIoU Results:")
    print(f"  TC IoU:   {df['tc_iou'].mean():.4f} ± {df['tc_iou'].std():.4f}")
    print(f"  ED IoU:   {df['ed_iou'].mean():.4f} ± {df['ed_iou'].std():.4f}")
    print(f"  Mean IoU: {df['mean_iou'].mean():.4f} ± {df['mean_iou'].std():.4f}")
    print(f"\nResults saved to: {args.output}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
