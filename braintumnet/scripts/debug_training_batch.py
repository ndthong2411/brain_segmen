"""
Debug script to test one training batch
"""
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np
from braintumnet.utils.io import load_yaml
from braintumnet.models import build_model
from braintumnet.data.dataset_factory import create_dataset
from torch.utils.data import DataLoader
from braintumnet.multiclass_metrics import MulticlassMetricsAccumulator

def merge_configs(base_cfg, override_cfg):
    """Deep merge two configs"""
    import copy
    merged = copy.deepcopy(base_cfg)
    for key, value in override_cfg.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

def test_training_batch():
    """Test one training batch end-to-end"""

    print("="*70)
    print("Training Batch Debug Test")
    print("="*70)

    # Load configs
    configs_dir = ROOT / "configs"
    base_cfg = load_yaml(str(configs_dir / "base.yaml"))
    model_cfg = load_yaml(str(configs_dir / "models" / "segunetv2_phase2.yaml"))

    cfg = merge_configs(base_cfg, model_cfg)
    cfg['model']['model_type'] = 'v2'
    cfg['data']['fold'] = 3

    print(f"\nConfig:")
    print(f"  Model: {cfg['model']['model_type']}")
    print(f"  num_classes_seg: {cfg['model']['num_classes_seg']}")
    print(f"  Backend: {cfg['data']['backend']}")
    print(f"  Fold: {cfg['data']['fold']}")

    # Build dataset
    print("\nBuilding dataset...")
    # Get data root based on backend
    backend = cfg['data']['backend']
    if backend == 'lmdb':
        data_root = cfg['data']['lmdb_root']
    elif backend == 'png':
        data_root = cfg['data']['proc_root']
    else:
        data_root = cfg['data'].get('data_root', cfg['data']['proc_root'])

    train_list = os.path.join(data_root, f"train_fold{cfg['data']['fold']}.csv")

    try:
        # Create dataset with NO augmentation for debugging
        cfg_no_aug = cfg.copy()
        cfg_no_aug['augment'] = {
            'rotate_deg': 0, 'hflip_p': 0.0, 'vflip_p': 0.0,
            'scale_range': [1.0, 1.0], 'brightness_range': [1.0, 1.0],
            'contrast_range': [1.0, 1.0], 'gamma_range': [1.0, 1.0],
            'gaussian_noise_p': 0.0, 'elastic_deform_p': 0.0,
            'bias_field_p': 0.0, 'gaussian_blur_p': 0.0,
            'gamma_p': 0.0, 'cutout_p': 0.0, 'local_shuffle_p': 0.0
        }

        dataset = create_dataset(cfg['data']['backend'], data_root, train_list, cfg_no_aug, train=False)
        print(f"✓ Dataset created: {len(dataset)} samples")

        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        print(f"✓ DataLoader created")
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Build model
    print("\nBuilding model...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = build_model(cfg).to(device)
        print(f"✓ Model built, device: {device}")
    except Exception as e:
        print(f"✗ Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Get first batch
    print("\n" + "="*70)
    print("Testing first batch...")
    print("="*70)

    model.eval()
    batch = next(iter(loader))

    img = batch['image'].to(device)
    mask = batch['mask'].to(device)
    label = batch['label'].to(device)

    print(f"\nBatch shapes:")
    print(f"  Image: {img.shape}, dtype: {img.dtype}")
    print(f"  Mask: {mask.shape}, dtype: {mask.dtype}")
    print(f"  Label: {label.shape}, dtype: {label.dtype}")

    # Check mask content
    print(f"\nMask statistics:")
    mask_np = mask.cpu().numpy()
    for b in range(mask.shape[0]):
        mask_sample = mask_np[b, 0]  # (H, W)
        unique, counts = np.unique(mask_sample, return_counts=True)
        print(f"  Sample {b}:")
        print(f"    Unique classes: {unique}")
        total = mask_sample.size
        for u, c in zip(unique, counts):
            pct = (c / total) * 100
            print(f"      Class {u}: {c:6d} pixels ({pct:5.2f}%)")

    # Forward pass
    print(f"\nForward pass...")
    with torch.no_grad():
        try:
            output = model(img)

            if isinstance(output, tuple) and len(output) >= 2:
                seg = output[0]
                cls = output[1]

                print(f"✓ Model output:")
                print(f"  Segmentation: {seg.shape}")
                print(f"  Classification: {cls.shape}")

                # Check segmentation predictions
                print(f"\nSegmentation predictions:")
                seg_pred = torch.argmax(seg, dim=1)  # (B, H, W)

                for b in range(seg_pred.shape[0]):
                    pred_np = seg_pred[b].cpu().numpy()
                    unique, counts = np.unique(pred_np, return_counts=True)
                    print(f"  Sample {b}:")
                    total = pred_np.size
                    for u, c in zip(unique, counts):
                        pct = (c / total) * 100
                        print(f"    Class {u}: {c:6d} pixels ({pct:5.2f}%)")

                # Compute metrics
                print(f"\nComputing metrics...")
                metrics_acc = MulticlassMetricsAccumulator(num_classes=3)
                metrics_acc.update(seg, mask)
                metrics = metrics_acc.get_metrics()

                print(f"✓ Metrics computed:")
                print(f"  WT Dice: {metrics['WT_dice']:.4f}, IoU: {metrics['WT_iou']:.4f}")
                print(f"  TC Dice: {metrics['TC_dice']:.4f}, IoU: {metrics['TC_iou']:.4f}")
                print(f"  ED Dice: {metrics['ED_dice']:.4f}, IoU: {metrics['ED_iou']:.4f}")
                print(f"  Mean Dice: {metrics['mean_dice']:.4f}, IoU: {metrics['mean_iou']:.4f}")

                # Check if metrics are all zeros
                if metrics['mean_dice'] == 0.0:
                    print("\n⚠ WARNING: All metrics are ZERO!")
                    print("This suggests:")
                    print("  1. Model is predicting all background (class 0)")
                    print("  2. OR mask labels are all background")
                    print("  3. OR there's no overlap between predictions and ground truth")

        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return

    print("\n" + "="*70)
    print("✓ Training batch test complete!")
    print("="*70)

if __name__ == "__main__":
    test_training_batch()
