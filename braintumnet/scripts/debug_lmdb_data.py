"""
Debug script to check LMDB data and labels
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np
from braintumnet.data.dataset_factory import create_dataset

def check_lmdb_data():
    """Check LMDB data for corruption or wrong labels"""

    # Config for LMDB dataset
    cfg = {
        'data': {
            'backend': 'lmdb',
            'data_root': 'braintumnet/data/lmdb_processed_multiclass_full',
            'img_size': 256,
            'in_channels': 4  # Added missing in_channels
        },
        'model': {
            'num_classes_seg': 3,
            'in_channels': 4  # Added missing in_channels
        },
        'augment': {
            'rotate_deg': 0,
            'hflip_p': 0.0,
            'vflip_p': 0.0,
            'scale_range': [1.0, 1.0],
            'brightness_range': [1.0, 1.0],
            'contrast_range': [1.0, 1.0],
            'gamma_range': [1.0, 1.0],
            'gaussian_noise_p': 0.0,
            'elastic_deform_p': 0.0,
            'bias_field_p': 0.0,
            'gaussian_blur_p': 0.0,
            'gamma_p': 0.0,
            'cutout_p': 0.0,
            'local_shuffle_p': 0.0
        }
    }

    data_root = cfg['data']['data_root']
    train_list = f"{data_root}/train_fold3.csv"

    print("="*70)
    print("LMDB Data Debug Check")
    print("="*70)
    print(f"Data root: {data_root}")
    print(f"Train list: {train_list}")

    # Create dataset (no augmentation for debugging)
    try:
        dataset = create_dataset('lmdb', data_root, train_list, cfg, train=False)
        print(f"✓ Dataset created successfully")
        print(f"  Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        return

    # Check first 5 samples
    print("\n" + "="*70)
    print("Checking first 5 samples...")
    print("="*70)

    for i in range(min(5, len(dataset))):
        try:
            batch = dataset[i]
            img = batch['image']  # (C, H, W)
            mask = batch['mask']  # (1, H, W)
            label = batch['label']  # scalar

            # Convert to numpy for analysis
            mask_np = mask.squeeze().numpy()

            # Count unique classes in mask
            unique_classes = np.unique(mask_np)
            class_counts = {int(c): int(np.sum(mask_np == c)) for c in unique_classes}

            # Compute percentages
            total_pixels = mask_np.size
            class_percentages = {c: (count / total_pixels) * 100 for c, count in class_counts.items()}

            print(f"\nSample {i}:")
            print(f"  Image shape: {img.shape}, dtype: {img.dtype}, range: [{img.min():.3f}, {img.max():.3f}]")
            print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"  Label: {label}")
            print(f"  Unique classes in mask: {unique_classes}")
            print(f"  Class distribution:")
            for c in sorted(class_counts.keys()):
                print(f"    Class {c}: {class_counts[c]:6d} pixels ({class_percentages[c]:5.2f}%)")

            # Check for issues
            if len(unique_classes) == 1 and unique_classes[0] == 0:
                print(f"  ⚠ WARNING: Mask is all background (class 0)!")

            if mask_np.min() < 0 or mask_np.max() > 2:
                print(f"  ✗ ERROR: Invalid mask values! Expected [0, 1, 2], got [{mask_np.min()}, {mask_np.max()}]")

        except Exception as e:
            print(f"\n✗ Error loading sample {i}: {e}")
            import traceback
            traceback.print_exc()

    # Statistics over entire validation set
    print("\n" + "="*70)
    print("Statistics over entire dataset (first 100 samples)...")
    print("="*70)

    all_bg_count = 0
    class_pixel_counts = {0: 0, 1: 0, 2: 0}

    for i in range(min(100, len(dataset))):
        try:
            batch = dataset[i]
            mask_np = batch['mask'].squeeze().numpy()
            unique_classes = np.unique(mask_np)

            if len(unique_classes) == 1 and unique_classes[0] == 0:
                all_bg_count += 1

            for c in range(3):
                class_pixel_counts[c] += int(np.sum(mask_np == c))

        except Exception as e:
            print(f"Error at sample {i}: {e}")

    print(f"\nSamples with only background: {all_bg_count} / 100")
    print(f"Overall class distribution (first 100 samples):")
    total_pixels_all = sum(class_pixel_counts.values())
    for c in sorted(class_pixel_counts.keys()):
        pct = (class_pixel_counts[c] / total_pixels_all) * 100 if total_pixels_all > 0 else 0
        print(f"  Class {c}: {class_pixel_counts[c]:10d} pixels ({pct:5.2f}%)")

    print("\n" + "="*70)
    print("✓ LMDB data check complete!")
    print("="*70)

if __name__ == "__main__":
    check_lmdb_data()
