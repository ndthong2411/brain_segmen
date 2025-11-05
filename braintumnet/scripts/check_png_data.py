"""
Check PNG data quality before LMDB conversion
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
import os

def check_png_data():
    """Check PNG data quality"""

    proc_root = "braintumnet/data/processed_multiclass_full"
    fold = 3
    train_csv = os.path.join(proc_root, f"train_fold{fold}.csv")

    print("="*70)
    print("PNG Data Quality Check")
    print("="*70)
    print(f"Data root: {proc_root}")
    print(f"Train CSV: {train_csv}")

    if not os.path.exists(train_csv):
        print(f"✗ Train CSV not found: {train_csv}")
        return

    # Read CSV
    with open(train_csv, 'r') as f:
        lines = f.readlines()

    print(f"✓ Found {len(lines)} samples in CSV")

    # Check first 10 samples
    print("\n" + "="*70)
    print("Checking first 10 samples...")
    print("="*70)

    for i, line in enumerate(lines[:10]):
        parts = line.strip().split(',')
        if len(parts) < 3:
            print(f"\n✗ Sample {i}: Invalid CSV format")
            continue

        img_path = parts[0]
        mask_path = parts[1]
        label = int(parts[2])

        print(f"\nSample {i}:")
        print(f"  Image: {img_path}")
        print(f"  Mask: {mask_path}")
        print(f"  Label: {label}")

        # Check if files exist
        if not os.path.exists(img_path):
            print(f"  ✗ Image file not found!")
            continue
        if not os.path.exists(mask_path):
            print(f"  ✗ Mask file not found!")
            continue

        # Load and check image
        try:
            # Image has 4 channels saved as separate files
            img_base = img_path.replace('_flair.png', '')
            channels = ['flair', 't1', 't1ce', 't2']

            img_data = []
            for ch in channels:
                ch_path = f"{img_base}_{ch}.png"
                if not os.path.exists(ch_path):
                    print(f"  ✗ Channel {ch} not found: {ch_path}")
                    break
                ch_img = np.array(Image.open(ch_path))
                img_data.append(ch_img)

            if len(img_data) == 4:
                img_array = np.stack(img_data, axis=0)  # (4, H, W)
                print(f"  ✓ Image shape: {img_array.shape}, dtype: {img_array.dtype}")
                print(f"    Range: [{img_array.min()}, {img_array.max()}]")

                # Check if image is all zeros
                if img_array.max() == 0:
                    print(f"  ⚠ WARNING: Image is all zeros!")

            # Load and check mask
            mask_img = Image.open(mask_path)
            mask_array = np.array(mask_img)

            unique_classes = np.unique(mask_array)
            class_counts = {int(c): int(np.sum(mask_array == c)) for c in unique_classes}
            total = mask_array.size

            print(f"  ✓ Mask shape: {mask_array.shape}, dtype: {mask_array.dtype}")
            print(f"    Unique classes: {unique_classes}")
            for c, count in class_counts.items():
                pct = (count / total) * 100
                print(f"      Class {c}: {count:6d} pixels ({pct:5.2f}%)")

            if len(unique_classes) == 1 and unique_classes[0] == 0:
                print(f"  ⚠ WARNING: Mask is all background!")

        except Exception as e:
            print(f"  ✗ Error loading files: {e}")

    # Statistics over more samples
    print("\n" + "="*70)
    print("Statistics over first 100 samples...")
    print("="*70)

    all_bg_count = 0
    all_zero_img_count = 0
    class_pixel_counts = {0: 0, 1: 0, 2: 0}

    for i, line in enumerate(lines[:100]):
        try:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue

            mask_path = parts[1]
            if not os.path.exists(mask_path):
                continue

            mask_array = np.array(Image.open(mask_path))
            unique_classes = np.unique(mask_array)

            if len(unique_classes) == 1 and unique_classes[0] == 0:
                all_bg_count += 1

            for c in range(3):
                class_pixel_counts[c] += int(np.sum(mask_array == c))

            # Check image
            img_path = parts[0]
            img_base = img_path.replace('_flair.png', '')
            flair_path = f"{img_base}_flair.png"
            if os.path.exists(flair_path):
                flair_img = np.array(Image.open(flair_path))
                if flair_img.max() == 0:
                    all_zero_img_count += 1

        except Exception as e:
            print(f"  Error at sample {i}: {e}")

    print(f"\nSamples with only background: {all_bg_count} / 100")
    print(f"Samples with all-zero images: {all_zero_img_count} / 100")
    print(f"Overall class distribution:")
    total_pixels = sum(class_pixel_counts.values())
    for c in sorted(class_pixel_counts.keys()):
        pct = (class_pixel_counts[c] / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"  Class {c}: {class_pixel_counts[c]:10d} pixels ({pct:5.2f}%)")

    print("\n" + "="*70)
    print("✓ PNG data check complete!")
    print("="*70)

if __name__ == "__main__":
    check_png_data()
