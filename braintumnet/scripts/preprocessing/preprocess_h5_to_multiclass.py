"""
Convert BraTS H5 format to 3-class PNG format for Multi-Class Segmentation
===========================================================================

Input: H5 files with mask shape (240, 240, 3) where each channel is binary
Output: PNG files with 3-class labels {0, 1, 2}

Mapping:
- Channel 0 → Unknown (ignore for now)
- Channel 1 → Tumor Core (TC) → class 1
- Channel 2 → Edema (ED) → class 2
- Background → class 0

Usage:
    python scripts/preprocessing/preprocess_h5_to_multiclass.py \
        --h5_dir E:\thong\code\brain_segmen\brats2020_data\bcs2020\archive\BraTS2020_training_data\content\data \
        --out_dir braintumnet\data\processed_multiclass \
        --img_size 256 \
        --num_folds 5

Author: BrainTumNet Multi-Class Extension
Date: 2025-10-10
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import KFold

# Add src to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from braintumnet.utils.seed import set_seed


def load_h5_data(h5_path):
    """Load H5 file and return image and mask.

    Args:
        h5_path: Path to H5 file

    Returns:
        image: (H, W, 4) numpy array - 4 modalities
        mask: (H, W, 3) numpy array - 3 binary channels
    """
    with h5py.File(h5_path, 'r') as f:
        image = f['image'][:]  # (H, W, 4)
        mask = f['mask'][:]    # (H, W, 3)
    return image, mask


def convert_mask_to_3class(mask_3ch):
    """Convert 3-channel binary mask to 3-class single-channel mask.

    Args:
        mask_3ch: (H, W, 3) binary mask where each channel is 0 or 1

    Returns:
        mask_3class: (H, W) uint8 with values {0, 1, 2}
            0 = Background
            1 = Tumor Core (TC) - from channel 1
            2 = Edema (ED) - from channel 2
    """
    H, W, C = mask_3ch.shape
    mask_3class = np.zeros((H, W), dtype=np.uint8)

    # Priority: TC > ED > Background
    # Channel 2 = Edema → class 2
    mask_3class[mask_3ch[:, :, 2] > 0] = 2

    # Channel 1 = Tumor Core → class 1 (overwrites edema if overlapping)
    mask_3class[mask_3ch[:, :, 1] > 0] = 1

    # Channel 0 is ignored (not used in BraTS standard regions)
    # Background remains 0

    return mask_3class


def normalize_image(image, modality_idx):
    """Normalize image to [0, 255] uint8.

    Args:
        image: (H, W) float array
        modality_idx: Modality index (0=FLAIR, 1=T1, 2=T1CE, 3=T2)

    Returns:
        normalized: (H, W) uint8 in [0, 255]
    """
    # Remove background
    brain_mask = image > 0

    if brain_mask.sum() == 0:
        return np.zeros_like(image, dtype=np.uint8)

    # Compute percentiles on brain region
    p1 = np.percentile(image[brain_mask], 1)
    p99 = np.percentile(image[brain_mask], 99)

    # Clip and normalize
    image_clipped = np.clip(image, p1, p99)
    image_norm = (image_clipped - p1) / (p99 - p1 + 1e-8)
    image_norm = (image_norm * 255).astype(np.uint8)

    return image_norm


def resize_array(arr, target_size=256, is_mask=False):
    """Resize 2D array to target size.

    Args:
        arr: (H, W) numpy array
        target_size: Target size (square)
        is_mask: If True, use nearest neighbor; else use bilinear

    Returns:
        resized: (target_size, target_size) array
    """
    if is_mask:
        img = Image.fromarray(arr, mode='L')
        img_resized = img.resize((target_size, target_size), Image.NEAREST)
    else:
        img = Image.fromarray(arr)
        img_resized = img.resize((target_size, target_size), Image.BILINEAR)

    return np.array(img_resized)


def process_h5_file(h5_path, out_dir, img_size=256):
    """Process a single H5 file and save PNG outputs.

    Args:
        h5_path: Path to H5 file
        out_dir: Output directory
        img_size: Target image size

    Returns:
        slice_info: Dict with metadata, or None if no tumor
    """
    # Extract slice_id from filename (e.g., volume_1_slice_50.h5)
    fname = Path(h5_path).stem  # volume_1_slice_50

    # Load data
    try:
        image, mask_3ch = load_h5_data(h5_path)
    except Exception as e:
        print(f"Error loading {h5_path}: {e}")
        return None

    # Convert mask to 3-class
    mask_3class = convert_mask_to_3class(mask_3ch)

    # Extract volume and slice ID
    # volume_1_slice_50 → vol1_slice50
    parts = fname.split('_')
    vol_id = f"vol{parts[1]}"
    slice_id = f"slice{parts[3]}"
    output_id = f"{vol_id}_{slice_id}"

    # Save each modality
    modality_names = ['flair', 't1', 't1ce', 't2']
    for mod_idx, mod_name in enumerate(modality_names):
        mod_dir = out_dir / mod_name
        mod_dir.mkdir(parents=True, exist_ok=True)

        # Normalize and resize
        img_2d = normalize_image(image[:, :, mod_idx], mod_idx)
        img_resized = resize_array(img_2d, img_size, is_mask=False)

        # Save as PNG
        save_path = mod_dir / f"{output_id}.png"
        Image.fromarray(img_resized).save(save_path)

    # Save segmentation mask
    seg_dir = out_dir / "seg"
    seg_dir.mkdir(parents=True, exist_ok=True)

    mask_resized = resize_array(mask_3class, img_size, is_mask=True)
    save_path = seg_dir / f"{output_id}.png"
    Image.fromarray(mask_resized, mode='L').save(save_path)

    # Compute statistics
    has_tc = (mask_resized == 1).any()
    has_ed = (mask_resized == 2).any()
    has_wt = has_tc or has_ed

    # Determine primary label
    if has_tc and has_ed:
        label = "WT"  # Whole tumor
    elif has_tc:
        label = "TC"  # Tumor core only
    elif has_ed:
        label = "ED"  # Edema only
    else:
        label = "Normal"

    # Create metadata
    slice_info = {
        'slice_id': output_id,
        'volume_id': vol_id,
        'slice_idx': int(parts[3]),
        'label': label,
        'has_wt': int(has_wt),
        'has_tc': int(has_tc),
        'has_ed': int(has_ed),
    }

    return slice_info


def create_kfold_splits(all_slices_df, num_folds=5, seed=42):
    """Create K-fold splits at volume level.

    Args:
        all_slices_df: DataFrame with all slices
        num_folds: Number of folds
        seed: Random seed

    Returns:
        splits: List of (train_indices, val_indices) tuples
    """
    # Get unique volumes
    volume_ids = all_slices_df['volume_id'].unique()

    # Create K-fold splitter
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    splits = []
    for train_vols, val_vols in kf.split(volume_ids):
        train_vol_ids = volume_ids[train_vols]
        val_vol_ids = volume_ids[val_vols]

        # Get slice indices
        train_indices = all_slices_df[all_slices_df['volume_id'].isin(train_vol_ids)].index.tolist()
        val_indices = all_slices_df[all_slices_df['volume_id'].isin(val_vol_ids)].index.tolist()

        splits.append((train_indices, val_indices))

    return splits


def main():
    parser = argparse.ArgumentParser(description="Convert H5 to Multi-Class PNG Format")
    parser.add_argument("--h5_dir", type=str, required=True,
                       help="Path to H5 data directory")
    parser.add_argument("--out_dir", type=str, default="braintumnet/data/processed_multiclass",
                       help="Output directory")
    parser.add_argument("--img_size", type=int, default=256,
                       help="Target image size (square)")
    parser.add_argument("--num_folds", type=int, default=5,
                       help="Number of K-fold splits")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Max files to process (for testing)")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all H5 files
    h5_dir = Path(args.h5_dir)
    if not h5_dir.exists():
        print(f"Error: H5 directory not found: {h5_dir}")
        sys.exit(1)

    h5_files = sorted([f for f in h5_dir.glob("*.h5")])

    if args.max_files:
        h5_files = h5_files[:args.max_files]

    print(f"Found {len(h5_files)} H5 files in {h5_dir}")

    # Process all files
    all_slices = []

    for h5_path in tqdm(h5_files, desc="Processing H5 files"):
        slice_info = process_h5_file(h5_path, out_dir, args.img_size)
        if slice_info is not None:
            all_slices.append(slice_info)

    # Create DataFrame
    df = pd.DataFrame(all_slices)

    # Save full dataset info
    df.to_csv(out_dir / "all_slices.csv", index=False)

    print(f"\n✓ Processed {len(df)} slices")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nTumor region statistics:")
    print(f"  Whole Tumor (WT): {df['has_wt'].sum():,} slices ({df['has_wt'].mean()*100:.1f}%)")
    print(f"  Tumor Core (TC):  {df['has_tc'].sum():,} slices ({df['has_tc'].mean()*100:.1f}%)")
    print(f"  Edema (ED):       {df['has_ed'].sum():,} slices ({df['has_ed'].mean()*100:.1f}%)")

    # Create K-fold splits
    splits = create_kfold_splits(df, num_folds=args.num_folds, seed=args.seed)

    # Save splits
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_df.to_csv(out_dir / f"train_fold{fold}.csv", index=False)
        val_df.to_csv(out_dir / f"val_fold{fold}.csv", index=False)

        print(f"\nFold {fold}:")
        print(f"  Train: {len(train_df):,} slices from {train_df['volume_id'].nunique()} volumes")
        print(f"  Val:   {len(val_df):,} slices from {val_df['volume_id'].nunique()} volumes")

    # Create label mapping file
    mapping = {
        "num_classes": 3,
        "class_names": ["Background", "TumorCore", "Edema"],
        "class_labels": [0, 1, 2],
        "regions": {
            "WT": "Whole Tumor = TC + ED (classes 1,2)",
            "TC": "Tumor Core = class 1",
            "ED": "Edema = class 2"
        },
        "h5_channel_mapping": {
            "channel_0": "Unused",
            "channel_1": "Tumor Core → class 1",
            "channel_2": "Edema → class 2"
        }
    }

    import json
    with open(out_dir / "class_mapping.json", 'w') as f:
        json.dump(mapping, f, indent=2)

    # Create labels.csv (case-level labels)
    volume_labels = []
    for vol_id in sorted(df['volume_id'].unique()):
        # In BraTS all cases have tumors, so label is always 0
        # (This matches the original format where all cases were labeled 0)
        volume_labels.append({'case_id': vol_id, 'label': 0})

    labels_df = pd.DataFrame(volume_labels)
    labels_df.to_csv(out_dir / "labels.csv", index=False)

    # Create mapping.csv (slice-to-case mapping)
    mapping_df = df[['slice_id', 'volume_id']].copy()
    mapping_df = mapping_df.rename(columns={'volume_id': 'case_id'})
    mapping_df = mapping_df.sort_values('slice_id')
    mapping_df.to_csv(out_dir / "mapping.csv", index=False)

    print(f"\n✓ Preprocessing complete!")
    print(f"✓ Data saved to: {out_dir}")
    print(f"✓ Class mapping saved to: {out_dir / 'class_mapping.json'}")
    print(f"\nNext steps:")
    print(f"1. Train multi-class model:")
    print(f"   python scripts/train.py --cfg configs/multiclass.yaml --fold 0")
    print(f"2. Or use A100 optimized:")
    print(f"   python scripts/train.py --cfg configs/multiclass_a100.yaml --fold 0")


if __name__ == "__main__":
    main()
