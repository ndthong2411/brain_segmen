"""
BraTS Multi-Class Data Preprocessing Script
============================================

Converts BraTS 2020 NIfTI files to 3-class PNG format for multi-class segmentation.

Original BraTS labels:
- 0: Background
- 1: NCR (Necrotic and Non-Enhancing Tumor)
- 2: ED (Peritumoral Edema)
- 4: ET (Enhancing Tumor)

Target 3-class mapping:
- 0: Background
- 1: Tumor Core (TC) = NCR + ET (original labels 1 + 4)
- 2: Edema (ED) = Peritumoral Edema (original label 2)

Tumor regions for evaluation:
- Whole Tumor (WT) = TC + ED (classes 1 + 2)
- Tumor Core (TC) = class 1 only
- Edema (ED) = class 2 only

Usage:
    python scripts/preprocess_multiclass.py \
        --raw_dir data/raw/BraTS2020 \
        --out_dir data/processed_multiclass \
        --img_size 256 \
        --slices_per_case 30 \
        --tumor_ratio 0.5 \
        --num_folds 5

Author: BrainTumNet Multi-Class Extension
Date: 2025-10-10
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import KFold

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from braintumnet.utils.seed import set_seed


def load_nifti(path):
    """Load NIfTI file and return numpy array."""
    return nib.load(str(path)).get_fdata()


def remap_labels_to_3class(seg_volume):
    """
    Remap BraTS labels {0, 1, 2, 4} to 3-class {0, 1, 2}.

    Args:
        seg_volume: (H, W, D) numpy array with original BraTS labels

    Returns:
        remapped: (H, W, D) numpy array with 3-class labels
    """
    remapped = np.zeros_like(seg_volume, dtype=np.uint8)

    # Background: 0 → 0
    remapped[seg_volume == 0] = 0

    # Tumor Core (TC) = NCR + ET: 1,4 → 1
    remapped[(seg_volume == 1) | (seg_volume == 4)] = 1

    # Edema (ED): 2 → 2
    remapped[seg_volume == 2] = 2

    return remapped


def normalize_modality(volume, modality):
    """
    Normalize MRI modality to [0, 255] range.

    Args:
        volume: (H, W, D) numpy array
        modality: str, one of ['flair', 't1', 't1ce', 't2']

    Returns:
        normalized: (H, W, D) uint8 array in [0, 255]
    """
    # Remove background (assumes background ~ 0)
    brain_mask = volume > 0

    if brain_mask.sum() == 0:
        return np.zeros_like(volume, dtype=np.uint8)

    # Compute percentiles on brain region only
    p1 = np.percentile(volume[brain_mask], 1)
    p99 = np.percentile(volume[brain_mask], 99)

    # Clip and normalize
    volume_clipped = np.clip(volume, p1, p99)
    volume_norm = (volume_clipped - p1) / (p99 - p1 + 1e-8)
    volume_norm = (volume_norm * 255).astype(np.uint8)

    return volume_norm


def select_slices(seg_volume, slices_per_case=30, tumor_ratio=0.5):
    """
    Select slices with balanced tumor/non-tumor distribution.

    Args:
        seg_volume: (H, W, D) segmentation volume with 3-class labels
        slices_per_case: Number of slices to select per case
        tumor_ratio: Ratio of tumor slices to select

    Returns:
        selected_indices: List of slice indices to use
    """
    D = seg_volume.shape[2]

    # Find slices with tumor (any non-zero label)
    tumor_slices = []
    non_tumor_slices = []

    for z in range(D):
        if seg_volume[:, :, z].max() > 0:
            tumor_slices.append(z)
        else:
            non_tumor_slices.append(z)

    # Calculate number of tumor and non-tumor slices to select
    n_tumor = int(slices_per_case * tumor_ratio)
    n_non_tumor = slices_per_case - n_tumor

    # Sample slices
    selected = []

    if len(tumor_slices) >= n_tumor:
        # Evenly sample from tumor slices
        indices = np.linspace(0, len(tumor_slices)-1, n_tumor, dtype=int)
        selected.extend([tumor_slices[i] for i in indices])
    else:
        # Use all tumor slices
        selected.extend(tumor_slices)
        n_non_tumor += (n_tumor - len(tumor_slices))

    if len(non_tumor_slices) >= n_non_tumor:
        # Evenly sample from non-tumor slices
        indices = np.linspace(0, len(non_tumor_slices)-1, n_non_tumor, dtype=int)
        selected.extend([non_tumor_slices[i] for i in indices])
    else:
        # Use all non-tumor slices
        selected.extend(non_tumor_slices)

    return sorted(selected)


def resize_slice(slice_2d, target_size=256):
    """
    Resize 2D slice to target size.

    Args:
        slice_2d: (H, W) numpy array
        target_size: Target image size (assumes square)

    Returns:
        resized: (target_size, target_size) numpy array
    """
    # For segmentation masks, use nearest neighbor to preserve labels
    # For images, use bilinear
    if slice_2d.dtype == np.uint8 and slice_2d.max() <= 2:
        # Segmentation mask
        img = Image.fromarray(slice_2d, mode='L')
        img_resized = img.resize((target_size, target_size), Image.NEAREST)
    else:
        # Image
        img = Image.fromarray(slice_2d)
        img_resized = img.resize((target_size, target_size), Image.BILINEAR)

    return np.array(img_resized)


def process_case(case_dir, out_dir, case_id, img_size=256,
                 slices_per_case=30, tumor_ratio=0.5):
    """
    Process a single BraTS case.

    Args:
        case_dir: Path to case directory (contains .nii.gz files)
        out_dir: Output directory
        case_id: Case identifier (e.g., 'BraTS20_Training_001')
        img_size: Target image size
        slices_per_case: Number of slices to extract
        tumor_ratio: Ratio of tumor slices

    Returns:
        slice_info: List of dicts with slice metadata
    """
    # Load all modalities
    modalities = ['flair', 't1', 't1ce', 't2']
    volumes = {}

    for mod in modalities:
        mod_path = case_dir / f"{case_id}_{mod}.nii.gz"
        if not mod_path.exists():
            print(f"Warning: {mod_path} not found, skipping case {case_id}")
            return []
        volumes[mod] = load_nifti(mod_path)

    # Load segmentation
    seg_path = case_dir / f"{case_id}_seg.nii.gz"
    if not seg_path.exists():
        print(f"Warning: {seg_path} not found, skipping case {case_id}")
        return []

    seg_volume = load_nifti(seg_path)

    # Remap labels to 3-class
    seg_volume = remap_labels_to_3class(seg_volume)

    # Normalize modalities
    for mod in modalities:
        volumes[mod] = normalize_modality(volumes[mod], mod)

    # Select slices
    selected_slices = select_slices(seg_volume, slices_per_case, tumor_ratio)

    # Create output directories
    for mod in modalities:
        (out_dir / mod).mkdir(parents=True, exist_ok=True)
    (out_dir / "seg").mkdir(parents=True, exist_ok=True)

    # Process and save slices
    slice_info = []

    for z in selected_slices:
        slice_id = f"{case_id}_slice{z:03d}"

        # Save each modality
        for mod in modalities:
            slice_2d = volumes[mod][:, :, z]
            slice_resized = resize_slice(slice_2d, img_size)

            save_path = out_dir / mod / f"{slice_id}.png"
            Image.fromarray(slice_resized).save(save_path)

        # Save segmentation mask
        seg_slice = seg_volume[:, :, z]
        seg_resized = resize_slice(seg_slice, img_size)

        save_path = out_dir / "seg" / f"{slice_id}.png"
        Image.fromarray(seg_resized, mode='L').save(save_path)

        # Compute label statistics
        has_tc = (seg_resized == 1).any()
        has_ed = (seg_resized == 2).any()
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

        slice_info.append({
            'slice_id': slice_id,
            'case_id': case_id,
            'slice_idx': z,
            'label': label,
            'has_wt': int(has_wt),
            'has_tc': int(has_tc),
            'has_ed': int(has_ed),
        })

    return slice_info


def create_kfold_splits(all_slices_df, num_folds=5, seed=42):
    """
    Create K-fold splits at the case level (not slice level).

    Args:
        all_slices_df: DataFrame with all slices
        num_folds: Number of folds
        seed: Random seed

    Returns:
        splits: List of (train_indices, val_indices) tuples
    """
    # Get unique cases
    case_ids = all_slices_df['case_id'].unique()

    # Create K-fold splitter
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    splits = []
    for train_cases, val_cases in kf.split(case_ids):
        train_case_ids = case_ids[train_cases]
        val_case_ids = case_ids[val_cases]

        # Get slice indices
        train_indices = all_slices_df[all_slices_df['case_id'].isin(train_case_ids)].index.tolist()
        val_indices = all_slices_df[all_slices_df['case_id'].isin(val_case_ids)].index.tolist()

        splits.append((train_indices, val_indices))

    return splits


def main():
    parser = argparse.ArgumentParser(description="BraTS Multi-Class Data Preprocessing")
    parser.add_argument("--raw_dir", type=str, required=True,
                       help="Path to BraTS raw data directory")
    parser.add_argument("--out_dir", type=str, default="data/processed_multiclass",
                       help="Output directory for processed data")
    parser.add_argument("--img_size", type=int, default=256,
                       help="Target image size (square)")
    parser.add_argument("--slices_per_case", type=int, default=30,
                       help="Number of slices to extract per case")
    parser.add_argument("--tumor_ratio", type=float, default=0.5,
                       help="Ratio of tumor slices to extract")
    parser.add_argument("--num_folds", type=int, default=5,
                       help="Number of K-fold splits")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all cases
    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print(f"Error: Raw data directory not found: {raw_dir}")
        sys.exit(1)

    case_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    print(f"Found {len(case_dirs)} cases in {raw_dir}")

    # Process all cases
    all_slices = []

    for case_dir in tqdm(case_dirs, desc="Processing cases"):
        case_id = case_dir.name
        slice_info = process_case(
            case_dir, out_dir, case_id,
            img_size=args.img_size,
            slices_per_case=args.slices_per_case,
            tumor_ratio=args.tumor_ratio
        )
        all_slices.extend(slice_info)

    # Create DataFrame
    df = pd.DataFrame(all_slices)

    # Save full dataset info
    df.to_csv(out_dir / "all_slices.csv", index=False)
    print(f"\nTotal slices: {len(df)}")
    print(f"Label distribution:")
    print(df['label'].value_counts())
    print(f"\nTumor region statistics:")
    print(f"  Whole Tumor (WT): {df['has_wt'].sum()} slices ({df['has_wt'].mean()*100:.1f}%)")
    print(f"  Tumor Core (TC):  {df['has_tc'].sum()} slices ({df['has_tc'].mean()*100:.1f}%)")
    print(f"  Edema (ED):       {df['has_ed'].sum()} slices ({df['has_ed'].mean()*100:.1f}%)")

    # Create K-fold splits
    splits = create_kfold_splits(df, num_folds=args.num_folds, seed=args.seed)

    # Save splits
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_df.to_csv(out_dir / f"train_fold{fold}.csv", index=False)
        val_df.to_csv(out_dir / f"val_fold{fold}.csv", index=False)

        print(f"\nFold {fold}:")
        print(f"  Train: {len(train_df)} slices from {train_df['case_id'].nunique()} cases")
        print(f"  Val:   {len(val_df)} slices from {val_df['case_id'].nunique()} cases")

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
        "original_brats_mapping": {
            0: "Background → 0",
            1: "NCR (Necrotic) → 1 (TC)",
            2: "ED (Edema) → 2",
            4: "ET (Enhancing) → 1 (TC)"
        }
    }

    import json
    with open(out_dir / "class_mapping.json", 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"\n✓ Preprocessing complete!")
    print(f"✓ Data saved to: {out_dir}")
    print(f"✓ Class mapping saved to: {out_dir / 'class_mapping.json'}")
    print(f"\nNext steps:")
    print(f"1. Update config to use multi-class:")
    print(f"   data:")
    print(f"     proc_root: \"{args.out_dir}\"")
    print(f"   model:")
    print(f"     num_classes_seg: 3")
    print(f"2. Train with multi-class losses:")
    print(f"   python scripts/train.py --cfg configs/multiclass.yaml --fold 0")


if __name__ == "__main__":
    main()
