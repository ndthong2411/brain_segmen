"""
Convert BraTS NIfTI format to Multi-Class PNG format
=====================================================

Input: NIfTI files (.nii) with 4 modalities + segmentation
Output: PNG files with 3-class labels {0, 1, 2}

Segmentation mapping (BraTS 3-class):
- Label 0: Background → class 0
- Label 1: Necrotic/Non-enhancing tumor (NCR/NET) → class 1 (Tumor Core)
- Label 2: Edema → class 2
- Label 4: Enhancing tumor (ET) → class 1 (Tumor Core)

For standard BraTS regions:
- Whole Tumor (WT) = class 1 + class 2 (all non-zero)
- Tumor Core (TC) = class 1 (labels 1,4)
- Enhancing Tumor (ET) = label 4 subset of class 1

Usage:
    python braintumnet/scripts/preprocess_nifti_to_multiclass.py \
        --nifti_dir braintumnet/data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
        --out_dir braintumnet/data/processed_multiclass \
        --img_size 256 \
        --slices_per_case 30 \
        --num_folds 5

Author: BrainTumNet NIfTI Preprocessing
Date: 2025-10-29
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


def load_nifti_volume(case_dir, modality):
    """Load NIfTI volume for a specific modality.

    Args:
        case_dir: Path to case directory
        modality: Modality name (flair, t1, t1ce, t2, seg)

    Returns:
        volume: (H, W, D) numpy array
    """
    case_name = Path(case_dir).name
    nii_file = case_dir / f"{case_name}_{modality}.nii"

    if not nii_file.exists():
        raise FileNotFoundError(f"Missing {modality} file: {nii_file}")

    volume = nib.load(str(nii_file)).get_fdata()
    return volume


def convert_brats_seg_to_3class(seg_slice):
    """Convert BraTS segmentation to 3-class format.

    BraTS labels:
        0: Background
        1: Necrotic/Non-enhancing tumor (NCR/NET)
        2: Edema
        4: Enhancing tumor (ET)

    Our 3-class mapping:
        0: Background
        1: Tumor Core (NCR/NET + ET) = labels 1,4
        2: Edema = label 2

    Args:
        seg_slice: (H, W) with values {0, 1, 2, 4}

    Returns:
        mask_3class: (H, W) with values {0, 1, 2}
    """
    mask_3class = np.zeros_like(seg_slice, dtype=np.uint8)

    # Edema → class 2
    mask_3class[seg_slice == 2] = 2

    # Tumor Core (NCR/NET + ET) → class 1
    mask_3class[(seg_slice == 1) | (seg_slice == 4)] = 1

    return mask_3class


def normalize_slice(img_slice):
    """Normalize image slice to [0, 255] uint8.

    Args:
        img_slice: (H, W) float array

    Returns:
        normalized: (H, W) uint8 in [0, 255]
    """
    # Brain mask
    brain_mask = img_slice > 0

    if brain_mask.sum() == 0:
        return np.zeros_like(img_slice, dtype=np.uint8)

    # Percentile normalization on brain region
    p1 = np.percentile(img_slice[brain_mask], 1)
    p99 = np.percentile(img_slice[brain_mask], 99)

    # Clip and normalize
    img_clipped = np.clip(img_slice, p1, p99)
    img_norm = (img_clipped - p1) / (p99 - p1 + 1e-8)
    img_norm = (img_norm * 255).astype(np.uint8)

    return img_norm


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


def select_slices_with_tumor(seg_volume, slices_per_case=None, tumor_ratio=0.7):
    """Select slices from volume.

    Args:
        seg_volume: (H, W, D) segmentation volume
        slices_per_case: Number of slices to select per case. If None, use ALL slices.
        tumor_ratio: Ratio of tumor slices vs non-tumor slices (only used if slices_per_case is set)

    Returns:
        selected_indices: List of slice indices
    """
    total_slices = seg_volume.shape[2]

    # If slices_per_case is None or >= total slices, return ALL slices
    if slices_per_case is None or slices_per_case >= total_slices:
        return list(range(total_slices))

    # Otherwise, sample slices with tumor ratio
    tumor_slices = []
    non_tumor_slices = []

    for z in range(total_slices):
        if seg_volume[:, :, z].sum() > 0:
            tumor_slices.append(z)
        else:
            non_tumor_slices.append(z)

    # Calculate how many of each type
    n_tumor = min(len(tumor_slices), int(slices_per_case * tumor_ratio))
    n_non_tumor = slices_per_case - n_tumor

    # Sample tumor slices uniformly
    if len(tumor_slices) > 0 and n_tumor > 0:
        indices_tumor = np.linspace(tumor_slices[0], tumor_slices[-1], n_tumor).astype(int).tolist()
    else:
        indices_tumor = []

    # Sample non-tumor slices uniformly
    if len(non_tumor_slices) > 0 and n_non_tumor > 0:
        step = max(1, len(non_tumor_slices) // n_non_tumor)
        indices_non_tumor = non_tumor_slices[::step][:n_non_tumor]
    else:
        indices_non_tumor = []

    # Combine and sort
    selected = sorted(set(indices_tumor + indices_non_tumor))

    return selected


def process_case(case_dir, out_dir, img_size=256, slices_per_case=None):
    """Process a single BraTS case.

    Args:
        case_dir: Path to case directory
        out_dir: Output directory
        img_size: Target image size
        slices_per_case: Number of slices per case. If None, use ALL slices (155).

    Returns:
        slice_infos: List of slice metadata dicts
    """
    case_id = case_dir.name

    try:
        # Load all modalities
        flair = load_nifti_volume(case_dir, 'flair')
        t1 = load_nifti_volume(case_dir, 't1')
        t1ce = load_nifti_volume(case_dir, 't1ce')
        t2 = load_nifti_volume(case_dir, 't2')
        seg = load_nifti_volume(case_dir, 'seg')

    except FileNotFoundError as e:
        print(f"Skip {case_id}: {e}")
        return []

    # Select slices
    selected_slices = select_slices_with_tumor(seg, slices_per_case)

    if len(selected_slices) == 0:
        print(f"Skip {case_id}: No valid slices")
        return []

    slice_infos = []

    # Process each selected slice
    for z in selected_slices:
        slice_id = f"{case_id}_slice{z:03d}"

        # Extract slices
        flair_slice = flair[:, :, z]
        t1_slice = t1[:, :, z]
        t1ce_slice = t1ce[:, :, z]
        t2_slice = t2[:, :, z]
        seg_slice = seg[:, :, z]

        # Normalize images
        flair_norm = normalize_slice(flair_slice)
        t1_norm = normalize_slice(t1_slice)
        t1ce_norm = normalize_slice(t1ce_slice)
        t2_norm = normalize_slice(t2_slice)

        # Convert segmentation to 3-class
        seg_3class = convert_brats_seg_to_3class(seg_slice)

        # Resize all
        flair_resized = resize_array(flair_norm, img_size, is_mask=False)
        t1_resized = resize_array(t1_norm, img_size, is_mask=False)
        t1ce_resized = resize_array(t1ce_norm, img_size, is_mask=False)
        t2_resized = resize_array(t2_norm, img_size, is_mask=False)
        seg_resized = resize_array(seg_3class, img_size, is_mask=True)

        # Save images
        for modality, img in [('flair', flair_resized), ('t1', t1_resized),
                              ('t1ce', t1ce_resized), ('t2', t2_resized)]:
            mod_dir = out_dir / modality
            mod_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(mod_dir / f"{slice_id}.png")

        # Save segmentation
        seg_dir = out_dir / "seg"
        seg_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(seg_resized, mode='L').save(seg_dir / f"{slice_id}.png")

        # Compute statistics
        has_tc = (seg_resized == 1).any()
        has_ed = (seg_resized == 2).any()
        has_wt = has_tc or has_ed

        # Determine label
        if has_tc and has_ed:
            label = "WT"
        elif has_tc:
            label = "TC"
        elif has_ed:
            label = "ED"
        else:
            label = "Normal"

        slice_info = {
            'slice_id': slice_id,
            'case_id': case_id,
            'slice_idx': z,
            'label': label,
            'has_wt': int(has_wt),
            'has_tc': int(has_tc),
            'has_ed': int(has_ed),
        }
        slice_infos.append(slice_info)

    return slice_infos


def create_kfold_splits(all_slices_df, num_folds=5, seed=42):
    """Create K-fold splits at case level.

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
    for train_cases_idx, val_cases_idx in kf.split(case_ids):
        train_case_ids = case_ids[train_cases_idx]
        val_case_ids = case_ids[val_cases_idx]

        # Get slice indices
        train_indices = all_slices_df[all_slices_df['case_id'].isin(train_case_ids)].index.tolist()
        val_indices = all_slices_df[all_slices_df['case_id'].isin(val_case_ids)].index.tolist()

        splits.append((train_indices, val_indices))

    return splits


def main():
    parser = argparse.ArgumentParser(description="Convert NIfTI to Multi-Class PNG Format")
    parser.add_argument("--nifti_dir", type=str, required=True,
                       help="Path to NIfTI data directory (MICCAI_BraTS2020_TrainingData)")
    parser.add_argument("--out_dir", type=str, default="braintumnet/data/processed_multiclass",
                       help="Output directory")
    parser.add_argument("--img_size", type=int, default=256,
                       help="Target image size (square)")
    parser.add_argument("--slices_per_case", type=int, default=None,
                       help="Number of slices per case. Use None or 0 for ALL slices (default: None = ALL)")
    parser.add_argument("--tumor_ratio", type=float, default=0.7,
                       help="Ratio of tumor slices (only used if slices_per_case is set)")
    parser.add_argument("--num_folds", type=int, default=5,
                       help="Number of K-fold splits")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--max_cases", type=int, default=None,
                       help="Max cases to process (for testing)")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # If slices_per_case is 0, treat as None (ALL slices)
    if args.slices_per_case == 0:
        args.slices_per_case = None

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all case directories
    nifti_dir = Path(args.nifti_dir)
    if not nifti_dir.exists():
        print(f"Error: NIfTI directory not found: {nifti_dir}")
        sys.exit(1)

    case_dirs = sorted([d for d in nifti_dir.iterdir() if d.is_dir()])

    if args.max_cases:
        case_dirs = case_dirs[:args.max_cases]

    print(f"Found {len(case_dirs)} cases in {nifti_dir}")
    if args.slices_per_case is None:
        print(f"Processing with ALL slices per case (full 3D volume)...")
    else:
        print(f"Processing with {args.slices_per_case} slices per case (sampled)...")

    # Process all cases
    all_slices = []

    for case_dir in tqdm(case_dirs, desc="Processing cases"):
        slice_infos = process_case(case_dir, out_dir, args.img_size, args.slices_per_case)
        all_slices.extend(slice_infos)

    # Create DataFrame
    df = pd.DataFrame(all_slices)

    if len(df) == 0:
        print("Error: No slices processed!")
        sys.exit(1)

    # Save full dataset info
    df.to_csv(out_dir / "all_slices.csv", index=False)

    print(f"\n[OK] Processed {len(df)} slices from {df['case_id'].nunique()} cases")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nTumor region statistics:")
    print(f"  Whole Tumor (WT): {df['has_wt'].sum():,} slices ({df['has_wt'].mean()*100:.1f}%)")
    print(f"  Tumor Core (TC):  {df['has_tc'].sum():,} slices ({df['has_tc'].mean()*100:.1f}%)")
    print(f"  Edema (ED):       {df['has_ed'].sum():,} slices ({df['has_ed'].mean()*100:.1f}%)")

    # Create K-fold splits (only if enough cases)
    num_cases = df['case_id'].nunique()
    if num_cases < args.num_folds:
        print(f"\nWarning: Only {num_cases} cases, need at least {args.num_folds} for K-fold. Skipping fold creation.")
        splits = []
    else:
        splits = create_kfold_splits(df, num_folds=args.num_folds, seed=args.seed)

    # Save splits
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_df.to_csv(out_dir / f"train_fold{fold}.csv", index=False)
        val_df.to_csv(out_dir / f"val_fold{fold}.csv", index=False)

        print(f"\nFold {fold}:")
        print(f"  Train: {len(train_df):,} slices from {train_df['case_id'].nunique()} cases")
        print(f"  Val:   {len(val_df):,} slices from {val_df['case_id'].nunique()} cases")

    # Create class mapping file
    import json
    mapping = {
        "num_classes": 3,
        "class_names": ["Background", "TumorCore", "Edema"],
        "class_labels": [0, 1, 2],
        "regions": {
            "WT": "Whole Tumor = TC + ED (classes 1,2)",
            "TC": "Tumor Core = class 1 (BraTS labels 1,4)",
            "ED": "Edema = class 2 (BraTS label 2)"
        },
        "brats_label_mapping": {
            "0": "Background → class 0",
            "1": "NCR/NET → class 1",
            "2": "Edema → class 2",
            "4": "Enhancing Tumor → class 1"
        }
    }

    with open(out_dir / "class_mapping.json", 'w') as f:
        json.dump(mapping, f, indent=2)

    # Create labels.csv (case-level labels from name_mapping.csv)
    # Load HGG/LGG grades from name_mapping.csv
    name_mapping_path = nifti_dir / "name_mapping.csv"

    if name_mapping_path.exists():
        print(f"\nLoading grade labels from: {name_mapping_path}")
        grade_df = pd.read_csv(name_mapping_path)

        # Create mapping: case_id -> grade (HGG=0, LGG=1)
        grade_map = {}
        for _, row in grade_df.iterrows():
            case_id = row['BraTS_2020_subject_ID']
            grade = row['Grade']
            grade_map[case_id] = 0 if grade == 'HGG' else 1

        # Create labels
        case_labels = []
        for case_id in sorted(df['case_id'].unique()):
            label = grade_map.get(case_id, 0)  # Default to 0 (HGG) if not found
            case_labels.append({'case_id': case_id, 'label': label})

        labels_df = pd.DataFrame(case_labels)
        labels_df.to_csv(out_dir / "labels.csv", index=False)

        # Print distribution
        print(f"\nGrade distribution:")
        print(f"  HGG (label=0): {(labels_df['label']==0).sum()} cases")
        print(f"  LGG (label=1): {(labels_df['label']==1).sum()} cases")
    else:
        print(f"\nWarning: name_mapping.csv not found at {name_mapping_path}")
        print(f"Using default label=0 for all cases")

        case_labels = []
        for case_id in sorted(df['case_id'].unique()):
            case_labels.append({'case_id': case_id, 'label': 0})

        labels_df = pd.DataFrame(case_labels)
        labels_df.to_csv(out_dir / "labels.csv", index=False)

    # Create mapping.csv (slice-to-case mapping)
    mapping_df = df[['slice_id', 'case_id']].copy()
    mapping_df = mapping_df.sort_values('slice_id')
    mapping_df.to_csv(out_dir / "mapping.csv", index=False)

    print(f"\n[OK] Preprocessing complete!")
    print(f"[OK] Data saved to: {out_dir}")
    print(f"[OK] Class mapping saved to: {out_dir / 'class_mapping.json'}")
    print(f"\nNext steps:")
    print(f"1. Train model:")
    print(f"   python braintumnet/scripts/train.py --cfg braintumnet/configs/phase2_small.yaml --fold 0")


if __name__ == "__main__":
    main()
