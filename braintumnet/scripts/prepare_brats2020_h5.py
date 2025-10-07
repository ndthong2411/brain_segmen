import os, argparse, csv, h5py
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

ROOT = Path(__file__).resolve().parents[1]  # braintumnet/
sys.path.append(str(ROOT / "src"))

def _rescale01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    nz = arr > 0
    if nz.sum() > 0:
        a = arr[nz]
        lo, hi = a.min(), a.max()
    else:
        lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    out[~np.isfinite(out)] = 0
    return out

def _save_png01(x: np.ndarray, path: str):
    x = (x * 255.0).clip(0,255).astype(np.uint8)
    Image.fromarray(x).save(path)

def process_h5_brats2020(h5_root: str, meta_csv: str, out_root: str,
                         img_size: int=256, modality_idx: int=2, max_slices: int=None,
                         multimodal: bool=False, min_tumor_ratio: float=0.001):
    """
    Process BraTS2020 HDF5 slices to PNG/NPY format.

    Args:
        h5_root: Directory containing .h5 files
        meta_csv: Path to meta_data.csv with columns: slice_path,target,volume,slice
        out_root: Output directory
        img_size: Target image size
        modality_idx: Which modality to use (0=FLAIR, 1=T1, 2=T1CE, 3=T2) - ignored if multimodal=True
        max_slices: Maximum number of slices to process (None = process all, recommended)
        multimodal: If True, save all 4 modalities stacked (4-channel)
        min_tumor_ratio: Minimum tumor pixel ratio to keep slice (0.001 = 0.1%)
    """
    os.makedirs(os.path.join(out_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "masks"), exist_ok=True)

    labels_path = os.path.join(out_root, "labels.csv")
    mapping_path = os.path.join(out_root, "mapping.csv")

    # Read metadata
    slice_info = []
    with open(meta_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slice_info.append(row)
            if max_slices and len(slice_info) >= max_slices:
                break

    total_in_metadata = len(slice_info)
    print(f"Found {total_in_metadata} slices in metadata")
    if max_slices:
        print(f"[WARNING] Processing only {max_slices} slices (limited mode)")
        print(f"          For full dataset, remove --max_slices argument")
    else:
        print(f"[OK] Processing ALL slices (full dataset mode)")

    # Track case labels
    case_labels = {}  # volume_id -> label
    slice_mapping = []  # (slice_id, case_id)

    # Statistics tracking
    processed = 0
    skipped = 0
    skipped_no_tumor = 0
    skipped_error = 0
    tumor_pixels_total = 0

    for info in tqdm(slice_info, desc="Processing slices"):
        h5_filename = os.path.basename(info['slice_path'])
        h5_path = os.path.join(h5_root, h5_filename)

        if not os.path.exists(h5_path):
            skipped += 1
            continue

        volume_id = info['volume']
        slice_idx = info['slice']
        label = int(info['target'])

        # Store case label
        case_labels[volume_id] = label

        try:
            with h5py.File(h5_path, 'r') as f:
                # Structure: image (H,W,4), mask (H,W,3) or (H,W)
                if 'image' in f and 'mask' in f:
                    image = np.array(f['image'])  # H,W,4 (4 modalities)
                    mask = np.array(f['mask'])    # H,W,3 or H,W

                    if multimodal:
                        # Process all 4 modalities
                        if image.ndim == 3 and image.shape[2] == 4:
                            # Normalize each modality separately
                            img_4ch = np.stack([_rescale01(image[:,:,i]) for i in range(4)], axis=-1)
                        else:
                            skipped += 1
                            continue
                    else:
                        # Extract single modality
                        if image.ndim == 3 and image.shape[2] >= modality_idx + 1:
                            img_modal = image[:, :, modality_idx]
                        else:
                            # Fallback: use mean if shape is unexpected
                            img_modal = image.mean(axis=2) if image.ndim == 3 else image

                        # Normalize
                        img_modal = _rescale01(img_modal)

                    # Handle multi-channel mask - combine all tumor regions
                    if mask.ndim == 3:
                        # Mask has multiple channels (e.g., 3 tumor regions)
                        # Combine all regions into a single binary mask
                        mask_bin = (mask.sum(axis=2) > 0).astype(np.float32)
                    else:
                        mask_bin = (mask > 0).astype(np.float32)

                    # Filter: Skip slices with too little tumor (optional)
                    if min_tumor_ratio > 0:
                        tumor_ratio = mask_bin.sum() / mask_bin.size
                        if tumor_ratio < min_tumor_ratio:
                            skipped_no_tumor += 1
                            continue

                    # Resize with padding to square
                    if multimodal:
                        h, w = img_4ch.shape[:2]
                    else:
                        h, w = img_modal.shape

                    s = max(h, w)
                    pad_h = s - h
                    pad_w = s - w

                    if multimodal:
                        # Pad all 4 channels
                        img_pad = np.pad(img_4ch, ((pad_h//2, pad_h - pad_h//2),
                                                   (pad_w//2, pad_w - pad_w//2),
                                                   (0, 0)), mode='constant')
                        # Resize each channel
                        img_resized = np.stack([
                            np.array(Image.fromarray((img_pad[:,:,i]*255).astype(np.uint8)).resize((img_size, img_size), Image.BILINEAR))
                            for i in range(4)
                        ], axis=-1).astype(np.float32) / 255.0
                    else:
                        img_pad = np.pad(img_modal, ((pad_h//2, pad_h - pad_h//2),
                                                      (pad_w//2, pad_w - pad_w//2)), mode='constant')
                        img_pil = Image.fromarray((img_pad*255).astype(np.uint8)).resize((img_size, img_size), Image.BILINEAR)

                    mask_pad = np.pad(mask_bin, ((pad_h//2, pad_h - pad_h//2),
                                                  (pad_w//2, pad_w - pad_w//2)), mode='constant')
                    mask_pil = Image.fromarray((mask_pad*255).astype(np.uint8)).resize((img_size, img_size), Image.NEAREST)

                    # Save
                    slice_id = f"vol{volume_id}_slice{slice_idx}"
                    if multimodal:
                        # Save as .npy for multi-channel
                        np.save(os.path.join(out_root, "images", f"{slice_id}.npy"), img_resized)
                    else:
                        # Save as PNG for single channel
                        _save_png01(np.array(img_pil).astype(np.float32)/255.0,
                                   os.path.join(out_root, "images", f"{slice_id}.png"))

                    Image.fromarray(np.array(mask_pil)).save(
                        os.path.join(out_root, "masks", f"{slice_id}.png"))

                    slice_mapping.append((slice_id, f"vol{volume_id}"))
                    processed += 1

                    # Track tumor statistics
                    tumor_pixels_total += mask_bin.sum()

        except Exception as e:
            print(f"Error processing {h5_filename}: {e}")
            skipped_error += 1
            skipped += 1
            continue

    # Print detailed statistics
    print("\n" + "="*70)
    print("PREPROCESSING SUMMARY")
    print("="*70)
    print(f"Total slices in metadata:    {total_in_metadata}")
    print(f"[OK] Successfully processed:    {processed}")
    print(f"[SKIP] Skipped (no tumor):      {skipped_no_tumor}")
    print(f"[ERROR] Skipped (errors):       {skipped_error}")
    print(f"[SKIP] Skipped (file not found):{skipped - skipped_error - skipped_no_tumor}")
    print(f"Total skipped:               {skipped}")
    print("-"*70)
    print(f"Unique cases:                {len(case_labels)}")
    print(f"Total tumor pixels:          {int(tumor_pixels_total):,}")
    if processed > 0:
        avg_tumor = tumor_pixels_total / processed
        print(f"Average tumor pixels/slice:  {int(avg_tumor):,}")
    print("="*70 + "\n")

    # Write labels.csv
    with open(labels_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['case_id', 'label'])
        for vol_id, label in sorted(case_labels.items()):
            writer.writerow([f"vol{vol_id}", label])

    # Write mapping.csv
    with open(mapping_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['slice_id', 'case_id'])
        for slice_id, case_id in slice_mapping:
            writer.writerow([slice_id, case_id])

    print(f"Labels written: {len(case_labels)} cases")
    print(f"Mapping written: {len(slice_mapping)} slices")

def make_folds(proc_root: str, num_folds: int=5):
    """Create stratified K-fold splits."""
    labels_csv = os.path.join(proc_root, "labels.csv")
    mapping_csv = os.path.join(proc_root, "mapping.csv")

    assert os.path.exists(labels_csv) and os.path.exists(mapping_csv), "Run processing first."

    # case -> label
    case_label = {}
    with open(labels_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            case_label[row["case_id"]] = int(row["label"])

    # case -> slice_ids
    case_slices = {}
    with open(mapping_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            case_slices.setdefault(row["case_id"], []).append(row["slice_id"])

    # Stratified split on cases
    cases0 = [c for c, l in case_label.items() if l == 0]
    cases1 = [c for c, l in case_label.items() if l == 1]

    random.seed(42)
    random.shuffle(cases0)
    random.shuffle(cases1)

    folds = [[] for _ in range(num_folds)]
    for i, c in enumerate(cases0):
        folds[i % num_folds].append(c)
    for i, c in enumerate(cases1):
        folds[i % num_folds].append(c)

    # Write slice ids per fold
    for k in range(num_folds):
        val_cases = set(folds[k])
        tr_cases = set(case_label.keys()) - val_cases
        tr_slices, val_slices = [], []
        for cid in tr_cases:
            tr_slices += case_slices[cid]
        for cid in val_cases:
            val_slices += case_slices[cid]

        with open(os.path.join(proc_root, f"split_train_fold{k}.txt"), "w") as f:
            f.write("\n".join(tr_slices))
        with open(os.path.join(proc_root, f"split_val_fold{k}.txt"), "w") as f:
            f.write("\n".join(val_slices))

    # Print fold statistics
    print("\n" + "="*70)
    print("CROSS-VALIDATION SPLITS")
    print("="*70)
    print(f"Number of folds:       {num_folds}")
    print(f"Total cases:           {len(case_label)}")
    print(f"  • LGG (class 0):     {len(cases0)} cases")
    print(f"  • HGG (class 1):     {len(cases1)} cases")
    print("-"*70)

    # Show per-fold statistics
    for k in range(num_folds):
        val_cases = len(folds[k])
        train_cases = len(case_label) - val_cases

        # Count slices
        val_slices_file = os.path.join(proc_root, f"split_val_fold{k}.txt")
        train_slices_file = os.path.join(proc_root, f"split_train_fold{k}.txt")

        with open(train_slices_file) as f:
            train_slices = len(f.readlines())
        with open(val_slices_file) as f:
            val_slices = len(f.readlines())

        print(f"Fold {k}: {train_cases:2d} train cases ({train_slices:5d} slices) | "
              f"{val_cases:2d} val cases ({val_slices:4d} slices)")

    print("="*70 + "\n")

def main():
    ap = argparse.ArgumentParser(
        description="Preprocess BraTS2020 HDF5 dataset for brain tumor segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process full dataset (RECOMMENDED):
  python prepare_brats2020_h5.py --h5_root data/raw --meta_csv data/raw/meta_data.csv --out data/processed

  # Process with multimodal (all 4 MRI sequences):
  python prepare_brats2020_h5.py --h5_root data/raw --meta_csv data/raw/meta_data.csv --out data/processed_multimodal --multimodal

  # Quick test (only 100 slices):
  python prepare_brats2020_h5.py --h5_root data/raw --meta_csv data/raw/meta_data.csv --out data/test --max_slices 100
        """)
    ap.add_argument("--h5_root", required=True, help="Directory containing .h5 files")
    ap.add_argument("--meta_csv", required=True, help="Path to meta_data.csv")
    ap.add_argument("--out", required=True, help="Output processed directory")
    ap.add_argument("--modality", default="t1ce", choices=["flair","t1","t1ce","t2"],
                    help="Single modality to use (default: t1ce, best for tumors)")
    ap.add_argument("--multimodal", action="store_true",
                    help="Process all 4 modalities (FLAIR, T1, T1CE, T2) as 4-channel .npy files")
    ap.add_argument("--img_size", type=int, default=256, help="Target image size (default: 256)")
    ap.add_argument("--num_folds", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    ap.add_argument("--max_slices", type=int, default=None,
                    help="Limit number of slices to process (default: None = process ALL, recommended for full training)")
    ap.add_argument("--min_tumor_ratio", type=float, default=0.001,
                    help="Minimum tumor pixel ratio to keep slice (default: 0.001 = 0.1%%)")
    args = ap.parse_args()

    # Map modality name to index
    modality_map = {"flair": 0, "t1": 1, "t1ce": 2, "t2": 3}
    modality_idx = modality_map[args.modality]

    os.makedirs(args.out, exist_ok=True)

    # Print configuration
    print("\n" + "="*70)
    print("BraTS2020 PREPROCESSING CONFIGURATION")
    print("="*70)
    print(f"Input directory:     {args.h5_root}")
    print(f"Metadata CSV:        {args.meta_csv}")
    print(f"Output directory:    {args.out}")
    print(f"Image size:          {args.img_size}×{args.img_size}")
    print(f"Min tumor ratio:     {args.min_tumor_ratio:.4f} ({args.min_tumor_ratio*100:.2f}%)")

    if args.multimodal:
        print(f"Mode:                Multi-modal (4 channels: FLAIR, T1, T1CE, T2)")
    else:
        print(f"Mode:                Single-modal ({args.modality.upper()} only)")

    if args.max_slices:
        print(f"Slice limit:         {args.max_slices} (TESTING MODE - WARNING)")
    else:
        print(f"Slice limit:         None (processing ALL data - OK)")

    print("="*70 + "\n")

    process_h5_brats2020(args.h5_root, args.meta_csv, args.out,
                         args.img_size, modality_idx, args.max_slices, args.multimodal,
                         args.min_tumor_ratio)
    make_folds(args.out, args.num_folds)

if __name__ == "__main__":
    main()
