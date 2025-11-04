"""
Convert PNG dataset to LMDB format for fast data loading
=========================================================

LMDB (Lightning Memory-Mapped Database) provides:
- 10-15x faster random access than PNG files
- Single file database (easy to manage)
- Memory-mapped I/O (zero-copy reads)
- Perfect for training on fast GPUs (A100)

Usage:
    python scripts/convert_to_lmdb.py \
        --input_dir braintumnet/data/processed_multiclass_with_grades \
        --output_dir braintumnet/data/lmdb_multiclass_with_grades \
        --map_size 50

Input structure:
    processed_multiclass_with_grades/
    ├── flair/*.png
    ├── t1/*.png
    ├── t1ce/*.png
    ├── t2/*.png
    ├── seg/*.png
    ├── labels.csv
    ├── mapping.csv
    └── *.csv (fold splits)

Output structure:
    lmdb_multiclass_with_grades/
    ├── data.mdb          # LMDB database file
    ├── lock.mdb          # LMDB lock file
    ├── labels.csv        # Copied from input
    ├── mapping.csv       # Copied from input
    ├── meta.json         # Metadata (num_samples, modalities, etc)
    └── *.csv             # Copied fold splits

Author: BrainTumNet Optimization
Date: 2025-10-30
"""

import os
import sys
from pathlib import Path
import argparse
import lmdb
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
import json
import shutil

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_multimodal_sample(input_dir, slice_id):
    """Load 4 modalities + segmentation for a single slice.

    Args:
        input_dir: Input directory with flair/, t1/, t1ce/, t2/, seg/
        slice_id: Slice ID (e.g., "BraTS20_Training_001_slice050")

    Returns:
        data_dict: {
            'image': np.array (4, H, W) uint8,
            'mask': np.array (H, W) uint8,
            'slice_id': str
        }
    """
    # Load 4 modalities
    flair = np.array(Image.open(input_dir / "flair" / f"{slice_id}.png"))
    t1 = np.array(Image.open(input_dir / "t1" / f"{slice_id}.png"))
    t1ce = np.array(Image.open(input_dir / "t1ce" / f"{slice_id}.png"))
    t2 = np.array(Image.open(input_dir / "t2" / f"{slice_id}.png"))

    # Stack to (4, H, W) - channel first for PyTorch
    image = np.stack([flair, t1, t1ce, t2], axis=0).astype(np.uint8)

    # Load segmentation mask
    mask = np.array(Image.open(input_dir / "seg" / f"{slice_id}.png")).astype(np.uint8)

    return {
        'image': image,
        'mask': mask,
        'slice_id': slice_id
    }


def get_all_slice_ids(input_dir):
    """Get all slice IDs from flair directory.

    Args:
        input_dir: Input directory with flair/ subfolder

    Returns:
        slice_ids: List of slice IDs (without .png extension)
    """
    flair_dir = input_dir / "flair"
    if not flair_dir.exists():
        raise FileNotFoundError(f"flair directory not found: {flair_dir}")

    slice_ids = sorted([
        f.stem for f in flair_dir.glob("*.png")
    ])

    return slice_ids


def convert_to_lmdb(input_dir, output_dir, map_size_gb=50):
    """Convert PNG dataset to LMDB format.

    Args:
        input_dir: Input directory with PNG files
        output_dir: Output directory for LMDB database
        map_size_gb: LMDB map size in GB (default: 50GB)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting PNG dataset to LMDB...")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Map size: {map_size_gb} GB")

    # Get all slice IDs
    slice_ids = get_all_slice_ids(input_dir)
    print(f"\nFound {len(slice_ids)} slices")

    # Create LMDB environment
    map_size = map_size_gb * 1024 * 1024 * 1024  # Convert to bytes
    env = lmdb.open(
        str(output_dir),
        map_size=map_size,
        readonly=False,
        meminit=False,
        map_async=True
    )

    # Write samples to LMDB
    with env.begin(write=True) as txn:
        for idx, slice_id in enumerate(tqdm(slice_ids, desc="Converting")):
            try:
                # Load sample
                sample = load_multimodal_sample(input_dir, slice_id)

                # Serialize with pickle (faster than JSON)
                sample_bytes = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)

                # Store in LMDB with index as key
                key = f"{idx:08d}".encode('ascii')
                txn.put(key, sample_bytes)

                # Also store slice_id -> index mapping
                id_key = f"id_{slice_id}".encode('ascii')
                txn.put(id_key, str(idx).encode('ascii'))

            except Exception as e:
                print(f"\nError processing {slice_id}: {e}")
                continue

        # Store metadata
        metadata = {
            'num_samples': len(slice_ids),
            'modalities': ['flair', 't1', 't1ce', 't2'],
            'num_channels': 4,
            'has_segmentation': True,
            'num_classes': 3,
            'slice_ids': slice_ids
        }
        txn.put(b'__metadata__', pickle.dumps(metadata))

    env.close()

    # Copy CSV files
    print("\nCopying metadata files...")
    for csv_file in input_dir.glob("*.csv"):
        shutil.copy(csv_file, output_dir / csv_file.name)
        print(f"  Copied: {csv_file.name}")

    # Save metadata as JSON for easy inspection
    with open(output_dir / "meta.json", 'w') as f:
        # Don't include full slice_ids list in JSON (too large)
        meta_json = metadata.copy()
        meta_json['slice_ids'] = f"<{len(slice_ids)} items>"
        json.dump(meta_json, f, indent=2)
    print(f"  Saved: meta.json")

    print(f"\n[OK] Conversion complete!")
    print(f"[OK] LMDB database saved to: {output_dir}")

    # Print size comparison
    lmdb_size = sum(f.stat().st_size for f in output_dir.glob("*.mdb"))
    print(f"\nDatabase size: {lmdb_size / 1024**3:.2f} GB")


def verify_lmdb(output_dir):
    """Verify LMDB database integrity.

    Args:
        output_dir: LMDB database directory
    """
    output_dir = Path(output_dir)
    env = lmdb.open(str(output_dir), readonly=True, lock=False)

    with env.begin() as txn:
        # Load metadata
        metadata = pickle.loads(txn.get(b'__metadata__'))
        print(f"\nVerifying LMDB database...")
        print(f"  Samples: {metadata['num_samples']}")
        print(f"  Modalities: {metadata['modalities']}")
        print(f"  Channels: {metadata['num_channels']}")
        print(f"  Classes: {metadata['num_classes']}")

        # Test random sample
        import random
        test_idx = random.randint(0, metadata['num_samples'] - 1)
        key = f"{test_idx:08d}".encode('ascii')
        sample_bytes = txn.get(key)

        if sample_bytes:
            sample = pickle.loads(sample_bytes)
            print(f"\nSample test (index {test_idx}):")
            print(f"  Slice ID: {sample['slice_id']}")
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Mask shape: {sample['mask'].shape}")
            print(f"  Image dtype: {sample['image'].dtype}")
            print(f"  Mask dtype: {sample['mask'].dtype}")
            print(f"  Mask values: {np.unique(sample['mask'])}")
            print(f"\n[OK] Verification passed!")
        else:
            print(f"\n[ERROR] Could not load sample {test_idx}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Convert PNG dataset to LMDB format")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory with PNG files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for LMDB database")
    parser.add_argument("--map_size", type=int, default=50,
                       help="LMDB map size in GB (default: 50)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify database after conversion")

    args = parser.parse_args()

    # Convert
    convert_to_lmdb(args.input_dir, args.output_dir, args.map_size)

    # Verify
    if args.verify:
        verify_lmdb(args.output_dir)


if __name__ == "__main__":
    main()
