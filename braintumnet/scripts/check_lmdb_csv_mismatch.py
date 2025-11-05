"""
Check for mismatch between LMDB data and CSV split files
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lmdb
import pickle
import pandas as pd

def check_lmdb_csv_mismatch():
    """Check if CSV split files match LMDB data"""

    lmdb_root = "braintumnet/data/lmdb_processed_multiclass_full"
    fold = 3

    print("="*70)
    print("LMDB vs CSV Mismatch Check")
    print("="*70)
    print(f"LMDB root: {lmdb_root}")
    print(f"Fold: {fold}")

    # Load LMDB metadata
    print("\nLoading LMDB metadata...")
    env = lmdb.open(lmdb_root, readonly=True, lock=False, meminit=False)
    with env.begin() as txn:
        metadata = pickle.loads(txn.get(b'__metadata__'))
        lmdb_slice_ids = set(metadata['slice_ids'])
        num_samples = metadata['num_samples']

    print(f"✓ LMDB contains {num_samples} samples")
    print(f"  First 5 slice IDs: {list(lmdb_slice_ids)[:5]}")

    # Load CSV split file
    csv_path = f"{lmdb_root}/train_fold{fold}.csv"
    print(f"\nLoading CSV: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        csv_slice_ids = set(df['slice_id'].tolist())
        print(f"✓ CSV contains {len(csv_slice_ids)} slice IDs")
        print(f"  First 5 slice IDs: {list(csv_slice_ids)[:5]}")
    except Exception as e:
        print(f"✗ Failed to load CSV: {e}")
        return

    # Check for mismatches
    print("\n" + "="*70)
    print("Checking for mismatches...")
    print("="*70)

    # Slice IDs in CSV but not in LMDB
    missing_in_lmdb = csv_slice_ids - lmdb_slice_ids
    if missing_in_lmdb:
        print(f"\n✗ {len(missing_in_lmdb)} slice IDs in CSV but NOT in LMDB:")
        for sid in list(missing_in_lmdb)[:10]:
            print(f"    {sid}")
        if len(missing_in_lmdb) > 10:
            print(f"    ... and {len(missing_in_lmdb) - 10} more")
    else:
        print(f"\n✓ All CSV slice IDs found in LMDB")

    # Slice IDs in LMDB but not in CSV
    extra_in_lmdb = lmdb_slice_ids - csv_slice_ids
    if extra_in_lmdb:
        print(f"\n  {len(extra_in_lmdb)} slice IDs in LMDB but NOT in CSV (this is OK)")
    else:
        print(f"\n✓ No extra slice IDs in LMDB")

    # Test loading actual samples
    print("\n" + "="*70)
    print("Testing actual sample loading...")
    print("="*70)

    # Get first 5 slice IDs from CSV
    test_slice_ids = list(csv_slice_ids)[:5]

    for slice_id in test_slice_ids:
        print(f"\nTesting slice_id: {slice_id}")

        # Find LMDB index
        if slice_id in metadata['slice_ids']:
            lmdb_idx = metadata['slice_ids'].index(slice_id)
            print(f"  ✓ Found in LMDB at index: {lmdb_idx}")

            # Load sample
            with env.begin() as txn:
                key = f"{lmdb_idx:08d}".encode('ascii')
                sample_bytes = txn.get(key)

                if sample_bytes:
                    sample = pickle.loads(sample_bytes)
                    print(f"  ✓ Sample loaded successfully")
                    print(f"    Image shape: {sample['image'].shape}")
                    print(f"    Mask shape: {sample['mask'].shape}")
                    print(f"    Image range: [{sample['image'].min()}, {sample['image'].max()}]")
                    print(f"    Mask unique: {list(set(sample['mask'].flatten()))}")

                    # Check if data is corrupt
                    if sample['image'].max() == 0:
                        print(f"  ⚠ WARNING: Image is all zeros!")
                    if len(set(sample['mask'].flatten())) == 1 and sample['mask'].max() == 0:
                        print(f"  ⚠ WARNING: Mask is all background!")
                else:
                    print(f"  ✗ Failed to load sample from LMDB")
        else:
            print(f"  ✗ NOT found in LMDB!")

    env.close()

    print("\n" + "="*70)
    print("✓ Check complete!")
    print("="*70)

if __name__ == "__main__":
    check_lmdb_csv_mismatch()
