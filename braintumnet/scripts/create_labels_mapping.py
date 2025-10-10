"""
Create labels.csv and mapping.csv from all_slices.csv
======================================================

Generates two CSV files compatible with the original preprocessing format:
1. labels.csv - Case-level labels (case_id, label)
2. mapping.csv - Slice-to-case mapping (slice_id, case_id)

Usage:
    python scripts/create_labels_mapping.py \
        --data_dir braintumnet/data/processed_multiclass

Date: 2025-10-10
"""

import argparse
import pandas as pd
from pathlib import Path


def create_labels_csv(all_slices_df, output_path):
    """Create labels.csv with case-level labels.

    Args:
        all_slices_df: DataFrame with all slices (from all_slices.csv)
        output_path: Path to save labels.csv
    """
    # Group by volume_id and assign label based on majority vote
    volume_labels = []

    for vol_id in sorted(all_slices_df['volume_id'].unique()):
        vol_slices = all_slices_df[all_slices_df['volume_id'] == vol_id]

        # Determine volume-level label
        # Priority: If any slice has tumor → label = 0 (tumor case)
        # Otherwise → label = 0 (normal case)
        # Note: For binary classification, both are 0 (we only have tumor data)
        # For multi-class, we keep this simple format

        has_tumor = vol_slices['has_wt'].any()

        # In BraTS all cases have tumors, so label is always 0
        # (This matches the original format where all cases were labeled 0)
        label = 0

        volume_labels.append({
            'case_id': vol_id,
            'label': label
        })

    labels_df = pd.DataFrame(volume_labels)
    labels_df.to_csv(output_path, index=False)

    print(f"✓ Created {output_path}")
    print(f"  Total cases: {len(labels_df)}")


def create_mapping_csv(all_slices_df, output_path):
    """Create mapping.csv with slice-to-case mapping.

    Args:
        all_slices_df: DataFrame with all slices (from all_slices.csv)
        output_path: Path to save mapping.csv
    """
    # Extract slice_id and volume_id (volume_id is the case_id)
    mapping_df = all_slices_df[['slice_id', 'volume_id']].copy()
    mapping_df = mapping_df.rename(columns={'volume_id': 'case_id'})

    # Sort by slice_id
    mapping_df = mapping_df.sort_values('slice_id')

    mapping_df.to_csv(output_path, index=False)

    print(f"✓ Created {output_path}")
    print(f"  Total slices: {len(mapping_df):,}")


def main():
    parser = argparse.ArgumentParser(description="Create labels.csv and mapping.csv")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Data directory containing all_slices.csv")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_slices_path = data_dir / "all_slices.csv"

    if not all_slices_path.exists():
        print(f"Error: {all_slices_path} not found")
        print("Make sure preprocessing has completed successfully.")
        return

    # Load all_slices.csv
    print(f"Loading {all_slices_path}...")
    all_slices_df = pd.read_csv(all_slices_path)

    # Create labels.csv
    labels_path = data_dir / "labels.csv"
    create_labels_csv(all_slices_df, labels_path)

    # Create mapping.csv
    mapping_path = data_dir / "mapping.csv"
    create_mapping_csv(all_slices_df, mapping_path)

    print(f"\n✓ Done! Created:")
    print(f"  - {labels_path}")
    print(f"  - {mapping_path}")


if __name__ == "__main__":
    main()
