"""
Create K-fold splits from all_slices.csv
Generates train_fold0-4.csv and val_fold0-4.csv
"""
import pandas as pd
from sklearn.model_selection import KFold
from pathlib import Path
import argparse

def create_kfold_splits(df, num_folds=5, seed=42):
    """
    Create K-fold splits at volume level (not slice level) to prevent data leakage.

    Args:
        df: DataFrame with columns [slice_id, volume_id, ...]
        num_folds: Number of folds
        seed: Random seed

    Returns:
        list of (train_indices, val_indices) tuples
    """
    # Get unique volume IDs
    volume_ids = df['volume_id'].unique()
    print(f"Total volumes: {len(volume_ids)}")

    # Create KFold splitter
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    splits = []
    for fold, (train_vols, val_vols) in enumerate(kf.split(volume_ids)):
        # Get volume IDs for train and val
        train_vol_ids = volume_ids[train_vols]
        val_vol_ids = volume_ids[val_vols]

        # Get slice indices belonging to these volumes
        train_indices = df[df['volume_id'].isin(train_vol_ids)].index.tolist()
        val_indices = df[df['volume_id'].isin(val_vol_ids)].index.tolist()

        splits.append((train_indices, val_indices))

        print(f"\nFold {fold}:")
        print(f"  Train volumes: {len(train_vol_ids)}, slices: {len(train_indices)}")
        print(f"  Val volumes:   {len(val_vol_ids)}, slices: {len(val_indices)}")

    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing all_slices.csv')
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_slices_csv = data_dir / 'all_slices.csv'

    if not all_slices_csv.exists():
        raise FileNotFoundError(f"File not found: {all_slices_csv}")

    print(f"Reading {all_slices_csv}...")
    df = pd.read_csv(all_slices_csv)
    print(f"Total slices: {len(df)}")

    # Create K-fold splits
    splits = create_kfold_splits(df, num_folds=args.num_folds, seed=args.seed)

    # Save splits
    print("\nSaving fold CSV files...")
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_csv = data_dir / f"train_fold{fold}.csv"
        val_csv = data_dir / f"val_fold{fold}.csv"

        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        print(f"  Saved {train_csv.name} ({len(train_df):,} slices)")
        print(f"  Saved {val_csv.name} ({len(val_df):,} slices)")

    print(f"\n✓ Created {args.num_folds * 2} fold CSV files in {data_dir}")


if __name__ == '__main__':
    main()
