"""
Quick test script to verify EDA notebook data loading works correctly
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project to path
project_root = Path('../braintumnet/src').resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Dataset paths
DATA_ROOT = Path('../braintumnet/data')
LMDB_ROOT = DATA_ROOT / 'lmdb_processed_multiclass_full'

print("=" * 60)
print("EDA NOTEBOOK DATA LOADING TEST")
print("=" * 60)

# Test 1: Load CSV files
print("\n1. Testing CSV file loading...")
try:
    all_slices_df = pd.read_csv(LMDB_ROOT / 'all_slices.csv')
    labels_df = pd.read_csv(LMDB_ROOT / 'labels.csv')
    mapping_df = pd.read_csv(LMDB_ROOT / 'mapping.csv')

    # Rename to avoid conflicts
    labels_df = labels_df.rename(columns={'label': 'grade'})

    print(f"   [OK] all_slices_df: {all_slices_df.shape}, columns: {list(all_slices_df.columns)}")
    print(f"   [OK] labels_df: {labels_df.shape}, columns: {list(labels_df.columns)}")
    print(f"   [OK] mapping_df: {mapping_df.shape}, columns: {list(mapping_df.columns)}")
except Exception as e:
    print(f"   [ERROR] Error loading CSV files: {e}")
    sys.exit(1)

# Test 2: Merge operation
print("\n2. Testing dataframe merge...")
try:
    df = all_slices_df.copy()
    df = df.merge(labels_df, on='case_id', how='left')
    print(f"   [OK] Merged df: {df.shape}, columns: {list(df.columns)}")
    print(f"   [OK] No NaN in grade: {df['grade'].notna().all()}")
except Exception as e:
    print(f"   [ERROR] Error merging dataframes: {e}")
    sys.exit(1)

# Test 3: Load fold data
print("\n3. Testing fold data loading...")
try:
    fold_data = {}
    for fold in range(5):
        train_df = pd.read_csv(LMDB_ROOT / f'train_fold{fold}.csv')
        val_df = pd.read_csv(LMDB_ROOT / f'val_fold{fold}.csv')
        fold_data[fold] = {
            'train': train_df,
            'val': val_df,
            'train_size': len(train_df),
            'val_size': len(val_df)
        }
    print(f"   [OK] Loaded {len(fold_data)} folds")
    for fold, data in fold_data.items():
        print(f"      Fold {fold}: train={data['train_size']:,}, val={data['val_size']:,}")
except Exception as e:
    print(f"   [ERROR] Error loading fold data: {e}")
    sys.exit(1)

# Test 4: Fold label distribution
print("\n4. Testing fold label distribution...")
try:
    fold_label_dist = []
    for fold in range(5):
        train_df = fold_data[fold]['train']
        val_df = fold_data[fold]['val']

        # Extract case IDs
        train_cases = train_df['case_id'].unique()
        val_cases = val_df['case_id'].unique()

        # Get labels
        train_labels = labels_df[labels_df['case_id'].isin(train_cases)]['grade']
        val_labels = labels_df[labels_df['case_id'].isin(val_cases)]['grade']

        fold_label_dist.append({
            'fold': fold,
            'train_lgg': (train_labels == 0).sum(),
            'train_hgg': (train_labels == 1).sum(),
            'val_lgg': (val_labels == 0).sum(),
            'val_hgg': (val_labels == 1).sum()
        })

    fold_dist_df = pd.DataFrame(fold_label_dist)
    print(f"   [OK] Fold distribution calculated")
    print(fold_dist_df.to_string(index=False))
except Exception as e:
    print(f"   [ERROR] Error calculating fold distribution: {e}")
    sys.exit(1)

# Test 5: Grade distribution
print("\n5. Testing grade distribution...")
try:
    label_counts = labels_df['grade'].value_counts().sort_index()
    label_names = {0: 'LGG (Low Grade)', 1: 'HGG (High Grade)'}

    print("   Grade Distribution:")
    for grade, count in label_counts.items():
        pct = 100 * count / len(labels_df)
        print(f"      {label_names[grade]}: {count:,} ({pct:.1f}%)")
except Exception as e:
    print(f"   [ERROR] Error calculating grade distribution: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nThe notebook should now run without errors.")
print("Please restart the kernel and run all cells.")
