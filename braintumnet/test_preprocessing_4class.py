"""
Quick test: Preprocess ONE case to verify 4-class conversion works
"""
import os
import sys
import subprocess

print("="*60)
print("QUICK TEST: Preprocessing ONE case to 4-class")
print("="*60)

# Run preprocessing with max_cases=1
cmd = [
    sys.executable,
    "scripts/preprocess_nifti_to_multiclass.py",
    "--nifti_dir", "E:/thong/code/brain_segmen/braintumnet/data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
    "--out_dir", "E:/thong/code/brain_segmen/braintumnet/data/test_4class",
    "--img_size", "256",
    "--num_folds", "5",
    "--max_cases", "1"  # Only process 1 case for testing
]

print("\nRunning command:")
print(" ".join(cmd))
print()

result = subprocess.run(cmd, capture_output=False)

if result.returncode == 0:
    print("\n" + "="*60)
    print("SUCCESS: Test preprocessing completed!")
    print("="*60)

    # Verify output
    import json
    import numpy as np
    from pathlib import Path

    test_dir = Path("E:/thong/code/brain_segmen/braintumnet/data/test_4class")

    # Check class_mapping.json
    mapping_file = test_dir / "class_mapping.json"
    if mapping_file.exists():
        with open(mapping_file) as f:
            mapping = json.load(f)
        print(f"\nClass mapping:")
        print(f"  num_classes: {mapping['num_classes']}")
        print(f"  class_names: {mapping['class_names']}")

        if mapping['num_classes'] == 4:
            print("\n[PASS] CORRECT: 4 classes detected")
        else:
            print(f"\n[FAIL] ERROR: Expected 4 classes, got {mapping['num_classes']}")

    # Check a sample mask
    seg_dir = test_dir / "seg"
    if seg_dir.exists():
        seg_files = list(seg_dir.glob("*.npy"))
        if seg_files:
            sample = np.load(seg_files[0])
            unique_labels = np.unique(sample)
            print(f"\nSample mask: {seg_files[0].name}")
            print(f"  Shape: {sample.shape}")
            print(f"  Unique labels: {unique_labels}")

            expected = np.array([0, 1, 2, 3])
            if set(unique_labels).issubset(set(expected)):
                print("\n[PASS] CORRECT: Labels are subset of [0,1,2,3]")
            else:
                print(f"\n[FAIL] ERROR: Unexpected labels found")

    print("\n" + "="*60)
    print("NEXT STEP: Run full preprocessing")
    print("="*60)
    print("\nWindows: run_preprocess_4class.bat")
    print("Linux/Mac: bash run_preprocess_4class.sh")
    print("\nOr run manually:")
    print("python scripts/preprocess_nifti_to_multiclass.py \\")
    print("    --nifti_dir E:/thong/code/brain_segmen/braintumnet/data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \\")
    print("    --out_dir E:/thong/code/brain_segmen/braintumnet/data/processed_multiclass_4class \\")
    print("    --img_size 256 --num_folds 5")

else:
    print("\n" + "="*60)
    print("ERROR: Test preprocessing failed!")
    print("="*60)
    print("\nPlease check the error messages above.")
    sys.exit(1)
