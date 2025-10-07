#!/usr/bin/env python
"""Quick verification that all paths and imports work correctly."""
import os
import sys

print("=" * 70)
print("BrainTumNet Setup Verification")
print("=" * 70)

# Check we're in the right directory
cwd = os.getcwd()
print(f"\nCurrent directory: {cwd}")
if not cwd.endswith('braintumnet'):
    print("WARNING: Run this script from the braintumnet/ directory")
    print("  cd braintumnet && python verify_setup.py")

# Check critical directories exist
print("\n1. Checking directory structure...")
dirs_to_check = [
    'configs',
    'data',
    'src/braintumnet',
    'scripts',
    'checkpoints',
    'runs',
    'docs',
    'tests'
]

all_ok = True
for d in dirs_to_check:
    exists = os.path.exists(d)
    status = "[OK]" if exists else "[MISSING]"
    print(f"   {status} {d}/")
    if not exists:
        all_ok = False

# Check critical files
print("\n2. Checking critical files...")
files_to_check = [
    'README.md',
    'requirements.txt',
    'configs/default.yaml',
    'configs/quick_test.yaml',
    'scripts/train.py',
    'scripts/evaluate.py',
    'scripts/predict.py',
    'scripts/prepare_brats2020_h5.py',
    'src/braintumnet/__init__.py',
    'src/braintumnet/models/braintumnet.py',
    'src/braintumnet/engine/trainer.py'
]

for f in files_to_check:
    exists = os.path.exists(f)
    status = "[OK]" if exists else "[MISSING]"
    print(f"   {status} {f}")
    if not exists:
        all_ok = False

# Test imports
print("\n3. Testing imports...")
sys.path.insert(0, 'src')

try:
    from braintumnet.models.braintumnet import BrainTumNet
    print("   [OK] Import BrainTumNet")
except Exception as e:
    print(f"   [FAIL] Import BrainTumNet: {e}")
    all_ok = False

try:
    from braintumnet.data.brats2020_dataset import SliceDataset
    print("   [OK] Import SliceDataset")
except Exception as e:
    print(f"   [FAIL] Import SliceDataset: {e}")
    all_ok = False

try:
    from braintumnet.engine.trainer import train_one_fold
    print("   [OK] Import train_one_fold")
except Exception as e:
    print(f"   [FAIL] Import train_one_fold: {e}")
    all_ok = False

try:
    from braintumnet.losses import MultiTaskLoss
    print("   [OK] Import MultiTaskLoss")
except Exception as e:
    print(f"   [FAIL] Import MultiTaskLoss: {e}")
    all_ok = False

try:
    from braintumnet.metrics import iou_score, dice_score
    print("   [OK] Import metrics")
except Exception as e:
    print(f"   [FAIL] Import metrics: {e}")
    all_ok = False

# Check data
print("\n4. Checking processed data...")
processed_data = 'data/processed'
if os.path.exists(processed_data):
    items = os.listdir(processed_data)
    print(f"   [OK] Processed data directory exists")
    print(f"   Found: {', '.join(items[:5])}...")

    # Check for images
    if os.path.exists(os.path.join(processed_data, 'images')):
        num_images = len(os.listdir(os.path.join(processed_data, 'images')))
        print(f"   [OK] {num_images} images found")
    else:
        print(f"   [MISSING] images/ directory")

    # Check for splits
    splits = [f for f in os.listdir(processed_data) if f.startswith('split_')]
    if splits:
        print(f"   [OK] {len(splits)} split files found")
    else:
        print(f"   [MISSING] split files")
else:
    print(f"   [MISSING] {processed_data} directory")
    print(f"   Run: python scripts/prepare_brats2020_h5.py ...")

# Check checkpoints
print("\n5. Checking checkpoints...")
if os.path.exists('checkpoints'):
    ckpts = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
    if ckpts:
        print(f"   [OK] {len(ckpts)} checkpoint(s) found")
        for ckpt in ckpts:
            size_mb = os.path.getsize(os.path.join('checkpoints', ckpt)) / 1024 / 1024
            print(f"       - {ckpt} ({size_mb:.1f} MB)")
    else:
        print(f"   [INFO] No checkpoints yet (train a model first)")
else:
    print(f"   [MISSING] checkpoints/ directory")

# Summary
print("\n" + "=" * 70)
if all_ok:
    print("Status: ALL CHECKS PASSED")
    print("\nYou can now:")
    print("  - Train: python scripts/train.py --cfg configs/quick_test.yaml --fold 0")
    print("  - Evaluate: python scripts/evaluate.py --cfg configs/quick_test.yaml --ckpt checkpoints/xxx.pth --fold 0")
    print("  - Predict: python scripts/predict.py --cfg configs/quick_test.yaml --ckpt checkpoints/xxx.pth --img path/to/image.png")
else:
    print("Status: SOME CHECKS FAILED")
    print("Please review the issues above")
print("=" * 70)
