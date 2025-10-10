import h5py
import numpy as np
import os

# Check multiple files to find one with tumor
data_dir = r"E:\thong\code\brain_segmen\brats2020_data\bcs2020\archive\BraTS2020_training_data\content\data"

files = [f for f in os.listdir(data_dir) if f.endswith('.h5')][:20]  # Check first 20

for fname in files:
    h5_path = os.path.join(data_dir, fname)
    with h5py.File(h5_path, 'r') as f:
        mask = f['mask'][:]
        if mask.max() > 0:
            print(f"\nFile: {fname}")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Mask channels: {mask.shape[2] if len(mask.shape) == 3 else 1}")
            for ch in range(mask.shape[2]):
                unique = np.unique(mask[:, :, ch])
                print(f"  Channel {ch}: unique values = {unique}, non-zero = {(mask[:, :, ch] > 0).sum()}")
            break
