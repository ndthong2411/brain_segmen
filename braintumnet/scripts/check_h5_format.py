import h5py
import numpy as np

# Check one H5 file to understand structure
h5_path = r"E:\thong\code\brain_segmen\brats2020_data\bcs2020\archive\BraTS2020_training_data\content\data\volume_1_slice_0.h5"

with h5py.File(h5_path, 'r') as f:
    print("Keys in H5 file:", list(f.keys()))
    for key in f.keys():
        data = f[key][:]
        print(f"\n{key}:")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        print(f"  Min: {data.min()}, Max: {data.max()}")
        if 'mask' in key.lower() or 'seg' in key.lower():
            unique_values = np.unique(data)
            print(f"  Unique values: {unique_values}")
