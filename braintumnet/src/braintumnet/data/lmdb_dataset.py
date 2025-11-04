"""
LMDB Dataset for fast data loading
===================================

LMDB (Lightning Memory-Mapped Database) backend for BraTS dataset.
Provides 10-15x faster loading than PNG files.

Compatible with SliceDataset API for drop-in replacement.
"""

import os
import lmdb
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from .advanced_transforms import MedicalAugmentation


class LMDBDataset(Dataset):
    """LMDB-backed dataset for fast loading.

    Compatible with SliceDataset API.

    Args:
        lmdb_root: Path to LMDB database directory
        split_file: Path to split file (CSV or TXT)
        img_size: Image size (not used, images pre-resized)
        rotate_deg: Rotation angle for augmentation (legacy, not used)
        hflip_p: Horizontal flip probability (legacy, not used)
        vflip_p: Vertical flip probability (legacy, not used)
        train: Whether this is training set
        in_channels: Number of input channels (should be 4 for multi-modal)
        augment_config: Dictionary of augmentation parameters for MedicalAugmentation (Phase 1)
    """

    def __init__(self, lmdb_root: str, split_file: str,
                 img_size: int=256, rotate_deg: int=30, hflip_p: float=0.5, vflip_p: float=0.5,
                 train: bool=True, in_channels: int=4, augment_config: Optional[Dict]=None):
        self.lmdb_root = lmdb_root
        self.train = train
        self.img_size = img_size
        self.rotate_deg, self.hflip_p, self.vflip_p = rotate_deg, hflip_p, vflip_p
        self.in_channels = in_channels

        # Phase 1: Advanced Medical Augmentation
        if train and augment_config is not None:
            self.medical_aug = MedicalAugmentation(
                elastic_deform_p=augment_config.get('elastic_deform_p', 0.3),
                elastic_alpha=augment_config.get('elastic_alpha', 30),
                elastic_sigma=augment_config.get('elastic_sigma', 4),
                bias_field_p=augment_config.get('bias_field_p', 0.5),
                bias_field_scale=augment_config.get('bias_field_scale', 0.3),
                gaussian_blur_p=augment_config.get('gaussian_blur_p', 0.2),
                gaussian_blur_sigma=augment_config.get('gaussian_blur_sigma', (0.5, 1.5)),
                gamma_p=augment_config.get('gamma_p', 0.5),
                gamma_range=augment_config.get('gamma_range', (0.7, 1.4)),
                cutout_p=augment_config.get('cutout_p', 0.2),
                cutout_n_holes=augment_config.get('cutout_n_holes', 3),
                cutout_size=augment_config.get('cutout_size', 20),
                local_shuffle_p=augment_config.get('local_shuffle_p', 0.15),
                local_shuffle_size=augment_config.get('local_shuffle_size', 3),
            )
        else:
            self.medical_aug = None

        # Delay LMDB environment creation (lazy init in __getitem__)
        # This is required for Windows multiprocessing (Environment objects can't be pickled)
        self.env = None

        # Load metadata from a temporary environment
        env_temp = lmdb.open(
            lmdb_root,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        with env_temp.begin() as txn:
            metadata = pickle.loads(txn.get(b'__metadata__'))
            self.num_samples = metadata['num_samples']
            self.all_slice_ids = metadata['slice_ids']
        env_temp.close()

        # Read split file to get slice IDs for this split
        if split_file.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(split_file)
            self.slice_ids: List[str] = df['slice_id'].tolist()
        else:
            with open(split_file, "r") as f:
                self.slice_ids: List[str] = [x.strip() for x in f if x.strip()]

        # Create slice_id -> LMDB index mapping
        self.slice_to_idx = {sid: idx for idx, sid in enumerate(self.all_slice_ids)}

        # Filter indices for this split
        self.indices = [self.slice_to_idx[sid] for sid in self.slice_ids if sid in self.slice_to_idx]

        # Load labels and mapping
        self.case_label: Dict[str, int] = {}
        labels_csv = os.path.join(lmdb_root, "labels.csv")
        if os.path.exists(labels_csv):
            with open(labels_csv) as f:
                next(f)  # skip header
                for line in f:
                    if "," in line:
                        cid, lab = line.strip().split(",")
                        self.case_label[cid] = int(lab)

        self.slice_case: Dict[str, str] = {}
        mapping_csv = os.path.join(lmdb_root, "mapping.csv")
        if os.path.exists(mapping_csv):
            with open(mapping_csv) as f:
                next(f)  # skip header
                for line in f:
                    if "," in line:
                        sid, cid = line.strip().split(",")
                        self.slice_case[sid] = cid

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Lazy initialization of LMDB environment (required for Windows multiprocessing)
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_root,
                readonly=True,
                lock=False,
                readahead=True,  # Enable OS-level readahead
                meminit=False
            )

        # Get LMDB index
        lmdb_idx = self.indices[idx]

        # Read from LMDB
        with self.env.begin() as txn:
            key = f"{lmdb_idx:08d}".encode('ascii')
            sample_bytes = txn.get(key)

            if sample_bytes is None:
                raise KeyError(f"Sample not found in LMDB: index {lmdb_idx}")

            sample = pickle.loads(sample_bytes)

        # Extract data
        image = sample['image']  # (4, H, W) uint8
        mask = sample['mask']    # (H, W) uint8
        slice_id = sample['slice_id']

        # Convert to torch tensors
        img_t = torch.from_numpy(image).float()  # (4, H, W)
        msk_t = torch.from_numpy(mask).long()  # (H, W)

        # Phase 1: Apply advanced medical augmentation
        if self.train and self.medical_aug is not None:
            img_t, msk_t = self.medical_aug(img_t, msk_t)

        # Ensure mask has channel dimension
        if msk_t.ndim == 2:
            msk_t = msk_t.unsqueeze(0)  # (1, H, W)

        # Get case label
        cid = self.slice_case.get(slice_id, slice_id.split("_")[0])
        label = self.case_label.get(cid, 0)

        return {
            "image": img_t,
            "mask": msk_t,
            "label": torch.tensor(label, dtype=torch.long),
            "slice_id": slice_id,
            "case_id": cid
        }

    def __del__(self):
        # Close LMDB environment when dataset is destroyed
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
