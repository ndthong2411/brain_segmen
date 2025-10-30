import os
from typing import List, Dict, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from functools import lru_cache
from .transforms import augment_pair

class SliceDataset(Dataset):
    """
    processed/
      images/<slice_id>.png    (grayscale or 4ch .npy if multi)
      masks/<slice_id>.png     (0/255)
      labels.csv               (case_id,label)
      mapping.csv              (slice_id,case_id)
      split_train_fold{k}.txt
      split_val_fold{k}.txt
    """
    def __init__(self, proc_root: str, split_file: str,
                 img_size: int=256, rotate_deg: int=30, hflip_p: float=0.5, vflip_p: float=0.5,
                 train: bool=True, in_channels: int=1, cache_size: int=1000):
        self.proc_root = proc_root
        self.train = train
        self.img_size = img_size
        self.rotate_deg, self.hflip_p, self.vflip_p = rotate_deg, hflip_p, vflip_p
        self.in_channels = in_channels
        self.cache_size = cache_size

        # Create cached loading functions
        if cache_size > 0:
            self._load_image_cached = lru_cache(maxsize=cache_size)(self._load_image_uncached)
            self._load_mask_cached = lru_cache(maxsize=cache_size)(self._load_mask_uncached)
        else:
            self._load_image_cached = self._load_image_uncached
            self._load_mask_cached = self._load_mask_uncached

        # Read CSV or TXT split file
        if split_file.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(split_file)
            self.slice_ids: List[str] = df['slice_id'].tolist()
        else:
            with open(split_file, "r") as f:
                self.slice_ids: List[str] = [x.strip() for x in f if x.strip()]

        # labels
        self.case_label: Dict[str, int] = {}
        labels_csv = os.path.join(proc_root, "labels.csv")
        if os.path.exists(labels_csv):
            with open(labels_csv) as f:
                next(f)  # skip header
                for line in f:
                    if "," in line:
                        cid, lab = line.strip().split(",")
                        self.case_label[cid] = int(lab)
        # mapping slice -> case
        self.slice_case: Dict[str, str] = {}
        mapping_csv = os.path.join(proc_root, "mapping.csv")
        if os.path.exists(mapping_csv):
            with open(mapping_csv) as f:
                next(f)  # skip header
                for line in f:
                    if "," in line:
                        sid, cid = line.strip().split(",")
                        self.slice_case[sid] = cid

    def __len__(self): return len(self.slice_ids)

    def _load_image_uncached(self, sid: str):
        # Check for multi-modal structure (flair/, t1/, t1ce/, t2/ folders)
        flair_path = os.path.join(self.proc_root, "flair", f"{sid}.png")
        t1_path = os.path.join(self.proc_root, "t1", f"{sid}.png")
        t1ce_path = os.path.join(self.proc_root, "t1ce", f"{sid}.png")
        t2_path = os.path.join(self.proc_root, "t2", f"{sid}.png")

        if all(os.path.exists(p) for p in [flair_path, t1_path, t1ce_path, t2_path]):
            # Multi-modal: Load all 4 modalities and stack
            flair = np.array(Image.open(flair_path).convert("L"))
            t1 = np.array(Image.open(t1_path).convert("L"))
            t1ce = np.array(Image.open(t1ce_path).convert("L"))
            t2 = np.array(Image.open(t2_path).convert("L"))
            # Stack to (H, W, 4)
            img_array = np.stack([flair, t1, t1ce, t2], axis=-1)
            return img_array
        else:
            # Try single-modal fallback
            png_path = os.path.join(self.proc_root, "images", f"{sid}.png")
            if os.path.exists(png_path):
                return Image.open(png_path).convert("L")
            else:
                raise FileNotFoundError(f"Multi-modal images not found for {sid}")

    def _load_mask_uncached(self, sid: str) -> Image.Image:
        # Try seg/ folder first (multiclass), then masks/ (binary)
        seg_path = os.path.join(self.proc_root, "seg", f"{sid}.png")
        msk_path = os.path.join(self.proc_root, "masks", f"{sid}.png")

        if os.path.exists(seg_path):
            return Image.open(seg_path).convert("L")
        elif os.path.exists(msk_path):
            return Image.open(msk_path).convert("L")
        else:
            raise FileNotFoundError(f"Mask not found: {seg_path} or {msk_path}")

    def __getitem__(self, idx):
        sid = self.slice_ids[idx]
        img = self._load_image_cached(sid)
        msk = self._load_mask_cached(sid)

        # Check if multi-modal (numpy array) or single-modal (PIL Image)
        if isinstance(img, np.ndarray):
            # Multi-modal: img is (H, W, 4)
            # For multi-modal, augmentation is already applied during preprocessing
            # We just need to convert to tensor with correct shape
            # NOTE: Multi-modal preprocessing should be done with same resize/pad as single-modal
            img_t = torch.from_numpy(img).permute(2, 0, 1).float()  # (4, H, W)

            # Process mask - keep as class labels (0, 1, 2, ...) not binary
            msk_arr = np.asarray(msk).astype(np.int64)  # Keep as integer class labels
            # Mask values are already 0, 1, 2 from preprocessing
            # No need to threshold or normalize - just convert to tensor
            msk_t = torch.from_numpy(msk_arr).unsqueeze(0)  # (1, H, W) with integer labels
        else:
            # Single-modal: img is PIL Image
            img_t, msk_t = augment_pair(img, msk, self.img_size, self.rotate_deg, self.hflip_p, self.vflip_p, self.train)

        cid = self.slice_case.get(sid, sid.split("_")[0])
        label = self.case_label.get(cid, 0)
        return {"image": img_t, "mask": msk_t, "label": torch.tensor(label, dtype=torch.long), "slice_id": sid, "case_id": cid}
