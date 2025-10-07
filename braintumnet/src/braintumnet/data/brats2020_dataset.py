import os
from typing import List, Dict
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
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
                 train: bool=True, in_channels: int=1):
        self.proc_root = proc_root
        self.train = train
        self.img_size = img_size
        self.rotate_deg, self.hflip_p, self.vflip_p = rotate_deg, hflip_p, vflip_p
        self.in_channels = in_channels
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

    def _load_image(self, sid: str):
        # Try multi-modal (.npy) first, then single-modal (.png)
        npy_path = os.path.join(self.proc_root, "images", f"{sid}.npy")
        png_path = os.path.join(self.proc_root, "images", f"{sid}.png")

        if os.path.exists(npy_path):
            # Multi-modal: Load 4-channel numpy array
            img_array = np.load(npy_path)  # Shape: (H, W, 4)
            return img_array
        elif os.path.exists(png_path):
            # Single-modal: Load grayscale PNG
            return Image.open(png_path).convert("L")
        else:
            raise FileNotFoundError(f"Neither {npy_path} nor {png_path} found")

    def _load_mask(self, sid: str) -> Image.Image:
        msk_path = os.path.join(self.proc_root, "masks", f"{sid}.png")
        if not os.path.exists(msk_path):
            raise FileNotFoundError(msk_path)
        return Image.open(msk_path).convert("L")

    def __getitem__(self, idx):
        sid = self.slice_ids[idx]
        img = self._load_image(sid)
        msk = self._load_mask(sid)

        # Check if multi-modal (numpy array) or single-modal (PIL Image)
        if isinstance(img, np.ndarray):
            # Multi-modal: img is (H, W, 4)
            # For multi-modal, augmentation is already applied during preprocessing
            # We just need to convert to tensor with correct shape
            # NOTE: Multi-modal preprocessing should be done with same resize/pad as single-modal
            img_t = torch.from_numpy(img).permute(2, 0, 1).float()  # (4, H, W)

            # Still need to process mask
            msk_arr = np.asarray(msk).astype(np.float32)
            if msk_arr.max() > 1.0:
                msk_arr /= 255.0
            msk_t = torch.from_numpy(msk_arr > 0.5).float().unsqueeze(0)  # (1, H, W)
        else:
            # Single-modal: img is PIL Image
            img_t, msk_t = augment_pair(img, msk, self.img_size, self.rotate_deg, self.hflip_p, self.vflip_p, self.train)

        cid = self.slice_case.get(sid, sid.split("_")[0])
        label = self.case_label.get(cid, 0)
        return {"image": img_t, "mask": msk_t, "label": torch.tensor(label, dtype=torch.long), "slice_id": sid, "case_id": cid}
