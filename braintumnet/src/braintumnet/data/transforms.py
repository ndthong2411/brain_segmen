import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch

def resize_pad_to_square(img: Image.Image, size: int, is_mask: bool=False) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    pad_l = (s - w) // 2
    pad_t = (s - h) // 2
    pad_r = s - w - pad_l
    pad_b = s - h - pad_t
    fill = 0
    if is_mask:
        img = TF.pad(img, [pad_l, pad_t, pad_r, pad_b], fill=0)
        img = img.resize((size, size), Image.NEAREST)
    else:
        img = TF.pad(img, [pad_l, pad_t, pad_r, pad_b], fill=0)
        img = img.resize((size, size), Image.BILINEAR)
    return img

def to_tensor01(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32)
    if arr.max() > 1.0: arr /= 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)

def augment_pair(img: Image.Image, msk: Image.Image, img_size: int,
                 rotate_deg: int=30, hflip_p: float=0.5, vflip_p: float=0.5,
                 train: bool=True):
    img = resize_pad_to_square(img, img_size, is_mask=False)
    msk = resize_pad_to_square(msk, img_size, is_mask=True)
    if train:
        angle = random.uniform(-rotate_deg, rotate_deg)
        img = TF.rotate(img, angle)
        msk = TF.rotate(msk, angle)
        if random.random() < hflip_p:
            img = TF.hflip(img); msk = TF.hflip(msk)
        if random.random() < vflip_p:
            img = TF.vflip(img); msk = TF.vflip(msk)
    return to_tensor01(img), (torch.from_numpy((np.asarray(msk)>127).astype(np.float32)).unsqueeze(0))
