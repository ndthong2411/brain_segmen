"""
Dataset Factory for automatic backend selection
================================================

Automatically selects the appropriate dataset class based on configuration.

Backends:
- "png": Standard PNG files (SliceDataset)
- "lmdb": LMDB database (LMDBDataset) - 10x faster

Usage:
    from braintumnet.data.dataset_factory import create_dataset

    train_ds = create_dataset(
        backend="lmdb",  # or "png"
        data_root="path/to/data",
        split_file="path/to/split.csv",
        cfg=config_dict,
        train=True
    )
"""

from typing import Dict, Any
from .brats2020_dataset import SliceDataset
from .lmdb_dataset import LMDBDataset


def create_dataset(
    backend: str,
    data_root: str,
    split_file: str,
    cfg: Dict[str, Any],
    train: bool = True
):
    """Create dataset with specified backend.

    Args:
        backend: "png" or "lmdb"
        data_root: Path to data directory
        split_file: Path to split file (CSV or TXT)
        cfg: Config dictionary
        train: Whether this is training set

    Returns:
        dataset: Dataset instance (SliceDataset or LMDBDataset)

    Raises:
        ValueError: If backend is not supported
    """
    # Extract common parameters from config
    img_size = cfg["data"]["img_size"]
    in_channels = cfg["model"]["in_channels"]

    if train:
        # Training: use augmentation
        rotate_deg = cfg["augment"]["rotate_deg"]
        hflip_p = cfg["augment"]["hflip_p"]
        vflip_p = cfg["augment"]["vflip_p"]
    else:
        # Validation: no augmentation
        rotate_deg = 0
        hflip_p = 0.0
        vflip_p = 0.0

    # Select backend
    if backend == "png":
        print(f"  Using PNG backend (SliceDataset)")
        cache_size = cfg["data"].get("cache_size", 1000)  # Default 1000 samples cached
        return SliceDataset(
            proc_root=data_root,
            split_file=split_file,
            img_size=img_size,
            rotate_deg=rotate_deg,
            hflip_p=hflip_p,
            vflip_p=vflip_p,
            train=train,
            in_channels=in_channels,
            cache_size=cache_size
        )

    elif backend == "lmdb":
        print(f"  Using LMDB backend (LMDBDataset) - Fast loading!")
        return LMDBDataset(
            lmdb_root=data_root,
            split_file=split_file,
            img_size=img_size,
            rotate_deg=rotate_deg,
            hflip_p=hflip_p,
            vflip_p=vflip_p,
            train=train,
            in_channels=in_channels
        )

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Supported backends: 'png', 'lmdb'"
        )


def get_data_root(cfg: Dict[str, Any]) -> str:
    """Get data root path based on backend.

    Args:
        cfg: Config dictionary

    Returns:
        data_root: Path to data directory
    """
    backend = cfg["data"].get("backend", "png")

    if backend == "png":
        return cfg["data"]["proc_root"]
    elif backend == "lmdb":
        return cfg["data"].get("lmdb_root", cfg["data"]["proc_root"])
    else:
        raise ValueError(f"Unknown backend: {backend}")
