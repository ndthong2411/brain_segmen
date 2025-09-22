import os, sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from braintumnet.utils.io import load_yaml
from braintumnet.data.brats2020_dataset import SliceDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    cfg = load_yaml(args.cfg)

    proc = cfg["data"]["proc_root"]
    split = os.path.join(proc, f"split_train_fold{args.fold}.txt")
    ds = SliceDataset(proc, split, cfg["data"]["img_size"], train=True, in_channels=cfg["model"]["in_channels"])
    dl = DataLoader(ds, batch_size=args.n, shuffle=True)
    batch = next(iter(dl))
    imgs = batch["image"]; msks = batch["mask"]
    n = imgs.size(0)
    cols = 4
    rows = (n*2 + cols - 1)//cols
    plt.figure(figsize=(cols*3, rows*3))
    for i in range(n):
        plt.subplot(rows, cols, i*2+1); plt.imshow(imgs[i,0].numpy(), cmap="gray"); plt.axis("off"); plt.title("img")
        plt.subplot(rows, cols, i*2+2); plt.imshow(msks[i,0].numpy(), cmap="gray"); plt.axis("off"); plt.title("mask")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
