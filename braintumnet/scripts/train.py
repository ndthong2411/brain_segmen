import os, argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # braintumnet/
sys.path.append(str(ROOT / "src"))

from braintumnet.utils.io import load_yaml
from braintumnet.utils.seed import set_seed
from braintumnet.engine.trainer import train_one_fold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--fold", type=int, default=None, help="0..K-1. If not set, uses cfg.data.fold")
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold

    set_seed(42, deterministic=False)
    best_iou = train_one_fold(cfg, cfg["data"]["fold"])
    print("Best IoU:", best_iou)

if __name__ == "__main__":
    main()
