import os, argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from braintumnet.utils.io import load_yaml
from braintumnet.engine.evaluator import evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--fold", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold

    evaluate(cfg, cfg["data"]["fold"], args.ckpt)

if __name__ == "__main__":
    main()
