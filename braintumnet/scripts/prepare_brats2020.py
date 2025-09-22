import os, argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # braintumnet/
sys.path.append(str(ROOT / "src"))

from braintumnet.data.preprocessing import process_brats2020, make_folds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Path to BraTS2020 TrainingData root (contains HGG/LGG)")
    ap.add_argument("--out", required=True, help="Output processed dir (e.g., braintumnet/data/processed)")
    ap.add_argument("--modality", default="t1ce", choices=["t1","t1ce","t2","flair"])
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--slices_per_case", type=int, default=20)
    ap.add_argument("--tumor_slice_ratio", type=float, default=0.7)
    ap.add_argument("--num_folds", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    process_brats2020(args.raw, args.out, args.modality, args.img_size, args.slices_per_case, args.tumor_slice_ratio)
    make_folds(args.out, args.num_folds)

if __name__ == "__main__":
    main()
