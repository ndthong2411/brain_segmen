import os, glob, csv
from typing import Tuple, List
import numpy as np
from PIL import Image
import nibabel as nib

def _rescale01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    nz = arr > 0
    if nz.sum() > 0:
        a = arr[nz]
        lo, hi = a.min(), a.max()
    else:
        lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    out[~np.isfinite(out)] = 0
    return out

def _save_png01(x: np.ndarray, path: str):
    x = (x * 255.0).clip(0,255).astype(np.uint8)
    Image.fromarray(x).save(path)

def _find_modal_paths(case_dir: str, modality: str) -> Tuple[str, str]:
    # modality file names contain: t1, t1ce, t2, flair
    patt = f"*{modality}*.nii*"
    img = glob.glob(os.path.join(case_dir, patt))
    seg = glob.glob(os.path.join(case_dir, "*seg*.nii*"))
    if len(img)==0 or len(seg)==0:
        return None, None
    return img[0], seg[0]

def _pick_slices(seg3d: np.ndarray, k: int) -> List[int]:
    zs = np.where(seg3d.sum(axis=(0,1)) > 0)[0]
    if len(zs) == 0:
        return []
    if len(zs) <= k:
        return zs.tolist()
    idx = np.linspace(zs[0], zs[-1], k).astype(int).tolist()
    return idx

def process_brats2020(raw_root: str, out_root: str, modality: str="t1ce",
                      img_size: int=256, slices_per_case: int=20, tumor_slice_ratio: float=0.7):
    os.makedirs(os.path.join(out_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "masks"), exist_ok=True)
    labels_path = os.path.join(out_root, "labels.csv")
    mapping_path = os.path.join(out_root, "mapping.csv")

    cases = []
    for grp, lab in [("HGG", 0), ("LGG", 1)]:
        grp_dir = os.path.join(raw_root, grp)
        if not os.path.isdir(grp_dir): continue
        for case_dir in sorted(glob.glob(os.path.join(grp_dir, "*"))):
            cases.append((case_dir, lab))

    with open(labels_path, "w", newline="") as lf, open(mapping_path, "w", newline="") as mf:
        lw, mw = csv.writer(lf), csv.writer(mf)
        lw.writerow(["case_id","label"])
        mw.writerow(["slice_id","case_id"])
        total_slices = 0

        for case_dir, lab in cases:
            case_id = os.path.basename(case_dir)
            lw.writerow([case_id, lab])

            img_path, seg_path = _find_modal_paths(case_dir, modality)
            if img_path is None: 
                print("Skip (missing files):", case_dir)
                continue

            img3d = nib.load(img_path).get_fdata()
            seg3d = nib.load(seg_path).get_fdata()
            img3d = _rescale01(img3d)
            wt = (seg3d > 0).astype(np.float32)

            tumor_z = np.where(wt.sum(axis=(0,1)) > 0)[0].tolist()
            non_z   = [z for z in range(img3d.shape[2]) if z not in set(tumor_z)]

            k_tum = min(len(tumor_z), int(round(slices_per_case * tumor_slice_ratio)))
            k_non = slices_per_case - k_tum
            pick_t = np.linspace(tumor_z[0], tumor_z[-1], k_tum).astype(int).tolist() if len(tumor_z)>0 and k_tum>0 else []
            if k_non > 0 and len(non_z)>0:
                # sample uniformly spread
                step = max(1, len(non_z)//k_non)
                pick_n = non_z[::step][:k_non]
            else:
                pick_n = []
            picks = sorted(set(pick_t + pick_n))

            for z in picks:
                img = img3d[:,:,z]
                msk = wt[:,:,z]
                # pad to square then resize
                h, w = img.shape
                s = max(h,w)
                pad_h = (s - h); pad_w = (s - w)
                img_p = np.pad(img, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode="constant")
                msk_p = np.pad(msk, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode="constant")
                img_p = Image.fromarray((img_p*255).astype(np.uint8)).resize((img_size,img_size), Image.BILINEAR)
                msk_p = Image.fromarray((msk_p*255).astype(np.uint8)).resize((img_size,img_size), Image.NEAREST)
                sid = f"{case_id}_{int(z):03d}"
                _save_png01(np.array(img_p).astype(np.float32)/255.0, os.path.join(out_root, "images", f"{sid}.png"))
                Image.fromarray(np.array(msk_p)).save(os.path.join(out_root, "masks", f"{sid}.png"))
                mw.writerow([sid, case_id])
                total_slices += 1
        print("Processed slices:", total_slices)

def make_folds(proc_root: str, num_folds: int=5):
    import csv, random
    labels_csv = os.path.join(proc_root, "labels.csv")
    mapping_csv = os.path.join(proc_root, "mapping.csv")
    assert os.path.exists(labels_csv) and os.path.exists(mapping_csv), "Run processing first."

    # case -> label
    case_label = {}
    with open(labels_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            case_label[row["case_id"]] = int(row["label"])
    # case -> slice_ids
    case_slices = {}
    with open(mapping_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            case_slices.setdefault(row["case_id"], []).append(row["slice_id"])

    # stratified split on cases
    cases0 = [c for c,l in case_label.items() if l==0]
    cases1 = [c for c,l in case_label.items() if l==1]
    random.seed(42)
    random.shuffle(cases0); random.shuffle(cases1)
    folds = [[] for _ in range(num_folds)]
    for i,c in enumerate(cases0): folds[i%num_folds].append(c)
    for i,c in enumerate(cases1): folds[i%num_folds].append(c)

    # write slice ids per fold
    for k in range(num_folds):
        val_cases = set(folds[k])
        tr_cases = set(case_label.keys()) - val_cases
        tr_slices, val_slices = [], []
        for cid in tr_cases: tr_slices += case_slices[cid]
        for cid in val_cases: val_slices += case_slices[cid]
        with open(os.path.join(proc_root, f"split_train_fold{k}.txt"), "w") as f: f.write("\n".join(tr_slices))
        with open(os.path.join(proc_root, f"split_val_fold{k}.txt"), "w") as f: f.write("\n".join(val_slices))
    print("Folds written:", num_folds)
