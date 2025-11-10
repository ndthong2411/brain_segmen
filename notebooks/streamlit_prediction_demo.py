#!/usr/bin/env python3
"""Streamlit gallery to showcase BrainTumNet predictions alongside baseline models."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

# Disable Streamlit's module watcher that conflicts with torch.classes (must happen before importing Streamlit)
os.environ.setdefault("STREAMLIT_SERVER_FILEWATCHERTYPE", "none")

import streamlit as st
import torch

# Make the BrainTumNet source importable without installing the package.
ROOT = Path(__file__).resolve().parents[1]
BRAIN_ROOT = ROOT / "braintumnet"
SRC_ROOT = BRAIN_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from braintumnet.models.braintumnet import BrainTumNet  # type: ignore  # pylint: disable=wrong-import-position
from braintumnet.models.braintumnet_v2 import BrainTumNetV2  # type: ignore  # pylint: disable=wrong-import-position
from braintumnet.models.lg_unetr_wrapper import LGUNETRWrapper  # type: ignore  # pylint: disable=wrong-import-position
from braintumnet.models.nnunet_wrapper import nnUNetWrapper  # type: ignore  # pylint: disable=wrong-import-position
from braintumnet.models.transunet_wrapper import TransUNetWrapper  # type: ignore  # pylint: disable=wrong-import-position
from braintumnet.models.unetr_wrapper import UNETRWrapper  # type: ignore  # pylint: disable=wrong-import-position
from braintumnet.utils.io import load_ckpt, load_yaml  # type: ignore  # pylint: disable=wrong-import-position
from braintumnet.data.transforms import resize_pad_to_square  # type: ignore  # pylint: disable=wrong-import-position

# Default assets inside the repository.
DATA_ROOT = BRAIN_ROOT / "data" / "test_4class"
SEG_DIR = DATA_ROOT / "seg"
MODALITY_DIRS: Dict[str, Path] = {
    "flair": DATA_ROOT / "flair",
    "t1": DATA_ROOT / "t1",
    "t1ce": DATA_ROOT / "t1ce",
    "t2": DATA_ROOT / "t2",
}
DISPLAY_MODALITIES = list(MODALITY_DIRS.keys())
STACK_MODALITIES = ("flair", "t1", "t1ce", "t2")
BASE_CFG_PATH = BRAIN_ROOT / "configs" / "base.yaml"
CHECKPOINT_ROOT = BRAIN_ROOT / "checkpoints"

CLASS_COLORS = {
    0: (0, 0, 0),       # background
    1: (255, 0, 0),     # NCR/NET
    2: (0, 255, 0),     # ED
    3: (0, 128, 255),   # ET
}
DEFAULT_MASK_COLOR = (255, 0, 0)

MODEL_REGISTRY = [
    {
        "id": "lg_unetr",
        "label": "LG-UNETR",
        "cfg": BRAIN_ROOT / "configs" / "models" / "lg_unetr.yaml",
        "ckpt": CHECKPOINT_ROOT / "lg_unetr" / "lg_unetr_fold3" / "braintumnet_best_fold3.pth",
    },
    {
        "id": "unetr",
        "label": "UNETR",
        "cfg": BRAIN_ROOT / "configs" / "models" / "unetr.yaml",
        "ckpt": CHECKPOINT_ROOT / "unetr" / "unetr_fold3" / "braintumnet_best_fold3.pth",
    },
    {
        "id": "transunet",
        "label": "TransUNet",
        "cfg": BRAIN_ROOT / "configs" / "models" / "transunet.yaml",
        "ckpt": CHECKPOINT_ROOT / "transunet" / "transunet_fold3" / "braintumnet_best_fold3.pth",
    },
    {
        "id": "nnunet",
        "label": "nnU-Net",
        "cfg": BRAIN_ROOT / "configs" / "models" / "nnunet.yaml",
        "ckpt": CHECKPOINT_ROOT / "nnunet" / "nnunet_fold4" / "braintumnet_best_fold4.pth",
    },
    {
        "id": "segunetv2",
        "label": "SegUNetV2",
        "cfg": BASE_CFG_PATH,
        "ckpt": CHECKPOINT_ROOT / "segunetv2" / "segunetv2_fold3" / "braintumnet_best_fold3.pth",
    },
    {
        "id": "v2",
        "label": "BrainTumNet V2",
        "cfg": BRAIN_ROOT / "configs" / "models" / "ourv2.yaml",
        "ckpt": CHECKPOINT_ROOT / "v2" / "v2_fold3" / "braintumnet_best_fold3.pth",
    },
]

MODEL_GRID_ROWS = [
    {"label": "Brain", "kind": "brain"},
    {"label": "Ground Truth", "kind": "ground_truth"},
] + [{"label": entry["label"], "kind": "model", "model_id": entry["id"]} for entry in MODEL_REGISTRY]


@dataclass(frozen=True)
class ModelColumn:
    label: str
    kind: str  # currently "braintumnet"
    style: str | None = None
    cfg: Path | None = None
    ckpt: Path | None = None
    model_id: str | None = None


@st.cache_data(show_spinner=False)
def collect_slice_names() -> List[str]:
    file_sets: List[set[str]] = []
    for path in list(MODALITY_DIRS.values()) + [SEG_DIR]:
        if not path.exists():
            continue
        file_sets.append({p.name for p in path.glob("*.png")})
    if not file_sets:
        return []
    common = set.intersection(*file_sets)
    return sorted(common)




def load_modality_image(modality: str, slice_name: str, size: int) -> Image.Image:
    img_path = MODALITY_DIRS[modality] / slice_name
    if not img_path.exists():
        raise FileNotFoundError(f"Missing slice {slice_name} for modality {modality}")
    img = Image.open(img_path).convert("L")
    return resize_pad_to_square(img, size, is_mask=False)


def normalize_label_mask(mask: np.ndarray) -> np.ndarray:
    """Convert grayscale mask (0-255) to class indices 0..3."""
    max_val = int(mask.max()) if mask.size else 0
    if max_val <= len(CLASS_COLORS) - 1:
        return mask.astype(np.uint8)
    scale = max(1, round(255 / max(len(CLASS_COLORS) - 1, 1)))
    normalized = np.round(mask / scale).clip(0, len(CLASS_COLORS) - 1)
    return normalized.astype(np.uint8)


def load_ground_truth_mask(slice_name: str, size: int) -> np.ndarray:
    mask_path = SEG_DIR / slice_name
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing ground-truth mask for {slice_name}")
    mask = Image.open(mask_path).convert("L")
    mask = resize_pad_to_square(mask, size, is_mask=True)
    mask_arr = np.asarray(mask, dtype=np.uint8)
    return normalize_label_mask(mask_arr)


def dice_score(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred = pred_mask > 0
    target = target_mask > 0
    inter = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def prepare_input_tensor(slice_name: str, size: int, device: str) -> torch.Tensor:
    channels: List[torch.Tensor] = []
    for modality in STACK_MODALITIES:
        img = load_modality_image(modality, slice_name, size)
        arr = np.asarray(img, dtype=np.float32)
        channels.append(torch.from_numpy(arr))
    tensor = torch.stack(channels, dim=0).unsqueeze(0)  # (1,4,H,W)
    return tensor.to(device)


def _extract_state_dict(raw: Dict) -> Dict:
    """Return the actual model state dict regardless of wrapper format."""
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    return raw


def infer_arch_from_checkpoint(ckpt_path: str) -> Dict[str, int]:
    """Peek into checkpoint weights to recover channel counts automatically."""
    meta: Dict[str, int] = {}
    try:
        raw = torch.load(ckpt_path, map_location="cpu")
        state = _extract_state_dict(raw)
    except Exception:  # pylint: disable=broad-except
        return meta

    def shape_of(key: str):
        tensor = state.get(key)
        return tensor.shape if tensor is not None else None

    w = shape_of("seg.e1.block.0.0.weight")
    if w is not None:
        meta["base"] = int(w[0])
        meta["in_channels"] = int(w[1])

    head = shape_of("seg.head.weight")
    if head is not None:
        meta["num_classes_seg"] = int(head[0])

    bottleneck = shape_of("seg.bottleneck_conv.0.weight")
    if bottleneck is not None:
        meta["dim"] = int(bottleneck[0])

    tr = shape_of("seg.tr_upsample.weight")
    if tr is not None:
        meta["patch_size"] = int(tr[2])

    mask_gen2 = shape_of("seg.amt.mask_gen.mlp.2.weight")
    if mask_gen2 is not None:
        meta["n_heads"] = int(mask_gen2[0])

    blocks = {k.split(".")[3] for k in state.keys() if k.startswith("seg.amt.blocks.")}
    if blocks:
        meta["depth"] = len(blocks)

    aux = any("aux_head" in k for k in state.keys())
    meta["deep_supervision"] = 1 if aux else 0

    meta["_state_dict"] = state
    return meta



def build_model_from_config(mcfg: Dict, device: str):
    common_kwargs = dict(
        in_ch=mcfg.get("in_channels", len(STACK_MODALITIES)),
        num_cls=mcfg.get("num_classes_cls", 2),
        base=mcfg.get("base", 64),
        dim=mcfg.get("dim", 512),
        patch=mcfg.get("patch_size", 8),
        depth=mcfg.get("depth", 4),
        n_heads=mcfg.get("n_heads", 8),
        roi_stop_grad=mcfg.get("roi_stop_grad", True),
        deep_supervision=mcfg.get("deep_supervision", False),
        num_classes_seg=mcfg.get("num_classes_seg", 1),
    )

    model_type = mcfg.get("model_type", "v1").lower()
    if "v2" in model_type:
        model = BrainTumNetV2(
            **common_kwargs,
            dropout=mcfg.get("dropout", 0.15),
            multi_scale_fusion=mcfg.get("multi_scale_fusion", True),
            boundary_refinement=mcfg.get("boundary_refinement", False),
            use_multiscale_transformer=mcfg.get("use_multiscale_transformer", False),
            use_attention_gates=mcfg.get("use_attention_gates", False),
        )
    elif "nnunet" in model_type:
        model = nnUNetWrapper(
            in_ch=mcfg.get("in_channels", len(STACK_MODALITIES)),
            num_classes_seg=mcfg.get("num_classes_seg", 4),
            base=mcfg.get("base", 32),
            deep_supervision=mcfg.get("deep_supervision", True),
        )
    elif "lg_unetr" in model_type:
        model = LGUNETRWrapper(
            in_ch=mcfg.get("in_channels", len(STACK_MODALITIES)),
            num_classes_seg=mcfg.get("num_classes_seg", 4),
            base=mcfg.get("base", 32),
            num_levels=mcfg.get("num_levels", 4),
            embed_dim=mcfg.get("embed_dim", 384),
            depth=mcfg.get("depth", 12),
            num_heads=mcfg.get("num_heads", 6),
        )
    elif "unetr" in model_type:
        model = UNETRWrapper(
            in_ch=mcfg.get("in_channels", len(STACK_MODALITIES)),
            num_classes_seg=mcfg.get("num_classes_seg", 4),
            img_size=mcfg.get("img_size", 256),
            hidden_size=mcfg.get("hidden_size", 768),
            feature_size=mcfg.get("feature_size", 16),
            num_heads=mcfg.get("num_heads", 12),
            mlp_dim=mcfg.get("mlp_dim"),
            dropout_rate=mcfg.get("dropout_rate", 0.1),
        )
    elif "transunet" in model_type:
        model = TransUNetWrapper(
            in_ch=mcfg.get("in_channels", len(STACK_MODALITIES)),
            num_classes_seg=mcfg.get("num_classes_seg", 4),
            img_size=mcfg.get("img_size", 256),
            embed_dim=mcfg.get("embed_dim", 768),
            depth=mcfg.get("depth", 12),
            num_heads=mcfg.get("num_heads", 12),
            base=mcfg.get("base", 64),
        )
    else:
        model = BrainTumNet(**common_kwargs)

    return model.to(device)


@st.cache_resource(show_spinner=False)
def _resolve_cfg_path(cfg_path: str | Path) -> Path:
    path = Path(cfg_path)
    if not path.is_absolute():
        candidate = BRAIN_ROOT / path
        if candidate.exists():
            return candidate
    return path


def _merge_dict(base: Dict, override: Dict) -> Dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_with_base(cfg_path: str | Path) -> Dict:
    base_cfg = load_yaml(str(BRAIN_ROOT / "configs" / "base.yaml"))
    resolved = _resolve_cfg_path(cfg_path)
    if resolved.exists():
        model_cfg = load_yaml(str(resolved))
        return _merge_dict(base_cfg, model_cfg)
    return base_cfg


def load_braintumnet_bundle(cfg_path: str, ckpt_path: str, device: str) -> Dict:
    cfg = _load_config_with_base(cfg_path)
    mcfg = dict(cfg["model"])
    meta = infer_arch_from_checkpoint(ckpt_path)

    overrides = {}
    for key in ("in_channels", "base", "dim", "patch_size", "depth", "n_heads", "num_classes_seg"):
        if key in meta:
            mcfg[key] = meta[key]
            overrides[key] = meta[key]

    if "deep_supervision" in meta:
        mcfg["deep_supervision"] = bool(meta["deep_supervision"])
    state = meta.get("_state_dict")

    model = build_model_from_config(mcfg, device)
    if state is not None:
        model.load_state_dict(state, strict=True)
        state = None
    else:
        load_ckpt(model, ckpt_path, map_location=device)
    model.eval()
    meta.pop("_state_dict", None)
    return {
        "model": model,
        "img_size": cfg["data"]["img_size"],
        "num_seg_classes": mcfg.get("num_classes_seg", 1),
        "overrides": overrides,
    }


def run_model_inference(bundle: Dict, slice_name: str, device: str) -> Tuple[np.ndarray, np.ndarray, str | None, float | None]:
    model = bundle["model"]
    img_size: int = bundle["img_size"]
    tensor = prepare_input_tensor(slice_name, img_size, device)

    with torch.no_grad():
        output = model(tensor)

    if isinstance(output, tuple):
        seg_logits = output[0]
        cls_logits = None
        if len(output) > 1:
            cls_candidate = output[1]
            if isinstance(cls_candidate, torch.Tensor):
                cls_logits = cls_candidate
    else:
        seg_logits = output
        cls_logits = None

    if seg_logits.shape[1] == 1:
        seg_prob = torch.sigmoid(seg_logits)
        seg_mask = (seg_prob >= 0.5).long()
    else:
        seg_soft = torch.softmax(seg_logits, dim=1)
        seg_prob = seg_soft[:, 1:, :, :].sum(dim=1, keepdim=True)
        seg_mask = torch.argmax(seg_soft, dim=1, keepdim=False).unsqueeze(1)

    seg_np = seg_prob.squeeze().cpu().numpy()
    seg_mask_np = seg_mask.squeeze().cpu().numpy().astype(np.uint8)

    cls_name: str | None = None
    cls_conf: float | None = None
    if cls_logits is not None:
        cls_prob = torch.softmax(cls_logits, dim=1).squeeze().cpu().numpy()
        cls_idx = int(cls_prob.argmax())
        cls_name = "HGG" if cls_idx == 0 else "LGG"
        cls_conf = float(cls_prob[cls_idx])

    return seg_np, seg_mask_np, cls_name, cls_conf


def prob_to_mask(prob_map: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    mask = (prob_map >= thresh).astype(np.uint8) * 255
    return mask


def colorize_mask(mask: np.ndarray, palette: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in palette.items():
        rgb[mask == cls_id] = color
    return rgb


def overlay_mask(base_img: Image.Image, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.array(base_img.convert("RGB")).astype(np.float32)
    mask = mask_rgb.astype(np.float32)
    mask_alpha = (mask.sum(axis=2, keepdims=True) > 0).astype(np.float32)
    blended = base * (1 - alpha * mask_alpha) + mask * (alpha * mask_alpha)
    return blended.clip(0, 255).astype(np.uint8)


def render_header(columns: Sequence[ModelColumn]):
    titles = ["MRI", "Ground Truth"] + [col.label for col in columns]
    cols = st.columns(len(titles))
    for col, title in zip(cols, titles, strict=True):
        col.markdown(f"**{title}**")


def render_row(
    slice_name: str,
    display_modality: str,
    columns: Sequence[ModelColumn],
    bundles: Dict[str, Dict],
    device: str,
    img_size: int,
):
    gt_mask = load_ground_truth_mask(slice_name, img_size)
    display_img = load_modality_image(display_modality, slice_name, img_size)
    row_cols = st.columns(len(columns) + 2)
    row_cols[0].image(display_img, caption=f"{slice_name} | {display_modality.upper()}", use_container_width=True, clamp=True)
    gt_rgb = colorize_mask(gt_mask, CLASS_COLORS)
    gt_overlay = overlay_mask(display_img, gt_rgb)
    row_cols[1].image(gt_overlay, caption="Ground Truth", clamp=True, use_container_width=True)

    for idx, col_cfg in enumerate(columns, start=2):
        if col_cfg.kind == "braintumnet":
            model_id = col_cfg.model_id
            bundle = bundles.get(model_id or "")
            if bundle is None:
                row_cols[idx].warning("Model unavailable")
                continue
            seg_prob, seg_mask, cls_name, cls_conf = run_model_inference(bundle, slice_name, device)
            pred_mask = prob_to_mask(seg_prob)
            dice = dice_score(pred_mask, gt_mask)
            seg_rgb = colorize_mask(seg_mask, CLASS_COLORS if seg_mask.max() > 1 else {0: (0, 0, 0), 1: DEFAULT_MASK_COLOR})
            seg_overlay = overlay_mask(display_img, seg_rgb)
            row_cols[idx].image(seg_overlay, clamp=True, use_container_width=True)
            if cls_name is None:
                row_cols[idx].caption(f"Dice {dice:.3f}")
            else:
                row_cols[idx].caption(f"Dice {dice:.3f} | {cls_name} ({cls_conf:.2f})")
        else:
            row_cols[idx].warning("Unsupported column type")


def render_comparison_grid(
    slice_names: List[str],
    display_modality: str,
    model_bundles: Dict[str, Dict],
    device: str,
    img_size: int,
):
    if not slice_names:
        st.info("Choose at least one slice to render the comparison grid.")
        return

    st.subheader("Comparison Grid")
    st.caption("Rows show each model; columns show different slices for a quick qualitative scan.")

    base_cache: Dict[str, Dict] = {}
    pred_cache: Dict[Tuple[str, str], Dict] = {}

    def get_base(slice_name: str) -> Dict:
        if slice_name not in base_cache:
            img = load_modality_image(display_modality, slice_name, img_size)
            gt_mask = load_ground_truth_mask(slice_name, img_size)
            gt_rgb = colorize_mask(gt_mask, CLASS_COLORS)
            gt_overlay = overlay_mask(img, gt_rgb)
            base_cache[slice_name] = {
                "image": img,
                "gt_mask": gt_mask,
                "gt_overlay": gt_overlay,
            }
        return base_cache[slice_name]

    def get_prediction(model_id: str, slice_name: str) -> Dict:
        key = (model_id, slice_name)
        if key not in pred_cache:
            bundle = model_bundles.get(model_id)
            if bundle is None:
                pred_cache[key] = {"overlay": None, "dice": None}
            else:
                seg_prob, seg_mask, _, _ = run_model_inference(bundle, slice_name, device)
                base = get_base(slice_name)
                palette = CLASS_COLORS if seg_mask.max() > 1 else {0: (0, 0, 0), 1: DEFAULT_MASK_COLOR}
                seg_rgb = colorize_mask(seg_mask, palette)
                overlay = overlay_mask(base["image"], seg_rgb)
                dice = dice_score(prob_to_mask(seg_prob), base["gt_mask"])
                pred_cache[key] = {"overlay": overlay, "dice": dice}
        return pred_cache[key]

    header_cols = st.columns(len(slice_names) + 1)
    header_cols[0].markdown("**Row**")
    for idx, slice_name in enumerate(slice_names, start=1):
        header_cols[idx].markdown(f"**{idx}**<br/><sub>{slice_name}</sub>", unsafe_allow_html=True)

    for row_cfg in MODEL_GRID_ROWS:
        cols = st.columns(len(slice_names) + 1)
        cols[0].markdown(f"**{row_cfg['label']}**")
        for idx, slice_name in enumerate(slice_names, start=1):
            base = get_base(slice_name)
            if row_cfg["kind"] == "brain":
                cols[idx].image(base["image"], use_container_width=True, clamp=True)
            elif row_cfg["kind"] == "ground_truth":
                cols[idx].image(base["gt_overlay"], use_container_width=True, clamp=True)
            elif row_cfg["kind"] == "model":
                pred = get_prediction(row_cfg["model_id"], slice_name)
                if pred["overlay"] is None:
                    cols[idx].warning("N/A")
                else:
                    cols[idx].image(pred["overlay"], use_container_width=True, clamp=True)
                    cols[idx].caption(f"Dice {pred['dice']:.3f}")
            else:
                cols[idx].write("—")


def main():
    st.set_page_config(page_title="BrainTumNet Prediction Demo", layout="wide")
    st.title("Streamlit Demo · Brain Tumor Segmentation")
    st.caption("Visualize MRI slices with segmentation masks from multiple models.")

    slice_names = collect_slice_names()
    if not slice_names:
        st.error("Could not find prepared PNG slices. Please run preprocessing first.")
        return

    st.sidebar.header("Display Settings")
    display_modality = st.sidebar.selectbox("MRI modality to show", DISPLAY_MODALITIES, index=2)

    max_rows = min(8, len(slice_names))
    num_rows = st.sidebar.slider("Slices per view", min_value=1, max_value=max_rows, value=min(4, max_rows))
    start_idx = st.sidebar.number_input(
        "Start index",
        min_value=0,
        max_value=max(0, len(slice_names) - num_rows),
        value=0,
        step=1,
    )
    selected = slice_names[start_idx : start_idx + num_rows]

    model_labels = {entry["id"]: entry["label"] for entry in MODEL_REGISTRY}
    primary_options = list(model_labels.keys())
    default_index = max(0, len(primary_options) - 1)
    primary_model_id = st.sidebar.selectbox(
        "Primary model for slice view",
        options=primary_options,
        index=default_index,
        format_func=lambda mid: model_labels.get(mid, mid),
    )
    use_cuda = st.sidebar.checkbox("Use CUDA (if available)", value=torch.cuda.is_available())
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    model_bundles: Dict[str, Dict] = {}
    load_errors: List[str] = []
    for entry in MODEL_REGISTRY:
        cfg_path = str(entry["cfg"])
        ckpt_path = str(entry["ckpt"])
        try:
            model_bundles[entry["id"]] = load_braintumnet_bundle(cfg_path, ckpt_path, device)
        except FileNotFoundError as exc:
            load_errors.append(f"{entry['label']}: {exc}")
        except RuntimeError as exc:
            load_errors.append(f"{entry['label']}: {exc}")

    if load_errors:
        st.warning("Some models could not be loaded:\n" + "\n".join(f"- {msg}" for msg in load_errors))
    if not model_bundles:
        st.error("No models were loaded successfully.")
        return
    if primary_model_id not in model_bundles:
        st.error(f"Primary model {model_labels.get(primary_model_id, primary_model_id)} is unavailable.")
        return

    st.sidebar.success(f"Loaded {len(model_bundles)} models on {device.upper()}")
    st.info("Comparison grid includes LG-UNETR, UNETR, TransUNet, nnU-Net, SegUNetV2, and BrainTumNet V2 (bottom).")

    primary_columns: Tuple[ModelColumn, ...] = (
        ModelColumn(model_labels.get(primary_model_id, primary_model_id), "braintumnet", model_id=primary_model_id),
    )

    render_header(primary_columns)
    for slice_name in selected:
        render_row(slice_name, display_modality, primary_columns, model_bundles, device, model_bundles[primary_model_id]["img_size"])

    available_grid = max(1, len(slice_names) - start_idx)
    grid_cols = st.slider(
        "Slices in comparison grid",
        min_value=1,
        max_value=min(10, available_grid),
        value=min(5, available_grid),
        help="Controls how many slice columns appear in the comparison grid below.",
    )
    grid_slices = slice_names[start_idx : start_idx + grid_cols]
    render_comparison_grid(grid_slices, display_modality, model_bundles, device, model_bundles[primary_model_id]["img_size"])

    st.markdown(
        """
        **Tips**

        - Update `MODEL_REGISTRY` in `notebooks/streamlit_prediction_demo.py` to plug in additional checkpoints or reorder models.
        - Use `streamlit run notebooks/streamlit_prediction_demo.py` inside the project root to launch this demo.
        """
    )


if __name__ == "__main__":
    main()
