"""
Model factory for BrainTumNet

Supports multiple architectures:
- segunetv2/v2: BrainTumNetV2 (baseline enhanced)
- swin_unetr: Swin-UNETR (MONAI)
- nnunet: nnU-Net style architecture
- unetr: UNETR (MONAI)
- transunet: TransUNet (ResNet + ViT)
- lg_unetr: Local-Global UNETR (CNN + Transformer dual-path)
"""

from typing import Dict


def build_model(cfg: Dict):
    """
    Factory function to build model based on config

    Args:
        cfg: Configuration dictionary with 'model' section

    Returns:
        nn.Module: Model instance
    """
    mcfg = cfg["model"]
    model_type = mcfg.get("model_type", "segunetv2").lower()

    # Common parameters
    in_ch = mcfg["in_channels"]
    num_cls = mcfg.get("num_classes_cls", 2)
    num_classes_seg = mcfg.get("num_classes_seg", 3)

    if model_type in ["segunetv2", "v2"]:
        from .braintumnet_v2 import BrainTumNetV2
        return BrainTumNetV2(
            in_ch=in_ch,
            num_cls=num_cls,
            base=mcfg["base"],
            dim=mcfg["dim"],
            patch=mcfg["patch_size"],
            depth=mcfg["depth"],
            n_heads=mcfg["n_heads"],
            num_classes_seg=num_classes_seg,
            dropout=mcfg.get("dropout", 0.15),
            roi_stop_grad=mcfg.get("roi_stop_grad", True),
            deep_supervision=mcfg.get("deep_supervision", True),
            multi_scale_fusion=mcfg.get("multi_scale_fusion", True),
            boundary_refinement=mcfg.get("boundary_refinement", False),
            use_multiscale_transformer=mcfg.get("use_multiscale_transformer", False),
            use_attention_gates=mcfg.get("use_attention_gates", False),
        )

    elif model_type == "swin_unetr":
        from .swin_unetr_wrapper import SwinUNETRWrapper
        return SwinUNETRWrapper(
            in_ch=in_ch,
            num_classes_seg=num_classes_seg,
            feature_size=mcfg.get("feature_size", 48),
            img_size=mcfg.get("img_size", 256),
            use_checkpoint=mcfg.get("use_checkpoint", True),
        )

    elif model_type == "nnunet":
        from .nnunet_wrapper import nnUNetWrapper
        return nnUNetWrapper(
            in_ch=in_ch,
            num_classes_seg=num_classes_seg,
            base=mcfg.get("base", 32),
            deep_supervision=mcfg.get("deep_supervision", True),
        )

    elif model_type == "unetr":
        from .unetr_wrapper import UNETRWrapper
        return UNETRWrapper(
            in_ch=in_ch,
            num_classes_seg=num_classes_seg,
            img_size=mcfg.get("img_size", 256),
            hidden_size=mcfg.get("hidden_size", 768),
            feature_size=mcfg.get("feature_size", 16),
            num_heads=mcfg.get("num_heads", 12),
        )

    elif model_type == "transunet":
        from .transunet_wrapper import TransUNetWrapper
        return TransUNetWrapper(
            in_ch=in_ch,
            num_classes_seg=num_classes_seg,
            img_size=mcfg.get("img_size", 256),
            embed_dim=mcfg.get("embed_dim", 768),
            depth=mcfg.get("depth", 12),
            num_heads=mcfg.get("num_heads", 12),
            base=mcfg.get("base", 64),
        )

    elif model_type == "lg_unetr":
        from .lg_unetr_wrapper import LGUNETRWrapper
        return LGUNETRWrapper(
            in_ch=in_ch,
            num_classes_seg=num_classes_seg,
            base=mcfg.get("base", 32),
            num_levels=mcfg.get("num_levels", 4),
            embed_dim=mcfg.get("embed_dim", 384),
            depth=mcfg.get("depth", 12),
            num_heads=mcfg.get("num_heads", 6),
        )

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Supported: segunetv2, swin_unetr, nnunet, unetr, transunet, lg_unetr"
        )


__all__ = ["build_model"]
