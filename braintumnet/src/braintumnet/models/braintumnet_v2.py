"""
BrainTumNet V2 - Phase 2 Enhanced Model

Improvements:
1. Uses SegUNetV2 with InstanceNorm, LeakyReLU, residuals
2. Larger capacity options
3. Better classification backbone
4. Multi-scale fusion support

Author: BrainTumNet Phase 2 Upgrade
Date: 2025-10-14
"""

import torch
import torch.nn as nn
from .seg_unet_v2 import SegUNetV2
from .t_inception import TInceptionNet


class BrainTumNetV2(nn.Module):
    """
    Multi-task brain tumor segmentation and classification model - V2

    Tasks:
    1. Segmentation: 3-class (background, tumor core, edema)
    2. Classification: Binary (HGG vs LGG)

    Args:
        in_ch: Input channels (4 for FLAIR, T1, T1CE, T2)
        num_cls: Number of classification classes (2 for HGG/LGG)
        base: Base feature channels (32 baseline, 48/64 Phase 2)
        dim: Transformer dimension (256 baseline, 384/512 Phase 2)
        patch: Transformer patch size
        depth: Transformer depth (2 baseline, 4 Phase 2)
        n_heads: Transformer heads (4 baseline, 8 Phase 2)
        num_classes_seg: Segmentation classes (3 for bg/TC/ED)
        dropout: Dropout rate (0.15 for large models)
        roi_stop_grad: Stop gradient in ROI path
        deep_supervision: Use deep supervision
        multi_scale_fusion: Use multi-scale fusion
    """
    def __init__(self, in_ch=4, num_cls=2, base=48, dim=384, patch=8,
                 depth=4, n_heads=8, num_classes_seg=3, dropout=0.15,
                 roi_stop_grad=True, deep_supervision=True, multi_scale_fusion=True):
        super().__init__()
        self.num_classes_seg = num_classes_seg
        self.roi_stop_grad = roi_stop_grad
        self.deep_supervision = deep_supervision

        # Segmentation network (V2 with enhancements)
        self.seg = SegUNetV2(
            in_ch=in_ch,
            base=base,
            dim=dim,
            patch=patch,
            depth=depth,
            n_heads=n_heads,
            num_classes=num_classes_seg,
            dropout=dropout,
            norm='instance',
            deep_supervision=deep_supervision,
            multi_scale_fusion=multi_scale_fusion
        )

        # Channel reduction for ROI (multi-modal to single channel)
        self.reduce = nn.Conv2d(in_ch, 1, 1, bias=False) if in_ch > 1 else nn.Identity()

        # Classification backbone
        self.cls_backbone = TInceptionNet(in_ch=1, num_classes=num_cls)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input image

        Returns:
            If deep_supervision=True:
                seg_logits: (B, num_classes_seg, H, W) main segmentation
                cls_logits: (B, num_cls) classification
                aux_outputs: List of auxiliary segmentations
            Else:
                seg_logits: (B, num_classes_seg, H, W)
                cls_logits: (B, num_cls)
        """
        # Segmentation forward
        seg_output = self.seg(x)

        # Handle deep supervision output
        if self.deep_supervision:
            seg_logits, aux_outputs = seg_output
        else:
            seg_logits = seg_output
            aux_outputs = None

        # ROI-guided classification
        # Compute whole tumor probability from segmentation
        if self.num_classes_seg == 1:
            # Binary segmentation
            seg_prob = torch.sigmoid(seg_logits)
        else:
            # Multi-class: whole tumor = sum of all tumor classes (exclude bg class 0)
            seg_prob = torch.softmax(seg_logits, dim=1)
            # Whole Tumor = TC (class 1) + ED (class 2)
            seg_prob = seg_prob[:, 1:, :, :].sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # ROI: mask input with tumor probability
        roi_input = self.reduce(x)  # (B, 1, H, W)

        if self.roi_stop_grad:
            roi = roi_input * seg_prob.detach()  # Stop gradient through segmentation
        else:
            roi = roi_input * seg_prob  # Allow gradient flow

        # Classification
        cls_logits = self.cls_backbone(roi)

        # Return
        if self.deep_supervision:
            return seg_logits, cls_logits, aux_outputs
        return seg_logits, cls_logits


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test BrainTumNetV2
    print("="*70)
    print("Testing BrainTumNetV2")
    print("="*70)

    # Baseline (V1-equivalent)
    model_v1 = BrainTumNetV2(
        in_ch=4, num_cls=2, base=32, dim=256, patch=8, depth=2, n_heads=4,
        num_classes_seg=3, dropout=0.0, deep_supervision=True, multi_scale_fusion=False
    )
    print(f"\nV1 Baseline: {count_parameters(model_v1)/1e6:.2f}M parameters")

    # Phase 2 Small (recommended)
    model_phase2_small = BrainTumNetV2(
        in_ch=4, num_cls=2, base=48, dim=384, patch=8, depth=4, n_heads=8,
        num_classes_seg=3, dropout=0.15, deep_supervision=True, multi_scale_fusion=True
    )
    print(f"Phase 2 Small: {count_parameters(model_phase2_small)/1e6:.2f}M parameters")

    # Phase 2 Large
    model_phase2_large = BrainTumNetV2(
        in_ch=4, num_cls=2, base=64, dim=512, patch=8, depth=4, n_heads=8,
        num_classes_seg=3, dropout=0.15, deep_supervision=True, multi_scale_fusion=True
    )
    print(f"Phase 2 Large: {count_parameters(model_phase2_large)/1e6:.2f}M parameters")

    # Test forward pass
    x = torch.randn(2, 4, 256, 256)

    print(f"\nForward pass test (Phase 2 Small):")
    seg, cls, aux = model_phase2_small(x)
    print(f"  Input: {x.shape}")
    print(f"  Seg output: {seg.shape}")
    print(f"  Cls output: {cls.shape}")
    print(f"  Aux outputs: [{aux[0].shape}, {aux[1].shape}, {aux[2].shape}]")

    print("\n✓ BrainTumNetV2 tests passed!")
