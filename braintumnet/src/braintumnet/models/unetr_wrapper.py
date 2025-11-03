"""
UNETR Wrapper for Brain Tumor Segmentation

Uses MONAI's UNETR (UNet with Transformers) implementation.

UNETR uses Vision Transformer (ViT) encoder with CNN decoder.
Difference from Swin-UNETR:
- UNETR: Vanilla ViT with global attention
- Swin-UNETR: Shifted window attention (more efficient)

Reference:
    Hatamizadeh et al. "UNETR: Transformers for 3D Medical Image Segmentation" (2022)
"""

import torch
import torch.nn as nn

try:
    from monai.networks.nets import UNETR
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("WARNING: MONAI not installed. UNETR will not be available.")
    print("Install with: pip install monai")


class UNETRWrapper(nn.Module):
    """
    UNETR wrapper for brain tumor segmentation

    This wrapper adapts MONAI's UNETR for:
    - 2D slice-based segmentation (spatial_dims=2)
    - Multi-class segmentation (3 classes: BG, TC, ED)
    - Segmentation-only task (no classification)

    Args:
        in_ch: Input channels (4 for FLAIR, T1, T1CE, T2)
        num_classes_seg: Number of segmentation classes (3)
        img_size: Input image size (256)
        hidden_size: ViT hidden dimension (768 for ViT-Base, 1024 for ViT-Large)
        feature_size: CNN decoder feature dimension (16)
        num_heads: Number of attention heads (12 for ViT-Base)
        mlp_dim: MLP dimension in transformer (3072 for ViT-Base = 4 * hidden_size)
        num_layers: Number of transformer layers (12 for ViT-Base)
        patch_size: Patch size for ViT (16)
        dropout_rate: Dropout rate (0.1)

    Forward:
        Returns (seg_logits, None) where None is placeholder for classification
        to maintain compatibility with dual-task models.
    """

    def __init__(
        self,
        in_ch=4,
        num_classes_seg=3,
        img_size=256,
        hidden_size=768,
        feature_size=16,
        num_heads=12,
        mlp_dim=None,
        dropout_rate=0.1,
    ):
        super().__init__()

        if not MONAI_AVAILABLE:
            raise ImportError(
                "MONAI is required for UNETR. "
                "Install with: pip install monai"
            )

        self.num_classes_seg = num_classes_seg

        # Default MLP dimension (4x hidden_size for ViT-Base)
        if mlp_dim is None:
            mlp_dim = hidden_size * 4

        # UNETR segmentation network
        self.seg = UNETR(
            in_channels=in_ch,
            out_channels=num_classes_seg,
            img_size=(img_size, img_size),
            spatial_dims=2,  # 2D mode
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            proj_type="conv",  # Convolutional projection (replaces pos_embed)
            norm_name="instance",  # InstanceNorm for medical imaging
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (B, 4, H, W) - 4 MRI modalities

        Returns:
            seg_logits: Segmentation logits (B, num_classes_seg, H, W)
            None: Placeholder for classification logits (compatibility)
        """
        seg_logits = self.seg(x)
        return seg_logits, None

    def get_num_params(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test code
if __name__ == "__main__":
    print("Testing UNETR Wrapper...")

    if not MONAI_AVAILABLE:
        print("MONAI not available, skipping test")
    else:
        # Create model (ViT-Base config)
        model = UNETRWrapper(
            in_ch=4,
            num_classes_seg=3,
            img_size=256,
            hidden_size=768,
            feature_size=16,
            num_heads=12,
            num_layers=12,
        )

        # Test forward pass
        x = torch.randn(2, 4, 256, 256)
        seg_logits, cls_logits = model(x)

        print(f"Input shape: {x.shape}")
        print(f"Segmentation output shape: {seg_logits.shape}")
        print(f"Classification output: {cls_logits}")
        print(f"Total parameters: {model.get_num_params():,}")
        print(f"Trainable parameters: {model.get_num_trainable_params():,}")

        assert seg_logits.shape == (2, 3, 256, 256), "Wrong output shape!"
        assert cls_logits is None, "Classification should be None!"

        print("✓ UNETR wrapper test passed!")
