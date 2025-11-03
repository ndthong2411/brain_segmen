"""
Swin-UNETR Wrapper for Brain Tumor Segmentation

Uses MONAI's Swin-UNETR implementation with 2D adaptation for BraTS dataset.

Reference:
    Hatamizadeh et al. "Swin UNETR: Swin Transformers for Semantic Segmentation
    of Brain Tumors in MRI Images" (2022)
"""

import torch
import torch.nn as nn

try:
    from monai.networks.nets import SwinUNETR
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("WARNING: MONAI not installed. Swin-UNETR will not be available.")
    print("Install with: pip install monai")


class SwinUNETRWrapper(nn.Module):
    """
    Swin-UNETR wrapper for brain tumor segmentation

    This wrapper adapts MONAI's Swin-UNETR for:
    - 2D slice-based segmentation (spatial_dims=2)
    - Multi-class segmentation (3 classes: BG, TC, ED)
    - Segmentation-only task (no classification)

    Args:
        in_ch: Input channels (4 for FLAIR, T1, T1CE, T2)
        num_classes_seg: Number of segmentation classes (3)
        feature_size: Base feature dimension (24 for small, 48 for large)
        img_size: Input image size (256)
        use_checkpoint: Enable gradient checkpointing (saves memory)
        depths: Swin Transformer depths per stage [2,2,6,2] for base model
        num_heads: Number of attention heads per stage [3,6,12,24]
        window_size: Window size for shifted window attention (7)

    Forward:
        Returns (seg_logits, None) where None is placeholder for classification
        to maintain compatibility with dual-task models.
    """

    def __init__(
        self,
        in_ch=4,
        num_classes_seg=3,
        feature_size=48,
        img_size=256,
        use_checkpoint=True,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
    ):
        super().__init__()

        if not MONAI_AVAILABLE:
            raise ImportError(
                "MONAI is required for Swin-UNETR. "
                "Install with: pip install monai"
            )

        self.num_classes_seg = num_classes_seg

        # Swin-UNETR segmentation network
        # Note: img_size parameter was deprecated in MONAI 1.3 and removed in later versions
        # We try to use it for backwards compatibility, but fall back to omitting it
        try:
            self.seg = SwinUNETR(
                img_size=(img_size, img_size),
                spatial_dims=2,  # 2D mode for slice-based segmentation
                in_channels=in_ch,
                out_channels=num_classes_seg,
                feature_size=feature_size,
                depths=depths,
                num_heads=num_heads,
                use_checkpoint=use_checkpoint,  # Gradient checkpointing
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                dropout_path_rate=dropout_path_rate,
                norm_name="instance",
            )
        except TypeError as e:
            # If img_size is not accepted (MONAI >= 1.5), create without it
            if "img_size" in str(e):
                self.seg = SwinUNETR(
                    spatial_dims=2,  # 2D mode for slice-based segmentation
                    in_channels=in_ch,
                    out_channels=num_classes_seg,
                    feature_size=feature_size,
                    depths=depths,
                    num_heads=num_heads,
                    use_checkpoint=use_checkpoint,  # Gradient checkpointing
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    dropout_path_rate=dropout_path_rate,
                    norm_name="instance",
                )
            else:
                raise

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
    print("Testing Swin-UNETR Wrapper...")

    if not MONAI_AVAILABLE:
        print("MONAI not available, skipping test")
    else:
        import monai
        print(f"MONAI version: {monai.__version__}")
        
        # Create model (compatible with MONAI 1.3+ including versions where img_size is removed)
        model = SwinUNETRWrapper(
            in_ch=4,
            num_classes_seg=3,
            feature_size=48,
            img_size=256,
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

        print("✓ Swin-UNETR wrapper test passed!")
