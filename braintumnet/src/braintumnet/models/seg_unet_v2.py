"""
Enhanced U-Net with Phase 2 Improvements

Changes from V1:
1. BatchNorm → InstanceNorm (medical imaging standard)
2. ReLU → LeakyReLU (better gradients)
3. Residual connections in all blocks
4. MaxPool → Strided convolution (learned downsampling)
5. Multi-scale fusion before final head
6. Dropout for regularization
7. Larger model capacity support

Author: BrainTumNet Phase 2 Upgrade
Date: 2025-10-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam import CBAM
from .masked_transformer import AdaptiveMaskedTransformer


def conv_norm_act(in_ch, out_ch, k=3, s=1, p=1, norm='instance', dropout=0.0):
    """
    Improved convolution block: Conv + Norm + LeakyReLU + Dropout

    Args:
        norm: 'instance' (medical imaging), 'batch', or 'group'
        dropout: dropout probability (0.0 = no dropout)
    """
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)]

    # Normalization
    if norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    elif norm == 'batch':
        layers.append(nn.BatchNorm2d(out_ch))
    elif norm == 'group':
        num_groups = min(32, out_ch // 4)  # Adaptive group size
        layers.append(nn.GroupNorm(num_groups, out_ch))

    # Activation
    layers.append(nn.LeakyReLU(0.01, inplace=True))  # slope=0.01 (nnUNet style)

    # Dropout
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))

    return nn.Sequential(*layers)


class ResidualConvBlock(nn.Module):
    """
    Residual convolutional block with InstanceNorm and LeakyReLU

    Structure: Conv-Norm-Act -> Conv-Norm -> Add-Act
    """
    def __init__(self, in_ch, out_ch, norm='instance', dropout=0.0):
        super().__init__()
        self.conv1 = conv_norm_act(in_ch, out_ch, norm=norm, dropout=dropout)
        # Second conv without activation (will be applied after residual add)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True) if norm == 'instance' else nn.BatchNorm2d(out_ch)
        )

        # Residual connection: 1x1 conv if channel mismatch
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        identity = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity  # Residual addition
        out = self.act(out)
        return out


class EncoderBlock(nn.Module):
    """
    Encoder block with residual convolutions and strided conv downsampling

    Improvements:
    - Residual connections
    - Strided conv instead of MaxPool (learned downsampling)
    """
    def __init__(self, in_ch, out_ch, norm='instance', dropout=0.0):
        super().__init__()
        self.block = ResidualConvBlock(in_ch, out_ch, norm=norm, dropout=dropout)
        # Strided convolution for downsampling (learnable, better than MaxPool)
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.block(x)
        x_down = self.downsample(x)
        return x, x_down


class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections (nnU-Net style, Phase 2)

    Highlights salient features from encoder using gating signal from decoder.
    Suppresses irrelevant regions in skip connections.

    Expected improvement: +1-2% Dice through better feature selection

    Args:
        F_g: Channels in gating signal (from decoder)
        F_l: Channels in skip connection (from encoder)
        F_int: Intermediate channels
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.InstanceNorm2d(F_int, affine=True)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.InstanceNorm2d(F_int, affine=True)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: (B, F_g, H, W) gating signal from decoder
            x: (B, F_l, H, W) skip connection from encoder

        Returns:
            (B, F_l, H, W) attention-weighted skip connection
        """
        # Project both to intermediate dimension
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Combine and compute attention
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Apply attention weights to skip connection
        return x * psi


class DecoderBlock(nn.Module):
    """
    Decoder block with residual convolutions and CBAM attention

    Improvements:
    - Residual connections
    - CBAM attention on skip connections
    - Dropout for regularization
    - (Phase 2) Optional Attention Gates
    """
    def __init__(self, in_ch, out_ch, norm='instance', dropout=0.0, use_attention_gate=False):
        super().__init__()
        self.use_attention_gate = use_attention_gate
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)

        # Phase 2: Attention Gate (optional)
        if use_attention_gate:
            self.attn_gate = AttentionGate(F_g=out_ch, F_l=out_ch, F_int=out_ch // 2)

        self.cbam = CBAM(out_ch)
        self.block = ResidualConvBlock(out_ch * 2, out_ch, norm=norm, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)

        # Phase 2: Apply attention gate if enabled
        if self.use_attention_gate:
            skip = self.attn_gate(g=x, x=skip)

        skip = self.cbam(skip)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class BoundaryRefinementModule(nn.Module):
    """
    Boundary Refinement Module for improved edge precision

    Addresses the IoU-Dice gap by explicitly modeling boundaries
    using edge detection and boundary-aware attention.

    Expected improvement: +2-3% Dice, reduces IoU-Dice gap from 10% to ~5%
    """
    def __init__(self, in_channels):
        super().__init__()
        # Edge detector (Sobel-like learnable filter)
        self.edge_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                                    groups=in_channels, bias=False)

        # Initialize with edge detection kernels
        with torch.no_grad():
            # Horizontal Sobel kernel
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            # Vertical Sobel kernel
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            # Combine for edge magnitude
            sobel = (sobel_x.abs() + sobel_y.abs()) / 2

            # Apply to all channels
            for i in range(in_channels):
                self.edge_conv.weight[i, 0] = sobel / sobel.sum()

        # Boundary attention mechanism
        self.boundary_attn = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.LeakyReLU(0.01),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.LeakyReLU(0.01),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, features):
        """
        Args:
            features: (B, C, H, W) feature map
        Returns:
            refined: (B, C, H, W) boundary-refined features
        """
        # Detect edges
        edges = self.edge_conv(features)

        # Concatenate original features with edge features
        combined = torch.cat([features, edges], dim=1)

        # Generate boundary attention map
        attn = self.boundary_attn(combined)

        # Apply attention with residual connection
        refined = features * (1 + attn)  # Multiplicative attention + identity

        return refined


class MultiScaleFusion(nn.Module):
    """
    Multi-scale feature fusion module

    Fuses features from multiple decoder levels to capture
    both fine-grained and coarse information.
    """
    def __init__(self, channels_list, out_channels):
        """
        Args:
            channels_list: List of channel dimensions [d1_ch, d2_ch, d3_ch, d4_ch]
            out_channels: Output channel dimension
        """
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1, bias=False) for ch in channels_list
        ])
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, features):
        """
        Args:
            features: List [d1, d2, d3, d4] with different spatial sizes
        Returns:
            fused: (B, out_channels, H, W) fused features
        """
        target_size = features[0].shape[2:]  # Use largest spatial size (d1)

        upsampled = []
        for i, feat in enumerate(features):
            # Project to same channel dimension
            feat = self.convs[i](feat)
            # Upsample to target size if needed
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled.append(feat)

        # Fuse by summation
        fused = sum(upsampled)
        fused = self.norm(fused)
        fused = self.act(fused)
        return fused


class SegUNetV2(nn.Module):
    """
    Enhanced Segmentation U-Net with Phase 2 improvements

    Key improvements:
    1. InstanceNorm instead of BatchNorm
    2. LeakyReLU instead of ReLU
    3. Residual connections in all blocks
    4. Strided conv instead of MaxPool
    5. Multi-scale fusion
    6. Dropout regularization
    7. Support for larger capacity

    Args:
        in_ch: Input channels (4 for multi-modal MRI)
        base: Base number of features (32 baseline, 48/64 for Phase 2)
        dim: Transformer dimension (256 baseline, 384/512 for Phase 2)
        patch: Transformer patch size
        depth: Transformer depth (2 baseline, 4 for Phase 2)
        n_heads: Transformer attention heads (4 baseline, 8 for Phase 2)
        num_classes: Number of segmentation classes
        dropout: Dropout probability (0.1-0.15 for large models)
        norm: Normalization type ('instance', 'batch', 'group')
        deep_supervision: Use deep supervision
        multi_scale_fusion: Use multi-scale fusion
        boundary_refinement: Use boundary refinement (Phase 1 optimization, improves IoU-Dice gap)
    """
    def __init__(self, in_ch=4, base=48, dim=384, patch=8, depth=4, n_heads=8,
                 num_classes=3, dropout=0.15, norm='instance',
                 deep_supervision=True, multi_scale_fusion=True, boundary_refinement=False,
                 use_multiscale_transformer=False, use_attention_gates=False):
        super().__init__()
        self.patch = patch
        self.deep_supervision = deep_supervision
        self.multi_scale_fusion = multi_scale_fusion
        self.boundary_refinement = boundary_refinement
        self.use_multiscale_transformer = use_multiscale_transformer
        self.use_attention_gates = use_attention_gates
        self.num_classes = num_classes

        # Encoder
        self.e1 = EncoderBlock(in_ch, base, norm=norm, dropout=0)
        self.e2 = EncoderBlock(base, base*2, norm=norm, dropout=0)
        self.e3 = EncoderBlock(base*2, base*4, norm=norm, dropout=dropout)
        self.e4 = EncoderBlock(base*4, base*8, norm=norm, dropout=dropout)

        # Transformer bottleneck (Phase 1 or Phase 2)
        self.bottleneck_conv = conv_norm_act(base*8, dim, k=1, s=1, p=0, norm=norm)

        if use_multiscale_transformer:
            # Phase 2: Multi-scale transformer
            from .multiscale_transformer import MultiScaleTransformerBottleneck
            self.bottleneck = MultiScaleTransformerBottleneck(
                in_ch=dim, dim=dim,
                patch_sizes=[4, 8, 16],  # Multi-scale
                depth=depth, n_heads=n_heads
            )
        else:
            # Phase 1: Single-scale transformer (original)
            self.amt = AdaptiveMaskedTransformer(
                in_ch=dim, dim=dim, patch_size=patch, depth=depth, n_heads=n_heads
            )
            self.tr_upsample = nn.ConvTranspose2d(dim, base*8, kernel_size=patch, stride=patch, bias=False)

        # Decoder (Phase 2: with optional attention gates)
        self.d4 = DecoderBlock(base*8, base*8, norm=norm, dropout=dropout, use_attention_gate=use_attention_gates)
        self.d3 = DecoderBlock(base*8, base*4, norm=norm, dropout=dropout, use_attention_gate=use_attention_gates)
        self.d2 = DecoderBlock(base*4, base*2, norm=norm, dropout=dropout/2, use_attention_gate=use_attention_gates)
        self.d1 = DecoderBlock(base*2, base, norm=norm, dropout=0, use_attention_gate=use_attention_gates)

        # Multi-scale fusion
        if self.multi_scale_fusion:
            self.ms_fusion = MultiScaleFusion(
                channels_list=[base, base*2, base*4, base*8],
                out_channels=base
            )
            self.fusion_conv = ResidualConvBlock(base*2, base, norm=norm, dropout=0)

        # Boundary refinement (Phase 1 optimization)
        if self.boundary_refinement:
            self.boundary_refine = BoundaryRefinementModule(base)

        # Segmentation head
        self.head = nn.Conv2d(base, num_classes, 1)

        # Deep supervision auxiliary heads
        if self.deep_supervision:
            self.aux_head3 = nn.Conv2d(base*4, num_classes, 1)
            self.aux_head2 = nn.Conv2d(base*2, num_classes, 1)
            self.aux_head1 = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        # Encoder
        s1, x1 = self.e1(x)      # base, H, W
        s2, x2 = self.e2(x1)     # base*2, H/2, W/2
        s3, x3 = self.e3(x2)     # base*4, H/4, W/4
        s4, x4 = self.e4(x3)     # base*8, H/8, W/8

        # Transformer bottleneck (Phase 1 or Phase 2)
        b = self.bottleneck_conv(x4)

        if self.use_multiscale_transformer:
            # Phase 2: Multi-scale transformer (already returns spatial)
            b = self.bottleneck(b)
        else:
            # Phase 1: Single-scale transformer
            b = self.amt(b)
            b = self.tr_upsample(b)

        # Decoder
        d4 = self.d4(b, s4)      # base*8, H/8, W/8

        d3 = self.d3(d4, s3)     # base*4, H/4, W/4
        aux3 = self.aux_head3(d3) if self.deep_supervision else None

        d2 = self.d2(d3, s2)     # base*2, H/2, W/2
        aux2 = self.aux_head2(d2) if self.deep_supervision else None

        d1 = self.d1(d2, s1)     # base, H, W
        aux1 = self.aux_head1(d1) if self.deep_supervision else None

        # Multi-scale fusion (optional)
        if self.multi_scale_fusion:
            decoder_features = [d1, d2, d3, d4]
            fused = self.ms_fusion(decoder_features)
            # Combine fused features with final decoder output
            combined = torch.cat([d1, fused], dim=1)
            final_features = self.fusion_conv(combined)
        else:
            final_features = d1

        # Boundary refinement (Phase 1 optimization - improves IoU-Dice gap)
        if self.boundary_refinement:
            final_features = self.boundary_refine(final_features)

        # Final segmentation
        seg = self.head(final_features)

        if self.deep_supervision:
            return seg, [aux3, aux2, aux1]
        return seg


# Utility function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test V2 model
    print("="*70)
    print("Testing SegUNetV2")
    print("="*70)

    # Baseline config (similar to V1)
    model_baseline = SegUNetV2(
        in_ch=4, base=32, dim=256, patch=8, depth=2, n_heads=4,
        num_classes=3, dropout=0.0, deep_supervision=True, multi_scale_fusion=False
    )
    print(f"\nBaseline (V1-like): {count_parameters(model_baseline)/1e6:.2f}M parameters")

    # Phase 2 small
    model_phase2_small = SegUNetV2(
        in_ch=4, base=48, dim=384, patch=8, depth=4, n_heads=8,
        num_classes=3, dropout=0.15, deep_supervision=True, multi_scale_fusion=True
    )
    print(f"Phase 2 Small: {count_parameters(model_phase2_small)/1e6:.2f}M parameters")

    # Phase 2 large
    model_phase2_large = SegUNetV2(
        in_ch=4, base=64, dim=512, patch=8, depth=4, n_heads=8,
        num_classes=3, dropout=0.15, deep_supervision=True, multi_scale_fusion=True
    )
    print(f"Phase 2 Large: {count_parameters(model_phase2_large)/1e6:.2f}M parameters")

    # Test forward pass
    x = torch.randn(2, 4, 256, 256)
    seg, aux = model_phase2_small(x)
    print(f"\nForward pass test:")
    print(f"  Input: {x.shape}")
    print(f"  Seg output: {seg.shape}")
    print(f"  Aux outputs: [{aux[0].shape}, {aux[1].shape}, {aux[2].shape}]")
    print("\n✓ SegUNetV2 tests passed!")
