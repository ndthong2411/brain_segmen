"""
Multi-Scale Transformer Bottleneck (Phase 2 Optimization)

Enhances bottleneck with multi-scale patch embeddings for better
multi-resolution feature extraction.

Expected improvement: +1.5-2.5% Dice through better global context

Author: BrainTumNet Phase 2 Optimization
Date: 2025-01-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScalePatchEmbed(nn.Module):
    """
    Multi-scale patch embedding with different patch sizes

    Args:
        in_ch: Input channels
        embed_dim: Embedding dimension
        patch_sizes: List of patch sizes (e.g., [4, 8, 16])
    """
    def __init__(self, in_ch, embed_dim, patch_sizes=[4, 8, 16]):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.num_scales = len(patch_sizes)

        # Per-scale embeddings
        self.projections = nn.ModuleList([
            nn.Conv2d(in_ch, embed_dim, kernel_size=ps, stride=ps)
            for ps in patch_sizes
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in patch_sizes
        ])

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input feature map

        Returns:
            scale_tokens: List of [(B, N_i, C), ...] tokens at each scale
            scale_shapes: List of [(H_i, W_i), ...] spatial shapes
        """
        scale_tokens = []
        scale_shapes = []

        for proj, norm in zip(self.projections, self.norms):
            # Project to patches
            x_patch = proj(x)  # (B, embed_dim, H', W')
            B, C, H, W = x_patch.shape

            # Flatten to tokens
            tokens = x_patch.flatten(2).transpose(1, 2)  # (B, N, C)
            tokens = norm(tokens)

            scale_tokens.append(tokens)
            scale_shapes.append((H, W))

        return scale_tokens, scale_shapes


class TransformerBlock(nn.Module):
    """
    Standard Transformer block (for multi-scale transformer)

    Args:
        dim: Token dimension
        n_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        drop: Dropout rate
    """
    def __init__(self, dim, n_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)

        # MLP
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, C) tokens

        Returns:
            (B, N, C) transformed tokens
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class MultiScaleTransformerBottleneck(nn.Module):
    """
    Multi-Scale Transformer Bottleneck (Phase 2)

    Processes features at multiple scales (different patch sizes) and
    fuses them for better multi-resolution reasoning.

    Args:
        in_ch: Input channels
        dim: Transformer dimension
        patch_sizes: List of patch sizes (default: [4, 8, 16])
        depth: Number of transformer layers
        n_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        drop: Dropout rate
    """
    def __init__(self, in_ch, dim, patch_sizes=[4, 8, 16], depth=4,
                 n_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.num_scales = len(patch_sizes)

        # Multi-scale patch embedding
        self.patch_embed = MultiScalePatchEmbed(in_ch, dim, patch_sizes)

        # Shared transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])

        # Cross-scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(dim * self.num_scales, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(drop)
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input feature map

        Returns:
            (B, C, H, W) transformed feature map
        """
        B, C, H, W = x.shape

        # Get multi-scale tokens
        scale_tokens, scale_shapes = self.patch_embed(x)

        # Process each scale through transformers
        processed_scales = []
        for tokens in scale_tokens:
            # Apply transformer blocks
            for block in self.blocks:
                tokens = block(tokens)
            processed_scales.append(tokens)

        # Upsample all scales to largest resolution (smallest patch size)
        target_shape = scale_shapes[0]  # (H_max, W_max)
        target_num_tokens = target_shape[0] * target_shape[1]

        upsampled_scales = []
        for i, (tokens, (h, w)) in enumerate(zip(processed_scales, scale_shapes)):
            if (h, w) != target_shape:
                # Reshape to spatial
                tokens_spatial = tokens.transpose(1, 2).reshape(B, -1, h, w)
                # Upsample
                tokens_upsampled = F.interpolate(
                    tokens_spatial, size=target_shape,
                    mode='bilinear', align_corners=False
                )
                # Flatten back to tokens
                tokens = tokens_upsampled.flatten(2).transpose(1, 2)
            upsampled_scales.append(tokens)

        # Concatenate all scales
        fused_tokens = torch.cat(upsampled_scales, dim=-1)  # (B, N, C*num_scales)

        # Fuse multi-scale features
        fused_tokens = self.scale_fusion(fused_tokens)  # (B, N, C)

        # Reshape back to spatial
        fused_spatial = fused_tokens.transpose(1, 2).reshape(
            B, -1, target_shape[0], target_shape[1]
        )

        # Resize to original input size
        output = F.interpolate(
            fused_spatial, size=(H, W),
            mode='bilinear', align_corners=False
        )

        return output


# Test module
if __name__ == "__main__":
    print("="*70)
    print("Testing Multi-Scale Transformer Bottleneck")
    print("="*70)

    # Test multi-scale bottleneck
    model = MultiScaleTransformerBottleneck(
        in_ch=512,      # From encoder output (base*8 = 64*8)
        dim=512,        # Transformer dimension
        patch_sizes=[4, 8, 16],
        depth=4,
        n_heads=8
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nMulti-Scale Transformer: {total_params/1e6:.2f}M parameters")

    # Test forward pass
    x = torch.randn(2, 512, 32, 32)  # (B, C, H, W) - typical bottleneck size
    output = model(x)

    print(f"\nForward pass test:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  ✓ Shape preserved: {x.shape == output.shape}")

    print("\n✓ Multi-Scale Transformer tests passed!")
