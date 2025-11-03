"""
TransUNet Wrapper for Brain Tumor Segmentation

Full TransUNet implementation combining:
- ResNet encoder (CNN for local features)
- Vision Transformer bottleneck (for global context)
- CNN decoder with skip connections

Reference:
    Chen et al. "TransUNet: Transformers Make Strong Encoders for
    Medical Image Segmentation" (2021)
    https://arxiv.org/abs/2102.04306
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResNetEncoder(nn.Module):
    """
    ResNet-style encoder with 4 stages
    Progressively downsample: 256 -> 128 -> 64 -> 32 -> 16
    """

    def __init__(self, in_ch=4, base=64):
        super().__init__()

        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 128x128, base channels
        self.stage1 = self._make_stage(base, base, 2)

        # Stage 2: 64x64, base*2 channels
        self.stage2 = self._make_stage(base, base*2, 2, stride=2)

        # Stage 3: 32x32, base*4 channels
        self.stage3 = self._make_stage(base*2, base*4, 2, stride=2)

        # Stage 4: 16x16, base*8 channels (for transformer)
        self.stage4 = self._make_stage(base*4, base*8, 2, stride=2)

    def _make_stage(self, in_ch, out_ch, num_blocks, stride=1):
        layers = []
        # First block with stride
        layers.append(ResNetBlock(in_ch, out_ch, stride))
        # Remaining blocks
        for _ in range(num_blocks - 1):
            layers.append(ResNetBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 4, 256, 256]

        x1 = self.conv1(x)      # [B, 64, 128, 128]
        x2 = self.stage1(x1)     # [B, 64, 128, 128]
        x3 = self.stage2(x2)     # [B, 128, 64, 64]
        x4 = self.stage3(x3)     # [B, 256, 32, 32]
        x5 = self.stage4(x4)     # [B, 512, 16, 16]

        return [x1, x2, x3, x4, x5]


class ResNetBlock(nn.Module):
    """Basic ResNet block with bottleneck"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Shortcut
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class VisionTransformer(nn.Module):
    """
    Vision Transformer bottleneck
    Processes 16x16 feature map with transformer layers
    """

    def __init__(self, in_ch=512, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = 1  # Treat each spatial location as a patch

        # Project to embedding dimension
        self.proj = nn.Conv2d(in_ch, embed_dim, 1)

        # Positional embedding for 16x16 grid
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, embed_dim))  # 16*16=256

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize pos embed
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, 512, 16, 16]
        B, C, H, W = x.shape

        # Project to embed_dim
        x = self.proj(x)  # [B, 768, 16, 16]

        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # [B, 256, 768]

        # Add positional embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Reshape back to spatial
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)  # [B, 768, 16, 16]

        return x


class TransformerBlock(nn.Module):
    """Standard transformer block with MHA + MLP"""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class TransUNetDecoder(nn.Module):
    """
    CNN decoder with skip connections from ResNet encoder
    Progressively upsample: 16 -> 32 -> 64 -> 128 -> 256
    """

    def __init__(self, embed_dim=768, base=64, num_classes=3):
        super().__init__()

        # Project transformer output to decoder channels
        self.proj = nn.Conv2d(embed_dim, base*8, 1)

        # Upsampling blocks with skip connections
        self.up1 = UpBlock(base*8, base*4, base*4)  # 16 -> 32, skip from stage3
        self.up2 = UpBlock(base*4, base*2, base*2)  # 32 -> 64, skip from stage2
        self.up3 = UpBlock(base*2, base, base)      # 64 -> 128, skip from stage1/conv1

        # Final upsample to 256x256 (without skip)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base, base, 2, 2),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(base, base, 3, 1, 1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, num_classes, 1),
        )

    def forward(self, x, enc_features):
        # x: [B, 768, 16, 16] from transformer
        # enc_features: [x1, x2, x3, x4, x5] from encoder
        # x1, x2: [B, 64, 128, 128] - both at same resolution

        x = self.proj(x)  # [B, 512, 16, 16]

        x = self.up1(x, enc_features[3])  # [B, 256, 32, 32]
        x = self.up2(x, enc_features[2])  # [B, 128, 64, 64]
        x = self.up3(x, enc_features[1])  # [B, 64, 128, 128] - use stage1 skip

        # Final upsample to 256x256
        x = self.up4(x)  # [B, 64, 256, 256]

        seg = self.seg_head(x)  # [B, 3, 256, 256]

        return seg


class UpBlock(nn.Module):
    """Upsampling block with skip connection"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class TransUNetWrapper(nn.Module):
    """
    Full TransUNet implementation

    Architecture:
        ResNet Encoder -> Vision Transformer -> CNN Decoder

    Args:
        in_ch: Input channels (4 for BraTS)
        num_classes_seg: Number of segmentation classes
        img_size: Input image size (must be 256)
        embed_dim: Transformer embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        base: Base number of channels in encoder/decoder
    """

    def __init__(
        self,
        in_ch=4,
        num_classes_seg=3,
        img_size=256,
        embed_dim=768,
        depth=12,
        num_heads=12,
        base=64,
    ):
        super().__init__()

        if img_size != 256:
            raise ValueError(f"TransUNet only supports img_size=256, got {img_size}")

        self.num_classes_seg = num_classes_seg

        # Three main components
        self.encoder = ResNetEncoder(in_ch, base)
        self.transformer = VisionTransformer(base*8, embed_dim, depth, num_heads)
        self.decoder = TransUNetDecoder(embed_dim, base, num_classes_seg)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: [B, 4, 256, 256]

        Returns:
            seg_logits: [B, num_classes_seg, 256, 256]
            cls_logits: None (segmentation only)
        """
        # Encode with ResNet
        enc_features = self.encoder(x)

        # Transform with ViT
        bottleneck = self.transformer(enc_features[-1])

        # Decode
        seg_logits = self.decoder(bottleneck, enc_features)

        return seg_logits, None  # No classification head

    def get_params_count(self):
        """Count parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test
    model = TransUNetWrapper(in_ch=4, num_classes_seg=3, img_size=256)
    x = torch.randn(2, 4, 256, 256)

    seg, cls = model(x)
    print(f"Input: {x.shape}")
    print(f"Seg output: {seg.shape}")
    print(f"Cls output: {cls}")
    print(f"Parameters: {model.get_params_count():,}")
