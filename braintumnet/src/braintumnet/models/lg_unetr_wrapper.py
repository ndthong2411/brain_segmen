"""
LG-UNETR: Local-Global UNet Transformer for Brain Tumor Segmentation

Hybrid architecture combining:
- CNN encoder for local features (multiple scales)
- Transformer encoder for global context
- Dual-path fusion at each decoder level
- Skip connections from both CNN and Transformer paths

Key innovation:
    At each encoder level, both CNN and Transformer process features,
    capturing both local details and global context simultaneously.

Reference:
    Inspired by "TransBTS" and "UNETR" architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalGlobalEncoder(nn.Module):
    """
    Dual-path encoder with CNN (local) and Transformer (global)

    At each level:
    - CNN path: Captures local spatial features
    - Transformer path: Captures global context
    - Both paths process features in parallel
    """

    def __init__(self, in_ch=4, base=32, num_levels=4, embed_dim=384, depth=12, num_heads=6):
        super().__init__()

        self.num_levels = num_levels
        self.embed_dim = embed_dim

        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # CNN encoder path (local features)
        self.cnn_encoders = nn.ModuleList()
        self.cnn_downs = nn.ModuleList()

        ch = base
        for i in range(num_levels):
            # Encoder block
            self.cnn_encoders.append(
                nn.Sequential(
                    ConvBlock(ch, ch*2),
                    ConvBlock(ch*2, ch*2),
                )
            )
            # Downsampling
            if i < num_levels - 1:
                self.cnn_downs.append(nn.Conv2d(ch*2, ch*2, 3, 2, 1))
            ch = ch * 2

        # Transformer encoder path (global features)
        # Input: 256x256 image -> patches for transformer
        self.patch_embed = PatchEmbed(in_ch, embed_dim, patch_size=16)  # 256 -> 16x16 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, embed_dim))  # 16*16=256

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Projection layers to match CNN feature dimensions at each level
        self.trans_projs = nn.ModuleList()
        ch = base
        for i in range(num_levels):
            ch = ch * 2
            self.trans_projs.append(
                nn.Conv2d(embed_dim, ch, 1)
            )

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, 4, 256, 256]
        B = x.shape[0]

        # === CNN Path (Local) ===
        cnn_feats = []
        x_cnn = self.stem(x)  # [B, 32, 256, 256]

        for i in range(self.num_levels):
            x_cnn = self.cnn_encoders[i](x_cnn)  # [B, 64/128/256/512, 256/128/64/32]
            cnn_feats.append(x_cnn)
            if i < self.num_levels - 1:
                x_cnn = self.cnn_downs[i](x_cnn)

        # === Transformer Path (Global) ===
        x_trans = self.patch_embed(x)  # [B, 256, 384]
        x_trans = x_trans + self.pos_embed

        for block in self.transformer_blocks:
            x_trans = block(x_trans)

        x_trans = self.norm(x_trans)  # [B, 256, 384]

        # Reshape to spatial: [B, 384, 16, 16]
        x_trans = x_trans.transpose(1, 2).reshape(B, self.embed_dim, 16, 16)

        # Project transformer features to each resolution level
        trans_feats = []
        for i in range(self.num_levels):
            # Target size from CNN features
            h, w = cnn_feats[i].shape[2], cnn_feats[i].shape[3]

            # Upsample transformer features to match CNN resolution
            trans_feat = F.interpolate(x_trans, size=(h, w), mode='bilinear', align_corners=False)

            # Project to match CNN channels
            trans_feat = self.trans_projs[i](trans_feat)

            trans_feats.append(trans_feat)

        return cnn_feats, trans_feats


class LocalGlobalDecoder(nn.Module):
    """
    Decoder that fuses local (CNN) and global (Transformer) features
    """

    def __init__(self, base=32, num_levels=4, num_classes=3):
        super().__init__()

        self.num_levels = num_levels

        # Calculate channel sizes for each level: [64, 128, 256, 512]
        cnn_channels = [base * (2 ** (i+1)) for i in range(num_levels)]

        # Fusion blocks at each encoder level
        self.fusions = nn.ModuleList()
        for ch in cnn_channels:
            self.fusions.append(FusionBlock(ch, ch))

        # Upsampling path
        self.ups = nn.ModuleList()
        # Build from bottleneck to top
        for i in range(num_levels - 1, 0, -1):
            in_ch = cnn_channels[i]
            out_ch = cnn_channels[i-1]
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 2, 2),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )

        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(cnn_channels[0], cnn_channels[0], 3, 1, 1),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_channels[0], num_classes, 1),
        )

    def forward(self, cnn_feats, trans_feats):
        # cnn_feats: list of [B, 64/128/256/512, 256/128/64/32]
        # trans_feats: list of [B, 64/128/256/512, 256/128/64/32]

        # Start from deepest level (bottleneck: 32x32)
        x = self.fusions[-1](cnn_feats[-1], trans_feats[-1])  # [B, 512, 32, 32]

        # Progressively upsample and fuse with encoder features
        # 32 -> 64 -> 128 -> 256
        for i in range(len(self.ups)):
            x = self.ups[i](x)  # Upsample
            # Fuse with corresponding encoder level
            enc_idx = self.num_levels - 2 - i
            x = self.fusions[enc_idx](x, trans_feats[enc_idx])

        # x is now at [B, 64, 256, 256] - already at full resolution!
        seg = self.seg_head(x)

        return seg


class ConvBlock(nn.Module):
    """Basic convolutional block"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FusionBlock(nn.Module):
    """Fuse local (CNN) and global (Transformer) features"""

    def __init__(self, cnn_ch, trans_ch):
        super().__init__()

        # Attention-based fusion
        self.cnn_gate = nn.Sequential(
            nn.Conv2d(cnn_ch + trans_ch, cnn_ch, 1),
            nn.Sigmoid(),
        )

        self.trans_gate = nn.Sequential(
            nn.Conv2d(cnn_ch + trans_ch, trans_ch, 1),
            nn.Sigmoid(),
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(cnn_ch + trans_ch, cnn_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cnn_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, cnn_feat, trans_feat):
        # Concatenate
        concat = torch.cat([cnn_feat, trans_feat], dim=1)

        # Gated fusion
        cnn_att = self.cnn_gate(concat)
        trans_att = self.trans_gate(concat)

        cnn_feat = cnn_feat * cnn_att
        trans_feat = trans_feat * trans_att

        # Fuse
        fused = torch.cat([cnn_feat, trans_feat], dim=1)
        out = self.fusion_conv(fused)

        return out


class PatchEmbed(nn.Module):
    """Image to patch embedding"""

    def __init__(self, in_ch=4, embed_dim=384, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)

    def forward(self, x):
        # x: [B, 4, 256, 256]
        x = self.proj(x)  # [B, 384, 16, 16]
        x = x.flatten(2).transpose(1, 2)  # [B, 256, 384]
        return x


class TransformerBlock(nn.Module):
    """Transformer block with MHA + FFN"""

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


class LGUNETRWrapper(nn.Module):
    """
    LG-UNETR: Local-Global UNet Transformer

    Dual-path encoder-decoder architecture:
    - CNN path: Captures local spatial details
    - Transformer path: Captures global context
    - Fusion at each decoder level

    Args:
        in_ch: Input channels (4 for BraTS)
        num_classes_seg: Number of segmentation classes
        base: Base number of CNN channels
        num_levels: Number of encoder-decoder levels
        embed_dim: Transformer embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        in_ch=4,
        num_classes_seg=3,
        base=32,
        num_levels=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
    ):
        super().__init__()

        self.num_classes_seg = num_classes_seg

        self.encoder = LocalGlobalEncoder(
            in_ch, base, num_levels, embed_dim, depth, num_heads
        )
        self.decoder = LocalGlobalDecoder(base, num_levels, num_classes_seg)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: [B, 4, 256, 256]

        Returns:
            seg_logits: [B, num_classes_seg, 256, 256]
            cls_logits: None (segmentation only)
        """
        cnn_feats, trans_feats = self.encoder(x)
        seg_logits = self.decoder(cnn_feats, trans_feats)

        return seg_logits, None

    def get_params_count(self):
        """Count parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test
    model = LGUNETRWrapper(in_ch=4, num_classes_seg=3)
    x = torch.randn(2, 4, 256, 256)

    seg, cls = model(x)
    print(f"Input: {x.shape}")
    print(f"Seg output: {seg.shape}")
    print(f"Cls output: {cls}")
    print(f"Parameters: {model.get_params_count():,}")
