"""
nnU-Net Style Architecture for Brain Tumor Segmentation

Implements nnU-Net-inspired architecture with:
- 5-level encoder/decoder (vs 4-level in baseline)
- InstanceNorm2d + LeakyReLU(0.01)
- Residual connections in all blocks
- Strided convolution for downsampling
- Deep supervision at all decoder levels

Reference:
    Isensee et al. "nnU-Net: a self-configuring method for deep learning-based
    biomedical image segmentation" (Nature Methods, 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class nnUNetResBlock(nn.Module):
    """
    nnU-Net residual block

    Structure:
        Conv-InstanceNorm-LeakyReLU → Conv-InstanceNorm → Add residual → LeakyReLU

    Exactly follows nnU-Net implementation:
    - InstanceNorm2d (affine=True)
    - LeakyReLU(negative_slope=0.01)
    - 1x1 conv for channel matching in residual
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        # First conv block
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        # Second conv block (no activation - applied after residual add)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)

        # Residual connection (1x1 conv if channel mismatch)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            self.skip = nn.Identity()

        self.act2 = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        identity = self.skip(x)

        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        out = out + identity  # Residual addition
        out = self.act2(out)

        return out


class nnUNetEncoder(nn.Module):
    """
    nnU-Net encoder with 5 levels

    Architecture:
        Level 1: 4   → 32   (256×256)
        Level 2: 32  → 64   (128×128)
        Level 3: 64  → 128  (64×64)
        Level 4: 128 → 256  (32×32)
        Level 5: 256 → 320  (16×16) - bottleneck

    Uses strided convolution for learned downsampling
    """

    def __init__(self, in_ch=4, base=32):
        super().__init__()

        # Level 1
        self.e1 = nnUNetResBlock(in_ch, base)
        self.down1 = nn.Conv2d(base, base, 3, stride=2, padding=1, bias=False)

        # Level 2
        self.e2 = nnUNetResBlock(base, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1, bias=False)

        # Level 3
        self.e3 = nnUNetResBlock(base * 2, base * 4)
        self.down3 = nn.Conv2d(base * 4, base * 4, 3, stride=2, padding=1, bias=False)

        # Level 4
        self.e4 = nnUNetResBlock(base * 4, base * 8)
        self.down4 = nn.Conv2d(base * 8, base * 8, 3, stride=2, padding=1, bias=False)

        # Level 5 (bottleneck)
        self.e5 = nnUNetResBlock(base * 8, base * 10)

    def forward(self, x):
        # Encoder with skip connections
        s1 = self.e1(x)  # (B, 32, 256, 256)
        x = self.down1(s1)

        s2 = self.e2(x)  # (B, 64, 128, 128)
        x = self.down2(s2)

        s3 = self.e3(x)  # (B, 128, 64, 64)
        x = self.down3(s3)

        s4 = self.e4(x)  # (B, 256, 32, 32)
        x = self.down4(s4)

        s5 = self.e5(x)  # (B, 320, 16, 16) - bottleneck

        return s5, [s1, s2, s3, s4]


class nnUNetDecoder(nn.Module):
    """
    nnU-Net decoder with deep supervision

    Uses:
    - Transposed convolution for upsampling
    - Concatenation with skip connections
    - Residual blocks
    - Deep supervision (auxiliary outputs at d4, d3, d2)
    """

    def __init__(self, base=32, num_classes=3, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Decoder level 4
        self.up4 = nn.ConvTranspose2d(base * 10, base * 8, 2, stride=2, bias=False)
        self.d4 = nnUNetResBlock(base * 16, base * 8)  # Concat: up4(320) + s4(256) = 576 → 256

        # Decoder level 3
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2, bias=False)
        self.d3 = nnUNetResBlock(base * 8, base * 4)  # Concat: up3(256) + s3(128) = 384 → 128

        # Decoder level 2
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2, bias=False)
        self.d2 = nnUNetResBlock(base * 4, base * 2)  # Concat: up2(128) + s2(64) = 192 → 64

        # Decoder level 1
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2, bias=False)
        self.d1 = nnUNetResBlock(base * 2, base)  # Concat: up1(64) + s1(32) = 96 → 32

        # Output heads
        self.head = nn.Conv2d(base, num_classes, 1)

        # Deep supervision heads
        if deep_supervision:
            self.aux_head4 = nn.Conv2d(base * 8, num_classes, 1)
            self.aux_head3 = nn.Conv2d(base * 4, num_classes, 1)
            self.aux_head2 = nn.Conv2d(base * 2, num_classes, 1)

    def forward(self, bottleneck, skips):
        s1, s2, s3, s4 = skips

        # Decoder level 4
        x = self.up4(bottleneck)  # (B, 256, 32, 32)
        x = torch.cat([x, s4], dim=1)  # (B, 512, 32, 32)
        d4 = self.d4(x)  # (B, 256, 32, 32)

        # Decoder level 3
        x = self.up3(d4)  # (B, 128, 64, 64)
        x = torch.cat([x, s3], dim=1)  # (B, 256, 64, 64)
        d3 = self.d3(x)  # (B, 128, 64, 64)

        # Decoder level 2
        x = self.up2(d3)  # (B, 64, 128, 128)
        x = torch.cat([x, s2], dim=1)  # (B, 128, 128, 128)
        d2 = self.d2(x)  # (B, 64, 128, 128)

        # Decoder level 1
        x = self.up1(d2)  # (B, 32, 256, 256)
        x = torch.cat([x, s1], dim=1)  # (B, 64, 256, 256)
        d1 = self.d1(x)  # (B, 32, 256, 256)

        # Final segmentation
        seg = self.head(d1)  # (B, num_classes, 256, 256)

        if self.deep_supervision:
            # Auxiliary outputs (will be resized to match target in loss computation)
            aux = [
                self.aux_head4(d4),  # (B, num_classes, 32, 32)
                self.aux_head3(d3),  # (B, num_classes, 64, 64)
                self.aux_head2(d2),  # (B, num_classes, 128, 128)
            ]
            return seg, aux
        else:
            return seg


class nnUNetWrapper(nn.Module):
    """
    nnU-Net style architecture for brain tumor segmentation

    This is a custom implementation inspired by nnU-Net design principles:
    - 5-level U-Net architecture
    - InstanceNorm + LeakyReLU
    - Residual connections
    - Strided convolution downsampling
    - Deep supervision

    Args:
        in_ch: Input channels (4 for multi-modal MRI)
        num_classes_seg: Number of segmentation classes (3)
        base: Base feature channels (32)
        deep_supervision: Enable deep supervision (True)

    Forward:
        Returns (seg_logits, None, aux_outputs) if deep_supervision
        Returns (seg_logits, None) otherwise
    """

    def __init__(
        self,
        in_ch=4,
        num_classes_seg=3,
        base=32,
        deep_supervision=True,
    ):
        super().__init__()

        self.num_classes_seg = num_classes_seg
        self.deep_supervision = deep_supervision

        # Encoder (5 levels)
        self.encoder = nnUNetEncoder(in_ch, base)

        # Decoder with deep supervision
        self.decoder = nnUNetDecoder(base, num_classes_seg, deep_supervision)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input (B, 4, 256, 256)

        Returns:
            If deep_supervision:
                (seg_logits, None, aux_outputs)
            Else:
                (seg_logits, None)
        """
        # Encode
        bottleneck, skips = self.encoder(x)

        # Decode
        if self.deep_supervision:
            seg, aux = self.decoder(bottleneck, skips)
            return seg, None, aux  # (seg, cls=None, aux)
        else:
            seg = self.decoder(bottleneck, skips)
            return seg, None  # (seg, cls=None)

    def get_num_params(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test code
if __name__ == "__main__":
    print("Testing nnU-Net Wrapper...")

    # Create model
    model = nnUNetWrapper(
        in_ch=4,
        num_classes_seg=3,
        base=32,
        deep_supervision=True,
    )

    # Test forward pass
    x = torch.randn(2, 4, 256, 256)
    output = model(x)

    if len(output) == 3:
        seg_logits, cls_logits, aux_outputs = output
    else:
        seg_logits, cls_logits = output
        aux_outputs = None

    print(f"Input shape: {x.shape}")
    print(f"Segmentation output shape: {seg_logits.shape}")
    print(f"Classification output: {cls_logits}")
    if aux_outputs:
        print(f"Auxiliary outputs: {[aux.shape for aux in aux_outputs]}")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")

    assert seg_logits.shape == (2, 3, 256, 256), "Wrong output shape!"
    assert cls_logits is None, "Classification should be None!"

    print("✓ nnU-Net wrapper test passed!")
