import torch, torch.nn as nn
from .cbam import CBAM
from .masked_transformer import AdaptiveMaskedTransformer

def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(conv_bn_relu(in_ch, out_ch), conv_bn_relu(out_ch, out_ch))
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.block(x)
        x_down = self.pool(x)
        return x, x_down

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.cbam = CBAM(out_ch)  # CBAM on skip connection which has out_ch channels
        self.block = nn.Sequential(conv_bn_relu(out_ch + out_ch, out_ch), conv_bn_relu(out_ch, out_ch))
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, self.cbam(skip)], dim=1)
        x = self.block(x)
        return x

class SegUNetMasked(nn.Module):
    def __init__(self, in_ch=1, base=32, dim=256, patch=8, depth=2, n_heads=4, deep_supervision=False):
        super().__init__()
        self.patch = patch
        self.deep_supervision = deep_supervision

        self.e1 = EncoderBlock(in_ch, base)
        self.e2 = EncoderBlock(base, base*2)
        self.e3 = EncoderBlock(base*2, base*4)
        self.e4 = EncoderBlock(base*4, base*8)
        # After 4 encoder blocks: spatial size is H/16 x W/16
        # Transformer will further reduce by patch size
        self.bottleneck_conv = conv_bn_relu(base*8, dim, k=1, s=1, p=0)
        self.amt = AdaptiveMaskedTransformer(in_ch=dim, dim=dim, patch_size=patch, depth=depth, n_heads=n_heads)
        # Upsample transformer output back to original bottleneck size
        self.tr_upsample = nn.ConvTranspose2d(dim, base*8, kernel_size=patch, stride=patch)
        self.d4 = DecoderBlock(base*8, base*8)
        self.d3 = DecoderBlock(base*8, base*4)
        self.d2 = DecoderBlock(base*4, base*2)
        self.d1 = DecoderBlock(base*2, base)
        self.head = nn.Conv2d(base, 1, 1)

        # Deep Supervision: auxiliary segmentation heads at intermediate decoder levels
        if self.deep_supervision:
            self.aux_head3 = nn.Conv2d(base*4, 1, 1)  # After d3: 64x64 resolution
            self.aux_head2 = nn.Conv2d(base*2, 1, 1)  # After d2: 128x128 resolution
            self.aux_head1 = nn.Conv2d(base, 1, 1)    # After d1: 256x256 resolution (same as main head)
    def forward(self, x):
        s1, x1 = self.e1(x)      # s1: base, H, W
        s2, x2 = self.e2(x1)     # s2: base*2, H/2, W/2
        s3, x3 = self.e3(x2)     # s3: base*4, H/4, W/4
        s4, x4 = self.e4(x3)     # s4: base*8, H/8, W/8
        b = self.bottleneck_conv(x4)  # dim, H/16, W/16
        b = self.amt(b)          # dim, H/16/patch, W/16/patch
        b = self.tr_upsample(b)  # base*8, H/16, W/16 (upsampled back)

        x = self.d4(b, s4)       # base*8, H/8, W/8

        x = self.d3(x, s3)       # base*4, H/4, W/4
        aux3 = self.aux_head3(x) if self.deep_supervision else None  # Auxiliary output at 64x64

        x = self.d2(x, s2)       # base*2, H/2, W/2
        aux2 = self.aux_head2(x) if self.deep_supervision else None  # Auxiliary output at 128x128

        x = self.d1(x, s1)       # base, H, W
        aux1 = self.aux_head1(x) if self.deep_supervision else None  # Auxiliary output at 256x256

        seg = self.head(x)       # 1, H, W - Main output

        if self.deep_supervision:
            return seg, [aux3, aux2, aux1]
        return seg
