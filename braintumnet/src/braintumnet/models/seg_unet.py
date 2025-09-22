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
        self.cbam = CBAM(out_ch + out_ch)
        self.block = nn.Sequential(conv_bn_relu(out_ch + out_ch, out_ch), conv_bn_relu(out_ch, out_ch))
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, self.cbam(skip)], dim=1)
        x = self.block(x)
        return x

class SegUNetMasked(nn.Module):
    def __init__(self, in_ch=1, base=32, dim=256, patch=8, depth=2, n_heads=4):
        super().__init__()
        self.e1 = EncoderBlock(in_ch, base)
        self.e2 = EncoderBlock(base, base*2)
        self.e3 = EncoderBlock(base*2, base*4)
        self.e4 = EncoderBlock(base*4, base*8)
        self.bottleneck_conv = conv_bn_relu(base*8, dim, k=1, s=1, p=0)
        self.amt = AdaptiveMaskedTransformer(in_ch=dim, dim=dim, patch_size=patch, depth=depth, n_heads=n_heads)
        self.bottleneck_out = conv_bn_relu(dim, base*16, k=1, s=1, p=0)
        self.d4 = DecoderBlock(base*16, base*8)
        self.d3 = DecoderBlock(base*8, base*4)
        self.d2 = DecoderBlock(base*4, base*2)
        self.d1 = DecoderBlock(base*2, base)
        self.head = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        s1, x1 = self.e1(x)
        s2, x2 = self.e2(x1)
        s3, x3 = self.e3(x2)
        s4, x4 = self.e4(x3)
        b = self.bottleneck_conv(x4)
        b = self.amt(b)
        b = self.bottleneck_out(b)
        x = self.d4(b, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        seg = self.head(x)
        return seg
