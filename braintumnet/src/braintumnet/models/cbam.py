import torch, torch.nn as nn, torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False),
        )
    def forward(self, x):
        att = torch.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))
        return x * att

class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k//2, bias=False)
    def forward(self, x):
        att = torch.cat([x.mean(1, True), x.amax(1, True)], dim=1)
        att = torch.sigmoid(self.conv(att))
        return x * att

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, k=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(k)
    def forward(self, x):
        return self.sa(self.ca(x))
