import torch.nn as nn
import torch

class InceptionBranch(nn.Module):
    def __init__(self, in_ch, out_ch, k=(3,3)):
        super().__init__()
        if k==(1,1):
            self.op = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        elif k==(1,3):
            self.op = nn.Conv2d(in_ch, out_ch, (1,3), padding=(0,1), bias=False)
        elif k==(3,1):
            self.op = nn.Conv2d(in_ch, out_ch, (3,1), padding=(1,0), bias=False)
        else:
            self.op = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.op(x)))

class TInceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        c = out_ch // 4
        self.b1 = InceptionBranch(in_ch, c, (1,1))
        self.b2 = InceptionBranch(in_ch, c, (3,3))
        self.b3 = InceptionBranch(in_ch, c, (1,3))
        self.b4 = InceptionBranch(in_ch, c, (3,1))
        self.fuse = nn.Conv2d(c*4, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.act(self.bn(self.fuse(x)))

class TInceptionNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.b1 = TInceptionBlock(64, 128)
        self.b2 = TInceptionBlock(128, 256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)
