import torch, torch.nn as nn
from .seg_unet import SegUNetMasked
from .t_inception import TInceptionNet

class BrainTumNet(nn.Module):
    def __init__(self, in_ch=1, num_cls=2, base=32, dim=256, patch=8, depth=2, n_heads=4, roi_stop_grad=True):
        super().__init__()
        self.seg = SegUNetMasked(in_ch=in_ch, base=base, dim=dim, patch=patch, depth=depth, n_heads=n_heads)
        self.roi_stop_grad = roi_stop_grad
        # classifier consumes ROI gated image (1-ch). If in_ch>1, we can reduce via 1x1 conv or mean.
        self.reduce = nn.Conv2d(in_ch, 1, 1, bias=False) if in_ch>1 else nn.Identity()
        self.cls_backbone = TInceptionNet(in_ch=1, num_classes=num_cls)

    def forward(self, x):
        seg_logits = self.seg(x)  # B,1,H,W
        seg_prob = torch.sigmoid(seg_logits)
        roi_input = self.reduce(x)
        if self.roi_stop_grad:
            roi = roi_input * seg_prob.detach()
        else:
            roi = roi_input * seg_prob
        cls_logits = self.cls_backbone(roi)
        return seg_logits, cls_logits
