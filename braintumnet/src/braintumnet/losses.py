import torch, torch.nn as nn, torch.nn.functional as F

def dice_loss_with_logits(logits, target, eps=1e-6):
    pred = torch.sigmoid(logits)
    num = 2 * (pred * target).sum(dim=(2,3))
    den = (pred.pow(2).sum(dim=(2,3)) + target.pow(2).sum(dim=(2,3))) + eps
    dice = 1 - (num + eps) / den
    return dice.mean()

class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, seg_logits, seg_mask):
        return dice_loss_with_logits(seg_logits, seg_mask) + self.bce(seg_logits, seg_mask)

class MultiTaskLoss(nn.Module):
    def __init__(self, seg_w=1.0, cls_w=0.7):
        super().__init__()
        self.seg_w = seg_w
        self.cls_w = cls_w
        self.seg_loss = DiceCELoss()
        self.cls_loss = nn.CrossEntropyLoss()
    def forward(self, seg_logits, seg_mask, cls_logits, cls_label):
        l_seg = self.seg_loss(seg_logits, seg_mask)
        l_cls = self.cls_loss(cls_logits, cls_label)
        return self.seg_w * l_seg + self.cls_w * l_cls, l_seg.detach(), l_cls.detach()
