import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedL1Loss(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, pred, targetargs):
        
        val_pixels = torch.ne(target, 0).float().detach()
        return F.l1_loss(pred*val_pixels, target*val_pixels)