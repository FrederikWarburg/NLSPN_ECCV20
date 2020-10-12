import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, pred, target):

        val_pixels = torch.ne(target, 0).float().detach()
        loss = F.smooth_l1_loss(pred*val_pixels, target*val_pixels, reduction='none')
        return torch.mean(loss)