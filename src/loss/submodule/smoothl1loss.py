
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothL1Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        val_pixels = torch.ne(target, 0).float().cuda()
        loss = F.smooth_l1_loss(outputs*val_pixels, target*val_pixels, reduction='none')
        return torch.mean(loss)

    