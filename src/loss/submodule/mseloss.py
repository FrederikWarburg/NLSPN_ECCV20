   

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        val_pixels = torch.ne(target, 0).float().cuda()
        loss = target*val_pixels - outputs*val_pixels
        return torch.mean(loss**2)
    