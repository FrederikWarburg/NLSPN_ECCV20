    

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):       
        val_pixels = (target>0).float().cuda()
        err = (target*val_pixels - outputs*val_pixels)**2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(torch.sqrt(loss/cnt))
 