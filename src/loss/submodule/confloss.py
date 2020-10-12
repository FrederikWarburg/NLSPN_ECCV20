

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
                      
    def forward(self, outputs, target, cout, *args):    
        val_pixels = torch.ne(target, 0).float().cuda()
        err = F.smooth_l1_loss(outputs*val_pixels, target*val_pixels, reduction='none')
        loss = err-cout*val_pixels+err*cout*val_pixels
        return torch.mean(loss)
