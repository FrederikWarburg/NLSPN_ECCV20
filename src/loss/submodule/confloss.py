

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfLoss(nn.Module):
    
    def __init__(self, args):
        super().__init__()
                      
    def forward(self, pred, target, cout):    
        val_pixels = torch.ne(target, 0).float().cuda()
        err = F.smooth_l1_loss(pred*val_pixels, target*val_pixels, reduction='none')
        loss = err-cout*val_pixels+err*cout*val_pixels
        return torch.mean(loss)
