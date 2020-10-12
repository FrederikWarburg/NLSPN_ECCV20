

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfDecayLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
                      
    def forward(self, pred, target, cout, epoch_num):           
        val_pixels = torch.ne(target, 0).float().cuda()
        err = F.smooth_l1_loss(pred*val_pixels, target*val_pixels, reduction='none')
        cert = cout*val_pixels - err*cout*val_pixels
        loss = err - (1/epoch_num) * cert
        return torch.mean(loss)

