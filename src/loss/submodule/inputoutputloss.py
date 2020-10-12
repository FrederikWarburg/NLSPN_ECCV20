
import torch
import torch.nn as nn
import torch.nn.functional as F

class InputOutputLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
                      
    def forward(self, pred, target, inputs):           
        val_pixels = torch.ne(target, 0).float().cuda()
        err = F.smooth_l1_loss(pred*val_pixels, target*val_pixels, reduction='none')
        
        val_pixels = torch.ne(inputs, 0).float().cuda()
        inp_loss = F.smooth_l1_loss(pred*val_pixels, inputs*val_pixels, reduction='none')
        
        loss = err + 0.1 * inp_loss
        
        return torch.mean(loss)