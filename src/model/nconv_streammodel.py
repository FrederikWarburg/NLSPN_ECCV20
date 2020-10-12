import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math
import importlib
import sys

from .nconv_unguidedmodel import NCONV_UNGUIDEDModel

class NCONV_STREAMModel(nn.Module):

    def __init__(self, pos_fn=None):
        super().__init__() 
        
        # Import the unguided network
        self.d_net = NCONV_UNGUIDEDModel()
                
        self.d = nn.Sequential(
          nn.Conv2d(1,16,3,1,1),
          nn.ReLU(),
          nn.Conv2d(16,16,3,1,1),
          nn.ReLU(),
          nn.Conv2d(16,16,3,1,1),
          nn.ReLU(),
          nn.Conv2d(16,16,3,1,1),
          nn.ReLU(),
          nn.Conv2d(16,16,3,1,1),
          nn.ReLU(),
          nn.Conv2d(16,16,3,1,1),
          nn.ReLU(),                                              
        )#11,664 Params
        
        # RGB stream
        self.rgb = nn.Sequential(
          nn.Conv2d(4,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),                                            
        )#186,624 Params

        # Fusion stream
        self.fuse = nn.Sequential(
          nn.Conv2d(80,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,64,3,1,1),
          nn.ReLU(),
          nn.Conv2d(64,32,3,1,1),
          nn.ReLU(),
          nn.Conv2d(32,32,3,1,1),
          nn.ReLU(),
          nn.Conv2d(32,32,3,1,1),
          nn.ReLU(),
          nn.Conv2d(32,1,1,1),
        )# 156,704 Params
            
        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

                
            
    def forward(self, sample ):  
        # unpack sample
        x0_d = sample['dep']
        c0 = sample['confidence']
        x0_rgb = sample['rgb']

        # Depth Network
        output = self.d_net(x0_d, c0)

        xout_d = output['dep']
        cout_d = output['confidence']
        
        xout_d = self.d(xout_d)
        
        self.xout_d = xout_d
        self.cout_d = cout_d
                
        # RGB network
        xout_rgb = self.rgb(torch.cat((x0_rgb, cout_d),1))
        self.xout_rgb = xout_rgb
        
        # Fusion Network
        xout = self.fuse(torch.cat((xout_rgb, xout_d),1))
        
        output = {'pred': xout, 'confidence': cout_d}

        return output
       