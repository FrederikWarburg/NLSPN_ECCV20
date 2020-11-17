import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math
import importlib
import sys

from .nconv import NConv2d

import torch
import torch.nn as nn
from .pnconv import UNet
from .nconv import NConvUNet


class NCONV_UNGUIDEDModel(nn.Module):
    def __init__(self, args):
        super().__init__() 
        self.__name__ = 'ncnn_conf'

        self.args = args

        if self.args.input_conf == 'learned':
            self.conf_estimator = UNet(1, 1)

        self.nconv = NConvUNet(1, 1)
        
    def forward(self, sample): 

        # unpack sample
        x0 = sample['dep']
        
        if self.args.input_conf == 'learned':
            c0 = self.conf_estimator(x0)
        else:
            # binary or heuristic
            c0 = sample['confidence']

        xout, cout = self.nconv(x0, c0)
        #out = torch.cat((xout, cout, c0), 1)
        
        output = {'pred': xout, 'confidence': cout}

        return output

