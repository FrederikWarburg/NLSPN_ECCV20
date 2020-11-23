import torch
import torch.nn as nn

from .pnconv import UNetSP
from .pnconv import UNet
from .nconv import NConvUNet

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

from .common import get_resnet18, get_resnet34, _remove_extra_pad
from .unetmodel import Upsample, conv_bn_relu
from .resnet18_unetmodel import RESNET18_UNETModel




class PNCONV_UNGUIDEDModel(nn.Module):
    def __init__(self, args):
        super().__init__() 
        self.__name__ = 'pncnn'
        self.args = args

        if self.args.input_conf == 'learned':
            self.conf_estimator = UNet(1, 1)
        
        self.nconv = NConvUNet(1, 1)
        self.var_estimator = UNetSP(1, 1)

        self.rgb_unet = RESNET18_UNETModel(args, activation = 'softplus')

    def forward(self, sample):

        x0 = sample['dep']  # Use only depth
        
        if self.args.input_conf == 'learned':
            c0 = self.conf_estimator(x0)
        else:
            # binary or heuristic
            c0 = sample['confidence']

        xout, cout = self.nconv(x0, c0)
        cout = self.var_estimator(cout)

        out = self.rgb_unet(sample)
        x_rgb = out['pred']
        c_rgb = out['confidence']

        alpha = c_rgb / (c_rgb + cout + 1e-6)

        xout = (1 - alpha) * xout + (alpha) * x_rgb
        
        output = {'pred': xout, 'confidence': cout, 'confidence_rgb': alpha, 'pred_rgb': x_rgb}

        return output
        
