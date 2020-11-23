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




class GUIDEDRESNET18Model(nn.Module):
    def __init__(self, args):
        super().__init__() 
        self.__name__ = 'pncnn'
        self.args = args

        self.depth_unet = RESNET18_UNETModel(args, activation = 'softplus', input_channels=1)
        self.rgb_unet = RESNET18_UNETModel(args, activation = 'softplus')

    def forward(self, sample):

        out = self.rgb_unet(sample)
        x_dep = out['pred']
        c_dep = out['confidence']

        out = self.rgb_unet(sample)
        x_rgb = out['pred']
        c_rgb = out['confidence']

        alpha = c_rgb / (c_rgb + c_dep + 1e-6)

        xout = (1 - alpha) * x_dep + (alpha) * x_rgb
        
        output = {'pred': xout, 'confidence': c_dep, 'confidence_rgb': alpha, 'pred_rgb': x_rgb}

        return output
        
