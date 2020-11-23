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


class NCONV_RESNET18UNETModel(nn.Module):
    def __init__(self, args):
        super().__init__() 
        self.__name__ = 'ncnn_conf'

        self.args = args

        if self.args.input_conf == 'learned':
            self.conf_estimator = UNet(1, 1)

        self.nconv = NConvUNet(1, 1)

        self.rgb_unet = RESNET18_UNETModel(args)
        
    def forward(self, sample): 

        # unpack sample
        x0 = sample['dep']
        c0 = sample['confidence']
        xout, cout = self.nconv(x0, c0)

        out = self.rgb_unet(sample)
        x_rgb = out['pred']
        c_rgb = out['confidence']

        xout = xout * cout + x_rgb * c_rgb

        output = {'pred': xout, 'confidence': cout, 'confidence_rgb': c_rgb, 'pred_rgb': x_rgb}

        return output

