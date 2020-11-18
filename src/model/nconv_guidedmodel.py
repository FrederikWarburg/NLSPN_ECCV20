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

class NCONV_GUIDEDModel(nn.Module):
    def __init__(self, args):
        super().__init__() 
        self.__name__ = 'ncnn_conf'

        self.args = args

        if self.args.input_conf == 'learned':
            self.conf_estimator = UNet(1, 1)

        self.nconv = NConvUNet(1, 1)

        self.rgb_unet = resnetUnet()
        
    def forward(self, sample): 

        # unpack sample
        x0 = sample['dep']
        c0 = sample['confidence']
        rgb = sample['rgb']

        x_rgb, c_rgb = self.rgb_unet(rgb)
        xout, cout = self.nconv(x0, c0)
        
        xout = xout * cout + x_rgb * c_rgb

        output = {'pred': xout, 'confidence': cout}

        return output

class resnetUnet(nn.Module):
    def __init__(self):
        super().__init__() 

        net = get_resnet18(False)
        self.upsampling = 'not_learnable'
        self.aggregate = 'sum'
        self.D_skip = 0

        self.conv1_rgb = torch.nn.Sequential(*[net.conv1, net.bn1, net.relu, net.maxpool]) #1/2
        self.conv2_rgb = net.layer1 #1/2
        self.conv3_rgb = net.layer2 #1/4
        self.conv4_rgb = net.layer3 #1/8
        self.conv5_rgb = net.layer4 #1/16
        
        self.bottleneck1 = conv_bn_relu(512, 512, kernel=3, stride=2, padding=1, bn=True, relu=True) # 1/32
        #self.bottleneck2 = conv_bn_relu(1024, 512, kernel=3, stride=1, padding=1, bn=True, relu=True) # 1/32

        # Decoder
        self.dec5_rgb = Upsample(512, self.D_skip * 512, 256, upsampling=self.upsampling, aggregate=self.aggregate) # 1/8
        self.dec4_rgb = Upsample(256, self.D_skip * 256, 128, upsampling=self.upsampling, aggregate=self.aggregate) # 1/4
        self.dec3_rgb = Upsample(128, self.D_skip * 128, 64,  upsampling=self.upsampling, aggregate=self.aggregate) # 1/2
        self.dec2_rgb = Upsample(64, self.D_skip * 64, 64,  upsampling=self.upsampling, aggregate=self.aggregate) # 1/2
        self.dec1_rgb = Upsample(64, 0, 64, upsampling=self.upsampling, bn=False) # 1/1

        # Depth Branch
        self.id_dec1_rgb = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1, bn=False, relu=True) # 1/1
        self.id_dec0_rgb = conv_bn_relu(64, 1, kernel=3, stride=1, padding=1, bn=False, relu=True, maxpool=False)

        # Confidence Branch
        self.conf_dec1_rgb = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1, bn=False, relu=True) # 1/1
        self.conf_dec0_rgb = conv_bn_relu(64, 1, kernel=3, stride=1, padding=1, bn=False, relu=True, maxpool=False)

    def forward(self, x): 

        # Encoding RGB
        fe1_rgb = self.conv1_rgb(x)
        fe2_rgb = self.conv2_rgb(fe1_rgb)
        fe3_rgb = self.conv3_rgb(fe2_rgb)
        fe4_rgb = self.conv4_rgb(fe3_rgb)
        fe5_rgb = self.conv5_rgb(fe4_rgb)

        # bottleneck
        bottleneck1_rgb = self.bottleneck1(fe5_rgb)
        #bottleneck2_rgb = self.bottleneck2(bottleneck1_rgb)
        
        # Decoding RGB
        fd5_rgb = self.dec5_rgb(bottleneck1_rgb, fe5_rgb)
        fd4_rgb = self.dec4_rgb(fd5_rgb, fe4_rgb)
        fd3_rgb = self.dec3_rgb(fd4_rgb, fe3_rgb)
        fd2_rgb = self.dec2_rgb(fd3_rgb, fe2_rgb)

        fd1_rgb = self.id_dec1_rgb(fd2_rgb)
        pred = self.id_dec0_rgb(fd1_rgb)

        conf1_rgb = self.conf_dec1_rgb(fd2_rgb)
        conf = F.softplus(self.conf_dec0_rgb(conf1_rgb))
        
        pred = _remove_extra_pad(pred, x)
        conf = _remove_extra_pad(conf, x)

        return pred, conf
        