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



import torchvision
resnet = torchvision.models.resnet.resnet50(pretrained=True)



class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """

        x = self.upsample(up_x)
        x = _remove_extra_pad(x, down_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class RESNET18_UNETModel(nn.Module):
    

    def __init__(self, args, activation = 'sigmoid', input_channels = 3):
        super().__init__()
        resnet = torchvision.models.resnet.resnet18(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_channels = input_channels
        if self.input_channels == 3:
            self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        else: 
            layers = []
            layers.append(nn.Conv2d(input_channels, 64, 7, 1, 1))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU())
            self.input_block = nn.Sequential(*layers)

        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(512, 512)
        up_blocks.append(UpBlockForUNetWithResNet(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet(256, 128))
        up_blocks.append(UpBlockForUNetWithResNet(128, 64))
        up_blocks.append(UpBlockForUNetWithResNet(in_channels=64 + 64, out_channels=64,
                                                    up_conv_in_channels=64, up_conv_out_channels=64))
        #up_blocks.append(UpBlockForUNetWithResNet(in_channels=64 + 3, out_channels=64,
        #                                            up_conv_in_channels=128, up_conv_out_channels=64))
        self.upsample = nn.Upsample(mode='bilinear', scale_factor=2)
        self.up_blocks = nn.ModuleList(up_blocks)

        self.pred = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.conf = nn.Conv2d(64, 1, kernel_size=1, stride=1)

        self.activation = activation

        self.DEPTH = 6

    def forward(self, sample, with_output_feature_map=False):
        
        if self.input_channels == 3:
            x = sample['rgb']
        else:
            x = sample['depth']
        
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (self.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{self.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        pred = self.pred(x)
        conf = self.conf(x)

        if self.activation == 'sigmoid':
            conf = F.sigmoid(conf)
        elif self.activation == 'softplus':
            conf = F.softplus(conf)

        pred = self.upsample(pred)
        conf = self.upsample(conf)

        del pre_pools
        if with_output_feature_map:
            return pred, conf, output_feature_map
        
        out = {'pred':pred, 'confidence': conf}
        return out