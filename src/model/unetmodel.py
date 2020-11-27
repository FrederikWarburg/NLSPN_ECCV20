

import torch
from torch import nn
from .visualtransformer import VisualTransformer
from .common import get_resnet18, get_resnet34, _remove_extra_pad
from torchvision.models.resnet import BasicBlock
from torchvision import models
import math
from .attention_module.simple_attention import build_simple_attention_module

def _concat(fd, fe, vt=None, aggregate='cat', dim=1):
    
    fd = _remove_extra_pad(fd, fe)

    if vt is None:
        if aggregate == 'cat':
            f = torch.cat((fd, fe), dim=dim)
        elif aggregate == 'sum':
            f = fd + fe
    else:
        if aggregate == 'cat':
            f = torch.cat((fd, fe, vt), dim=dim)
        elif aggregate == 'sum':
            f = fd + fe + vt 

    return f


def _upsampling(ch_in, ch_out, bn=True, relu=True, upsampling = 'learnable'):

    layers = []
    if upsampling == 'learnable':
        layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel=3, stride=2, padding=0, output_padding=0))
    else:
        layers.append(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False))
        layers.append(nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1))
    
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers

def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True, relu=True, maxpool=False):

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    if maxpool:
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))

    layers = nn.Sequential(*layers)

    return layers


def double_conv(ch_in, ch_mid, ch_out, bn=True, relu=True):

    layers = []

    layers.append(conv_bn_relu(ch_in, ch_mid, kernel=3, stride=1, padding=1, bn=relu, relu=relu))
    layers.append(conv_bn_relu(ch_mid, ch_out, kernel=3, stride=1, padding=1, bn=relu, relu=relu))

    layers = nn.Sequential(*layers)

    return layers


class Upsample(nn.Module):
    def __init__(self, ch_in1, ch_in2, ch_out, bn=True, relu=True, upsampling = 'learnable', aggregate = 'cat'):
        super(Upsample, self).__init__()

        self.aggregate = aggregate

        self.upsampling = _upsampling(ch_in1, ch_in1, bn=bn, relu=relu, upsampling = upsampling)
        self.conv = double_conv(ch_in1+ch_in2, (ch_in1+ch_in2)//2, ch_out, bn=bn, relu=relu)

    def forward(self, x, x1 = None):
        
        x = self.upsampling(x)
        if x1 is not None:
            x = _concat(x, x1, aggregate=self.aggregate, dim=1)
        x = self.conv(x)

        return x

class Guide(nn.Module):
    def __init__(self, ch_in1, ch_in2, ch_out, vt = None, bn=True, relu=True, aggregate = 'cat'):
        super(Guide, self).__init__()

        self.aggregate = aggregate

        self.conv1 = conv_bn_relu(ch_in1, ch_out, kernel=1, stride=1, padding=0, bn=bn, relu=relu, maxpool=False)
        self.conv2 = conv_bn_relu(ch_in2, ch_out, kernel=1, stride=1, padding=0, bn=bn, relu=relu, maxpool=False)
        self.vt = vt

    def forward(self, fe_dep, fd_rgb):

        fd_rgb = self.conv1(fd_rgb)
        if self.vt is not None:
            proj_ = self.vt(fd_rgb, fe_dep)
            if self.aggregate == 'vt_only':
                x = _concat(proj_, fe_dep, aggregate='cat') 
            else:   
                x = _concat(fd_rgb, fe_dep, proj_, aggregate=self.aggregate)
        else:
            x = _concat(fd_rgb, fe_dep, aggregate=self.aggregate)

        x = self.conv2(x)

        return x

class UNETModel(nn.Module):
    def __init__(self, args = None):
        super(UNETModel, self).__init__()

        self.args = args

        self.network = self.args.network
        self.aggregate = self.args.aggregate
        self.guide = self.args.guide
        self.upsampling = 'not_learnable' #'leanable' # not_learnable
        self.attention_type = self.args.attention_type
        self.supervision = self.args.supervision
        self.num_tokens = [int(l) for l in self.args.num_tokens.split("+")]
        self.token_size = [int(l) for l in self.args.token_size.split("+")]
        self.num_heads = [int(l) for l in self.args.num_heads.split("+")]
        self.groups = [int(l) for l in self.args.groups.split("+")]
        self.kqv_groups = [int(l) for l in self.args.kqv_groups.split("+")]


        if self.guide == 'cat':
            self.D_guide = 2
        elif self.guide == 'sum':
            self.D_guide = 1
        elif self.guide == 'none':
            self.D_guide = 1 
        else:
            raise NotImplementedError    

        if self.aggregate == 'cat':
            self.D_skip = 1
        elif self.aggregate == 'sum':
            self.D_skip = 0
        else:
            raise NotImplementedError       
        
        if self.network == 'resnet18':
            net = get_resnet18(False)
        elif self.network == 'resnet34':
            net = get_resnet34(False)
        else:
            raise NotImplementedError

        ####
        # RGB Stream
        ####


        # Encoder
        self.conv1_rgb = torch.nn.Sequential(*[net.conv1, net.bn1, net.relu]) #1/2
        self.conv2_rgb = net.layer1 #1/2
        self.conv3_rgb = net.layer2 #1/4
        self.conv4_rgb = net.layer3 #1/8
        self.conv5_rgb = net.layer4 #1/16
        
        self.bottleneck1 = conv_bn_relu(512, 1024, kernel=3, stride=2, padding=1, bn=True, relu=True) # 1/32
        self.bottleneck2 = conv_bn_relu(1024, 512, kernel=3, stride=1, padding=1, bn=True, relu=True) # 1/32

        # Decoder
        self.dec5_rgb = Upsample(512, self.D_skip * 512, 256, upsampling=self.upsampling, aggregate=self.aggregate) # 1/8
        self.dec4_rgb = Upsample(256, self.D_skip * 256, 128, upsampling=self.upsampling, aggregate=self.aggregate) # 1/4
        self.dec3_rgb = Upsample(128, self.D_skip * 128, 64,  upsampling=self.upsampling, aggregate=self.aggregate) # 1/2
        self.dec2_rgb = Upsample(64, self.D_skip * 64, 64,  upsampling=self.upsampling, aggregate=self.aggregate) # 1/2

        self.guide1 = Guide(64, self.D_guide * 64, 64, None, aggregate=self.guide)
        self.guide2 = Guide(64, self.D_guide * 128, 128, None, aggregate=self.guide)
        self.guide3 = Guide(128, self.D_guide * 256, 256, None, aggregate=self.guide)
        self.guide4 = Guide(256, self.D_guide * 512, 512, None, aggregate=self.guide)
           
        ####
        # Depth Stream
        ####
       
        # Encoder
        self.conv1_dep = conv_bn_relu(1, 64, kernel=7, stride=2, padding=3, bn=True, relu=True, maxpool=False) # 1/2
        self.conv2_dep = net.layer1 # 1/2
        self.conv3_dep = net.layer2 # 1/4
        self.conv4_dep = net.layer3 # 1/8
        self.conv5_dep = net.layer4 # 1/16

        self.bottleneck1_dep = conv_bn_relu(512, 1024, kernel=3, stride=2, padding=1, bn=True, relu=True) # 1/32
        self.bottleneck2_dep = conv_bn_relu(1024, 512, kernel=3, stride=1, padding=1, bn=True, relu=True) # 1/32

        # Decoder
        self.dec5_dep = Upsample(512, self.D_skip * 512, 256,  upsampling=self.upsampling, aggregate=self.aggregate) # 1/16
        self.dec4_dep = Upsample(256, self.D_skip * 256, 128, upsampling=self.upsampling, aggregate=self.aggregate) # 1/8
        self.dec3_dep = Upsample(128, self.D_skip * 128, 64,  upsampling=self.upsampling, aggregate=self.aggregate) # 1/4
        self.dec2_dep = Upsample(64, self.D_skip * 64, 64, upsampling=self.upsampling, aggregate=self.aggregate) # 1/2
        self.dec1_dep = Upsample(64, 0, 64, upsampling=self.upsampling, bn=False) # 1/1

        # Depth Branch
        self.id_dec1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1, bn=False, relu=True) # 1/1
        self.id_dec0 = conv_bn_relu(64, 1, kernel=3, stride=1, padding=1, bn=False, relu=True, maxpool=False)

        if 'confidence' in self.supervision:
            # Confidence Branch
            self.cf_dec1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1, bn=False, relu=True) # 1/1
            self.cf_dec0 = nn.Sequential(
                nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
                nn.Softplus()
            )

    def forward(self, sample):

        rgb = sample['rgb']
        dep = sample['dep']
        output = {}
        
        # Encoding RGB
        fe1_rgb = self.conv1_rgb(rgb)
        fe2_rgb = self.conv2_rgb(fe1_rgb)
        fe3_rgb = self.conv3_rgb(fe2_rgb)
        fe4_rgb = self.conv4_rgb(fe3_rgb)
        fe5_rgb = self.conv5_rgb(fe4_rgb)

        # bottleneck
        bottleneck1_rgb = self.bottleneck1(fe5_rgb)
        bottleneck2_rgb = self.bottleneck2(bottleneck1_rgb)

        # Decoding RGB
        fd5_rgb = self.dec5_rgb(bottleneck2_rgb, fe5_rgb)
        fd4_rgb = self.dec4_rgb(fd5_rgb, fe4_rgb)
        fd3_rgb = self.dec3_rgb(fd4_rgb, fe3_rgb)
        fd2_rgb = self.dec2_rgb(fd3_rgb, fe2_rgb)

        ###
        # DEPTH UNET
        ###
    
        # Encoding Depth
        fe1_dep = self.conv1_dep(dep)

        fe2_dep = self.conv2_dep(fe1_dep)
        fe2_dep = self.guide1(fe2_dep, fd2_rgb)

        fe3_dep = self.conv3_dep(fe2_dep)
        fe3_dep = self.guide2(fe3_dep, fd3_rgb)

        fe4_dep = self.conv4_dep(fe3_dep)
        fe4_dep = self.guide3(fe4_dep, fd4_rgb)

        fe5_dep = self.conv5_dep(fe4_dep)
        fe5_dep = self.guide4(fe5_dep, fd5_rgb)

        # bottleneck
        bottleneck1_dep = self.bottleneck1_dep(fe5_dep)
        bottleneck2_dep = self.bottleneck2_dep(bottleneck1_dep)

        # Decoding Depth
        fd5_dep = self.dec5_dep(bottleneck2_dep, fe5_dep)
        fd4_dep = self.dec4_dep(fd5_dep, fe4_dep)
        fd3_dep = self.dec3_dep(fd4_dep, fe3_dep)
        fd2_dep = self.dec2_dep(fd3_dep, fe2_dep)
        fd1_dep = self.dec1_dep(fd2_dep)

        ###
        # PREDICTION HEADS
        ###

        # Depth Decoding
        id_fd1 = self.id_dec1(fd1_dep)
        pred = self.id_dec0(id_fd1)
        pred = _remove_extra_pad(pred, dep)
        output['pred'] = pred

        # Confidence Decoding
        if  'confidence' in self.supervision:
            cf_fd1 = self.cf_dec1(fd1_dep)
            confidence = self.cf_dec0(cf_fd1)
            confidence = _remove_extra_pad(confidence, dep)
            output['confidence'] = confidence

        return output

if __name__ == "__main__":
    
    rgb = torch.FloatTensor(torch.randn((1,3, 1216,300)))
    dep = torch.FloatTensor(torch.randn((1,1, 1216,300)))

    sample = {'rgb':rgb,'dep':dep}

    model = UNETModel()

    out = model(sample)
    print(out['pred'].shape)
