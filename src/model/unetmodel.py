

import torch
from torch import nn
from .visualtransformer import VisualTransformer
from .common import get_resnet18, get_resnet34, _remove_extra_pad
from torchvision.models.resnet import BasicBlock
from torchvision import models
import math
from .attention_module.attention_module import build_attention_module, build_simple_attention_module

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
        layers.append(nn.Upsample(mode='bilinear', scale_factor=2))
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
            if self.attention_type == "VT":
                self.D_guide = 3
            else:
                self.D_guide = 2
        elif self.guide == 'vt_only':
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

        if 'guided' in self.supervision:

            # Encoder
            self.conv1_rgb = torch.nn.Sequential(*[net.conv1, net.bn1, net.relu]) #1/2
            self.conv2_rgb = net.layer1 #1/2
            self.conv3_rgb = net.layer2 #1/4
            self.conv4_rgb = net.layer3 #1/8
            self.conv5_rgb = net.layer4 #1/16
            
            self.bottleneck1 = conv_bn_relu(512, 1024, kernel=3, stride=2, padding=1, bn=True, relu=True) # 1/32
            self.bottleneck2 = conv_bn_relu(1024, 512, kernel=3, stride=1, padding=1, bn=True, relu=True) # 1/32

            if self.attention_type == 'attention':
                self.multihead_attn = build_simple_attention_module(512, 512)
            else:
                # Decoder
                self.dec5_rgb = Upsample(512, self.D_skip * 512, 256, upsampling=self.upsampling, aggregate=self.aggregate) # 1/8
                self.dec4_rgb = Upsample(256, self.D_skip * 256, 128, upsampling=self.upsampling, aggregate=self.aggregate) # 1/4
                self.dec3_rgb = Upsample(128, self.D_skip * 128, 64,  upsampling=self.upsampling, aggregate=self.aggregate) # 1/2
                self.dec2_rgb = Upsample(64, self.D_skip * 64, 64,  upsampling=self.upsampling, aggregate=self.aggregate) # 1/2

                if 'rgb' in self.supervision:
                    self.dec1_rgb = Upsample(64, 0, 64, upsampling=self.upsampling, bn=False) # 1/1

                    # Depth Branch
                    self.id_dec1_rgb = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1, bn=False, relu=True) # 1/1
                    self.id_dec0_rgb = conv_bn_relu(64, 1, kernel=3, stride=1, padding=1, bn=False, relu=True, maxpool=False)
                
                if self.attention_type == 'VT':
                    vt1 = VisualTransformer(L=self.num_tokens[0], CT=self.token_size[0], C=64, size = 512, num_downsample = 4, head=self.num_heads[0], groups=self.groups[0], kqv_groups=self.kqv_groups[0], dynamic=False)
                    vt2 = VisualTransformer(L=self.num_tokens[1], CT=self.token_size[1], C=128, size = 256, num_downsample = 2,head=self.num_heads[1], groups=self.groups[1], kqv_groups=self.kqv_groups[1], dynamic=False)
                    vt3 = VisualTransformer(L=self.num_tokens[2], CT=self.token_size[2], C=256, size = 128, num_downsample = 2,head=self.num_heads[2], groups=self.groups[2], kqv_groups=self.kqv_groups[2], dynamic=False)
                    vt4 = VisualTransformer(L=self.num_tokens[3], CT=self.token_size[3], C=512, size = 64, num_downsample = 2,head=self.num_heads[3], groups=self.groups[3], kqv_groups=self.kqv_groups[3], dynamic=False)
                else:
                    vt1 = None
                    vt2 = None
                    vt3 = None
                    vt4 = None

                self.guide1 = Guide(64, self.D_guide * 64, 64, vt1, aggregate=self.guide)
                self.guide2 = Guide(64, self.D_guide * 128, 128, vt2, aggregate=self.guide)
                self.guide3 = Guide(128, self.D_guide * 256, 256, vt3, aggregate=self.guide)
                self.guide4 = Guide(256, self.D_guide * 512, 512, vt4, aggregate=self.guide)
           

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
        
        if 'guided' in self.supervision:
            # Encoding RGB
            fe1_rgb = self.conv1_rgb(rgb)
            fe2_rgb = self.conv2_rgb(fe1_rgb)
            fe3_rgb = self.conv3_rgb(fe2_rgb)
            fe4_rgb = self.conv4_rgb(fe3_rgb)
            fe5_rgb = self.conv5_rgb(fe4_rgb)

            # bottleneck
            bottleneck1_rgb = self.bottleneck1(fe5_rgb)
            bottleneck2_rgb = self.bottleneck2(bottleneck1_rgb)

            if hasattr(self, "dec5_rgb"):
                # Decoding RGB
                fd5_rgb = self.dec5_rgb(bottleneck2_rgb, fe5_rgb)
                fd4_rgb = self.dec4_rgb(fd5_rgb, fe4_rgb)
                fd3_rgb = self.dec3_rgb(fd4_rgb, fe3_rgb)
                fd2_rgb = self.dec2_rgb(fd3_rgb, fe2_rgb)

                if 'rgb' in self.supervision:
                    fd1_rgb = self.dec1_rgb(fd2_rgb)
        
        ###
        # DEPTH UNET
        ###
        
        # Encoding Depth
        fe1_dep = self.conv1_dep(dep)
        fe2_dep = self.conv2_dep(fe1_dep)

        if hasattr(self, "guide1"):
            fe2_dep = self.guide1(fe2_dep, fd2_rgb)

        fe3_dep = self.conv3_dep(fe2_dep)
        if hasattr(self, "guide2"):
            fe3_dep = self.guide2(fe3_dep, fd3_rgb)
 
        fe4_dep = self.conv4_dep(fe3_dep)
        if hasattr(self, "guide3"):
            fe4_dep = self.guide3(fe4_dep, fd4_rgb)

        fe5_dep = self.conv5_dep(fe4_dep)
        if hasattr(self, "guide4"):
            fe5_dep = self.guide4(fe5_dep, fd5_rgb)

        # bottleneck
        bottleneck1_dep = self.bottleneck1_dep(fe5_dep)
        bottleneck2_dep = self.bottleneck2_dep(bottleneck1_dep)

        if hasattr(self, "multihead_attn"):
            bottleneck2_dep = self.multihead_attn(bottleneck2_dep, bottleneck2_rgb)
        
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

        # Confidence Decoding
        if  'confidence' in self.supervision:
            cf_fd1 = self.cf_dec1(fd1_dep)
            confidence = self.cf_dec0(cf_fd1)
            confidence = _remove_extra_pad(confidence, dep)

        # RGB Decoding
        if  'rgb' in self.supervision:
            id_fd1_rgb = self.id_dec1_rgb(fd1_rgb)
            pred_rgb = self.id_dec0_rgb(id_fd1_rgb)
            pred_rgb = _remove_extra_pad(pred_rgb, dep)
        
        output = {'pred': pred}

        if 'confidence' in self.supervision:
            output['confidence'] = confidence
        else:
            output['confidence'] = None

        if self.attention_type == 'VT':
            output['vt1'] = self.guide1.vt
            output['vt2'] = self.guide2.vt
            output['vt3'] = self.guide3.vt
            output['vt4'] = self.guide4.vt
        elif self.attention_type == 'attention':
            output['num_layers'] = len(self.multihead_attn.transformer.decoder.layers)
            output['size'] = bottleneck2_rgb.shape[-2:]
            for i in range(output['num_layers']):
                output['self_attn_map_{}'.format(i)] = self.multihead_attn.transformer.decoder.layers[i].attn_map1
                output['attn_map_{}'.format(i)] = self.multihead_attn.transformer.decoder.layers[i].attn_map2

        if 'rgb' in self.supervision:
            output['pred_rgb'] = pred_rgb
            
        return output

if __name__ == "__main__":
    
    rgb = torch.FloatTensor(torch.randn((1,3, 1216,300)))
    dep = torch.FloatTensor(torch.randn((1,1, 1216,300)))

    sample = {'rgb':rgb,'dep':dep}

    model = UNETModel()

    out = model(sample)
    print(out['pred'].shape)
