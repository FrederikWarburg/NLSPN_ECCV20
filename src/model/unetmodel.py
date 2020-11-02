

import torch
from .visualtransformer import *
from .common import get_resnet18, get_resnet34
from torchvision.models.resnet import BasicBlock
from torchvision import models

def _guide( fd_rgb, fe_rgb, fe_dep, guide = 'cat', dim = 1):

    if guide == 'cat':
        return _concat(_concat(fd_rgb, fe_rgb, aggregate='cat', dim=1), fe_dep, aggregate='cat', dim=1)
    elif guide == 'sum':
        return _concat(_concat(fd_rgb, fe_rgb, aggregate='sum', dim=1), fe_dep, aggregate='sum', dim=1)
    elif guide == 'none':
        return fe_dep

def _concat(fd, fe, aggregate='cat', dim=1):
    
    fd = _remove_extra_pad(fd, fe)

    if aggregate == 'cat':
        f = torch.cat((fd, fe), dim=dim)
    elif aggregate == 'sum':
        f = fd + fe

    return f

def _remove_extra_pad(fd, fe, dim=1):

    # Decoder feature may have additional padding
    _, _, Hd, Wd = fd.shape
    _, _, He, We = fe.shape

    if abs(Hd - He) > 1 or abs(Wd - We) > 1:
        print("warning", fd.shape, fe.shape)

    # Remove additional padding
    if Hd > He:
        h = Hd - He
        fd = fd[:, :, :-h, :]

    if Wd > We:
        w = Wd - We
        fd = fd[:, :, :, :-w]

    return fd

def _make_layer(inplanes, planes, blocks=1, stride=1):
    downsample = None
    block = BasicBlock

    if stride != 1 or inplanes != planes * block.expansion:
        downsample = torch.nn.Sequential(
            torch.nn.Conv2d(inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            torch.nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes))

    return torch.nn.Sequential(*layers)


def _upsampling(ch_in, ch_out, kernel, bn=True, relu=True, upsampling = 'learnable'):

    layers = []
    if upsampling == 'learnable':
        layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride=2, padding=0, output_padding=0))
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
    def __init__(self, ch_in1, ch_in2, ch_out, kernel, stride=1, padding=0, output_padding=0, bn=True, relu=True, upsampling = 'learnable', aggregate = 'cat'):
        super(Upsample, self).__init__()


        self.aggregate = aggregate

        self.upsampling = _upsampling(ch_in1, ch_in1, kernel, bn=bn, relu=relu, upsampling = upsampling)
        self.conv = double_conv(ch_in1+ch_in2, ch_in1, ch_out, bn=bn, relu=relu)


    def forward(self, x, x1 = None):
        print("x", x.shape)
        x = self.upsampling(x)
        print("x!", x.shape)
        if x1 is not None:
            print("hello")
            print(x.shape, x1.shape)
            x = _concat(x, x1, aggregate=self.aggregate, dim=1)
            print("cat x", x.shape)
        print("out", x.shape)            
        x = self.conv(x)
        print("out", x.shape)
        return x

class UNETModel(nn.Module):
    def __init__(self, args = None):
        super(UNETModel, self).__init__()

        self.args = args

        self.network = self.args.network
        self.aggregate = self.args.aggregate
        self.guide = self.args.guide
        self.upsampling = 'not_learnable' #'leanable' # not_learnable

        if self.guide == 'cat':
            self.D_guide = 2
        elif self.guide == 'sum':
            self.D_guide = 0
        elif self.guide == 'none':
            self.D_guide = 0
        else:
            raise NotImplementedError    

        if self.aggregate == 'cat':
            self.D_skip = 1
        elif self.aggregate == 'sum':
            self.D_skip = 0
        else:
            raise NotImplementedError       
        
        if self.network == 'resnet18':
            net = get_resnet18(not self.args.from_scratch)
        elif self.network == 'resnet34':
            net = get_resnet34(not self.args.from_scratch)
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
        #self.dec5_rgb = Upsample(1024, 0, 512, kernel=3, stride=2, padding=1, output_padding=1, upsampling=self.upsampling,aggregate=self.aggregate) # 1/16
        self.dec4_rgb = Upsample(512, self.D_skip * 512, 256, kernel=3, stride=2, padding=1, output_padding=1, upsampling=self.upsampling, aggregate=self.aggregate) # 1/8
        self.dec3_rgb = Upsample(256, self.D_skip * 256, 128, kernel=3, stride=2, padding=1, output_padding=1, upsampling=self.upsampling,aggregate=self.aggregate) # 1/4
        self.dec2_rgb = Upsample(128, self.D_skip * 128, 64, kernel=3, stride=2, padding=1, output_padding=1, upsampling=self.upsampling, aggregate=self.aggregate) # 1/2
        self.dec1_rgb = Upsample(64, self.D_skip * 64, 64, kernel=3, stride=2, padding=1, output_padding=1, upsampling=self.upsampling, aggregate=self.aggregate) # 1/2

        ####
        # Depth Stream
        ####

        # Encoder
        self.conv1_dep = conv_bn_relu(1, 64, kernel=7, stride=2, padding=3, bn=True, relu=True, maxpool=False) # 1/2
        if self.aggregate == 'sum':
            self.conv2_dep = net.layer1 # 1/2
            self.conv3_dep = net.layer2 # 1/4
            self.conv4_dep = net.layer3 # 1/8
            self.conv5_dep = net.layer4 # 1/16
        elif self.aggregate == 'cat':
            self.conv2_dep = _make_layer(64 + self.D_guide * 64, 64, stride=1, blocks=2) # 1/2
            self.conv3_dep = _make_layer(64 + self.D_guide * 64, 128, stride=2, blocks=2) # 1/4
            self.conv4_dep = _make_layer(128 + self.D_guide * 128, 256, stride=2, blocks=2) # 1/8
            self.conv5_dep = _make_layer(256 + self.D_guide * 256, 512, stride=2, blocks=2) # 1/16
        self.conv6_dep = conv_bn_relu(512, 1024, kernel=3, stride=2, padding=1, bn=True, relu=True, maxpool=False) # 1/32

        # Decoder
        self.dec5_dep = Upsample(1024, 0, 512, kernel=3, stride=2, padding=1, output_padding=1, upsampling=self.upsampling, aggregate=self.aggregate) # 1/16
        self.dec4_dep = Upsample(512, self.D_skip * 512, 256, kernel=3, stride=2, padding=1, output_padding=1, upsampling=self.upsampling, aggregate=self.aggregate) # 1/8
        self.dec3_dep = Upsample(256, self.D_skip * 256, 128, kernel=3, stride=2, padding=1, output_padding=1, upsampling=self.upsampling, aggregate=self.aggregate) # 1/4
        self.dec2_dep = Upsample(128, self.D_skip * 128, 64, kernel=3, stride=2, padding=1, output_padding=1, upsampling=self.upsampling, aggregate=self.aggregate) # 1/2
        self.dec1_dep = Upsample(64, self.D_skip * 64, 64, kernel=3, stride=2, padding=1, output_padding=1, upsampling=self.upsampling, aggregate=self.aggregate) # 1/1

        # Depth Branch
        self.id_dec1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1, bn=False, relu=True) # 1/1
        self.id_dec0 = conv_bn_relu(64, 1, kernel=3, stride=1, padding=1, bn=False, relu=True, maxpool=False)

        # Confidence Branch
        self.cf_dec1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1, bn=False, relu=True) # 1/1
        self.cf_dec0 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Softplus()
        )

        ####
        # VISUAL TRANSFORMER
        ####

        if args.attention_type == 'VT':
            self.vt1 = VisualTransformer(L=args.num_tokens, CT=args.token_size, C=64, size = 512, num_downsample = 6, head=args.num_heads, groups=args.groups, kqv_groups=args.kqv_groups, dynamic=False)
            self.vt2 = VisualTransformer(L=args.num_tokens, CT=args.token_size, C=128, size = 256, num_downsample = 5,head=args.num_heads, groups=args.groups, kqv_groups=args.kqv_groups, dynamic=False)
            self.vt3 = VisualTransformer(L=args.num_tokens, CT=args.token_size, C=256, size = 128, num_downsample = 4,head=args.num_heads, groups=args.groups, kqv_groups=args.kqv_groups, dynamic=False)
            self.vt4 = VisualTransformer(L=args.num_tokens, CT=args.token_size, C=512, size = 64, num_downsample = 3,head=args.num_heads, groups=args.groups, kqv_groups=args.kqv_groups, dynamic=False)


    def forward(self, sample):

        rgb = sample['rgb']
        dep = sample['dep']
        
        # Encoding RGB
        fe1_rgb = self.conv1_rgb(rgb)
        fe2_rgb = self.conv2_rgb(fe1_rgb)
        fe3_rgb = self.conv3_rgb(fe2_rgb)
        fe4_rgb = self.conv4_rgb(fe3_rgb)
        fe5_rgb = self.conv5_rgb(fe4_rgb)

        # bottlenect
        bottleneck = self.bottleneck1(fe5_rgb)
        bottleneck = self.bottleneck2(bottleneck)

        # Decoding RGB
        fd4_rgb = self.dec4_rgb(bottleneck, fe5_rgb)
        fd3_rgb = self.dec3_rgb(fd4_rgb, fe4_rgb)
        fd2_rgb = self.dec2_rgb(fd3_rgb, fe3_rgb)
        fd1_rgb = self.dec1_rgb(fd2_rgb, fe2_rgb)
        
        ###
        # DEPTH UNET
        ###
        print("depth prediction")
        
        # Encoding Depth
        fe1_dep = self.conv1_dep(dep)
        print("1", fd1_rgb.shape, fe1_rgb.shape, fe1_dep.shape)
        fe2_dep = self.conv2_dep(_guide(fd1_rgb, fe1_rgb, fe1_dep, guide=self.guide, dim=1))
        print("2", fe2_dep.shape, fd2_rgb.shape)
        if self.args.attention_type == 'VT':
            # we need first to remove some extra padding which is added in the decoding stage
            fd2_rgb = _remove_extra_pad(fd2_rgb, fe2_dep)
            fe2_dep = self.vt1(fd2_rgb, fe2_dep)

        fe3_dep = self.conv3_dep(_guide(fd2_rgb, fe2_rgb, fe2_dep, guide=self.guide, dim=1))

        if self.args.attention_type == 'VT':
            # we need first to remove some extra padding which is added in the decoding stage
            fd3_rgb = _remove_extra_pad(fd3_rgb, fe3_dep)
            fe3_dep = self.vt2(fd3_rgb, fe3_dep)

        fe4_dep = self.conv4_dep(_guide(fd3_rgb, fe3_rgb, fe3_dep, guide=self.guide, dim=1))

        if self.args.attention_type == 'VT':
            # we need first to remove some extra padding which is added in the decoding stage
            fd4_rgb = _remove_extra_pad(fd4_rgb, fe4_dep)
            fe4_dep = self.vt3(fd4_rgb, fe4_dep)

        fe5_dep = self.conv5_dep(_guide(fd4_rgb, fe4_rgb, fe4_dep, guide=self.guide, dim=1))

        if self.args.attention_type == 'VT':
            # we need first to remove some extra padding which is added in the decoding stage
            fd5_rgb = _remove_extra_pad(fd5_rgb, fe5_dep)
            fe5_dep = self.vt4(fd5_rgb, fe5_dep)

        fe6_dep = self.conv6_dep(fe5_dep)
        
        # Decoding Depth
        fd5_dep = self.dec5_dep(fe6_dep)
        fd4_dep = self.dec4_dep(fd5_dep, fe5_dep)
        fd3_dep = self.dec3_dep(fd4_dep, fe4_dep)
        fd2_dep = self.dec2_dep(fd3_dep, fe3_dep)   
        fd1_dep = self.dec1_dep(fd2_dep, fe2_dep)

        ###
        # PREDICTION HEADS
        ###

        # Depth Decoding
        id_fd1 = self.id_dec1(fd1_dep)
        pred = self.id_dec0(id_fd1)

        # Confidence Decoding
        cf_fd1 = self.cf_dec1(fd1_dep)
        confidence = self.cf_dec0(cf_fd1)

        pred = _remove_extra_pad(pred, dep)
        confidence = _remove_extra_pad(confidence, dep)

        if self.args.attention_type == 'VT':
            output = {'pred': pred, 'confidence': confidence, 'vt1': self.vt1, 'vt2': self.vt2, 'vt3':self.vt3, 'vt4':self.vt4}
        else:
            output = {'pred': pred, 'confidence': confidence}

        return output

if __name__ == "__main__":
    
    rgb = torch.FloatTensor(torch.randn((1,3, 300,65)))
    dep = torch.FloatTensor(torch.randn((1,1, 300,65)))

    sample = {'rgb':rgb,'dep':dep}

    model = UNETModel()

    out = model(sample)
    print(out['pred'].shape)
