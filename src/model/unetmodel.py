

import torch
from .common import *
from .visualtransformer import *
from torchvision.models.resnet import BasicBlock
from torchvision import models


class UNETModel(nn.Module):
    def __init__(self, args = None):
        super(UNETModel, self).__init__()

        self.args = args

        self.network = self.args.network
        self.aggregate = self.args.aggregate
        self.guide = self.args.guide

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
        self.conv1_rgb = torch.nn.Sequential(*[net.conv1, net.bn1, net.relu])#1/2
        self.conv2_rgb = net.layer1 #1/2
        self.conv3_rgb = net.layer2 #1/4
        self.conv4_rgb = net.layer3 #1/8
        self.conv5_rgb = net.layer4 #1/16
        self.conv6_rgb = conv_bn_relu(512, 512, kernel=3, stride=2, padding=1, bn=False) # 1/32

        # Decoder
        self.dec5_rgb = convt_bn_relu(512, 512, kernel=3, stride=2, padding=1, output_padding=1) # 1/16
        self.dec4_rgb = convt_bn_relu(512+self.D_skip * 512, 256, kernel=3, stride=2, padding=1, output_padding=1) # 1/8
        self.dec3_rgb = convt_bn_relu(256+self.D_skip * 256, 128, kernel=3, stride=2, padding=1, output_padding=1) # 1/4
        self.dec2_rgb = convt_bn_relu(128+self.D_skip * 128, 64, kernel=3, stride=2, padding=1, output_padding=1) # 1/2
        self.dec1_rgb = conv_bn_relu(64+self.D_skip * 64, 64, kernel=3, stride=1, padding=1) # 1/2

        ####
        # Depth Stream
        ####

        # Encoder
        self.conv1_dep = conv_bn_relu(1, 64, kernel=7, stride=2, padding=3, bn=False) # 1/2
        if self.aggregate == 'sum':
            self.conv2_dep = net.layer1 # 1/2
            self.conv3_dep = net.layer2 # 1/4
            self.conv4_dep = net.layer3 # 1/8
            self.conv5_dep = net.layer4 # 1/16
        elif self.aggregate == 'cat':
            self.conv2_dep = self._make_layer(64 + self.D_guide * 64, 64, stride=1, blocks=2) # 1/2
            self.conv3_dep = self._make_layer(64 + self.D_guide * 64, 128, stride=2, blocks=2) # 1/4
            self.conv4_dep = self._make_layer(128 + self.D_guide * 128, 256, stride=2, blocks=2) # 1/8
            self.conv5_dep = self._make_layer(256 + self.D_guide * 256, 512, stride=2, blocks=2) # 1/16
        self.conv6_dep = conv_bn_relu(512, 512, kernel=3, stride=2, padding=1, bn=False) # 1/32

        # Decoder
        self.dec5_dep = convt_bn_relu(512, 512, kernel=3, stride=2, padding=1, output_padding=1) # 1/16
        self.dec4_dep = convt_bn_relu(512+self.D_skip * 512, 256, kernel=3, stride=2, padding=1, output_padding=1) # 1/8
        self.dec3_dep = convt_bn_relu(256+self.D_skip * 256, 128, kernel=3, stride=2, padding=1, output_padding=1) # 1/4
        self.dec2_dep = convt_bn_relu(128+self.D_skip * 128, 64, kernel=3, stride=2, padding=1, output_padding=1) # 1/2
        self.dec1_dep = convt_bn_relu(64+self.D_skip * 64, 64, kernel=3, stride=2, padding=1, output_padding=1) # 1/1

        # Depth Branch
        self.id_dec1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1) # 1/1
        self.id_dec0 = conv_bn_relu(64, 1, kernel=3, stride=1, padding=1, bn=False, relu=True)

        # Confidence Branch
        self.cf_dec1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1) # 1/1
        self.cf_dec0 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Softplus()
        )

        ####
        # VISUAL TRANSFORMER
        ####

        self.vt1 = VisualTransformer(L=8, CT=1024, C=64, head=args.heads, groups=args.groups, kqv_groups=args.kqv_groups, dynamic=False)
        self.vt2 = VisualTransformer(L=8, CT=1024, C=128, head=args.heads, groups=args.groups, kqv_groups=args.kqv_groups, dynamic=False)
        self.vt3 = VisualTransformer(L=8, CT=1024, C=256, head=args.heads, groups=args.groups, kqv_groups=args.kqv_groups, dynamic=False)
        self.vt4 = VisualTransformer(L=8, CT=1024, C=512, head=args.heads, groups=args.groups, kqv_groups=args.kqv_groups, dynamic=False)


    def _make_layer(self, inplanes, planes, blocks=1, stride=1):
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

    def _guide(self, fd_rgb, fe_rgb, fe_dep, guide = 'cat', dim = 1):

        if guide == 'cat':
            return self._concat(self._concat(fd_rgb, fe_rgb, aggregate='cat', dim=1), fe_dep, aggregate='cat', dim=1)
        elif guide == 'sum':
            return self._concat(self._concat(fd_rgb, fe_rgb, aggregate='sum', dim=1), fe_dep, aggregate='sum', dim=1)
        elif guide == 'none':
            return fe_dep

    def _concat(self, fd, fe, aggregate='cat', dim=1):
        
        fd = self._remove_extra_pad(fd, fe)

        if aggregate == 'cat':
            f = torch.cat((fd, fe), dim=dim)
        elif aggregate == 'sum':
            f = fd + fe

        return f

    def _remove_extra_pad(self, fd, fe, dim=1):

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

    def forward(self, sample):

        rgb = sample['rgb']
        dep = sample['dep']
        
        # Encoding RGB
        fe1_rgb = self.conv1_rgb(rgb)
        fe2_rgb = self.conv2_rgb(fe1_rgb)
        fe3_rgb = self.conv3_rgb(fe2_rgb)
        fe4_rgb = self.conv4_rgb(fe3_rgb)
        fe5_rgb = self.conv5_rgb(fe4_rgb)
        fe6_rgb = self.conv6_rgb(fe5_rgb)

        # Decoding RGB
        fd5_rgb = self.dec5_rgb(fe6_rgb)
        fd4_rgb = self.dec4_rgb(self._concat(fd5_rgb, fe5_rgb, aggregate=self.aggregate, dim=1))
        fd3_rgb = self.dec3_rgb(self._concat(fd4_rgb, fe4_rgb, aggregate=self.aggregate, dim=1))
        fd2_rgb = self.dec2_rgb(self._concat(fd3_rgb, fe3_rgb, aggregate=self.aggregate, dim=1))
        fd1_rgb = self.dec1_rgb(self._concat(fd2_rgb, fe2_rgb, aggregate=self.aggregate, dim=1))
        
        ###
        # DEPTH UNET
        ###
        
        # Encoding Depth
        fe1_dep = self.conv1_dep(dep)
        fe2_dep = self.conv2_dep(self._guide(fd1_rgb, fe1_rgb, fe1_dep, guide=self.guide, dim=1))

        # we need first to remove some extra padding which is added in the decoding stage
        fd2_rgb = self._remove_extra_pad(fd2_rgb, fe2_dep)
        fe2_dep = self.vt1(fd2_rgb, fe2_dep)

        fe3_dep = self.conv3_dep(self._guide(fd2_rgb, fe2_rgb, fe2_dep, guide=self.guide, dim=1))

        # we need first to remove some extra padding which is added in the decoding stage
        fd3_rgb = self._remove_extra_pad(fd3_rgb, fe3_dep)
        fe3_dep = self.vt2(fd3_rgb, fe3_dep)

        fe4_dep = self.conv4_dep(self._guide(fd3_rgb, fe3_rgb, fe3_dep, guide=self.guide, dim=1))

        # we need first to remove some extra padding which is added in the decoding stage
        fd4_rgb = self._remove_extra_pad(fd4_rgb, fe4_dep)
        fe4_dep = self.vt3(fd5_rgb, fe4_dep)

        fe5_dep = self.conv5_dep(self._guide(fd4_rgb, fe4_rgb, fe4_dep, guide=self.guide, dim=1))

        # we need first to remove some extra padding which is added in the decoding stage
        fd5_rgb = self._remove_extra_pad(fd5_rgb, fe5_dep)
        fe5_dep = self.vt4(fd5_rgb, fe5_dep)

        fe6_dep = self.conv6_dep(fe5_dep)
        
        # Decoding Depth
        fd5_dep = self.dec5_dep(fe6_dep)
        fd4_dep = self.dec4_dep(self._concat(fd5_dep, fe5_dep, aggregate=self.aggregate, dim=1))
        fd3_dep = self.dec3_dep(self._concat(fd4_dep, fe4_dep, aggregate=self.aggregate, dim=1))
        fd2_dep = self.dec2_dep(self._concat(fd3_dep, fe3_dep, aggregate=self.aggregate, dim=1))   
        fd1_dep = self.dec1_dep(self._concat(fd2_dep, fe2_dep, aggregate=self.aggregate, dim=1))

        ###
        # PREDICTION HEADS
        ###

        # Depth Decoding
        id_fd1 = self.id_dec1(fd1_dep)
        pred = self.id_dec0(id_fd1)

        # Confidence Decoding
        cf_fd1 = self.cf_dec1(fd1_dep)
        confidence = self.cf_dec0(cf_fd1)

        pred = self._remove_extra_pad(pred, dep)
        confidence = self._remove_extra_pad(confidence, dep)

        output = {'pred': pred, 'confidence': confidence, 'vt1': self.vt1, 'vt2': self.vt2, 'vt3':self.vt3, 'vt4':self.vt4)}

        return output

if __name__ == "__main__":
    
    rgb = torch.FloatTensor(torch.randn((1,3, 300,65)))
    dep = torch.FloatTensor(torch.randn((1,1, 300,65)))

    sample = {'rgb':rgb,'dep':dep}

    model = UNETModel()

    out = model(sample)
    print(out['pred'].shape)
