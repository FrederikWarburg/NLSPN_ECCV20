

import torch
from .common import *
from visualtransformer import *
from torchvision.models.resnet import BasicBlock
from torchvision import models


class UNETModel(nn.Module):
    def __init__(self, args = None):
        super(UNETModel, self).__init__()

        self.args = args

        self.network = self.args.network
        self.aggregate = self.args.aggregate

        if self.aggregate == 'cat':
            self.D = 1
        elif self.aggregate == 'sum':
            self.D = 0
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
        self.dec4_rgb = convt_bn_relu(512+self.D * 512, 256, kernel=3, stride=2, padding=1, output_padding=1) # 1/8
        self.dec3_rgb = convt_bn_relu(256+self.D * 256, 128, kernel=3, stride=2, padding=1, output_padding=1) # 1/4
        self.dec2_rgb = convt_bn_relu(128+self.D * 128, 64, kernel=3, stride=2, padding=1, output_padding=1) # 1/2
        self.dec1_rgb = conv_bn_relu(64+self.D * 64, 64, kernel=3, stride=1, padding=1) # 1/2

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
            self.conv2_dep = self._make_layer(64 + 2*self.D * 64, 64, stride=1, blocks=2) # 1/2
            self.conv3_dep = self._make_layer(64 + 2*self.D * 64, 128, stride=2, blocks=2) # 1/4
            self.conv4_dep = self._make_layer(128 + 2*self.D * 128, 256, stride=2, blocks=2) # 1/8
            self.conv5_dep = self._make_layer(256 + 2*self.D * 256, 512, stride=2, blocks=2) # 1/16
        self.conv6_dep = conv_bn_relu(512, 512, kernel=3, stride=2, padding=1, bn=False) # 1/32

        # Decoder
        self.dec5_dep = convt_bn_relu(512, 512, kernel=3, stride=2, padding=1, output_padding=1) # 1/16
        self.dec4_dep = convt_bn_relu(512+self.D * 512, 256, kernel=3, stride=2, padding=1, output_padding=1) # 1/8
        self.dec3_dep = convt_bn_relu(256+self.D * 256, 128, kernel=3, stride=2, padding=1, output_padding=1) # 1/4
        self.dec2_dep = convt_bn_relu(128+self.D * 128, 64, kernel=3, stride=2, padding=1, output_padding=1) # 1/2
        self.dec1_dep = convt_bn_relu(64+self.D * 64, 64, kernel=3, stride=2, padding=1, output_padding=1) # 1/1

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

        L = 8 # number of tokens
        CT = 1024 # size of tokens
        C = 512 # number of channels for features
        head = 16
        groups = 16
        kqv_groups = 8

        self.tokenizer = Tokenizer(L, CT, C, head=head, groups=groups)
        self.transformer = Transformer(CT, head=head, kqv_groups=kqv_groups)
        self.projector = Projector(CT, C, head=head, groups=groups)

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

    def _concat(self, fd, fe, dim=1):
        
        fd = self._remove_extra_pad(fd, fe)

        if self.aggregate == 'cat':
            f = torch.cat((fd, fe), dim=dim)
        elif self.aggregate == 'sum':
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
        fd4_rgb = self.dec4_rgb(self._concat(fd5_rgb, fe5_rgb, dim=1))
        fd3_rgb = self.dec3_rgb(self._concat(fd4_rgb, fe4_rgb, dim=1))
        fd2_rgb = self.dec2_rgb(self._concat(fd3_rgb, fe3_rgb, dim=1))
        fd1_rgb = self.dec1_rgb(self._concat(fd2_rgb, fe2_rgb, dim=1))
        
        ###
        # DEPTH UNET
        ###

        # Encoding Depth
        fe1_dep = self.conv1_dep(dep)
        fe2_dep = self.conv2_dep(self._concat(self._concat(fd1_rgb, fe1_rgb, dim=1), fe1_dep, dim=1))
        fe3_dep = self.conv3_dep(self._concat(self._concat(fd2_rgb, fe2_rgb, dim=1), fe2_dep, dim=1))
        fe4_dep = self.conv4_dep(self._concat(self._concat(fd3_rgb, fe3_rgb, dim=1), fe3_dep, dim=1))
        fe5_dep = self.conv5_dep(self._concat(self._concat(fd4_rgb, fe4_rgb, dim=1), fe4_dep,  dim=1))

        # VT
        tokens_in = self.tokenizer(fd5_rgb)
        tokens_out = self.transformer(tokens_in)
        fe5_dep = self.projector(fe5_dep, tokens_out)

        fe6_dep = self.conv6_dep(fe5_dep)
        
        # Decoding Depth
        fd5_dep = self.dec5_dep(fe6_dep)
        fd4_dep = self.dec4_dep(self._concat(fd5_dep, fe5_dep, dim=1))
        fd3_dep = self.dec3_dep(self._concat(fd4_dep, fe4_dep, dim=1))
        fd2_dep = self.dec2_dep(self._concat(fd3_dep, fe3_dep, dim=1))   
        fd1_dep = self.dec1_dep(self._concat(fd2_dep, fe2_dep, dim=1))

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

        output = {'pred': pred, 'confidence': confidence, 'token_coef': self.tokenizer.token_coef}

        return output

if __name__ == "__main__":
    
    rgb = torch.FloatTensor(torch.randn((1,3, 300,65)))
    dep = torch.FloatTensor(torch.randn((1,1, 300,65)))

    sample = {'rgb':rgb,'dep':dep}

    model = UNETModel()

    out = model(sample)
    print(out['pred'].shape)