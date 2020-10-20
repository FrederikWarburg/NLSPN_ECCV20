

import torch
from .common import *
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
        ####
        # RGB Stream
        ####

        if self.network == 'resnet18':
            net = get_resnet18(not self.args.from_scratch)
        elif self.network == 'resnet34':
            net = get_resnet34(not self.args.from_scratch)
        else:
            raise NotImplementedError

        # Encoder
        self.conv1_rgb = conv_bn_relu(3, 64, kernel=3, stride=1, padding=1, bn=False)
        self.conv2_rgb = net.layer1 #self._make_layer(64, 64, stride=1) # 1/2
        self.conv3_rgb = net.layer2 #self._make_layer(64, 128, stride=2) # 1/4
        self.conv4_rgb = net.layer3 #self._make_layer(128, 256, stride=2) # 1/8
        self.conv5_rgb = net.layer4 #self._make_layer(256, 512, stride=2) # 1/16

        # Bottleneck
        self.conv6_rgb = conv_bn_relu(512, 512, kernel=3, stride=2, padding=1) # 1/16

        # Decoder
        self.dec5_rgb = convt_bn_relu(512, 512, kernel=3, stride=2, padding=1, output_padding=1) # 1/8
        self.dec4_rgb = convt_bn_relu(512+self.D * 512, 256, kernel=3, stride=2, padding=1, output_padding=1) # 1/4
        self.dec3_rgb = convt_bn_relu(256+self.D * 256, 128, kernel=3, stride=2, padding=1, output_padding=1) # 1/2
        self.dec2_rgb = convt_bn_relu(128+self.D * 128, 64, kernel=3, stride=2, padding=1, output_padding=1) # 1/1

        ####
        # Depth Stream
        ####

        # Encoder
        self.conv1_dep = conv_bn_relu(1, 64, kernel=3, stride=1, padding=1, bn=False)
        if self.aggregate == 'cat':
            self.conv2_dep = self._make_layer(64, 64, stride=1) # 1/2
            self.conv3_dep = self._make_layer(64 + self.D*(self.D+1) * 64, 128, stride=2) # 1/4
            self.conv4_dep = self._make_layer(128+ self.D*(self.D+1) * 128, 256, stride=2) # 1/8
            self.conv5_dep = self._make_layer(256+ self.D*(self.D+1) * 256, 512, stride=2) # 1/16
        elif self.aggregate == 'sum':
            self.conv2_dep = net.layer1 # 1/2
            self.conv3_dep = net.layer2 # 1/4
            self.conv4_dep = net.layer3 # 1/8
            self.conv5_dep = net.layer4 # 1/16
        self.conv6_dep = conv_bn_relu(512 + self.D*(self.D+1) * 512, 512, kernel=3, stride=2, padding=1) # 1/16
        del net

        # Decoder
        self.dec5_dep = convt_bn_relu(512, 512, kernel=3, stride=2, padding=1, output_padding=1) # 1/8
        self.dec4_dep = convt_bn_relu(512+self.D * 512, 256, kernel=3, stride=2, padding=1, output_padding=1) # 1/4
        self.dec3_dep = convt_bn_relu(256+self.D * 256, 128, kernel=3, stride=2, padding=1, output_padding=1) # 1/2
        self.dec2_dep = convt_bn_relu(128+self.D * 128, 64, kernel=3, stride=2, padding=1, output_padding=1) # 1/1

        # Depth Branch
        self.id_dec1 = conv_bn_relu(64+self.D*64, 64, kernel=3, stride=1, padding=1) # 1/1
        self.id_dec0 = conv_bn_relu(64+self.D*64, 1, kernel=3, stride=1, padding=1, bn=False, relu=True)

        # Confidence Branch
        self.cf_dec1 = conv_bn_relu(64+self.D*64, 64, kernel=3, stride=1, padding=1) # 1/1
        self.cf_dec0 = nn.Sequential(
            nn.Conv2d(64+self.D*64, 1, kernel_size=3, stride=1, padding=1),
            nn.Softplus()
        )

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
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        if self.aggregate == 'cat':
            f = torch.cat((fd, fe), dim=dim)
        elif self.aggregate == 'sum':
            f = fd + fe

        return f

    def forward(self, sample):

        rgb = sample['rgb']
        dep = sample['dep']
        
        # Encoding RGB
        fe1_rgb = self.conv1_rgb(rgb)
        fe2_rgb = self.conv2_rgb(fe1_rgb)
        fe3_rgb = self.conv3_rgb(fe2_rgb)
        fe4_rgb = self.conv4_rgb(fe3_rgb)
        fe5_rgb = self.conv5_rgb(fe4_rgb)

        # Bottleneck RGB
        fe6_rgb = self.conv6_rgb(fe5_rgb)

        # Decoding RGB
        fd5_rgb = self.dec5_rgb(fe6_rgb)
        fd4_rgb = self.dec4_rgb(self._concat(fd5_rgb, fe5_rgb, dim=1))
        fd3_rgb = self.dec3_rgb(self._concat(fd4_rgb, fe4_rgb, dim=1))
        fd2_rgb = self.dec2_rgb(self._concat(fd3_rgb, fe3_rgb, dim=1))
         
        # Encoding Depth
        fe1_dep = self.conv1_dep(dep)
        fe2_dep = self.conv2_dep(fe1_dep)
        fe3_dep = self.conv3_dep(self._concat(fe2_dep, self._concat(fd2_rgb, fe2_rgb, dim=1), dim=1))
        fe4_dep = self.conv4_dep(self._concat(fe3_dep, self._concat(fd3_rgb, fe3_rgb, dim=1), dim=1))
        fe5_dep = self.conv5_dep(self._concat(fe4_dep, self._concat(fd4_rgb, fe4_rgb, dim=1), dim=1))

        # Bottleneck Depth
        fe6_dep = self.conv6_dep(self._concat(fe5_dep,self._concat(fd5_rgb, fe5_rgb, dim=1), dim=1))

        # Decoding Depth
        fd5_dep = self.dec5_dep(fe6_dep)
        fd4_dep = self.dec4_dep(self._concat(fd5_dep, fe5_dep, dim=1))
        fd3_dep = self.dec3_dep(self._concat(fd4_dep, fe4_dep, dim=1))
        fd2_dep = self.dec2_dep(self._concat(fd3_dep, fe3_dep, dim=1))      

        # Depth Decoding
        id_fd1 = self.id_dec1(self._concat(fd2_dep, fe2_dep, dim=1))
        pred = self.id_dec0(self._concat(id_fd1, fe1_dep, dim=1))

        # Confidence Decoding
        cf_fd1 = self.cf_dec1(self._concat(fd2_dep, fe2_dep, dim=1))
        confidence = self.cf_dec0(self._concat(cf_fd1, fe1_dep, dim=1))

        output = {'pred': pred, 'confidence': confidence}

        return output

if __name__ == "__main__":
    
    rgb = torch.FloatTensor(torch.randn((1,3, 300,65)))
    dep = torch.FloatTensor(torch.randn((1,1, 300,65)))

    sample = {'rgb':rgb,'dep':dep}

    model = UNETModel()

    out = model(sample)
