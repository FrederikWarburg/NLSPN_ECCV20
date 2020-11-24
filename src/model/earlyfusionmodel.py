"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    NLSPN implementation
"""


from .common import *
import torch
import torch.nn as nn


class EARLYFUSIONModel(nn.Module):
    def __init__(self, args):
        super(EARLYFUSIONModel, self).__init__()

        self.args = args

        # Encoder
        self.conv1_rgb = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1,
                                      bn=False)
        if self.args.dep_src in ['slam', 'sgbm']:
            self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                        bn=False)
        else:
            self.conv1_dep0 = conv_bn_relu(1, 8, kernel=3, stride=1, padding=1,
                                bn=False)
            self.conv1_dep1 = conv_bn_relu(1, 8, kernel=3, stride=1, padding=1,
                                bn=False)


        if self.args.network == 'resnet18':
            net = get_resnet18(not self.args.from_scratch)
        elif self.args.network == 'resnet34':
            net = get_resnet34(not self.args.from_scratch)
        else:
            raise NotImplementedError

        # 1/1
        self.conv2 = net.layer1
        # 1/2
        self.conv3 = net.layer2
        # 1/4
        self.conv4 = net.layer3
        # 1/8
        self.conv5 = net.layer4

        del net

        # 1/16
        self.conv6 = conv_bn_relu(512, 512, kernel=3, stride=2, padding=1)

        # Shared Decoder
        # 1/8
        self.dec5 = convt_bn_relu(512, 256, kernel=3, stride=2,
                                  padding=1, output_padding=1)
        # 1/4
        self.dec4 = convt_bn_relu(256+512, 128, kernel=3, stride=2,
                                  padding=1, output_padding=1)
        # 1/2
        self.dec3 = convt_bn_relu(128+256, 64, kernel=3, stride=2,
                                  padding=1, output_padding=1)

        # 1/1
        self.dec2 = convt_bn_relu(64+128, 64, kernel=3, stride=2,
                                  padding=1, output_padding=1)

        # Init Depth Branch
        # 1/1
        self.id_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
                                    padding=1)
        self.id_dec0 = conv_bn_relu(64+64, 1, kernel=3, stride=1,
                                    padding=1, bn=False, relu=True)

        # Confidence Branch
        # Confidence is shared for propagation and mask generation
        # 1/1
        self.cf_dec1 = conv_bn_relu(64+64, 32, kernel=3, stride=1,
                                    padding=1)
        self.cf_dec0 = nn.Sequential(
            nn.Conv2d(32+64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )



        # Set parameter groups
        params = []
        for param in self.named_parameters():
            if param[1].requires_grad:
                params.append(param[1])

        params = nn.ParameterList(params)

        self.param_groups = [
            {'params': params, 'lr': self.args.lr}
        ]

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

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, sample):
        rgb = sample['rgb']
        
        # Encoding
        fe1_rgb = self.conv1_rgb(rgb)

        if self.args.dep_src in ['slam', 'sgbm']:
            dep = sample['dep']
            
            fe1_dep = self.conv1_dep(dep)
            fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
        else:
            dep0 = sample['dep0']
            dep1 = sample['dep1']

            fe1_dep0 = self.conv1_dep0(dep0)
            fe1_dep1 = self.conv1_dep1(dep1)
            fe1 = torch.cat((fe1_rgb, fe1_dep0, fe1_dep1), dim=1)
        
        fe2 = self.conv2(fe1)
        fe3 = self.conv3(fe2)
        fe4 = self.conv4(fe3)
        fe5 = self.conv5(fe4)
        fe6 = self.conv6(fe5)

        # Shared Decoding
        fd5 = self.dec5(fe6)
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))
        fd2 = self.dec2(self._concat(fd3, fe3))

        # Init Depth Decoding
        id_fd1 = self.id_dec1(self._concat(fd2, fe2))
        pred_init = self.id_dec0(self._concat(id_fd1, fe1))

        # Confidence Decoding
        cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
        confidence = self.cf_dec0(self._concat(cf_fd1, fe1))

        # Remove negative depth
        y = torch.clamp(pred_init, min=0)

        output = {'pred': y, 'confidence': confidence}

        return output
