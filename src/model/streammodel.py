

import torch
from .common import *
from .visualtransformer import *
from torchvision.models.resnet import BasicBlock
from torchvision import models


class StreamModel(nn.Module):
    def __init__(self, args = None):
        super(StreamModel, self).__init__()

        self.args = args

        self.network = self.args.network
        
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
        self.conv1 = self._make_layer(3,16,2,1)
        self.conv2 = self._make_layer(3,16,2,1)
        self.conv3 = self._make_layer(3,16,2,1)
        self.conv4 = self._make_layer(3,16,2,1)
        
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

        C = 16
        num_downsample = 3
        size = 128

        self.tokenizer1 = Tokenizer(args.num_tokens, args.token_size, C, head=args.num_heads, groups=args.groups, num_downsample=num_downsample, size=size)
        self.transformer1 = Transformer(args.token_size, head=args.num_heads, kqv_groups=args.kqv_groups)
        self.projector1 = Projector(args.token_size, C, head=args.num_heads, groups=args.groups)

        self.tokenizer2 = Tokenizer(args.num_tokens, args.token_size, C, head=args.num_heads, groups=args.groups, num_downsample=num_downsample, size=size)
        self.transformer2 = Transformer(args.token_size, head=args.num_heads, kqv_groups=args.kqv_groups)
        self.projector2 = Projector(args.token_size, C, head=args.num_heads, groups=args.groups)

        self.tokenizer3 = Tokenizer(args.num_tokens, args.token_size, C, head=args.num_heads, groups=args.groups, num_downsample=num_downsample, size=size)
        self.transformer3 = Transformer(args.token_size, head=args.num_heads, kqv_groups=args.kqv_groups)
        self.projector3 = Projector(args.token_size, C, head=args.num_heads, groups=args.groups)

        self.tokenizer4 = Tokenizer(args.num_tokens, args.token_size, C, head=args.num_heads, groups=args.groups, num_downsample=num_downsample, size=size)
        self.transformer4 = Transformer(args.token_size, head=args.num_heads, kqv_groups=args.kqv_groups)
        self.projector4 = Projector(args.token_size, C, head=args.num_heads, groups=args.groups)

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

    def forward(self, sample):

        rgb = sample['rgb']
        dep = sample['dep']

        # Encoding Depth
        fe1 = self.conv1(rgb)

        tokens_in = self.tokenizer1(fe1)
        tokens_out = self.transformer1(tokens_in)
        fe1 = self.projector1(fe1, tokens_out)

        fe2 = self.conv2(fe1)

        tokens_in = self.tokenizer2(fe2)
        tokens_out = self.transformer2(tokens_in)
        fe2 = self.projector2(fe2, tokens_out)

        fe3 = self.conv3(fe2)

        tokens_in = self.tokenizer3(fe3)
        tokens_out = self.transformer3(tokens_in)
        fe3 = self.projector3(fe3, tokens_out)

        fe4 = self.conv4(fe3)

        tokens_in = self.tokenizer4(fe4)
        tokens_out = self.transformer4(tokens_in)
        fe4 = self.projector4(fe4, tokens_out)

        ###
        # PREDICTION HEADS
        ###

        # Depth Decoding
        fe5 = self.id_dec1_rgb(fe4)
        pred_rgb = self.id_dec0_rgb(fe5)

        # Confidence Decoding
        fe5 = self.cf_dec1_rgb(fe4)
        confidence_rgb = self.cf_dec0_rgb(fe5)


        if self.args.attention_stage == 'none':
            output = {'pred': pred_rgb, 'confidence': confidence_rgb}
        else:
            output = {'pred': pred_rgb, 'confidence': confidence_rgb, 'token_coef': self.tokenizer2.token_coef, 'kq': self.transformer2.kq, 'proj_coef': self.projector2.proj_coef, 'size': size}

        return output

if __name__ == "__main__":
    
    rgb = torch.FloatTensor(torch.randn((1,3, 300,65)))
    dep = torch.FloatTensor(torch.randn((1,1, 300,65)))

    sample = {'rgb':rgb,'dep':dep}

    model = UNETModel()

    out = model(sample)
    #print(out['pred'].shape)
