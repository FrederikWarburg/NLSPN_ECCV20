import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import _remove_extra_pad


def conv1x1_1d(channel_in, channel_out, stride=1, groups = 1):
    return nn.Conv1d(channel_in, channel_out, kernel_size=1, stride=stride, groups=groups)

def conv1x1_2d(channel_in, channel_out, stride=1, groups=1, padding=0):
    return nn.Conv2d(channel_in, channel_out, kernel_size=(1,1), stride=stride, padding=padding, groups=groups)

def conv3x3_2d(channel_in, channel_out, stride=1, groups=1, padding=1):
    return nn.Conv2d(channel_in, channel_out, kernel_size=(3,3), stride=stride, groups=groups, padding=padding)


class VisualTransformer(nn.Module):
    def __init__(self, L, CT, C, size =14, num_downsample = 1, head = 16, groups = 16, kqv_groups = 16, dynamic = False):
        super(VisualTransformer,self).__init__()
        self.L = L # number of tokens
        self.CT = CT # size of tokens
        self.C = C # number of channels for features
        self.head = head
        self.groups = groups
        self.kqv_groups = kqv_groups
        self.size = size
        self.num_downsample = num_downsample

        self.tokenizer = Tokenizer(L, CT, C, size=size, num_downsample=num_downsample, head=head, groups=groups)
        self.transformer = Transformer(CT, head=head, kqv_groups=kqv_groups)
        self.projector = Projector(CT, C, head=head, groups=groups)

    def forward(self, src, dst):

        src = _remove_extra_pad(src, dst)
        
        #print(src.shape, dst.shape)
        assert src.shape == dst.shape
        self.size = src.shape[-2:]
        self.tokens_in = self.tokenizer(src)
        self.tokens_out = self.transformer(self.tokens_in)
        dst = self.projector(dst, self.tokens_out)

        return dst

class Tokenizer(nn.Module):
    def __init__(self,L,CT,C,size = 14, num_downsample = 1, head = 16,groups = 16,dynamic = False):
        super(Tokenizer,self).__init__()
        if not dynamic:
            #use static weights to compute token coefficients.
            self.conv_token_coef = conv1x1_2d(C,L)
        else:
            #use previous tokens to compute a query weight, which is
            #then used to compute token coefficients.
            self.conv_query = conv1x1_1d(CT,C)
            self.conv_key = conv1x1_2d(C,C,groups = groups)
        
        self.conv_value = conv1x1_2d(C,CT,groups = groups)

        num_downsample = num_downsample
        size = size
        self.CT = CT
        self.pos_encoding = PosEncoder(size, num_downsample)
        self.conv_token = conv1x1_1d(self.CT+self.pos_encoding.pos_dim, self.CT)
        self.head = head
        self.dynamic = dynamic

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature, tokens = None):
        #compute token coefficients
        # feature: N,C,H,W, tokens: N, CT, L
        if not self.dynamic:
            #print(feature.shape)
            token_coef = self.conv_token_coef(feature)
            #print(torch.min(token_coef), torch.max(token_coef))
            #print("1", token_coef.is_contiguous())
            #print("tio", token_coef.shape)
            #token_coef = token_coef.permute(0,1,3,2)
            #print("tio1", token_coef.shape)
            N, L, H, W = token_coef.shape
            token_coef = token_coef.view(N,1,L,H*W)
            token_coef = token_coef.permute(0,1,3,2) # N,1,HW,L
            #print(np.sqrt(feature.shape[1]), feature.shape[1])
            token_coef = token_coef/np.sqrt(feature.shape[1])
            #print(torch.min(token_coef), torch.max(token_coef))
            #print("2", token_coef.is_contiguous())


        else:
            L = tokens.shape[2]
            # Split input tokens
            # T_a,T_b: N, CT, L//2
            T_a,T_b = tokens[:,:,:L//2],tokens[:,:,L//2:]
            query = self.conv_query(T_a)
            N,C,L_a = query.shape
            #N,h,C//h,L_a
            query = query.view(N,token_coef.head,C//self.head,L_a)
            N,C,H,W = feature.shape
            key = self.conv_key(feature).view(N,self.head,C//self.head,H*W) # N,h,C//h,HW
            #Compute token coefficients.
            #N,h,HW,L_a
            token_coef = torch.matmul(key.permute(0,1,3,2),query)
            #token_coef = token_coef/np.sqrt(C/self.head)
        

      
        N, C, H, W = feature.shape

        token_coef = F.softmax(token_coef,dim = 2)
        value = self.conv_value(feature).view(N, self.head, self.CT//self.head , H*W) # N,h,C//h,HW

        # extract tokens from the feature map
        # static tokens: N,C,L
        # dynamic tokens: N,C,L_a
        tokens = torch.matmul(value,token_coef).view(N,self.CT,-1)

        # compute position encoding
        # if static: pos_encoding: N, Cp, L  else: N,Cp,L_a
        pos_encoding = self.pos_encoding(token_coef, (H,W))

        tokens = torch.cat((tokens,pos_encoding),dim = 1)

        if not self.dynamic:
            # N, C+Cp , L -> N, CT , L
            tokens = self.conv_token(tokens)
        else:
            # N, C+Cp , L_a -> N, CT , L_a , then cat to N, CT , (L_a + L_b )
            tokens = torch.cat(( T_b, self.conv_token(tokens)), dim = 2)

        
        """
        print("tokencoef", token_coef.shape)
        print("token", tokens.shape)
        print("pos_encoding", pos_encoding.shape)
        print("value", value.shape)
        """

        # store token_coef for visualizations
        self.token_coef = token_coef.clone()

        return tokens

class PosEncoder(nn.Module):
    def __init__(self, size, num_downsample):
        super(PosEncoder,self).__init__()

        self.size=size # H

        dim = 1
        ds_conv=[]
        for _ in range(num_downsample):
            ds_conv.append(conv3x3_2d(dim,dim,stride=2, padding=1))
        ds_conv.append(conv1x1_2d(dim,1))

        self.ds_conv=nn.Sequential(*ds_conv)

        self.pos_dim=size**2//(4**num_downsample) # Cp

        self.pos_conv=conv1x1_1d(self.pos_dim,self.pos_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,token_coef,input_size):

        H,W = input_size
        N, h, HW, L = token_coef.shape

        token_coef=token_coef.view(N*L,h,H,W)

        #interpolation to deal with input with varying sizes
        token_coef=F.interpolate(token_coef,size=(self.size,self.size))

        #downsampling
        token_coef=self.ds_conv(token_coef)
        token_coef=token_coef.view(N, L, -1).permute(0,2,1)

        #compress and compute the position encoding.
        return self.pos_conv(token_coef) # N, Cp, L


class Transformer(nn.Module):
    def __init__(self,CT,head=16,kqv_groups=8):
        super(Transformer,self).__init__()

        self.k_conv=conv1x1_1d(CT,CT//2,groups=kqv_groups)
        self.q_conv=conv1x1_1d(CT,CT//2,groups=kqv_groups)
        self.v_conv=conv1x1_1d(CT,CT,groups=kqv_groups)
        self.ff_conv=conv1x1_1d(CT,CT)
        self.head=head
        self.CT=CT

        # Implementation of Feedforward model
        dropout = 0.2
        self.linear1 = nn.Linear(CT, CT)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(CT, CT)

        self.norm1 = nn.LayerNorm(CT)
        self.norm2 = nn.LayerNorm(CT)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,tokens):

        N = tokens.shape[0]

        # k : N,h,CT//2//h,L
        k = self.k_conv(tokens).view(N,self.head,self.CT//2//self.head,-1)

        # q : N,h,CT//2//h,L
        q = self.q_conv(tokens).view(N,self.head,self.CT//2//self.head,-1)

        # v : N,h,CT//h,L
        v = self.v_conv(tokens).view(N,self.head,self.CT//self.head,-1)

        # kq : N,h,L,L
        kq = torch.matmul(k.permute(0,1,3,2),q)
        kq = F.softmax(kq/np.sqrt(kq.shape[2]),dim=2)

        #kqv : N,CT,L
        kqv = torch.matmul(v,kq).view(N,self.CT,-1)
        
        # How they had it
        #tokens = tokens+kqv
        #tokens = tokens+self.ff_conv(tokens)

        # How DETR has it:
        #src = src + self.dropout1(src2)
        #src2 = self.norm1(src)
        #src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        #src = src + self.dropout2(src2)
        #src = self.norm2(src)

        tokens = tokens + self.dropout1(kqv)
        kqv = self.norm1(tokens.permute(0,2,1))
        kqv = self.linear2(self.dropout(self.activation(self.linear1(kqv))))
        tokens = tokens + self.dropout2(kqv.permute(0,2,1))
        tokens = self.norm2(tokens.permute(0,2,1)).permute(0,2,1)

        # save for visualization purposes
        self.kq = kq
        
        """
        print("k", k.shape)
        print("q", q.shape)
        print("kq", kq.shape)
        print("kqv", kqv.shape)
        print("tokens", tokens.shape)
        """
        return tokens

class Projector(nn.Module):
    def __init__(self,CT,C,head=16,groups=16):
        super(Projector,self).__init__()

        self.proj_value_conv = conv1x1_1d(CT,C)
        self.proj_key_conv = conv1x1_1d(CT,C)
        self.proj_query_conv = conv1x1_2d(C,C,groups=groups)
        self.head = head

        # Implementation of Feedforward model
        dropout = 0.2
        self.linear1 = nn.Linear(C, C)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(C, C)

        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,feature,token):
        N,_,L = token.shape
        h = self.head
        
        proj_v = self.proj_value_conv(token).view(N,h,-1,L)
        proj_k = self.proj_key_conv(token).view(N,h,-1,L)
        proj_q = self.proj_query_conv(feature)
        
        N,C,H,W = proj_q.shape
        proj_q = proj_q.view(N,h,C//h,H*W).permute(0,1,3,2)
        
        #proj_coef : N,h,HW,L
        proj_coef=F.softmax(torch.matmul(proj_q,proj_k)/np.sqrt(C/h),dim=3)
        
        #proj : N,h,C//h,HW
        proj=torch.matmul(proj_v,proj_coef.permute(0,1,3,2))
        _,_,H,W=feature.shape

        # save for visualization purposes
        self.proj_coef = proj_coef
        """
        print("proj_v", proj_v.shape)
        print("proj_k", proj_k.shape)
        print("proj_q", proj_q.shape)
        print("proj_coef", proj_coef.shape)
        print("proj", proj.shape)
        """

        #print(proj.shape)        

        #feature = feature + self.dropout1(proj).view(N,-1,H,W)

        """
        proj = self.norm1(proj.view(N*h, C, H*W).permute(0,2,1))
        proj = self.linear2(self.dropout(self.activation(self.linear1(proj))))
        feature = feature + self.dropout2(proj.permute(0,2,1)).view(N,-1,H,W)
        feature = self.norm2(feature.permute(0,2,3,1)).permute(0,3,1,2)
        """

        return self.dropout1(proj).view(N,-1,H,W)