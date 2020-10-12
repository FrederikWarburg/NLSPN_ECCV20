import torch
import torch.nn as nn

from .pnconv import UNetSP
from .pnconv import NConvUNet


class PNCONV_UNGUIDEDModel(nn.Module):
    def __init__(self, args):
        super().__init__() 
        self.__name__ = 'pncnn'
        self.args = args

        if self.args.input_conf == 'learned':
            self.conf_estimator = UNet(1, 1)
        
        self.nconv = NConvUNet(1, 1)
        self.var_estimator = UNetSP(1, 1)

    def forward(self, sample):

        x0 = sample['dep']  # Use only depth
        
        if self.args.input_conf == 'learned':
            c0 = self.conf_estimator(x0)
        else:
            c0 = sample['confidence']

        xout, cout = self.nconv(x0, c0)
        cout = self.var_estimator(cout)
        
        output = {'pred': xout, 'confidence': cout}

        return output
        