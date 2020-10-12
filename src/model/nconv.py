########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
import numpy as np
from scipy.stats import poisson
from scipy import signal



# Normalized Convolution Layer
class NConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='n', stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=True):
        
        # Call _ConvNd constructor
        super(NConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, (0,0), groups, bias, padding_mode='zeros')
        
        self.eps = 1e-20
        self.pos_fn = pos_fn
        self.init_method = init_method
        
        # Initialize weights and bias
        self.init_parameters()
        
        if self.pos_fn is not None :
            EnforcePos.apply(self, 'weight', pos_fn)
        
    def forward(self, data, conf):
        # Normalized Convolution
        denom = F.conv2d(conf, self.weight, None, self.stride,
                        self.padding, self.dilation, self.groups)        
        nomin = F.conv2d(data*conf, self.weight, None, self.stride,
                        self.padding, self.dilation, self.groups)        
        nconv = nomin / (denom+self.eps)

        # Add bias
        b = self.bias
        sz = b.size(0)
        b = b.view(1,sz,1,1)
        b = b.expand_as(nconv)
        nconv += b
        
        # Propagate confidence
        cout = denom
        sz = cout.size()
        cout = cout.view(sz[0], sz[1], -1)
        
        k = self.weight
        k_sz = k.size()
        k = k.view(k_sz[0], -1)
        s = torch.sum(k, dim=-1, keepdim=True)        

        cout = cout / s
        cout = cout.view(sz)
        
        return nconv, cout
    
    def enforce_pos(self):
        p = self.weight
        if self.pos_fn.lower() == 'softmax':
            p_sz = p.size()
            p = p.view(p_sz[0],p_sz[1], -1)
            p = F.softmax(p, -1).data
            self.weight.data = p.view(p_sz)
        elif self.pos_fn.lower() == 'exp':
            self.weight.data = torch.exp(p).data
        elif self.pos_fn.lower() == 'softplus':
            self.weight.data = F.softplus(p, beta=10).data
        elif self.pos_fn.lower() == 'sigmoid':
            self.weight.data = F.sigmoid(p).data
        else:
            print('Undefined positive function!')
            return 
    
    def init_parameters(self):
        # Init weights
        if self.init_method == 'x': # Xavier            
            torch.nn.init.xavier_uniform_(self.weight)
        elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.weight)
        elif self.init_method == 'n': # Normal dist
            n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
            self.weight.data.normal_(2, math.sqrt(2. / n))
        elif self.init_method == 'p': # Poisson
            mu=self.kernel_size[0]/2 
            dist = poisson(mu)
            x = np.arange(0, self.kernel_size[0])
            y = np.expand_dims(dist.pmf(x),1)
            w = signal.convolve2d(y, y.transpose(), 'full')
            w = torch.Tensor(w).type_as(self.weight)
            w = torch.unsqueeze(w,0)
            w = torch.unsqueeze(w,1)
            w = w.repeat(self.out_channels, 1, 1, 1)
            w = w.repeat(1, self.in_channels, 1, 1)
            self.weight.data = w + torch.rand(w.shape)
            
        # Init bias
        self.bias = torch.nn.Parameter(torch.zeros(self.out_channels)+0.01)

class EnforcePos(object):
    def __init__(self, name, pos_fn):
            self.name = name
            self.pos_fn = pos_fn

    def compute_weight(self, module):
        return _pos(getattr(module, self.name + '_p'), self.pos_fn)

    @staticmethod
    def apply(module, name, pos_fn):
        fn = EnforcePos(name, pos_fn)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        #
        module.register_parameter(name + '_p', Parameter(_pos(weight, pos_fn).data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_p']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))

def _pos(p, pos_fn):
    pos_fn = pos_fn.lower()
    if pos_fn == 'softmax':
        p_sz = p.size()
        p = p.view(p_sz[0],p_sz[1], -1)
        p = F.softmax(p, -1)
        return p.view(p_sz)
    elif pos_fn == 'exp':
        return torch.exp(p)
    elif pos_fn == 'softplus':
        return F.softplus(p, beta=10)
    elif pos_fn == 'sigmoid':
        return F.sigmoid(p)
    else:
        print('Undefined positive function!')
        return 


def remove_weight_pos(module, name='weight'):
    r"""Removes the weight normalization reparameterization from a module.
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, EnforcePos) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))