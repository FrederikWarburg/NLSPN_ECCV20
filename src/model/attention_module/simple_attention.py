import torch
import torch.nn.functional as F
from torch import nn
import copy
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .position_encoding import PositionEmbeddingSine

class AttentionModuleSimple(nn.Module):
    def __init__(self, transformer, position_embedding, num_channels, hidden_dim):
        super().__init__()
        self.transformer = transformer
        self.position_embedding = position_embedding
        self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(hidden_dim, num_channels, kernel_size=1)

    def forward(self, x_rgb, x_dep):
        b, c, h, w = x_rgb.shape
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=x_rgb.device)
        pos = self.position_embedding(x_rgb, mask)
        proj_x = self.input_proj(x_rgb)
        hs = self.transformer(proj_x, x_dep, pos)

        # project back to features
        hs = hs.permute(1,2,0).view(b,c,h,w)
        new_x = x_dep.contiguous() + self.out_proj(hs)

        return new_x

def build_simple_attention_module(num_channels, hidden_dim=256):

    transformer = TransformerSimple(
        d_model=hidden_dim,
        dropout=0.1,
        nhead=8,
        dim_feedforward=2048
    )
    position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

    return AttentionModuleSimple(transformer, position_embedding, num_channels, hidden_dim)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



class TransformerSimple(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
 
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.value = 'dep'

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, rgb, dep, pos):

        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = rgb.shape
        rgb = rgb.flatten(2).permute(2, 0, 1)
        dep = dep.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)

        rgb2 = self.norm1(rgb)
        q = k = self.with_pos_embed(rgb2, pos)

        value = dep if self.value == 'dep' else rgb

        dep2, self.attn_map1 = self.multihead_attn(q, k, value=value)

        dep = dep + self.dropout1(dep2)

        dep2 = self.norm2(dep)
        dep2 = self.linear2(self.dropout(self.activation(self.linear1(dep2))))
        dep = dep + self.dropout2(dep2)
        return dep
