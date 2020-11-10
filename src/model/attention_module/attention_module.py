# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer, TransformerTemporal, TransformerReverse, TransformerSimple

class Projector(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.transformer = TransformerReverse(
            d_model=hidden_dim,
            dropout=0.1,
            nhead=8,
            dim_feedforward=hidden_dim,
            num_decoder_layers=1,
            normalize_before=False,
            return_intermediate_dec=False,
        )

    def forward(self, x, hs):
        new_x = self.transformer(hs, x)
        return new_x



class AttentionModuleSimple(nn.Module):
    def __init__(self, transformer, position_embedding, num_channels, hidden_dim):
        super().__init__()
        self.transformer = transformer
        self.position_embedding = position_embedding
        self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)
        self.projector = Projector(hidden_dim)
        self.out_proj = nn.Conv2d(hidden_dim, num_channels, kernel_size=1)

    def forward(self, x_rgb, x_dep):
        b, c, h, w = x_rgb.shape
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=x_rgb.device)
        pos = self.position_embedding(x_rgb, mask)
        proj_x = self.input_proj(x_rgb)
        hs, _ = self.transformer(proj_x, x_dep, pos)

        # project back to features
        new_x = self.projector(proj_x, hs)
        new_x = x_dep.contiguous() + self.out_proj(new_x)

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


class AttentionModule(nn.Module):
    def __init__(self, transformer, transformer_temporal, position_embedding, num_queries, num_channels, hidden_dim):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.transformer_temporal = transformer_temporal
        self.position_embedding = position_embedding
        #self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)
        self.projector = Projector(hidden_dim)
        self.out_proj = nn.Conv2d(hidden_dim, num_channels, kernel_size=1)

    def forward(self, x_rgb, x_dep):
        b, c, h, w = x_rgb.shape
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=x_rgb.device)
        pos = self.position_embedding(x_rgb, mask)
        proj_x = self.input_proj(x_rgb)
        hs, _ = self.transformer(proj_x, x_dep, mask, pos)

        # project back to features
        new_x = self.projector(proj_x, hs)
        new_x = x_dep.contiguous() + self.out_proj(new_x)

        return new_x

def build_attention_module(num_channels, hidden_dim=256, num_queries=100, temporal=False):
    transformer_temporal = None
    if temporal:
        transformer_temporal = TransformerTemporal(
            d_model=hidden_dim,
            dropout=0.1,
            nhead=8,
            dim_feedforward=2048,
            num_decoder_layers=1,
            normalize_before=False,
            return_intermediate_dec=False,
        )
    transformer = Transformer(
        d_model=hidden_dim,
        dropout=0.1,
        nhead=8,
        dim_feedforward=2048,
        num_decoder_layers=1,
        normalize_before=False,
        return_intermediate_dec=False,
    )
    position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

    return AttentionModule(transformer, transformer_temporal, position_embedding, num_queries, num_channels, hidden_dim)
