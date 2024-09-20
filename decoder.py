#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as f
from .encoder import MultiHeadSelfAttention, FeedForward

class Decoder:
    def __init__(self, 
                 vocab_size: int,
                 num_layers: int = 6,
                 max_output_tokens: int = 1000, 
                 num_heads: int = 4,
                 embed_dim: int = 128,
                 d_model: int = 512,
                 ffwd_dim: int = 2048):
        self.position_embedding = nn.Embedding(max_output_tokens, embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.Sequential(
            DecoderLayer(num_heads, embed_dim, ffwd_dim) for _ in range(num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T) -> (B,T,C)
        x = self.position_embedding(x) + self.token_embedding(x)
        # (B,T,C) -> (B,T,C)
        x = self.layers(x)
        return x

class DecoderLayer:
    def __init__(self,
                 num_heads: int,
                 embed_dim: int,
                 d_model: int,
                 ffwd_dim: int):
        self.masked_mha = MultiHeadSelfAttention(num_heads, embed_dim, d_model, mask=True)
        self.mha = MultiHeadSelfAttention(num_heads, embed_dim, d_model)
        self.ffwd = FeedForward(embed_dim, ffwd_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B,T,C -> B,T,H
        x = f.layer_norm(x + self.masked_mha(x))
        # B,T,H -> B,T,H

        # B,T,H -> B,T,H
        x = f.layer_norm(x + self.ffwd(x))
        return x
