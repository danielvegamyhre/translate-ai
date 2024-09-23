#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as f
from math import sqrt

class Encoder(nn.Module):
    def __init__(self, 
                 num_layers: int,
                 vocab_size: int, 
                 embed_dim: int = 64,
                 d_model: int = 512,
                 max_seq_len: int = 512, 
                 num_attention_heads: int = 4,
                 ffwd_dim: int = 2048):
        super(Encoder, self).__init__()
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.Sequential(
            *[EncoderLayer(embed_dim, d_model, num_attention_heads, ffwd_dim) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T) -> (B,T,C)
        B, T = x.shape
        tok_embed = self.token_embedding(x)
        pos_embed = self.position_embedding(torch.arange(T).to(x.device))
        return self.layers(tok_embed + pos_embed) 


class EncoderLayer(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 d_model: int,
                 num_attention_heads: int,
                 ffwd_dim: int):
        super(EncoderLayer, self).__init__()
        self.ffwd = FeedForward(d_model, ffwd_dim)
        self.mha = MultiHeadSelfAttention(num_attention_heads, embed_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,C) -> (B,T,H)
        x = f.layer_norm(x + self.mha(x), x.shape)

        # (B,T,H) -> (B,T,H)
        x = f.layer_norm(x + self.ffwd(x), x.shape)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                 num_heads: int,
                 embed_dim: int,
                 d_model: int,
                 mask: bool = False):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_dim, head_dim, mask)
            for _ in range(num_heads)
        ])
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # head(x) => (B,T,head_size)
        # concat them all along head_size dim -> (B,T,H)
        # since H = head_size * num_heads 
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        # (B,T,H) -> B,T,H
        x = self.linear(x)
        return x

class SelfAttentionHead(nn.Module):
    '''Self attention head takes 2 parameters:
    embed_dim: embedding dimension from token embeddings.
    head_dim: total hidden dimension size divided by the number of attention heads.'''
    def __init__(self,
                 embed_dim: int = 128,
                 head_dim: int = 128,
                 mask: bool = False):
        super(SelfAttentionHead, self).__init__()
        self.query_layer = nn.Linear(embed_dim, head_dim)
        self.key_layer = nn.Linear(embed_dim, head_dim)
        self.value_layer = nn.Linear(embed_dim, head_dim)
    
    def forward(self, x: torch.Tensor, mask:bool = False) -> torch.Tensor:
        # batch, token, channel dimensions
        B, T, C = x.shape

        # (B,T,C) @ (C,H) = (B,T,H) -> (batch, token, hidden dimension)
        queries = self.query_layer(x) 

        # (B,T,C) @ (C,H) = (B,T,H) -> (batch, token, hidden dimension)
        keys = self.key_layer(x)

        # (B,T,H) @ (B,H,T) = (B,T,T)
        similarities = (queries @ keys.transpose(-2,-1)) / sqrt(keys.shape[-1])

        # optionally mask tokens in subsequent positions (so token at t[i] can only depend on t[:i]
        # for predictions)
        if mask:
            # B,T,T with bottom left triangle the same but top right triangle set to -inf.
            # we can use this to prevent the decoder from attending to future tokens it should
            # not have access to and preserve the auto-regressive property.
            mask = torch.tril(torch.ones(T,T))
            similarities = similarities.masked_fill(mask, -float('inf'))

        # element-wise, shape stays same
        weighted_similarities = torch.softmax(similarities, dim=-1)

        # (B,T,C) @ (C,H) = (B,T,H) -> (batch, token, hidden dimension)
        values = self.value_layer(x)

        # (B,T,T) @ (B,T,H) = B,T,H
        out = weighted_similarities @ values
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffwd_dim: int):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, ffwd_dim)
        self.linear2 = nn.Linear(ffwd_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,H) @ (H,M) = (B,T,M)
        x = f.relu(self.linear1(x))

        # (B,T,M) @ (M,H) = (B,T,H)
        x = f.relu(self.linear2(x))
        return x
    