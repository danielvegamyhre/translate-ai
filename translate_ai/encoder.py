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
                 ffwd_dim: int = 2048,
                 dropout: float = 0.1):
        super(Encoder, self).__init__()
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, d_model, num_attention_heads, ffwd_dim)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_padding_mask: torch.Tensor) -> torch.Tensor:
        # (B,T) -> (B,T,C)
        B, T = x.shape
        tok_embed = self.token_embedding(x)
        pos_embed = self.position_embedding(torch.arange(T).to(x.device))
        out = tok_embed + pos_embed

        # B,T -> B,1,T so it can be broadcasted across attention scores
        encoder_padding_mask = encoder_padding_mask.unsqueeze(1)
        for layer in self.layers:
            # B,T,C -> B,T,C
            out = layer(out, encoder_padding_mask)
        out = self.dropout(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 d_model: int,
                 num_attention_heads: int,
                 ffwd_dim: int):
        
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadSelfAttention(num_attention_heads, embed_dim, d_model)
        self.ffwd = FeedForward(embed_dim, ffwd_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)


    def forward(self, x: torch.Tensor, encoder_padding_mask: torch.Tensor) -> torch.Tensor:
        # (B,T,C) -> (B,T,H)
        x = self.ln1(x + self.mha(x, encoder_padding_mask))

        # (B,T,H) -> (B,T,H)
        x = self.ln2(x + self.ffwd(x))
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                 num_heads: int,
                 embed_dim: int,
                 d_model: int,
                 dropout: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_dim, head_dim)
            for _ in range(num_heads)
        ])
        self.linear = nn.Linear(d_model, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # head(x) => (B,T,head_size)
        # concat them all along head_size dim -> (B,T,H)
        # since H = head_size * num_heads 
        x = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        # project to embed dimension to enable residual (i.e. x + mha(x))
        x = self.linear(x)  #  (B,T,H) -> B,T,C
        x = self.dropout(x)
        return x

class SelfAttentionHead(nn.Module):
    '''Self attention head takes 2 parameters:
    embed_dim: embedding dimension from token embeddings.
    head_dim: total hidden dimension size divided by the number of attention heads.'''
    def __init__(self,
                 embed_dim: int = 128,
                 head_dim: int = 128,
                 dropout: float = 0.1):
        super(SelfAttentionHead, self).__init__()
        self.query_layer = nn.Linear(embed_dim, head_dim)
        self.key_layer = nn.Linear(embed_dim, head_dim)
        self.value_layer = nn.Linear(embed_dim, head_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # (B,T,C) @ (C,head_dim) = (B,T,head_dim)
        queries = self.query_layer(x) 

        # (B,T,C) @ (C,head_dim) = (B,T,head_dim)
        keys = self.key_layer(x)

        # (B,T,H) @ (B,H,T) = (B,T,T)
        scores = (queries @ keys.transpose(-2,-1)) / sqrt(keys.shape[-1])

        # optionally mask tokens (e.g., mask padding tokens to prevent model from attending
        # to them, or mask tokens temporally ahead of each token to preserve autoregressive property).
        if mask is not None:
            scores = scores.masked_fill(mask, -float('inf'))

        # element-wise, shape stays same
        weighted_scores = torch.softmax(scores, dim=-1)

        # apply dropout
        weighted_scores = self.dropout(weighted_scores)

        # (B,T,C) @ (C,head_dim) = (B,T,head_dim)
        values = self.value_layer(x)

        # (B,T,T) @ (B,T,head_dim) = B,T,head_dim
        out = weighted_scores @ values
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffwd_dim: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ffwd_dim)
        self.linear2 = nn.Linear(ffwd_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,H) @ (H,M) = (B,T,M)
        x = f.relu(self.linear1(x))

        # (B,T,M) @ (M,H) = (B,T,H)
        x = f.relu(self.linear2(x))

        x = self.dropout(x)
        return x
    