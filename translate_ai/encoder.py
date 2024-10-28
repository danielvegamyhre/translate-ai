#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as f

from attention import DifferentialMultiHeadSelfAttention

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
        self.mha = DifferentialMultiHeadSelfAttention(num_attention_heads, embed_dim, d_model)
        self.ffwd = FeedForward(embed_dim, ffwd_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)


    def forward(self, x: torch.Tensor, encoder_padding_mask: torch.Tensor) -> torch.Tensor:
        # (B,T,C) -> (B,T,H)
        x = self.ln1(x + self.mha(x, encoder_padding_mask))

        # (B,T,H) -> (B,T,H)
        x = self.ln2(x + self.ffwd(x))
        return x

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
