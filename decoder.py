#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as f
from math import sqrt

from encoder import MultiHeadSelfAttention, FeedForward

class Decoder(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 num_layers: int = 6,
                 max_output_tokens: int = 1000, 
                 num_heads: int = 4,
                 embed_dim: int = 128,
                 d_model: int = 512,    
                 ffwd_dim: int = 2048,
                 max_seq_len: int = 128):
        super(Decoder, self).__init__()
        self.position_embedding = nn.Embedding(max_output_tokens, embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(num_heads, embed_dim, d_model, ffwd_dim, max_seq_len) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, decoder_padding_mask: torch.Tensor) -> torch.Tensor:
        # (B,T) -> (B,T,C)
        B, T = x.shape
        pos_embed = self.position_embedding(torch.arange(T).to(x.device))
        tok_embed =  + self.token_embedding(x)

        # (B,T,C) -> (B,T,H)
        x = tok_embed + pos_embed
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_padding_mask)
        # (B,T,H) -> (B,T,output_vocab_size)
        x = self.linear(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self,
                 num_heads: int,
                 embed_dim: int,
                 d_model: int,
                 ffwd_dim: int,
                 max_seq_len: int):
        super(DecoderLayer, self).__init__()
        self.masked_mha = MultiHeadSelfAttention(num_heads, embed_dim, d_model)
        self.mh_cross_attention = MultiHeadCrossAttention(num_heads, d_model)
        self.ffwd = FeedForward(embed_dim, ffwd_dim)
        self.register_buffer('tril', torch.tril(torch.ones((max_seq_len, max_seq_len))))

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, decoder_padding_mask: torch.Tensor) -> torch.Tensor:
        # B,T,C -> B,T,H
        B,T,C = x.shape
        decoder_causal_mask = (self.tril[:T, :T] == 0)
        # (B,T) -> B,T,T
        expanded_padding_mask = decoder_padding_mask.unsqueeze(1).expand(-1, T, -1)
        # B,T,T
        combined_mask = expanded_padding_mask | decoder_causal_mask
        x = f.layer_norm(x + self.masked_mha(x, combined_mask), x.shape)
        # B,T,H -> B,T,H
        x = f.layer_norm(x + self.mh_cross_attention(x, encoder_out), x.shape)
        # B,T,H -> B,T,H
        x = f.layer_norm(x + self.ffwd(x), x.shape)
        return x


class MultiHeadCrossAttention(nn.Module):#
    def __init__(self, num_heads: int, d_model: int):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList([
            CrossAttentionHead(d_model, head_dim)
            for _ in range(num_heads)
        ])
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        # B,T,H -> B,T,H
        x = torch.cat([head(x, encoder_out) for head in self.heads], dim=-1)
        # B,T,H -> B,T,H
        x = self.linear(x)
        return x

class CrossAttentionHead(nn.Module):
    def __init__(self, d_model: int, head_dim: int):
        super(CrossAttentionHead, self).__init__()
        self.q = nn.Linear(d_model, head_dim)
        self.k = nn.Linear(d_model, head_dim)
        self.v = nn.Linear(d_model, head_dim)
    
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        # queries come from previous decoder layer
        # B,T,H @ H,H -> B,T,H
        queries = self.q(x) 

        # keys and values come from encoder output
        # B,T,H -> B,T,H
        keys = self.k(encoder_out)
        # B,T,H -> B,T,H
        values = self.v(encoder_out)

        # decoder queries can attend to all parts of the encoder output
        # (B,T,H) @ (B,H,T) = (B,T,T)
        similarities = (queries @ keys.transpose(-2,-1)) / sqrt(keys.shape[-1])

        # B,T,T
        weighted_similarities = torch.softmax(similarities, dim=-1)

        # (B,T,T) @ (B,T,H) = B,T,H
        out = weighted_similarities @ values
        return out