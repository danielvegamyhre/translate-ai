#!/usr/bin/env python3

import torch
from torch import nn

from encoder import FeedForward
from attention import DifferentialMultiHeadCrossAttention, DifferentialMultiHeadSelfAttention

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
        self.layers = nn.ModuleList([
            DecoderLayer(num_heads, embed_dim, d_model, ffwd_dim, max_seq_len, layer_index=i) 
             for i in range(num_layers)
        ])
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, decoder_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # (B,T) -> (B,T,C)
        B, T = x.shape
        pos_embed = self.position_embedding(torch.arange(T).to(x.device))
        tok_embed = self.token_embedding(x)
        x = tok_embed + pos_embed
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_padding_mask)     # (B,T,C) -> (B,T,C)

        # (B,T,C) @ (C, output_vocab_size) = (B,T,output_vocab_size)
        x = self.linear(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self,
                 num_heads: int,
                 embed_dim: int,
                 d_model: int,
                 ffwd_dim: int,
                 max_seq_len: int,
                 layer_index: int):
        super(DecoderLayer, self).__init__()
        self.masked_mha = DifferentialMultiHeadSelfAttention(num_heads, embed_dim, d_model)
        self.mh_cross_attention = DifferentialMultiHeadCrossAttention(num_heads, embed_dim, d_model, layer_index=layer_index)
        self.ffwd = FeedForward(embed_dim, ffwd_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.register_buffer('tril', torch.tril(torch.ones((max_seq_len, max_seq_len))))

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, decoder_padding_mask: torch.Tensor = None) -> torch.Tensor:
        B,T,C = x.shape
        decoder_causal_mask = (self.tril[:T, :T] == 0)
        if decoder_padding_mask is not None:
            decoder_padding_mask = decoder_padding_mask.unsqueeze(1)   # B,T -> B,T,1
            decoder_padding_mask = decoder_padding_mask.expand(-1,T,-1) # B,T,1 -> B,T,T

        # B,T,T
        combined_mask = decoder_causal_mask
        if decoder_padding_mask is not None:
            combined_mask = (decoder_padding_mask | decoder_causal_mask)

        x = self.ln1(x + self.masked_mha(x, combined_mask))         # B,T,C -> B,T,C
        x = self.ln2(x + self.mh_cross_attention(x, encoder_out))   # B,T,C -> B,T,C
        x = self.ln3(x + self.ffwd(x))                              # B,T,H -> B,T,H
        return x

