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
                 num_heads: int = 4,
                 embed_dim: int = 128,
                 d_model: int = 512,    
                 ffwd_dim: int = 2048,
                 output_seq_len: int = 128,
                 device: str = "cpu"):
        super(Decoder, self).__init__()
        self.position_embedding = nn.Embedding(output_seq_len, embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(num_heads, embed_dim, d_model, ffwd_dim, output_seq_len=output_seq_len) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.device = device

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, decoder_padding_mask: torch.Tensor = None) -> tuple[torch.tensor, float]:
        # (B,T) -> (B,T,C)
        B, T = x.shape
        pos_embed = self.position_embedding(torch.arange(T).to(x.device))
        tok_embed = self.token_embedding(x)
        x = tok_embed + pos_embed
        sparsity_loss = torch.tensor(0.0, device=self.device)
        for layer in self.layers:
            x, layer_sparsity_loss = layer(x, encoder_out, decoder_padding_mask)     # (B,T,C) -> (B,T,C)
            sparsity_loss += layer_sparsity_loss

        # (B,T,C) @ (C, output_vocab_size) = (B,T,output_vocab_size)
        x = self.linear(x)
        return x, sparsity_loss

class DecoderLayer(nn.Module):
    def __init__(self,
                 num_heads: int,
                 embed_dim: int,
                 d_model: int,
                 ffwd_dim: int,
                 output_seq_len: int = 128):
        super(DecoderLayer, self).__init__()
        self.masked_mha = MultiHeadSelfAttention(num_heads, embed_dim, d_model, seq_len=output_seq_len)
        self.mh_cross_attention = MultiHeadCrossAttention(num_heads, embed_dim, d_model)
        self.ffwd = FeedForward(embed_dim, ffwd_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.register_buffer('tril', torch.tril(torch.ones((output_seq_len, output_seq_len))))

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, decoder_padding_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        B,T,C = x.shape
        decoder_causal_mask = (self.tril[:T, :T] == 0)
        if decoder_padding_mask is not None:
            decoder_padding_mask = decoder_padding_mask.unsqueeze(1)   # B,T -> B,T,1
            decoder_padding_mask = decoder_padding_mask.expand(-1,T,-1) # B,T,1 -> B,T,T

        # B,T,T
        combined_mask = decoder_causal_mask
        if decoder_padding_mask is not None:
            combined_mask = (decoder_padding_mask | decoder_causal_mask)

        # we keep attention scores to do sparsity loss during training
        outputs, sparsity_loss = self.masked_mha(x, combined_mask)    # B,T,C -> B,T,C
        x = self.ln1(x + outputs)
        x = self.ln2(x + self.mh_cross_attention(x, encoder_out))   # B,T,C -> B,T,C
        x = self.ln3(x + self.ffwd(x))                              # B,T,H -> B,T,H
        return x, sparsity_loss


class MultiHeadCrossAttention(nn.Module):#
    def __init__(self, num_heads: int, embed_dim: int, d_model: int, dropout: float = 0.1):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList([
            CrossAttentionHead(embed_dim, head_dim)
            for _ in range(num_heads)
        ])
        self.linear = nn.Linear(d_model, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        # each head: B,T,C -> B,T,head_dim
        # after concat: B,T,d_model
        x = torch.cat([head(x, encoder_out) for head in self.heads], dim=-1)
        x = self.linear(x)  # B,T,d_model -> B,T,C
        x = self.dropout(x)
        return x

class CrossAttentionHead(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 head_dim: int,
                 dropout: float = 0.1):
        super(CrossAttentionHead, self).__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        # queries come from previous decoder layer
        # B,T,C @ C,head_dim -> B,T,head_dim
        queries = self.q(x) 

        # keys and values come from encoder output
        # B,T,C @ C,head_dim -> B,T,head_dim
        keys = self.k(encoder_out)
        
        # B,T,C @ C,head_dim -> B,T,head_dim
        values = self.v(encoder_out)

        # no masking, decoder queries can attend to all parts of the encoder output.
        # (B,T,C) @ (B,C,T) = (B,T,T)
        scores = (queries @ keys.transpose(-2,-1)) / sqrt(keys.shape[-1])

        # B,T,T
        weighted_scores = torch.softmax(scores, dim=-1)
        weighted_scores = self.dropout(weighted_scores)

        # (B,T,T) @ (B,T,head_dim) = B,T,head_dim
        out = weighted_scores @ values
        return out