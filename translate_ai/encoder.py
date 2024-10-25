#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as f
from math import sqrt

from utils import log, calculate_sparsity

class Encoder(nn.Module):
    def __init__(self, 
                 num_layers: int,
                 vocab_size: int, 
                 embed_dim: int = 64,
                 d_model: int = 512,
                 input_seq_len: int = 512, 
                 num_attention_heads: int = 4,
                 ffwd_dim: int = 2048,
                 dropout: float = 0.1,
                 device: str = "cpu"):
        super(Encoder, self).__init__()
        self.position_embedding = nn.Embedding(input_seq_len, embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, d_model, num_attention_heads, ffwd_dim, input_seq_len=input_seq_len)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x: torch.Tensor, encoder_padding_mask: torch.Tensor) -> torch.Tensor:
        # (B,T) -> (B,T,C)
        B, T = x.shape
        tok_embed = self.token_embedding(x)
        pos_embed = self.position_embedding(torch.arange(T).to(x.device))
        out = tok_embed + pos_embed

        # B,T -> B,1,T so it can be broadcasted across attention scores
        encoder_padding_mask = encoder_padding_mask.unsqueeze(1)
        sparsity_loss = torch.tensor(0.0, device=self.device)
        for layer in self.layers:
            # B,T,C -> B,T,C
            out, layer_sparsity_loss = layer(out, encoder_padding_mask)
            sparsity_loss += layer_sparsity_loss
        out = self.dropout(out)
        return out, sparsity_loss


class EncoderLayer(nn.Module):
    def __init__(self, 
                 embed_dim: int = 128,
                 d_model: int = 512,
                 num_attention_heads: int = 4,
                 ffwd_dim: int = 2048,
                 input_seq_len: int = 128):
        
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadSelfAttention(num_attention_heads, embed_dim, d_model, seq_len=input_seq_len)
        self.ffwd = FeedForward(embed_dim, ffwd_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, encoder_padding_mask: torch.Tensor) -> torch.Tensor:
        # (B,T,C) -> (B,T,H)
        out, sparsity_loss = self.mha(x, encoder_padding_mask)
        x = self.ln1(x + out)

        # (B,T,H) -> (B,T,H)
        x = self.ln2(x + self.ffwd(x))
        return x, sparsity_loss

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                 num_heads: int,
                 embed_dim: int,
                 d_model: int,
                 seq_len: int = 128,
                 dropout: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList([
            SparseSelfAttentionHead(embed_dim, head_dim, seq_len=seq_len)
            for _ in range(num_heads)
        ])
        self.linear = nn.Linear(d_model, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.tensor, list[torch.tensor], list[torch.tensor]]:
        head_outputs = []
        sparsity_loss = torch.tensor(0.0, device="mps")
        for head in self.heads:
            out, head_sparsity_loss = head(x, mask)
            head_outputs.append(out)
            sparsity_loss += head_sparsity_loss

        # head(x) => (B,T,head_size)
        # concat them all along head_size dim -> (B,T,H)
        # since H = head_size * num_heads
        x = torch.cat(head_outputs, dim=-1)

        # project to embed dimension to enable residual (i.e. x + mha(x))
        x = self.linear(x)  #  (B,T,H) -> B,T,C
        x = self.dropout(x)
        return x, sparsity_loss

class SparseSelfAttentionHead(nn.Module):
    '''Self attention head takes 2 parameters:
    embed_dim: embedding dimension from token embeddings.
    head_dim: total hidden dimension size divided by the number of attention heads.'''
    def __init__(self,
                 embed_dim: int = 128,
                 head_dim: int = 128,
                 seq_len: int = 128,
                 dropout: float = 0.1,
                 sparsity_lambda: float = 1e-3):
        super(SparseSelfAttentionHead, self).__init__()

        self.sparsity_lambda = sparsity_lambda
        self.Q1 = nn.Linear(embed_dim, head_dim)
        #self.Q2 = nn.Linear(embed_dim, head_dim)
        self.K1 = nn.Linear(embed_dim, head_dim)
        #self.K2 = nn.Linear(embed_dim, head_dim)
        self.V = nn.Linear(embed_dim, head_dim)
        self.sae1 = SparseAutoencoderLayer(seq_len, seq_len//4)
        #self.sae2 = SparseAutoencoderLayer(seq_len, seq_len//4)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        '''Returns tuple of (output, raw attention scores, sparse attention scores).'''
        # (B,T,C) @ (C,head_dim) = (B,T,head_dim)
        q1 = self.Q1(x) 

        # (B,T,C) @ (C,head_dim) = (B,T,head_dim)
        k1 = self.K1(x)

        # (B,T,H) @ (B,H,T) = (B,T,T)
        raw_scores = (q1 @ k1.transpose(-2,-1)) * sqrt(1.0/k1.shape[-1])

        # sae layer, reconstruct scores with increased sparsity to reduce noise
        # (B,T,T) @ (T, T//4) @ (T//4, T)= (B,T,T)
        sparse_scores, latent = self.sae1(raw_scores)
        
        # optionally mask tokens (e.g., mask padding tokens to prevent model from attending
        # to them, or mask tokens temporally ahead of each token to preserve autoregressive property).
        if mask is not None:
            sparse_scores = sparse_scores.masked_fill(mask, -float('inf'))

        # element-wise, shape stays same
        weighted_scores = torch.softmax(sparse_scores, dim=-1)

        log(f"attention score sparsity: {calculate_sparsity(weighted_scores)}")

        # compute sparsity loss (L1 penalty)
        sparsity_loss = self.sparsity_lambda * torch.sum(torch.abs(weighted_scores))

        # apply dropout
        weighted_scores = self.dropout(weighted_scores)

        # (B,T,C) @ (C,head_dim) = (B,T,head_dim)
        values = self.V(x)

        # (B,T,T) @ (B,T,head_dim) = B,T,head_dim
        out = weighted_scores @ values
        return out, sparsity_loss

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

class SparseAutoencoderLayer(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity_lambda=1e-3):
        super(SparseAutoencoderLayer, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x):
        latent = torch.relu(self.encoder(x))
        reconstructed = torch.sigmoid(self.decoder(latent))
        return reconstructed, latent
