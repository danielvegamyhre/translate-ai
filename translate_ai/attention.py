
#!/usr/bin/env python3
import torch
from torch import nn
from math import sqrt

class DifferentialMultiHeadCrossAttention(nn.Module):#
    def __init__(self, num_heads: int, embed_dim: int, d_model: int, dropout: float = 0.1, lambda_init: float = 0.8):
        super(DifferentialMultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.d_model = d_model
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList([
            DifferentialCrossAttentionHead(embed_dim, head_dim, lambda_init=lambda_init)
            for _ in range(num_heads)
        ])
        self.linear = nn.Linear(d_model, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        # each head: B,T,C -> B,T,head_dim
        # after concat: B,T,d_model
        x = torch.cat([head(x, encoder_out) for head in self.heads], dim=-1)

        # group norm expects dims to be (batch, channels, seq len), 
        # so permute B,T,C -> B,C,T
        x = x.permute(0, 2, 1)
        x = self.group_norm(x)

        # reshape B,C,T back to B,T,C
        x = x.permute(0, 2, 1)
        x = self.linear(x)  # B,T,d_model -> B,T,C
        x = self.dropout(x)
        return x

class DifferentialCrossAttentionHead(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 head_dim: int,
                 dropout: float = 0.1,
                 lambda_init: float = 0.8):
        super(DifferentialCrossAttentionHead, self).__init__()
        self.Q1 = nn.Linear(embed_dim, head_dim)
        self.Q2 = nn.Linear(embed_dim, head_dim)
        self.K1 = nn.Linear(embed_dim, head_dim)
        self.K2 = nn.Linear(embed_dim, head_dim)
        self.V = nn.Linear(embed_dim, head_dim)
        self.dropout = nn.Dropout(dropout)
        self._lambda = nn.Parameter(torch.tensor(lambda_init))
    
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        # queries come from previous decoder layer
        # B,T,C @ C,head_dim -> B,T,head_dim
        q1 = self.Q1(x) 
        q2 = self.Q2(x)

        # keys and values come from encoder output
        # B,T,C @ C,head_dim -> B,T,head_dim
        k1 = self.K1(encoder_out)
        k2 = self.K2(encoder_out)
        
        # no masking, decoder queries can attend to all parts of the encoder output.
        # (B,T,C) @ (B,C,T) = (B,T,T)
        scores1 = (q1 @ k1.transpose(-2,-1)) / sqrt(k1.shape[-1])
        scores2 = (q2 @ k2.transpose(-2,-1)) / sqrt(k2.shape[-1])

        # B,T,T
        weighted_scores1 = torch.softmax(scores1, dim=-1)
        weighted_scores2 = torch.softmax(scores2, dim=-1)

        # subtract weighted scores to reduce noise
        weighted_scores = weighted_scores1 - self._lambda * weighted_scores2
        weighted_scores = self.dropout(weighted_scores)

        # B,T,C @ C,head_dim -> B,T,head_dim
        values = self.V(encoder_out)

        # (B,T,T) @ (B,T,head_dim) = B,T,head_dim
        out = weighted_scores @ values
        return out
    
class DifferentialMultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                 num_heads: int,
                 embed_dim: int,
                 d_model: int,
                 dropout: float = 0.1,
                 lambda_init: float = 0.8):
        super(DifferentialMultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.d_model = d_model
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList([
            DifferentialSelfAttentionHead(embed_dim, head_dim, lambda_init=lambda_init)
            for _ in range(num_heads)
        ])
        self.linear = nn.Linear(d_model, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # head(x) => (B,T,head_size)
        # concat them all along head_size dim -> (B,T,H)
        # since H = head_size * num_heads 
        x = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        
        # group norm expects dims to be (batch, channels, seq len), 
        # so permute B,T,C -> B,C,T
        x = x.permute(0, 2, 1)
        x = self.group_norm(x)
        
        # reshape B,C,T back to B,T,C
        x = x.permute(0, 2, 1)

        # project to embed dimension to enable residual (i.e. x + mha(x))
        x = self.linear(x)  #  (B,T,H) -> B,T,C
        x = self.dropout(x)
        return x

class DifferentialSelfAttentionHead(nn.Module):
    '''Self attention head takes 2 parameters:
    embed_dim: embedding dimension from token embeddings.
    head_dim: total hidden dimension size divided by the number of attention heads.'''
    def __init__(self,
                 embed_dim: int = 128,
                 head_dim: int = 128,
                 dropout: float = 0.1,
                 lambda_init: float = 0.8):
        super(DifferentialSelfAttentionHead, self).__init__()
        self.Q1 = nn.Linear(embed_dim, head_dim)
        self.Q2 = nn.Linear(embed_dim, head_dim)
        self.K1 = nn.Linear(embed_dim, head_dim)
        self.K2 = nn.Linear(embed_dim, head_dim)
        self.V = nn.Linear(embed_dim, head_dim)
        self.dropout = nn.Dropout(dropout)
        self._lambda = nn.Parameter(torch.tensor(lambda_init))
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # (B,T,C) @ (C,head_dim) = (B,T,head_dim)
        q1 = self.Q1(x) 
        q2 = self.Q2(x)

        # (B,T,C) @ (C,head_dim) = (B,T,head_dim)
        k1 = self.K1(x)
        k2 = self.K2(x)

        # (B,T,H) @ (B,H,T) = (B,T,T)
        scores1 = (q1 @ k1.transpose(-2,-1)) / sqrt(k1.shape[-1])
        scores2 = (q2 @ k2.transpose(-2,-1)) / sqrt(k2.shape[-1])

        # optionally mask tokens (e.g., mask padding tokens to prevent model from attending
        # to them, or mask tokens temporally ahead of each token to preserve autoregressive property).
        if mask is not None:
            scores1 = scores1.masked_fill(mask, -float('inf'))
            scores2 = scores2.masked_fill(mask, -float('inf'))

        # element-wise, shape stays same
        weighted_scores1 = torch.softmax(scores1, dim=-1)
        weighted_scores2 = torch.softmax(scores2, dim=-1)
        
        # subtract weighted scores to reduce noise in attention scores
        weighted_scores = weighted_scores1 - self._lambda * weighted_scores2

        # apply dropout
        weighted_scores = self.dropout(weighted_scores)

        # (B,T,C) @ (C,head_dim) = (B,T,head_dim)
        values = self.V(x)

        # (B,T,T) @ (B,T,head_dim) = B,T,head_dim
        out = weighted_scores @ values
        return out