
#!/usr/bin/env python3
import torch
from torch import nn
from math import sqrt

class DifferentialMultiHeadCrossAttention(nn.Module):
    '''Differential Attention (Ye, et al. 2024: https://arxiv.org/pdf/2410.05258)
    applied to cross-attention layer in encoder-decoder transformer.'''
    def __init__(self, num_heads: int, embed_dim: int, d_model: int, dropout: float = 0.0, lambda_init: float = 0.8, layer_index: int = 0):
        super(DifferentialMultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.d_model = d_model
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList([
            DifferentialCrossAttentionHead(embed_dim, head_dim, lambda_init=lambda_init, layer_index=layer_index)
            for _ in range(num_heads)
        ])
        self.linear = nn.Linear(d_model, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rms_norms = nn.ModuleList([nn.RMSNorm(embed_dim) for _ in range(num_heads)])
        self.lambda_init = lambda_init

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        # each head: B,T,C -> B,T,head_dim
        # after concat: B,T,d_model
        heads = [head(x, encoder_out) for head in self.heads]

        # apply RMS norm to each head (GroupRMSNorm)
        heads = [(1 - self.lambda_init) * self.rms_norms[i](head) for i, head in enumerate(heads)]

        x = torch.cat(heads, dim=-1)
        x = self.linear(x) # B,T,d_model -> B,T,C
        x = self.dropout(x)
        return x

class DifferentialCrossAttentionHead(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 head_dim: int,
                 dropout: float = 0.0,
                 lambda_init: float = 0.8,
                 layer_index: int = 0):
        super(DifferentialCrossAttentionHead, self).__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.W_q = nn.Linear(embed_dim, head_dim)
        self.W_k = nn.Linear(embed_dim, head_dim)
        self.W_v = nn.Linear(embed_dim, head_dim)
        self.lambda_q1 = nn.Parameter(torch.randn(head_dim))
        self.lambda_k1 = nn.Parameter(torch.randn(head_dim))
        self.lambda_q2 = nn.Parameter(torch.randn(head_dim))
        self.lambda_k2 = nn.Parameter(torch.randn(head_dim))
        self.lambda_init = nn.Parameter(lambda_init - 0.6 * torch.exp(torch.tensor(-0.3) * layer_index))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        # compute lambda from reparameterization
        lambda_value = torch.exp(self.lambda_q1 @ self.lambda_k1) \
                     - torch.exp(self.lambda_q2 @ self.lambda_k2) \
                     + self.lambda_init
        
        # queries come from previous decoder layer
        # B,T,C @ C,head_dim -> B,T,head_dim
        q1, q2 = torch.chunk(self.W_q(x), 2, dim=-1)

        # keys and values come from encoder output
        # B,T,C @ C,head_dim -> B,T,head_dim
        k1, k2 = torch.chunk(self.W_k(encoder_out), 2, dim=-1)
        
        # no masking, decoder queries can attend to all parts of the encoder output.
        # (B,T,C) @ (B,C,T) = (B,T,T)
        scores1 = (q1 @ k1.transpose(-2,-1)) / sqrt(k1.shape[-1])
        scores2 = (q2 @ k2.transpose(-2,-1)) / sqrt(k2.shape[-1])

        # B,T,T
        weighted_scores1 = torch.softmax(scores1, dim=-1)
        weighted_scores2 = torch.softmax(scores2, dim=-1)

        # subtract weighted scores to reduce noise
        weighted_scores = weighted_scores1 - lambda_value * weighted_scores2
        weighted_scores = self.dropout(weighted_scores)

        # B,T,C @ C,head_dim -> B,T,head_dim
        values = self.W_v(encoder_out)

        # (B,T,T) @ (B,T,head_dim) = B,T,head_dim
        out = weighted_scores @ values
        return out

class DifferentialMultiHeadSelfAttention(nn.Module):
    '''Differential Attention (Ye, et al. 2024: https://arxiv.org/pdf/2410.05258)'''   
    def __init__(self, 
                 num_heads: int,
                 embed_dim: int,
                 d_model: int,
                 dropout: float = 0.0,
                 lambda_init: float = 0.8, 
                 layer_index: int = 0):
        super(DifferentialMultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.d_model = d_model
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList([
            DifferentialSelfAttentionHead(embed_dim, head_dim, lambda_init=lambda_init, layer_index=layer_index)
            for _ in range(num_heads)
        ])
        self.linear = nn.Linear(d_model, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rms_norms = nn.ModuleList([nn.RMSNorm(embed_dim) for _ in range(num_heads)])
        self.lambda_init = lambda_init

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # head(x) => (B,T,head_size)
        heads = [head(x, mask) for head in self.heads]

        # apply RMS norm on each head separately
        heads = [(1 - self.lambda_init) * self.rms_norms[i](head) for i, head in enumerate(heads)]

        # concat them all along head_size dim -> (B,T,H)
        # since H = head_size * num_heads 
        x = torch.cat(heads, dim=-1)

        # project to embed dimension to enable residual (i.e. x + mha(x))
        # (B,T,H) -> B,T,C
        x = self.linear(x)
        x = self.dropout(x)
        return x

class DifferentialSelfAttentionHead(nn.Module):
    def __init__(self,
                 embed_dim: int = 128,
                 head_dim: int = 128,
                 dropout: float = 0.0,
                 lambda_init: float = 0.8,
                 layer_index: int = 0):
        super(DifferentialSelfAttentionHead, self).__init__()
        self.W_q = nn.Linear(embed_dim, head_dim)
        self.W_k = nn.Linear(embed_dim, head_dim)
        self.W_v = nn.Linear(embed_dim, head_dim)
        self.dropout = nn.Dropout(dropout)
        self.lambda_q1 = nn.Parameter(torch.randn(head_dim))
        self.lambda_k1 = nn.Parameter(torch.randn(head_dim))
        self.lambda_q2 = nn.Parameter(torch.randn(head_dim))
        self.lambda_k2 = nn.Parameter(torch.randn(head_dim))
        self.lambda_init = nn.Parameter(lambda_init - 0.6 * torch.exp(torch.tensor(-0.3) * layer_index))
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # compute lambda from reparameterization
        lambda_value = torch.exp(self.lambda_q1 @ self.lambda_k1) \
                     - torch.exp(self.lambda_q2 @ self.lambda_k2) \
                     + self.lambda_init
        
        # (B,T,C) @ (C,head_dim) = (B,T,head_dim)
        q1, q2 = torch.chunk(self.W_q(x), 2, dim=-1)

        # (B,T,C) @ (C,head_dim) = (B,T,head_dim)
        k1, k2 = torch.chunk(self.W_k(x), 2, dim=-1)

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
        weighted_scores = weighted_scores1 - lambda_value * weighted_scores2

        # apply dropout
        weighted_scores = self.dropout(weighted_scores)

        # (B,T,C) @ (C,head_dim) = (B,T,head_dim)
        values = self.W_v(x)

        # (B,T,T) @ (B,T,head_dim) = B,T,head_dim
        out = weighted_scores @ values
        return out