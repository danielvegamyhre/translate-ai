#!/usr/bin/env python3
import torch
import torch.nn.functional as f

def log(msg: str, rank: int = 0):
    if rank == 0:
        print(msg)

def sparse_attention_loss(attention_scores: list[torch.tensor], recon_attention_scores: list[torch.tensor], sparsity_lambda: float = 1e-3):
    recon_loss = f.mse_loss(recon_attention_scores, attention_scores)
    sparsity_loss = sparsity_lambda * torch.mean(torch.abs(recon_attention_scores))
    return recon_loss + sparsity_loss