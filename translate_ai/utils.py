#!/usr/bin/env python3
import torch
import torch.nn.functional as f

def log(msg: str, rank: int = 0):
    if rank == 0:
        print(msg)

def calculate_sparsity(attn_scores, threshold=1e-2):
    """
    Calculate the fraction of attention scores that are below a given threshold.
    
    attn_weights: (batch_size, num_heads, seq_len, seq_len)
    threshold: value below which attention weights are considered sparse.
    """
    total_elements = attn_scores.numel()
    sparse_elements = (attn_scores.abs() < threshold).sum().item()
    
    return sparse_elements / total_elements
