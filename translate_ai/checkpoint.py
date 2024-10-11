#!/usr/bin/env python3

import torch
from torch import nn
from torch.optim.optimizer import Optimizer

def save_checkpoint(path: str, epoch: int, cfg, model: nn.Module, optim: Optimizer):
    print(f'checkpointing to {path}') 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'config': cfg,
            }, path)

def load_checkpoint(path: str):
    print(f'loading checkpiont from path {path}')
    checkpoint = torch.load(path)
    print(f'loaded checkpoint from {path}')
    return checkpoint