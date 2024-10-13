#!/usr/bin/env python3

import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from config import TrainingConfig

def save_checkpoint(cfg: TrainingConfig, epoch: int, model: nn.Module, optim: Optimizer):
    path = cfg.save_checkpoint
    print(f'checkpointing to {path}') 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'config': cfg,
            }, path)

def load_checkpoint(path: str) -> dict:
    print(f'loading checkpoint from path {path}')
    checkpoint = torch.load(path)
    print(f'loaded checkpoint from {path}')
    return checkpoint