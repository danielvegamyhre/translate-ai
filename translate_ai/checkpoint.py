#!/usr/bin/env python3
import os
import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from config import TrainingConfig

def save_checkpoint(cfg: TrainingConfig, epoch: int, model: nn.Module, optim: Optimizer):
    path = cfg.save_checkpoint
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

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

def load_state_dict(model: nn.Module, checkpoint: dict) -> None:
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key in state_dict.keys():
        # models trained with DDP will have a 'module.' prefix
        # on the state dict keys that need to be removed in order
        # to restore the checkpoint.
        if key.startswith("module."):
            new_key = key[len("module."):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]

    model.load_state_dict(new_state_dict)
