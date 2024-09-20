#!/usr/bin/env python3

import torch
from torch.nn import functional as f
from dataclasses import dataclass
from argparse import ArgumentParser

from .transformer import TransformerTranslator

@dataclass
class TrainingConfig:
    steps: int
    learning_rate: float
    num_layers: int
    eval_interval: int
    checkpoint_interval: int
    batch_size: int

def train(cfg: TrainingConfig) -> None:
    pass

if __name__ == '__main__':
    argparser = ArgumentParser()
    args = argparser.parse()
    argparser.add_argument("steps", type=int, default=100)
    argparser.add_argument("learning_rate", type=float, default=1e-3)
    argparser.add_argument("num_layers", type=int, default=6)
    argparser.add_argument("eval_interval", type=int, default=100)
    argparser.add_argument("checkpoint_interval", type=int, default=100)  
    argparser.add_argument("batch_size", type=int, default=32) 
    cfg = TrainingConfig(
        steps=args.steps,
        learning_reate=args.learning_rate,
        num_layers=args.num_layers,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.eval_interval,
        batch_size=args.batch_size,
    )
    train(cfg)