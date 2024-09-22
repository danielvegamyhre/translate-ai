#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as f
from torch.utils.data import Dataset
from dataclasses import dataclass
from argparse import ArgumentParser
import tiktoken
from tqdm import tqdm

from transformer import TransformerTranslator
from dataset import EnglishToSpanishDataset


@dataclass
class TrainingConfig:
    steps: int
    learning_rate: float
    num_layers: int
    eval_interval: int
    checkpoint_interval: int
    batch_size: int
    dataset_file: str
    seq_len: int

def train(cfg: TrainingConfig) -> None:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    dataset = EnglishToSpanishDataset(cfg.dataset_file, tokenizer)
    model = TransformerTranslator(
        input_vocab_size=tokenizer.n_vocab, 
        output_vocab_size=tokenizer.n_vocab,
        embed_dim=512,
        d_model=512,
        num_encoder_layers=cfg.num_layers,
        num_decoder_layers=cfg.num_layers,
        num_attention_heads=8,
        ffwd_dim=2048,
        max_seq_len=128,
        max_output_tokens=128)
    
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    for step in tqdm(range(cfg.steps)):
        x, y = get_batch(dataset, cfg.seq_len, cfg.batch_size)
        out = model(x, y)
        pred_probs = f.softmax(out, dim=-1)
        B,T,C = pred_probs.shape
        # B,T,vocab_size -> B*T,vocab_size
        pred_probs = pred_probs.view(B*T,C)
        # B,T -> B*T
        y = y.view(-1)
        loss = f.cross_entropy(pred_probs, y)
        print(f"step: {step}, loss: {loss}")
        optim.zero_grad()
        loss.backward()
        optim.step()

def get_batch(dataset: Dataset, seq_len: int = 8, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    # get one random index per batch
    indexes = torch.randint(len(dataset) - seq_len, (batch_size,))
    x, y = [], []
    for idx in indexes:
        xb, yb = dataset[idx]
        x.append(xb)
        y.append(yb)
    return torch.stack(x), torch.stack(y)

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--steps", type=int, default=100)
    argparser.add_argument("--learning-rate", type=float, default=1e-3)
    argparser.add_argument("--num-layers", type=int, default=6)
    argparser.add_argument("--eval-interval", type=int, default=100)
    argparser.add_argument("--checkpoint-interval", type=int, default=100)  
    argparser.add_argument("--batch-size", type=int, default=32) 
    argparser.add_argument("--dataset-file", type=str, required=True)
    argparser.add_argument("--seq-len", type=int, default=128)
    args = argparser.parse_args()
    cfg = TrainingConfig(
        steps=args.steps,
        learning_rate=args.learning_rate,
        num_layers=args.num_layers,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.eval_interval,
        batch_size=args.batch_size,
        dataset_file=args.dataset_file,
        seq_len=args.seq_len,
    )
    train(cfg)