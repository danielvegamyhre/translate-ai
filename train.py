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
from checkpoint import save_checkpoint, load_checkpoint

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
    device: str
    save_checkpoint: str
    load_checkpoint: str

def train(cfg: TrainingConfig) -> None:
    device = torch.device(cfg.device)
    print("device: ", device)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    print(f"tokenizer: tiktoken cl100k_base")
    print(f"vocab size: {tokenizer.n_vocab}")
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
    
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # load checkpoint if specified
    curr_step = 0
    if cfg.load_checkpoint:
        checkpoint = load_checkpoint(cfg.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_step = checkpoint['step']

    for step in tqdm(range(curr_step, cfg.steps)):
        encoder_input, decoder_targets = get_batch(dataset, cfg.seq_len, cfg.batch_size)
        encoder_input = encoder_input.to(device)

        # remove last token of targets to get decoder inputs
        decoder_input = decoder_targets[:, :-1].to(device)
        # remove first token for targets so the target seq is offset from input by 1
        decoder_targets = decoder_targets[:, 1:].to(device)

        out = model(encoder_input, decoder_input)
        pred_probs = f.softmax(out, dim=-1)

        # flatten predicted probs and targets for cross entropy loss
        B,T,C = pred_probs.shape
        # B,T,vocab_size -> B*T,vocab_size
        pred_probs = pred_probs.reshape(B*T,C)
        # B,T -> B*T
        decoder_targets = decoder_targets.reshape(-1)

        loss = f.cross_entropy(pred_probs, decoder_targets)
        print(f"step: {step}, loss: {loss}")

        optim.zero_grad()
        loss.backward()
        optim.step()
    
    # save checkpoint if specified
    if cfg.save_checkpoint:
        save_checkpoint(cfg.save_checkpoint, step, cfg, model, optim)


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
    argparser.add_argument("--device", type=str, default="cpu")
    argparser.add_argument("--load-checkpoint", type=str)
    argparser.add_argument("--save-checkpoint", type=str)
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
        device=args.device,
        load_checkpoint=args.load_checkpoint,
        save_checkpoint=args.save_checkpoint,
    )
    train(cfg)