#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as f
from torch.utils.data import Dataset, DataLoader, random_split
from dataclasses import dataclass
from argparse import ArgumentParser
import tiktoken
from tqdm import tqdm

from transformer import TransformerTranslator
from dataset import EnglishToSpanishDataset
from checkpoint import save_checkpoint, load_checkpoint
from plotting import plot_learning_curves
from scheduler import NoamLR

@dataclass
class TrainingConfig:
    epochs: int
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
    debug: bool
    plot_learning_curves: bool

def train(cfg: TrainingConfig) -> None:
    # configure device
    device = torch.device(cfg.device)
    print("device: ", device)

    # initialize tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    print(f"tokenizer: tiktoken cl100k_base")
    print(f"vocab size: {tokenizer.n_vocab}")

    # initialize dataset
    dataset = EnglishToSpanishDataset(cfg.dataset_file, tokenizer)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # initalize dataloaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # initialize model
    d_model = 512
    model = TransformerTranslator(
        input_vocab_size=tokenizer.n_vocab, 
        output_vocab_size=tokenizer.n_vocab,
        embed_dim=512,
        d_model=d_model,
        num_encoder_layers=cfg.num_layers,
        num_decoder_layers=cfg.num_layers,
        num_attention_heads=8,
        ffwd_dim=2048,
        max_seq_len=128,
        max_output_tokens=128).to(device)

    # set up optimizer and learning rate scheduler 
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    warmup_steps = 4000
    lr_scheduler = NoamLR(optim, warmup_steps)

    # load checkpoint if specified
    curr_epoch = 0
    if cfg.load_checkpoint:
        checkpoint = load_checkpoint(cfg.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_epoch = checkpoint['epoch']

    # training loop
    train_losses, val_losses = [], []
    model.train()
    for epoch in range(curr_epoch, curr_epoch + cfg.epochs):
        print(f"Epoch: {epoch}")
        total_train_loss = 0.0
        for step, (encoder_inputs, decoder_targets) in tqdm(enumerate(train_loader)):
            # encoder_input, decoder_targets = get_batch(dataset, cfg.seq_len, cfg.batch_size)
            encoder_input = encoder_inputs.to(device)

            # remove last token of targets to get decoder inputs
            decoder_input = decoder_targets[:, :-1].to(device)

            # remove first token for targets so the target seq is offset from input by 1
            decoder_targets = decoder_targets[:, 1:].to(device)
            if cfg.debug:
                print('decoder targets min', decoder_targets.min().item(), 'max', decoder_targets.max().item())

            logits = model(encoder_input, decoder_input)
            if cfg.debug:
                print('logits min', logits.min().item(), 'max', logits.max().item(), 'mean', logits.mean().item())

            # flatten predicted probs and targets for cross entropy loss
            B,T,C = logits.shape
            logits = logits.reshape(B*T,C)                  # B,T,vocab_size -> B*T,vocab_size
            decoder_targets = decoder_targets.reshape(-1)   # B,T -> B*T
            
            loss = f.cross_entropy(logits, decoder_targets)
            total_train_loss += loss.item()
            print(f"step: {step}, loss: {loss}")

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            # after each step, do a lr scheduler step
            lr_scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)
        print(f"Average train loss: {avg_train_loss}")

        # validation loop
        print(f"Estimating validation loss")
        model.eval()
        total_val_loss = 0.0
        for encoder_inputs, decoder_targets in val_loader:
            # encoder_input, decoder_targets = get_batch(dataset, cfg.seq_len, cfg.batch_size)
            encoder_input = encoder_inputs.to(device)

            # remove last token of targets to get decoder inputs
            decoder_input = decoder_targets[:, :-1].to(device)

            # remove first token for targets so the target seq is offset from input by 1
            decoder_targets = decoder_targets[:, 1:].to(device)
            if cfg.debug:
                print('decoder targets min', decoder_targets.min().item(), 'max', decoder_targets.max().item())

            logits = model(encoder_input, decoder_input)
            if cfg.debug:
                print('logits min', logits.min().item(), 'max', logits.max().item(), 'mean', logits.mean().item())

            # flatten predicted probs and targets for cross entropy loss
            B,T,C = logits.shape
            logits = logits.reshape(B*T,C)                  # B,T,vocab_size -> B*T,vocab_size
            decoder_targets = decoder_targets.reshape(-1)   # B,T -> B*T
            
            val_loss = f.cross_entropy(logits, decoder_targets)
            total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataset)      
        val_losses.append(avg_val_loss)
        print(f"Average validation loss: {avg_val_loss}")

        # save checkpoint if specified
        if cfg.save_checkpoint:
            save_checkpoint(cfg.save_checkpoint, epoch, cfg, model, optim)

    # plot learning curves if specified
    if cfg.plot_learning_curves:
        plot_learning_curves(train_losses, val_losses)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--epochs", type=int, default=100)
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
    argparser.add_argument("--debug", action="store_true", default=False)
    argparser.add_argument("--plot-learning-curves", action="store_true", default=False)
    args = argparser.parse_args()

    cfg = TrainingConfig(
        epochs=args.epochs,
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
        debug=args.debug,
        plot_learning_curves=args.plot_learning_curves
    )
    train(cfg)