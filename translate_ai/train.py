#!/usr/bin/env python3
import os
import torch
from torch import nn
from torch.nn import functional as f
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass
from argparse import ArgumentParser, Namespace
import tiktoken
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter


from transformer import TransformerTranslator
from datasets.multi_un import MultiUNDataset
from checkpoint import save_checkpoint, load_checkpoint
from plotting import plot_learning_curves
from scheduler import NoamLR

@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    warmup_steps: int
    num_layers: int
    embed_dim: int
    d_model: int
    ffwd_dim: int
    num_attention_heads: int
    eval_interval: int
    eval_iters: int
    checkpoint_interval: int
    batch_size: int
    dataset_file: str
    dataset_dir: str
    seq_len: int
    device: str
    mixed_precision: str
    save_checkpoint: str
    load_checkpoint: str
    debug: bool
    plot_learning_curves: bool
    log_dir: str

def train(cfg: TrainingConfig) -> None:
    # set up logdir
    if cfg.log_dir:
        os.makedirs(cfg.log_dir, exist_ok=True)

    # configure device
    device = torch.device(cfg.device)
    print("device: ", device)

    # initialize tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    vocab_size = tokenizer.n_vocab + 3 # +3 for BOS, EOS, PAD tokens
    pad_token = vocab_size - 1
    bos_token = vocab_size - 2
    eos_token = vocab_size - 3
    print(f"tokenizer: tiktoken cl100k_base, PAD: {pad_token}, BOS: {bos_token}, EOS: {eos_token}")
    print(f"vocab size: {vocab_size}")

    # initialize dataset
    dataset = MultiUNDataset(cfg.dataset_dir, 
                             tokenizer, 
                             max_length=cfg.seq_len, 
                             pad_token=pad_token, 
                             begin_text_token=bos_token,
                             end_text_token=eos_token)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"total dataset examples: {len(dataset)}")

    # initalize dataloaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # initialize model
    model = TransformerTranslator(
        input_vocab_size=vocab_size, 
        output_vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        d_model=cfg.d_model,
        num_encoder_layers=cfg.num_layers,
        num_decoder_layers=cfg.num_layers,
        num_attention_heads=cfg.num_attention_heads,
        ffwd_dim=cfg.ffwd_dim,
        max_seq_len=cfg.seq_len,
        max_output_tokens=128).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {total_params}")
    print(f"num layers: {cfg.num_layers}")

    # set up optimizer and learning rate scheduler 
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    lr_scheduler = NoamLR(optim, cfg.warmup_steps)

    # configure mixed precision training if specified
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    model = model.to(weight_dtype)

    model, optim, lr_scheduler, train_loader = accelerator.prepare(
        model, optim, lr_scheduler, train_loader 
    )

    # load checkpoint if specified
    curr_epoch = 0
    if cfg.load_checkpoint:
        checkpoint = load_checkpoint(cfg.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_epoch = checkpoint['epoch']

    # training loop
    writer = SummaryWriter(cfg.log_dir)
    train_losses, val_losses = [], []
    try:
        model.train()
        for epoch in range(curr_epoch, curr_epoch + cfg.epochs):
            for step, (encoded_inputs, encoded_targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
                # encoder_input, decoder_targets = get_batch(dataset, cfg.seq_len, cfg.batch_size)
                encoder_input = encoded_inputs.to(device)

                # remove last token of targets to get decoder inputs
                decoder_input = encoded_targets[:, :-1].to(device)

                # remove first token for targets so the target seq is offset from input by 1
                decoder_targets = encoded_targets[:, 1:].to(device)
                if cfg.debug:
                    print('decoder targets min', decoder_targets.min().item(), 'max', decoder_targets.max().item())

                # create padding masks to ensure model doesn't attend to padding tokens
                encoder_padding_mask = (encoder_input == pad_token).to(device) # (B,T)
                decoder_padding_mask = (decoder_input == pad_token).to(device) # (B,T)

                logits = model(encoder_input, decoder_input, encoder_padding_mask, decoder_padding_mask)

                if cfg.debug:
                    import pdb; pdb.set_trace()
                    print('logits min', logits.min().item(), 'max', logits.max().item(), 'mean', logits.mean().item())
                    print('logits argmax', torch.argmax(logits, dim=-1))

                # flatten predicted probs and targets for cross entropy loss
                B,T,C = logits.shape
                logits = logits.reshape(B*T,C)                  # B,T,vocab_size -> B*T,vocab_size
                decoder_targets = decoder_targets.reshape(-1)   # B,T -> B*T
                
                loss = f.cross_entropy(logits, decoder_targets, ignore_index=pad_token)
                if cfg.debug:
                    print(f"epoch: {epoch}, step: {step}, loss: {loss}")

                accelerator.backward(loss)
                optim.step()
                lr_scheduler.step()
                optim.zero_grad(set_to_none=True)

                # estimate loss periodically
                is_eval_step = step % cfg.eval_interval == 0
                if is_eval_step:
                    print("Estimating loss")
                    model.eval()
                    avg_train_loss = estimate_loss(model, train_loader, ignore_index=pad_token)
                    avg_val_loss = estimate_loss(model, val_loader, ignore_index=pad_token)
                    model.train()

                    # log to stdout
                    train_losses.append(avg_train_loss)
                    val_losses.append(avg_val_loss)
                    print(f"Train loss: {avg_train_loss}")
                    print(f"Validation loss: {avg_val_loss}")

                    # log to tensorboard
                    writer.add_scalars('loss', {
                            'training': avg_train_loss.item(),
                            'validation': avg_val_loss.item(),
                        }, epoch + step)

                # save checkpoint if specified
                is_checkpoint_step = step > 0 and step % cfg.checkpoint_interval == 0
                if cfg.save_checkpoint and is_checkpoint_step:
                    save_checkpoint(cfg.save_checkpoint, epoch, cfg, model, optim)

    # catch ctrl+C to allow us to interrupt training early and plot learning curves,
    # for faster development iteration loop.
    except KeyboardInterrupt:
        pass

    # plot learning curves if specified
    if cfg.plot_learning_curves:
        plot_learning_curves(train_losses, val_losses)

@torch.no_grad()
def estimate_loss(model: nn.Module, 
                  dataloader: torch.utils.data.DataLoader,
                  ignore_index: int = 0,
                  eval_iters: int = 10):
    """Returns the average loss after `eval_iters` iterations using the given model and dataloader."""
    device = next(model.parameters()).device
    losses = torch.zeros(eval_iters)
    for i, (encoder_inputs, decoder_targets) in enumerate(dataloader):
        if i == eval_iters:
            break
        
        encoder_input = encoder_inputs.to(device)

        # remove last token of targets to get decoder inputs
        decoder_input = decoder_targets[:, :-1].to(device)

        # remove first token for targets so the target seq is offset from input by 1
        decoder_targets = decoder_targets[:, 1:].to(device)

        if cfg.debug:
            print('decoder targets min', decoder_targets.min().item(), 'max', decoder_targets.max().item())

        # create padding masks to ensure model doesn't attend to padding tokens
        encoder_padding_mask = (encoder_input == ignore_index).to(device) # (B,T)
        decoder_padding_mask = (decoder_input == ignore_index).to(device) # (B,T)

        logits = model(encoder_input, decoder_input, encoder_padding_mask, decoder_padding_mask)

        if cfg.debug:
            print('logits min', logits.min().item(), 'max', logits.max().item(), 'mean', logits.mean().item())

        # flatten predicted probs and targets for cross entropy loss
        B,T,C = logits.shape
        logits = logits.reshape(B*T,C)                  # B,T,vocab_size -> B*T,vocab_size
        decoder_targets = decoder_targets.reshape(-1)   # B,T -> B*T
        
        loss = f.cross_entropy(logits, decoder_targets, ignore_index=ignore_index)
        losses[i] = loss
    return losses.mean()

def validate_args(args: Namespace) -> None:
    if not args.dataset_file and not args.dataset_dir:
        raise ValueError("--dataset-dir or --dataset-file must be specified")
    if args.mixed_precision and args.mixed_precision not in {"fp16","bf16"}:
        raise ValueError(f"unsupported mixed precision data type '{args.mixed_precision}' - must be one of: 'fp16', 'bf16'")

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--learning-rate", type=float, default=1e-3)
    argparser.add_argument("--warmup-steps", type=int, default=100)
    argparser.add_argument("--num-layers", type=int, default=6)
    argparser.add_argument("--embed-dim", type=int, default=128)
    argparser.add_argument("--ffwd-dim", type=int, default=512)
    argparser.add_argument("--d-model", type=int, default=128)
    argparser.add_argument("--num-attention-heads", type=int, default=2)
    argparser.add_argument("--eval-interval", type=int, default=100)
    argparser.add_argument("--eval-iters", type=int, default=10)
    argparser.add_argument("--checkpoint-interval", type=int, default=100)  
    argparser.add_argument("--batch-size", type=int, default=32) 
    argparser.add_argument("--dataset-file", type=str)
    argparser.add_argument("--dataset-dir", type=str)
    argparser.add_argument("--seq-len", type=int, default=128)
    argparser.add_argument("--device", type=str, default="cpu")
    argparser.add_argument("--load-checkpoint", type=str)
    argparser.add_argument("--save-checkpoint", type=str)
    argparser.add_argument("--debug", action="store_true", default=False)
    argparser.add_argument("--plot-learning-curves", action="store_true", default=False)
    argparser.add_argument("--mixed-precision", required=False, help="fp16, bfloat16")
    argparser.add_argument("--log-dir", type=str)
    args = argparser.parse_args()

    validate_args(args)

    cfg = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_layers=args.num_layers,
        embed_dim=args.embed_dim,
        ffwd_dim=args.ffwd_dim,
        d_model=args.d_model,
        num_attention_heads=args.num_attention_heads,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        checkpoint_interval=args.checkpoint_interval,
        batch_size=args.batch_size,
        dataset_file=args.dataset_file,
        dataset_dir=args.dataset_dir,
        seq_len=args.seq_len,
        device=args.device,
        mixed_precision=args.mixed_precision,
        load_checkpoint=args.load_checkpoint,
        save_checkpoint=args.save_checkpoint,
        debug=args.debug,
        plot_learning_curves=args.plot_learning_curves,
        log_dir=args.log_dir,
    )
    train(cfg)