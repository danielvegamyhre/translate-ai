#!/usr/bin/env python3
import os
from datetime import datetime
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from time import perf_counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.nn import functional as f
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.elastic.multiprocessing.errors import record

import tiktoken
import wandb
from fvcore.nn import parameter_count

from transformer import TransformerTranslator
from datasets.english_spanish import EnglishToSpanishDataset
from checkpoint import save_checkpoint, load_checkpoint
from plotting import plot_learning_curves
from scheduler import NoamLR
from perf_analysis import estimate_mfu
from utils import log
from config import (
    TrainingConfig,
    SUPPORTED_MIXED_PRECISION_DTYPES,
    _get_dist_configs
)

HARDWARE_PEAK_FLOPS_PER_SECOND = 149e12 # NVIDIA A40

@record
def train(cfg: TrainingConfig) -> None:
    device = torch.device(cfg.device)

    # set up distributed training
    local_rank = 0
    if cfg.distributed:
        _setup_distributed_training()
        local_rank = dist.get_rank()

    # initialize tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    vocab_size = tokenizer.n_vocab + 3 # +3 for BOS, EOS, PAD tokens
    pad_token = vocab_size - 1
    bos_token = vocab_size - 2
    eos_token = vocab_size - 3
    log(f"tokenizer: tiktoken cl100k_base, PAD: {pad_token}, BOS: {bos_token}, EOS: {eos_token}", local_rank)
    log(f"vocab size: {vocab_size}", local_rank)

    # initialize dataset
    dataset = EnglishToSpanishDataset(cfg.dataset_file,
                                      tokenizer,
                                      pad_token,
                                      bos_token,
                                      eos_token)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    log(f"total dataset examples: {len(dataset)}", local_rank)

    # initalize dataloaders
    train_sampler, val_sampler = None, None
    shuffle = True
    if cfg.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=shuffle, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=shuffle, sampler=val_sampler)

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
        max_output_tokens=cfg.max_output_tokens).to(device)

    if cfg.distributed:
        # wrap model in DDP and configure the device the code will be operating
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # get model param count
    param_counts = parameter_count(model)
    total_params_key = '' # https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.nn.parameter_count
    total_params = param_counts[total_params_key]
    log(f"total model parameters: {total_params}", local_rank)
    log(f"encoder parameters: {param_counts['encoder']}")
    log(f"decoder parameters: {param_counts['decoder']}") 

    # set up optimizer and learning rate scheduler 
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    lr_scheduler = NoamLR(optim, cfg.warmup_steps)

    # configure mixed precision training
    weight_dtype = torch.float32
    if cfg.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif cfg.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    model = model.to(weight_dtype)

    # load checkpoint
    curr_epoch = 0
    if cfg.load_checkpoint:
        checkpoint = load_checkpoint(cfg.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_epoch = checkpoint['epoch']

    # configure wandb
    if cfg.wandb_project:
        log(f"initializing wandb project {cfg.wandb_project}", local_rank)
        wandb.login(key=cfg.wandb_api_key)
        wandb.init(project=cfg.wandb_project, 
                   config={
                       "learning_rate": cfg.learning_rate,
                       "epochs": cfg.epochs,
                       "batch_size": cfg.batch_size,
                       "num_layers:": cfg.num_layers,
                       "num_attention_heads": cfg.num_attention_heads,
                       "embed_dim": cfg.embed_dim,
                       "d_model": cfg.d_model,
                       "total_params": total_params})
        wandb.watch(model, log="all")

    # configure tensorboard
    if cfg.tensorboard_log_dir:
        os.makedirs(cfg.tensorboard_log_dir, exist_ok=True)
        writer = SummaryWriter(f"runs/{datetime.now()}")

    # training loop
    train_losses, val_losses = [], []
    try:
        model.train()
        for epoch in range(curr_epoch, curr_epoch + cfg.epochs):
            # make sure data shuffling it different between epochs for distributed sampler
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            start_time = perf_counter()    
            for step, (encoded_inputs, encoded_targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
                encoder_input = encoded_inputs.to(device)

                # remove last token of targets to get decoder inputs
                decoder_input = encoded_targets[:, :-1].to(device)

                # remove first token for targets so the target seq is offset from input by 1
                decoder_targets = encoded_targets[:, 1:].to(device)
                if cfg.debug:
                    log(f'decoder targets min: {decoder_targets.min().item()}, max {decoder_targets.max().item()}', local_rank)

                # create padding masks to ensure model doesn't attend to padding tokens
                encoder_padding_mask = (encoder_input == pad_token).to(device) # (B,T)
                decoder_padding_mask = (decoder_input == pad_token).to(device) # (B,T)

                logits = model(encoder_input, decoder_input, encoder_padding_mask, decoder_padding_mask)

                if cfg.debug:
                    import pdb; pdb.set_trace()
                    log(f'logits min: {logits.min().item()}, max: {logits.max().item()}, mean: {logits.mean().item()}', local_rank)
                    log(f'logits argmax: {torch.argmax(logits, dim=-1)}', local_rank)

                # flatten predicted probs and targets for cross entropy loss
                B,T,C = logits.shape
                logits = logits.reshape(B*T,C)                  # B,T,vocab_size -> B*T,vocab_size
                decoder_targets = decoder_targets.reshape(-1)   # B,T -> B*T
                
                loss = f.cross_entropy(logits, decoder_targets, ignore_index=pad_token)
                if cfg.debug:
                    log(f"epoch: {epoch}, step: {step}, loss: {loss}", local_rank)

                loss.backward()
                optim.step()
                lr_scheduler.step()
                optim.zero_grad(set_to_none=True)

                # estimate loss periodically
                is_eval_step = step % cfg.eval_interval == 0
                if is_eval_step:
                    log("Estimating loss", local_rank)
                    model.eval()
                    avg_train_loss = estimate_loss(model, train_loader, ignore_index=pad_token)
                    avg_val_loss = estimate_loss(model, val_loader, ignore_index=pad_token)
                    model.train()

                    # log to stdout
                    train_losses.append(avg_train_loss)
                    val_losses.append(avg_val_loss)
                    log(f"Train loss: {avg_train_loss}", local_rank)
                    log(f"Validation loss: {avg_val_loss}", local_rank)

                    # log to tensorboard
                    if cfg.tensorboard_log_dir:
                        global_step = epoch + step
                        scalars = {'training':avg_train_loss.item(), 'validation': avg_val_loss.item()}
                        writer.add_scalars('loss', scalars, global_step)

                    # log to wandb
                    if cfg.wandb_project:
                        wandb.log({
                            'train_loss':  avg_train_loss.item(),
                            'val_loss':  avg_val_loss.item(),
                        })

                # save checkpoint if specified
                is_checkpoint_step = step > 0 and step % cfg.checkpoint_interval == 0
                if cfg.save_checkpoint and is_checkpoint_step:
                    save_checkpoint(cfg, epoch, model, optim)
                
            # estimate MFU after each epoch
            epoch_duration = perf_counter() - start_time
            steps_per_epoch = len(train_loader)
            steps_per_second = epoch_duration / steps_per_epoch

            mfu = estimate_mfu(cfg, param_counts['encoder'], param_counts['decoder'], steps_per_second, HARDWARE_PEAK_FLOPS_PER_SECOND)
            mfu_pct = mfu * 100
            log(f"Esimated MFU: {mfu_pct:.4f}%", local_rank)

    # catch ctrl+C to allow us to interrupt training early and plot learning curves,
    # for faster development iteration loop.
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.save_checkpoint:
            save_checkpoint(cfg, epoch, model, optim)
        if cfg.plot_learning_curves:
            plot_learning_curves(train_losses, val_losses)
        if cfg.distributed:
            _dist_cleanup()


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

        # create padding masks to ensure model doesn't attend to padding tokens
        encoder_padding_mask = (encoder_input == ignore_index).to(device) # (B,T)
        decoder_padding_mask = (decoder_input == ignore_index).to(device) # (B,T)

        logits = model(encoder_input, decoder_input, encoder_padding_mask, decoder_padding_mask)

        # flatten predicted probs and targets for cross entropy loss
        B,T,C = logits.shape
        logits = logits.reshape(B*T,C)                  # B,T,vocab_size -> B*T,vocab_size
        decoder_targets = decoder_targets.reshape(-1)   # B,T -> B*T
        
        loss = f.cross_entropy(logits, decoder_targets, ignore_index=ignore_index)
        losses[i] = loss
    return losses.mean()


def _validate_args(args: Namespace) -> None:
    if not args.dataset_file and not args.dataset_dir:
        raise ValueError("--dataset-dir or --dataset-file must be specified")
    if args.mixed_precision and args.mixed_precision not in SUPPORTED_MIXED_PRECISION_DTYPES:
        raise ValueError(f"unsupported mixed precision data type '{args.mixed_precision}' - must be one of: {SUPPORTED_MIXED_PRECISION_DTYPES}")
    if args.distributed:
        dist_configs: dict = _get_dist_configs()
        for env_var, value in dist_configs.items():
            if value is None:
                raise ValueError(f"environment variable {env_var} must be set for torch distributed training")

def _init_debug_config():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1" 
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

def _setup_distributed_training():
    # initialize the process group (NCCL backend is for GPUs, GLOO for CPUs)
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, init_method='env://')

    # ensure one process per GPU
    torch.cuda.set_device(dist.get_rank())

def _dist_cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    argparser = ArgumentParser()
    # hyperparams
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--learning-rate", type=float, default=1e-3)
    argparser.add_argument("--warmup-steps", type=int, default=100)
    argparser.add_argument("--batch-size", type=int, default=32) 

    # acceleration
    argparser.add_argument("--mixed-precision", help="fp16, bfloat16")
    argparser.add_argument("--device", type=str, default="cpu")

    # model dims
    argparser.add_argument("--num-layers", type=int, default=6)
    argparser.add_argument("--embed-dim", type=int, default=128)
    argparser.add_argument("--ffwd-dim", type=int, default=512)
    argparser.add_argument("--d-model", type=int, default=128)
    argparser.add_argument("--num-attention-heads", type=int, default=2)
    argparser.add_argument("--seq-len", type=int, default=128)
    argparser.add_argument("--max-output-tokens", type=int, default=128)

    # evaluation
    argparser.add_argument("--eval-interval", type=int, default=100)
    argparser.add_argument("--eval-iters", type=int, default=10)

    # checkpointing
    argparser.add_argument("--checkpoint-interval", type=int, default=100)  
    argparser.add_argument("--load-checkpoint", type=str)
    argparser.add_argument("--save-checkpoint", type=str)

    # dataset
    argparser.add_argument("--dataset-file", type=str)
    argparser.add_argument("--dataset-dir", type=str)

    # observability
    argparser.add_argument("--tensorboard-log-dir", type=str)
    argparser.add_argument("--wandb-project", type=str)
    argparser.add_argument("--wandb-api-key", type=str)
    argparser.add_argument("--plot-learning-curves", action="store_true", default=False)
    argparser.add_argument("--debug", action="store_true", default=False)

    # distributed training
    argparser.add_argument('--distributed', action='store_true', help="multi-GPU or multi-node training with distributed data parallelism")

    args = argparser.parse_args()

    _validate_args(args)
    if args.debug:
        _init_debug_config()

    cfg = TrainingConfig(
        # hyperparams
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,

        # acceleration
        device=args.device,
        mixed_precision=args.mixed_precision,

        # model dims
        num_layers=args.num_layers,
        embed_dim=args.embed_dim,
        ffwd_dim=args.ffwd_dim,
        d_model=args.d_model,
        num_attention_heads=args.num_attention_heads,
        seq_len=args.seq_len,
        max_output_tokens=args.max_output_tokens,

        # evaluation
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,

        # checkpointing
        checkpoint_interval=args.checkpoint_interval,
        load_checkpoint=args.load_checkpoint,
        save_checkpoint=args.save_checkpoint,

        # dataset
        dataset_file=args.dataset_file,
        dataset_dir=args.dataset_dir,

        # distributed training
        distributed=args.distributed,

        # observability and debugging
        plot_learning_curves=args.plot_learning_curves,
        tensorboard_log_dir=args.tensorboard_log_dir,
        wandb_project=args.wandb_project,
        wandb_api_key=args.wandb_api_key,
        debug=args.debug,
    )
    train(cfg)