#!/usr/bin/env python3
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # hyperparams
    epochs: int
    learning_rate: float
    warmup_steps: int
    batch_size: int

    # accelerators
    device: str
    mixed_precision: str

    # model dims
    num_layers: int
    embed_dim: int
    d_model: int
    ffwd_dim: int
    num_attention_heads: int
    seq_len: int
    max_output_tokens: int

    # eval configs
    eval_interval: int
    eval_iters: int

    # checkpointing
    save_checkpoint: str
    load_checkpoint: str
    checkpoint_interval: int

    # dataset configs
    dataset_file: str
    dataset_dir: str

    # distributed training
    multi_gpu: bool
    multi_node: bool
    world_size: int
    dist_url: str
    rank: int

    # observability and debugging
    tensorboard_log_dir: str
    wandb_project: str
    plot_learning_curves: bool
    debug: bool