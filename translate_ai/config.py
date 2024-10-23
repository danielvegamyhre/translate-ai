#!/usr/bin/env python3
import os
from dataclasses import dataclass

# env vars automatically set by torchrun for distributed training
LOCAL_RANK = "LOCAL_RANK"
RANK = "RANK"
WORLD_SIZE = "WORLD_SIZE"
LOCAL_WORLD_SIZE = "LOCAL_WORLD_SIZE"
MASTER_ADDR = "MASTER_ADDR"
MASTER_PORT = "MASTER_PORT"

SUPPORTED_MIXED_PRECISION_DTYPES = {"fp16","bf16"}

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
    output_seq_len: int

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
    distributed: bool

    # performance analysis
    estimate_mfu: bool
    hardware_peak_tflops: float

    # observability and debugging
    tensorboard_log_dir: str
    wandb_project: str
    wandb_api_key: str
    plot_learning_curves: bool
    debug: bool


def _get_dist_configs():
    local_rank = int(os.environ.get(LOCAL_RANK))
    global_rank = int(os.environ.get(RANK))
    world_size = int(os.environ.get(WORLD_SIZE))
    local_world_size = int(os.environ.get(LOCAL_WORLD_SIZE))
    master_addr = os.environ.get(MASTER_ADDR)
    master_port = os.environ.get(MASTER_PORT)
    return {
        LOCAL_RANK: local_rank,
        RANK: global_rank,
        WORLD_SIZE: world_size,
        LOCAL_WORLD_SIZE: local_world_size,
        MASTER_ADDR: master_addr,
        MASTER_PORT: master_port,
    }