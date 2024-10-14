#!/usr/bin/env python3
import torch
from torch import nn
import torch.distributed as dist
from fvcore.nn import FlopCountAnalysis, parameter_count

from config import TrainingConfig

def estimate_mfu(cfg: TrainingConfig,
                 model: nn.Module,
                 inputs: tuple,
                 steps_per_second: float,
                 hardware_peak_flops_per_sec: float) -> float:
    '''
    MFU (Model Flops Utilization) can be estimated as:

    MFU = (actual FLOPs per second when trainng) / (theoretical peak FLOPs per second of hardware)

    Args:
        model: torch nn.Module
        input_shape: batch dimension must be first at index 0
        steps_per_second: training steps per second measured
        hardware_peak_flops_per_sec: check your GPU documentation for this value
    '''
    # adjust batch size based on world size
    # batch_size = input_shape[0]
    # world_size = dist.get_world_size() if cfg.distributed else 1
    # effective_batch_size = batch_size // world_size
    # new_input_shape = (effective_batch_size, *input_shape[1:])

    # input_tensor = torch.rand(new_input_shape)
    flop_counter = FlopCountAnalysis(model, inputs)
    flops_per_fwd_pass = flop_counter.total()

    # total flops for one forward and backward pass can be roughly estimated as 2x flops for forward pass
    flops_fwd_bkwd = 2 * flops_per_fwd_pass 
    actual_flops_per_second = flops_fwd_bkwd * steps_per_second
    mfu = actual_flops_per_second / hardware_peak_flops_per_sec
    return mfu

