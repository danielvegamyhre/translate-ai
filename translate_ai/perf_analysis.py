#!/usr/bin/env python3
import torch
from torch import nn
from fvcore.nn import FlopCountAnalysis, parameter_count

def estimate_mfu(model: nn.Module, 
                 input_shape: tuple,
                 steps_per_second: float,
                 hardware_peak_flops_per_sec: float) -> float:
    '''
    MFU (Model Flops Utilization) can be estimated as:

    MFU = (actual FLOPs per second when trainng) / (theoretical peak FLOPs per second of hardware)
    '''
    input_tensor = torch.tensor(input_shape)
    flop_counter = FlopCountAnalysis(model, input_tensor)
    flops_per_fwd_pass = flop_counter.total()
    # total flops for one forward and backward pass can be roughly estimated as 2x flops for forward pass
    flops_fwd_bkwd = 2 * flops_per_fwd_pass 
    actual_flops_per_second = flops_fwd_bkwd * steps_per_second
    mfu = actual_flops_per_second / hardware_peak_flops_per_sec
    return mfu

