#!/usr/bin/env python3
from config import TrainingConfig

def estimate_mfu(cfg: TrainingConfig,
                 num_encoder_params: int,
                 num_decoder_params: int,
                 steps_per_second: float,
                 hardware_peak_flops_per_sec: float) -> float:
    '''
    MFU (Model Flops Utilization) can be estimated as:

    MFU = (actual FLOPs per second when trainng) / (theoretical peak FLOPs per second of hardware)

    Args:
        steps_per_second: training steps per second measured
        hardware_peak_flops_per_sec: check your GPU documentation for this value
    '''
    # total flops for one forward and backward pass can be roughly estimated as 2x flops for forward pass
    flops_per_step = 6 * (num_encoder_params * cfg.seq_len + num_decoder_params * cfg.max_output_tokens)
    actual_flops_per_second = flops_per_step * steps_per_second
    mfu = actual_flops_per_second / hardware_peak_flops_per_sec
    return mfu

