#!/usr/bin/env python3
from time import perf_counter
from torch import nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from fvcore.nn import parameter_count

from config import TrainingConfig
from utils import log

def estimate_mfu(cfg: TrainingConfig,
                 num_encoder_params: int,
                 num_decoder_params: int,
                 steps_per_second: float) -> float:
    '''
    MFU (Model Flops Utilization) can be estimated as:

    MFU = (actual FLOPs per second when trainng) / (theoretical peak FLOPs per second of hardware)

    Args:
        steps_per_second: training steps per second measured
        hardware_peak_flops_per_sec: check your GPU documentation for this value
    '''
    hardware_peak_flops = float(cfg.hardware_peak_tflops) * 10**12
    flops_per_step = 6 * (num_encoder_params * cfg.seq_len + num_decoder_params * cfg.max_output_tokens) * cfg.batch_size
    actual_flops_per_second = flops_per_step * steps_per_second
    mfu = actual_flops_per_second / hardware_peak_flops
    log(f"flops per step: {flops_per_step:.4e}")
    log(f"steps per second: {steps_per_second:.2f}")
    log(f"actual flops per second: {actual_flops_per_second:.4e}")
    log(f"hardware peak flops per second: {hardware_peak_flops:.4e}")
    return mfu

def run_perf_analysis(model: nn.Module, 
                      cfg: TrainingConfig,
                      dataloader: DataLoader, 
                      optimizer: Optimizer,
                      lr_scheduler: LRScheduler,
                      warmup_steps: int = 10, 
                      test_steps: int = 100,
                      device: str = "cpu",
                      pad_token: int = 0) -> None:
    log(f"running performance analysis with {warmup_steps} warmup steps and {test_steps} test steps")

    # get param counts for MFU calculation
    param_counts = parameter_count(model)
    log(f"encoder parameters: {param_counts['encoder']}")
    log(f"decoder parameters: {param_counts['decoder']}") 

    # run test
    for step, (encoded_inputs, encoded_targets) in enumerate(dataloader):
        # start timer after warmup steps are completed
        if step == warmup_steps:
            start_time = perf_counter()
        elif step == warmup_steps + test_steps:
            break

        encoder_input = encoded_inputs.to(device)

        # remove last token of targets to get decoder inputs
        decoder_input = encoded_targets[:, :-1].to(device)

        # remove first token for targets so the target seq is offset from input by 1
        decoder_targets = encoded_targets[:, 1:].to(device)

        # create padding masks to ensure model doesn't attend to padding tokens
        encoder_padding_mask = (encoder_input == pad_token).to(device) # (B,T)
        decoder_padding_mask = (decoder_input == pad_token).to(device) # (B,T)

        logits = model(encoder_input, decoder_input, encoder_padding_mask, decoder_padding_mask)

        # flatten predicted probs and targets for cross entropy loss
        B,T,C = logits.shape
        logits = logits.reshape(B*T,C)                  # B,T,vocab_size -> B*T,vocab_size
        decoder_targets = decoder_targets.reshape(-1)   # B,T -> B*T
        
        loss = f.cross_entropy(logits, decoder_targets, ignore_index=pad_token)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    test_duration = perf_counter() - start_time
    steps_per_second = test_steps / test_duration 
    log(f"test duration: {test_duration:.2f} seconds")
    log(f"steps per second: {steps_per_second:.2f}")

    mfu = estimate_mfu(cfg, param_counts['encoder'], param_counts['decoder'], steps_per_second)
    mfu_pct = mfu * 100
    log(f"Esimated MFU: {mfu_pct:.2f}%")
