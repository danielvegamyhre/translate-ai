#!/usr/bin/env python3
import torch
from torch import nn


def compute_flops(model: nn.Module, input_tensor: torch.Tensor) -> int:
    # Forward pass with hooks to count FLOPs
    flops = 0

    def count_flops(module, input, output):
        if isinstance(module, nn.Linear):
            batch_size = input[0].size(0)
            flops_per_instance = 2 * module.in_features * module.out_features
            flops_per_layer = batch_size * flops_per_instance
            nonlocal flops
            flops += flops_per_layer

    hooks = []
    for layer in model.modules():
        hooks.append(layer.register_forward_hook(count_flops))

    # Forward pass to trigger hooks
    with torch.no_grad():
        model(input_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return flops