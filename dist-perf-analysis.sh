#!/bin/bash
torchrun --nproc_per_node=2 --master_port=12345 translate_ai/train.py \
    --dataset-file data/english-spanish.csv \
    --device cuda \
    --mixed-precision bf16 \
    --learning-rate .001 \
    --batch-size 32 \
    --num-layers 2 \
    --embed-dim 128 \
    --d-model 128 \
    --ffwd-dim 512 \
    --seq-len 128 \
    --max-output-tokens 128 \
    --hardware-peak-tflops 19.05 \
    --distributed \
    --estimate-mfu
