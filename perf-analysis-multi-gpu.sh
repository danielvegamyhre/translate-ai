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
    --output-seq-len 128 \
    --hardware-peak-tflops ${HARDWARE_PEAK_TFLOPS} \
    --distributed \
    --estimate-mfu
