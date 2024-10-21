#!/bin/bash
torchrun --nproc_per_node=2 --master_port=12345 \
    translate_ai/train.py \
    --dataset-file data/english-spanish.csv \
    --device cuda \
    --mixed-precision bf16 \
    --epochs 10 \
    --learning-rate .004 \
    --batch-size 128 \
    --num-layers 2 \
    --embed-dim 128 \
    --d-model 128 \
    --ffwd-dim 512 \
    --seq-len 128 \
    --max-output-tokens 128 \
    --eval-interval 200 \
    --eval-iters 10 \
    --checkpoint-interval 200 \
    --save-checkpoint checkpoints/chkpt.pt \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-api-key ${WANDB_API_KEY} \
    --distributed
