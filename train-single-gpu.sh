#!/bin/bash
python3 translate_ai/train.py \
    --dataset-file data/english-spanish.csv \
    --device mps \
    --epochs 1 \
    --learning-rate .001 \
    --batch-size 2 \
    --num-layers 2 \
    --embed-dim 128 \
    --d-model 256 \
    --ffwd-dim 512 \
    --seq-len 128 \
    --output-seq-len 128 \
    --eval-interval 100 \
    --eval-iters 10 \
    --checkpoint-interval 100 \
    --save-checkpoint checkpoints/chkpt.pt \
    --wandb-project dvm \
    --wandb-api-key ${WANDB_API_KEY}
