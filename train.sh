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
    --max-output-tokens 128 \
    --eval-interval 100 \
    --eval-iters 10 \
    --checkpoint-interval 100 \
    --save-checkpoint checkpoints/chkpt.pt
