#!/bin/bash
accelerate launch train.py \
    --dataset-file=data/data.csv \
    --device=mps \
    --epochs 1 \
    --learning-rate .001 \
    --save-checkpoint checkpoints/chkpt.pt \
    --eval-interval 100 \
    --checkpoint-interval 100 \
    --batch-size 2 \
    --plot-learning-curves