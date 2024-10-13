#!/bin/bash
accelerate launch translate_ai/train.py \
    --dataset-file=data/english-spanish.csv \
    --device=mps \
    --epochs 10 \
    --learning-rate .001 \
    --save-checkpoint checkpoints/chkpt.pt \
    --eval-interval 100 \
    --checkpoint-interval 100 \
    --batch-size 2 \
    --plot-learning-curves