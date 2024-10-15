#!/bin/bash
python3 translate_ai/train.py \
    --dataset-file data/english-spanish.csv \
    --device mps \
    --learning-rate .001 \
    --batch-size 2 \
    --num-layers 2 \
    --embed-dim 128 \
    --d-model 128 \
    --ffwd-dim 512 \
    --seq-len 128 \
    --max-output-tokens 128 \
    --hardware-peak-tflops 1.0 \
    --estimate-mfu

