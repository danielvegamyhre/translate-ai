#!/bin/bash
#!/bin/bash
python3 train.py \
    --dataset-file=data/data.csv \
    --device=cuda \
    --epochs 1 \
    --learning-rate .001 \
    --save-checkpoint checkpoints/chkpt.pt \
    --eval-interval 100 \
    --checkpoint-interval 100 \
    --batch-size 64 \
    --plot-learning-curves