# Translate AI

Making encoder-decoder transfomers cool again.

English to Spanish translation model with custom [Datasets](https://github.com/danielvegamyhre/translate-ai/tree/main/translate_ai/datasets) for:

- [English to Spanish Dataset](https://www.kaggle.com/datasets/lonnieqin/englishspanish-translation-dataset/data)
- [MultiUN dataset](https://opus.nlpl.eu/legacy/MultiUN.php) 

### Feature overview 
- Encoder decoder transformer architecture closely following the [research paper](https://arxiv.org/pdf/1706.03762) from Google.
- Easily configurable acceleration (mixed precision training, device type)
- Easily configurable model (number of layers, number of attention heads embed dim, hidden dim, feedforward dim)
- Observability
    - [Tensorboard](https://www.tensorflow.org/tensorboard) support
    - [Weights and Biases](https://wandb.ai/site/) support
- [Performance analysis instrumentation](https://github.com/danielvegamyhre/translate-ai/blob/main/dist-perf-analysis.sh) to estimate MFU your model + training config is getting given your hardware specs
    - MFU estimation supports both single-GPU and distributed training
- Supports distributed training (multi-GPU and multi-node)
- Auto-checkpointing support
- Fast BPE tokenization
- [Noam learning rate scheduler](https://nn.labml.ai/optimizers/noam.html) as per the paper


### Setup

Install requirements:
```
python3 -m pip install -r requirements.txt
```

### Performance Analysis

- `perf-analysis.sh` contains a script template estimate MFU, which you can modify with your own batch size, model dims, hardware specs, etc.
- `dist-perf-analysis.sh` contains a similar script template for estimating MFU for distributed training.

### Training

- `train.sh` includes command template for kicking off a simple training run using accelerate.
- `dist-train.sh` contains a similar command template for running distributed training.

Example:
```
python3 translate_ai/train.py \
    --dataset-file data/english-spanish.csv \
    --device cuda \
    --epochs 10 \
    --learning-rate .001 \
    --batch-size 32 \
    --num-layers 2 \
    --embed-dim 128 \
    --d-model 512 \
    --ffwd-dim 1024 \
    --seq-len 128 \
    --max-output-tokens 128 \
    --eval-interval 100 \
    --eval-iters 10 \
    --checkpoint-interval 100 \
    --save-checkpoint checkpoints/chkpt.pt \
    --wandb-project dvm \
    --wandb-api-key ${WANDB_API_KEY} \
    --distributed
  ```

### Inference

Example: 

```
python3 translate_ai/translate.py --english-query "The cat is blue." --checkpoint-file checkpoints/chkpt.pt
...
"El gato es azul."
```