# Translate AI

Making encoder-decoder transfomers cool again.

English to Spanish translation model with custom [Datasets](https://github.com/danielvegamyhre/translate-ai/tree/main/translate_ai/datasets) for:

- [English to Spanish Dataset](https://www.kaggle.com/datasets/lonnieqin/englishspanish-translation-dataset/data)
- [MultiUN dataset](https://opus.nlpl.eu/legacy/MultiUN.php) 

### Feature overview 
- Encoder decoder transformer architecture closely following the [research paper](https://arxiv.org/pdf/1706.03762) from Google.
- Fast BPE tokenization
- [Noam learning rate scheduler](https://nn.labml.ai/optimizers/noam.html) as per the paper
- Configurable hyperparams and model dimensions
- Configurable acceleration (mixed precision training, device type)
    - Support for [accelerate](https://huggingface.co/docs/accelerate/en/index)
- Auto-checkpointing support
- Observability
    - [Tensorboard](https://www.tensorflow.org/tensorboard) support
    - [Weights and Biases](https://wandb.ai/site/) support

### Quickstart

Install requirements:
```
python3 -m pip install -r requirements.txt
```

Kick off training run


### Training

`train.sh` includes generic template for kicking off a simple training run using accelerate.

Example:
```
accelerate launch translate_ai/train.py \
    --dataset-file data/english-spanish.csv \
    --device mps \
    --epochs 1 \
    --learning-rate .001 \
    --batch-size 2 \
    --num-layers 2 \
    --embed-dim 128 \
    --d-model 128 \
    --ffwd-dim 512 \
    --seq-len 128 \
    --max-output-tokens 128 \
    --eval-interval 100 \
    --eval-iters 10 \
    --checkpoint-interval 100 \
    --save-checkpoint checkpoints/chkpt.pt \
    --wandb-project dvm
  ```

### Inference

Example: 

```
python3 translate_ai/translate.py --english-query "The cat is blue." --checkpoint-file checkpoints/chkpt.pt
...
"El gato es azul."
```