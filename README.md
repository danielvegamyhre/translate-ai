# Translate AI

Making encoder-decoder transfomers cool again.

English to Spanish translation model with custom [Datasets](https://github.com/danielvegamyhre/translate-ai/tree/main/translate_ai/datasets) for:

- [English to Spanish Dataset](https://www.kaggle.com/datasets/lonnieqin/englishspanish-translation-dataset/data)
- [MultiUN dataset](https://opus.nlpl.eu/legacy/MultiUN.php) 

### Feature overview 
- Encoder decoder transformer architecture closely following the [research paper](https://arxiv.org/pdf/1706.03762) from Google.
- Fast BPE tokenization via [tiktoken](https://github.com/openai/tiktoken)
- [Noam learning rate scheduler](https://nn.labml.ai/optimizers/noam.html)
- Configurable hyperparams (batch size, epochs, learning rate, warmup steps)
- Configurable model dimensions (number of layers, embed dim, d-model, feedforward dim, number of attention heads, sequence length, max output sequence length)
- Configurable acceleration (mixed precision training, device type)
    - Support for [accelerate](https://huggingface.co/docs/accelerate/en/index)
- Auto-checkpointing support
- Observability
    - [Tensorboard](https://www.tensorflow.org/tensorboard) support
    - [Weights and Biases](https://wandb.ai/site/) support

### Usage
```
usage: train.py [-h] [--epochs EPOCHS] [--learning-rate LEARNING_RATE] [--warmup-steps WARMUP_STEPS] [--batch-size BATCH_SIZE]
                [--mixed-precision MIXED_PRECISION] [--device DEVICE] [--num-layers NUM_LAYERS] [--embed-dim EMBED_DIM]
                [--ffwd-dim FFWD_DIM] [--d-model D_MODEL] [--num-attention-heads NUM_ATTENTION_HEADS] [--seq-len SEQ_LEN]
                [--max-output-tokens MAX_OUTPUT_TOKENS] [--eval-interval EVAL_INTERVAL] [--eval-iters EVAL_ITERS]
                [--checkpoint-interval CHECKPOINT_INTERVAL] [--load-checkpoint LOAD_CHECKPOINT] [--save-checkpoint SAVE_CHECKPOINT]
                [--dataset-file DATASET_FILE] [--dataset-dir DATASET_DIR] [--tensorboard-log-dir TENSORBOARD_LOG_DIR]
                [--wandb-project WANDB_PROJECT] [--plot-learning-curves] [--debug]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  --learning-rate LEARNING_RATE
  --warmup-steps WARMUP_STEPS
  --batch-size BATCH_SIZE
  --mixed-precision MIXED_PRECISION
  --device DEVICE
  --num-layers NUM_LAYERS
  --embed-dim EMBED_DIM
  --ffwd-dim FFWD_DIM
  --d-model D_MODEL
  --num-attention-heads NUM_ATTENTION_HEADS
  --seq-len SEQ_LEN
  --max-output-tokens MAX_OUTPUT_TOKENS
  --eval-interval EVAL_INTERVAL
  --eval-iters EVAL_ITERS
  --checkpoint-interval CHECKPOINT_INTERVAL
  --load-checkpoint LOAD_CHECKPOINT
  --save-checkpoint SAVE_CHECKPOINT
  --dataset-file DATASET_FILE
  --dataset-dir DATASET_DIR
  --tensorboard-log-dir TENSORBOARD_LOG_DIR
  --wandb-project WANDB_PROJECT
  --plot-learning-curves
  --debug
  ```