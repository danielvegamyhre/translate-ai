# Translate AI

Making encoder-decoder transfomers cool again.

English to Spanish translation model with custom [Datasets](https://github.com/danielvegamyhre/translate-ai/tree/main/translate_ai/datasets) for:

- [English to Spanish Dataset](https://www.kaggle.com/datasets/lonnieqin/englishspanish-translation-dataset/data)
- [MultiUN dataset](https://opus.nlpl.eu/legacy/MultiUN.php) 

## Feature overview 
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

## Training

Training script templates to estimate MFU for:

- Single GPU training
- Multi-GPU training
- Multi-node training

Example for multi-node training on 2 nodes with 1 GPU each:
```
torchrun --nproc_per_node=1 --nnodes 2 --master_port=12345 \
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
  ```

## Performance Analysis

Performance analysis script templates to estimate MFU for:

- Single GPU training
- Multi-GPU training
- Multi-node training

## Inference

Example: 

```
python3 translate_ai/translate.py --english-query "The cat is blue." --checkpoint-file checkpoints/chkpt.pt
...
"El gato es azul."
```

## Training workload orchestration with Kubernetes

Using Kubernetes and the [JobSet API](https://github.com/kubernetes-sigs/jobset) simplifies the process of orchestrating
distributed training workloads, especially for very large scale workloads.

To deploy a training workload using Kubernetes, you can follow these steps. This guide will use Google Cloud to provision
the infrastructure, but you can use any k8s cluster (on-prem or cloud based).

### Prerequisites

1. Google Cloud account set up.
2. [gcloud](https://cloud.google.com/sdk/docs/install-sdk) installed in your local development environment.
3. [Docker](https://docs.docker.com/engine/install/) installed in your local development environment.
4. The following Google Cloud APIs are enabled: GKE, Artifact Repository 

### Steps

1. Build the container image and push it to Artifact Repository with the following command:

```bash
PROJECT_ID=your-gcp-project REPO_NAME=your-ar-repo IMAGE_NAME=translate TAG=latest ./build_and_push.sh
```

2. Create a GKE cluster with a single GPU nodepool. The example script below will provision a GKE cluster
called `demo` in zone `us-central1-c`, with a GPU node pool called `gpu-pool`. The GPU pool will have 2 nodes of type `n1-standard-4`, each with 1 [NVIDIA Tesla T4](https://www.nvidia.com/en-us/data-center/tesla-t4/) GPUs attached. Note these GPUs have peak hardware capacity of 8.1 TFLOPS (fp32) or 65 TFLOPS (fp16, bf16) - this will be needed if you want to run performance analysis to estimate MFU. If you use a different GPU, refer to the vendor spec to get the hardware peak FLOPS.   

```bash
./create_cluster.sh
```

3. Install the JobSet API, a k8s native distributed ML training orchestrator.

```bash
VERSION=v0.6.0
kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/$VERSION/manifests.yaml
```

You can find more detailed installation information [here](https://jobset.sigs.k8s.io/docs/installation/).

4. Modify the reference [JobSet manifest](jobset.yaml). Note the items marked optional are not needed in this example
as they are already configured to correctly match the infrastructure provisioned in these steps, but if you train on
different infrastructure (more/fewer nodes, more/fewer GPUs per node, etc.) you'll need to configure these parameters as well.

- Set the container image parameter to be your own container image which you built and pushed in step 1.
- (Optional) Set the Job template `parallelism` and `completions` fields to match the number of nodes in your GPU pool.
- (Optional) Set the environment variable `NPROC_PER_NODE` as the number of GPUs per node (in this case, 1).
- (Optional) Set the environment variable `NNODES` as the number of nodes in your GPU pool.
- (Optional) Set the environment variable `WANDB_PROJECT` to be your Weights and Biases project name.
- (Optional) Set the environment variable `WANDB_API_KEY` to be your Weights and Biases API key. The best practices for doing this
securely can be found [here](https://kubernetes.io/docs/tasks/inject-data-application/distribute-credentials-secure/).

5. Deploy the workload!

```bash
kubectl apply -f jobset.yaml
```

6. Verify the training is working, either by viewing container logs, Tensorboard, or Weights and Biases (depending on what
observability instrumentation you have set up).