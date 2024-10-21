#!/bin/bash
gcloud container clusters create demo --zone us-central1-c

gcloud container node-pools create gpu-pool \
  --accelerator type=nvidia-tesla-t4,count=1,gpu-driver-version=LATEST \
  --machine-type n1-standard-4 \
  --zone us-central1-c --cluster demo \
  --node-locations us-central1-c \
  --num-nodes 2