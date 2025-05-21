FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    git config --global --add safe.directory /workspace

RUN pip install wandb==0.17.5 \
    tqdm \
    datasets==3.2.0 \
    transformers==4.48.0
