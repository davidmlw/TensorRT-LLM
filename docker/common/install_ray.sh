#!/bin/bash

set -ex

# Install systemctl
apt-get update && \
    apt-get install -y -o Dpkg::Options::="--force-confdef" systemd && \
    apt-get clean

# Install tini
apt-get update && \
    apt-get install -y tini aria2 fish sudo && \
    apt-get clean

apt-get remove -y ibverbs-providers libibverbs1
apt-get install -y libibverbs-dev
apt-get remove --purge --allow-change-held-packages cuda-nvrtc-dev-12-9

# Change pip source
pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    python -m pip install --upgrade pip

pip install --no-cache-dir --no-deps vllm torchaudio ray

pip install hf-transfer liger-kernel mathruler pre-commit  qwen-vl-utils ruff
#RUN pip install pyext

pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.0rc3

pip3 install triton==3.3.1

pip3 show -q nvidia-cudnn-cu12 nvidia-ml-py fastapi optree pydantic grpcio opencv-python opencv-fixer transformers accelerate datasets peft hf-transfer numpy pyarrow pandas codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler pytest py-spy pyext pre-commit ruff