#!/usr/bin/env bash
# Create the conda env used for every run. Usage: CONDA_ROOT=... scripts/setup_env.sh
set -euo pipefail
ENV_NAME=act-quant
source "$CONDA_ROOT/etc/profile.d/conda.sh"

conda create -y -n "$ENV_NAME" python=3.12
conda activate "$ENV_NAME"
conda install -y -c conda-forge ffmpeg=7

# LeRobot 0.6.1 expects CUDA 12.8 torch wheels (works with NVIDIA driver >= 570).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "lerobot[aloha,training]==0.6.1"
pip freeze > requirements.lock.txt
