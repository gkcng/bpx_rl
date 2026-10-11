#!/usr/bin/env bash
# ==============================================================================
# Remote VM Setup Script for BPX Reinforcement Learning & Isaac Lab
# Run this script on the Ubuntu 22.04 LTS instance with NVIDIA GPU.
# ==============================================================================
set -euo pipefail

echo "===> [1/6] Updating system packages and installing graphics/Vulkan dependencies..."
sudo apt-get update
sudo apt-get install -y \
  git \
  git-lfs \
  build-essential \
  cmake \
  libgl1-mesa-glx \
  libegl1 \
  libxrandr2 \
  libxinerama1 \
  libxcursor1 \
  libxi6 \
  ffmpeg \
  tmux \
  curl \
  wget

echo "===> [2/6] Checking NVIDIA driver and CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found. Installing NVIDIA driver 550..."
    sudo apt-get install -y linux-headers-$(uname -r) nvidia-driver-550 nvidia-utils-550
    echo "NVIDIA driver installed. A system reboot may be required before continuing."
fi

echo "===> [3/6] Setting up Miniconda..."
if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
fi

# Activate conda
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

echo "===> [4/6] Creating Python 3.11 environment (bpx_train)..."
if conda info --envs | grep -q "bpx_train"; then
    echo "Environment 'bpx_train' already exists. Activating..."
else
    conda create -n bpx_train python=3.11 -y
fi
conda activate bpx_train

echo "===> [5/6] Installing Isaac Lab 2.3.0, PyTorch, and RSL-RL..."
python -m pip install --upgrade pip
python -m pip install "setuptools<81" wheel
python -m pip install --no-build-isolation flatdict==4.0.1

# Install Isaac Lab with Isaac Sim & RSL-RL
python -m pip install "isaaclab[isaacsim,rsl-rl]==2.3.0" \
  --extra-index-url https://pypi.nvidia.com

# PyTorch 2.7.0 with CUDA 12.8 wheels
python -m pip install -U torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

echo "===> [6/6] Verifying PyTorch and Isaac Lab installation..."
python -c "from importlib.metadata import version; import torch; print('Isaac Lab:', version('isaaclab')); print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"

echo "=============================================================================="
echo "Setup completed successfully! Activate your environment with:"
echo "    conda activate bpx_train"
echo "=============================================================================="
