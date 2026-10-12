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
  libglu1-mesa \
  libsm6 \
  libxext6 \
  libxrandr2 \
  libxinerama1 \
  libxcursor1 \
  libxi6 \
  libxxf86vm1 \
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

echo "===> [3/3] Setting up Miniconda & configuring conda-forge..."
if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
fi

# Activate and persist conda across all shell sessions
export PATH="$HOME/miniconda3/bin:$PATH"
"$HOME/miniconda3/bin/conda" init bash
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Set conda-forge as the ONLY channel and remove Anaconda's 'defaults' channel completely.
# This prevents Anaconda from prompting for Terms of Service (TOS) during conda create.
conda config --set custom_multichannels.defaults "" 2>/dev/null || true
conda config --remove channels defaults 2>/dev/null || true
conda config --add channels conda-forge
conda config --set channel_priority strict
conda config --set notify_outdated_conda false
conda config --set auto_update_conda false

echo "=============================================================================="
echo "VM host setup complete!"
echo "Next step: Run 'bash scripts/setup_bpx_repo.sh' to install BPX RL & Isaac Lab."
echo "=============================================================================="
