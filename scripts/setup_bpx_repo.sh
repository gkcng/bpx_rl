#!/usr/bin/env bash
# ==============================================================================
# Helper Script to Setup BPX RL Repository and Verification
# Run this inside the VM under conda environment `bpx_train`.
# ==============================================================================
set -euo pipefail

TARGET_DIR="$HOME/mirrorme_rl_train"

if [ ! -d "$TARGET_DIR" ]; then
    echo "===> Cloning mirrorme_rl_train..."
    git clone https://github.com/gkcng/mirrorme_rl_train.git "$TARGET_DIR"
fi

cd "$TARGET_DIR"

echo "===> Updating submodules (BPX robot asset URDF/meshes)..."
git submodule update --init --recursive

echo "===> Ensuring conda channel is strictly conda-forge (removes Anaconda defaults TOS)..."
conda config --remove channels defaults 2>/dev/null || true
conda config --add channels conda-forge
conda config --set channel_priority strict

echo "===> Running official MIRRORME environment setup (Isaac Lab, PyTorch, RSL-RL)..."
./bpx.sh setup bpx_train

echo "=============================================================================="
echo "BPX RL setup complete! Activate your environment with:"
echo "    conda activate bpx_train"
echo "To test or train:"
echo "    ./bpx.sh train --task bpx_flat --num_envs 64 --max_iterations 20 --headless"
echo "=============================================================================="
