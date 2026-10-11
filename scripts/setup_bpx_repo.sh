#!/usr/bin/env bash
# ==============================================================================
# Helper Script to Setup BPX RL Repository and Verification
# Run this inside the VM under conda environment `bpx_train`.
# ==============================================================================
set -euo pipefail

TARGET_DIR="$HOME/mirrorme_rl_train"

if [ ! -d "$TARGET_DIR" ]; then
    echo "===> Cloning mirrorme_rl_train..."
    git clone https://github.com/mirrormerobotics/mirrorme_rl_train.git "$TARGET_DIR"
fi

cd "$TARGET_DIR"

echo "===> Updating submodules (BPX robot asset URDF/meshes)..."
git submodule update --init --recursive

echo "===> Installing BPX package..."
./bpx.sh install

echo "===> Testing task registration..."
python scripts/list_envs.py

echo "===> Validating runtime contract (Actor 45-D, Critic 48-D, Action 12-D)..."
./bpx.sh contract --headless

echo "=============================================================================="
echo "BPX RL is ready! To start training, run:"
echo "    ./bpx.sh train --task bpx_flat --num_envs 1024 --max_iterations 3000 --run_name baseline --headless"
echo "=============================================================================="
