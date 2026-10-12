#!/usr/bin/env bash
# ==============================================================================
# Helper Script to Start WebRTC Streaming on Remote VM
# Usage: ./start_stream.sh /path/to/checkpoint/model_xxxx.pt
# ==============================================================================
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_checkpoint.pt> [num_envs]"
    echo "Example: $0 logs/bpx_flat/2026-xx-xx_baseline/model_2999.pt 1"
    exit 1
fi

CHECKPOINT="$1"
NUM_ENVS="${2:-1}"

cd "$HOME/mirrorme_rl_train"

echo "===> Launching interactive evaluation with WebRTC streaming enabled..."
echo "Ensure you have Isaac Sim WebRTC Streaming Client installed on your Mac."
echo "Launch the client, input your server IP address and use the default ports, "
echo "then press 'Connect'."
echo "=============================================================================="

./bpx.sh play \
  --checkpoint "$CHECKPOINT" \
  --livestream 1 \
  --num_envs "$NUM_ENVS"
