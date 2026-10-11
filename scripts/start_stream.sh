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
echo "Ensure you have forwarded port 8011 from your Mac:"
echo "    ssh -N -L 8011:localhost:8011 -L 8211:localhost:8211 user@<VM_IP>"
echo "Then navigate to: http://localhost:8011/streaming/client/ on your Mac browser."
echo "=============================================================================="

./bpx.sh play \
  --checkpoint "$CHECKPOINT" \
  --livestream 1 \
  --num_envs "$NUM_ENVS"
