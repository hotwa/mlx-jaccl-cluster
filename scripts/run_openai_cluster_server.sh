#!/usr/bin/env bash
set -euo pipefail

# ---------
# Settings
# ---------
ENV_NAME="${ENV_NAME:-mlxjccl}"
HOSTFILE="${HOSTFILE:-/Users/alex/Code/mlx-jaccl-cluster/hostfiles/m3-ultra-jaccl.local.json}"
SERVER_PY="${SERVER_PY:-/Users/alex/Code/mlx-jaccl-cluster/server/openai_cluster_server.py}"

MODEL_DIR="${MODEL_DIR:-/Users/alex/models_mlx/mlx-community/Qwen3-4B-Instruct-2507-4bit}"
MODEL_ID="${MODEL_ID:-Qwen3-4B-Instruct-2507-4bit}"

HTTP_HOST="${HTTP_HOST:-0.0.0.0}"
HTTP_PORT="${HTTP_PORT:-8080}"

CTRL_HOST="${CTRL_HOST:-192.168.0.36}"   # rank0 LAN IP (same as coordinator host)
CTRL_PORT="${CTRL_PORT:-18080}"

QUEUE_MAX="${QUEUE_MAX:-8}"
REQ_TIMEOUT="${REQ_TIMEOUT:-120}"

# ---------
# Stop any old copies
# ---------
for h in macstudio1.local macstudio2.local macstudio3.local macstudio4.local; do
  ssh "$h" 'pkill -f openai_cluster_server.py || true' || true
done

# ---------
# Start
# ---------
/Users/alex/miniconda3/bin/conda run -n "$ENV_NAME" mlx.launch --verbose --backend jaccl \
  --hostfile "$HOSTFILE" \
  --env MLX_METAL_FAST_SYNCH=1 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env MODEL_DIR="$MODEL_DIR" \
  --env MODEL_ID="$MODEL_ID" \
  --env HOST="$HTTP_HOST" \
  --env PORT="$HTTP_PORT" \
  --env CTRL_HOST="$CTRL_HOST" \
  --env CTRL_PORT="$CTRL_PORT" \
  --env QUEUE_MAX="$QUEUE_MAX" \
  --env REQ_TIMEOUT="$REQ_TIMEOUT" -- \
  "$SERVER_PY"
