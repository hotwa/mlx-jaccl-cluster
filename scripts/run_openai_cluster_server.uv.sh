#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MLX-JACCL Cluster OpenAI Server Launcher
# =============================================================================
# Starts an OpenAI-compatible API server distributed across your MLX cluster.
#
# Required:
#   MODEL_DIR    Path to the MLX model directory (must exist on all nodes)
#
# Optional environment variables:
#   HOSTFILE     Path to cluster hostfile (default: hostfiles/hosts.json)
#   MODEL_ID     Model identifier for API responses (default: basename of MODEL_DIR)
#   ENV_NAME     Conda environment name (default: mlxjccl)
#   HTTP_HOST    HTTP server bind address (default: 0.0.0.0)
#   HTTP_PORT    HTTP server port (default: 8080)
#   CTRL_HOST    Coordinator IP for rank0 (default: auto-detect from hostfile)
#   CTRL_PORT    Coordinator port (default: 18080)
#   CTRL_DONE_TIMEOUT  Max wait seconds for worker done-acks per request (default: 25)
#   QUEUE_MAX    Max queued requests (default: 8)
#   REQ_TIMEOUT  Request timeout in seconds (default: 120)
#   MAX_TOKENS   Default max_tokens per request when client omits it (default: 512)
#
# Example:
#   MODEL_DIR=/path/to/model ./run_openai_cluster_server.sh
#   MODEL_DIR=/path/to/model HOSTFILE=/path/to/hosts.json ./run_openai_cluster_server.sh
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ---------
# Required settings
# ---------
if [[ -z "${MODEL_DIR:-}" ]]; then
  echo "ERROR: MODEL_DIR is required. Set it to the path of your MLX model."
  echo "Example: MODEL_DIR=/path/to/model ./run_openai_cluster_server.sh"
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: MODEL_DIR does not exist: $MODEL_DIR"
  exit 1
fi

# ---------
# Settings with defaults
# ---------
ENV_NAME="${ENV_NAME:-mlxjccl}"
HOSTFILE="${HOSTFILE:-$REPO_DIR/hostfiles/hosts.json}"
SERVER_PY="${SERVER_PY:-$REPO_DIR/server/openai_cluster_server.py}"

# Default MODEL_ID to the basename of MODEL_DIR
MODEL_ID="${MODEL_ID:-$(basename "$MODEL_DIR")}"

HTTP_HOST="${HTTP_HOST:-0.0.0.0}"
HTTP_PORT="${HTTP_PORT:-8080}"
CTRL_PORT="${CTRL_PORT:-18080}"
CTRL_DONE_TIMEOUT="${CTRL_DONE_TIMEOUT:-25}"
QUEUE_MAX="${QUEUE_MAX:-8}"
REQ_TIMEOUT="${REQ_TIMEOUT:-120}"
MAX_TOKENS="${MAX_TOKENS:-512}"
SKIP_KILL="${SKIP_KILL:-0}"
SSH_OPTS="${SSH_OPTS:- -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null}"
MLX_HARD_MAX_TOKENS="${MLX_HARD_MAX_TOKENS:-}"

# ---------
# Validate paths
# ---------
if [[ ! -f "$HOSTFILE" ]]; then
  echo "ERROR: Hostfile not found: $HOSTFILE"
  echo "Create a hostfile or set HOSTFILE=/path/to/your/hostfile.json"
  exit 1
fi

if [[ ! -f "$SERVER_PY" ]]; then
  echo "ERROR: Server script not found: $SERVER_PY"
  exit 1
fi

# ---------
# Auto-detect CTRL_HOST from hostfile if not set
# ---------
if [[ -z "${CTRL_HOST:-}" ]]; then
  # Extract the first IP from the first host's "ips" array
  CTRL_HOST=$(python3 -c "
import json, sys
with open('$HOSTFILE') as f:
    hosts = json.load(f)
ips = hosts[0].get('ips', [])
if ips:
    print(ips[0])
else:
    # Fallback: try to get IP of first host via ssh hostname
    print('')
" 2>/dev/null || echo "")

  if [[ -z "$CTRL_HOST" ]]; then
    echo "ERROR: Could not auto-detect CTRL_HOST from hostfile."
    echo "Set CTRL_HOST to the LAN IP of rank0 (first host in hostfile)."
    exit 1
  fi
fi

# ---------
# Extract hosts from hostfile for cleanup
# ---------
HOSTS=$(python3 -c "
import json
with open('$HOSTFILE') as f:
    hosts = json.load(f)
print(' '.join(h['ssh'] for h in hosts))
" 2>/dev/null || echo "")

# ---------
# Preflight: validate hostfile IPs + SSH reachability
# ---------
if [[ "${SKIP_SSH_CHECK:-0}" != "1" ]]; then
  echo "Preflight: checking hostfile IPs and SSH reachability..."
  python3 - <<'PY' "$HOSTFILE"
import json, sys
path = sys.argv[1]
with open(path) as f:
    hosts = json.load(f)
bad = False
for i, h in enumerate(hosts):
    ips = h.get("ips", [])
    if not ips:
        print(f"WARNING: host[{i}] {h.get('ssh','<missing>')} has empty ips[] in hostfile.")
        bad = True
if bad:
    print("WARNING: Empty ips[] may cause barrier hangs. Consider filling LAN IPs.")
PY

  for h in $HOSTS; do
    echo "  - $h"
    python3 - <<'PY' "$h" 2>/dev/null || true
import socket, sys
name = sys.argv[1]
try:
    print("    resolves_to:", socket.gethostbyname(name))
except Exception as e:
    print("    resolves_to: <unresolved>")
PY
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" 'echo "    ssh: ok"' 2>/dev/null || \
      echo "    ssh: FAILED (password prompt or unreachable)"
  done
  echo
fi

# ---------
# Print configuration
# ---------
echo "=== MLX-JACCL Cluster Server ==="
echo "Model:      $MODEL_DIR"
echo "Model ID:   $MODEL_ID"
echo "Hostfile:   $HOSTFILE"
echo "Hosts:      $HOSTS"
echo "Ctrl Host:  $CTRL_HOST:$CTRL_PORT"
echo "HTTP:       $HTTP_HOST:$HTTP_PORT"
echo "Ctrl done:  ${CTRL_DONE_TIMEOUT}s"
echo "Max tokens: $MAX_TOKENS (default when request omits max_tokens)"
echo "================================"
echo

# ---------
# Stop any old copies on cluster nodes
# ---------
if [[ -n "$HOSTS" && "$SKIP_KILL" != "1" ]]; then
  echo "Stopping any existing server processes..."
  for h in $HOSTS; do
    if [[ "$h" == "$CTRL_HOST" || "$h" == "127.0.0.1" || "$h" == "localhost" ]]; then
      pkill -f openai_cluster_server.py || true
      pkill -f mlx.launch || true
    else
      ssh $SSH_OPTS "$h" \
        'pkill -f openai_cluster_server.py || true; pkill -f mlx.launch || true' \
        2>/dev/null || true
    fi
  done
fi

# ---------
# Start the server
# ---------
echo "Starting cluster server..."
EXTRA_ENV=()
if [[ -n "$MLX_HARD_MAX_TOKENS" ]]; then
  EXTRA_ENV+=(--env MLX_HARD_MAX_TOKENS="$MLX_HARD_MAX_TOKENS")
fi

"$REPO_DIR/.venv/bin/mlx.launch" --verbose --backend jaccl \
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
  --env CTRL_DONE_TIMEOUT="$CTRL_DONE_TIMEOUT" \
  --env QUEUE_MAX="$QUEUE_MAX" \
  --env REQ_TIMEOUT="$REQ_TIMEOUT" \
  --env MAX_TOKENS="$MAX_TOKENS" \
  "${EXTRA_ENV[@]}" -- \
  "$SERVER_PY"
