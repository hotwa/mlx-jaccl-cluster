#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MLX-JACCL Cluster OpenAI Server Launcher
# =============================================================================
# Starts an OpenAI-compatible API server distributed across your MLX cluster.
# Uses the .venv created by scripts/setup.sh — no conda required.
#
# Required:
#   MODEL_DIR    Path to the MLX model directory (must exist on all nodes)
#
# Optional environment variables:
#   HOSTFILE     Path to cluster hostfile (default: hostfiles/hosts-2node.json)
#   MODEL_ID     Model identifier for API responses (default: basename of MODEL_DIR)
#   VENV_DIR     Path to the virtualenv (default: <repo>/.venv)
#   HTTP_HOST    HTTP server bind address (default: 0.0.0.0)
#   HTTP_PORT    HTTP server port (default: 8080)
#   CTRL_HOST    Coordinator IP for rank0 (default: auto-detect from hostfile)
#   CTRL_PORT    Coordinator port (default: 18080)
#   QUEUE_MAX    Max queued requests (default: 8)
#   REQ_TIMEOUT  Request timeout in seconds (default: 120)
#
# Examples:
#   MODEL_DIR=/path/to/model ./scripts/run_openai_cluster_server.sh
#   MODEL_DIR=/path/to/model HOSTFILE=hostfiles/hosts-2node.json ./scripts/run_openai_cluster_server.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ── Colours ──────────────────────────────────────────────────────────────────
_c()    { printf "\033[%sm%s\033[0m\n" "$1" "$2"; }
info()  { _c "36" "  → $*"; }
error() { _c "31" "  ✗ $*"; exit 1; }

# ── Required: MODEL_DIR ──────────────────────────────────────────────────────
if [[ -z "${MODEL_DIR:-}" ]]; then
  error "MODEL_DIR is required.\n  Example: MODEL_DIR=/path/to/model ./scripts/run_openai_cluster_server.sh"
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  error "MODEL_DIR does not exist: $MODEL_DIR"
fi

# ── Defaults ─────────────────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
HOSTFILE="${HOSTFILE:-$REPO_DIR/hostfiles/hosts-2node.json}"
SERVER_PY="${SERVER_PY:-$REPO_DIR/server/openai_cluster_server.py}"

MODEL_ID="${MODEL_ID:-$(basename "$MODEL_DIR")}"
HTTP_HOST="${HTTP_HOST:-0.0.0.0}"
HTTP_PORT="${HTTP_PORT:-8080}"
CTRL_PORT="${CTRL_PORT:-18080}"
QUEUE_MAX="${QUEUE_MAX:-8}"
REQ_TIMEOUT="${REQ_TIMEOUT:-120}"

# Auto-detect RDMA devices from hostfile (comma-separated, one per rank)
# Can be overridden with RDMA_DEVICES="rdma_en4,rdma_en4"
if [[ -z "${RDMA_DEVICES:-}" ]]; then
  RDMA_DEVICES=$(python3 -c "
import json
with open('$HOSTFILE') as f:
    hosts = json.load(f)
devs = []
for h in hosts:
    rdma = h.get('rdma', [])
    # rdma is a list like [null, 'rdma_en4'] — pick the first non-null entry
    dev = next((d for d in rdma if d), 'rdma_en4')
    devs.append(dev)
print(','.join(devs))
" 2>/dev/null || echo "rdma_en4,rdma_en4")
fi

# ── Validate paths ────────────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
  error ".venv not found at $VENV_DIR\n  Run: ./scripts/setup.sh"
fi

MLX_LAUNCH="$VENV_DIR/bin/mlx.launch"
if [[ ! -f "$MLX_LAUNCH" ]]; then
  error "mlx.launch not found at $MLX_LAUNCH\n  Run: ./scripts/setup.sh"
fi

if [[ ! -f "$HOSTFILE" ]]; then
  error "Hostfile not found: $HOSTFILE\n  Set HOSTFILE= or edit hostfiles/hosts-2node.json"
fi

if [[ ! -f "$SERVER_PY" ]]; then
  error "Server script not found: $SERVER_PY"
fi

# ── Auto-detect CTRL_HOST from rank0's ips[] in hostfile ─────────────────────
if [[ -z "${CTRL_HOST:-}" ]]; then
  CTRL_HOST=$(python3 -c "
import json
with open('$HOSTFILE') as f:
    hosts = json.load(f)
ips = hosts[0].get('ips', [])
print(ips[0] if ips else '')
" 2>/dev/null || echo "")

  if [[ -z "$CTRL_HOST" ]]; then
    error "Could not auto-detect CTRL_HOST from hostfile.\n  Set CTRL_HOST to the LAN IP of rank0 (first entry in hostfile)."
  fi
fi

# ── Extract SSH host list for pre-flight cleanup ──────────────────────────────
HOSTS=$(python3 -c "
import json
with open('$HOSTFILE') as f:
    hosts = json.load(f)
print(' '.join(h['ssh'] for h in hosts))
" 2>/dev/null || echo "")

# ── Print configuration ───────────────────────────────────────────────────────
echo ""
printf "\033[1m%s\033[0m\n" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "\033[1m%s\033[0m\n" "  MLX-JACCL Cluster Server"
printf "\033[1m%s\033[0m\n" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "Model      : $MODEL_DIR"
info "Model ID   : $MODEL_ID"
info "Hostfile   : $HOSTFILE"
info "Hosts      : $HOSTS"
info "Ctrl       : $CTRL_HOST:$CTRL_PORT"
info "HTTP       : $HTTP_HOST:$HTTP_PORT"
info "RDMA devs  : $RDMA_DEVICES"
info "venv       : $VENV_DIR"
info "mlx.launch : $MLX_LAUNCH"
echo ""
printf "\033[1;32m  Dashboard → http://localhost:%s/dashboard\033[0m\n" "$HTTP_PORT"
echo ""

# ── Stop any stale server processes on all nodes ──────────────────────────────
if [[ -n "$HOSTS" ]]; then
  echo "Stopping any existing server processes..."
  for h in $HOSTS; do
    ssh "$h" 'pkill -f openai_cluster_server.py 2>/dev/null || true' 2>/dev/null || true
  done
  echo ""
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo "Starting cluster server..."
"$MLX_LAUNCH" --verbose --backend jaccl \
  --hostfile "$HOSTFILE" \
  --env MLX_METAL_FAST_SYNCH=1 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env PYTHONPATH="$REPO_DIR/server" \
  --env MODEL_DIR="$MODEL_DIR" \
  --env RDMA_DEVICES="$RDMA_DEVICES" \
  --env MODEL_ID="$MODEL_ID" \
  --env HOST="$HTTP_HOST" \
  --env PORT="$HTTP_PORT" \
  --env CTRL_HOST="$CTRL_HOST" \
  --env CTRL_PORT="$CTRL_PORT" \
  --env QUEUE_MAX="$QUEUE_MAX" \
  --env REQ_TIMEOUT="$REQ_TIMEOUT" -- \
  python "$SERVER_PY"
