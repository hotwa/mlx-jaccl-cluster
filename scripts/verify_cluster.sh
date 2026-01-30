#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Verify MLX-JACCL Cluster Connectivity
# =============================================================================
# Checks SSH connectivity and RDMA devices on all cluster nodes.
#
# Optional:
#   HOSTFILE  Path to hostfile (default: hostfiles/hosts.json)
#   HOSTS     Space-separated list of hosts (overrides hostfile)
#
# Example:
#   ./verify_cluster.sh
#   HOSTFILE=/path/to/hosts.json ./verify_cluster.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Get hosts from HOSTS env var, or extract from hostfile
if [[ -z "${HOSTS:-}" ]]; then
  HOSTFILE="${HOSTFILE:-$REPO_DIR/hostfiles/hosts.json}"

  if [[ -f "$HOSTFILE" ]]; then
    HOSTS=$(python3 -c "
import json
with open('$HOSTFILE') as f:
    hosts = json.load(f)
print(' '.join(h['ssh'] for h in hosts))
" 2>/dev/null || echo "")
  fi
fi

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: No hosts found. Set HOSTS or create a hostfile."
  exit 1
fi

echo "== SSH connectivity check =="
for h in $HOSTS; do
  echo -n "### $h: "
  if ssh -o ConnectTimeout=5 "$h" 'hostname' 2>/dev/null; then
    :
  else
    echo "FAILED"
  fi
done

echo
echo "== RDMA devices =="
for h in $HOSTS; do
  echo "### $h"
  ssh "$h" 'ibv_devices 2>/dev/null | grep -E "rdma_en[0-9]" || echo "(no RDMA devices found)"' || true
done

echo
echo "Done."
