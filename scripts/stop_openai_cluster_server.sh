#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Stop MLX-JACCL Cluster Server
# =============================================================================
# Stops the OpenAI cluster server on all nodes.
#
# Optional:
#   HOSTFILE  Path to hostfile (default: hostfiles/hosts.json)
#   HOSTS     Space-separated list of hosts (overrides hostfile)
#
# Example:
#   ./stop_openai_cluster_server.sh
#   HOSTFILE=/path/to/hosts.json ./stop_openai_cluster_server.sh
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

echo "Stopping cluster server on: $HOSTS"
for h in $HOSTS; do
  echo "### stopping on $h"
  ssh "$h" 'pkill -f openai_cluster_server.py || true' 2>/dev/null || true
done

echo "Done."
