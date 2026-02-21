#!/usr/bin/env bash
# =============================================================================
# cluster_info.sh — Side-by-side node alignment report
# =============================================================================
# SSHes into every node in the hostfile, collects hardware + software info,
# and prints a unified table. Highlights any version mismatches in red.
#
# Usage:
#   ./scripts/cluster_info.sh
#   HOSTFILE=hostfiles/hosts-2node.json ./scripts/cluster_info.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

HOSTFILE="${HOSTFILE:-$REPO_DIR/hostfiles/hosts-2node.json}"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
CYAN="\033[36m"
BOLD="\033[1m"
DIM="\033[2m"
RESET="\033[0m"

ok()   { printf "${GREEN}✓${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}!${RESET} %s\n" "$*"; }
err()  { printf "${RED}✗${RESET} %s\n" "$*"; }
sep()  { printf "${DIM}%s${RESET}\n" "────────────────────────────────────────────────────────────────────────────"; }

# ── Validate inputs ───────────────────────────────────────────────────────────
if [[ ! -f "$HOSTFILE" ]]; then
  err "Hostfile not found: $HOSTFILE"
  exit 1
fi

if [[ ! -f "$VENV_PYTHON" ]]; then
  err ".venv not found at $REPO_DIR/.venv"
  err "Run: ./scripts/setup.sh first"
  exit 1
fi

# ── Parse hostfile ────────────────────────────────────────────────────────────
HOSTS=$("$VENV_PYTHON" -c "
import json
with open('$HOSTFILE') as f:
    hosts = json.load(f)
print(' '.join(h['ssh'] for h in hosts))
")

# Compute relative path from HOME to REPO_DIR (works on both Macs if same username)
REPO_REL=$(python3 -c "import os; print(os.path.relpath('$REPO_DIR', os.path.expanduser('~')))")

# ── Probe script — runs on each remote node via SSH ───────────────────────────
# NOTE: single-quoted heredoc so local variables are NOT expanded.
#       We pass REPO_REL as an env var via SSH to avoid quoting issues.
build_probe() {
  local repo_rel="$1"
  cat <<'PROBE_EOF'
set -uo pipefail

REPO="$HOME/$REPO_REL"
VENV="$REPO/.venv/bin/python"

# system_profiler helper — uses grep + sed to avoid awk $2 issues
sp() {
  local key="$1"
  system_profiler SPHardwareDataType 2>/dev/null \
    | grep "$key" \
    | sed 's/.*: //' \
    | head -1 \
    | xargs
}

echo "hostname=$(hostname)"
echo "model=$(sp 'Model Name')"
echo "chip=$(sp 'Chip')"
echo "memory=$(sp 'Memory:')"
echo "cores=$(sp 'Total Number of Cores')"
echo "macos=$(sw_vers -productVersion 2>/dev/null || echo unknown)"
echo "build=$(sw_vers -buildVersion 2>/dev/null || echo unknown)"

if [[ -f "$VENV" ]]; then
  "$VENV" - <<PYEOF
import mlx.core as mx, sys
try:
    d = mx.device_info()
    mem_gb  = d["memory_size"] / (1024**3)
    wset_gb = d["max_recommended_working_set_size"] / (1024**3)
    buf_gb  = d["max_buffer_length"] / (1024**3)
    print("mlx_version=" + mx.__version__)
    print("gpu=" + d.get("device_name","unknown"))
    print("arch=" + d.get("architecture","unknown"))
    print("memory_gb=" + str(round(mem_gb)))
    print("wset_gb="   + str(round(wset_gb, 1)))
    print("buf_gb="    + str(round(buf_gb,  1)))
    print("python="    + sys.version.split()[0])
except Exception as e:
    print("mlx_error=" + str(e))
    print("mlx_version=ERROR")
    print("python=" + sys.version.split()[0])
PYEOF
else
  echo "mlx_version=NOT_INSTALLED"
  echo "python=NOT_INSTALLED"
  echo "gpu=N/A"
  echo "arch=N/A"
  echo "memory_gb=N/A"
  echo "wset_gb=N/A"
  echo "buf_gb=N/A"
fi

# RDMA
if command -v ibv_devices &>/dev/null; then
  DEVS=$(ibv_devices 2>/dev/null | grep -c "rdma_en" || true)
  ACTIVE=$(ibv_devinfo 2>/dev/null \
    | grep -E "hca_id|PORT_ACTIVE" \
    | grep -B1 "PORT_ACTIVE" \
    | grep "hca_id" \
    | sed 's/.*hca_id:[[:space:]]*//' \
    | tr '\n' ',' \
    | sed 's/,$//')
  echo "rdma_devices=$DEVS"
  echo "rdma_active=${ACTIVE:-none}"
else
  echo "rdma_devices=0"
  echo "rdma_active=none"
fi

# mlx.launch
if [[ -f "$REPO/.venv/bin/mlx.launch" ]]; then
  echo "mlx_launch=present"
else
  echo "mlx_launch=MISSING"
fi
PROBE_EOF
}

# ── Collect data from all nodes ───────────────────────────────────────────────
printf "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
printf "${BOLD}  MLX JACCL Cluster — Node Alignment Report${RESET}\n"
printf "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
printf "  Hostfile : ${DIM}%s${RESET}\n" "$HOSTFILE"
printf "  Nodes    : ${DIM}%s${RESET}\n" "$HOSTS"
printf "\n"

PROBE=$(build_probe "$REPO_REL")

declare -A NODE_DATA

echo "  Probing nodes..."
for h in $HOSTS; do
  printf "  ${DIM}→ %-24s${RESET} " "$h"
  if DATA=$(ssh -o ConnectTimeout=6 -o BatchMode=yes "$h" \
      "REPO_REL='$REPO_REL' bash -s" <<< "$PROBE" 2>/dev/null); then
    NODE_DATA["$h"]="$DATA"
    printf "${GREEN}OK${RESET}\n"
  else
    NODE_DATA["$h"]="hostname=$h
error=SSH_FAILED
mlx_version=SSH_FAILED
python=SSH_FAILED
gpu=SSH_FAILED
arch=SSH_FAILED
memory_gb=SSH_FAILED
wset_gb=SSH_FAILED
buf_gb=SSH_FAILED
rdma_devices=0
rdma_active=none
mlx_launch=SSH_FAILED
macos=SSH_FAILED
build=SSH_FAILED
chip=SSH_FAILED
memory=SSH_FAILED
cores=SSH_FAILED
model=SSH_FAILED"
    printf "${RED}FAILED${RESET}\n"
  fi
done

echo ""

# ── Helper: extract value from a node's data ─────────────────────────────────
get() {
  local host="$1" key="$2"
  echo "${NODE_DATA[$host]}" | grep "^${key}=" | cut -d'=' -f2- | head -1
}

# ── Build side-by-side table ──────────────────────────────────────────────────
COL_KEY=22
COL_VAL=26

# Header row
printf "${BOLD}${CYAN}%-${COL_KEY}s${RESET}" "Property"
for h in $HOSTS; do
  HN=$(get "$h" "hostname")
  printf " ${BOLD}${CYAN}%-${COL_VAL}s${RESET}" "${HN:-$h}"
done
echo ""
sep

# ── Print one row ─────────────────────────────────────────────────────────────
print_row() {
  local label="$1"
  local key="$2"
  local check_match="${3:-true}"

  printf "${DIM}%-${COL_KEY}s${RESET}" "$label"

  declare -a vals=()
  for h in $HOSTS; do
    vals+=("$(get "$h" "$key")")
  done

  local first="${vals[0]}"
  local all_match=true
  for v in "${vals[@]}"; do
    [[ "$v" != "$first" ]] && all_match=false && break
  done

  for v in "${vals[@]}"; do
    if [[ "$v" == "NOT_INSTALLED" || "$v" == "MISSING" || "$v" == "SSH_FAILED" || "$v" == "ERROR" ]]; then
      printf " ${RED}%-${COL_VAL}s${RESET}" "$v"
    elif [[ "$check_match" == "true" && "$all_match" == "false" ]]; then
      printf " ${YELLOW}%-${COL_VAL}s${RESET}" "$v"
    else
      printf " ${GREEN}%-${COL_VAL}s${RESET}" "$v"
    fi
  done

  if [[ "$check_match" == "true" && "$all_match" == "false" ]]; then
    printf "  ${YELLOW}⚠ MISMATCH${RESET}"
  fi

  echo ""
}

# ── Table sections ────────────────────────────────────────────────────────────

# Hardware
print_row "Model"              "model"       false
print_row "Chip"               "chip"        true
print_row "Memory"             "memory"      true
print_row "CPU Cores"          "cores"       false
sep

# Software
print_row "macOS"              "macos"       true
print_row "macOS Build"        "build"       true
print_row "Python"             "python"      true
print_row "MLX version"        "mlx_version" true
print_row "GPU"                "gpu"         true
print_row "Architecture"       "arch"        true
sep

# Memory / compute capacity
print_row "Unified RAM (GB)"   "memory_gb"   true
print_row "Max working set GB" "wset_gb"     true
print_row "Max buffer GB"      "buf_gb"      true
sep

# Cluster / RDMA
print_row "RDMA devices"       "rdma_devices" false
print_row "RDMA active port"   "rdma_active"  false
print_row "mlx.launch"         "mlx_launch"   true
sep

# ── Alignment verdict ─────────────────────────────────────────────────────────
echo ""
printf "${BOLD}  Alignment Verdict${RESET}\n"
echo ""

ISSUES=0

check_align() {
  local label="$1"
  local key="$2"

  declare -a vals=()
  for h in $HOSTS; do
    vals+=("$(get "$h" "$key")")
  done

  local first="${vals[0]}"
  local all_ok=true
  for v in "${vals[@]}"; do
    [[ "$v" != "$first" ]] && all_ok=false && break
  done

  if [[ "$all_ok" == "true" ]]; then
    printf "  ${GREEN}✓${RESET}  %-34s ${DIM}%s${RESET}\n" "$label" "$first"
  else
    printf "  ${RED}✗${RESET}  %-34s " "$label"
    for h in $HOSTS; do
      HN=$(get "$h" "hostname")
      V=$(get "$h" "$key")
      printf "${YELLOW}%s${RESET}=${RED}%s${RESET}  " "${HN:-$h}" "$V"
    done
    echo ""
    ISSUES=$((ISSUES + 1))
  fi
}

check_align "macOS version"      "macos"
check_align "MLX version"        "mlx_version"
check_align "Python version"     "python"
check_align "Chip"               "chip"
check_align "Unified memory GB"  "memory_gb"
check_align "Architecture"       "arch"
check_align "mlx.launch"         "mlx_launch"

echo ""

if [[ "$ISSUES" -eq 0 ]]; then
  printf "${BOLD}${GREEN}  ✓ All nodes are aligned — cluster is ready.${RESET}\n"
else
  printf "${BOLD}${RED}  ✗ %d alignment issue(s) found — resolve before running inference.${RESET}\n" "$ISSUES"
  echo ""
  printf "  ${DIM}Common fixes:${RESET}\n"
  printf "  ${DIM}  MLX mismatch    → ssh <node> && cd repo && ./scripts/setup.sh${RESET}\n"
  printf "  ${DIM}  macOS mismatch  → System Settings → Software Update${RESET}\n"
  printf "  ${DIM}  Python mismatch → rm -rf .venv && ./scripts/setup.sh${RESET}\n"
fi

printf "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n\n"
