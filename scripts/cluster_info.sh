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
G="\033[32m"   # green
R="\033[31m"   # red
Y="\033[33m"   # yellow
C="\033[36m"   # cyan
B="\033[1m"    # bold
D="\033[2m"    # dim
X="\033[0m"    # reset

sep() { printf "${D}%s${X}\n" "────────────────────────────────────────────────────────────────────────────────"; }

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ ! -f "$HOSTFILE" ]]; then
  printf "${R}✗${X} Hostfile not found: %s\n" "$HOSTFILE"; exit 1
fi
if [[ ! -f "$VENV_PYTHON" ]]; then
  printf "${R}✗${X} .venv not found — run ./scripts/setup.sh first\n"; exit 1
fi

# ── Parse hosts from hostfile ─────────────────────────────────────────────────
HOSTS=()
while IFS= read -r _h; do
  [[ -n "$_h" ]] && HOSTS+=("$_h")
done <<< "$("$VENV_PYTHON" -c "
import json
with open('$HOSTFILE') as f:
    hosts = json.load(f)
for h in hosts:
    print(h['ssh'])
")"

NUM_HOSTS=${#HOSTS[@]}

# Relative path from HOME to REPO_DIR (same on both Macs — same username)
REPO_REL=$(python3 -c "import os; print(os.path.relpath('$REPO_DIR', os.path.expanduser('~')))")

# ── Probe script ──────────────────────────────────────────────────────────────
# Echoes key=value lines. Uses no associative arrays, no awk $2.
# REPO_REL is injected via SSH env.
PROBE=$(cat <<'PROBE_EOF'
REPO="$HOME/$REPO_REL"
VENV="$REPO/.venv/bin/python"

sp() {
  local key="$1"
  system_profiler SPHardwareDataType 2>/dev/null \
    | grep "$key" | sed 's/.*: //' | head -1 | xargs
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
    print("gpu="         + d.get("device_name","unknown"))
    print("arch="        + d.get("architecture","unknown"))
    print("memory_gb="   + str(round(mem_gb)))
    print("wset_gb="     + str(round(wset_gb,1)))
    print("buf_gb="      + str(round(buf_gb,1)))
    print("python="      + sys.version.split()[0])
except Exception as e:
    print("mlx_version=ERROR:" + str(e))
    print("python="      + sys.version.split()[0])
    print("gpu=ERROR")
    print("arch=ERROR")
    print("memory_gb=ERROR")
    print("wset_gb=ERROR")
    print("buf_gb=ERROR")
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

if command -v ibv_devices &>/dev/null; then
  DEVS=$(ibv_devices 2>/dev/null | grep -c "rdma_en" || echo 0)
  ACTIVE=$(ibv_devinfo 2>/dev/null \
    | grep -E "hca_id|PORT_ACTIVE" \
    | grep -B1 "PORT_ACTIVE" \
    | grep "hca_id" \
    | sed 's/.*hca_id:[[:space:]]*//' \
    | tr '\n' ',' | sed 's/,$//')
  echo "rdma_devices=$DEVS"
  echo "rdma_active=${ACTIVE:-none}"
else
  echo "rdma_devices=0"
  echo "rdma_active=none"
fi

if [[ -f "$REPO/.venv/bin/mlx.launch" ]]; then
  echo "mlx_launch=present"
else
  echo "mlx_launch=MISSING"
fi
PROBE_EOF
)

# ── Header ────────────────────────────────────────────────────────────────────
printf "\n${B}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${X}\n"
printf "${B}  MLX JACCL Cluster — Node Alignment Report${X}\n"
printf "${B}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${X}\n"
printf "  Hostfile : ${D}%s${X}\n" "$HOSTFILE"
printf "  Nodes    : ${D}%s${X}\n" "${HOSTS[*]}"
printf "\n"

# ── Probe each node — store output in temp files indexed by position ──────────
TMPDIR_PROBE=$(mktemp -d)
trap 'rm -rf "$TMPDIR_PROBE"' EXIT

printf "  Probing nodes...\n"
for i in "${!HOSTS[@]}"; do
  h="${HOSTS[$i]}"
  TMPFILE="$TMPDIR_PROBE/node_${i}.txt"
  printf "  ${D}→ %-26s${X} " "$h"
  if ssh -o ConnectTimeout=8 -o BatchMode=yes "$h" \
       "REPO_REL='$REPO_REL' bash -s" <<< "$PROBE" > "$TMPFILE" 2>/dev/null; then
    printf "${G}OK${X}\n"
  else
    # Write failure sentinel
    cat > "$TMPFILE" <<FAIL_EOF
hostname=$h
model=SSH_FAILED
chip=SSH_FAILED
memory=SSH_FAILED
cores=SSH_FAILED
macos=SSH_FAILED
build=SSH_FAILED
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
FAIL_EOF
    printf "${R}FAILED${X}\n"
  fi
done

printf "\n"

# ── Helpers ───────────────────────────────────────────────────────────────────

# get <node_index> <key>
get() {
  local idx="$1" key="$2"
  grep "^${key}=" "$TMPDIR_PROBE/node_${idx}.txt" 2>/dev/null \
    | head -1 | cut -d'=' -f2-
}

# get_hostname <node_index>
get_hn() {
  local v
  v=$(get "$1" "hostname")
  echo "${v:-node_$1}"
}

COL_KEY=24
COL_VAL=26

# ── Table header ──────────────────────────────────────────────────────────────
printf "${B}${C}%-${COL_KEY}s${X}" "Property"
for i in "${!HOSTS[@]}"; do
  printf " ${B}${C}%-${COL_VAL}s${X}" "$(get_hn "$i")"
done
printf "\n"
sep

# ── print_row <label> <key> [check_match=true] ────────────────────────────────
print_row() {
  local label="$1"
  local key="$2"
  local check_match="${3:-true}"

  printf "${D}%-${COL_KEY}s${X}" "$label"

  # Collect values
  local vals=()
  for i in "${!HOSTS[@]}"; do
    vals+=("$(get "$i" "$key")")
  done

  # Check if all match
  local first="${vals[0]}"
  local all_match=true
  for v in "${vals[@]}"; do
    [[ "$v" != "$first" ]] && all_match=false && break
  done

  # Print each value
  for v in "${vals[@]}"; do
    if [[ "$v" == *"FAILED"* || "$v" == *"MISSING"* || "$v" == *"NOT_INSTALLED"* || "$v" == "ERROR"* ]]; then
      printf " ${R}%-${COL_VAL}s${X}" "$v"
    elif [[ "$check_match" == "true" && "$all_match" == "false" ]]; then
      printf " ${Y}%-${COL_VAL}s${X}" "$v"
    else
      printf " ${G}%-${COL_VAL}s${X}" "$v"
    fi
  done

  if [[ "$check_match" == "true" && "$all_match" == "false" ]]; then
    printf "  ${Y}⚠ MISMATCH${X}"
  fi

  printf "\n"
}

# ── Hardware ──────────────────────────────────────────────────────────────────
print_row "Model"              "model"       false
print_row "Chip"               "chip"        true
print_row "Memory"             "memory"      true
print_row "CPU Cores"          "cores"       false
sep

# ── Software ──────────────────────────────────────────────────────────────────
print_row "macOS"              "macos"       true
print_row "macOS Build"        "build"       true
print_row "Python"             "python"      true
print_row "MLX version"        "mlx_version" true
print_row "GPU"                "gpu"         true
print_row "Architecture"       "arch"        true
sep

# ── Compute capacity ──────────────────────────────────────────────────────────
print_row "Unified RAM (GB)"   "memory_gb"   true
print_row "Max working set GB" "wset_gb"     true
print_row "Max buffer GB"      "buf_gb"      true
sep

# ── RDMA / cluster ────────────────────────────────────────────────────────────
print_row "RDMA devices"       "rdma_devices" false
print_row "RDMA active port"   "rdma_active"  false
print_row "mlx.launch"         "mlx_launch"   true
sep

# ── Alignment verdict ─────────────────────────────────────────────────────────
printf "\n${B}  Alignment Verdict${X}\n\n"

ISSUES=0

check_align() {
  local label="$1"
  local key="$2"

  local vals=()
  for i in "${!HOSTS[@]}"; do
    vals+=("$(get "$i" "$key")")
  done

  local first="${vals[0]}"
  local all_ok=true
  for v in "${vals[@]}"; do
    [[ "$v" != "$first" ]] && all_ok=false && break
  done

  if [[ "$all_ok" == "true" ]]; then
    printf "  ${G}✓${X}  %-34s ${D}%s${X}\n" "$label" "$first"
  else
    printf "  ${R}✗${X}  %-34s " "$label"
    for i in "${!HOSTS[@]}"; do
      local hn v
      hn=$(get_hn "$i")
      v=$(get "$i" "$key")
      printf "${Y}%s${X}=${R}%s${X}  " "$hn" "$v"
    done
    printf "\n"
    ISSUES=$((ISSUES + 1))
  fi
}

check_align "macOS version"       "macos"
check_align "MLX version"         "mlx_version"
check_align "Python version"      "python"
check_align "Chip"                "chip"
check_align "Unified memory (GB)" "memory_gb"
check_align "Architecture"        "arch"
check_align "mlx.launch"          "mlx_launch"

printf "\n"

if [[ "$ISSUES" -eq 0 ]]; then
  printf "${B}${G}  ✓ All nodes are aligned — cluster is ready.${X}\n"
else
  printf "${B}${R}  ✗ %d alignment issue(s) found.${X}\n" "$ISSUES"
  printf "\n"
  printf "  ${D}Fixes:${X}\n"
  printf "  ${D}  MLX mismatch    → ssh <node> 'cd %s && ./scripts/setup.sh'${X}\n" "$REPO_REL"
  printf "  ${D}  macOS mismatch  → System Settings → Software Update${X}\n"
  printf "  ${D}  Python mismatch → rm -rf .venv && ./scripts/setup.sh${X}\n"
fi

printf "\n${B}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${X}\n\n"
