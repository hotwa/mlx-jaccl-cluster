# Scripts & Makefile Reference

> All paths relative to the repo root (`mlx-jaccl-cluster/`).

---

## Quick Reference

| What you want to do | Command |
|---|---|
| Install everything on this node | `make setup` |
| Bootstrap a remote node from scratch | `REMOTE=mac2.local make bootstrap` |
| Check SSH + RDMA devices | `make verify` |
| Side-by-side node comparison | `make cluster-info` |
| Quick RDMA smoke test | `make rdma-quick` |
| Full RDMA bandwidth test | `make rdma-test` |
| RDMA stress test (large tensors, 100 rounds) | `make rdma-stress` |
| Run tokens/sec benchmark | `MODEL_DIR=~/models_mlx/Qwen3-4B make bench` |
| Start the OpenAI server | `MODEL_DIR=~/models_mlx/Qwen3-4B make server` |
| Stop the server | `make server-stop` |
| Check server health | `make health` |
| Send a test chat message | `make chat-test` |
| Open the live dashboard | `make dashboard` |
| Show current metrics (JSON) | `make metrics` |
| Download a model and sync to all nodes | `MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit make download` |
| List locally downloaded models | `make models-local` |
| Verify model exists on all nodes | `MODEL_DIR=~/models_mlx/Qwen3-4B make models-check` |
| Full cluster status (nodes, memory, server) | `make status` |
| Live-updating cluster status | `make monitor` |
| Tail server logs | `make logs` |
| Show version info for all components | `make version` |
| Run code quality checks | `make lint` |
| Run full test suite | `make test` |
| Count lines of code | `make loc` |
| Pull latest code on all nodes | `make sync` |
| Kill everything (emergency) | `make kill-all` |
| Show all targets | `make help` |

---

## Makefile Targets

### Setup & Bootstrap

| Target | Description |
|---|---|
| `make setup` | Run `scripts/setup.sh` on the local node. Creates `.venv`, installs all Python dependencies, verifies imports, checks RDMA devices, saves a hardware fingerprint JSON. |
| `make bootstrap` | Bootstrap a remote node over SSH. Requires `REMOTE=<host>`. Installs Homebrew, git, uv, clones the repo, and runs `setup.sh` — all in one command. |
| `make clean` | Remove `.venv`, `__pycache__`, and fingerprint files on the local node. |
| `make clean-all` | Same as `clean` but runs on **all** nodes in the hostfile via SSH. |

### Cluster Verification

| Target | Description |
|---|---|
| `make verify` | SSH into each node in the hostfile, check connectivity, list RDMA devices. Fast — does not send data over RDMA. |
| `make cluster-info` | Probe every node and print a side-by-side table: hardware, macOS version, MLX version, Python version, RDMA devices, working set size. Highlights mismatches in yellow/red. |
| `make sync` | `git pull` on all nodes in parallel, then verify all nodes are on the same commit. |
| `make fingerprint` | Print this node's hardware + MLX info as JSON. |

### RDMA Tests

| Target | Description | Approx. duration |
|---|---|---|
| `make rdma-quick` | 5 rounds, small tensors (4 KB, 256 KB). Smoke test — confirms RDMA works at all. | ~10 s |
| `make rdma-test` | 20 rounds, 4 sizes (4 KB → 64 MB). Full correctness + latency + bandwidth sweep. | ~30 s |
| `make rdma-verbose` | Same as `rdma-test` but prints per-round timing. | ~30 s |
| `make rdma-stress` | 100 rounds, large tensors (4 MB → 512 MB). Stability + sustained bandwidth test. | ~5 min |

All RDMA targets accept these overrides:

```bash
RDMA_ROUNDS=50 RDMA_SIZES=1024,65536,1048576 RDMA_VERBOSE=1 RDMA_MAX_MB=512 make rdma-test
```

### Benchmarks

| Target | Description |
|---|---|
| `make bench` | Distributed tokens/sec benchmark using `jaccl_tps_bench.py`. Requires `MODEL_DIR`. Prints prompt tokens, generated tokens, time, and tok/s. |

Override the prompt and token count:

```bash
MODEL_DIR=~/models_mlx/Qwen3-4B \
BENCH_PROMPT="Explain quantum computing in simple terms." \
BENCH_TOKENS=512 \
make bench
```

### Server

| Target | Description |
|---|---|
| `make server` | Start the OpenAI-compatible cluster server. Requires `MODEL_DIR`. Launches via `mlx.launch` with JACCL backend. Dashboard available at `http://localhost:8080/dashboard`. |
| `make server-stop` | Stop the server on all nodes (sends `pkill` via SSH). |
| `make server-restart` | Stop then start (requires `MODEL_DIR`). |
| `make health` | `curl` the `/health` endpoint. |
| `make models` | `curl` the `/v1/models` endpoint. |
| `make chat-test` | Send a quick chat completion request to the running server. |
| `make queue` | Show current request queue depth. |
| `make dashboard` | Open the live dashboard in the default browser (`open` on macOS). |
| `make metrics` | Fetch `/metrics/snapshot` and pretty-print the JSON. |

Server configuration overrides:

```bash
MODEL_DIR=~/models_mlx/Qwen3-4B \
HTTP_PORT=9000 \
QUEUE_MAX=16 \
REQ_TIMEOUT=300 \
make server
```

### Model Management

| Target | Description |
|---|---|
| `make download` | Download a HuggingFace model and rsync to all nodes. Requires `MODEL=<hf-repo-id>`. Optionally set `MODELS_DIR` (default: `~/models_mlx`). |
| `make models-local` | Scan `MODELS_DIR` for directories containing `config.json` and list them with sizes and quantization info. |
| `make models-check` | SSH into every node in the hostfile and verify `MODEL_DIR` exists with a valid `config.json`. Reports per-node status. |

Download + sync a model in one command:

```bash
# Download and sync to all nodes
MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit make download

# Download to a custom directory
MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit MODELS_DIR=~/my_models make download

# Then serve it
MODEL_DIR=~/models_mlx/Qwen3-4B-Instruct-2507-4bit make server
```

### Monitoring & Status

| Target | Description |
|---|---|
| `make status` | Full cluster status snapshot: node connectivity (SSH + RDMA ports), server health (model, world size, queue), and local memory usage (active, cache, peak as a visual bar). |
| `make monitor` | Live-updating version of `make status`. Clears the terminal and refreshes every `MONITOR_INTERVAL` seconds (default: 5). Press Ctrl+C to stop. |
| `make logs` | Tail the server log file at `/tmp/mlx-jaccl-cluster.log`. If no log file exists, prints instructions for capturing logs via `tee`. |
| `make version` | Print version info for all components: mlx-jaccl-cluster, mlx, mlx-lm, fastapi, uvicorn, transformers, Python, macOS, and chip. |

```bash
# Full status snapshot
make status

# Live monitoring (refreshes every 3 seconds)
MONITOR_INTERVAL=3 make monitor
```

### Quality & Testing

| Target | Description |
|---|---|
| `make lint` | Run `py_compile` on all Python files in `server/` and `scripts/`. If `shellcheck` is installed, also checks all `.sh` scripts. Reports pass/fail per file. |
| `make test` | Run the full test suite in order: (1) `make lint`, (2) `make rdma-quick`, (3) server health check (skipped if server not running). |
| `make loc` | Count lines of code by component (server, scripts, Makefile, docs) and print a summary table with totals. |

```bash
# Run full test suite
make test

# Just lint
make lint

# Install shellcheck for better shell script checks
brew install shellcheck
```

### Utilities

| Target | Description |
|---|---|
| `make kill-all` | Emergency stop — kills all MLX and server processes on every node in the hostfile. |
| `make fingerprint` | Print this node's hardware + MLX info as JSON (hostname, mlx version, GPU, architecture, RAM, working set). |
| `make docs` | List all documentation files in the project. |
| `make loc` | Count lines of code by component. |

---

## Global Variables

These environment variables are respected by all targets:

| Variable | Default | Description |
|---|---|---|
| `HOSTFILE` | `hostfiles/hosts-2node.json` | Path to the JACCL hostfile |
| `MODEL_DIR` | *(none — required for server/bench)* | Path to the MLX model directory |
| `MODEL` | *(none — required for download)* | HuggingFace repo ID (e.g. `mlx-community/Qwen3-4B-Instruct-2507-4bit`) |
| `MODELS_DIR` | `~/models_mlx` | Base directory for downloaded models |
| `HTTP_PORT` | `8080` | HTTP server bind port |
| `CTRL_PORT` | `18080` | Coordinator control-plane port |
| `MONITOR_INTERVAL` | `5` | Seconds between refreshes for `make monitor` |

---

## Scripts Detail

### `scripts/setup.sh`

**One-shot installer for a single node.**

Runs on each Mac in the cluster. Does the following in order:

1. Checks macOS + Apple Silicon (rejects non-arm64)
2. Installs `uv` via Homebrew if missing
3. Creates `.venv` with Python 3.12 (`uv venv`)
4. Installs dependencies: `mlx`, `mlx-lm`, `fastapi`, `uvicorn`, `pydantic`, `transformers`, `tokenizers`, `mistral_common`, `huggingface_hub`
5. Verifies all packages are importable
6. Checks `mlx.launch` supports the JACCL backend
7. Prints full hardware fingerprint (chip, memory, MLX device info, GPU architecture, safe RDMA tensor sizes)
8. Detects RDMA devices via `ibv_devices` / `ibv_devinfo` and reports port state
9. Saves a `.node_fingerprint_<hostname>.json` file in the repo root

```bash
# Run locally
./scripts/setup.sh

# Run on a remote node
ssh mac2.local "cd ~/path/to/mlx-jaccl-cluster && ./scripts/setup.sh"
```

---

### `scripts/bootstrap_node.sh`

**Bootstrap a remote node from Mac 1 (the coordinator).**

Run this on Mac 1 to fully set up Mac 2 without touching Mac 2's keyboard:

1. Validates SSH connectivity
2. Installs Homebrew on the remote Mac (if missing)
3. Installs `git` and `uv` (if missing)
4. Clones this repo from GitHub to the same path as on Mac 1
5. Runs `setup.sh` on the remote Mac

```bash
./scripts/bootstrap_node.sh mac2.local
./scripts/bootstrap_node.sh 192.168.0.50
```

**Prerequisites:** SSH key-based auth to the remote Mac (`ssh-copy-id mac2.local`).

---

### `scripts/verify_cluster.sh`

**Fast connectivity check — SSH + RDMA device presence.**

For each node in the hostfile:
- Tests SSH connectivity (5-second timeout)
- Runs `ibv_devices` and lists any `rdma_en*` entries

Does **not** send data over RDMA. Use `make rdma-test` for that.

```bash
./scripts/verify_cluster.sh
HOSTFILE=hostfiles/hosts-2node.json ./scripts/verify_cluster.sh
```

---

### `scripts/cluster_info.sh`

**Side-by-side node alignment report.**

SSHes into every node, collects ~20 metrics, prints a table, and flags any mismatches:

| Category | Metrics collected |
|---|---|
| Hardware | Model, Chip, Memory, CPU cores |
| Software | macOS version + build, Python version, MLX version |
| GPU / Metal | GPU name, architecture, working set, max buffer |
| RDMA | Device count, active ports, `mlx.launch` presence |

At the bottom, prints an **alignment verdict** — green if all nodes match, red with details if they don't.

```bash
./scripts/cluster_info.sh
HOSTFILE=hostfiles/hosts-2node.json ./scripts/cluster_info.sh
```

---

### `scripts/sync_nodes.sh`

**Pull latest git changes on all nodes in parallel.**

1. `git pull origin main` on every node simultaneously
2. Reports whether each node was already up to date or received updates
3. Compares HEAD commits across all nodes
4. Flags if any node is on a different commit

```bash
./scripts/sync_nodes.sh
HOSTFILE=hostfiles/hosts-2node.json ./scripts/sync_nodes.sh
```

---

### `scripts/rdma_test.py`

**RDMA connectivity, correctness, latency, and bandwidth test.**

Runs via `mlx.launch --backend jaccl`. No model download required. Tests actual RDMA data transfer between all ranks using MLX distributed `all_sum` collectives.

**Phases:**

| Phase | What it tests |
|---|---|
| Phase 0 — Barrier | All ranks can reach a global barrier |
| Phase 1 — Correctness | `all_sum` produces the correct mathematical result across ranks |
| Phase 2 — Latency | Round-trip time for a 1-element (4-byte) `all_sum` |
| Phase 3 — Bandwidth | Sweep across tensor sizes, reports avg + peak GB/s |

**Summary output** includes a visual bar chart and an overall verdict:
- `EXCELLENT` (> 5 GB/s) — saturating the TB link
- `GREAT` (> 2 GB/s) — strong performance
- `OK` (> 500 MB/s) — working but moderate
- `LOW` (< 500 MB/s) — check cable and `rdma_ctl enable`

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `RDMA_ROUNDS` | `20` | Number of benchmark rounds per tensor size |
| `RDMA_SIZES` | `1024,65536,1048576,16777216` | Comma-separated tensor sizes (float32 elements) |
| `RDMA_VERBOSE` | `0` | Set to `1` for per-round timing |
| `RDMA_MAX_MB` | `256` | Safety cap — skip tensors whose peak memory exceeds this |

**Memory safety:** Explicitly deletes tensors and calls `mx.clear_cache()` between tests. Cache memory is reported inline to confirm cleanup.

```bash
# Standard test
make rdma-test

# Custom configuration
RDMA_ROUNDS=50 RDMA_SIZES=1048576,16777216,67108864 RDMA_VERBOSE=1 make rdma-test
```

---

### `scripts/jaccl_tps_bench.py`

**Distributed tokens/sec benchmark.**

Loads a model via `mlx_lm.sharded_load`, runs a warmup generation, then times a full generation and reports tokens/sec. Supports custom tokenizers (auto-detects `tokenization_*.py` in the model directory).

```bash
.venv/bin/mlx.launch --backend jaccl \
  --hostfile hostfiles/hosts-2node.json \
  --env MLX_METAL_FAST_SYNCH=1 \
  --env HF_HUB_OFFLINE=1 -- \
  scripts/jaccl_tps_bench.py \
  --model ~/models_mlx/Qwen3-4B \
  --prompt "Explain RDMA in simple terms." \
  --max-tokens 256
```

Output (rank 0 only):

```
==========
model=~/models_mlx/Qwen3-4B
world_size=2
prompt_tokens=8
gen_tokens=256
seconds=4.123
tokens_per_sec=62.091
```

---

### `scripts/run_openai_cluster_server.sh`

**Start the OpenAI-compatible cluster server.**

1. Validates `MODEL_DIR` exists
2. Auto-detects RDMA devices from the hostfile
3. Auto-detects `CTRL_HOST` from rank 0's `ips[]` in the hostfile
4. Kills any stale server processes on all nodes
5. Launches via `mlx.launch --backend jaccl` with all env vars forwarded

The server exposes:
- `GET /health` — cluster status
- `GET /v1/models` — list served models
- `POST /v1/chat/completions` — OpenAI chat (streaming + non-streaming)
- `POST /v1/completions` — OpenAI completions (streaming + non-streaming)
- `GET /queue` — request queue depth
- `GET /dashboard` — live HTMX dashboard
- `GET /metrics/stream` — SSE metrics endpoint
- `GET /docs` — auto-generated API docs (FastAPI/Swagger)

```bash
MODEL_DIR=~/models_mlx/Qwen3-4B ./scripts/run_openai_cluster_server.sh
```

---

### `scripts/stop_openai_cluster_server.sh`

**Stop the server on all nodes.**

Sends `pkill -f openai_cluster_server.py` to every node in the hostfile via SSH.

```bash
./scripts/stop_openai_cluster_server.sh
HOSTFILE=hostfiles/hosts-2node.json ./scripts/stop_openai_cluster_server.sh
```

---

## Hostfile Format

JACCL hostfiles are JSON arrays. Each entry represents one node:

```json
[
  {
    "ssh": "mac.home",
    "ips": ["192.168.1.14"],
    "rdma": [null, "rdma_en4"]
  },
  {
    "ssh": "mac2",
    "ips": [],
    "rdma": ["rdma_en4", null]
  }
]
```

| Field | Description |
|---|---|
| `ssh` | SSH hostname for this node (used by `mlx.launch` and all scripts) |
| `ips` | LAN IP addresses. Only rank 0 needs an IP (for the coordinator control-plane). |
| `rdma` | RDMA device matrix. `rdma[i]` is the device name to reach node `i` from this node. `null` = self (no RDMA to yourself). |

### How to find your RDMA device name

```bash
ibv_devinfo 2>/dev/null | grep -E "hca_id|state"
```

Look for a device with `PORT_ACTIVE` — that is the Thunderbolt port with a cable connected. Typically `rdma_en4` on Mac mini M4 Pro.

---

## Typical Workflows

### First-time setup (2-node cluster)

```bash
# On Mac 1
git clone https://github.com/omar-karray/mlx-jaccl-cluster.git
cd mlx-jaccl-cluster
make setup
REMOTE=mac2.local make bootstrap
# Edit hostfiles/hosts-2node.json with your hostnames + IPs
make verify
make cluster-info
make rdma-test
```

### Daily development

```bash
# After pushing changes on Mac 1
make sync          # pull on all nodes
make rdma-quick    # smoke test RDMA
make server        # or: MODEL_DIR=... make server
```

### Before running inference

```bash
make verify        # SSH + RDMA devices OK?
make rdma-quick    # RDMA data path working?
make bench         # MODEL_DIR=... — how fast are we?
make server        # MODEL_DIR=... — serve it
make chat-test     # quick smoke test
```

### Debugging a failed RDMA test

```bash
make verify                    # are RDMA devices present?
make cluster-info              # any version mismatches?
RDMA_VERBOSE=1 make rdma-test  # per-round timing to spot outliers
make rdma-stress               # sustained load to find intermittent failures
```
