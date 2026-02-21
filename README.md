# ⚡ mlx-jaccl-cluster

**Multi-Mac MLX inference over Thunderbolt RDMA — confirmed 8 GB/s on Apple M4 Pro.**

A lightweight, production-ready toolkit for running distributed [MLX](https://github.com/ml-explore/mlx) inference across Apple Silicon Macs connected via Thunderbolt, using [JACCL](https://machinelearning.apple.com/research/jaccl) (RDMA over Thunderbolt) as the transport layer. Exposes an OpenAI-compatible API with a live dashboard.

> **Why this exists:** [exo](https://github.com/exo-explore/exo) is a much larger project (32K lines) that also targets TB RDMA, but its auto-discovery and planner logic failed to produce working RDMA on our M4 Pro Mac minis. This fork takes the opposite approach — explicit configuration, minimal code, proven RDMA. See [docs/comparison-vs-exo.md](docs/comparison-vs-exo.md) for the full analysis.

---

## Highlights

| | |
|---|---|
| 🔗 **RDMA verified** | 8.05 GB/s peak bandwidth · 25.5 µs latency over Thunderbolt |
| 🧠 **Distributed inference** | Tensor-parallel `mlx_lm.sharded_load` across 2+ Macs |
| 🌐 **OpenAI-compatible API** | `/v1/chat/completions` + `/v1/completions` with SSE streaming |
| 📊 **Live dashboard** | HTMX + SSE — real-time tok/s, latency, queue depth, sparkline, chat UI |
| 🔧 **Makefile-driven** | `make setup` → `make rdma-test` → `make server` — every operation is one command |
| 📦 **Zero build toolchain** | No Rust, no Node.js, no npm — pure Python + Bash, managed by [uv](https://github.com/astral-sh/uv) |
| 🍎 **Stock MLX** | Uses official `mlx` from PyPI — no custom forks |

---

## Verified Hardware

| | Mac 1 | Mac 2 |
|---|---|---|
| Model | Mac mini (Mac16,11) | Mac mini (Mac16,11) |
| Chip | Apple M4 Pro | Apple M4 Pro |
| Memory | 48 GB unified | 48 GB unified |
| macOS | 26.3 (25D125) | 26.3 (25D125) |
| RDMA device | `rdma_en4` (PORT_ACTIVE) | `rdma_en4` (PORT_ACTIVE) |

**RDMA benchmark results** (from `make rdma-test`):

| Tensor size | Avg bandwidth | Peak bandwidth | Avg latency |
|---|---|---|---|
| 4 KB | — | — | 25.5 µs |
| 256 KB | 1.82 GB/s | 2.14 GB/s | 0.12 ms |
| 4 MB | 6.71 GB/s | 7.38 GB/s | 0.56 ms |
| 64 MB | 7.94 GB/s | **8.05 GB/s** | 7.60 ms |

---

## Repository Layout

```text
mlx-jaccl-cluster/
├── Makefile                          # All operations as make targets
├── pyproject.toml                    # uv / pip dependency manifest
├── hostfiles/
│   ├── hosts-2node.json              # Working 2-node hostfile
│   ├── hosts-1node.json              # Single-node (local testing)
│   └── hosts.json.example            # Template for custom setups
├── server/
│   ├── openai_cluster_server.py      # OpenAI-compatible API (rank 0 HTTP, all ranks compute)
│   └── dashboard.py                  # HTMX + SSE live dashboard
├── scripts/
│   ├── setup.sh                      # One-shot node installer (uv + .venv + deps + fingerprint)
│   ├── bootstrap_node.sh             # Remote node setup over SSH
│   ├── rdma_test.py                  # RDMA correctness + latency + bandwidth test
│   ├── jaccl_tps_bench.py            # Distributed tokens/sec benchmark
│   ├── cluster_info.sh               # Side-by-side node alignment report
│   ├── verify_cluster.sh             # SSH + RDMA device checks
│   ├── sync_nodes.sh                 # git pull on all nodes in parallel
│   ├── run_openai_cluster_server.sh  # Start the cluster server
│   └── stop_openai_cluster_server.sh # Stop the cluster server
└── docs/
    ├── architecture.md               # Deep technical architecture reference
    ├── roadmap.md                    # Feature roadmap + gap analysis vs exo
    ├── from-scratch.md               # Full setup guide (RDMA enable → running server)
    ├── comparison-vs-exo.md          # Deep comparison with exo project
    └── scripts-reference.md          # All scripts + Makefile targets reference
```

---

## Quickstart

### Prerequisites

- 2 Apple Silicon Macs connected via Thunderbolt cable
- RDMA enabled on both Macs (one-time, in macOS Recovery — see [docs/from-scratch.md](docs/from-scratch.md))
- SSH key-based auth between the Macs
- [Homebrew](https://brew.sh) installed on both

### 1. Clone and set up Mac 1

```bash
git clone https://github.com/omar-karray/mlx-jaccl-cluster.git
cd mlx-jaccl-cluster
make setup
```

This installs [uv](https://github.com/astral-sh/uv), creates a `.venv`, installs all Python dependencies, verifies imports, checks RDMA devices, and saves a hardware fingerprint.

### 2. Bootstrap Mac 2 (from Mac 1)

```bash
REMOTE=mac2.local make bootstrap
```

This SSHes into Mac 2, installs Homebrew/git/uv if needed, clones the repo to the same path, and runs `setup.sh` — all in one command.

### 3. Configure the hostfile

Edit `hostfiles/hosts-2node.json` with your actual hostnames and IPs:

```json
[
  {
    "ssh": "mac1.local",
    "ips": ["192.168.1.14"],
    "rdma": [null, "rdma_en4"]
  },
  {
    "ssh": "mac2.local",
    "ips": [],
    "rdma": ["rdma_en4", null]
  }
]
```

Find your RDMA device name: `ibv_devinfo 2>/dev/null | grep -E "hca_id|state"` — look for `PORT_ACTIVE`.

### 4. Verify the cluster

```bash
make verify        # SSH + RDMA device checks
make cluster-info  # Side-by-side version/hardware alignment
```

### 5. Test RDMA (no model needed)

```bash
make rdma-test
```

Expected output: correctness check → latency measurement → bandwidth sweep with GB/s readings. A healthy TB link shows **> 5 GB/s**.

### 6. Download a model and serve it

```bash
# Download on Mac 1
source .venv/bin/activate
huggingface-cli download mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --local-dir ~/models_mlx/Qwen3-4B

# Sync to Mac 2
rsync -avz ~/models_mlx/Qwen3-4B/ mac2.local:~/models_mlx/Qwen3-4B/

# Start the cluster server
MODEL_DIR=~/models_mlx/Qwen3-4B make server
```

### 7. Use it

```bash
# Health check
make health

# Chat
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":64}'

# Dashboard
open http://localhost:8080/dashboard
```

---

## Makefile Targets

Run `make help` for the full list. Key targets:

### Setup

| Target | Description |
|---|---|
| `make setup` | Install deps on this node (uv + .venv) |
| `REMOTE=mac2 make bootstrap` | Set up a remote node over SSH |

### Cluster Verification

| Target | Description |
|---|---|
| `make verify` | SSH + RDMA device checks on all nodes |
| `make cluster-info` | Side-by-side node alignment report |
| `make sync` | Pull latest code on all nodes |

### RDMA Tests

| Target | Description | Duration |
|---|---|---|
| `make rdma-quick` | 5 rounds, small tensors — smoke test | ~10 s |
| `make rdma-test` | 20 rounds, 4 sizes — full benchmark | ~30 s |
| `make rdma-verbose` | Same as above with per-round timing | ~30 s |
| `make rdma-stress` | 100 rounds, large tensors — stability test | ~5 min |

Override defaults:

```bash
RDMA_ROUNDS=50 RDMA_SIZES=1048576,16777216,67108864 RDMA_VERBOSE=1 make rdma-test
```

### Server

| Target | Description |
|---|---|
| `MODEL_DIR=... make server` | Start the OpenAI-compatible cluster server |
| `make server-stop` | Stop the server on all nodes |
| `make server-restart` | Stop then start (requires `MODEL_DIR`) |
| `make health` | Check `/health` endpoint |
| `make models` | List served models |
| `make chat-test` | Send a test chat completion |
| `make queue` | Show request queue depth |
| `make dashboard` | Open the live dashboard in the default browser |
| `make metrics` | Show current metrics snapshot (JSON) |

### Model Management

| Target | Description |
|---|---|
| `MODEL=mlx-community/... make download` | Download a model from HuggingFace and rsync to all nodes |
| `make models-local` | List locally downloaded models with sizes |
| `MODEL_DIR=... make models-check` | Verify model exists on all nodes |

```bash
# Download and sync a model to the whole cluster in one command
MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit make download

# Then serve it
MODEL_DIR=~/models_mlx/Qwen3-4B-Instruct-2507-4bit make server
```

### Monitoring & Status

| Target | Description |
|---|---|
| `make status` | Full cluster snapshot: nodes, memory, RDMA, server, model |
| `make monitor` | Live-updating status (refreshes every 5 s, Ctrl+C to stop) |
| `make logs` | Tail server log file |
| `make version` | Show version info for all components (mlx, Python, macOS, chip) |

### Quality & Testing

| Target | Description |
|---|---|
| `make lint` | Syntax check (py_compile) + shellcheck on all code |
| `make test` | Full test suite: lint → RDMA quick → health check |
| `make loc` | Count lines of code by component |

### Utilities

| Target | Description |
|---|---|
| `make bench` | Distributed tokens/sec benchmark (requires `MODEL_DIR`) |
| `make kill-all` | Emergency stop — kill all MLX processes on all nodes |
| `make fingerprint` | Print this node's hardware/MLX info as JSON |
| `make clean` | Remove `.venv` locally |
| `make clean-all` | Remove `.venv` on all nodes |

---

## API Endpoints

The server (rank 0) exposes:

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI chat completions (streaming + non-streaming) |
| `/v1/completions` | POST | OpenAI text completions (streaming + non-streaming) |
| `/v1/models` | GET | List served models |
| `/health` | GET | Cluster health (world size, queue depth) |
| `/queue` | GET | Request queue status |
| `/dashboard` | GET | Live HTMX dashboard |
| `/metrics/stream` | GET | SSE metrics stream (tok/s, latency, queue) |
| `/metrics/snapshot` | GET | Current metrics as JSON |
| `/docs` | GET | Auto-generated Swagger/OpenAPI docs |

### Server Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | *(required)* | Path to the MLX model directory |
| `HOSTFILE` | `hostfiles/hosts-2node.json` | Path to cluster hostfile |
| `MODEL_ID` | basename of `MODEL_DIR` | Model identifier for API responses |
| `HTTP_HOST` | `0.0.0.0` | HTTP server bind address |
| `HTTP_PORT` | `8080` | HTTP server port |
| `CTRL_HOST` | auto-detect from hostfile | Coordinator IP (rank 0 LAN IP) |
| `CTRL_PORT` | `18080` | Coordinator control-plane port |
| `QUEUE_MAX` | `8` | Max queued requests |
| `REQ_TIMEOUT` | `120` | Per-request timeout in seconds |

---

## Dashboard

The server includes a built-in live dashboard at `/dashboard`:

- **Cluster topology** — rank, role, RDMA device, status for each node
- **RDMA banner** — Thunderbolt 5 / JACCL badge with peak bandwidth
- **Live metrics** — avg tok/s, peak tok/s, request count, latency (updated via SSE every 2s)
- **Queue depth** — visual bar with color coding (green → yellow → red)
- **Sparkline** — tok/s history over the last 40 generations
- **Chat UI** — full streaming chat interface, send messages directly from the dashboard
- **Uptime** — server uptime and total tokens generated

No build step. No Node.js. Pure HTMX + SSE served from Python.

---

## MLX Environment Variables

These are passed to all nodes via `mlx.launch --env`:

| Variable | Description |
|---|---|
| `MLX_METAL_FAST_SYNCH=1` | **Critical.** Enables fast Metal synchronization. Without this, expect 5–6× slower inference. |
| `HF_HUB_OFFLINE=1` | Prevents HuggingFace from downloading models at runtime. |
| `TRANSFORMERS_OFFLINE=1` | Same for the `transformers` library. |

**Why offline mode?** In a distributed cluster, every node would attempt to download the model simultaneously — causing races, inconsistent states, and unpredictable startup. Always download once on rank 0, then `rsync` to other nodes.

---

## Documentation

| Document | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Deep technical architecture: data plane, control plane, process model, request lifecycle |
| [docs/roadmap.md](docs/roadmap.md) | Feature roadmap: gap analysis vs exo, dashboard v2 design, tool support plan, phased priorities |
| [docs/from-scratch.md](docs/from-scratch.md) | Full setup guide: RDMA enablement → uv install → hostfile → RDMA test → model download → server |
| [docs/comparison-vs-exo.md](docs/comparison-vs-exo.md) | Deep comparison with exo — architecture, failure modes, benchmarks, closing the gap |
| [docs/scripts-reference.md](docs/scripts-reference.md) | Complete reference for all scripts and Makefile targets |

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                     Mac 1 (rank 0)                          │
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  FastAPI +    │   │  mlx_lm      │   │  Dashboard     │  │
│  │  uvicorn      │──▶│  .generate() │   │  (HTMX + SSE) │  │
│  │  :8080        │   │  (rank 0     │   │  /dashboard    │  │
│  │               │   │   shards)    │   │                │  │
│  └──────────────┘   └──────┬───────┘   └────────────────┘  │
│                            │                                │
│         ┌──────────────────┼──────────────────┐             │
│         │  TCP ctrl-plane  │  RDMA all_sum    │             │
│         │  :18080          │  (JACCL/TB)      │             │
│         │  {"type":"task"} │  ~8 GB/s         │             │
│         └──────────────────┼──────────────────┘             │
│                            │                                │
└────────────────────────────┼────────────────────────────────┘
             Thunderbolt ════╪════ rdma_en4
┌────────────────────────────┼────────────────────────────────┐
│                            │                                │
│         ┌──────────────────┼──────────────────┐             │
│         │  TCP ctrl-plane  │  RDMA all_sum    │             │
│         │  worker_loop()   │  (JACCL/TB)      │             │
│         │  recv task →     │  ~8 GB/s         │             │
│         │  generate() →    │                  │             │
│         │  send done       │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            │                                │
│                     ┌──────┴───────┐                        │
│                     │  mlx_lm      │                        │
│                     │  .generate() │                        │
│                     │  (rank 1     │                        │
│                     │   shards)    │                        │
│                     └──────────────┘                        │
│                                                             │
│                     Mac 2 (rank 1)                          │
└─────────────────────────────────────────────────────────────┘
```

**Data path:** Tensor-parallel collective operations (`all_sum`) run over RDMA at ~8 GB/s.

**Control path:** Rank 0 broadcasts `{"type":"task","prompt":"...","max_tokens":N}` to workers over a simple TCP socket. Workers call `generate()` (which triggers the RDMA collectives), then reply `{"type":"done"}`. This is ~120 lines of code total.

---

## Troubleshooting

### RDMA test shows LOW bandwidth or fails

1. Confirm RDMA is enabled: boot into macOS Recovery → `rdma_ctl enable` → reboot
2. Check devices: `ibv_devinfo` — look for `PORT_ACTIVE`
3. Make sure `MLX_METAL_FAST_SYNCH=1` is set (without it, bandwidth is 5–6× lower)
4. Re-seat the Thunderbolt cable

### `make rdma-test` hangs

Both nodes must have matching environments. Run `make cluster-info` and check for yellow/red mismatches.

### Server starts but curl hangs

All ranks must enter `generate()` per request. Confirm workers connected:
```bash
# Check server logs for "all workers connected"
# If not, check CTRL_HOST and CTRL_PORT are reachable from workers
```

### Unexpected model downloads at startup

The server passes `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to all nodes. If you're still seeing downloads, the model path may be wrong — confirm `MODEL_DIR` exists on **all** nodes.

### Kill stuck processes

```bash
make kill-all       # kills all MLX processes on all nodes
make server-stop    # just the server
```

### Re-run setup after dependency changes

```bash
make clean    # or: make clean-all (all nodes)
make setup
```

---

## Roadmap

We're actively closing the gap between our minimal-but-working approach and exo's richer feature set — **without** sacrificing reliability, simplicity, or the zero-build-toolchain promise.

Planned in upcoming phases:

| Phase | What | Status |
|---|---|---|
| **1 — Observability** | Live RAM/memory per node, RDMA link probe, worker health detection, Prometheus `/metrics`, `make status` / `make monitor` | 🔜 Next |
| **2 — Dashboard v2** | Memory gauges, SVG topology graph, request history table, error log, responsive layout | Planned |
| **3 — Model Management** | `make download MODEL=...` with progress + auto-sync to all nodes, model registry | ✅ Makefile targets shipped |
| **4 — Tool Support** | Function calling (`tools=`), structured output (`response_format`), sampling params (`temperature`, `top_p`, `stop`) | Planned |
| **5 — API Parity** | Ollama API compatibility (`/api/generate`, `/api/chat`), client SDK testing | Planned |

> Full details, wireframes, and architecture decisions: **[docs/roadmap.md](docs/roadmap.md)**
> Deep architecture reference: **[docs/architecture.md](docs/architecture.md)**

---

## Notes

- `mlx_lm.server` is single-host only. This repo's server runs HTTP on rank 0 while all ranks participate in sharded compute.
- For 4 nodes, JACCL requires a **fully connected Thunderbolt mesh** (6 cables total).
- RDMA must be enabled in **macOS Recovery** on each Mac (`rdma_ctl enable`).
- The dashboard requires no build step — it's pure HTMX + SSE served from `dashboard.py`.

---

## License

MIT