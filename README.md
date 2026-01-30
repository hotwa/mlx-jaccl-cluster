# MLX JACCL (Thunderbolt RDMA) Cluster + OpenAI-Compatible Server

This repo helps you stand up a multi‑Mac **MLX** cluster using **JACCL** (RDMA over Thunderbolt) and expose it via **OpenAI-compatible HTTP endpoints** from rank 0.

You get:

- **From-scratch runbook** (`docs/from-scratch.md`)
- JACCL **hostfile template** (`hostfiles/hosts.json.example`)
- **Distributed tokens/sec benchmark** (`scripts/jaccl_tps_bench.py`)
- **OpenAI-compatible server** (rank0 HTTP + all ranks participate in `generate()`)
  - `GET /v1/models`
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
  - request **queue + backpressure**
  (`server/openai_cluster_server.py`)
- Start/stop/verify helper scripts (`scripts/`)

---

## Repository layout

- `docs/from-scratch.md` — full setup instructions (RDMA enablement, conda env, hostfile, troubleshooting)
- `hostfiles/`
  - `hosts.json.example` — template hostfile (copy to `hosts.json` and edit)
- `server/`
  - `openai_cluster_server.py` — OpenAI-compatible server (queue + backpressure)
- `scripts/`
  - `verify_cluster.sh` — SSH + RDMA device checks
  - `jaccl_tps_bench.py` — distributed tokens/sec benchmark (prints tokens/sec)
  - `run_openai_cluster_server.sh` — starts the server via `mlx.launch`
  - `stop_openai_cluster_server.sh` — stops the server on all nodes

---

## Quickstart

### 1) Follow the full setup guide

Read:

- `docs/from-scratch.md`

### 2) Create your hostfile

Copy the template:

```bash
cp hostfiles/hosts.json.example hostfiles/hosts.json
```

Edit `hostfiles/hosts.json`:

- set your `ssh` hostnames
- set rank 0 `"ips": ["<RANK0_LAN_IP>"]` (Ethernet recommended)
- confirm the `rdma` matrix matches your cabling

> `hostfiles/hosts.json` is ignored by git.

### 3) Verify the cluster is reachable + RDMA is enabled

```bash
./scripts/verify_cluster.sh
```

### 4) Run a distributed tokens/sec benchmark

```bash
# Replace <MODEL_PATH> with your actual model path, e.g. ~/models_mlx/Qwen3-4B
mlx.launch --backend jaccl \
  --hostfile hostfiles/hosts.json \
  --env MLX_METAL_FAST_SYNCH=1 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 -- \
  python scripts/jaccl_tps_bench.py \
  --model <MODEL_PATH> \
  --prompt "Write 5 sentences about Thunderbolt RDMA." \
  --max-tokens 256
```

Rank 0 prints:

- `prompt_tokens`, `gen_tokens`, `seconds`, `tokens_per_sec`

### 5) Start the OpenAI-compatible server (rank 0 HTTP)

```bash
# Replace <MODEL_PATH> with your actual model path
MODEL_DIR=<MODEL_PATH> ./scripts/run_openai_cluster_server.sh
```

Test:

```bash
curl -s http://<rank0-host>:8080/v1/models

curl -s http://<rank0-host>:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<MODEL_ID>",
    "messages": [{"role":"user","content":"hello"}],
    "max_tokens": 64
  }'

curl -s http://<rank0-host>:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<MODEL_ID>",
    "prompt": "Write 5 sentences about Thunderbolt RDMA.",
    "max_tokens": 128
  }'
```

Stop:

```bash
./scripts/stop_openai_cluster_server.sh
```

### Server configuration

The server script accepts these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | (required) | Path to the MLX model directory |
| `HOSTFILE` | `hostfiles/hosts.json` | Path to cluster hostfile |
| `MODEL_ID` | basename of `MODEL_DIR` | Model identifier for API responses |
| `ENV_NAME` | `mlxjccl` | Conda environment name |
| `HTTP_HOST` | `0.0.0.0` | HTTP server bind address |
| `HTTP_PORT` | `8080` | HTTP server port |
| `CTRL_HOST` | auto-detect | Coordinator IP (rank0 LAN IP) |
| `CTRL_PORT` | `18080` | Coordinator port |
| `QUEUE_MAX` | `8` | Max queued requests |
| `REQ_TIMEOUT` | `120` | Request timeout in seconds |

### MLX environment variables

These environment variables are passed to all nodes via `mlx.launch --env`:

| Variable | Description |
|----------|-------------|
| `MLX_METAL_FAST_SYNCH=1` | **Critical for performance.** Enables fast Metal synchronization. Without this, you may see 5-6x slower inference speeds. |
| `HF_HUB_OFFLINE=1` | **Prevents automatic model downloads.** See below. |
| `TRANSFORMERS_OFFLINE=1` | **Prevents automatic model downloads.** See below. |

#### Why use offline mode?

The `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` flags prevent HuggingFace from automatically downloading models. This is important for distributed clusters because:

1. **All nodes would download simultaneously** — wasteful and slow
2. **Nodes may have different network access** — some might fail while others succeed
3. **Race conditions** — nodes may end up with inconsistent model states
4. **Unpredictable startup times** — downloading large models can take a long time

**Best practice:** Download models once on rank0, then sync to all nodes:

```bash
# 1. Download on rank0
python -c "from mlx_lm import load; load('mlx-community/Qwen3-4B-Instruct-2507-4bit')"

# 2. Sync to other nodes (reads hosts from your hostfile)
MODEL_PATH=~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit
OTHER_HOSTS=$(python3 -c "import json; print(' '.join(h['ssh'] for h in json.load(open('hostfiles/hosts.json'))[1:]))")
for h in $OTHER_HOSTS; do
  rsync -a --progress "$MODEL_PATH/" "$h:$MODEL_PATH/"
done
```

Or use a shared model directory that exists on all nodes at the same path.

---

## Notes

- `mlx_lm.server` is single-host; this repo's server runs rank0 HTTP while all ranks participate in sharded compute.
- For 4 nodes, JACCL requires a **fully connected Thunderbolt mesh** (6 cables).
- RDMA must be enabled in **macOS Recovery** (`rdma_ctl enable`) and verified via `ibv_devices`.

---

## License

MIT
