# MLX JACCL (Thunderbolt RDMA) Cluster + OpenAI-Compatible Server

This repo helps you stand up a multi‑Mac **MLX** cluster using **JACCL** (RDMA over Thunderbolt) and expose it via **OpenAI-compatible HTTP endpoints** from rank 0.

You get:

- **From-scratch runbook** (`docs/from-scratch.md`)
- JACCL **hostfile template** (`hostfiles/m3-ultra-jaccl.template.json`)
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
  - `m3-ultra-jaccl.template.json` — template hostfile (copy to a local file and edit)
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

### 2) Verify the cluster is reachable + RDMA is enabled

```bash
scripts/verify_cluster.sh
```

### 3) Create your real hostfile (not committed)

Copy the template:

```bash
cp hostfiles/m3-ultra-jaccl.template.json hostfiles/m3-ultra-jaccl.local.json
```

Edit `hostfiles/m3-ultra-jaccl.local.json`:

- set your `ssh` hostnames
- set rank 0 `"ips": ["<RANK0_LAN_IP>"]` (Ethernet recommended)
- confirm the `rdma` matrix matches your cabling

> `hostfiles/*.local.json` is ignored by git.

### 4) Run a distributed tokens/sec benchmark

```bash
MODEL_DIR=/path/to/local/mlx-model

/Users/alex/miniconda3/bin/conda run -n mlxjccl mlx.launch --backend jaccl   --hostfile hostfiles/m3-ultra-jaccl.local.json   --env MLX_METAL_FAST_SYNCH=1   --env HF_HUB_OFFLINE=1   --env TRANSFORMERS_OFFLINE=1 --   scripts/jaccl_tps_bench.py   --model "$MODEL_DIR"   --prompt "Write 5 sentences about Thunderbolt RDMA."   --max-tokens 256
```

Rank 0 prints:

- `prompt_tokens`, `gen_tokens`, `seconds`, `tokens_per_sec`

### 5) Start the OpenAI-compatible server (rank 0 HTTP)

```bash
scripts/run_openai_cluster_server.sh
```

Test:

```bash
curl -s http://<rank0-host>:8080/v1/models

curl -s http://<rank0-host>:8080/v1/chat/completions   -H 'Content-Type: application/json'   -d '{
    "model": "<MODEL_ID>",
    "messages": [{"role":"user","content":"hello"}],
    "max_tokens": 64
  }'

curl -s http://<rank0-host>:8080/v1/completions   -H 'Content-Type: application/json'   -d '{
    "model": "<MODEL_ID>",
    "prompt": "Write 5 sentences about Thunderbolt RDMA.",
    "max_tokens": 128
  }'
```

Stop:

```bash
scripts/stop_openai_cluster_server.sh
```

---

## Notes

- `mlx_lm.server` is single-host; this repo’s server runs rank0 HTTP while all ranks participate in sharded compute.
- For 4 nodes, JACCL requires a **fully connected Thunderbolt mesh** (6 cables).
- RDMA must be enabled in **macOS Recovery** (`rdma_ctl enable`) and verified via `ibv_devices`.

---

## License

Choose a license before sharing broadly (MIT/Apache-2.0 are common).
