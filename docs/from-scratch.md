# From-scratch: Multi-Mac MLX JACCL (Thunderbolt RDMA) Cluster

This guide sets up a multi‑Mac, fully connected Thunderbolt mesh using **MLX JACCL** (RDMA over Thunderbolt) and runs distributed jobs via `mlx.launch --backend jaccl`.

---

## 0) Hardware topology

For **4 nodes**, JACCL requires a **fully connected mesh**:

- 6 Thunderbolt cables total (every pair directly connected)

For **2 nodes**:

- 1 Thunderbolt cable directly between the two Macs

![Thunderbolt wiring diagram for 4-Mac cluster](images/figure1.jpeg)

---

## 1) Enable RDMA (one-time per Mac)

RDMA over Thunderbolt must be enabled locally in **macOS Recovery**:

1. Boot into Recovery
2. Open Terminal
3. Run:
   ```bash
   rdma_ctl enable
   ```
4. Reboot
5. Verify:
   ```bash
   ibv_devices
   ```

You should see `rdma_en*` devices (e.g. `rdma_en3`, `rdma_en4`, `rdma_en5`).

---

## 2) Install uv and set up the Python environment

This project uses [uv](https://github.com/astral-sh/uv) instead of conda — it is faster, lighter, and requires no extra tool beyond Homebrew.

Do this on **each** Mac:

### Install uv (if not already installed)

```bash
brew install uv
```

### Run the one-shot setup script

From the repo root:

```bash
./scripts/setup.sh
```

This will:
- Create a `.venv` virtualenv in the repo root (Python 3.12)
- Install all dependencies: `mlx`, `mlx-lm`, `fastapi`, `uvicorn`, `transformers`, `tokenizers`, `mistral_common`, `huggingface_hub`
- Verify all packages are importable
- Check for RDMA devices

### Manual install (alternative)

If you prefer to do it step by step:

```bash
# Create virtualenv
uv venv .venv --python 3.12

# Activate
source .venv/bin/activate

# Install dependencies
uv pip install "mlx>=0.30.4" "mlx-lm>=0.30.5"
uv pip install "fastapi>=0.110.0" "uvicorn[standard]>=0.29.0" "pydantic>=2.0"
uv pip install "transformers>=4.50.0" tokenizers mistral_common "huggingface_hub[cli]"
```

### Verify

```bash
source .venv/bin/activate
python -m pip show mlx mlx-lm transformers | grep -E "Name|Version"
```

---

## 3) Pick rank-0 coordinator IP (LAN)

JACCL uses RDMA for the data path, but needs a TCP coordinator address that all nodes can reach.

On rank0, prefer **Ethernet**:

```bash
ipconfig getifaddr en0
```

---

## 4) Create a JACCL hostfile

Copy the appropriate template:

```bash
# For 2 nodes:
cp hostfiles/hosts-2node.json hostfiles/hosts.json

# For 4 nodes:
cp hostfiles/hosts.json.example hostfiles/hosts.json
```

Edit `hostfiles/hosts.json`:

- set `ssh` hostnames (e.g. `mac1.local`, `mac2.local`, …)
- set rank0 `"ips": ["<rank0_lan_ip>"]`
- keep the `rdma` matrix consistent with your wiring (use `ibv_devices` to find device names)

> `hostfiles/hosts.json` is ignored by git.

---

## 5) Verify the cluster

```bash
HOSTFILE=hostfiles/hosts-2node.json ./scripts/verify_cluster.sh
```

This checks:
- SSH connectivity to each node
- RDMA devices present on each node (`ibv_devices`)

> Note: this does **not** send data over RDMA. It only checks SSH + device presence.

---

## 6) Test RDMA data transfer (no model needed)

Run the minimal RDMA test to confirm actual data flows over Thunderbolt between both Macs:

```bash
.venv/bin/mlx.launch --backend jaccl \
  --hostfile hostfiles/hosts-2node.json \
  --env MLX_METAL_FAST_SYNCH=1 -- \
  scripts/rdma_test.py
```

Expected output on rank0:
- **Phase 0**: barrier smoke test (all ranks reached barrier)
- **Phase 1**: correctness check on `all_sum` results
- **Phase 2**: latency of a 1-element all_sum in µs
- **Phase 3**: bandwidth sweep across tensor sizes with GB/s readings

A healthy TB5 RDMA link should show **> 5 GB/s** peak bandwidth.

Optional env vars:
```bash
RDMA_ROUNDS=50 RDMA_VERBOSE=1 \
  .venv/bin/mlx.launch --backend jaccl \
  --hostfile hostfiles/hosts-2node.json \
  --env MLX_METAL_FAST_SYNCH=1 -- \
  scripts/rdma_test.py
```

---

## 7) Download and sync the model to all nodes

The same model path must exist on every node. Download once on rank0, then sync to other nodes.

**Download a model from HuggingFace to a local directory:**

```bash
# Activate the venv first
source .venv/bin/activate

huggingface-cli download mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --local-dir ~/models_mlx/Qwen3-4B-Instruct-2507-4bit
```

**Sync to other nodes:**

```bash
# Replace paths and hostnames with your actual values
ssh mac2.local "mkdir -p ~/models_mlx"
rsync -avz -e ssh ~/models_mlx/Qwen3-4B-Instruct-2507-4bit/ \
  mac2.local:/Users/yourusername/models_mlx/Qwen3-4B-Instruct-2507-4bit/
```

**Verify all nodes have the model:**

```bash
HOSTS=$(python3 -c "import json; print(' '.join(h['ssh'] for h in json.load(open('hostfiles/hosts.json'))))")
for h in $HOSTS; do
  echo -n "$h: "
  ssh "$h" "test -d '$MODEL_DIR' && echo OK || echo MISSING"
done
```

> **Tip:** For large models (100GB+), copying via an external SSD is much faster than rsync over the network.

---

## 8) Run the distributed tokens/sec benchmark

```bash
.venv/bin/mlx.launch --verbose --backend jaccl \
  --hostfile hostfiles/hosts-2node.json \
  --env MLX_METAL_FAST_SYNCH=1 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 -- \
  scripts/jaccl_tps_bench.py \
  --model "$MODEL_DIR" \
  --prompt "Write 5 sentences about Thunderbolt RDMA." \
  --max-tokens 256
```

Rank0 prints `prompt_tokens`, `gen_tokens`, `seconds`, `tokens_per_sec`.

---

## 9) Run the OpenAI-compatible server

Start:

```bash
MODEL_DIR=~/models_mlx/your-model-name ./scripts/run_openai_cluster_server.sh
```

Or with custom settings:

```bash
MODEL_DIR=~/models_mlx/your-model-name \
HTTP_PORT=8000 \
HOSTFILE=hostfiles/hosts-2node.json \
./scripts/run_openai_cluster_server.sh
```

Stop:

```bash
./scripts/stop_openai_cluster_server.sh
```

Test:

```bash
curl -s http://<rank0-host>:8080/v1/models

curl -s http://<rank0-host>:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<MODEL_ID>","messages":[{"role":"user","content":"hello"}],"max_tokens":64}'

curl -s http://<rank0-host>:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<MODEL_ID>","prompt":"Hello","max_tokens":64}'
```

---

## 10) MLX environment variables

These environment variables are passed to all nodes via `mlx.launch --env`:

| Variable | Description |
|----------|-------------|
| `MLX_METAL_FAST_SYNCH=1` | **Critical for performance.** Enables fast Metal synchronization. Without this, you may see 5-6x slower inference speeds. |
| `HF_HUB_OFFLINE=1` | **Prevents automatic model downloads.** |
| `TRANSFORMERS_OFFLINE=1` | **Prevents automatic model downloads.** |

### Why use offline mode?

The `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` flags prevent HuggingFace from automatically downloading models. This is critical for distributed clusters because:

1. **All nodes would download simultaneously** — wasteful and slow
2. **Nodes may have different network access** — some might fail while others succeed
3. **Race conditions** — nodes may end up with inconsistent model states
4. **Unpredictable startup times** — downloading large models can take a long time

**Best practice:** Always download models once on rank0, then sync to all other nodes (see step 7 above), and run with offline mode enabled.

---

## 11) Troubleshooting

### Curl hangs forever

For sharded distributed inference, **all ranks must enter `generate()` per request**.

- Confirm all nodes are running the server (rank0 + workers)
- Confirm the server control-plane port is reachable (`CTRL_PORT`)

### RDMA test fails / low bandwidth

- Confirm `rdma_ctl enable` was run in Recovery on **both** Macs
- Run `ibv_devices` on each Mac — you must see `rdma_en*` entries
- Confirm the Thunderbolt cable is seated properly
- Try `MLX_METAL_FAST_SYNCH=1` — without it bandwidth will appear 5-6x lower

### Unexpected HF downloads

Pass offline env vars via `mlx.launch --env`:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

### Stop stuck runs (no reboot)

```bash
./scripts/stop_openai_cluster_server.sh

# If needed, also kill any other MLX processes:
HOSTS=$(python3 -c "import json; print(' '.join(h['ssh'] for h in json.load(open('hostfiles/hosts.json'))))")
for h in $HOSTS; do
  ssh "$h" 'pkill -f "python.*mlx" || true'
done
```

### Re-run setup after dependency changes

```bash
# Remove old venv and start fresh
rm -rf .venv
./scripts/setup.sh
```
