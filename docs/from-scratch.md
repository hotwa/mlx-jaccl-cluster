# From-scratch: 4-Mac MLX JACCL (Thunderbolt RDMA) Cluster

This guide sets up a 4‑Mac, fully connected Thunderbolt mesh using **MLX JACCL** (RDMA over Thunderbolt) and runs distributed jobs via `mlx.launch --backend jaccl`.

---

## 0) Hardware topology

For **4 nodes**, JACCL requires a **fully connected mesh**:

- 6 Thunderbolt cables total (every pair directly connected)

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

## 2) Create the conda env and install MLX

Do this on **each** Mac:

```bash
conda create -n mlxjccl python=3.12 -y
conda activate mlxjccl

python -m pip install -U pip setuptools wheel
python -m pip install -U "mlx>=0.30.4" "mlx-lm==0.30.5" fastapi uvicorn
python -m pip install -U "transformers==5.0.0rc3" tokenizers mistral_common
```

Verify:

```bash
python -m pip show mlx mlx-lm transformers | egrep "Name|Version"
mlx.distributed_config -h | grep -i jaccl || true
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

Copy the template:

```bash
cp hostfiles/m3-ultra-jaccl.template.json hostfiles/m3-ultra-jaccl.local.json
```

Edit `hostfiles/m3-ultra-jaccl.local.json`:

- set `ssh` hostnames (e.g. `macstudio1.local` …)
- set rank0 `"ips": ["<rank0_lan_ip>"]`
- keep the `rdma` matrix consistent with your wiring

> `hostfiles/*.local.json` is ignored by git.

---

## 5) Verify the cluster

```bash
scripts/verify_cluster.sh
```

---

## 6) Ensure the model exists on all nodes

If you want to run a model from a local folder, the same path must exist on every node.

Example:

```bash
MODEL_DIR=/Users/alex/models_mlx/mlx-community/Qwen3-4B-Instruct-2507-4bit

for h in macstudio1.local macstudio2.local macstudio3.local macstudio4.local; do
  ssh "$h" "test -f '$MODEL_DIR/model.safetensors' && echo OK || echo MISSING"
done
```

Copy (rank0 → others) if needed:

```bash
for h in macstudio2.local macstudio3.local macstudio4.local; do
  ssh "$h" "mkdir -p "$(dirname "$MODEL_DIR")""
  rsync -a --progress -e ssh "$MODEL_DIR/" "$h:$MODEL_DIR/"
done
```

---

## 7) Run the distributed tokens/sec benchmark

```bash
/Users/alex/miniconda3/bin/conda run -n mlxjccl mlx.launch --verbose --backend jaccl \
  --hostfile hostfiles/m3-ultra-jaccl.local.json \
  --env MLX_METAL_FAST_SYNCH=1 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 -- \
  scripts/jaccl_tps_bench.py \
  --model "$MODEL_DIR" \
  --prompt "Write 5 sentences about Thunderbolt RDMA." \
  --max-tokens 256
```

Rank0 prints tokens/sec.

---

## 8) Run the OpenAI-compatible server

Start:

```bash
scripts/run_openai_cluster_server.sh
```

Stop:

```bash
scripts/stop_openai_cluster_server.sh
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

## 9) Troubleshooting

### Curl hangs forever
For sharded distributed inference, **all ranks must enter `generate()` per request**.

- Confirm all nodes are running the server (rank0 + workers)
- Confirm the server control-plane port is reachable (`CTRL_PORT`)

### Unexpected HF downloads
Pass offline env vars via `mlx.launch --env`:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

### Stop stuck runs (no reboot)

```bash
scripts/stop_openai_cluster_server.sh
for h in macstudio1.local macstudio2.local macstudio3.local macstudio4.local; do
  ssh "$h" 'pkill -f "python.*-m mlx_lm" || true'
done
```
