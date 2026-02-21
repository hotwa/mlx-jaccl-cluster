# RDMA JACCL Cluster — Test & Command Reference

> All commands run from the repo root (`mlx-jaccl-cluster/`).
> Tests are ordered: hardware → RDMA → model → inference → server.

---

## Pre-flight

Keep mac2 awake so the Thunderbolt port doesn't drop:

```/dev/null/awake.sh#L1
ssh mac2 "nohup caffeinate -s -t 3600 >/dev/null 2>&1 & echo 'mac2 awake for 1h'"
```

Kill it when done:

```/dev/null/sleep.sh#L1
ssh mac2 "pkill caffeinate"
```

---

## 🧪 Test 1 — SSH + RDMA hardware presence

```/dev/null/t1.sh#L1
HOSTFILE=hostfiles/hosts-2node.json ./scripts/verify_cluster.sh
```

✅ Already passed. Tests: SSH reachability + `ibv_devices` on both nodes.

---

## 🧪 Test 2 — Full cluster alignment report

```/dev/null/t2.sh#L1
./scripts/cluster_info.sh
```

✅ Already passed. Tests: versions, RAM, chip, RDMA ports — side by side.

---

## 🧪 Test 3 — RDMA bandwidth (default: 20 rounds)

```/dev/null/t3.sh#L1-3
.venv/bin/mlx.launch --backend jaccl \
  --hostfile hostfiles/hosts-2node.json \
  --env MLX_METAL_FAST_SYNCH=1 -- scripts/rdma_test.py
```

✅ Already passed. 8 GB/s peak.

---

## 🧪 Test 4 — RDMA stress (100 rounds, verbose per-round timing)

```/dev/null/t4.sh#L1-4
RDMA_ROUNDS=100 RDMA_VERBOSE=1 \
  .venv/bin/mlx.launch --backend jaccl \
  --hostfile hostfiles/hosts-2node.json \
  --env MLX_METAL_FAST_SYNCH=1 -- scripts/rdma_test.py
```

100 rounds, prints every single round timing — good for spotting jitter or instability.

---

## 🧪 Test 5 — RDMA with bigger tensors (push the link harder)

```/dev/null/t5.sh#L1-5
RDMA_SIZES=1048576,16777216,67108864,134217728 \
RDMA_ROUNDS=10 \
  .venv/bin/mlx.launch --backend jaccl \
  --hostfile hostfiles/hosts-2node.json \
  --env MLX_METAL_FAST_SYNCH=1 -- scripts/rdma_test.py
```

Tensors: 4 MB, 64 MB, 256 MB, 512 MB. Tests sustained bandwidth under heavy load.
Safe — 512 MB × 3 = 1.5 GB peak, well under your 37 GB working set.

---

## 🧪 Test 6 — RDMA without `MLX_METAL_FAST_SYNCH` (compare the difference)

```/dev/null/t6.sh#L1-3
.venv/bin/mlx.launch --backend jaccl \
  --hostfile hostfiles/hosts-2node.json -- \
  scripts/rdma_test.py
```

Shows you exactly how much `MLX_METAL_FAST_SYNCH=1` matters. Expect 5–6× slower.

---

## 🧪 Test 7 — Verify nodes stay in sync

```/dev/null/t7.sh#L1
./scripts/sync_nodes.sh
```

Runs `git pull` on all nodes in parallel, then checks all are on the same commit.

---

## 🧪 Test 8 — Download a model and sync to all nodes

```/dev/null/t8.sh#L1-2
MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit make download
```

Downloads the model on mac.home, then rsyncs to mac2 automatically.
After it finishes, verify both nodes have it:

```/dev/null/t8-verify.sh#L1
MODEL_DIR=~/models_mlx/Qwen3-4B-Instruct-2507-4bit make models-check
```

**Alternatives (different model sizes):**

| Model | Size | Use Case |
|---|---|---|
| `mlx-community/Qwen3-0.6B-4bit` | ~0.4 GB | Tiny — fastest for testing |
| `mlx-community/Qwen3-4B-Instruct-2507-4bit` | ~2.4 GB | Sweet spot — good quality, fast |
| `mlx-community/Llama-3.1-8B-Instruct-4bit` | ~4.5 GB | Larger — more capable |
| `mlx-community/Qwen3-14B-4bit` | ~8 GB | Big — real use case for 2-node sharding |
| `mlx-community/Qwen3-30B-A3B-4bit` | ~17 GB | MoE — benefits from 2× memory pool |

---

## 🧪 Test 9 — Distributed tokens/sec benchmark

```/dev/null/t9.sh#L1
MODEL_DIR=~/models_mlx/Qwen3-4B-Instruct-2507-4bit make bench
```

Loads the model sharded across both nodes, runs warmup + timed generation, prints tok/s.

Custom prompt and token count:

```/dev/null/t9-custom.sh#L1-4
MODEL_DIR=~/models_mlx/Qwen3-4B-Instruct-2507-4bit \
BENCH_PROMPT="Explain RDMA over Thunderbolt in simple terms." \
BENCH_TOKENS=512 \
make bench
```

Expected output (rank 0 only):

```/dev/null/t9-output.txt#L1-6
==========
model=~/models_mlx/Qwen3-4B-Instruct-2507-4bit
world_size=2
prompt_tokens=8
gen_tokens=256
tokens_per_sec=62.0
```

---

## 🧪 Test 10 — Start the server

```/dev/null/t10.sh#L1
MODEL_DIR=~/models_mlx/Qwen3-4B-Instruct-2507-4bit make server
```

Starts the OpenAI-compatible API on port 8080, dashboard at `/dashboard`.
Wait until you see `all workers connected` and `Uvicorn running` in the output.

---

## 🧪 Test 11 — Server health + model listing

Run in a **separate terminal** while the server is running:

```/dev/null/t11.sh#L1-5
# Health check
make health

# List models
make models
```

Expected:

```/dev/null/t11-output.json#L1-7
{
  "ok": true,
  "world_size": 2,
  "rank": 0,
  "model": "Qwen3-4B-Instruct-2507-4bit",
  "queue_max": 8,
  "queue_size": 0
}
```

---

## 🧪 Test 12 — Chat completion (non-streaming)

```/dev/null/t12.sh#L1
make chat-test
```

Or with a custom message:

```/dev/null/t12-custom.sh#L1-4
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is RDMA and why is it fast?"}],"max_tokens":128}' \
  | python3 -m json.tool
```

Check the response includes `usage.completion_tokens`, `timing.tokens_per_sec`, and actual content.

---

## 🧪 Test 13 — Streaming chat (SSE)

```/dev/null/t13.sh#L1-4
curl -sN http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count from 1 to 20, one per line."}],"max_tokens":128,"stream":true}'
```

You should see `data: {...}` chunks arriving one by one, ending with `data: [DONE]`.
The `-N` flag disables buffering so you see tokens appear in real time.

---

## 🧪 Test 14 — Streaming completions (raw text, not chat)

```/dev/null/t14.sh#L1-4
curl -sN http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The fastest way to transfer data between two Macs is","max_tokens":128,"stream":true}'
```

Same SSE format, but uses the `/v1/completions` endpoint instead of chat.

---

## 🧪 Test 15 — Burst test (queue + backpressure)

Fire 5 requests in parallel to test the queue:

```/dev/null/t15.sh#L1-9
for i in 1 2 3 4 5; do
  curl -s http://localhost:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Say the number $i and nothing else.\"}],\"max_tokens\":16}" &
done
wait
echo "All 5 done"
```

While running, check the queue in another terminal:

```/dev/null/t15-queue.sh#L1
make queue
```

You should see `"size"` go from 0 → up to 4 → back to 0 as requests drain.

---

## 🧪 Test 16 — Dashboard (visual check)

```/dev/null/t16.sh#L1
make dashboard
```

Opens `http://localhost:8080/dashboard` in your browser. Verify:

- [ ] Cluster topology table shows rank 0 (coordinator) + rank 1 (worker)
- [ ] RDMA badge shows ~8 GB/s
- [ ] Tok/s sparkline updates after you send a chat
- [ ] Queue depth bar moves when requests are in flight
- [ ] Chat UI works — type a message, see streaming response

---

## 🧪 Test 17 — Server stop + restart

```/dev/null/t17.sh#L1-5
# Stop
make server-stop

# Restart
MODEL_DIR=~/models_mlx/Qwen3-4B-Instruct-2507-4bit make server
```

---

## 🧪 Test 18 — Full cluster status

```/dev/null/t18.sh#L1
make status
```

Shows nodes (online/offline + RDMA ports), server health, and local memory usage in one view.

---

## Suggested test order

### Phase A — RDMA (no model needed)

| # | Command | What you learn | Status |
|---|---|---|---|
| 1 | `make verify` | SSH + RDMA device presence | ✅ Passed |
| 2 | `make cluster-info` | Node alignment | ✅ Passed |
| 3 | `make rdma-test` | Bandwidth + correctness | ✅ 8.05 GB/s |
| 4 | Stress 100 rounds | Link stability over time | ⬜ |
| 5 | Bigger tensors | Peak sustained BW | ⬜ |
| 6 | No FAST_SYNCH | How much that flag matters | ⬜ |
| 7 | `make sync` | Git alignment | ⬜ |

### Phase B — Model setup

| # | Command | What you learn | Status |
|---|---|---|---|
| 8 | `make download MODEL=...` | Download + sync works | ⬜ |

### Phase C — Inference

| # | Command | What you learn | Status |
|---|---|---|---|
| 9 | `make bench` | Distributed tok/s | ⬜ |
| 10 | `make server` | Server starts, workers connect | ⬜ |
| 11 | `make health` + `make models` | API responds correctly | ⬜ |
| 12 | `make chat-test` | Non-streaming inference works | ⬜ |
| 13 | Streaming curl | SSE streaming works | ⬜ |
| 14 | Completions endpoint | Both API styles work | ⬜ |
| 15 | Burst test | Queue + backpressure | ⬜ |
| 16 | `make dashboard` | Dashboard renders + updates | ⬜ |
| 17 | Stop + restart | Clean lifecycle | ⬜ |
| 18 | `make status` | Full cluster overview | ⬜ |

---

## Troubleshooting quick reference

| Problem | Fix |
|---|---|
| `PORT_DOWN` on mac2 | Reseat cable on mac2, or run `ssh mac2 "nohup caffeinate -s -t 3600 >/dev/null 2>&1 &"` |
| `Couldn't allocate protection domain` | Kill stale processes: `make kill-all`, wait 3s, retry |
| `errno 60` (ETIMEDOUT) | One side is PORT_DOWN — check `ibv_devinfo` on both nodes |
| `errno 57` (ENOTCONN) | Stale process from a crashed run — `make kill-all`, wait, retry |
| Server curl hangs | Both ranks must be running — check server logs for `all workers connected` |
| Queue full (429) | Wait for requests to drain, or increase `QUEUE_MAX=16 make server` |
| Model not found | Run `MODEL_DIR=... make models-check` to verify all nodes have it |