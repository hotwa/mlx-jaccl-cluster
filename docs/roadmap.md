# Roadmap — mlx-jaccl-cluster

> Living document. Last updated: 2025-07-14
>
> This document sketches what we have, what we're missing relative to exo,
> and the phased plan to close the gap — dashboard v2, tool support,
> model management, observability, and beyond.

---

## Table of Contents

- [1. Current State](#1-current-state)
- [2. Gap Analysis vs exo](#2-gap-analysis-vs-exo)
- [3. Priority Matrix](#3-priority-matrix)
- [4. Phase 1 — Observability & Monitoring](#4-phase-1--observability--monitoring)
- [5. Phase 2 — Dashboard v2](#5-phase-2--dashboard-v2)
- [6. Phase 3 — Model Management](#6-phase-3--model-management)
- [7. Phase 4 — Tool Support & Structured Output](#7-phase-4--tool-support--structured-output)
- [8. Phase 5 — API Parity & Ecosystem](#8-phase-5--api-parity--ecosystem)
- [9. Phase 6 — Advanced Inference](#9-phase-6--advanced-inference)
- [10. Non-Goals](#10-non-goals)
- [11. Architecture Decisions](#11-architecture-decisions)
- [12. Dashboard v2 — Wireframe](#12-dashboard-v2--wireframe)
- [13. Implementation Notes](#13-implementation-notes)
- [14. Dependency Budget](#14-dependency-budget)
- [15. Success Criteria](#15-success-criteria)

---

## 1. Current State

### What Works Today (v0.1)

| Area | Status | Details |
|---|---|---|
| RDMA transport | ✅ Production-ready | 8.05 GB/s peak, 25.5 µs latency, stress-tested |
| Tensor-parallel inference | ✅ Working | `mlx_lm.sharded_load` across 2 nodes |
| OpenAI-compatible API | ✅ Working | `/v1/chat/completions`, `/v1/completions`, streaming SSE |
| Dashboard | ✅ Basic | HTMX+SSE, tok/s sparkline, queue depth, chat UI |
| Cluster tooling | ✅ Solid | Makefile, setup, bootstrap, verify, sync, benchmarks |
| RDMA test suite | ✅ Comprehensive | Correctness, latency, bandwidth, stress modes |
| Documentation | ✅ Good | Quickstart, from-scratch, comparison, scripts reference |

### What's Missing (Honest Assessment)

| Area | Status | Impact |
|---|---|---|
| Live RAM / memory monitoring | ❌ | Can't see if we're approaching OOM during inference |
| Live RDMA link health | ❌ | Static label only — no real-time bandwidth probe |
| Model download & management | ❌ | Manual `huggingface-cli` + `rsync` every time |
| Tool calls / function calling | ❌ | Can't use with agents, LangChain, OpenAI SDK tools |
| Structured output / JSON mode | ❌ | No `response_format` support |
| KV prefix cache | ❌ | Every request re-processes the full prompt |
| Ollama API compatibility | ❌ | Can't use with Ollama-native clients |
| Multi-model serving | ❌ | One model per server instance |
| Request logging / tracing | ❌ | No persistent logs, no request history |
| Prometheus / Grafana export | ❌ | No standard metrics format |
| Node failure detection | ❌ | Worker disconnect = silent hang |
| Image generation | ❌ | No Flux / image pipeline |

---

## 2. Gap Analysis vs exo

Detailed feature-by-feature comparison showing what exo has, what we have, and whether closing the gap makes sense for our use case.

### Server / API Layer

| Feature | exo | Us (v0.1) | Gap | Priority |
|---|---|---|---|---|
| `/v1/chat/completions` | ✅ | ✅ | — | — |
| `/v1/completions` | ✅ | ✅ | — | — |
| SSE streaming | ✅ | ✅ | — | — |
| Tool calls / function calling | ✅ | ❌ | **Big** | 🔴 High |
| Structured output (`response_format`) | ✅ | ❌ | **Big** | 🔴 High |
| `temperature`, `top_p`, `top_k` | ✅ | ❌ | Medium | 🟡 Medium |
| `stop` sequences | ✅ | ❌ | Medium | 🟡 Medium |
| `n` (multiple completions) | ✅ | ❌ | Small | 🟢 Low |
| `logprobs` | ✅ | ❌ | Small | 🟢 Low |
| Token usage in streaming | ✅ | ❌ | Medium | 🟡 Medium |
| Ollama `/api/generate` | ✅ | ❌ | Medium | 🟡 Medium |
| Ollama `/api/chat` | ✅ | ❌ | Medium | 🟡 Medium |
| `/v1/embeddings` | ❌ | ❌ | — | Future |

### Dashboard / Observability

| Feature | exo | Us (v0.1) | Gap | Priority |
|---|---|---|---|---|
| Live tok/s + sparkline | ✅ | ✅ | — | — |
| Cluster topology table | ✅ | ✅ | — | — |
| Chat UI (streaming) | ✅ | ✅ | — | — |
| Queue depth indicator | ✅ via inference | ✅ | — | — |
| **RAM / unified memory usage** | ❌ | ❌ | **Both miss** | 🔴 High |
| **Live RDMA bandwidth probe** | ❌ | ❌ static | **We should own this** | 🔴 High |
| **Per-node GPU memory** | ❌ | ❌ | **Both miss** | 🔴 High |
| D3.js topology graph | ✅ animated | ❌ | Medium | 🟡 Medium |
| Model download progress | ✅ | ❌ | Medium | 🟡 Medium |
| Token heatmap / attention | ✅ | ❌ | Small | 🟢 Low |
| Generation traces | ✅ | ❌ | Small | 🟢 Low |
| Error log viewer | ❌ | ❌ | Medium | 🟡 Medium |
| Request history table | ❌ | ❌ | Medium | 🟡 Medium |
| Prometheus `/metrics` | ❌ | ❌ | Medium | 🟡 Medium |

### Model Management

| Feature | exo | Us (v0.1) | Gap | Priority |
|---|---|---|---|---|
| Built-in model download | ✅ coordinator | ❌ manual CLI | **Big** | 🔴 High |
| Download progress tracking | ✅ per-shard | ❌ | **Big** | 🔴 High |
| Auto-sync to all nodes | ✅ via tasks | ❌ manual rsync | Medium | 🟡 Medium |
| Model registry / list | ✅ in code | ❌ | Medium | 🟡 Medium |
| Hot model swap | ❌ | ❌ | — | Future |

### Infrastructure / Operations

| Feature | exo | Us (v0.1) | Gap | Priority |
|---|---|---|---|---|
| Auto-discovery (libp2p) | ✅ | ❌ explicit | **Not a gap** | — |
| Leader election | ✅ | ❌ | Not needed (2 nodes) | — |
| Node health monitoring | Partial | ❌ | Medium | 🟡 Medium |
| Worker disconnect detection | ✅ | ❌ | Medium | 🟡 Medium |
| Graceful shutdown | Partial | ❌ | Medium | 🟡 Medium |
| Server logs (persistent) | ❌ | ❌ | Medium | 🟡 Medium |
| CI / automated tests | Partial | ❌ | Medium | 🟡 Medium |

### Things We Do Better (Keep / Protect)

| Advantage | Details |
|---|---|
| **RDMA actually works** | 8.05 GB/s proven; exo's auto-mapping is broken on M4 Pro |
| **Deterministic startup** | No race conditions, no election timeouts |
| **Zero build toolchain** | No Rust, no Node.js, no npm, no Swift |
| **Debuggable** | ~2K lines; any failure is traceable in minutes |
| **Stock MLX** | Uses official `mlx` from PyPI — no custom forks |
| **3-minute setup** | `make setup` → `make rdma-test` → `make server` |
| **Explicit configuration** | Hostfile gives full control; no magic |

---

## 3. Priority Matrix

Quadrant view — **Impact** (value to daily use) vs **Effort** (implementation complexity).

```
                          HIGH IMPACT
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           │  RAM monitoring  │  Tool calls      │
           │  RDMA live probe │  Structured out   │
           │  Model download  │  Ollama compat    │
           │  Worker health   │  KV prefix cache  │
           │                  │                  │
LOW EFFORT ├──────────────────┼──────────────────┤ HIGH EFFORT
           │                  │                  │
           │  Sampling params │  D3 topology      │
           │  Stop sequences  │  Token heatmap    │
           │  Error log view  │  Image generation  │
           │  Request history │  Pipeline parallel │
           │  Prometheus      │  Multi-model       │
           │                  │                  │
           └──────────────────┼──────────────────┘
                              │
                          LOW IMPACT
```

**Do first** (top-left): High impact, low effort — observability, model management
**Do next** (top-right): High impact, high effort — tool calls, Ollama, KV cache
**Do later** (bottom-left): Low impact, low effort — sampling params, logging
**Probably never** (bottom-right): Low impact, high effort — image gen, pipeline parallel

---

## 4. Phase 1 — Observability & Monitoring

> **Goal:** See everything happening in the cluster in real time.
> **Effort:** ~2–3 days. **Impact:** Transforms daily operations.

### 4.1 Live RAM / Unified Memory Monitoring

**Problem:** We have 48 GB unified memory per node but no visibility into usage during inference. A large model + long context can silently approach OOM and crash.

**Design:**

```
┌─────────────────────────────────────────────────────────┐
│  Memory (rank 0 — mac.home)                             │
│  ┌───────────────────────────────────────┐  37.4 / 48 GB│
│  │████████████████████████████░░░░░░░░░░░│  78% used     │
│  └───────────────────────────────────────┘              │
│  Model: 14.2 GB │ KV cache: 2.1 GB │ OS: 21.1 GB       │
│                                                         │
│  Memory (rank 1 — mac2)                                 │
│  ┌───────────────────────────────────────┐  36.8 / 48 GB│
│  │███████████████████████████░░░░░░░░░░░░│  77% used     │
│  └───────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

**Data source — per-node memory probe:**

```python
# Runs on every rank, reports to rank 0 via control-plane
import mlx.core as mx
import resource

def memory_snapshot() -> dict:
    info = mx.device_info()
    return {
        "total_gb": round(info["memory_size"] / (1024**3), 1),
        "working_set_gb": round(info["max_recommended_working_set_size"] / (1024**3), 1),
        "cache_gb": round(mx.metal.get_cache_memory() / (1024**3), 2),
        "active_gb": round(mx.metal.get_active_memory() / (1024**3), 2),
        "peak_gb": round(mx.metal.get_peak_memory() / (1024**3), 2),
        "rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3), 2),
    }
```

**Transport:** Workers periodically send memory snapshots to rank 0 over the existing TCP control-plane (new message type: `{"type": "metrics", "memory": {...}}`). No new connections needed.

**Dashboard integration:** New panel in dashboard, updated via SSE every 2 seconds. Color-coded bars (green < 70%, yellow 70–85%, red > 85%).

### 4.2 Live RDMA Link Health Probe

**Problem:** The dashboard shows a static "~8 GB/s" label. We need to know if the link is degraded or down.

**Design:**

```
┌─────────────────────────────────────────────────────────┐
│  RDMA Link                                              │
│  rdma_en4 ←→ rdma_en4                                  │
│  ┌────────────────────────────────┐                     │
│  │████████████████████████████████│  7.92 GB/s          │
│  └────────────────────────────────┘                     │
│  Latency: 26.1 µs │ Status: ● ACTIVE │ Last check: 3s  │
└─────────────────────────────────────────────────────────┘
```

**Implementation approach — lightweight background probe:**

- A background thread on rank 0 performs a small `all_sum` (e.g., 4 KB) every 10 seconds
- Measures round-trip latency
- Every 60 seconds, performs a larger probe (4 MB) to estimate bandwidth
- Reports results to the dashboard via the existing `MetricsStore`
- Does NOT interfere with inference (runs only when queue is empty)

**Key constraint:** The probe must NOT run during active generation. Use a lock shared with the `_queue_worker` to ensure mutual exclusion.

```python
# Pseudocode for RDMA health probe
class RDMAProbe:
    def __init__(self, world):
        self.world = world
        self.last_bw_gbps = 0.0
        self.last_latency_us = 0.0
        self.link_active = True
        self._generation_lock = asyncio.Lock()  # shared with queue_worker

    async def probe_latency(self):
        """4-byte all_sum — measures pure round-trip."""
        async with self._generation_lock:
            x = mx.ones(1)
            t0 = time.perf_counter()
            mx.distributed.all_sum(x)
            mx.eval(x)
            self.last_latency_us = (time.perf_counter() - t0) * 1e6

    async def probe_bandwidth(self):
        """4 MB all_sum — estimates sustained bandwidth."""
        async with self._generation_lock:
            x = mx.ones(1_048_576)  # 4 MB float32
            t0 = time.perf_counter()
            mx.distributed.all_sum(x)
            mx.eval(x)
            elapsed = time.perf_counter() - t0
            self.last_bw_gbps = (4.0 / 1024) / elapsed  # GB/s
            del x
            mx.clear_cache()
```

### 4.3 Worker Health & Disconnect Detection

**Problem:** If a worker process dies or the TB cable is unplugged, rank 0 hangs forever in `rank0_wait_done()`.

**Design:**

- TCP control-plane sockets get a heartbeat: workers send `{"type": "heartbeat"}` every 5 seconds
- Rank 0 tracks last heartbeat time per worker
- If no heartbeat for 15 seconds → mark worker as `DISCONNECTED`
- Dashboard shows per-node status: `ACTIVE` / `DEGRADED` / `DISCONNECTED`
- On disconnect, queued requests get a 503 error instead of hanging forever

### 4.4 Request History & Error Log

**Problem:** No way to see past requests, errors, or debug failed generations.

**Design:**

- Ring buffer of last 200 requests with: timestamp, kind, prompt (truncated), tokens, latency, status
- Ring buffer of last 50 errors with: timestamp, error type, message, traceback
- Exposed via:
  - `GET /requests` → JSON array of recent requests
  - `GET /errors` → JSON array of recent errors
  - Dashboard panel with scrollable table

### 4.5 Prometheus Metrics Export

**Design:**

```
GET /metrics

# HELP mlx_cluster_requests_total Total inference requests
# TYPE mlx_cluster_requests_total counter
mlx_cluster_requests_total 1423

# HELP mlx_cluster_tokens_generated_total Total tokens generated
# TYPE mlx_cluster_tokens_generated_total counter
mlx_cluster_tokens_generated_total 182947

# HELP mlx_cluster_tokens_per_second Current tokens per second
# TYPE mlx_cluster_tokens_per_second gauge
mlx_cluster_tokens_per_second 62.3

# HELP mlx_cluster_queue_depth Current queue depth
# TYPE mlx_cluster_queue_depth gauge
mlx_cluster_queue_depth 2

# HELP mlx_cluster_memory_used_bytes Unified memory used per rank
# TYPE mlx_cluster_memory_used_bytes gauge
mlx_cluster_memory_used_bytes{rank="0"} 40265318400
mlx_cluster_memory_used_bytes{rank="1"} 39528046592

# HELP mlx_cluster_rdma_bandwidth_gbps Last measured RDMA bandwidth
# TYPE mlx_cluster_rdma_bandwidth_gbps gauge
mlx_cluster_rdma_bandwidth_gbps 7.92

# HELP mlx_cluster_rdma_latency_us Last measured RDMA latency
# TYPE mlx_cluster_rdma_latency_us gauge
mlx_cluster_rdma_latency_us 25.8
```

**No new dependency.** Plain text Prometheus exposition format is trivial to generate.

### 4.6 New Makefile Targets

```makefile
make status          # Full cluster status: nodes, memory, RDMA, queue, model
make logs            # Tail server logs (rank 0)
make monitor         # Watch mode: refresh status every 5s
make download MODEL=mlx-community/Qwen3-4B  # Download + sync model (Phase 3)
```

---

## 5. Phase 2 — Dashboard v2

> **Goal:** A dashboard that rivals exo's SvelteKit UI — but still zero build step.
> **Effort:** ~3–4 days. **Impact:** Professional-grade monitoring.

### 5.1 New Panels

The dashboard v2 adds these panels to the existing layout:

| Panel | Data Source | Update Frequency |
|---|---|---|
| **Memory gauges** (per node) | Worker heartbeats via control-plane | Every 2s |
| **RDMA link monitor** | Background probe | Every 10s (latency), 60s (bandwidth) |
| **Node health grid** | Worker heartbeats | Every 5s |
| **Request history table** | Ring buffer | On each request |
| **Error log** | Ring buffer | On error |
| **Model info card** | Static at startup | Once |
| **D3-lite topology** | Hostfile + RDMA probe | Every 10s |

### 5.2 Dashboard Layout (Wireframe)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ⚡ mlx-jaccl-cluster  │  Qwen3-4B-Instruct  │  ● Online  │  /docs   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─── Cluster Overview ──────────┐  ┌─── RDMA Link ──────────────────┐ │
│  │                               │  │                                │ │
│  │  Nodes: 2/2 online            │  │  ● ACTIVE — rdma_en4 ↔ rdma_en4│
│  │  Model: Qwen3-4B (4-bit)      │  │  Bandwidth: 7.92 GB/s          │ │
│  │  World size: 2                │  │  Latency:   26.1 µs            │ │
│  │  Uptime: 2h 14m               │  │  Last probe: 3s ago            │ │
│  │  Total requests: 1,423        │  │                                │ │
│  │  Total tokens: 182,947        │  │  ┌──────────────────────────┐  │ │
│  │                               │  │  │████████████████████████░░│  │ │
│  └───────────────────────────────┘  │  └──────────────────────────┘  │ │
│                                     │  99% of theoretical max        │ │
│  ┌─── Memory (rank 0) ──────────┐  └────────────────────────────────┘ │
│  │  ┌──────────────────────┐    │                                     │
│  │  │███████████████████░░░│    │  ┌─── Performance ─────────────────┐ │
│  │  └──────────────────────┘    │  │                                 │ │
│  │  37.4 / 48 GB  (78%)        │  │  Avg tok/s (60s): 62.3          │ │
│  │  Active: 14.2 │ Cache: 2.1  │  │  Peak tok/s:      71.8          │ │
│  │  Peak: 16.3 GB              │  │  Avg latency:     4.12s         │ │
│  └──────────────────────────────┘  │  Queue: 1/8  ██░░░░░░           │ │
│                                     │                                 │ │
│  ┌─── Memory (rank 1) ──────────┐  │  ┌───── tok/s sparkline ─────┐ │ │
│  │  ┌──────────────────────┐    │  │  │  ╱╲   ╱╲  ╱╲             │ │ │
│  │  │██████████████████░░░░│    │  │  │ ╱  ╲_╱  ╲╱  ╲_╱╲         │ │ │
│  │  └──────────────────────┘    │  │  └───────────────────────────┘ │ │
│  │  36.8 / 48 GB  (77%)        │  └─────────────────────────────────┘ │
│  │  Active: 14.0 │ Cache: 2.0  │                                     │
│  └──────────────────────────────┘                                     │
│                                                                         │
│  ┌─── Topology ─────────────────────────────────────────────────────┐  │
│  │                                                                   │  │
│  │   ┌──────────┐          rdma_en4          ┌──────────┐           │  │
│  │   │  rank 0  │  ◄════════════════════►   │  rank 1  │           │  │
│  │   │  mac.home│        8.05 GB/s           │  mac2    │           │  │
│  │   │  coord   │        25.5 µs             │  worker  │           │  │
│  │   └──────────┘                            └──────────┘           │  │
│  │                                                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─── Request History ──────────────────────────────────────────────┐  │
│  │  Time       │ Kind │ Tokens │ Latency │ tok/s │ Status           │  │
│  │─────────────┼──────┼────────┼─────────┼───────┼──────────────────│  │
│  │  14:23:01   │ chat │ 128    │ 2.06s   │ 62.1  │ ✅ ok            │  │
│  │  14:22:45   │ chat │ 256    │ 4.12s   │ 62.1  │ ✅ ok            │  │
│  │  14:22:30   │ cmpl │  64    │ 1.03s   │ 62.1  │ ✅ ok            │  │
│  │  14:21:58   │ chat │  32    │ 0.52s   │ 61.5  │ ✅ ok            │  │
│  │  14:21:12   │ chat │ 512    │ 8.31s   │ 61.6  │ ⚠️ slow          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─── Chat ──────────────────────────────────────────────────────────┐ │
│  │  (existing chat UI — keep as-is)                                  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 D3-lite Topology (No D3.js Dependency)

Instead of importing D3.js (which exo does), we draw the topology with **pure SVG** updated via HTMX/SSE:

- Nodes rendered as rounded rectangles
- RDMA links as animated dashed lines (CSS animation, no JS library)
- Link color: green = healthy, yellow = degraded, red = down
- Bandwidth label on the link, updated from the RDMA probe
- Works with 2-node and 4-node topologies (reads from hostfile)

### 5.4 SSE Event Schema (v2)

Current SSE pushes a flat JSON blob. v2 adds structured sections:

```json
{
  "uptime_s": 8040,
  "total_requests": 1423,
  "total_tokens": 182947,
  "avg_tps_60s": 62.3,
  "peak_tps_60s": 71.8,
  "avg_latency_60s": 4.12,
  "queue_size": 1,
  "queue_max": 8,
  "history": [ ... ],

  "memory": {
    "0": { "active_gb": 14.2, "cache_gb": 2.1, "peak_gb": 16.3, "total_gb": 48.0 },
    "1": { "active_gb": 14.0, "cache_gb": 2.0, "peak_gb": 16.1, "total_gb": 48.0 }
  },
  "rdma": {
    "bandwidth_gbps": 7.92,
    "latency_us": 26.1,
    "link_active": true,
    "last_probe_s": 3
  },
  "nodes": {
    "0": { "status": "active", "hostname": "mac.home", "last_heartbeat_s": 0 },
    "1": { "status": "active", "hostname": "mac2", "last_heartbeat_s": 2 }
  }
}
```

---

## 6. Phase 3 — Model Management

> **Goal:** Download, sync, and manage models without leaving the terminal (or the dashboard).
> **Effort:** ~2–3 days. **Impact:** Eliminates the most tedious manual step.

### 6.1 `make download` Target

```bash
# Download a model from HuggingFace and sync to all nodes
make download MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit

# Download to a custom directory
make download MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit MODELS_DIR=~/models_mlx

# List downloaded models
make models-local
```

**Implementation:**

```bash
# scripts/download_model.sh
# 1. huggingface-cli download $MODEL --local-dir $MODELS_DIR/$MODEL_NAME
# 2. For each node in hostfile (except rank 0):
#      ssh $node "mkdir -p $MODELS_DIR"
#      rsync -avz --progress $LOCAL_PATH/ $node:$MODELS_DIR/$MODEL_NAME/
# 3. Verify all nodes have the model (checksum on config.json)
```

### 6.2 Model Registry

A simple JSON file tracking downloaded models:

```json
// ~/.mlx-jaccl-cluster/models.json
{
  "models": [
    {
      "id": "Qwen3-4B-Instruct-2507-4bit",
      "source": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
      "path": "/Users/omar/models_mlx/Qwen3-4B-Instruct-2507-4bit",
      "size_gb": 2.4,
      "downloaded_at": "2025-07-14T10:23:00Z",
      "synced_nodes": ["mac.home", "mac2"],
      "quantization": "4-bit"
    }
  ]
}
```

### 6.3 Dashboard Model Manager Panel

```
┌─── Models ─────────────────────────────────────────────────────────────┐
│                                                                         │
│  ● Active: Qwen3-4B-Instruct-2507-4bit (4-bit, 2.4 GB)               │
│                                                                         │
│  Downloaded:                                                            │
│  ┌────────────────────────────────┬──────┬───────┬──────────────────┐  │
│  │ Model                          │ Size │ Quant │ Synced           │  │
│  ├────────────────────────────────┼──────┼───────┼──────────────────┤  │
│  │ Qwen3-4B-Instruct-2507-4bit   │ 2.4G │ 4-bit │ ✅ 2/2 nodes    │  │
│  │ Llama-3.1-8B-Instruct-4bit    │ 4.5G │ 4-bit │ ✅ 2/2 nodes    │  │
│  │ Mistral-7B-v0.3-4bit          │ 3.8G │ 4-bit │ ⚠️ 1/2 nodes    │  │
│  └────────────────────────────────┴──────┴───────┴──────────────────┘  │
│                                                                         │
│  [Download New Model]  input: ______________________________  [Go]     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Download Progress via SSE

When a download is in progress, the dashboard shows a progress bar:

```
Downloading: mlx-community/Qwen3-8B-Instruct-4bit
┌──────────────────────────────────────────────┐
│██████████████████████████░░░░░░░░░░░░░░░░░░░│ 62% — 1.5 / 2.4 GB — 45 MB/s
└──────────────────────────────────────────────┘
Syncing to mac2... waiting
```

**Implementation:** A background asyncio task wraps `huggingface-cli download` subprocess, parses progress from stderr, pushes updates via a new SSE event type.

---

## 7. Phase 4 — Tool Support & Structured Output

> **Goal:** Support OpenAI function calling and JSON mode so agents and LangChain work.
> **Effort:** ~4–5 days. **Impact:** Unlocks the agent/tool ecosystem.

### 7.1 What Tool Calls Look Like

OpenAI tool calling request:

```json
{
  "model": "Qwen3-4B",
  "messages": [
    {"role": "user", "content": "What's the weather in Paris?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

Expected response:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"Paris\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

### 7.2 Implementation Plan

**Step 1: Prompt formatting.** Convert tools + messages into a prompt the model understands. Most instruct models (Qwen, Llama, Mistral) have specific chat templates for tool use that `tokenizer.apply_chat_template` already handles when `tools=` is passed.

```python
# In _build_chat_prompt(), add tools support:
def _build_chat_prompt(messages, tools=None):
    msgs = [{"role": m.role, "content": m.content} for m in messages]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    return _tok.apply_chat_template(msgs, **kwargs)
```

**Step 2: Response parsing.** After generation, detect if the output contains a tool call (model-specific format) and parse it into the OpenAI tool_calls structure.

```python
# Tool call detection (model-dependent patterns)
# Qwen3:   <tool_call>{"name": "...", "arguments": {...}}</tool_call>
# Llama:   <|python_tag|>{"name": "...", "parameters": {...}}
# Mistral: [TOOL_CALLS][{"name": "...", "arguments": {...}}]

def parse_tool_calls(text: str, model_family: str) -> list[dict] | None:
    """Extract tool calls from model output. Returns None if no tool call detected."""
    ...
```

**Step 3: Schema updates.** Extend `ChatCompletionsReq` and response schemas:

```python
class Tool(BaseModel):
    type: str = "function"
    function: dict  # {name, description, parameters}

class ChatCompletionsReq(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[str] = None  # "auto", "none", or {"type":"function","function":{"name":"..."}}
    response_format: Optional[dict] = None  # {"type": "json_object"}
```

**Step 4: Streaming tool calls.** SSE chunks must include `tool_calls` delta objects per the OpenAI spec.

### 7.3 Structured Output / JSON Mode

When `response_format: {"type": "json_object"}` is set:

1. Append `"Respond with valid JSON."` to the system prompt
2. After generation, validate that the output is valid JSON
3. If not valid, retry once with a stronger prompt
4. Return `finish_reason: "stop"` only if valid JSON

### 7.4 Sampling Parameters

Currently missing — easy wins:

| Parameter | Default | Notes |
|---|---|---|
| `temperature` | 1.0 | Pass to `generate()` / `stream_generate()` |
| `top_p` | 1.0 | Nucleus sampling |
| `top_k` | -1 | Top-k sampling (-1 = disabled) |
| `repetition_penalty` | 1.0 | Penalize repeated tokens |
| `stop` | `[]` | Stop sequences — check after each token |

All of these are already supported by `mlx_lm.generate()` — we just need to plumb them through from the HTTP request.

---

## 8. Phase 5 — API Parity & Ecosystem

> **Goal:** Drop-in replacement for more clients.
> **Effort:** ~3–4 days. **Impact:** Works with Ollama clients, LangChain, etc.

### 8.1 Ollama API Compatibility

Many tools (Open WebUI, Continue.dev, etc.) speak Ollama's API:

| Endpoint | Method | Description |
|---|---|---|
| `/api/generate` | POST | Text generation (Ollama format) |
| `/api/chat` | POST | Chat (Ollama format) |
| `/api/tags` | GET | List models |
| `/api/show` | POST | Model info |
| `/api/ps` | GET | Running models |

**Implementation:** Thin adapter layer that translates Ollama requests → our internal format → Ollama responses. ~200 lines.

### 8.2 Additional OpenAI Endpoints

| Endpoint | Effort | Notes |
|---|---|---|
| `/v1/embeddings` | Medium | Requires an embedding model or adapter |
| `/v1/models/{id}` | Trivial | Return model details |
| `/v1/chat/completions` with `n > 1` | Medium | Multiple completions per request |

### 8.3 Client SDK Compatibility Testing

Verify against:

- [ ] OpenAI Python SDK (`openai.ChatCompletion.create()`)
- [ ] OpenAI Node SDK
- [ ] LangChain (`ChatOpenAI`)
- [ ] LlamaIndex
- [ ] Continue.dev (VS Code)
- [ ] Open WebUI
- [ ] Cursor / Cody (via OpenAI-compatible endpoint)
- [ ] `curl` (already tested)

---

## 9. Phase 6 — Advanced Inference

> **Goal:** Performance and capability improvements.
> **Effort:** ~1–2 weeks per feature. **Impact:** Competitive with production inference servers.

### 9.1 KV Prefix Cache

**What it does:** Caches the key-value tensors for shared prompt prefixes. If 10 users ask questions with the same system prompt, the KV cache for that prefix is computed once.

**Impact:** Dramatic latency reduction for chat applications with long system prompts.

**Complexity:** High — requires modifying how we call `generate()` and managing a cache eviction policy. May need `mlx_lm` updates.

### 9.2 Continuous Batching

**What it does:** Instead of processing one request at a time (current behavior), interleave tokens from multiple requests.

**Impact:** Higher throughput under concurrent load. Currently our queue processes requests serially.

**Complexity:** Very high — requires rewriting the generation loop. The control-plane protocol would need significant changes since all ranks must agree on batch composition.

### 9.3 Speculative Decoding

**What it does:** Use a small draft model to propose tokens, then verify with the large model in parallel.

**Impact:** 2–3× speedup for large models.

**Complexity:** High — requires loading two models and coordinating draft/verify cycles across ranks.

---

## 10. Non-Goals

Things we deliberately choose NOT to implement:

| Feature | Reason |
|---|---|
| **Auto-discovery** | Explicit hostfile is simpler, more reliable, and correct for 1–4 node clusters |
| **Leader election** | Adds complexity; rank 0 is always the coordinator; 2-node clusters don't need it |
| **Image generation** | Different workload; use a dedicated tool (mflux, ComfyUI) |
| **SvelteKit dashboard** | Requires Node.js build toolchain; HTMX+SSE is sufficient and zero-build |
| **Custom MLX fork** | We use stock PyPI `mlx`; this is a core advantage |
| **Rust / Swift components** | Pure Python + Bash; zero build toolchain is a feature |
| **N > 4 node scaling** | JACCL requires fully connected TB mesh; 4 nodes = 6 cables, already impractical |
| **Multi-tenant isolation** | Single-user inference server; auth/isolation adds complexity for no benefit |

---

## 11. Architecture Decisions

### AD-01: Keep HTMX+SSE for Dashboard v2

**Context:** exo uses SvelteKit + D3.js for a richer dashboard.

**Decision:** Stay with HTMX + SSE + inline HTML/CSS/JS.

**Rationale:**
- Zero build step is a core project value
- HTMX can handle all planned features (memory bars, topology, tables)
- SVG can replace D3.js for the topology graph
- SSE is already working and battle-tested
- Adding Node.js + npm + Svelte contradicts our "zero toolchain" promise

### AD-02: Use Control-Plane for Metrics Transport

**Context:** Workers need to report memory/health to rank 0 for the dashboard.

**Decision:** Extend the existing TCP control-plane protocol with new message types (`metrics`, `heartbeat`).

**Rationale:**
- No new connections or ports needed
- Protocol is already framed JSON, easy to extend
- Workers already have an open socket to rank 0
- Alternative (HTTP from workers) would require each worker to run a server

### AD-03: Model Downloads via CLI, Not HTTP

**Context:** exo has a built-in `DownloadCoordinator` that downloads models via HTTP from a leader.

**Decision:** Use `huggingface-cli download` + `rsync` wrapped in a script.

**Rationale:**
- HuggingFace CLI handles auth, resume, checksums, LFS
- rsync is battle-tested for large file sync
- Building a download coordinator is high effort, low marginal value for 2 nodes
- Script approach is debuggable and composable

### AD-04: Tool Call Parsing is Model-Specific

**Context:** Different model families use different formats for tool calls.

**Decision:** Implement a pluggable parser with model-family detection.

**Rationale:**
- Qwen, Llama, and Mistral all use different tool call formats
- A single regex won't work
- Auto-detect model family from `config.json` or tokenizer config
- Start with Qwen3 (our primary model), add others incrementally

---

## 12. Dashboard v2 — Wireframe

### Mobile / Narrow Viewport

For access from phones or narrow windows, the grid collapses to single column:

```
┌─────────────────────────┐
│ ⚡ mlx-jaccl-cluster    │
│ Qwen3-4B │ ● Online     │
├─────────────────────────┤
│ Cluster Overview         │
│ 2/2 nodes │ 1,423 reqs  │
├─────────────────────────┤
│ Performance              │
│ 62.3 tok/s │ 4.12s lat  │
│ ▁▃▅▇▅▃▅▇▅▃ (sparkline) │
├─────────────────────────┤
│ RDMA: 7.92 GB/s ● UP    │
├─────────────────────────┤
│ Memory                   │
│ R0: ████████░░ 78%       │
│ R1: ███████░░░ 77%       │
├─────────────────────────┤
│ Queue: 1/8 ██░░░░░░      │
├─────────────────────────┤
│ Chat UI                  │
│ [message input]  [Send]  │
└─────────────────────────┘
```

### Navigation

No SPA routing needed. Single page with anchor links and collapsible sections:

```
[Overview] [Memory] [RDMA] [Requests] [Models] [Chat]
```

Each section is an HTMX fragment that auto-updates via SSE. No full page reloads.

---

## 13. Implementation Notes

### File Changes by Phase

**Phase 1 — Observability:**

| File | Change |
|---|---|
| `server/openai_cluster_server.py` | Add heartbeat protocol, memory probe, worker health tracking |
| `server/dashboard.py` | Add memory panel, RDMA panel, request history, error log |
| `server/rdma_probe.py` | **New** — Background RDMA health probe |
| `server/prometheus.py` | **New** — `/metrics` endpoint |
| `Makefile` | Add `status`, `logs`, `monitor` targets |
| `scripts/cluster_status.sh` | **New** — Full cluster status script |

**Phase 2 — Dashboard v2:**

| File | Change |
|---|---|
| `server/dashboard.py` | Major rewrite — new layout, panels, SSE v2 schema |
| `server/openai_cluster_server.py` | Pass new data sources to dashboard |

**Phase 3 — Model Management:**

| File | Change |
|---|---|
| `scripts/download_model.sh` | **New** — Download + sync script |
| `Makefile` | Add `download`, `models-local`, `models-sync` targets |
| `server/openai_cluster_server.py` | Add `/models/download` endpoint (optional) |
| `server/dashboard.py` | Add models panel |

**Phase 4 — Tool Support:**

| File | Change |
|---|---|
| `server/openai_cluster_server.py` | Tool calls in request/response, sampling params, stop sequences |
| `server/tool_parser.py` | **New** — Model-specific tool call parser |
| `server/schemas.py` | **New** — Extracted Pydantic models with tools support |

**Phase 5 — API Parity:**

| File | Change |
|---|---|
| `server/ollama_compat.py` | **New** — Ollama API adapter |
| `server/openai_cluster_server.py` | Mount Ollama routes |

### Control-Plane Protocol v2

Current message types:

```
→ worker→rank0:  {"type": "hello", "rank": N}
→ rank0→worker:  {"type": "task", "prompt": "...", "max_tokens": N}
→ worker→rank0:  {"type": "done", "rank": N}
```

v2 additions:

```
→ worker→rank0:  {"type": "heartbeat", "rank": N, "memory": {...}, "timestamp": T}
→ rank0→worker:  {"type": "config", "probe_interval_s": 5}
→ worker→rank0:  {"type": "metrics", "rank": N, "memory": {...}}
→ rank0→worker:  {"type": "shutdown"}  (graceful stop)
```

---

## 14. Dependency Budget

We're strict about dependencies. Every new package must justify itself.

### Current Dependencies (8 packages)

```
mlx >= 0.30.4
mlx-lm >= 0.30.5
fastapi >= 0.110.0
uvicorn[standard] >= 0.29.0
pydantic >= 2.0
transformers >= 4.50.0
tokenizers
mistral_common
huggingface_hub
```

### Planned Additions

| Package | Phase | Justification | Alternative Considered |
|---|---|---|---|
| `psutil` | Phase 1 | Cross-platform memory/CPU stats | `/proc` parsing (Linux only, we're macOS) |
| — | — | — | — |

That's it. **One new dependency** across all phases. Everything else is built with the standard library, MLX APIs, or inline code.

- Prometheus export: hand-written text format (no `prometheus_client`)
- Topology SVG: inline SVG (no D3.js)
- RDMA probe: `mx.distributed.all_sum` (already available)
- Download: `huggingface-cli` subprocess (already installed)

---

## 15. Success Criteria

### Phase 1 (Observability) — Done When:

- [ ] Dashboard shows live RAM usage per node (auto-refreshing)
- [ ] Dashboard shows live RDMA bandwidth and latency (from actual probes)
- [ ] Worker disconnect is detected within 15 seconds and shown on dashboard
- [ ] `make status` prints a complete cluster snapshot in the terminal
- [ ] `/metrics` returns valid Prometheus exposition format
- [ ] Request history is visible in dashboard (last 50 requests)

### Phase 2 (Dashboard v2) — Done When:

- [ ] Dashboard has all panels from the wireframe
- [ ] SVG topology graph shows nodes and RDMA links with live status
- [ ] Dashboard works on mobile (responsive layout)
- [ ] SSE pushes v2 schema with memory + RDMA + node data
- [ ] Zero build step maintained (no npm, no bundler)

### Phase 3 (Model Management) — Done When:

- [ ] `make download MODEL=...` downloads and syncs to all nodes
- [ ] `make models-local` lists all downloaded models with sizes
- [ ] Dashboard shows downloaded models and active model
- [ ] Download progress is visible (terminal and/or dashboard)

### Phase 4 (Tool Support) — Done When:

- [ ] OpenAI SDK `tools=` parameter works with Qwen3 models
- [ ] `response_format: {"type": "json_object"}` works
- [ ] `temperature`, `top_p`, `stop` parameters are plumbed through
- [ ] Streaming tool calls work per OpenAI spec
- [ ] LangChain `ChatOpenAI` with tools works against our server

### Phase 5 (API Parity) — Done When:

- [ ] Open WebUI connects via Ollama API and works for chat
- [ ] Continue.dev (VS Code) works with our server as OpenAI backend
- [ ] All Ollama core endpoints return valid responses

---

## Timeline (Estimated)

```
Week 1-2:   Phase 1 — Observability & Monitoring
Week 3-4:   Phase 2 — Dashboard v2
Week 5:     Phase 3 — Model Management
Week 6-7:   Phase 4 — Tool Support
Week 8:     Phase 5 — API Parity
Ongoing:    Phase 6 — Advanced Inference (opportunistic)
```

---

## Appendix: exo Features We're Deliberately Skipping

For completeness, these exo features are **not** on our roadmap and why:

| exo Feature | Lines of Code | Why We Skip It |
|---|---|---|
| libp2p auto-discovery | ~2,000 | Explicit hostfile is more reliable for 1–4 nodes |
| Raft leader election | ~1,500 | 2-node cluster doesn't need it |
| `DownloadCoordinator` | ~800 | `huggingface-cli` + `rsync` is simpler and more robust |
| Topology-aware shard placement | ~600 | 2-node = trivial placement |
| Flux image generation | ~400 | Use dedicated tools (mflux, ComfyUI) |
| Swift `SystemProfiler` integration | ~300 | `mlx.device_info()` gives us what we need |
| Nix flake | ~200 | `uv` + `pyproject.toml` is sufficient |
| Ring-buffer P2P topology | ~500 | JACCL handles topology via hostfile |
| Custom MLX fork maintenance | ongoing | Using stock PyPI mlx is a core advantage |

**Total lines we avoid maintaining: ~6,300+**

---

*This roadmap is a living document. Update it as phases are completed or priorities shift.*