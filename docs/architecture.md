# Architecture — mlx-jaccl-cluster

> Technical reference for contributors and operators.
> Last updated: 2025-07-14

---

## Table of Contents

- [1. System Overview](#1-system-overview)
- [2. Physical Topology](#2-physical-topology)
- [3. Software Stack](#3-software-stack)
- [4. Process Model](#4-process-model)
- [5. Data Plane — RDMA / JACCL](#5-data-plane--rdma--jaccl)
- [6. Control Plane — TCP Framed JSON](#6-control-plane--tcp-framed-json)
- [7. HTTP Layer — FastAPI + Uvicorn](#7-http-layer--fastapi--uvicorn)
- [8. Request Lifecycle](#8-request-lifecycle)
- [9. Dashboard — HTMX + SSE](#9-dashboard--htmx--sse)
- [10. Sharded Model Loading](#10-sharded-model-loading)
- [11. Streaming (SSE) Pipeline](#11-streaming-sse-pipeline)
- [12. Queue & Backpressure](#12-queue--backpressure)
- [13. Hostfile Format & RDMA Matrix](#13-hostfile-format--rdma-matrix)
- [14. Environment Variables](#14-environment-variables)
- [15. Startup Sequence (Detailed)](#15-startup-sequence-detailed)
- [16. Failure Modes & Recovery](#16-failure-modes--recovery)
- [17. Security Model](#17-security-model)
- [18. Performance Characteristics](#18-performance-characteristics)
- [19. File Map](#19-file-map)
- [20. Design Principles](#20-design-principles)

---

## 1. System Overview

mlx-jaccl-cluster is a distributed inference server for Apple Silicon Macs connected via Thunderbolt RDMA. It splits a large language model across multiple GPUs using tensor parallelism and exposes an OpenAI-compatible HTTP API from the coordinator node (rank 0).

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Clients                                       │
│  curl / OpenAI SDK / LangChain / Open WebUI / Continue.dev          │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ HTTP (port 8080)
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Rank 0 — Coordinator                             │
│                                                                      │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────┐  ┌──────────┐ │
│  │ FastAPI     │  │ Queue Worker │  │ Control Plane │  │ Dashboard│ │
│  │ (HTTP API)  │──│ (sequential) │──│ (TCP :18080)  │  │ (HTMX)  │ │
│  │ :8080       │  │              │  │               │  │ /dashboard│ │
│  └────────────┘  └──────┬───────┘  └───────┬───────┘  └──────────┘ │
│                          │                  │                        │
│                  ┌───────┴──────┐   ┌───────┴──────┐                │
│                  │ mlx_lm       │   │ Broadcast    │                │
│                  │ .generate()  │   │ task / done  │                │
│                  │ (rank 0      │   │              │                │
│                  │  shards)     │   │              │                │
│                  └───────┬──────┘   └───────┬──────┘                │
│                          │                  │                        │
└──────────────────────────┼──────────────────┼────────────────────────┘
          RDMA (all_sum)   │                  │  TCP (framed JSON)
          ~8 GB/s          │                  │  ~1 KB per message
═══════════════════════════╪══════════════════╪════════════════════════
          Thunderbolt      │                  │  Ethernet / TB
═══════════════════════════╪══════════════════╪════════════════════════
┌──────────────────────────┼──────────────────┼────────────────────────┐
│                          │                  │                        │
│                  ┌───────┴──────┐   ┌───────┴──────┐                │
│                  │ mlx_lm       │   │ worker_loop  │                │
│                  │ .generate()  │   │ recv task →  │                │
│                  │ (rank 1      │   │ generate() → │                │
│                  │  shards)     │   │ send done    │                │
│                  └──────────────┘   └──────────────┘                │
│                                                                      │
│                     Rank 1 — Worker                                  │
└──────────────────────────────────────────────────────────────────────┘
```

**Key insight:** The HTTP server only runs on rank 0. Workers have no HTTP server — they run a tight loop that receives tasks over TCP, calls `generate()` (which triggers RDMA collectives internally), and sends back a "done" acknowledgment.

---

## 2. Physical Topology

### 2-Node (Current Production Setup)

```
┌──────────────┐                              ┌──────────────┐
│   Mac mini   │     Thunderbolt cable         │   Mac mini   │
│   M4 Pro     │◄════════════════════════════►│   M4 Pro     │
│   48 GB      │     rdma_en4 ↔ rdma_en4      │   48 GB      │
│   rank 0     │                              │   rank 1     │
│   (coord)    │     8.05 GB/s peak           │   (worker)   │
└──────┬───────┘     25.5 µs latency          └──────────────┘
       │
       │ Ethernet / Wi-Fi (LAN)
       │ SSH for setup + control
       ▼
   Client access
   http://<rank0>:8080
```

- **1 Thunderbolt cable** connects the two Macs
- **RDMA** carries tensor data (all_sum collectives during inference)
- **Ethernet/Wi-Fi** carries SSH (setup), TCP control-plane, and HTTP API traffic
- **No network reconfiguration** required — default macOS settings work

### 4-Node (Fully Connected Mesh)

```
         ┌──── TB ────┐
    ┌────┤            ├────┐
    │    │            │    │
┌───┴──┐ │  ┌──────┐ │ ┌──┴───┐
│rank 0│─┼──│rank 1│─┼─│rank 2│
└───┬──┘ │  └──┬───┘ │ └──┬───┘
    │    │     │     │    │
    │    │  ┌──┴───┐ │    │
    └────┼──│rank 3│─┼────┘
         │  └──────┘ │
         └──── TB ────┘

6 cables total (every pair directly connected)
```

JACCL requires a **fully connected Thunderbolt mesh** for N > 2 nodes. This means:
- 2 nodes → 1 cable
- 3 nodes → 3 cables
- 4 nodes → 6 cables (each Mac needs 3 TB ports)

---

## 3. Software Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  openai_cluster_server.py  │  dashboard.py              │
│  OpenAI API + queue        │  HTMX + SSE UI             │
├─────────────────────────────────────────────────────────┤
│                    Framework Layer                       │
│  FastAPI + Uvicorn         │  Pydantic (schemas)        │
├─────────────────────────────────────────────────────────┤
│                    ML Layer                              │
│  mlx_lm                                                 │
│  ├─ sharded_load()  — distribute model across ranks     │
│  ├─ generate()      — autoregressive token generation   │
│  └─ stream_generate() — streaming token generation      │
├─────────────────────────────────────────────────────────┤
│                    Compute Layer                         │
│  mlx (Apple ML framework)                               │
│  ├─ mx.distributed   — collective operations            │
│  ├─ mx.metal         — Metal GPU compute                │
│  └─ mx.eval()        — lazy → eager evaluation          │
├─────────────────────────────────────────────────────────┤
│                    Transport Layer                       │
│  JACCL (Apple's RDMA over Thunderbolt)                  │
│  ├─ mlx.launch --backend jaccl                          │
│  ├─ ibverbs (RDMA verbs API)                            │
│  └─ rdma_en* kernel devices                             │
├─────────────────────────────────────────────────────────┤
│                    Hardware Layer                        │
│  Apple Silicon (M4 Pro)  │  Thunderbolt controller      │
│  Unified memory (48 GB)  │  rdma_ctl enable (Recovery)  │
└─────────────────────────────────────────────────────────┘
```

### Dependency Graph

```
openai_cluster_server.py
├── mlx_lm.utils.sharded_load
├── mlx_lm.utils.generate / mlx_lm.generate.generate
├── mlx_lm.utils.stream_generate (optional)
├── mlx.core (mx)
│   ├── mx.distributed.init()
│   ├── mx.distributed.all_sum()
│   └── mx.eval()
├── fastapi.FastAPI
├── uvicorn
├── pydantic.BaseModel
├── transformers (tokenizer loading)
├── tokenizers
├── mistral_common (optional, for Mistral tokenizer)
├── huggingface_hub (model download CLI)
└── dashboard.py
    └── (no external deps beyond fastapi + starlette)
```

---

## 4. Process Model

### How `mlx.launch` Works

`mlx.launch --backend jaccl --hostfile hosts.json -- script.py` does the following:

1. Parses the hostfile to determine world size and per-node RDMA devices
2. SSHes into each remote node and launches `script.py` there
3. Starts `script.py` locally (rank 0)
4. Sets environment variables on each process:
   - `MLX_JACCL_COORDINATOR=<rank0_ip>:<port>` — JACCL coordinator address
   - `OMPI_COMM_WORLD_RANK=<rank>` — rank of this process
   - `OMPI_COMM_WORLD_SIZE=<world_size>` — total number of processes
5. Each process calls `mx.distributed.init()` which establishes RDMA connections
6. After init, all processes can call collective operations (e.g., `all_sum`)

### Process Layout (2-Node)

```
Mac 1 (rank 0):
  PID 1234  python server/openai_cluster_server.py
  ├── Main thread:     uvicorn HTTP server (port 8080)
  ├── Thread:          rank0_accept_workers() — TCP control-plane listener (port 18080)
  ├── asyncio task:    _queue_worker() — sequential request processing
  └── (JACCL threads): RDMA communication (managed by mlx runtime)

Mac 2 (rank 1):
  PID 5678  python server/openai_cluster_server.py
  ├── Main thread:     worker_loop() — blocking TCP recv → generate() → send done
  └── (JACCL threads): RDMA communication (managed by mlx runtime)
```

**Important:** There is exactly **one Python process per node**. No multiprocessing, no fork. The MLX runtime manages Metal GPU threads and RDMA threads internally.

---

## 5. Data Plane — RDMA / JACCL

### What RDMA Does Here

During tensor-parallel inference, each rank holds a **shard** of the model's weight matrices. When a layer computes its output, partial results must be summed across all ranks. This is the `all_sum` (a.k.a. `all_reduce`) collective.

```
rank 0: partial_output_0 ──┐
                            ├──► all_sum ──► full_output (on all ranks)
rank 1: partial_output_1 ──┘
```

RDMA (Remote Direct Memory Access) makes this fast because:
- **Zero-copy:** Data moves directly from GPU memory to the network, bypassing the CPU and OS kernel
- **Low latency:** No TCP/IP stack overhead — measured at 25.5 µs for a 4-byte transfer
- **High bandwidth:** Saturates the Thunderbolt link at 8.05 GB/s

### RDMA Device Mapping

The hostfile's `rdma` field is a **matrix**. For node `i`, `rdma[j]` is the device name used to reach node `j`:

```json
[
  { "ssh": "mac.home", "rdma": [null, "rdma_en4"] },
  { "ssh": "mac2",     "rdma": ["rdma_en4", null] }
]
```

Reading this matrix:
- Node 0 → Node 1: use `rdma_en4`
- Node 1 → Node 0: use `rdma_en4`
- Node 0 → Node 0: `null` (self — no RDMA needed)

For 4 nodes with 3 TB ports each:

```json
[
  { "ssh": "mac1", "rdma": [null,       "rdma_en3", "rdma_en4", "rdma_en5"] },
  { "ssh": "mac2", "rdma": ["rdma_en3", null,       "rdma_en4", "rdma_en5"] },
  { "ssh": "mac3", "rdma": ["rdma_en3", "rdma_en4", null,       "rdma_en5"] },
  { "ssh": "mac4", "rdma": ["rdma_en3", "rdma_en4", "rdma_en5", null      ] }
]
```

### Finding Your RDMA Devices

```bash
# List all RDMA devices
ibv_devices

# Show details (look for PORT_ACTIVE)
ibv_devinfo

# Typical output on Mac mini M4 Pro:
#   hca_id: rdma_en3    port_state: PORT_DOWN     ← no cable on this port
#   hca_id: rdma_en4    port_state: PORT_ACTIVE   ← cable connected here
#   hca_id: rdma_en5    port_state: PORT_DOWN     ← no cable on this port
```

### Bandwidth Expectations

| Thunderbolt | Theoretical Max | Measured Peak | Notes |
|---|---|---|---|
| TB3 (20 Gbps) | ~2.5 GB/s | ~2.1 GB/s | M1/M2 Mac mini |
| TB4 (40 Gbps) | ~5 GB/s | ~4.5 GB/s | M3 Mac mini |
| TB5 (80 Gbps) | ~10 GB/s | **8.05 GB/s** | M4 Pro Mac mini (our setup) |

The ~20% gap between theoretical and measured is due to protocol overhead, encoding, and the all_sum collective needing both send and receive.

---

## 6. Control Plane — TCP Framed JSON

### Protocol

The control plane is a minimal framed JSON protocol over TCP. Each message is:

```
┌──────────┬──────────────────────────────────┐
│ 4 bytes  │ N bytes                          │
│ uint32   │ UTF-8 JSON                       │
│ (length) │ (payload)                        │
└──────────┴──────────────────────────────────┘
```

### Wire format implementation

```python
def send_msg(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)))  # 4-byte big-endian length
    sock.sendall(data)

def recv_msg(sock):
    hdr = recvall(sock, 4)
    (n,) = struct.unpack("!I", hdr)
    body = recvall(sock, n)
    return json.loads(body.decode("utf-8"))
```

### Message Types

| Direction | Type | Fields | When |
|---|---|---|---|
| Worker → Rank 0 | `hello` | `rank` | Worker connects at startup |
| Rank 0 → Worker | `task` | `prompt`, `max_tokens` | Before each generation |
| Worker → Rank 0 | `done` | `rank` | After worker completes `generate()` |

### Connection Lifecycle

```
Startup:
  Worker ──── TCP connect ────► Rank 0 (:18080)
  Worker ──── {"type":"hello","rank":1} ────► Rank 0
  Rank 0 stores socket in _worker_socks[1]

Per request:
  Rank 0 ──── {"type":"task","prompt":"...","max_tokens":256} ────► Worker
  (both ranks call generate() simultaneously — RDMA collectives happen)
  Worker ──── {"type":"done","rank":1} ────► Rank 0

Shutdown:
  Connection drops when process exits.
```

### Why TCP and Not RDMA for Control?

Control messages are tiny (~200 bytes) and infrequent (once per request). TCP is simpler, debuggable with standard tools (`tcpdump`, `netstat`), and doesn't require RDMA setup. The RDMA path is reserved for the high-bandwidth tensor data that actually benefits from zero-copy.

---

## 7. HTTP Layer — FastAPI + Uvicorn

### Route Table

| Method | Path | Handler | Auth | Notes |
|---|---|---|---|---|
| `POST` | `/v1/chat/completions` | `chat_completions()` | None | OpenAI chat API |
| `POST` | `/v1/completions` | `completions()` | None | OpenAI completions API |
| `GET` | `/v1/models` | `list_models()` | None | List served models |
| `GET` | `/health` | `health()` | None | Cluster health check |
| `GET` | `/queue` | `queue_status()` | None | Queue depth |
| `GET` | `/dashboard` | `dashboard_page()` | None | HTMX dashboard |
| `GET` | `/` | `dashboard_root()` | None | Redirects to dashboard |
| `GET` | `/metrics/stream` | `metrics_stream()` | None | SSE event stream |
| `GET` | `/metrics/snapshot` | `metrics_snapshot()` | None | Current metrics JSON |
| `GET` | `/docs` | (auto) | None | Swagger / OpenAPI UI |

### Threading Model

```
┌─────────────────────────────────────────────────────┐
│                  Rank 0 Process                      │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │         uvicorn event loop (asyncio)         │    │
│  │                                              │    │
│  │  ┌────────┐  ┌──────────┐  ┌─────────────┐ │    │
│  │  │ HTTP   │  │ Queue    │  │ SSE streams │ │    │
│  │  │ routes │  │ Worker   │  │ (dashboard) │ │    │
│  │  └────────┘  └──────────┘  └─────────────┘ │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  ┌──────────────────────────────────────┐           │
│  │  Thread: rank0_accept_workers()      │           │
│  │  (blocking TCP accept loop)          │           │
│  └──────────────────────────────────────┘           │
│                                                      │
│  ┌──────────────────────────────────────┐           │
│  │  MLX runtime threads                 │           │
│  │  (Metal compute + RDMA — managed     │           │
│  │   internally by mlx)                 │           │
│  └──────────────────────────────────────┘           │
└─────────────────────────────────────────────────────┘
```

The `rank0_accept_workers()` runs in a daemon thread so it doesn't block the asyncio event loop. The queue worker is an asyncio task that processes requests sequentially (one at a time) because tensor-parallel generation requires all ranks to call `generate()` in lockstep.

---

## 8. Request Lifecycle

### Non-Streaming Request

```
Client                  Rank 0                          Rank 1
  │                       │                               │
  │  POST /v1/chat/...    │                               │
  │──────────────────────►│                               │
  │                       │                               │
  │                       │  1. Build prompt from messages │
  │                       │  2. Create asyncio.Future      │
  │                       │  3. Put (kind, prompt, max_t,  │
  │                       │     future, stream=False)      │
  │                       │     into _queue                │
  │                       │                               │
  │                       │  --- queue_worker picks up --- │
  │                       │                               │
  │                       │  4. broadcast_task()           │
  │                       │─────────────────────────────►│
  │                       │     {"type":"task",            │
  │                       │      "prompt":"...",           │
  │                       │      "max_tokens":256}         │
  │                       │                               │
  │                       │  5. generate() on rank 0       │  5. generate() on rank 1
  │                       │     ◄══ RDMA all_sum ══►      │
  │                       │     ◄══ RDMA all_sum ══►      │
  │                       │     ◄══ RDMA all_sum ══►      │
  │                       │     (one all_sum per layer     │
  │                       │      per token)                │
  │                       │                               │
  │                       │  6. mx.eval()                  │  6. mx.eval()
  │                       │                               │
  │                       │                               │  7. send {"type":"done"}
  │                       │◄─────────────────────────────│
  │                       │                               │
  │                       │  8. Build OpenAI response JSON │
  │                       │  9. Resolve future             │
  │                       │ 10. Record metrics             │
  │                       │                               │
  │  JSON response         │                               │
  │◄──────────────────────│                               │
  │                       │                               │
```

### Streaming Request

```
Client                  Rank 0                          Rank 1
  │                       │                               │
  │  POST /v1/chat/...    │                               │
  │  stream: true          │                               │
  │──────────────────────►│                               │
  │                       │                               │
  │  SSE connection opens │  1. Create asyncio.Queue       │
  │◄──────────────────────│  2. Put into _queue            │
  │                       │                               │
  │                       │  --- queue_worker picks up --- │
  │                       │                               │
  │                       │  3. broadcast_task()           │
  │                       │─────────────────────────────►│
  │                       │                               │
  │                       │  4. stream_generate() loop     │  4. generate() (blocking)
  │                       │     for each token:            │
  │  data: {chunk}         │     ◄══ RDMA all_sum ══►      │
  │◄──────────────────────│        put chunk in queue      │
  │  data: {chunk}         │     ◄══ RDMA all_sum ══►      │
  │◄──────────────────────│        put chunk in queue      │
  │  data: {chunk}         │     ...                       │
  │◄──────────────────────│                               │
  │                       │                               │
  │  data: {stop}          │  5. Final chunk               │
  │◄──────────────────────│                               │
  │  data: [DONE]          │                               │  6. send {"type":"done"}
  │◄──────────────────────│◄─────────────────────────────│
  │                       │                               │
```

**Critical detail:** The worker always calls the non-streaming `generate()`, even when the client requests streaming. Only rank 0 uses `stream_generate()`. Both functions trigger the same RDMA collectives internally — the difference is only in how rank 0 surfaces the output (all-at-once vs. token-by-token).

---

## 9. Dashboard — HTMX + SSE

### Architecture

```
Browser                    Rank 0 Server
  │                            │
  │  GET /dashboard            │
  │───────────────────────────►│
  │  ◄── Full HTML page ──────│  (pre-rendered, static)
  │                            │
  │  GET /metrics/stream       │
  │───────────────────────────►│
  │  ◄── SSE: data: {...} ────│  (every 2 seconds)
  │  ◄── SSE: data: {...} ────│
  │  ◄── SSE: data: {...} ────│
  │       ...                  │
  │                            │
  │  POST /v1/chat/completions │  (from chat UI)
  │───────────────────────────►│
  │  ◄── SSE: data: {chunk} ──│  (streaming response)
  │       ...                  │
```

### How Updates Work

1. **Page load:** Server returns a fully rendered HTML page (~800 lines) with all styling inline. No external CSS/JS files except HTMX (loaded from CDN).

2. **Live metrics:** The page opens an SSE connection to `/metrics/stream`. Every 2 seconds, the server pushes a JSON blob with current metrics.

3. **Client-side update:** HTMX's SSE extension receives the event and swaps specific DOM elements. JavaScript updates numeric displays, sparkline SVG, and queue bar.

4. **Chat UI:** The chat form sends a streaming request to `/v1/chat/completions`. Tokens are appended to the chat window as they arrive.

### MetricsStore

```python
class MetricsStore:
    _history: deque[GenerationStats]  # ring buffer, maxlen=200
    _total_requests: int
    _total_tokens: int
    _total_prompt_tokens: int
    _error_count: int
    _server_start: float

    async def snapshot() -> dict:
        # Returns: uptime, totals, 60s averages, last 40 points for sparkline
```

The store is **thread-safe** (asyncio lock) and **memory-bounded** (ring buffer). No database, no disk I/O.

### Why No SPA Framework

| Consideration | HTMX + SSE | SvelteKit (exo) |
|---|---|---|
| Build step | None | `npm install` + `npm run build` |
| Dependencies | 1 (htmx.js from CDN) | ~200 npm packages |
| Lines of code | ~800 | ~4,000 |
| Update mechanism | SSE push + DOM swap | WebSocket + Svelte reactivity |
| Offline capable | Yes (HTML is self-contained) | Yes (built bundle) |
| Customization | Edit Python string, restart | Edit Svelte, rebuild, restart |

For a monitoring dashboard that displays server-pushed data, HTMX + SSE is sufficient and eliminates the entire Node.js/npm toolchain.

---

## 10. Sharded Model Loading

### How Tensor Parallelism Works

When `mlx_lm.sharded_load(model_path)` is called on N ranks:

1. Each rank reads the model config to determine the architecture
2. Each rank loads only its **shard** of the weights (1/N of each tensor)
3. The model is partitioned along specific dimensions:
   - Attention Q/K/V projections: split along head dimension
   - MLP up/gate projections: split along intermediate dimension
   - MLP down projection: split along input dimension
   - Embedding and output layers: split along vocabulary dimension

```
Full weight matrix (hidden × intermediate):
┌──────────────────────────────────────┐
│                                      │
│            W (full)                  │
│                                      │
└──────────────────────────────────────┘

Sharded across 2 ranks:
┌──────────────────┐ ┌──────────────────┐
│                  │ │                  │
│   W_0 (rank 0)  │ │   W_1 (rank 1)  │
│                  │ │                  │
└──────────────────┘ └──────────────────┘
```

### Memory Usage

For a 4-bit quantized 4B parameter model on 2 nodes:

```
Total model size:     ~2.4 GB
Per-rank shard:       ~1.2 GB
KV cache per rank:    ~0.5–2 GB (depends on context length)
MLX overhead:         ~0.5 GB
OS / other:           ~20 GB

Total per node:       ~22–24 GB of 48 GB unified memory
```

The key advantage of distributing across nodes: **you get 2× the memory**. A model that needs 60 GB can run on two 48 GB nodes (30 GB per node).

### Fallback Loading

The server implements `sharded_load_with_fallback()`:

```python
def sharded_load_with_fallback(model_path):
    try:
        model, tokenizer = sharded_load(model_path)
    except Exception:
        # Fallback: load_model + manual distribute
        model, tokenizer = load_model(model_path)
    return model, tokenizer
```

This handles older versions of `mlx_lm` that may not have `sharded_load` or where the model format isn't compatible with the sharded loader.

---

## 11. Streaming (SSE) Pipeline

### Token Flow Through the System

```
mlx_lm.stream_generate()
    │
    │  yields GenerationResponse(text="Hello")
    │  yields GenerationResponse(text=" world")
    │  yields GenerationResponse(text="!")
    │
    ▼
_queue_worker()
    │
    │  Formats as OpenAI SSE chunk:
    │  data: {"choices":[{"delta":{"content":"Hello"}}]}
    │
    │  Puts into asyncio.Queue (chunk_queue)
    │
    ▼
_stream_generator()
    │
    │  async generator — yields from chunk_queue
    │
    ▼
StreamingResponse(media_type="text/event-stream")
    │
    │  Sent to client as HTTP chunked transfer
    │
    ▼
Client
```

### SSE Format (per OpenAI spec)

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1721000000,"model":"Qwen3-4B","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1721000000,"model":"Qwen3-4B","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1721000000,"model":"Qwen3-4B","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

---

## 12. Queue & Backpressure

### Why Sequential Processing

Tensor-parallel inference requires **all ranks to execute the same operations in the same order**. If rank 0 started processing request B while rank 1 was still on request A, the RDMA collectives would mismatch and produce garbage or deadlock.

Therefore, requests are processed **strictly sequentially** — one at a time, in FIFO order.

### Queue Architecture

```
HTTP request arrives
       │
       ▼
  ┌─────────┐     Queue full?
  │ Enqueue  │────── Yes ──► HTTP 429 Too Many Requests
  └────┬────┘
       │ No
       ▼
  ┌──────────────┐
  │ asyncio.Queue │  maxsize = QUEUE_MAX (default 8)
  │              │
  │  [req1]      │  ◄── waiting
  │  [req2]      │  ◄── waiting
  │  [req3]      │  ◄── processing (queue_worker has it)
  └──────────────┘
       │
       ▼
  _queue_worker()
       │
       │  1. broadcast_task() to workers
       │  2. generate() / stream_generate()
       │  3. rank0_wait_done()
       │  4. resolve future / push chunks
       │
       ▼
  Response sent to client
```

### Queue Item Format

```python
# Each queue item is a tuple:
(kind, prompt, max_tokens, result_target, is_stream)

# kind:           "chat" | "completions"
# prompt:         str — the full formatted prompt
# max_tokens:     int
# result_target:  asyncio.Future (non-streaming) or asyncio.Queue (streaming)
# is_stream:      bool
```

### Backpressure Behavior

| Queue state | HTTP response | Client experience |
|---|---|---|
| Queue has space | 200 (enqueued, waiting) | Request proceeds normally |
| Queue full | 429 Too Many Requests | Client should retry with backoff |
| Server processing | 200 (eventual) | Client waits for response (timeout: REQ_TIMEOUT) |

---

## 13. Hostfile Format & RDMA Matrix

### Full Schema

```json
[
  {
    "ssh": "<hostname_or_ip>",
    "ips": ["<lan_ip>", ...],
    "rdma": [<device_or_null>, ...]
  },
  ...
]
```

| Field | Type | Required | Description |
|---|---|---|---|
| `ssh` | string | Yes | SSH target for `mlx.launch` and all scripts. Can be hostname, FQDN, or IP. |
| `ips` | string[] | Rank 0 only | LAN IP addresses. Used to set `CTRL_HOST` for the TCP control-plane. Workers' `ips` can be empty. |
| `rdma` | (string\|null)[] | Yes | RDMA device matrix. Length must equal total number of nodes. `rdma[i]` = device to reach node `i`. `null` for self. |

### Validation Rules

1. All nodes must have the same-length `rdma` array (= world size)
2. `rdma[self_index]` must be `null`
3. Non-null `rdma` entries must correspond to `PORT_ACTIVE` devices on that node
4. Rank 0 must have at least one entry in `ips`
5. `ssh` must be reachable via passwordless SSH from rank 0

### Example: 2-Node

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

---

## 14. Environment Variables

### Required

| Variable | Set By | Description |
|---|---|---|
| `MODEL_DIR` | User | Path to the MLX model directory. Must exist on all nodes. |

### Server Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | basename of MODEL_DIR | Model identifier in API responses |
| `HOST` | `0.0.0.0` | HTTP bind address (rank 0) |
| `PORT` | `8080` | HTTP port (rank 0) |
| `CTRL_HOST` | auto-detect from hostfile | TCP control-plane host (rank 0 LAN IP) |
| `CTRL_PORT` | `18080` | TCP control-plane port |
| `QUEUE_MAX` | `8` | Maximum queued requests |
| `REQ_TIMEOUT` | `120` | Per-request timeout (seconds) |
| `MAX_TOKENS` | `512` | Default max_tokens when not specified in request |

### MLX Performance

| Variable | Recommended | Description |
|---|---|---|
| `MLX_METAL_FAST_SYNCH` | `1` | **Critical.** Enables fast Metal synchronization. Without it, inference is 5–6× slower. |
| `HF_HUB_OFFLINE` | `1` | Prevents HuggingFace from downloading models at runtime. |
| `TRANSFORMERS_OFFLINE` | `1` | Prevents the `transformers` library from downloading at runtime. |

### Set Automatically by `mlx.launch`

| Variable | Description |
|---|---|
| `MLX_JACCL_COORDINATOR` | JACCL coordinator address (`<ip>:<port>`) |
| `OMPI_COMM_WORLD_RANK` | Rank of this process (0, 1, ...) |
| `OMPI_COMM_WORLD_SIZE` | Total number of processes |

---

## 15. Startup Sequence (Detailed)

### Timeline

```
t=0.0s   make server (or run_openai_cluster_server.sh)
         │
         ├── Validates MODEL_DIR exists
         ├── Parses hostfile for CTRL_HOST and RDMA_DEVICES
         ├── Kills stale processes on all nodes
         │
t=0.5s   mlx.launch --backend jaccl --hostfile hosts.json -- server/openai_cluster_server.py
         │
         ├── mlx.launch SSHes into rank 1, starts script there
         ├── mlx.launch starts script locally (rank 0)
         │
t=1.0s   Both ranks: mx.distributed.init()
         │
         ├── JACCL establishes RDMA connections between all rank pairs
         ├── RDMA queue pairs created, memory regions registered
         │
t=2.0s   Both ranks: sharded_load_with_fallback(MODEL_DIR)
         │
         ├── Each rank loads its shard of the model weights
         ├── Tokenizer loaded (identical on both ranks)
         │
t=5.0s   Rank 0:                              Rank 1:
         │                                      │
         ├── mount_dashboard()                  ├── worker_loop() starts
         ├── Start accept_workers thread        │   ├── TCP connect to rank 0
         │   └── Listen on :18080               │   ├── Send {"type":"hello","rank":1}
         │                                      │   └── Block waiting for tasks
         ├── rank0_wait_for_workers()
         │   └── Wait until all workers connected
         │
t=6.0s   ├── "all workers connected"
         ├── uvicorn.run(app, port=8080)
         │
t=6.5s   Server ready — accepting requests
         │
         Dashboard: http://localhost:8080/dashboard
         API:       http://localhost:8080/v1/chat/completions
         Health:    http://localhost:8080/health
```

### Failure Points

| Failure | Symptom | Cause |
|---|---|---|
| SSH fails | `mlx.launch` exits immediately | SSH keys not set up, wrong hostname |
| RDMA init fails | Hang at `mx.distributed.init()` | RDMA not enabled, wrong device in hostfile, cable unplugged |
| Model load fails | Crash with file not found | MODEL_DIR doesn't exist on one/both nodes |
| Worker connect fails | Timeout in `rank0_wait_for_workers()` | CTRL_HOST wrong, firewall blocking port 18080 |
| Port in use | uvicorn bind error | Previous server still running — use `make server-stop` first |

---

## 16. Failure Modes & Recovery

### Worker Process Dies

**Symptom:** Next HTTP request hangs forever (rank 0 waits for `done` that never comes).

**Current behavior:** No timeout on `rank0_wait_done()` — will hang indefinitely.

**Recovery:** `make kill-all` then `make server`.

**Planned fix (Phase 1):** Heartbeat protocol + timeout detection. See [roadmap.md](roadmap.md#43-worker-health--disconnect-detection).

### Thunderbolt Cable Disconnected

**Symptom:** RDMA collectives fail. Likely manifests as a segfault or hang in the MLX runtime.

**Recovery:** Reconnect cable, `make kill-all`, `make server`.

**Detection:** `ibv_devinfo` shows `PORT_DOWN` instead of `PORT_ACTIVE`.

### Queue Full

**Symptom:** HTTP 429 responses.

**Recovery:** Automatic — queue drains as requests complete. Increase `QUEUE_MAX` if needed.

### Model Not Found on Worker

**Symptom:** Crash on startup with `FileNotFoundError`.

**Recovery:** `rsync` the model to the worker node, or use `make download MODEL=...` (planned).

### Memory Exhaustion (OOM)

**Symptom:** Process killed by macOS, or MLX throws allocation error.

**Recovery:** Use a smaller model, reduce `max_tokens`, or add more nodes.

**Detection (planned):** Memory monitoring in dashboard (Phase 1).

---

## 17. Security Model

### Current State: Development / Trusted Network

The server has **no authentication, no encryption, no rate limiting**. It is designed for use on a private network (e.g., home lab, research cluster).

| Attack Surface | Status | Notes |
|---|---|---|
| HTTP API | Unauthenticated | Anyone on the network can send requests |
| TCP control-plane | Unauthenticated | Anyone could connect and send fake `hello` messages |
| SSH | Key-based auth | Passwordless SSH required between nodes |
| RDMA | No encryption | Data in flight is unencrypted (but on a direct TB cable) |
| Model weights | On disk | No encryption at rest |

### Recommendations for Production

If exposing beyond a trusted network:

1. **Reverse proxy:** Put nginx/caddy in front with TLS + API key auth
2. **Firewall:** Restrict ports 8080 and 18080 to trusted IPs
3. **SSH hardening:** Disable password auth, use ed25519 keys
4. **Network isolation:** Keep the TB link physically separate from internet-facing interfaces

---

## 18. Performance Characteristics

### Latency Breakdown (Single Request)

```
Client → HTTP → Rank 0:           ~1 ms   (network)
Queue wait (if empty):             ~0 ms   (immediate)
Broadcast task to workers:         ~1 ms   (TCP)
Prompt encoding (tokenizer):       ~2 ms   (CPU)
Prefill (all prompt tokens):       ~50-200 ms  (GPU + RDMA)
Decode (per token):                ~15-25 ms   (GPU + RDMA)
  ├── GPU compute:                 ~10-15 ms
  └── RDMA all_sum:                ~5-10 ms    (25.5 µs + collective overhead)
Worker done → response:            ~1 ms   (TCP + JSON)
```

### Throughput

| Metric | Value | Notes |
|---|---|---|
| Tokens/sec (single request) | ~40-70 tok/s | Depends on model size and quantization |
| Concurrent requests | 1 (serial) | Limited by sequential queue processing |
| Max queue depth | 8 (default) | Configurable via QUEUE_MAX |
| RDMA bandwidth (sustained) | ~7.5 GB/s | From rdma_test.py stress mode |
| RDMA latency | ~25.5 µs | Single-element all_sum |

### Scaling Expectations

| Nodes | Theoretical Speedup | Practical Speedup | Notes |
|---|---|---|---|
| 1 | 1× | 1× | Baseline — single node |
| 2 | 2× | ~1.6–1.8× | RDMA overhead reduces gains |
| 4 | 4× | ~2.5–3× | More collectives, more overhead |

The primary benefit of multi-node is **increased memory**, not raw speed. A 30B model that doesn't fit in 48 GB can run on 2 × 48 GB nodes.

---

## 19. File Map

### Repository Structure with Annotations

```
mlx-jaccl-cluster/
│
├── Makefile                            # 230 lines — all operations as make targets
│                                       # setup, verify, rdma-test, server, bench, etc.
│
├── pyproject.toml                      # uv / pip dependency manifest (8 packages)
│
├── hostfiles/
│   ├── hosts-2node.json                # Working 2-node config (committed)
│   ├── hosts-1node.json                # Single-node for local testing
│   └── hosts.json.example              # Template for custom setups
│
├── server/
│   ├── openai_cluster_server.py        # ~785 lines — the main server
│   │   ├── Configuration (env vars)
│   │   ├── Framed JSON protocol (send_msg, recv_msg)
│   │   ├── OpenAI request schemas (Pydantic)
│   │   ├── Control-plane (accept_workers, broadcast_task, wait_done)
│   │   ├── Worker loop
│   │   ├── Queue worker (sequential processing)
│   │   ├── HTTP endpoints (health, models, chat, completions)
│   │   └── main() — init, load model, start server
│   │
│   └── dashboard.py                    # ~919 lines — HTMX + SSE dashboard
│       ├── GenerationStats dataclass
│       ├── MetricsStore (ring buffer + counters)
│       ├── _render_dashboard() — generates full HTML page
│       ├── _metrics_event_generator() — SSE push loop
│       └── mount_dashboard() — registers routes on FastAPI app
│
├── scripts/
│   ├── setup.sh                        # One-shot node installer
│   ├── bootstrap_node.sh               # Remote node setup over SSH
│   ├── rdma_test.py                    # RDMA correctness + benchmark
│   ├── jaccl_tps_bench.py              # Tokens/sec benchmark
│   ├── cluster_info.sh                 # Side-by-side node report
│   ├── verify_cluster.sh               # SSH + RDMA device check
│   ├── sync_nodes.sh                   # git pull on all nodes
│   ├── run_openai_cluster_server.sh    # Start server
│   └── stop_openai_cluster_server.sh   # Stop server
│
├── docs/
│   ├── architecture.md                 # ← This document
│   ├── roadmap.md                      # Feature roadmap + gap analysis
│   ├── from-scratch.md                 # Full setup guide
│   ├── comparison-vs-exo.md            # Deep comparison with exo
│   └── scripts-reference.md            # Scripts + Makefile reference
│
└── .gitignore
```

### Line Count Summary

| Component | Lines | Language |
|---|---|---|
| `openai_cluster_server.py` | ~785 | Python |
| `dashboard.py` | ~919 | Python (HTML/CSS/JS inline) |
| `rdma_test.py` | ~300 | Python |
| `jaccl_tps_bench.py` | ~150 | Python |
| Shell scripts (8 files) | ~600 | Bash |
| `Makefile` | ~230 | Make |
| Documentation (5 files) | ~2,500 | Markdown |
| **Total** | **~5,500** | — |

Compare to exo: **~32,000 lines** of Python + Rust + Svelte + Swift.

---

## 20. Design Principles

### 1. Explicit Over Automatic

We use a static hostfile instead of auto-discovery. We manually download and sync models instead of coordinated downloads. Every configuration is visible and auditable.

**Tradeoff:** More manual steps. **Benefit:** No race conditions, no mysterious failures, no magic.

### 2. Minimal Dependencies

Eight Python packages. No Rust compiler, no Node.js, no npm, no Swift. Every dependency must justify its presence.

**Tradeoff:** Some features are harder to build (e.g., dashboard without React). **Benefit:** No supply chain risk, fast installs, reproducible environments.

### 3. Stock MLX

We use the official `mlx` from PyPI. No custom forks, no patches, no vendor branches.

**Tradeoff:** We can't use features that only exist in forks. **Benefit:** We're always compatible with the latest official MLX release. No fork maintenance burden.

### 4. One Process Per Node

Each node runs exactly one Python process. No multiprocessing, no process pools, no worker spawning.

**Tradeoff:** Can't do CPU-parallel tokenization. **Benefit:** Simple process model, easy to debug, no IPC complexity.

### 5. Sequential Request Processing

Requests are processed one at a time. The queue provides backpressure but not parallelism.

**Tradeoff:** Lower throughput under concurrent load. **Benefit:** Correct tensor-parallel execution guaranteed. All ranks always agree on what to compute.

### 6. Zero Build Step

The dashboard is inline HTML/CSS/JS generated by Python. No bundler, no transpiler, no asset pipeline.

**Tradeoff:** Less ergonomic to develop than a modern frontend framework. **Benefit:** `git clone` → `make setup` → `make server` — that's it. No build artifacts to manage.

### 7. Debuggable

Total application code is ~2,000 lines of Python. Any developer can read the entire codebase in an afternoon and understand every component.

**Tradeoff:** Features are more limited than a 32K-line project. **Benefit:** When something breaks, you can find and fix it in minutes, not days.

---

*For the feature roadmap and planned enhancements, see [roadmap.md](roadmap.md).*
*For a comparison with the exo project, see [comparison-vs-exo.md](comparison-vs-exo.md).*
```

Now let me enhance the Makefile with the new targets: