# Server Improvement Plan — mlx-jaccl-cluster

> **Created:** 2025-07-15
> **Status:** Planning
> **Reference:** `refs/exo/` (cloned for code comparison)
>
> This plan details concrete server improvements informed by studying
> [exo-explore/exo](https://github.com/exo-explore/exo) (41K+ stars).
> Each phase builds on the previous one. Phases are ordered by
> impact-to-effort ratio: the biggest safety/performance wins come first.
>
> **Related docs:**
> - [Memory Management Guide](memory-management.md)
> - [Memory Comparison: exo vs us](memory-comparison-exo.md)
> - [Full Roadmap](roadmap.md)
> - [Architecture](architecture.md)

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Current State (What We Have)](#current-state-what-we-have)
- [Gap Summary (What We're Missing)](#gap-summary-what-were-missing)
- [Phase S1 — Background Memory Monitor + Tiered Thresholds](#phase-s1--background-memory-monitor--tiered-thresholds)
- [Phase S2 — KV Prefix Cache with LRU Eviction](#phase-s2--kv-prefix-cache-with-lru-eviction)
- [Phase S3 — Subprocess Model Runner (Optional Safety Mode)](#phase-s3--subprocess-model-runner-optional-safety-mode)
- [Phase S4 — Memory-Proportional Layer Allocation](#phase-s4--memory-proportional-layer-allocation)
- [Phase S5 — Request Queue Hardening & Observability](#phase-s5--request-queue-hardening--observability)
- [Implementation Order & Dependencies](#implementation-order--dependencies)
- [Risk Assessment](#risk-assessment)
- [Success Criteria](#success-criteria)
- [Appendix: Exo Reference File Map](#appendix-exo-reference-file-map)

---

## Executive Summary

Our server already prevents the IOGPU kernel panic through **proactive MLX memory limits**,
per-token pressure checks, hard token caps, and a careful model unload lifecycle.
Exo takes a fundamentally different approach: **eviction + isolation** (KV cache LRU,
subprocess runners, distributed pressure via `mx.distributed.all_gather`).

Neither approach alone is complete. This plan combines the best of both:

| Phase | What | Impact | Effort | Files |
|-------|------|--------|--------|-------|
| **S1** | Background memory monitor + tiered thresholds | 🟢 High (foundation for everything) | ~150 lines | `server/memory_monitor.py`, edits to `memory_manager.py`, `dashboard.py` |
| **S2** | KV prefix cache with LRU eviction | 🔴 Critical (biggest perf + safety win) | ~500 lines | `server/kv_cache.py`, edits to `openai_cluster_server.py` |
| **S3** | Subprocess model runner | 🟡 Medium (guaranteed crash recovery) | ~400 lines | `server/runner_supervisor.py`, `server/runner_process.py` |
| **S4** | Memory-proportional layer allocation | 🟡 Medium (heterogeneous cluster support) | ~200 lines | Edits to `openai_cluster_server.py`, `memory_manager.py` |
| **S5** | Request queue hardening + observability | 🟢 High (production readiness) | ~300 lines | Edits to `openai_cluster_server.py`, `server/request_log.py` |

**Total estimated new code: ~1,550 lines across 4 new files + edits to 3 existing files.**

---

## Current State (What We Have)

### server/memory_manager.py (~1,500 lines)

Our strongest component. Already implements:

- **Adaptive 4-candidate MLX memory limit** — probes actual free RAM via `vm_stat`/`sysctl`
  at startup and again before each model load; picks the most conservative of:
  `fraction_of_total`, `total_minus_reserve`, `apple_recommended`, `adaptive_available`
- **Per-token generation guard** — `generation_guard()` checks pressure every 16 tokens,
  aborts with `MemoryPressureError` before the OS reaches critical state
- **Hard token cap** — default 4,096 tokens, configurable via `MAX_TOKENS` env var
- **Full model unload lifecycle** — `del` refs → triple `gc.collect()` → `mx.clear_cache()`
  → sleep → second pass → `mx.reset_peak_memory()`
- **Emergency cleanup** — triggered when pressure exceeds critical threshold
- **Qwen3 thinking mode disable** — patches `apply_chat_template` to inject
  `enable_thinking=False`, preventing runaway thinking token generation
- **Cross-node awareness** — `cluster_memory_snapshot()` queries remote nodes via SSH
- **Readjust on load** — `readjust_limits()` re-measures system memory before each
  model load, accounting for changed conditions since server startup

### server/openai_cluster_server.py (~1,500 lines)

- OpenAI-compatible `/v1/chat/completions` and `/v1/completions` (streaming SSE)
- Sequential request queue (`asyncio.Queue`) with single `_queue_worker()`
- Memory-pressure-aware generation loop (checks every 16 tokens)
- Post-request `gc_cycle()` to reclaim KV cache memory
- Model load/unload/gc REST endpoints
- Rank-0 control plane: accepts workers, broadcasts tasks, waits for completion

### server/dashboard.py (~1,600 lines)

- HTMX + SSE live dashboard with tok/s sparkline, queue depth, chat UI
- `HardwarePoller` — background thread polling `macmon` on all cluster nodes
  (local subprocess + remote via SSH), stores latest metrics per host
- `MetricsStore` — records generation stats (tokens, latency, errors)
- SSE event stream at `/dashboard/metrics` pushing snapshots every 2s

---

## Gap Summary (What We're Missing)

Ordered by severity:

| # | Gap | Why It Matters | Exo Reference |
|---|-----|---------------|---------------|
| 1 | **No KV prefix cache** | Every request re-processes full prompt; repeated prompts waste compute; no eviction under memory pressure — only hard token cap prevents blowout | `refs/exo/src/exo/worker/engines/mlx/cache.py` — `KVPrefixCache` |
| 2 | **No continuous memory monitoring** | `snapshot()` is on-demand only; no live pressure feed; dashboard `HardwarePoller` runs `macmon` externally but doesn't feed back into `MemoryManager` decisions | `refs/exo/src/exo/utils/info_gatherer/info_gatherer.py` — `_monitor_memory_usage()` |
| 3 | **No distributed pressure check** | `cluster_memory_snapshot()` is SSH-based and slow; can't react in real-time to the weakest node's memory state | `refs/exo/src/exo/worker/engines/mlx/cache.py` — `get_memory_used_percentage()` using `mx.distributed.all_gather` |
| 4 | **No subprocess isolation** | Model crash (OOM, Metal error) kills the entire server process; no guaranteed memory reclaim without process death | `refs/exo/src/exo/worker/runner/runner_supervisor.py` — `RunnerSupervisor` |
| 5 | **Static layer allocation** | `sharded_load` splits layers evenly; doesn't account for heterogeneous available memory across nodes | `refs/exo/src/exo/master/placement_utils.py` — `allocate_layers_proportionally()` |
| 6 | **No tiered memory thresholds** | Same thresholds regardless of machine size (48 GB vs 192 GB behave the same) | `refs/exo/src/exo/worker/engines/mlx/cache.py` — `_default_memory_threshold()` |
| 7 | **No `OVERRIDE_MEMORY_MB`** | Can't simulate constrained environments for testing without actually consuming memory | `refs/exo/src/exo/utils/info_gatherer/info_gatherer.py` — `override_memory_env` |
| 8 | **No request history/logging** | Can't diagnose production issues after the fact | N/A (general best practice) |

---

## Phase S1 — Background Memory Monitor + Tiered Thresholds

> **Priority:** HIGH — foundation for S2 (KV eviction decisions) and S5 (observability)
> **Effort:** ~150 lines new code + ~50 lines edits
> **Risk:** LOW — additive, no changes to generation path
> **Exo reference:** `refs/exo/src/exo/utils/info_gatherer/info_gatherer.py` lines 462–478

### 1.1 Goals

- Always-available live `MemorySnapshot` updated every 1–2 seconds
- Feed live pressure data into `MemoryManager` (currently only measured on-demand)
- Tiered default thresholds by machine size (matching exo's 70/75/80/85% scheme)
- `OVERRIDE_MEMORY_MB` env var for testing constrained environments
- Dashboard integration: `HardwarePoller` consumes the new monitor instead of running
  its own independent `macmon` subprocess

### 1.2 New File: `server/memory_monitor.py`

```
server/memory_monitor.py (~120 lines)

Classes:
  MemoryMonitor
    - __init__(manager: MemoryManager, interval: float = 1.5)
    - start() → threading.Thread (daemon)
    - stop()
    - _poll_loop()                # runs in background thread
    - latest: MemorySnapshot      # always up-to-date, thread-safe via lock
    - pressure_history: deque     # last 60 snapshots (1 min at 1s intervals)
    - on_critical: Callable       # callback when pressure > critical threshold
    - peak_pressure_1m: float     # max pressure in last 60s

Functions:
  default_memory_threshold(total_ram_gb: float) → float
    # Tiered: >=128 GB → 0.85, >=64 → 0.80, >=32 → 0.75, <32 → 0.70
    # (matches exo's _default_memory_threshold)

  override_available_memory() → int | None
    # Reads OVERRIDE_MEMORY_MB env var, returns bytes or None
```

### 1.3 Edits to Existing Files

**`server/memory_manager.py`:**
- Add `tiered_threshold` property that uses `default_memory_threshold()`
- Modify `_apply_limits()` to use tiered threshold when calculating `adaptive_available`
- Modify `_get_local_memory_usage()` to respect `OVERRIDE_MEMORY_MB`
- Add `update_live_snapshot(snap: MemorySnapshot)` method for the monitor to push data
- Expose `_PRESSURE_WARN` and `_PRESSURE_CRITICAL` as tiered (currently hardcoded)

**`server/dashboard.py`:**
- Modify `HardwarePoller` to optionally accept a `MemoryMonitor` instance for the
  local node, falling back to `macmon` subprocess only for remote nodes
- Include `peak_pressure_1m` in SSE events

**`server/openai_cluster_server.py`:**
- Start `MemoryMonitor` in `_lifespan()`
- Pass monitor instance to dashboard `mount_dashboard()`
- Expose `GET /memory/live` endpoint returning last 60 snapshots (for graphs)

### 1.4 Environment Variables (New)

| Variable | Default | Description |
|----------|---------|-------------|
| `OVERRIDE_MEMORY_MB` | (unset) | Override reported available memory for testing |
| `MEMORY_POLL_INTERVAL` | `1.5` | Seconds between background memory polls |
| `MEMORY_THRESHOLD` | (auto, tiered) | Override the tiered memory pressure threshold (0.0–1.0) |

### 1.5 Implementation Notes

- Use `threading.Thread(daemon=True)` not `asyncio` — the monitor must not be blocked
  by the event loop during generation (generation is CPU-bound and holds the GIL
  for extended periods during `mlx_lm.stream_generate`)
- Thread-safe: `MemorySnapshot` is a frozen dataclass, assign atomically; `deque` with
  `maxlen=60` is thread-safe for append/read in CPython
- The monitor calls `mm.snapshot()` which is already documented as thread-safe
- If `OVERRIDE_MEMORY_MB` is set, `_get_local_memory_usage()` returns that value as
  `available_gb`, allowing us to simulate a 16 GB machine on a 48 GB machine

### 1.6 Testing

```
# Simulate a 16 GB machine
OVERRIDE_MEMORY_MB=16384 make server

# Verify tiered threshold was applied
curl localhost:8000/memory | jq .threshold_pct
# → 70.0 (because 16 GB < 32 GB)

# Check live pressure feed
curl localhost:8000/memory/live | jq '.[0].pressure_pct'
```

---

## Phase S2 — KV Prefix Cache with LRU Eviction

> **Priority:** CRITICAL — biggest single improvement for both performance and safety
> **Effort:** ~500 lines new code + ~80 lines edits
> **Risk:** MEDIUM — touches generation path; needs careful integration with mlx_lm cache types
> **Exo reference:** `refs/exo/src/exo/worker/engines/mlx/cache.py` (entire file, ~280 lines)

### 2.1 Goals

- **Prefix reuse:** If a new request shares a prompt prefix with a cached request,
  reuse the KV cache for the shared prefix and only prefill the new suffix
- **LRU eviction:** When memory pressure exceeds the tiered threshold (from S1),
  evict the least-recently-used cache entries until pressure drops below threshold
- **Distributed pressure:** Use `mx.distributed.all_gather` to take the **max**
  pressure across all nodes — eviction triggers when *any* node is stressed
- **Memory accounting:** Track approximate memory consumption of each cached entry

### 2.2 New File: `server/kv_cache.py`

```
server/kv_cache.py (~400 lines)

Classes:
  CacheEntry
    - prompt_tokens: mx.array     # the full prompt as token IDs
    - cache: list[KVCache]        # the mlx_lm KV cache objects
    - token_count: int            # number of tokens in this entry
    - approx_bytes: int           # estimated memory footprint
    - last_used: int              # monotonic access counter

  KVPrefixCache
    - __init__(group: mx.distributed.Group | None, threshold: float)
    - add(prompt_tokens: mx.array, cache: list[KVCache])
    - get(model, prompt_tokens: mx.array) → (cache, remaining_tokens, matched_index | None)
    - update(index: int, prompt_tokens: mx.array, cache: list[KVCache])
    - clear()
    - evict_if_needed()           # LRU eviction loop
    - memory_used_pct() → float   # distributed-aware pressure
    - entry_count: int
    - total_cached_tokens: int

Functions:
  get_prefix_length(a: mx.array, b: mx.array) → int
    # Vectorised prefix match using mx.equal + mx.cumprod
    # (direct port from exo's implementation)

  trim_cache(cache: list[KVCache], num_tokens: int) → None
    # Trim N tokens from the end of a KV cache

  make_kv_cache(model) → list[KVCache]
    # Create a fresh KV cache for the given model architecture

  estimate_cache_bytes(cache: list[KVCache]) → int
    # Estimate memory consumption: n_layers × 2 × seq_len × n_heads × head_dim × dtype_size
```

### 2.3 How It Works

#### Lookup flow (on each new request):

```
1. Tokenize prompt → prompt_tokens
2. kv_cache.get(model, prompt_tokens)
   ├── Scan all entries for longest common prefix
   ├── If exact match (minus last token): return cached KV + last token as remaining
   ├── If partial match: return trimmed KV + remaining suffix tokens
   └── If no match: return fresh KV + full prompt_tokens
3. Run prefill on remaining_tokens (may be just 1 token for exact match)
4. Run generation
5. After generation: kv_cache.add(prompt_tokens, final_cache) or kv_cache.update(...)
```

#### Eviction flow (called before each `add()`):

```
1. local_pressure = psutil.virtual_memory().percent / 100
2. if distributed group exists:
     all_pressure = mx.distributed.all_gather([local_pressure])
     pressure = max(all_pressure)
   else:
     pressure = local_pressure
3. while pressure > threshold AND cache has entries:
     pop entry with smallest last_used counter
     del entry  →  gc.collect()  →  mx.clear_cache()
     re-measure pressure
```

### 2.4 Distributed Pressure Check (Critical Detail)

This is the most important design element borrowed from exo. Currently, our
`cluster_memory_snapshot()` uses SSH to query remote nodes — it's slow (~500ms+)
and can't be called during generation without blocking.

Exo's approach is elegant:

```
# From refs/exo/src/exo/worker/engines/mlx/cache.py line 176-186
def get_memory_used_percentage(self) -> float:
    local_pressure: float = get_memory_used_percentage()
    if self._group is None:
        return local_pressure
    all_pressure = mx.distributed.all_gather(
        mx.array([local_pressure], dtype=mx.float32),
        group=self._group,
    )
    max_pressure = float(mx.max(all_pressure).item())
    return max_pressure
```

We can do exactly this because we already have `mx.distributed.Group` set up
for tensor-parallel inference. The `all_gather` is a single small message (4 bytes
per node) over the existing RDMA transport — effectively free.

**Key insight:** This means the **weakest node** controls eviction for the entire
cluster. If Node B has 2 GB free while Node A has 10 GB free, both nodes evict.
This is correct because KV caches are replicated across nodes in tensor-parallel mode.

### 2.5 Edits to Existing Files

**`server/openai_cluster_server.py`:**
- Import and instantiate `KVPrefixCache` at model load time
- Modify `_queue_worker()` to use `kv_cache.get()` before generation
- After generation completes, call `kv_cache.add()` or `kv_cache.update()`
- Pass `mx.distributed.Group` to `KVPrefixCache.__init__()` when in distributed mode
- Clear KV cache on model unload

**`server/memory_manager.py`:**
- Add `kv_cache_info()` method returning entry count, total tokens, estimated bytes
- Include KV cache stats in `snapshot()` output

**`server/dashboard.py`:**
- Show KV cache stats in dashboard (entries, hit rate, total cached tokens)

### 2.6 Integration with Existing Memory Safety

The KV prefix cache works *alongside* our existing protections, not instead of them:

| Layer | What | Still Active? |
|-------|------|--------------|
| MLX memory limit | Hard cap on Metal allocations | ✅ Yes — unchanged |
| Per-token pressure check | `generation_guard()` every 16 tokens | ✅ Yes — unchanged |
| Hard token cap | `clamp_max_tokens()` | ✅ Yes — unchanged |
| Qwen3 thinking disable | `patch_chat_template_no_thinking()` | ✅ Yes — unchanged |
| Post-request GC | `gc_cycle()` after each request | ✅ Yes — unchanged |
| **NEW: KV LRU eviction** | Evict cached KV entries when pressure rises | 🆕 Added |
| **NEW: Distributed pressure** | Weakest node triggers eviction cluster-wide | 🆕 Added |

### 2.7 Performance Impact

- **Cache hit (exact prefix match):** Skip entire prefill → saves ~200ms–2s depending
  on prompt length. For chat applications with system prompts, this is nearly every request.
- **Cache miss:** One extra scan over cached prompts (vectorised `mx.equal`). For 10
  cached entries of 2K tokens each, this is <1ms. Negligible.
- **Memory overhead:** Each cached entry stores KV tensors. For a 32-layer model with
  2K context, this is approximately `32 × 2 × 2048 × n_heads × head_dim × 2 bytes`.
  For Qwen3-8B: ~512 MB per 2K-token entry. The LRU eviction ensures this stays bounded.

### 2.8 Testing

```
# Test prefix reuse
curl -X POST localhost:8000/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 50
}'
# Second identical request should show "cache_hit: true" in logs

# Test eviction
MEMORY_THRESHOLD=0.30 make server  # very aggressive threshold
# Send a few requests, watch logs for "KV cache evicted LRU entry"

# Test distributed pressure (2-node cluster)
# Kill memory on node B, watch node A evict its cache too
```

---

## Phase S3 — Subprocess Model Runner (Optional Safety Mode)

> **Priority:** MEDIUM — provides guaranteed crash recovery at the cost of complexity
> **Effort:** ~400 lines across 2 new files + ~60 lines edits
> **Risk:** MEDIUM — significant architectural change, but opt-in
> **Exo reference:** `refs/exo/src/exo/worker/runner/runner_supervisor.py` (entire file)

### 3.1 Goals

- Run the model inference engine in a **child process** so that:
  - OOM or Metal crash kills only the child, not the API server
  - The OS guarantees all GPU memory is reclaimed when the child dies
  - The supervisor can restart the runner automatically
- **Opt-in mode** — default remains in-process for simplicity; enable with `--subprocess`
  flag or `RUNNER_MODE=subprocess` env var
- Clean IPC between API server and runner process using `multiprocessing.Queue`

### 3.2 New Files

#### `server/runner_process.py` (~200 lines)

```
The child process entry point. Receives tasks via multiprocessing Queue,
runs inference, sends results back.

Functions:
  runner_entrypoint(
    model_path: str,
    task_queue: mp.Queue,        # receives (prompt, max_tokens, request_id)
    result_queue: mp.Queue,      # sends back (request_id, chunk | done | error)
    cancel_queue: mp.Queue,      # receives request_id to cancel
    event_queue: mp.Queue,       # sends status events (loading, ready, failed)
    world_config: dict | None,   # RDMA/distributed config for multi-node
  )

Flow:
  1. Send RunnerLoading event
  2. Load model + tokenizer (with MemoryManager limits applied)
  3. Send RunnerReady event
  4. Loop:
     a. Get task from task_queue (blocking, with timeout)
     b. Run stream_generate(), sending chunks to result_queue
     c. Send RunnerDone event
     d. Check cancel_queue between tokens
  5. On exception: send RunnerFailed event, exit with non-zero code
```

#### `server/runner_supervisor.py` (~200 lines)

```
Manages the runner child process lifecycle from the API server side.

Classes:
  RunnerSupervisor
    - __init__(model_path: str, world_config: dict | None)
    - start() → None                  # spawn child process
    - stop(timeout: float = 5.0)      # graceful shutdown: join → SIGTERM → SIGKILL
    - submit_task(prompt, max_tokens, request_id) → AsyncIterator[str]
    - cancel_task(request_id)
    - is_alive: bool
    - status: RunnerStatus            # idle | loading | ready | running | failed
    - restart()                       # kill + start (after crash)

  RunnerStatus (enum)
    IDLE, LOADING, READY, RUNNING, FAILED, SHUTTING_DOWN

Shutdown sequence (matches exo's three-stage approach):
  1. Close task queue (signal child to exit)
  2. process.join(5)                  # wait 5 seconds
  3. If still alive: process.terminate()  →  process.join(1)
  4. If STILL alive: process.kill()   # SIGKILL — guaranteed death
```

### 3.3 Edits to Existing Files

**`server/openai_cluster_server.py`:**
- Add `--subprocess` CLI flag and `RUNNER_MODE` env var
- In `_lifespan()`: if subprocess mode, create `RunnerSupervisor` instead of
  loading model in-process
- In `_queue_worker()`: if subprocess mode, delegate to `supervisor.submit_task()`
  instead of calling `stream_generate()` directly
- Auto-restart on runner crash: log the failure, restart supervisor, retry the request

**`server/memory_manager.py`:**
- No changes needed — the child process creates its own `MemoryManager` instance

### 3.4 IPC Design

```
                    API Server Process                Runner Child Process
                    ─────────────────                 ────────────────────
  HTTP request →    _queue_worker()
                    │
                    ├── task_queue.put(task) ────────→ runner_entrypoint()
                    │                                  │
                    │   ┌─── result_queue.get() ◄──────┤ stream_generate()
                    │   │    (chunk by chunk)           │ yield token
                    │   │                               │
                    ├── yield SSE chunk to client       │
                    │   │                               │
                    │   └─── result_queue.get() ◄──────┤ "DONE" sentinel
                    │                                  │
                    ├── record metrics                 ▼
                    ▼
```

Uses `multiprocessing.Queue` (not pipes) for simplicity and because Queue handles
serialization automatically. Chunks are small strings (single tokens), so serialization
overhead is negligible.

### 3.5 Why This Is Opt-In

In-process mode is:
- **Simpler** — no IPC, no serialization, no process management
- **Faster** — no cross-process overhead (though it's minimal for token-level granularity)
- **Easier to debug** — single process, single debugger

Subprocess mode is better for:
- **Production deployments** — guaranteed crash recovery
- **Long-running servers** — memory leaks in the runner don't accumulate in the API server
- **Untrusted models** — runner crash can't take down the API

### 3.6 Testing

```
# Start in subprocess mode
RUNNER_MODE=subprocess make server

# Verify model loads in child process
curl localhost:8000/health | jq .runner_mode
# → "subprocess"

# Simulate crash: kill the runner child process
kill -9 $(pgrep -f runner_entrypoint)
# API server should log "Runner terminated (signal=9 (Killed))"
# and automatically restart the runner within ~5 seconds

# Normal request should work after restart
curl localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}]}'
```

---

## Phase S4 — Memory-Proportional Layer Allocation

> **Priority:** MEDIUM — enables heterogeneous clusters (e.g., M4 Max 128GB + M4 Pro 48GB)
> **Effort:** ~200 lines edits (no new files)
> **Risk:** LOW-MEDIUM — touches model loading, but only for multi-node
> **Exo reference:** `refs/exo/src/exo/master/placement_utils.py` lines 52–89

### 4.1 Goals

- Distribute model layers across nodes **proportionally to available RAM**, not evenly
- Support heterogeneous clusters where nodes have different total RAM or different
  amounts of free RAM (e.g., one node is running other processes)
- Validate that each node has enough memory for its assigned layers before loading
- Fall back to even split if memory data is unavailable

### 4.2 The Algorithm

Port exo's `allocate_layers_proportionally()` — a clean largest-remainder method:

```
Input:
  total_layers = 32 (e.g., Qwen3-8B)
  node_available = [80 GB, 30 GB]  (node A has 80 GB free, node B has 30 GB)

Step 1: Compute fractions
  total = 110 GB
  fractions = [80/110, 30/110] = [0.727, 0.273]

Step 2: Raw allocation
  raw = [0.727 × 32, 0.273 × 32] = [23.27, 8.73]

Step 3: Floor + distribute remainder by fractional part
  floored = [23, 8]  → sum = 31, remainder = 1
  fractional parts: [0.27, 0.73]  → node B gets the extra layer
  result = [23, 9]

Step 4: Validate minimum (at least 1 layer per node)
  ✅ Both nodes have ≥ 1 layer

Result: Node A gets layers 0–22, Node B gets layers 23–31
```

### 4.3 Edits to Existing Files

**`server/openai_cluster_server.py`:**
- In `sharded_load_with_fallback()`: before calling `mlx_lm.sharded_load`, compute
  proportional layer allocation based on each node's available memory
- Pass layer ranges to `sharded_load` (or implement custom layer assignment if
  `sharded_load` doesn't support unequal splits — needs investigation)
- Add CLI flag `--proportional-layers` / env var `PROPORTIONAL_LAYERS=1`

**`server/memory_manager.py`:**
- Add `allocate_layers_proportionally(total_layers: int, node_available_gb: list[float]) → list[int]`
- Add `validate_layer_allocation(allocations: list[int], node_available_gb: list[float], model_size_gb: float) → bool`

### 4.4 Compatibility Investigation Required

**Before implementing**, we need to answer:

1. **Does `mlx_lm.sharded_load` support unequal layer splits?**
   - If yes: pass layer ranges directly
   - If no: we may need to implement our own layer-range loading on top of
     `mlx_lm.load_model` with manual weight slicing

2. **Does tensor-parallel mode work with unequal splits?**
   - Tensor parallel typically requires all nodes to have ALL layers (each node
     holds a slice of each layer's weight matrices)
   - Unequal splits only make sense for **pipeline parallel** mode
   - If we're tensor-parallel only today, this phase becomes pipeline-parallel
     support + proportional allocation (bigger scope)

3. **How does JACCL handle unequal world configuration?**
   - Need to check if `jaccl` supports heterogeneous process groups

**Action item:** Investigate these questions before committing to implementation.
If tensor-parallel is the only mode we support, this phase should be deferred
until pipeline-parallel is added (Phase 6+ in the main roadmap).

### 4.5 Testing

```
# Two-node cluster: Node A (128 GB), Node B (48 GB)
PROPORTIONAL_LAYERS=1 make server

# Check allocation in logs
# Expected: "Layer allocation: Node A → 23 layers, Node B → 9 layers"

# Verify both nodes have enough headroom after load
make memory
# Both nodes should show positive headroom_gb
```

---

## Phase S5 — Request Queue Hardening & Observability

> **Priority:** HIGH — production readiness; prevents silent failures
> **Effort:** ~300 lines (1 new file + edits)
> **Risk:** LOW — additive improvements to existing queue
> **Exo reference:** General best practices (exo's task management in runner_supervisor.py)

### 5.1 Goals

- **Bounded queue** with configurable max depth + HTTP 429 when full
- **Request timeouts** — kill generation if a request exceeds wall-clock limit
- **Request cancellation** — client disconnect triggers abort
- **Per-request memory tracking** — log memory delta before/after each request
- **Persistent request log** — JSON-lines file of all requests with timing, tokens, errors
- **Metrics endpoint** — structured JSON for external monitoring (Prometheus-compatible later)

### 5.2 New File: `server/request_log.py`

```
server/request_log.py (~150 lines)

Classes:
  RequestRecord
    - request_id: str
    - timestamp: float
    - kind: "chat" | "completions"
    - model_id: str
    - prompt_tokens: int
    - generated_tokens: int
    - max_tokens_requested: int
    - wall_time_s: float
    - tokens_per_second: float
    - memory_before_gb: float
    - memory_after_gb: float
    - memory_delta_gb: float
    - kv_cache_hit: bool
    - status: "ok" | "error" | "timeout" | "cancelled" | "pressure_abort"
    - error_message: str | None

  RequestLog
    - __init__(path: str = "logs/requests.jsonl", max_entries: int = 10_000)
    - record(entry: RequestRecord)   # appends to file + in-memory ring buffer
    - recent(n: int = 50) → list[RequestRecord]
    - stats() → dict                 # aggregated: total, errors, avg_tps, p50/p95 latency
    - clear()
```

### 5.3 Edits to Existing Files

**`server/openai_cluster_server.py`:**

Queue hardening:
- Replace `asyncio.Queue()` (unbounded) with `asyncio.Queue(maxsize=MAX_QUEUE_DEPTH)`
  - Default `MAX_QUEUE_DEPTH=32` (env var configurable)
  - Return HTTP 429 `{"error": "Server busy, queue full"}` when queue is full
- Add `REQUEST_TIMEOUT_S` env var (default: 300 seconds / 5 minutes)
  - Wrap generation in `asyncio.wait_for()` or check wall clock in `_queue_worker`
  - On timeout: abort generation, send partial response with `finish_reason: "length"`
- Client disconnect detection:
  - In streaming mode: check if the SSE connection is still alive between chunks
  - If disconnected: set `_mm.request_abort()` and skip remaining generation

Per-request tracking:
- Before generation: `mem_before = _mm.active_gb()`
- After generation: `mem_after = _mm.active_gb()`
- Log `RequestRecord` with all timing/memory data
- Include `kv_cache_hit` field (from S2's cache lookup result)

New endpoints:
- `GET /requests/recent?n=50` — last N request records
- `GET /requests/stats` — aggregated statistics
- `GET /metrics` — Prometheus-style text format (future, but schema ready)

**`server/dashboard.py`:**
- Add "Request History" table to dashboard (last 20 requests)
- Show aggregate stats: total requests, error rate, avg tok/s, p95 latency
- Color-code requests by status (green=ok, red=error, yellow=timeout)

### 5.4 Environment Variables (New)

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_QUEUE_DEPTH` | `32` | Maximum pending requests before returning 429 |
| `REQUEST_TIMEOUT_S` | `300` | Wall-clock timeout per request (seconds) |
| `REQUEST_LOG_PATH` | `logs/requests.jsonl` | Path to persistent request log |
| `REQUEST_LOG_MAX` | `10000` | Max entries in memory ring buffer |

### 5.5 Testing

```
# Test queue backpressure
# Fire 50 concurrent requests at a server with MAX_QUEUE_DEPTH=5
for i in $(seq 1 50); do
  curl -s -o /dev/null -w "%{http_code}\n" \
    -X POST localhost:8000/v1/chat/completions \
    -d '{"messages":[{"role":"user","content":"Count to 100"}],"max_tokens":200}' &
done
# Should see mostly 200s for first 5, then 429s

# Test timeout
REQUEST_TIMEOUT_S=5 make server
curl localhost:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Write a novel"}],"max_tokens":99999}'
# Should timeout after 5s with partial response

# Check request log
cat logs/requests.jsonl | jq -s 'length'
# → number of completed requests

# Check stats endpoint
curl localhost:8000/requests/stats | jq .
```

---

## Implementation Order & Dependencies

```
     ┌──────────────┐
     │    Phase S1   │  Background Monitor + Tiered Thresholds
     │   ~2-3 hours  │  (foundation — no dependencies)
     └──────┬───────┘
            │
            │  S2 depends on S1 for live pressure data
            ▼
     ┌──────────────┐
     │    Phase S2   │  KV Prefix Cache + LRU Eviction
     │   ~6-8 hours  │  (biggest impact — needs S1's live pressure)
     └──────┬───────┘
            │
            │  S5 can start in parallel with S2
            │  (independent, but benefits from S2's cache_hit metric)
            │
     ┌──────┴───────┐
     │              │
     ▼              ▼
┌──────────┐  ┌──────────────┐
│ Phase S3 │  │   Phase S5   │  Request Queue Hardening
│ ~4 hours │  │   ~3 hours   │  (can parallelize with S3)
│Subprocess│  │              │
│ Runner   │  │              │
└──────────┘  └──────────────┘
     │              │
     └──────┬───────┘
            ▼
     ┌──────────────┐
     │    Phase S4   │  Proportional Layer Allocation
     │   ~3 hours   │  (needs investigation first; may defer)
     │  (if viable) │
     └──────────────┘
```

### Recommended Sprint Plan

| Sprint | Phases | Duration | Deliverable |
|--------|--------|----------|-------------|
| **Sprint 1** | S1 + S5 | 1 day | Live monitoring, tiered thresholds, queue hardening, request log |
| **Sprint 2** | S2 | 1–2 days | KV prefix cache with distributed eviction |
| **Sprint 3** | S3 | 1 day | Optional subprocess runner mode |
| **Sprint 4** | S4 (investigate) | 0.5 day | Proportional layers (or defer with findings documented) |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| KV cache integration breaks `mlx_lm.stream_generate` | Medium | High | Test with multiple model architectures (Qwen3, Llama, Mistral); keep in-process `stream_generate` as fallback with cache disabled |
| `mx.distributed.all_gather` blocks during generation | Low | Medium | The all_gather is a single float32 — should complete in <1ms over RDMA; add timeout wrapper |
| Subprocess IPC overhead reduces tok/s | Low | Low | Benchmark in-process vs subprocess; overhead should be <1% for token-level messages |
| `multiprocessing.Queue` serialization of MLX arrays | Medium | Medium | Don't send MLX arrays over IPC — send only text tokens and control messages |
| Proportional layers not supported by `sharded_load` | High | Low | Defer S4; document findings; propose pipeline-parallel mode |
| Background monitor thread contention with GIL | Low | Low | Monitor only reads `mx.get_active_memory()` (C call, releases GIL) and sleeps |

---

## Success Criteria

### Phase S1 — Background Monitor
- [ ] `GET /memory/live` returns 60 data points updated every 1.5s
- [ ] Tiered threshold matches machine size (verify on 48 GB and 128 GB machines)
- [ ] `OVERRIDE_MEMORY_MB=16384` correctly simulates 16 GB available
- [ ] Dashboard shows live pressure graph sourced from monitor (not separate macmon process)
- [ ] No measurable impact on generation tok/s (monitor runs in background thread)

### Phase S2 — KV Prefix Cache
- [ ] Repeated identical chat prompts show >90% prefill time reduction
- [ ] Memory pressure above threshold triggers LRU eviction (visible in logs)
- [ ] 2-node distributed: high pressure on Node B triggers eviction on Node A
- [ ] Cache hit/miss rate visible in `/memory` endpoint and dashboard
- [ ] Existing memory safety guards still function (hard token cap, pressure check, etc.)
- [ ] No generation correctness regression (outputs match non-cached generation)

### Phase S3 — Subprocess Runner
- [ ] `RUNNER_MODE=subprocess make server` starts model in child process
- [ ] `kill -9 <runner_pid>` is detected; server logs failure and auto-restarts
- [ ] After runner restart, next request succeeds
- [ ] `GET /health` reports `runner_mode: "subprocess"` and `runner_status: "ready"`
- [ ] tok/s within 5% of in-process mode

### Phase S4 — Proportional Layers
- [ ] Investigation document answers compatibility questions (sharded_load, tensor vs pipeline)
- [ ] If viable: 2-node heterogeneous cluster loads with proportional split
- [ ] If not viable: documented reason and timeline for pipeline-parallel support

### Phase S5 — Queue Hardening
- [ ] Queue full returns HTTP 429 (not hang or crash)
- [ ] Request timeout produces partial response with `finish_reason: "length"`
- [ ] Client disconnect aborts generation within 1 second
- [ ] `logs/requests.jsonl` contains complete request records
- [ ] `GET /requests/stats` returns aggregate metrics
- [ ] Dashboard shows request history table

---

## Appendix: Exo Reference File Map

These are the key exo files to study for each phase. All paths are relative to `refs/exo/`.

| Phase | Exo File | What to Study |
|-------|----------|---------------|
| S1 | `src/exo/utils/info_gatherer/info_gatherer.py` L462-478 | `_monitor_memory_usage()` polling loop, `OVERRIDE_MEMORY_MB` |
| S1 | `src/exo/worker/engines/mlx/cache.py` L30-38 | `_default_memory_threshold()` tiered by RAM |
| S1 | `src/exo/shared/types/memory.py` | Clean `Memory` value type with arithmetic |
| S2 | `src/exo/worker/engines/mlx/cache.py` L73-186 | `KVPrefixCache` — add, get, update, evict |
| S2 | `src/exo/worker/engines/mlx/cache.py` L176-186 | `get_memory_used_percentage()` — distributed all_gather |
| S2 | `src/exo/worker/engines/mlx/cache.py` L189-203 | `trim_cache()` and `get_prefix_length()` |
| S3 | `src/exo/worker/runner/runner_supervisor.py` | Full `RunnerSupervisor` class — spawn, IPC, 3-stage shutdown |
| S3 | `src/exo/worker/runner/bootstrap.py` | `entrypoint()` — child process entry point |
| S4 | `src/exo/master/placement_utils.py` L52-89 | `allocate_layers_proportionally()` — largest remainder method |
| S4 | `src/exo/master/placement_utils.py` L92-130 | `_allocate_and_validate_layers()` — memory validation |
| S4 | `src/exo/master/placement.py` | `place_instance()` — cycle selection by available RAM |
| S5 | `src/exo/worker/runner/runner_supervisor.py` L119-145 | `start_task()` / `cancel_task()` — task lifecycle |
| S5 | `src/exo/shared/types/tasks.py` | `Task`, `TaskStatus` types |

---

## What's NOT In This Plan

These are important but belong in the main [roadmap.md](roadmap.md), not here:

- **Tool calls / function calling** → Roadmap Phase 4
- **Ollama API compatibility** → Roadmap Phase 5
- **Image generation** → Non-goal for now
- **Continuous batching** → Roadmap Phase 6 (requires KV cache from S2 first)
- **Speculative decoding** → Roadmap Phase 6
- **Pipeline-parallel mode** → Prerequisite for S4 if tensor-parallel can't do unequal splits
- **Dashboard v2 redesign** → Roadmap Phase 2 (this plan adds to existing dashboard)

---

> **Next step:** Start Phase S1 (Background Memory Monitor + Tiered Thresholds).
> It's the foundation everything else builds on, it's low-risk, and it can be
> completed in a single session.