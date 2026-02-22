# Memory Management & Kernel Panic Prevention

> How mlx-jaccl-cluster protects your Mac from GPU memory exhaustion.
> Last updated: 2025-07-23
>
> **See also:** [Memory Handling: exo vs mlx-jaccl-cluster](memory-comparison-exo.md) — deep comparison of how exo handles memory measurement, KV cache eviction, model loading/unloading, and recommended improvements we can adopt.

---

## Table of Contents

- [1. The Problem — IOGPUGroupMemory Kernel Panic](#1-the-problem--iogpugroupmemory-kernel-panic)
- [2. Root Cause Analysis](#2-root-cause-analysis)
- [3. The MemoryManager Module](#3-the-memorymanager-module)
- [4. Memory Limits — What Gets Set and Why](#4-memory-limits--what-gets-set-and-why)
- [5. Model Load / Unload Lifecycle](#5-model-load--unload-lifecycle)
- [6. Generation Safety Guards](#6-generation-safety-guards)
- [7. Qwen3 Thinking Mode](#7-qwen3-thinking-mode)
- [8. API Endpoints](#8-api-endpoints)
- [9. Makefile Targets](#9-makefile-targets)
- [10. Environment Variables](#10-environment-variables)
- [11. Tuning Guide](#11-tuning-guide)
- [12. Troubleshooting](#12-troubleshooting)
- [13. Technical Details — Metal Buffer Lifecycle](#13-technical-details--metal-buffer-lifecycle)
- [14. Files Changed](#14-files-changed)

---

## 1. The Problem — IOGPUGroupMemory Kernel Panic

Running distributed inference on Apple Silicon can trigger a **macOS kernel panic** that
instantly shuts down the machine with no warning or opportunity to save state:

```
panic(cpu 10 caller 0xfffffe0042d90c3c):
  "Memory object unexpectedly not found in fPendingMemorySet"
  @IOGPUGroupMemory.cpp:219
```

This is not a software crash — it is a kernel-level assertion failure in Apple's GPU
driver (`IOGPUFamily`). The panicked process is `python3.12` running MLX inference.
The machine reboots immediately.

This document explains why it happens, how we prevent it, and how to operate the
cluster safely.

---

## 2. Root Cause Analysis

### The Chain of Events

```
1. MLX allocates Metal GPU buffers
   (model weights, KV cache, activations, compilation cache)
        │
2. Qwen3 thinking mode generates unbounded <think> tokens
   (KV cache grows ~0.5–2 MB per token, thousands of tokens)
        │
3. Total Metal allocations approach or exceed physical RAM (48 GB)
        │
4. macOS enters critical memory pressure
   (compressor full, swap saturated)
        │
5. IOGPUFamily driver attempts emergency Metal buffer eviction
        │
6. Driver's internal bookkeeping (fPendingMemorySet) gets out of sync
   with force-evicted buffers — a buffer it expects to find is gone
        │
7. KERNEL PANIC — machine reboots instantly
```

### Why MLX's Defaults Are Dangerous

On a 48 GB M4 Pro, the relevant numbers are:

| Metric | Value |
|--------|-------|
| Total physical RAM | 48.0 GB |
| Apple's max recommended working set | 37.4 GB |
| MLX default memory limit (1.5× recommended) | **56.1 GB** |
| macOS system overhead | ~4–6 GB |
| Safe headroom needed | ~8–14 GB |

**MLX's default limit of 56 GB exceeds the 48 GB of physical RAM.** The framework will
happily allocate up to 56 GB before complaining, but the machine has already panicked
long before that.

### Contributing Factors

1. **Qwen3 thinking mode**: Generates unbounded `<think>` sequences (often 1,000–10,000+
   tokens) before producing the actual answer. Each token grows the KV cache.

2. **No model unload**: Before this fix, the server loaded a model once at startup and
   kept it forever. No way to free memory without killing the process.

3. **No generation guards**: No memory checks during token generation. The KV cache grew
   until the kernel panicked.

4. **No post-request cleanup**: KV cache memory from completed requests was not reclaimed
   between requests, accumulating over time.

---

## 3. The MemoryManager Module

**File:** `server/memory_manager.py`

The MemoryManager is a single class that handles all memory safety for the cluster:

```
┌──────────────────────────────────────────────────────┐
│                   MemoryManager                       │
│                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ Limit Setup  │  │  Model Slot  │  │  Pressure   │ │
│  │              │  │              │  │  Monitor    │ │
│  │ memory_limit │  │ model ref    │  │             │ │
│  │ cache_limit  │  │ tokenizer    │  │ check()     │ │
│  │ wired_limit  │  │ metadata     │  │ abort()     │ │
│  │ max_tokens   │  │ load/unload  │  │ emergency() │ │
│  └─────────────┘  └──────────────┘  └─────────────┘ │
│                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Qwen3 Safe  │  │  GC Cycle    │  │  Snapshot   │ │
│  │              │  │              │  │             │ │
│  │ detect model │  │ gc.collect() │  │ active_gb   │ │
│  │ patch tmpl   │  │ clear_cache  │  │ peak_gb     │ │
│  │ no_thinking  │  │ reclaim mem  │  │ pressure_%  │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
└──────────────────────────────────────────────────────┘
```

### Initialization

The MemoryManager is created **before any model load** — this is critical because it
probes actual free RAM via `vm_stat` / `sysctl` and sets MLX memory limits based on
what is **really available right now**, not just total RAM:

```
from memory_manager import init_manager

mm = init_manager()
# Probes actual free RAM → e.g. 32.8 GB available (15.2 GB already used by macOS + apps)
# Computes 4 candidate limits, picks the smallest (safest):
#   mx.set_memory_limit(29.8 GB)   ← adaptive, was 45.6 GB (dangerous!)
#   mx.set_cache_limit(14.9 GB)    ← adaptive
```

### When Limits Are (Re-)Calculated

| Event | What Happens |
|-------|-------------|
| `MemoryManager.__init__()` | Probes free RAM via `vm_stat`, sets initial limits |
| `mm.load_model(...)` | **Re-probes** free RAM before loading, readjusts limits if system state changed |
| `mm.readjust_limits()` | Manual re-probe — call anytime to adapt to changed conditions |
| During generation | Checks `active_memory / limit` every 16 tokens, aborts if critical |

This means if the user closes Chrome between init and model load, the limit goes
**up**. If Spotlight indexing kicks in, the limit goes **down**. The system always
reflects reality.

### Key Properties

```
mm.model          # currently loaded model (or raises ModelNotLoadedError)
mm.tokenizer      # currently loaded tokenizer
mm.model_id       # e.g. "Qwen3-8B-4bit"
mm.model_loaded   # bool
mm.active_gb()    # current Metal memory usage in GB
mm.peak_gb()      # peak Metal memory usage in GB
mm.pressure_pct() # usage as % of limit (0–100)
mm.headroom_gb()  # GB available before hitting limit
mm.slot_info      # dict with model metadata
mm.snapshot()     # full MemorySnapshot dataclass (includes system_used_gb, actual_available_gb)
mm.cluster_memory_snapshot()  # all nodes in the cluster (local + remote via SSH)
mm.readjust_limits()          # re-probe free RAM and update limits
```

---

## 4. Memory Limits — What Gets Set and Why

### Adaptive Calculation (4 Candidates)

Every time limits are computed (at init AND before each model load), the manager
probes actual free RAM via `vm_stat` and evaluates **four** candidate limits.
The **smallest** wins — whichever is safest for the current system state:

| # | Method | Formula | Example (48 GB, 15 GB used) |
|---|--------|---------|---------------------------|
| 1 | Fraction of total (static ceiling) | `total × 0.70` | 33.6 GB |
| 2 | Total minus reserve (static floor) | `total − 6 GB` | 42.0 GB |
| 3 | Apple's recommendation | `max_recommended_working_set` | 37.4 GB |
| 4 | **Adaptive available** | **`actual_free − 3 GB safety`** | **29.8 GB** ← wins |

On a clean boot with only 5 GB used, Method 1 (33.6 GB) would win.
On a busy system with 15 GB used, Method 4 (adaptive) wins — because
33.6 + 15 = 48.6 GB would exceed physical RAM and risk a kernel panic.

The 3 GB "safety margin" in Method 4 accounts for macOS memory fluctuations
after the limit is set (Spotlight indexing, daemon spikes, etc.).

### Real Output (Measured on Your Machine)

```
Memory limits applied on Apple M4 Pro (ADAPTIVE):
  Total RAM:         48.0 GB
  System used now:   15.2 GB  (non-MLX: macOS + apps)
  Actually available: 32.8 GB
  Max recommended:   37.4 GB  (Apple's guidance)
  ──────────────────────────────────────
  Candidates:
    fraction_of_total         →   33.6 GB
    total_minus_reserve       →   42.0 GB
    apple_recommended         →   37.4 GB
    adaptive_available        →   29.8 GB ← selected
  ──────────────────────────────────────
  MLX memory limit:  29.8 GB  (was 45.6 GB)  [chosen by: adaptive_available]
  MLX cache limit:   14.9 GB  (was 45.6 GB)
```

### Limits Readjust Before Model Load

When `load_model()` is called, it re-probes free RAM **before** loading weights.
If the system state changed since init, the limit changes:

```
# At init (Chrome + Xcode running):
#   System used: 15.2 GB → limit: 29.8 GB

# User closes Chrome, then loads model:
#   System used: 8.1 GB  → limit: 33.6 GB (fraction_of_total wins now)

# Spotlight indexing starts during load:
#   System used: 18.0 GB → limit: 27.0 GB (adaptive wins, tighter)
```

### Before vs After

| Setting | Before (MLX defaults) | After (MemoryManager) |
|---------|----------------------|----------------------|
| Memory limit | 56.1 GB (above physical RAM!) | **29.8 GB** (adaptive to actual free RAM) |
| Cache limit | 56.1 GB | 14.9 GB (adaptive) |
| Limit calculation | Static (1.5× recommended) | Adaptive (4 candidates, smallest wins) |
| Re-check before load | never | yes — re-probes `vm_stat` before every load |
| Re-check during generation | never | every 16 tokens (pressure guard) |
| Cross-node awareness | none | queries remote nodes via SSH |
| Wired limit | 0 (unset) | 0 (configurable) |
| System reserve | 0 (none) | 6.0 GB minimum + 3.0 GB safety margin |
| Max tokens | unlimited | 4,096 |
| Post-request GC | none | gc.collect() + mx.clear_cache() |
| Model unload | not possible | full Metal buffer cleanup |

### Cross-Node Memory Awareness

The manager reads the hostfile and can query all nodes via SSH:

```
cluster = mm.cluster_memory_snapshot()
# Returns:
# {
#   "node_count": 2,
#   "nodes": [
#     {"hostname": "macstudio1", "total_gb": 48.0, "free_gb": 12.0, "used_gb": 15.2, ...},
#     {"hostname": "macstudio2", "total_gb": 48.0, "free_gb": 14.5, "used_gb": 13.1, ...}
#   ],
#   "cluster_total_gb": 96.0,
#   "cluster_available_gb": 26.5,
#   "weakest_node": "macstudio1",
#   "weakest_available_gb": 12.0
# }
```

Each node runs its own MemoryManager independently — limits are per-node because
each node has its own physical RAM. But the cluster snapshot lets you see the full
picture and identify the weakest node before it panics.

---

## 5. Model Load / Unload Lifecycle

### Loading a Model

```
model, tok = mm.load_model(
    model_path="/Users/you/models_mlx/Qwen3-8B-4bit",
    world=world,           # mx.distributed world (or None for single-node)
    model_id="Qwen3-8B",   # human-readable name
    lazy=False,             # eager load for JACCL safety
)
```

**Steps performed internally:**

1. **Unload previous model** (if any) — auto-calls `unload_model()`
2. **Pre-flight memory check** — estimates model size from disk, checks headroom
3. **Load weights** (eager mode — no lazy tensors, avoids JACCL eval deadlock)
4. **Barrier + shard** across distributed nodes (if `world.size() > 1`)
5. **Post-shard barrier** — both ranks synchronized
6. **Load tokenizer**
7. **Record metadata** — model size, load time, strategy, timestamps

### Unloading a Model

This is the critical operation. Getting it wrong means Metal buffers leak and
memory accumulates until the next kernel panic.

```
# CRITICAL: caller must delete their references FIRST
_model = None   # clear server global
_tok = None     # clear server global

result = mm.unload_model()
# Returns: {"status": "unloaded", "freed_gb": 4.291, ...}
```

**Steps performed internally:**

```
Step 1: Save metadata, destroy ModelSlot (drops MemoryManager's refs)
Step 2: Explicit `del` on model + tokenizer objects
        → Python refcount hits 0 → Metal buffers freed immediately
Step 3: gc.collect() × 3 (catches nn.Module reference cycles)
Step 4: mx.clear_cache() (returns freed Metal buffers to OS)
Step 5: Sleep 50ms (Metal allocator processes frees asynchronously)
Step 6: Second gc.collect() + mx.clear_cache() pass
Step 7: mx.reset_peak_memory()
```

### Why `del` Matters

MLX array objects (`mx.array`) release their backing Metal allocations when the
Python reference count drops to zero. Setting a variable to `None` is **not enough**
if another reference exists elsewhere.

This is why the caller must clear their globals before calling `unload_model()`:

```
# WRONG — model still referenced by _model global, freed_gb = 0
result = mm.unload_model()
_model = None  # too late, Metal buffers still allocated

# RIGHT — globals cleared first, freed_gb = 4.291
_model = None
_tok = None
result = mm.unload_model()  # Metal buffers freed during this call
```

### Load/Unload Cycle (Verified)

Tested on M4 Pro 48 GB with Qwen3-8B-4bit:

```
Before load:  0.000 GB active
After load:   4.291 GB active  (12.8% pressure)
After unload: 0.000 GB active  (4.291 GB freed)
After reload: 4.291 GB active  (12.8% pressure)
After unload: 0.000 GB active  (4.291 GB freed)
```

Memory returns to zero every time. No leaks.

---

## 6. Generation Safety Guards

### Token Clamping

Every request's `max_tokens` is clamped before generation starts:

| Requested | Clamped To | Reason |
|-----------|-----------|--------|
| `None` or `0` | 512 | Safe default |
| `64` | 64 | Within limit |
| `4096` | 4096 | At limit |
| `8192` | 4096 | Hard cap applied |
| `100000` | 4096 | Hard cap applied |

### Pressure Checks During Generation

Every 16 tokens during `stream_generate()`, the memory manager checks:

1. **Memory pressure**: `active_memory / memory_limit`
2. **Token count** vs hard limit
3. **Abort flag** (can be set externally by health monitoring)

If pressure exceeds 92%, generation is **aborted immediately** with a clean error
response instead of letting the system reach a kernel panic:

```
{
  "error": "CRITICAL memory pressure during stream_generate/chat: 
            31.47 GB active = 93.7% of 33.6 GB limit. 
            Aborting to prevent kernel panic."
}
```

### Pre-Request Rejection

Before a request even enters the generation queue, the memory pressure is checked.
If it's already critical, the request is rejected with HTTP 507 (Insufficient Storage):

```
HTTP/1.1 507 Insufficient Storage
{
  "detail": "Memory pressure too high to accept request: ..."
}
```

### Post-Request Cleanup

After every completed (or aborted) request, the queue worker runs:

```
mm.gc_cycle(reason="chat request done")
# → gc.collect() + mx.clear_cache()
# → KV cache memory from the request is reclaimed
```

This prevents memory from accumulating across requests.

---

## 7. Qwen3 Thinking Mode

### The Problem

Qwen3 models include a "thinking" mode in their chat template. When enabled (the
default), the model generates `<think>...</think>` sequences that can be thousands
of tokens long before producing the actual answer.

This is catastrophic for memory:

| Scenario | Tokens Generated | Approx KV Cache | Risk |
|----------|-----------------|-----------------|------|
| Thinking disabled | 50–500 | 25–250 MB | Low |
| Thinking enabled, simple question | 500–2,000 | 250 MB–1 GB | Medium |
| Thinking enabled, complex question | 2,000–10,000+ | 1–5+ GB | **Kernel panic** |

### The Fix

The MemoryManager auto-detects Qwen3 models and disables thinking by default:

```
# Auto-detection
mm.should_disable_thinking("Qwen3-8B-4bit")    # → True
mm.should_disable_thinking("Qwen2.5-7B")       # → False
mm.should_disable_thinking("Llama-3-8B")        # → False
mm.should_disable_thinking("qwen3-72B")         # → True
```

When building chat prompts, the manager passes `enable_thinking=False` to the
tokenizer's `apply_chat_template()`. This produces a prompt with an empty
`<think></think>` block, signaling the model to skip thinking and answer directly.

### Re-Enabling Thinking

If you want thinking mode (and accept the memory risk), set:

```
export QWEN3_ENABLE_THINKING=1
```

When doing this, also increase the hard max tokens limit:

```
export MLX_HARD_MAX_TOKENS=16384
```

⚠️ **Warning**: With thinking enabled, monitor memory closely via `make memory`
and be prepared for the generation to be aborted by the pressure guard.

---

## 8. API Endpoints

All endpoints are available on rank 0 only.

### `GET /memory`

Full memory status including MLX allocations, macOS system pressure, model slot
info, and configured limits.

```
curl -s http://localhost:8080/memory | python3 -m json.tool
```

Response:

```
{
    "ok": true,
    "mlx": {
        "timestamp": 1771758796.19,
        "active_gb": 4.291,
        "peak_gb": 5.102,
        "cache_gb": 0.812,
        "limit_gb": 33.6,
        "cache_limit_gb": 16.8,
        "total_ram_gb": 48.0,
        "system_reserve_gb": 6.0,
        "pressure_pct": 12.8,
        "model_loaded": true,
        "model_id": "Qwen3-8B-4bit",
        "model_size_gb": 4.291
    },
    "system": {
        "available": true,
        "pressure_level": "normal",
        "free_gb": 12.02,
        "wired_gb": 2.85,
        "compressed_gb": 0.0
    },
    "slot": {
        "loaded": true,
        "model_id": "Qwen3-8B-4bit",
        "strategy": "Tensor Parallelism x2",
        "load_time_s": 3.42,
        "size_gb": 4.291,
        "uptime_s": 1204.5
    },
    "limits": {
        "memory_limit_gb": 33.6,
        "cache_limit_gb": 16.8,
        "hard_max_tokens": 4096,
        "system_reserve_gb": 6.0
    }
}
```

### `GET /model/info`

Metadata about the currently loaded model.

```
curl -s http://localhost:8080/model/info | python3 -m json.tool
```

### `POST /model/unload`

Unload the current model and free all GPU memory.

```
curl -s -X POST http://localhost:8080/model/unload | python3 -m json.tool
```

Response:

```
{
    "status": "unloaded",
    "model_id": "Qwen3-8B-4bit",
    "before_active_gb": 4.291,
    "after_active_gb": 0.000,
    "freed_gb": 4.291,
    "memory_after": { ... }
}
```

**Constraints:**
- Rejects with 409 if requests are queued (wait for them to finish first)
- After unload, inference requests will fail until a new model is loaded

### `POST /model/load`

Load a new model (automatically unloads any existing model first).

```
curl -s -X POST http://localhost:8080/model/load \
  -H 'Content-Type: application/json' \
  -d '{"model_dir": "/Users/you/models_mlx/Qwen3-8B-4bit"}' \
  | python3 -m json.tool
```

Body:

```
{
    "model_dir": "/path/to/model",    // required
    "model_id": "optional-name"       // defaults to directory name
}
```

**Constraints:**
- Rejects with 409 if requests are queued
- Rejects with 507 if not enough memory headroom
- Rejects with 400 if model_dir doesn't exist

**Note for distributed mode:** Workers must also load the new model. Currently,
workers must be restarted to pick up a model change. A future enhancement will
broadcast model change commands to workers.

### `POST /model/gc`

Force a garbage collection + Metal cache clear cycle without unloading the model.
Use this to reclaim KV cache memory between heavy requests.

```
curl -s -X POST http://localhost:8080/model/gc | python3 -m json.tool
```

### `GET /health` (enhanced)

The health endpoint now includes memory information:

```
{
    "ok": true,
    "model_loaded": true,
    "memory": {
        "active_gb": 4.291,
        "peak_gb": 5.102,
        "limit_gb": 33.6,
        "pressure_pct": 12.8,
        "headroom_gb": 29.31
    },
    "memory_safe": true
}
```

The `memory_safe` field is `false` when pressure exceeds 80% — use this in
load balancers or monitoring to divert traffic.

---

## 9. Makefile Targets

### `make memory`

Show GPU/unified memory status with color-coded pressure indicator:

```
  🧠 Memory Status

  MLX Active:      4.291 GB
  MLX Peak:        5.102 GB
  MLX Cache:       0.812 GB
  Memory Limit:    33.6 GB
  Cache Limit:     16.8 GB
  System Reserve:  6 GB
  Total RAM:       48 GB
  Pressure:        12.8%          ← green < 70%, yellow < 85%, red ≥ 85%
  Hard Max Tokens: 4096

  Model Loaded:    True
  Model ID:        Qwen3-8B-4bit
  Model Size:      4.29 GB
  Strategy:        Tensor Parallelism x2

  macOS Pressure:  normal
  System Free:     12.02 GB
  System Wired:    2.85 GB
```

### `make model-unload`

Unload the current model and free GPU memory:

```
  ⚠ Unloading model...

  ✓ Unloaded: Qwen3-8B-4bit
    Freed:  4.291 GB
    Before: 4.291 GB active
    After:  0.000 GB active
```

### `make model-load MODEL_DIR=~/models_mlx/NewModel`

Load a model via the API:

```
  Loading model: /Users/you/models_mlx/NewModel

  ✓ Loaded: NewModel
    Strategy:  Tensor Parallelism x2
    Size:      4.29 GB
    Active:    4.291 GB
    Pressure:  12.8%
```

### `make model-swap MODEL_DIR=~/models_mlx/NewModel`

Safe three-step model swap (unload → gc → load):

```
  ⚡ Safe Model Swap
  ─────────────────────────────────────────────

  Step 1/3: Unloading current model...
    Freed: 4.291 GB
  Step 2/3: Garbage collection...
    Active: 0.000 GB
  Step 3/3: Loading /Users/you/models_mlx/NewModel...

  ✓ Swap complete: NewModel
    Active: 4.291 GB  |  Pressure: 12.8%
```

### `make model-gc`

Force GC without unloading:

```
  Running GC cycle...
  ✓ Freed 0.812 GB
    Active: 5.102 → 4.291 GB
    Cache:  0.812 → 0.000 GB
```

### `make model-info`

Show loaded model metadata as JSON.

---

## 10. Environment Variables

All variables are optional. Defaults are tuned for 48 GB M4 Pro.

### Memory Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `MLX_SYSTEM_RESERVE_GB` | `6.0` | RAM reserved for macOS (WindowServer, Finder, daemons) |
| `MLX_MAX_RAM_FRACTION` | `0.70` | Maximum fraction of physical RAM for MLX |
| `MLX_CACHE_FRACTION` | `0.50` | Cache limit as fraction of memory limit |
| `MLX_WIRED_LIMIT_GB` | `0` | Metal wired memory limit (0 = don't set; macOS 15+ only) |

### Generation Safety

| Variable | Default | Description |
|----------|---------|-------------|
| `MLX_HARD_MAX_TOKENS` | `4096` | Absolute maximum tokens per request |
| `MLX_PRESSURE_WARN` | `0.80` | Log warning when usage exceeds this fraction of limit |
| `MLX_PRESSURE_CRITICAL` | `0.92` | Abort generation when usage exceeds this fraction |

### Model Behavior

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_ENABLE_THINKING` | `0` | Set to `1` to allow Qwen3 thinking mode (dangerous) |

### Recommendations by Machine Size

| Machine | RAM | `MLX_MAX_RAM_FRACTION` | `MLX_SYSTEM_RESERVE_GB` | Effective Limit |
|---------|-----|----------------------|------------------------|----------------|
| Mac Mini M4 Pro | 48 GB | 0.70 | 6.0 | 33.6 GB |
| Mac Mini M4 Pro | 64 GB | 0.75 | 6.0 | 48.0 GB |
| Mac Studio M4 Ultra | 128 GB | 0.80 | 8.0 | 102.4 GB |
| Mac Studio M4 Ultra | 192 GB | 0.80 | 8.0 | 153.6 GB |
| MacBook Pro M4 Max | 128 GB | 0.70 | 8.0 | 89.6 GB |

For machines with more RAM, you can safely increase the fraction since macOS has
proportionally more headroom.

---

## 11. Tuning Guide

### For Maximum Throughput (Accept More Risk)

```
export MLX_MAX_RAM_FRACTION=0.80        # 38.4 GB limit on 48 GB
export MLX_CACHE_FRACTION=0.60          # 23.0 GB cache
export MLX_HARD_MAX_TOKENS=8192         # allow longer generations
export MLX_PRESSURE_CRITICAL=0.95       # abort later
```

### For Maximum Safety (Production)

```
export MLX_MAX_RAM_FRACTION=0.60        # 28.8 GB limit on 48 GB
export MLX_CACHE_FRACTION=0.40          # 11.5 GB cache
export MLX_HARD_MAX_TOKENS=2048         # short generations only
export MLX_PRESSURE_CRITICAL=0.85       # abort early
export MLX_SYSTEM_RESERVE_GB=8.0        # extra headroom for macOS
```

### For Benchmarking

```
export MLX_MAX_RAM_FRACTION=0.75        # balanced
export MLX_HARD_MAX_TOKENS=4096         # standard bench length
export QWEN3_ENABLE_THINKING=0          # deterministic output
export BENCH_RUNS=5                     # more runs for statistics
```

---

## 12. Troubleshooting

### "Memory object unexpectedly not found in fPendingMemorySet"

**This is the kernel panic.** If you see this in a panic log after a reboot:

1. The memory manager was not active, or its limits were too high
2. Check that `memory_manager.py` is importable from the server directory
3. Verify limits: start the server and run `make memory` — confirm
   the limit is well below your physical RAM

### Server Shows "running WITHOUT memory limits"

The `memory_manager` module failed to import. Check:

```
cd mlx-jaccl-cluster
.venv/bin/python3 -c "import sys; sys.path.insert(0, 'server'); from memory_manager import MemoryManager; print('OK')"
```

### Model Unload Reports "freed_gb: 0.0"

The caller still holds a reference to the model. In server code, make sure
`_model = None` and `_tok = None` are set **before** calling `mm.unload_model()`.

### Memory Doesn't Return to Zero After Unload

1. Other Python objects may reference model tensors (e.g., cached KV states)
2. Run `make model-gc` after unload to force a second cleanup pass
3. Check for any background threads that might hold references
4. As a last resort, restart the server process (`make server-restart`)

### Generation Aborted by Pressure Guard

```
CRITICAL memory pressure during stream_generate/chat:
31.47 GB active = 93.7% of 33.6 GB limit.
```

This means the KV cache grew too large during generation. Options:

1. Reduce `max_tokens` in the request
2. Ensure Qwen3 thinking is disabled (`QWEN3_ENABLE_THINKING=0`)
3. Run `make model-gc` to reclaim cache between requests
4. Increase the memory limit if you have headroom (check `make memory`)
5. Use a smaller or more aggressively quantized model

### macOS System Pressure Shows "critical"

The system-level memory pressure (not just MLX) is critical. This can happen even
if MLX's own pressure is below threshold, because other apps are using RAM.

1. Close memory-hungry apps (browsers, IDEs, Docker)
2. Run `make model-unload` to free model memory
3. If pressure persists, reduce `MLX_MAX_RAM_FRACTION` and
   increase `MLX_SYSTEM_RESERVE_GB`

---

## 13. Technical Details — Metal Buffer Lifecycle

### How MLX Manages GPU Memory on Apple Silicon

Apple Silicon uses **unified memory** — CPU and GPU share the same physical RAM.
There is no discrete VRAM. MLX allocates Metal buffers through Apple's Metal API.

```
Python mx.array object
    │
    ├── Holds reference to MLX internal buffer
    │       │
    │       └── Holds reference to Metal MTLBuffer
    │               │
    │               └── Backed by unified memory page(s)
    │
    └── When Python refcount → 0:
            │
            ├── MLX buffer released
            │       │
            │       └── Metal buffer released (or moved to MLX cache)
            │
            └── If in MLX cache:
                    │
                    └── mx.clear_cache() → MTLBuffer deallocated
                            │
                            └── Physical pages returned to macOS
```

### Why `del` is More Reliable Than `= None`

```
# Setting to None:
_model = None
# Only removes ONE reference. If any other variable, closure, or data structure
# still points to the model, the refcount stays > 0, Metal buffers stay allocated.

# Explicit del:
del _model
# Also removes one reference, but signals intent more clearly. The key insight is
# that ALL references must be cleared before Metal buffers are freed.
```

### The `gc.collect()` Triple-Pass

Neural network modules (`nn.Module`) typically have reference cycles:

```
model.layers[0].parent = model        # parent → child → parent cycle
model.layers[0].attention.proj.weight  # deep nesting
```

Python's garbage collector handles cycles, but needs multiple passes:

- **Pass 1**: Breaks top-level cycles (model ↔ layers)
- **Pass 2**: Breaks nested cycles (attention ↔ projection)
- **Pass 3**: Catches finalizer-created references

### The `mx.clear_cache()` Step

Even after Python objects are freed and GC runs, MLX maintains a **free-list cache**
of Metal buffers for performance (reusing allocations is faster than asking Metal for
new ones). `mx.clear_cache()` returns these cached buffers to Metal, which returns the
physical pages to macOS.

Without `clear_cache()`, memory appears freed in Python but the physical pages are
still wired by Metal.

---

## 14. Files Changed

### New Files

| File | Purpose |
|------|---------|
| `server/memory_manager.py` | Core memory safety module (~1,000 lines) |
| `docs/memory-management.md` | This document |

### Modified Files

| File | Changes |
|------|---------|
| `server/openai_cluster_server.py` | Integrated MemoryManager: safe limits on startup, model load/unload endpoints, pressure guards in generation loop, post-request GC, Qwen3 thinking disable, enhanced health endpoint |
| `scripts/jaccl_tps_bench.py` | Memory limits on startup, pressure checks during bench runs, model unload after bench, fallback manual limits if MemoryManager unavailable |
| `Makefile` | Added targets: `memory`, `model-info`, `model-unload`, `model-load`, `model-gc`, `model-swap` |

### Architecture Decisions

1. **Conservative defaults**: 70% RAM fraction leaves 14.4 GB for macOS — more than
   enough, even under heavy system load.

2. **Abort over crash**: A generation that's aborted at 92% pressure returns an error
   to the client. A kernel panic loses everything. We choose the error every time.

3. **Caller-responsible unload**: The MemoryManager can only free its own references.
   The caller (server globals, bench locals) must clear theirs first. This is documented
   and enforced in the API endpoints.

4. **Qwen3 thinking off by default**: The risk/reward ratio is not worth it for a
   production inference server. Users who want thinking can opt in explicitly.

5. **No wired limit by default**: `mx.set_wired_limit()` can improve performance by
   keeping model weights resident, but setting it incorrectly can cause allocation
   failures. Left at 0 (system-managed) for safety.