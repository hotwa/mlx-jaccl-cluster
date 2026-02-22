# Memory Handling: exo vs mlx-jaccl-cluster — Deep Comparison

> **Date:** 2025-02-22
> **Context:** After a macOS kernel panic (`IOGPUGroupMemory.cpp:219 — "Memory object unexpectedly not found in fPendingMemorySet"`) during 2-node Qwen3-8B RDMA inference, we built a comprehensive memory safety layer (`server/memory_manager.py`). This document compares our approach with [exo-explore/exo](https://github.com/exo-explore/exo) (41K+ stars) as a reference.
>
> **Related docs:**
> - [Memory Management Guide](memory-management.md)
> - [Full exo Comparison](comparison-vs-exo.md)
> - [Architecture](architecture.md)

---

## TL;DR

| Area | exo's Approach | Our Approach | Who Does It Better |
|------|---------------|--------------|-------------------|
| Memory measurement | Continuous background polling (macmon / psutil) | On-demand vm_stat + sysctl parsing | **Exo** (continuous) / **Us** (more granular) |
| MLX memory limits | Not used — relies on OS | Adaptive 4-candidate system | **Us** (proactive prevention) |
| KV cache management | LRU eviction with distributed pressure check | None (token capping instead) | **Exo** (significant gap) |
| Model isolation | Subprocess per model (process death = clean reclaim) | In-process del + gc + clear_cache | **Exo** (guaranteed cleanup) |
| Pre-load safety | Memory-aware placement at scheduling time | readjust_limits() + headroom estimation | **Tie** (different strategies, both effective) |
| Generation-time safety | No per-token checks; relies on KV eviction | check_pressure() every 16 tokens + hard token cap | **Us** (catches runaway generation) |
| Cluster-wide awareness | all_gather max pressure across nodes | SSH-based cluster_memory_snapshot() | **Exo** (real-time, in-band) |

**Bottom line:** Exo prevents memory exhaustion through *eviction* (KV cache LRU + subprocess isolation). We prevent it through *limits* (MLX memory cap + token cap + pressure checks). Both strategies work, but combining them would be strongest.

---

## 1. Memory Measurement & Monitoring

### Exo: Continuous Background Polling

Exo runs a dedicated `InfoGatherer` service that continuously streams memory data:

**Primary source — macmon (macOS native, preferred):**
```python
# src/exo/utils/info_gatherer/macmon.py
class MacmonMetrics(TaggedModel):
    @classmethod
    def from_raw(cls, raw: RawMacmonMetrics) -> Self:
        return cls(
            memory=MemoryUsage.from_bytes(
                ram_total=raw.memory.ram_total,
                ram_available=(raw.memory.ram_total - raw.memory.ram_usage),
                swap_total=raw.memory.swap_total,
                swap_available=(raw.memory.swap_total - raw.memory.swap_usage),
            ),
        )
```

**Fallback — psutil (non-macOS or macmon not installed):**
```python
# src/exo/shared/types/profiling.py
@classmethod
def from_psutil(cls, *, override_memory: int | None) -> Self:
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    return cls.from_bytes(
        ram_total=vm.total,
        ram_available=vm.available if override_memory is None else override_memory,
        swap_total=sm.total,
        swap_available=sm.free,
    )
```

**Polling loop — runs every 1 second:**
```python
# src/exo/utils/info_gatherer/info_gatherer.py
async def _monitor_memory_usage(self):
    while True:
        await self.info_sender.send(
            MemoryUsage.from_psutil(override_memory=override_memory)
        )
        await anyio.sleep(self.memory_poll_rate)  # default: 1s
```

Key details:
- macmon is a native macOS system monitor that streams JSON metrics via pipe
- Memory data feeds into the master's placement engine in real-time
- `OVERRIDE_MEMORY_MB` env var lets you simulate constrained machines for testing
- Non-macOS falls back to psutil with a 1s poll interval

### Our Code: On-Demand vm_stat + sysctl

We parse macOS kernel stats directly when needed:

```python
# server/memory_manager.py — _get_local_memory_usage()
out = subprocess.run(["sysctl", "-n", "hw.memsize"], ...)
# ... then parse vm_stat for detailed page counts:
free       = _pages_to_gb("pages free")
inactive   = _pages_to_gb("pages inactive")
purgeable  = _pages_to_gb("pages purgeable")
wired      = _pages_to_gb("pages wired down")
compressed = _pages_to_gb("pages occupied by compressor")
available  = free + inactive + purgeable
```

Key details:
- Called at init and before each model load (`readjust_limits()`)
- Distinguishes free / inactive / purgeable / wired / compressed pages
- Does NOT count compressed pages as available (conservative and correct)
- No background thread — measurement is on-demand only

### Comparison

| Aspect | Exo | Us |
|--------|-----|----|
| Primary source | macmon (streaming native) | vm_stat + sysctl (subprocess) |
| Fallback | psutil | MLX device_info |
| Monitoring mode | **Continuous background (1s)** | On-demand (init + pre-load) |
| Page-level granularity | ❌ (total/available only) | **✅ (free/inactive/purgeable/wired/compressed)** |
| `OVERRIDE_MEMORY_MB` env | ✅ | ❌ |
| Cluster-wide | Via master state machine | Via SSH + remote vm_stat |

### Verdict

Exo wins on **continuous monitoring** — they always know the current pressure. We win on **granularity** — we know exactly *what kind* of memory is available. The ideal approach combines both.

---

## 2. Memory Limits & Thresholds

### Exo: No MLX Memory Limit — Rely on Eviction

Exo deliberately does **not** call `mx.set_memory_limit()` or `mx.set_cache_limit()`. Their philosophy is: let the OS manage memory pressure, and evict KV caches when things get tight.

The only MLX limit they set is **wired limit** (pins model weights in physical RAM):

```python
# src/exo/worker/engines/mlx/utils_mlx.py
def set_wired_limit_for_model(model_size: Memory):
    max_rec_size = Memory.from_bytes(
        int(mx.metal.device_info()["max_recommended_working_set_size"])
    )
    if model_size > 0.9 * max_rec_size:
        logger.warning(
            f"Model requires {model_size.in_float_mb:.1f} MB "
            f"which is close to max recommended {max_rec_size.in_float_mb:.1f} MB."
        )
    mx.set_wired_limit(max_rec_size.in_bytes)
```

For cache eviction, they use a **tiered memory threshold** based on machine size:

```python
# src/exo/worker/engines/mlx/cache.py
def _default_memory_threshold() -> float:
    total_gb = Memory.from_bytes(psutil.virtual_memory().total).in_gb
    if total_gb >= 128: return 0.85   # 128 GB+ machines: evict at 85%
    if total_gb >= 64:  return 0.80   # 64 GB machines:   evict at 80%
    if total_gb >= 32:  return 0.75   # 32 GB machines:   evict at 75%
    return 0.70                        # Smaller machines: evict at 70%

_MEMORY_THRESHOLD = float(
    os.environ.get("EXO_MEMORY_THRESHOLD", _default_memory_threshold())
)
```

### Our Code: Adaptive 4-Candidate MLX Limits

We proactively tell MLX not to allocate past a computed limit:

```python
# server/memory_manager.py — _apply_limits()

# Method 1 (static ceiling): fraction of total RAM
limit_by_fraction = self._total_ram * self._max_ram_fraction

# Method 2 (static floor): total minus configured reserve
limit_by_reserve = self._total_ram - (self._system_reserve_gb * _GiB)

# Method 3 (Apple's guidance): max recommended working set
limit_by_recommended = self._max_recommended

# Method 4 (ADAPTIVE): actual available RAM now minus safety margin
limit_by_available = (system_available_gb - 3.0) * _GiB  # 3 GB safety margin

# Pick the SMALLEST (most conservative)
candidates = [
    ("fraction_of_total",    limit_by_fraction),
    ("total_minus_reserve",  limit_by_reserve),
    ("apple_recommended",    limit_by_recommended),
    ("adaptive_available",   limit_by_available),
]
valid = [(n, v) for n, v in candidates if v > 2 * _GiB]
winning_name, winning_value = min(valid, key=lambda x: x[1])

mx.set_memory_limit(int(winning_value))
mx.set_cache_limit(int(winning_value * cache_fraction))
```

On a 48 GB M4 Pro with 15 GB already used:
- Exo: no limit set — relies on KV eviction at ~70% system usage
- Us: limit = min(33.6, 45, 46, 30.0) = **30.0 GB** (adaptive wins)

### Comparison

| Aspect | Exo | Us |
|--------|-----|----|
| `mx.set_memory_limit()` | ❌ Not used | **✅ Adaptive 4-candidate** |
| `mx.set_cache_limit()` | ❌ Not used | **✅ Fraction of memory limit** |
| `mx.set_wired_limit()` | ✅ = max_recommended | ✅ Optional, configurable |
| Tiered thresholds by RAM | ✅ (70–85% for eviction) | ✅ (via fraction config) |
| Re-adjustment at load | ❌ | **✅ readjust_limits()** |
| Philosophy | Let OS manage, evict when tight | **Proactive: don't allocate past limit** |

### Verdict

**Different philosophies, both valid.** Exo trusts the OS and uses eviction as a relief valve. We tell MLX "never go past X" which is more defensive — and the reason the IOGPU panic stopped recurring after our implementation. For machines running other workloads (browsers, IDEs), our adaptive approach is safer because it accounts for what's already using RAM.

---

## 3. KV Cache Management — Exo's Biggest Differentiator

This is the single largest architectural gap between our projects.

### Exo: LRU Prefix Cache with Distributed Pressure Check

Exo maintains a `KVPrefixCache` that stores KV caches from previous requests and reuses them via prefix matching:

```python
# src/exo/worker/engines/mlx/cache.py
class KVPrefixCache:
    def __init__(self, group: mx.distributed.Group | None):
        self.prompts: list[mx.array] = []
        self.caches: list[KVCacheType] = []
        self._last_used: list[int] = []       # LRU tracking
        self._access_counter: int = 0
        self._group = group                    # for distributed pressure check

    def add_kv_cache(self, prompt_tokens, cache, ssm_snapshots=None):
        self._evict_if_needed()                # evict BEFORE adding new entry
        self.prompts.append(prompt_tokens)
        self.caches.append(deepcopy(cache))
        self._access_counter += 1
        self._last_used.append(self._access_counter)

    def get_kv_cache(self, model, prompt_tokens):
        """Find best prefix match, reuse cached KV state."""
        best_index, best_length = None, 0
        for i, cached_prompt in enumerate(self.prompts):
            length = get_prefix_length(prompt_tokens, cached_prompt)
            if length > best_length:
                best_index, best_length = i, length
        if best_index is None:
            return make_kv_cache(model), prompt_tokens, None
        # Trim cache to match point, return remaining tokens
        prompt_cache = deepcopy(self.caches[best_index])
        remaining = prompt_tokens[restore_pos:]
        return prompt_cache, remaining, best_index
```

**LRU eviction with memory pressure check:**
```python
    def _evict_if_needed(self):
        while (len(self.caches) > 0
               and self.get_memory_used_percentage() > _MEMORY_THRESHOLD):
            lru_index = self._last_used.index(min(self._last_used))
            evicted_tokens = len(self.prompts[lru_index])
            self.prompts.pop(lru_index)
            self.caches.pop(lru_index)
            self._snapshots.pop(lru_index)
            self._last_used.pop(lru_index)
            logger.info(f"KV cache evicted LRU ({evicted_tokens} tokens)")
```

**Distributed pressure — the clever part:**
```python
    def get_memory_used_percentage(self) -> float:
        local_pressure = get_memory_used_percentage()  # psutil
        if self._group is None:
            return local_pressure
        # In distributed mode: get MAX pressure across ALL nodes
        all_pressure = mx.distributed.all_gather(
            mx.array([local_pressure], dtype=mx.float32),
            group=self._group,
        )
        return float(mx.max(all_pressure).item())
```

This means: if *any* node in the cluster is under pressure, *all* nodes evict. This prevents the weakest node from hitting a panic while stronger nodes happily cache more.

### Our Code: No KV Cache — Token Capping Instead

We don't maintain a KV prefix cache. Our strategy for preventing unbounded memory growth during generation:

```python
# server/memory_manager.py
def clamp_max_tokens(self, requested: int) -> int:
    return min(requested, self.hard_max_tokens)  # default: 4096

def generation_guard(self, context: str = "generation"):
    """Call every N tokens during generation."""
    self._generation_tokens += 1
    if self._generation_tokens % 16 == 0:
        self.check_pressure(context)

def check_pressure(self, context: str = "generation"):
    pct = self.pressure_pct()
    if pct >= _PRESSURE_CRITICAL * 100:
        self._emergency_cleanup()
        raise MemoryPressureError(f"CRITICAL: {pct:.1f}% of limit")
```

### Comparison

| Aspect | Exo | Us |
|--------|-----|----|
| KV prefix cache | **✅ Full LRU implementation** | ❌ None |
| Prefix reuse across requests | **✅ (major perf win for chat)** | ❌ |
| Memory-aware eviction | **✅ Evict when pressure > threshold** | ❌ |
| Distributed pressure check | **✅ all_gather max across nodes** | ❌ (SSH-based, not real-time) |
| SSM/state snapshot support | **✅ (Mamba-style models)** | ❌ |
| Hard token cap | ❌ | **✅ (4096 default)** |
| Per-token pressure check | ❌ | **✅ (every 16 tokens)** |
| Emergency cleanup | ❌ | **✅ (gc + clear_cache)** |

### Verdict

**Exo wins decisively here.** Their KV prefix cache is both a performance feature (prefix reuse avoids re-prefilling) and a safety feature (LRU eviction prevents unbounded growth). The distributed `all_gather` pressure check is particularly clever for multi-node setups like ours.

---

## 4. Model Loading & Unloading

### Exo: Subprocess Isolation

Exo runs each model in a **separate child process** via `RunnerSupervisor`:

```python
# src/exo/worker/runner/runner_supervisor.py
runner_process = Process(
    target=entrypoint,
    args=(bound_instance, ev_send, task_recv, cancel_recv, logger),
    daemon=True,
)

def shutdown(self):
    self.runner_process.join(5)
    if not self.runner_process.is_alive():
        return
    logger.warning("Runner didn't shutdown, terminating")
    self.runner_process.terminate()    # SIGTERM
    self.runner_process.join(1)
    if not self.runner_process.is_alive():
        return
    logger.critical("Runner didn't respond to SIGTERM, killing")
    self.runner_process.kill()         # SIGKILL — guaranteed death
```

Inside the runner, shutdown is simple:
```python
# src/exo/worker/runner/llm_inference/runner.py
case Shutdown():
    del inference_model, tokenizer, group
    mx.clear_cache()
    gc.collect()
```

**Why this matters:** When a process dies (or is killed), the OS reclaims ALL its memory — Metal buffers, Python objects, everything. No ref cycles, no lingering allocations, no "did gc actually find everything?" questions.

### Our Code: In-Process Aggressive Cleanup

We do the unload in the same process with a multi-pass cleanup:

```python
# server/memory_manager.py — unload_model()

# Step 1: Wipe the slot (drops ModelSlot references)
model_ref = self._slot.model
tok_ref = self._slot.tokenizer
self._slot = ModelSlot()

# Step 2: Explicit del (drops refcount to zero immediately)
del model_ref
del tok_ref

# Step 3: Triple GC (catches reference cycles in nn.Module trees)
gc.collect()
gc.collect()
gc.collect()

# Step 4: Clear MLX Metal buffer cache
mx.clear_cache()

# Step 5: Sleep + second pass (Metal may free asynchronously)
time.sleep(0.05)
gc.collect()
mx.clear_cache()

# Step 6: Reset peak tracker
mx.reset_peak_memory()
```

**Important caller requirement:** The caller (server globals or bench locals) must clear their own references to model/tokenizer *before* calling `unload_model()`, or Metal buffers won't be freed.

### Comparison

| Aspect | Exo | Us |
|--------|-----|----|
| Isolation | **Subprocess (Process)** | In-process |
| Memory guaranteed free? | **✅ (process death = OS reclaim)** | ⚠️ (depends on no lingering refs) |
| Cleanup steps | del + clear_cache + gc | del + 3× gc + 2× clear_cache + sleep |
| Load timeout handling | Configurable, scales with model size | Pre-flight headroom estimation |
| Pre-load memory check | Via placement engine (before scheduling) | readjust_limits() + headroom |
| Swap overhead | Higher (new process spin-up) | **Lower (just gc + load)** |
| Ref cycle risk | **None (process is dead)** | Must be careful with caller refs |

### Verdict

**Exo wins on safety** (subprocess = guaranteed cleanup). **We win on speed** (no process restart overhead). For a production system, the subprocess model is objectively safer. For a dev/research cluster where you swap models frequently, in-process is more convenient.

---

## 5. Memory-Aware Placement & Scheduling

### Exo: Filter + Rank by Available RAM

Before a model is even loaded, exo's master checks if any node topology has enough RAM:

```python
# src/exo/master/placement_utils.py
def filter_cycles_by_memory(cycles, node_memory, required_memory):
    filtered_cycles = []
    for cycle in cycles:
        total_mem = sum(
            (node_memory[node_id].ram_available for node_id in cycle.node_ids),
            start=Memory(),
        )
        if total_mem >= required_memory:
            filtered_cycles.append(cycle)
    return filtered_cycles
```

When multiple topologies qualify, they pick the one with **most available RAM**:

```python
# src/exo/master/placement.py
selected_cycle = max(
    candidate_cycles,
    key=lambda cycle: sum(
        (node_memory[node_id].ram_available for node_id in cycle),
        start=Memory(),
    ),
)
```

And they allocate layers **proportionally** based on per-node available RAM:

```python
# src/exo/master/placement_utils.py
layer_allocations = allocate_layers_proportionally(
    total_layers=model_card.n_layers,
    memory_fractions=[
        node_memory[node_id].ram_available / total_memory
        for node_id in node_ids
    ],
)
```

This means: if node A has 40 GB free and node B has 20 GB free, node A gets ~2/3 of the layers. Nodes are never overloaded relative to their capacity.

### Our Code: Pre-Load Headroom Check

We check headroom before loading but don't influence sharding decisions:

```python
# server/memory_manager.py — load_model()

# Re-measure system memory state
readjust = self.readjust_limits(reason=f"pre-load {model_id}")

# Estimate if we have enough headroom
headroom = self.headroom_gb()
model_disk_gb = self._estimate_model_disk_size(path)
nodes = world.size() if world else 1
estimated_peak_load = model_disk_gb * 1.3

if estimated_peak_load > headroom * 1.5:
    raise MemoryPressureError(
        f"Not enough memory to load {model_id}: "
        f"estimated {estimated_peak_load:.1f} GB needed, "
        f"only {headroom:.1f} GB available."
    )
```

JACCL handles the actual sharding (tensor or pipeline parallel), which currently divides layers equally across nodes.

### Comparison

| Aspect | Exo | Us |
|--------|-----|----|
| Pre-schedule memory filter | **✅ (reject if insufficient)** | ✅ (MemoryPressureError if too tight) |
| Best-topology selection | **✅ (pick max available RAM)** | ❌ (single topology from hostfile) |
| Proportional layer allocation | **✅ (RAM-weighted)** | ❌ (JACCL does equal split) |
| Re-measure before load | ❌ | **✅ readjust_limits()** |
| Cluster-wide snapshot | ✅ (master state machine) | ✅ (SSH-based cluster_memory_snapshot) |

### Verdict

**Exo wins on proportional allocation** — critical for heterogeneous clusters. Our `readjust_limits()` before every load is a nice touch exo doesn't have.

---

## 6. Generation-Time Safety

### Exo: No Per-Token Checks — KV Eviction is the Safety Net

During generation, exo does not check memory pressure per-token. Their safety comes from:
1. KV cache LRU eviction at cache insertion time (before generation starts)
2. Task cancellation via IPC pipe (checked every N tokens, where N is calibrated during warmup)
3. Subprocess isolation (if the runner crashes, the supervisor catches it)

```python
# src/exo/worker/runner/llm_inference/runner.py
tokens_since_last_cancel_check = check_for_cancel_every
for response in mlx_generator:
    tokens_since_last_cancel_check += 1
    if tokens_since_last_cancel_check >= check_for_cancel_every:
        tokens_since_last_cancel_check = 0
        cancelled_tasks.update(cancel_receiver.collect())
        want_to_cancel = (task.task_id in cancelled_tasks) or ...
        if mx_any(want_to_cancel, group):
            break
```

### Our Code: Periodic Pressure Checks + Hard Cap

We actively monitor during generation:

```python
# server/memory_manager.py
def check_pressure(self, context="generation"):
    pct = self.pressure_pct()
    if pct >= _PRESSURE_CRITICAL * 100:
        self._emergency_cleanup()           # gc + clear_cache
        raise MemoryPressureError(...)      # abort cleanly
    if pct >= _PRESSURE_WARN * 100:
        log.warning(f"High pressure: {pct:.1f}%")

def _emergency_cleanup(self):
    gc.collect()
    gc.collect()
    mx.clear_cache()
    time.sleep(0.1)
    gc.collect()
    mx.clear_cache()
```

Plus a hard token cap (critical for Qwen3 thinking mode):
```python
def clamp_max_tokens(self, requested: int) -> int:
    return min(requested, self.hard_max_tokens)  # default 4096
```

### Comparison

| Aspect | Exo | Us |
|--------|-----|----|
| Per-token memory pressure check | ❌ | **✅ Every 16 tokens** |
| Hard token cap | ❌ | **✅ 4096 default (configurable)** |
| Emergency cleanup mid-generation | ❌ | **✅ gc + clear_cache** |
| Cancellation support | ✅ IPC pipe + all_gather | ✅ Abort flag |
| Crash recovery | **✅ Supervisor restarts runner** | ❌ (process crash = server down) |

### Verdict

**We win on proactive detection** (catching pressure mid-generation). **Exo wins on crash recovery** (subprocess supervisor restarts the runner). The hard token cap is essential for models like Qwen3 that can generate unbounded internal reasoning.

---

## 7. Summary: What Each Project Does Better

### Exo Does Better

| Feature | Description | Impact |
|---------|-------------|--------|
| **KV prefix cache with LRU eviction** | Reuses KV caches across requests, evicts when memory is tight | Performance + safety |
| **Distributed pressure via all_gather** | Takes max pressure across all nodes for eviction decisions | Prevents weakest-node panic |
| **Subprocess isolation** | Model runs in child process; death = guaranteed cleanup | Ultimate memory safety |
| **Continuous memory monitoring** | Background poller (macmon / psutil) every 1s | Always-current pressure data |
| **Memory-proportional layer allocation** | More layers on nodes with more available RAM | Optimal for heterogeneous clusters |
| **Tiered memory thresholds** | 70% for small machines, 85% for large ones | Right-sized protection |
| **OVERRIDE_MEMORY_MB** | Simulate constrained machines for testing | Dev/CI convenience |

### Our Code Does Better

| Feature | Description | Impact |
|---------|-------------|--------|
| **Adaptive MLX memory limits** | 4-candidate system probing actual free RAM | Proactive prevention of over-allocation |
| **Pre-load readjustment** | Re-measures system state before every model load | Adapts to changing conditions |
| **Per-token pressure monitoring** | Catches runaway generation mid-flight | Prevents slow-building panics |
| **Hard token caps** | Prevents Qwen3 thinking mode from infinite generation | Essential for reasoning models |
| **Emergency cleanup** | gc + clear_cache as last resort before panic | Avoids kernel panic |
| **Detailed vm_stat parsing** | Distinguishes free/inactive/purgeable/wired/compressed | More accurate available-memory estimate |
| **Explicit unload protocol** | Documented multi-step Metal buffer release | Predictable memory reclaim |

---

## 8. Recommended Improvements (Inspired by Exo)

These are concrete enhancements we should implement, ordered by impact:

### 8.1 — KV Prefix Cache with LRU Eviction (HIGH IMPACT)

**What exo does:** Maintains cached KV states for previous prompts. On new request, finds longest matching prefix and reuses the KV cache, only prefilling the new suffix. Evicts oldest entries when memory pressure exceeds threshold.

**Why we need it:**
- Without prefix caching, every request re-prefills from scratch — wasteful for chat conversations where the system prompt + history is repeated
- The eviction mechanism doubles as memory safety — prevents unbounded KV cache growth
- In our RDMA cluster, the distributed `all_gather` pressure check would prevent the weaker node from panicking

**Implementation plan:**
```
Priority:  ★★★★★ (highest)
Effort:    ~400 lines
Dependencies: None (can use our existing vm_stat for pressure)
Phase:     Phase 6 in roadmap (now elevated to Phase 2)
```

**Key design decisions to make:**
1. Use `psutil.virtual_memory().percent` or our existing `vm_stat` parser for pressure?
2. How many cached entries to allow before LRU eviction starts?
3. For RDMA mode: add `mx.distributed.all_gather` pressure check?

### 8.2 — Continuous Background Memory Monitoring (HIGH IMPACT)

**What exo does:** A background thread polls memory every 1s and feeds it into a shared state object that placement, eviction, and API endpoints all read from.

**Why we need it:**
- We only measure at init and pre-load — pressure spikes between requests go undetected
- A background thread would let the `/memory` endpoint return truly current data
- Could trigger alerts or pre-emptive GC when pressure rises

**Implementation plan:**
```
Priority:  ★★★★☆
Effort:    ~100 lines (thread + shared MemorySnapshot)
Dependencies: None
Phase:     Phase 1
```

**Design:**
```python
# Proposed: background memory monitor thread
import threading

class MemoryMonitor:
    def __init__(self, manager: MemoryManager, interval: float = 2.0):
        self._manager = manager
        self._interval = interval
        self._latest: MemorySnapshot = manager.snapshot()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while True:
            self._latest = self._manager.snapshot()
            if self._latest.pressure_pct > 80:
                self._manager.gc_cycle()  # pre-emptive cleanup
            time.sleep(self._interval)

    @property
    def current(self) -> MemorySnapshot:
        return self._latest
```

### 8.3 — Subprocess Model Runner (MEDIUM IMPACT)

**What exo does:** Each model runs in a child process. The supervisor communicates via IPC pipes. If the runner crashes, the supervisor catches the exit code and can restart.

**Why we need it:**
- In-process unload depends on no lingering Python references — fragile
- If MLX hits an unrecoverable Metal error, the entire server dies
- Subprocess isolation guarantees memory reclaim on unload

**Implementation plan:**
```
Priority:  ★★★☆☆
Effort:    ~500 lines (subprocess + IPC protocol)
Dependencies: Significant refactor of server architecture
Phase:     Phase 3
```

**Trade-offs:**
- Pro: Guaranteed memory cleanup, crash isolation
- Con: Higher latency for model swaps, more complex IPC, debugging is harder
- Compromise: Keep in-process mode as default, add `--subprocess` flag for production

### 8.4 — Memory-Proportional Layer Allocation (MEDIUM IMPACT)

**What exo does:** When sharding across nodes, allocates more layers to nodes with more available RAM.

**Why we need it:**
- Our 2-node cluster may have asymmetric available RAM (one running more apps)
- Equal sharding can overload the constrained node while the other has headroom
- Especially important if we add a third node with different specs

**Implementation plan:**
```
Priority:  ★★★☆☆
Effort:    ~150 lines (query both nodes, compute fractions, pass to JACCL)
Dependencies: JACCL must support unequal layer splits
Phase:     Phase 4 (needs JACCL investigation first)
```

**Open question:** Does JACCL's tensor parallel mode support unequal splits? Pipeline parallel should work, but tensor parallel typically requires equal splits.

### 8.5 — Tiered Memory Threshold by Machine Size (LOW EFFORT)

**What exo does:** Different eviction thresholds: 70% for small machines, 85% for large ones.

**Why we need it:**
- Our fixed fraction doesn't account for absolute RAM size
- A 24 GB machine needs more conservative limits than a 192 GB machine
- Simple to implement, zero risk

**Implementation plan:**
```
Priority:  ★★☆☆☆
Effort:    ~20 lines
Dependencies: None
Phase:     Phase 1 (trivial, do immediately)
```

```python
# Proposed: tiered defaults
def _default_ram_fraction(total_gb: float) -> float:
    if total_gb >= 128: return 0.85
    if total_gb >= 64:  return 0.80
    if total_gb >= 48:  return 0.75
    if total_gb >= 32:  return 0.70
    return 0.65
```

### 8.6 — OVERRIDE_MEMORY_MB Environment Variable (LOW EFFORT)

**What exo does:** Lets you override reported available memory for testing constrained scenarios.

**Why we need it:**
- Test memory pressure handling without actually filling RAM
- CI/CD can simulate a 16 GB machine on a 48 GB host
- Zero production risk (only active when env var is set)

**Implementation plan:**
```
Priority:  ★★☆☆☆
Effort:    ~15 lines
Dependencies: None
Phase:     Phase 1 (trivial, do immediately)
```

---

## 9. Implementation Priority & Phasing

```
Phase 1 (Immediate — low effort, high value):
  ├── 8.5  Tiered memory threshold by machine size     ~20 lines
  ├── 8.6  OVERRIDE_MEMORY_MB env var                  ~15 lines
  └── 8.2  Background memory monitor thread            ~100 lines

Phase 2 (High impact — core safety improvement):
  └── 8.1  KV prefix cache with LRU eviction           ~400 lines
           (includes distributed all_gather pressure)

Phase 3 (Architecture — significant refactor):
  └── 8.3  Subprocess model runner                     ~500 lines

Phase 4 (Optimization — depends on JACCL):
  └── 8.4  Memory-proportional layer allocation        ~150 lines
```

**Total new code: ~1,200 lines**

After implementation, the combined safety model would be:

```
Before request:
  ├── MLX memory limit (adaptive, 4-candidate)        [existing - ours]
  ├── Pre-load readjust_limits()                       [existing - ours]
  ├── KV prefix cache eviction                         [new - from exo]
  └── Continuous pressure monitoring                   [new - from exo]

During generation:
  ├── Per-token pressure check (every 16 tokens)       [existing - ours]
  ├── Hard token cap (4096)                            [existing - ours]
  ├── Distributed all_gather pressure                  [new - from exo]
  └── Emergency cleanup (gc + clear_cache)             [existing - ours]

After request / on swap:
  ├── KV cache LRU eviction                            [new - from exo]
  ├── In-process unload (del + gc + clear_cache)       [existing - ours]
  └── Optional subprocess isolation                    [new - from exo]
```

This layered approach combines the best of both strategies: **proactive limits** (ours) + **reactive eviction** (exo's) + **guaranteed cleanup** (exo's subprocess model).

---

## Appendix: Key Source Files Referenced

### Exo (exo-explore/exo @ `1780e4a`)

| File | Purpose |
|------|---------|
| `src/exo/worker/engines/mlx/cache.py` | KV prefix cache, LRU eviction, distributed pressure |
| `src/exo/worker/engines/mlx/utils_mlx.py` | Model loading, wired limit, distributed init |
| `src/exo/worker/runner/llm_inference/runner.py` | Generation loop, cancellation, model lifecycle |
| `src/exo/worker/runner/runner_supervisor.py` | Subprocess management, graceful/forced shutdown |
| `src/exo/master/placement.py` | Memory-aware topology selection |
| `src/exo/master/placement_utils.py` | filter_cycles_by_memory, proportional layer allocation |
| `src/exo/shared/types/profiling.py` | MemoryUsage type, psutil integration |
| `src/exo/shared/types/memory.py` | Memory value type with unit conversions |
| `src/exo/utils/info_gatherer/info_gatherer.py` | Background polling (macmon, psutil, system_profiler) |
| `src/exo/utils/info_gatherer/macmon.py` | macmon JSON parsing |

### Our Code (mlx-jaccl-cluster)

| File | Purpose |
|------|---------|
| `server/memory_manager.py` | Adaptive limits, model load/unload, pressure checks |
| `server/openai_cluster_server.py` | API endpoints, generation loop integration |