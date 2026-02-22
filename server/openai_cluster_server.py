#!/usr/bin/env python3
import asyncio
import atexit
import gc
import json
import logging
import os
import signal
import socket
import struct
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional, Union

import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer

# generate() import differs across mlx-lm branches
try:
    from mlx_lm.utils import generate
except Exception:
    from mlx_lm.generate import generate

# stream_generate for SSE streaming
try:
    from mlx_lm.utils import stream_generate
except ImportError:
    try:
        from mlx_lm.generate import stream_generate
    except ImportError:
        stream_generate = None

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

# Memory manager — prevents kernel panics from GPU memory exhaustion
try:
    from memory_manager import (
        MemoryManager,
        MemoryPressureError,
        ModelNotLoadedError,
        get_manager,
        init_manager,
    )

    _MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    _MEMORY_MANAGER_AVAILABLE = False
    MemoryManager = None
    MemoryPressureError = RuntimeError
    ModelNotLoadedError = RuntimeError

# Background memory & hardware monitor (macmon-based, like exo)
try:
    from memory_monitor import MemoryMonitor

    _MONITOR_AVAILABLE = True
except ImportError:
    _MONITOR_AVAILABLE = False
    MemoryMonitor = None

# Per-request logging with JSONL persistence
try:
    from request_log import (
        STATUS_CANCELLED,
        STATUS_ERROR,
        STATUS_OK,
        STATUS_PRESSURE_ABORT,
        STATUS_TIMEOUT,
        RequestLog,
        RequestRecord,
        get_request_log,
        init_request_log,
    )

    _REQUEST_LOG_AVAILABLE = True
except ImportError:
    _REQUEST_LOG_AVAILABLE = False
    RequestLog = None
    RequestRecord = None

# Dashboard (optional — only mounted on rank 0)
try:
    from dashboard import GenerationStats, metrics_store, mount_dashboard

    _DASHBOARD_AVAILABLE = True
except ImportError:
    _DASHBOARD_AVAILABLE = False
    metrics_store = None
    GenerationStats = None

log = logging.getLogger("cluster_server")


# -------------------------
# Custom tokenizer support
# -------------------------
class TokenizerWrapper:
    """Wrapper to handle encode kwargs that some custom tokenizers don't support."""

    def __init__(self, tokenizer):
        self._tok = tokenizer

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def encode(self, text, **kwargs):
        return self._tok.encode(text)

    def decode(self, tokens, **kwargs):
        return self._tok.decode(tokens)


def load_custom_tokenizer(model_path):
    """Load custom tokenizer directly when AutoTokenizer fails."""
    model_path = Path(model_path)
    sys.path.insert(0, str(model_path))

    for tok_file in model_path.glob("tokenization_*.py"):
        module_name = tok_file.stem
        mod = __import__(module_name)
        for attr in dir(mod):
            cls = getattr(mod, attr)
            if isinstance(cls, type) and hasattr(cls, "from_pretrained"):
                try:
                    tok = cls.from_pretrained(model_path)
                    return TokenizerWrapper(tok)
                except:
                    continue
    raise RuntimeError(f"Could not load custom tokenizer from {model_path}")


def sharded_load_with_fallback(repo):
    """
    Load and shard the model using EAGER (non-lazy) weight loading.

    Why eager instead of lazy?
      lazy=True + mx.eval(model.parameters()) triggers a distributed
      computation graph that deadlocks in JACCL — rank 1 hangs inside
      mx.eval() even with barriers. Eager loading materializes weights
      from disk immediately (no mx.eval needed), then shard() just
      redistributes the already-concrete tensors. This completely
      sidesteps the JACCL eval deadlock.

    This matches the proven approach from jaccl_tps_bench.py.

    Now delegates to MemoryManager when available for:
      - Hard memory limits (prevents kernel panic)
      - Proper unload of any previous model
      - Pre-flight memory headroom check
      - Qwen3 thinking mode disable
    """
    global _mm

    model_path = Path(repo)
    world = mx.distributed.init()
    rank = world.rank()

    # Use MemoryManager if available (the safe path)
    if _MEMORY_MANAGER_AVAILABLE:
        model_id = os.path.basename(str(repo).rstrip("/"))
        model, tok = _mm.load_model(
            str(model_path), world=world, model_id=model_id, lazy=False
        )

        # Patch tokenizer for Qwen3 thinking safety
        if _mm.should_disable_thinking(model_id):
            tok = _mm.patch_chat_template_no_thinking(tok)
            print(
                f"  [rank {rank}] ⚠ Qwen3 thinking mode DISABLED "
                f"(set QWEN3_ENABLE_THINKING=1 to re-enable)",
                flush=True,
            )

        _mm.print_status()
        return model, tok

    # Fallback: original path (no memory safety)
    print(
        f"  [rank {rank}] ⚠ memory_manager not available — "
        f"running WITHOUT memory limits (kernel panic risk!)",
        flush=True,
    )

    # Step 1: EAGER load — weights are fully materialized from disk
    print(f"  [rank {rank}] loading model (eager) ...", flush=True)
    t0 = time.time()
    model, _ = load_model(model_path, lazy=False)
    print(f"  [rank {rank}] model loaded in {time.time() - t0:.2f}s", flush=True)

    # Step 2: barrier — ensure both ranks loaded before sharding
    x = mx.zeros((1,))
    mx.eval(mx.distributed.all_sum(x))
    print(f"  [rank {rank}] pre-shard barrier done", flush=True)

    # Step 3: shard
    if hasattr(model, "shard"):
        model.shard(world)
        print(f"  [rank {rank}] model sharded (Tensor Parallelism)", flush=True)
    else:
        print(f"  [rank {rank}] no shard method — running replicated", flush=True)

    # Step 4: post-shard barrier
    mx.eval(mx.distributed.all_sum(mx.zeros((1,))))
    print(f"  [rank {rank}] post-shard barrier done", flush=True)

    # Step 5: load tokenizer
    try:
        tok = load_tokenizer(
            model_path, {"trust_remote_code": True}, eos_token_ids=None
        )
    except Exception:
        # Fallback for custom tokenizers
        tok = load_custom_tokenizer(model_path)
    print(f"  [rank {rank}] tokenizer loaded", flush=True)

    return model, tok


# -------------------------
# Configuration (env vars)
# -------------------------
MODEL_DIR = os.environ["MODEL_DIR"]  # REQUIRED
MODEL_ID = os.environ.get("MODEL_ID", os.path.basename(MODEL_DIR.rstrip("/")))

HOST = os.environ.get("HOST", "0.0.0.0")  # HTTP bind on rank0
PORT = int(os.environ.get("PORT", "8080"))  # HTTP port on rank0

# Control-plane (rank0 <-> workers) for coordinating "everyone call generate()"
CTRL_PORT = int(os.environ.get("CTRL_PORT", "18080"))


def _default_ctrl_host() -> str:
    c = os.environ.get("MLX_JACCL_COORDINATOR", "")
    if ":" in c:
        return c.split(":", 1)[0]
    return "macstudio1.local"


CTRL_HOST = os.environ.get("CTRL_HOST", _default_ctrl_host())

DEFAULT_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))

# Backpressure / queueing
QUEUE_MAX = int(os.environ.get("QUEUE_MAX", "8"))  # max queued requests
REQ_TIMEOUT = float(
    os.environ.get("REQ_TIMEOUT", "120")
)  # per request timeout (seconds)


# -------------------------
# Globals
# -------------------------

# Memory manager — initialized early, before any model load.
# This MUST happen before load_model to set safe memory limits.
_mm: Optional["MemoryManager"] = None
if _MEMORY_MANAGER_AVAILABLE:
    _mm = init_manager()
    log.info("MemoryManager initialized with safe limits")
else:
    log.warning(
        "memory_manager module not available — running WITHOUT memory limits. "
        "This risks a macOS kernel panic under memory pressure!"
    )

# Background memory monitor (macmon pipe, like exo's InfoGatherer)
_monitor: Optional["MemoryMonitor"] = None

# Per-request log
_rlog: Optional["RequestLog"] = None


# -------------------------
# Lifespan (replaces deprecated @app.on_event)
# -------------------------
@asynccontextmanager
async def _lifespan(application):
    """Startup/shutdown lifespan for FastAPI — runs the queue worker on rank0."""
    global _monitor, _rlog

    if _world and _world.rank() == 0:
        # Start background memory monitor (macmon persistent pipe, like exo)
        if _MONITOR_AVAILABLE and _mm is not None:
            _monitor = MemoryMonitor(manager=_mm)
            _monitor.start()
            log.info(
                f"MemoryMonitor started: source={'macmon' if _monitor.using_macmon else 'vm_stat'}, "
                f"threshold={_monitor.memory_threshold:.0%}"
            )

        # Start request log
        if _REQUEST_LOG_AVAILABLE:
            _rlog = init_request_log()
            log.info(f"RequestLog initialized: {_rlog.log_path}")

        asyncio.create_task(_queue_worker())
        _print_ready_banner()

    yield

    # Shutdown: stop monitor cleanly (kills macmon subprocess)
    if _monitor is not None:
        _monitor.stop()


app = FastAPI(
    title="mlx-jaccl-cluster",
    description="OpenAI-compatible API for a multi-Mac MLX cluster over RDMA/Thunderbolt (JACCL)",
    version="0.1.0",
    lifespan=_lifespan,
)
_model = None
_tok = None
_world = None

_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)  # rank0 only uses it


# -------------------------
# Tiny framed JSON protocol
# -------------------------
def _recvall(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def send_msg(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)


def recv_msg(sock: socket.socket) -> Optional[dict]:
    hdr = _recvall(sock, 4)
    if hdr is None:
        return None
    (n,) = struct.unpack("!I", hdr)
    body = _recvall(sock, n)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))


# -------------------------
# OpenAI-ish schemas
# -------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionsReq(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class CompletionsReq(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, list[str]]
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


def _build_chat_prompt(messages: list[ChatMessage]) -> str:
    # Use memory manager's safe prompt builder (disables Qwen3 thinking)
    if _mm is not None:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        return _mm.build_safe_chat_prompt(msgs, tokenizer=_tok, model_id=MODEL_ID)

    # Fallback: use tokenizer chat template directly
    if hasattr(_tok, "apply_chat_template"):
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        return _tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    # Last resort: simple "ROLE: content" format
    parts = [f"{m.role.upper()}: {m.content}" for m in messages]
    parts.append("ASSISTANT:")
    return "\n".join(parts)


def _safe_max_tokens(requested: Optional[int]) -> int:
    """Clamp max_tokens to prevent runaway generation / KV cache explosion."""
    if _mm is not None:
        return _mm.clamp_max_tokens(requested)
    # Fallback hard cap without memory manager
    hard_cap = int(os.environ.get("MLX_HARD_MAX_TOKENS", "4096"))
    if requested is None or requested <= 0:
        return min(512, hard_cap)
    return min(requested, hard_cap)


def _tok_len(text: str) -> int:
    return len(_tok.encode(text))


# -------------------------
# Rank0 worker connections
# -------------------------
_worker_socks: dict[int, socket.socket] = {}  # rank -> socket
_worker_lock = threading.Lock()


def rank0_accept_workers(expected_world_size: int) -> None:
    """
    Rank0 listens for worker control-plane connections.
    Each worker sends {"type":"hello","rank":N}.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, CTRL_PORT))
    srv.listen(16)
    print(f"[rank0] control-plane listening on {HOST}:{CTRL_PORT}", flush=True)

    while True:
        conn, addr = srv.accept()
        hello = recv_msg(conn)
        if not hello or hello.get("type") != "hello" or "rank" not in hello:
            conn.close()
            continue
        r = int(hello["rank"])
        with _worker_lock:
            _worker_socks[r] = conn
        print(f"[rank0] worker connected rank={r} from {addr}", flush=True)


def rank0_wait_for_workers(expected_world_size: int, timeout_s: int = 60) -> bool:
    t0 = time.time()
    while True:
        with _worker_lock:
            ok = all(r in _worker_socks for r in range(1, expected_world_size))
        if ok:
            print("[rank0] all workers connected", flush=True)
            return True
        if time.time() - t0 > timeout_s:
            return False
        time.sleep(0.1)


def rank0_broadcast_task(task: dict) -> None:
    """
    Send the same task to all worker ranks (1..N-1).
    """
    with _worker_lock:
        items = list(_worker_socks.items())
    for r, s in items:
        send_msg(s, {"type": "task", **task})


def rank0_wait_done(expected_world_size: int) -> None:
    """
    Wait for {"type":"done"} from all workers.
    """
    done: set[int] = set()
    while len(done) < (expected_world_size - 1):
        with _worker_lock:
            items = list(_worker_socks.items())
        for r, s in items:
            if r in done:
                continue
            s.settimeout(0.2)
            try:
                msg = recv_msg(s)
            except Exception:
                msg = None
            if msg and msg.get("type") == "done":
                done.add(r)


# -------------------------
# Worker loop
# -------------------------
def worker_loop(rank: int) -> None:
    """
    Workers connect to rank0 control-plane, block waiting for tasks.
    For each task: call generate() (so collectives match rank0), then send done.
    Exits cleanly when the control socket closes (rank0 shutdown / Ctrl+C).
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((CTRL_HOST, CTRL_PORT))
    send_msg(s, {"type": "hello", "rank": rank})
    print(
        f"[worker {rank}] connected to control-plane {CTRL_HOST}:{CTRL_PORT}",
        flush=True,
    )

    _none_count = 0  # track consecutive None reads (socket dead)

    while True:
        try:
            msg = recv_msg(s)
        except (ConnectionResetError, BrokenPipeError, OSError):
            print(
                f"\n[worker {rank}] control socket lost — shutting down.",
                flush=True,
            )
            break

        if not msg:
            _none_count += 1
            if _none_count >= 3:
                # Socket is dead (rank0 exited) — exit cleanly
                print(
                    f"\n[worker {rank}] coordinator disconnected — shutting down.",
                    flush=True,
                )
                break
            time.sleep(0.1)
            continue

        _none_count = 0  # reset on valid message

        if msg.get("type") == "shutdown":
            print(f"[worker {rank}] received shutdown — exiting.", flush=True)
            break

        if msg.get("type") != "task":
            continue

        prompt = msg["prompt"]
        max_tokens = int(msg["max_tokens"])

        _ = generate(_model, _tok, prompt, max_tokens=max_tokens)
        mx.eval()
        send_msg(s, {"type": "done", "rank": rank})

    # Clean exit
    try:
        s.close()
    except Exception:
        pass
    print(f"[worker {rank}] stopped. GPU memory released.", flush=True)


# -------------------------
# Queue worker (rank0 only)
# -------------------------
async def _queue_worker() -> None:
    """
    Processes queued requests sequentially.
    Each request triggers:
      - broadcast task to workers
      - rank0 generate() or stream_generate()
      - wait for worker completion
      - fulfill per-request future with an OpenAI-shaped response (or stream chunks)

    Memory safety:
      - max_tokens is clamped by _safe_max_tokens() before reaching here
      - Memory pressure is checked every 16 tokens during streaming
      - GC cycle runs after every request to reclaim KV cache memory
      - MemoryPressureError aborts generation cleanly (no kernel panic)

    Request logging:
      - Records memory before/after, timing, token counts, status per request
      - Writes to JSONL file + in-memory ring buffer for /requests/* endpoints
    """
    while True:
        item = await _queue.get()
        if item is None:
            _queue.task_done()
            continue

        kind, prompt, max_t, result_target, is_stream = (
            item  # kind: "chat" | "completions"
        )

        # ── Per-request tracking ──────────────────────────────────
        req_id = (
            f"chatcmpl-{uuid.uuid4().hex[:24]}"
            if kind == "chat"
            else f"cmpl-{uuid.uuid4().hex[:24]}"
        )
        mem_before = _mm.active_gb() if _mm else 0.0
        req_timestamp = time.time()
        req_status = STATUS_OK if _REQUEST_LOG_AVAILABLE else "ok"
        req_error_msg = None
        req_finish_reason = "stop"
        req_generated_tokens = 0
        req_prompt_tokens = _tok_len(prompt) if _tok else 0

        try:
            rank0_broadcast_task({"prompt": prompt, "max_tokens": max_t})

            if is_stream and stream_generate is not None:
                # Streaming mode: yield chunks via async queue
                chunk_queue: asyncio.Queue = result_target
                created = int(time.time())

                t0 = time.time()
                token_count = 0
                _aborted = False

                for response in stream_generate(_model, _tok, prompt, max_tokens=max_t):
                    token_count += 1
                    req_generated_tokens = token_count

                    # --- Memory pressure guard (every 16 tokens) ---
                    if _mm is not None and token_count % 16 == 0:
                        try:
                            _mm.generation_guard(
                                token_count, context=f"stream_generate/{kind}"
                            )
                        except (MemoryPressureError, RuntimeError) as mp_err:
                            log.error(
                                f"Generation aborted at token {token_count}: {mp_err}"
                            )
                            await chunk_queue.put(
                                f"data: {json.dumps({'error': str(mp_err)})}\n\n"
                            )
                            _aborted = True
                            req_status = (
                                STATUS_PRESSURE_ABORT
                                if _REQUEST_LOG_AVAILABLE
                                else "pressure_abort"
                            )
                            req_error_msg = str(mp_err)
                            req_finish_reason = "memory_pressure"
                            break

                    token_text = (
                        response.text
                    )  # GenerationResponse.text contains the decoded text
                    if kind == "chat":
                        chunk = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": token_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    else:  # completions
                        chunk = {
                            "id": req_id,
                            "object": "text_completion",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [
                                {
                                    "index": 0,
                                    "text": token_text,
                                    "finish_reason": None,
                                    "logprobs": None,
                                }
                            ],
                        }
                    await chunk_queue.put(f"data: {json.dumps(chunk)}\n\n")

                mx.eval()
                t1 = time.time()

                # Record streaming stats (best-effort token count)
                if _DASHBOARD_AVAILABLE and metrics_store is not None:
                    elapsed = t1 - t0
                    pt = _tok_len(prompt)
                    asyncio.create_task(
                        metrics_store.record_generation(
                            GenerationStats(
                                timestamp=t1,
                                prompt_tokens=pt,
                                completion_tokens=token_count,
                                elapsed_s=round(elapsed, 3),
                                tokens_per_sec=round(
                                    token_count / max(elapsed, 1e-9), 1
                                ),
                                model_id=MODEL_ID,
                                kind=kind,
                            )
                        )
                    )

                # Send final chunk with finish_reason
                _finish = "stop" if not _aborted else "memory_pressure"
                req_finish_reason = _finish
                if kind == "chat":
                    final_chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": _finish,
                            }
                        ],
                    }
                else:
                    final_chunk = {
                        "id": req_id,
                        "object": "text_completion",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "text": "",
                                "finish_reason": _finish,
                                "logprobs": None,
                            }
                        ],
                    }
                await chunk_queue.put(f"data: {json.dumps(final_chunk)}\n\n")
                await chunk_queue.put("data: [DONE]\n\n")
                await chunk_queue.put(None)  # Signal end of stream

                rank0_wait_done(_world.size())

            else:
                # Non-streaming mode: use future
                fut: asyncio.Future = result_target
                t0 = time.time()
                out_text = generate(_model, _tok, prompt, max_tokens=max_t)
                mx.eval()
                t1 = time.time()

                rank0_wait_done(_world.size())

                completion = (
                    out_text[len(prompt) :] if out_text.startswith(prompt) else out_text
                )
                pt = _tok_len(prompt)
                ct = _tok_len(completion)
                elapsed = t1 - t0
                req_generated_tokens = ct
                req_prompt_tokens = pt

                timing = {
                    "seconds": round(elapsed, 3),
                    "tokens_per_sec": round(ct / max(elapsed, 1e-9), 3),
                }

                # Record non-streaming stats
                if _DASHBOARD_AVAILABLE and metrics_store is not None:
                    asyncio.create_task(
                        metrics_store.record_generation(
                            GenerationStats(
                                timestamp=t1,
                                prompt_tokens=pt,
                                completion_tokens=ct,
                                elapsed_s=round(elapsed, 3),
                                tokens_per_sec=timing["tokens_per_sec"],
                                model_id=MODEL_ID,
                                kind=kind,
                            )
                        )
                    )

                if kind == "chat":
                    resp = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": completion},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": pt,
                            "completion_tokens": ct,
                            "total_tokens": pt + ct,
                        },
                        "timing": timing,
                    }
                elif kind == "completions":
                    resp = {
                        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "text": completion,
                                "finish_reason": "stop",
                                "logprobs": None,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": pt,
                            "completion_tokens": ct,
                            "total_tokens": pt + ct,
                        },
                        "timing": timing,
                    }
                else:
                    raise RuntimeError(f"Unknown request kind: {kind}")

                fut.set_result(resp)

        except MemoryPressureError as mpe:
            # Memory pressure — log prominently and clean up
            log.error(f"MEMORY PRESSURE in queue worker: {mpe}")
            req_status = (
                STATUS_PRESSURE_ABORT if _REQUEST_LOG_AVAILABLE else "pressure_abort"
            )
            req_error_msg = str(mpe)
            req_finish_reason = "memory_pressure"
            if _DASHBOARD_AVAILABLE and metrics_store is not None:
                asyncio.create_task(metrics_store.record_error())
            if is_stream:
                chunk_queue: asyncio.Queue = result_target
                await chunk_queue.put(
                    f"data: {json.dumps({'error': f'Memory pressure: {mpe}'})}\n\n"
                )
                await chunk_queue.put("data: [DONE]\n\n")
                await chunk_queue.put(None)
            else:
                result_target.set_exception(mpe)
        except Exception as e:
            req_status = STATUS_ERROR if _REQUEST_LOG_AVAILABLE else "error"
            req_error_msg = str(e)
            req_finish_reason = "error"
            if _DASHBOARD_AVAILABLE and metrics_store is not None:
                asyncio.create_task(metrics_store.record_error())
            if is_stream:
                chunk_queue: asyncio.Queue = result_target
                await chunk_queue.put(f"data: {json.dumps({'error': str(e)})}\n\n")
                await chunk_queue.put("data: [DONE]\n\n")
                await chunk_queue.put(None)
            else:
                result_target.set_exception(e)
        finally:
            # Post-request GC: reclaim KV cache + intermediate tensors.
            # This is critical — without it, memory from previous requests
            # accumulates and eventually triggers the kernel panic.
            if _mm is not None:
                _mm.gc_cycle(reason=f"{kind} request done")
            else:
                gc.collect()
                mx.clear_cache()

            # ── Record request to log ─────────────────────────────
            if _rlog is not None and _REQUEST_LOG_AVAILABLE:
                mem_after = _mm.active_gb() if _mm else 0.0
                wall = time.time() - req_timestamp
                tps = (
                    req_generated_tokens / max(wall, 1e-9)
                    if req_generated_tokens > 0
                    else 0.0
                )
                _rlog.record(
                    RequestRecord(
                        request_id=req_id,
                        timestamp=req_timestamp,
                        kind=kind,
                        model_id=MODEL_ID,
                        prompt_tokens=req_prompt_tokens,
                        generated_tokens=req_generated_tokens,
                        max_tokens_requested=max_t,
                        wall_time_s=wall,
                        tokens_per_second=tps,
                        memory_before_gb=mem_before,
                        memory_after_gb=mem_after,
                        memory_delta_gb=mem_after - mem_before,
                        kv_cache_hit=False,  # Phase S2 will populate this
                        status=req_status,
                        error_message=req_error_msg,
                        finish_reason=req_finish_reason,
                        is_stream=is_stream,
                    )
                )

            _queue.task_done()


def _print_ready_banner() -> None:
    """Print a production-grade startup banner once the server is fully ready."""
    W = 71  # inner width between ║ chars (matches 71 ═ in top/bottom borders)

    def _row(text: str = "") -> str:
        """Return a box row: '  ║' + text padded to W + '║'"""
        # Emoji like ⚡🧠🌐 occupy 2 display columns but len() counts 1.
        # Count them and subtract from padding budget.
        extra = sum(1 for ch in text if ord(ch) > 0xFFFF)
        return f"  ║{text}{' ' * max(0, W - len(text) - extra)}║"

    def _sep(color: str = "") -> str:
        rst = "\033[0m" if color else ""
        return f"{color}  ╔{'═' * W}╗{rst}"

    def _bot(color: str = "") -> str:
        rst = "\033[0m" if color else ""
        return f"{color}  ╚{'═' * W}╝{rst}"

    host_display = "localhost" if HOST == "0.0.0.0" else HOST
    base_url = f"http://{host_display}:{PORT}"
    world_size = _world.size() if _world else 1

    # ── Gather model metadata ────────────────────────────────────────────
    model_arch = ""
    model_quant = ""
    model_size = ""
    try:
        config_path = Path(MODEL_DIR) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            model_arch = cfg.get(
                "model_type",
                cfg.get("architectures", [""])[0] if cfg.get("architectures") else "",
            )
            hidden = cfg.get("hidden_size", "")
            layers = cfg.get("num_hidden_layers", "")
            if hidden and layers:
                model_size = f"{hidden}h / {layers}L"
            q = cfg.get("quantization", {})
            if q:
                bits = q.get("bits", "")
                group = q.get("group_size", "")
                model_quant = f"{bits}-bit" + (f" (g{group})" if group else "")
    except Exception:
        pass

    # ── Gather disk size ─────────────────────────────────────────────────
    disk_size = ""
    try:
        total = sum(f.stat().st_size for f in Path(MODEL_DIR).rglob("*") if f.is_file())
        disk_size = f"{total / (1024**3):.1f} GB"
    except Exception:
        pass

    # ── Gather node info from hostfile ───────────────────────────────────
    hosts_data = []
    hostfile = os.environ.get("HOSTFILE", "")
    if hostfile and os.path.isfile(hostfile):
        try:
            with open(hostfile) as f:
                hosts_data = json.load(f)
        except Exception:
            pass

    node_rows = []
    if hosts_data:
        for i, h in enumerate(hosts_data):
            role = "coordinator" if i == 0 else "worker"
            ssh = h.get("ssh", "?")
            rdma_devs = h.get("rdma", [])
            rdma = next((d for d in rdma_devs if d), "—")
            marker = "★" if i == 0 else "●"
            node_rows.append(
                _row(f"  {marker} rank {i}  {ssh:<20s}  {role:<13s} rdma: {rdma}")
            )
    else:
        node_rows.append(_row(f"  ● {world_size} node(s)"))

    # RDMA link line
    if len(hosts_data) >= 2:
        n0 = hosts_data[0].get("ssh", "node0")
        n1 = hosts_data[1].get("ssh", "node1")
        node_rows.append(_row())
        node_rows.append(_row(f"    {n0}  <==== RDMA (Thunderbolt) ====>  {n1}"))

    # ── Model detail line ────────────────────────────────────────────────
    detail_parts = []
    if model_arch:
        detail_parts.append(model_arch)
    if model_quant:
        detail_parts.append(model_quant)
    if model_size:
        detail_parts.append(model_size)
    if disk_size:
        detail_parts.append(disk_size)
    model_detail = "  |  ".join(detail_parts)

    shard_info = ""
    if disk_size and world_size > 1:
        try:
            gb = float(disk_size.replace(" GB", ""))
            shard_info = f"  ~{gb / world_size:.1f} GB/node (sharded)"
        except Exception:
            pass

    tp_line = f"Parallelism: Tensor Parallel x {world_size} nodes{shard_info}"

    # ── Endpoint rows ────────────────────────────────────────────────────
    endpoints = [
        ("Chat API", f"{base_url}/v1/chat/completions"),
        ("Completions", f"{base_url}/v1/completions"),
        ("Models", f"{base_url}/v1/models"),
        ("Dashboard", f"{base_url}/dashboard"),
        ("Health", f"{base_url}/health"),
    ]
    ep_rows = []
    for label, url in endpoints:
        ep_rows.append(_row(f"  {label:<14s} {url}"))

    # ── Assemble banner ──────────────────────────────────────────────────
    C = "\033[1;36m"  # cyan bold   — logo
    Y = "\033[1;33m"  # yellow bold — cluster
    B = "\033[1m"  # bold        — model
    G = "\033[1;32m"  # green bold  — API
    D = "\033[2m"  # dim         — hints
    R = "\033[0m"  # reset

    lines = [
        "",
        f"{C}{_sep()}",
        f"{C}{_row()}",
        f"{C}{_row('       ██╗ █████╗  ██████╗ ██████╗██╗')}",
        f"{C}{_row('       ██║██╔══██╗██╔════╝██╔════╝██║')}",
        f"{C}{_row('       ██║███████║██║     ██║     ██║')}",
        f"{C}{_row('  ██   ██║██╔══██║██║     ██║     ██║')}",
        f"{C}{_row('  ╚█████╔╝██║  ██║╚██████╗╚██████╗███████╗')}",
        f"{C}{_row('   ╚════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═════╝╚══════╝')}",
        f"{C}{_row()}",
        f"{C}{_row('  Distributed ML Inference over RDMA / Thunderbolt')}",
        f"{C}{_row()}",
        f"{C}{_bot()}{R}",
        "",
        f"{Y}{_sep()}",
        f"{Y}{_row()}",
        f"{Y}{_row('  ⚡ Cluster Online')}",
        f"{Y}{_row()}",
    ]
    for nr in node_rows:
        lines.append(f"{Y}{nr}")
    lines += [
        f"{Y}{_row()}",
        f"{Y}{_bot()}{R}",
        "",
        f"{B}{_sep()}",
        f"{B}{_row()}",
        f"{B}{_row(f'  🧠 Model: {MODEL_ID}')}",
        f"{B}{_row(f'     {model_detail}')}",
        f"{B}{_row(f'     {tp_line}')}",
        f"{B}{_row()}",
        f"{B}{_bot()}{R}",
        "",
        f"{G}{_sep()}",
        f"{G}{_row()}",
        f"{G}{_row('  🌐 API & Dashboard Ready')}",
        f"{G}{_row()}",
        f"{G}{_row(f'  {base_url}')}",
        f"{G}{_row()}",
    ]
    for er in ep_rows:
        lines.append(f"{G}{er}")
    lines += [
        f"{G}{_row()}",
        f"{G}{_bot()}{R}",
        "",
        f"  {D}Queue: {QUEUE_MAX} max concurrent  |  Timeout: {REQ_TIMEOUT}s per request{R}",
        "",
        f"  {D}Usage:  curl {base_url}/v1/chat/completions \\{R}",
        f"  {D}        -H 'Content-Type: application/json' \\{R}",
        f'  {D}        -d \'{{"messages":[{{"role":"user","content":"Hello!"}}],"max_tokens":64}}\'{R}',
        "",
        f'  {D}Python: client = OpenAI(base_url="{base_url}/v1", api_key="none"){R}',
        "",
        f"  {G}✓ Running on {HOST}:{PORT} (CTRL + C to quit){R}",
        "",
    ]

    print("\n".join(lines), flush=True)


# -------------------------
# Dashboard mounting helper
# -------------------------
def _mount_dashboard_now() -> None:
    """Mount dashboard routes. Called on rank-0 startup after _world is set."""
    if not _DASHBOARD_AVAILABLE:
        return

    # Detect RDMA devices from environment / hostfile heuristic
    rdma_raw = os.environ.get("RDMA_DEVICES", "")
    if rdma_raw:
        rdma_devices = [d.strip() for d in rdma_raw.split(",")]
    else:
        # Default: both nodes use rdma_en4 (confirmed working on M4 Pro)
        rdma_devices = ["rdma_en4"] * (_world.size() if _world else 2)

    hostfile = os.environ.get("HOSTFILE", "")

    mount_dashboard(
        app,
        get_state=lambda: {},
        get_queue_info=lambda: {"queue_size": _queue.qsize(), "queue_max": QUEUE_MAX},
        model_id=MODEL_ID,
        world_size=_world.size() if _world else 1,
        rank=_world.rank() if _world else 0,
        queue_max=QUEUE_MAX,
        rdma_devices=rdma_devices,
        host=HOST,
        port=PORT,
        hostfile=hostfile,
        memory_monitor=_monitor,  # share macmon data with dashboard
    )
    print(
        f"[rank0] dashboard mounted at http://{HOST if HOST != '0.0.0.0' else 'localhost'}:{PORT}/dashboard",
        flush=True,
    )


# -------------------------
# HTTP endpoints (rank0 only)
# -------------------------
@app.get("/health")
def health() -> dict:
    result = {
        "ok": True,
        "world_size": _world.size(),
        "rank": _world.rank(),
        "model": MODEL_ID,
        "model_loaded": _model is not None,
        "queue_max": QUEUE_MAX,
        "queue_size": _queue.qsize(),
    }
    if _mm is not None:
        snap = _mm.snapshot()
        result["memory"] = {
            "active_gb": round(snap.active_gb, 3),
            "peak_gb": round(snap.peak_gb, 3),
            "limit_gb": round(snap.limit_gb, 1),
            "pressure_pct": round(snap.pressure_pct, 1),
            "headroom_gb": round(_mm.headroom_gb(), 2),
        }
        result["memory_safe"] = snap.pressure_pct < 80.0
    if _monitor is not None:
        result["monitor"] = {
            "running": _monitor.running,
            "source": "macmon" if _monitor.using_macmon else "vm_stat",
            "peak_pressure_1m": round(_monitor.peak_pressure_1m, 1),
            "memory_threshold": _monitor.memory_threshold,
        }
    if _rlog is not None:
        result["requests_total"] = _rlog.entry_count
    return result


@app.get("/v1/models")
def list_models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.get("/queue")
def queue_status() -> dict:
    return {"size": _queue.qsize(), "max": QUEUE_MAX}


# -------------------------
# Memory & model management endpoints
# -------------------------
@app.get("/memory")
def memory_status() -> dict:
    """Return current memory state — use this to monitor before/during/after loads."""
    if _mm is not None:
        snap = _mm.snapshot()
        sys_pressure = _mm.system_memory_pressure()
        result = {
            "ok": True,
            "mlx": snap.to_dict(),
            "system": sys_pressure,
            "slot": _mm.slot_info,
            "limits": {
                "memory_limit_gb": round(_mm._memory_limit / (1024**3), 2),
                "cache_limit_gb": round(_mm._cache_limit / (1024**3), 2),
                "hard_max_tokens": _mm.hard_max_tokens,
                "system_reserve_gb": _mm._system_reserve_gb,
                "pressure_warn": _mm._pressure_warn,
                "pressure_critical": _mm._pressure_critical,
            },
        }
        # Include live monitor data if available
        if _monitor is not None:
            result["monitor"] = _monitor.summary()
        return result
    # Fallback without memory manager
    return {
        "ok": False,
        "error": "memory_manager not available",
        "active_gb": round(mx.get_active_memory() / (1024**3), 3),
        "peak_gb": round(mx.get_peak_memory() / (1024**3), 3),
        "cache_gb": round(mx.get_cache_memory() / (1024**3), 3),
    }


@app.get("/memory/live")
def memory_live(n: int = 60) -> dict:
    """Return rolling pressure history from the background monitor.

    Query params:
        n: number of recent data points (default 60, max = history size)

    Data source: macmon persistent pipe (like exo) + MLX allocator polls.
    """
    if _monitor is None:
        return {"ok": False, "error": "MemoryMonitor not running"}
    return {
        "ok": True,
        "source": "macmon" if _monitor.using_macmon else "vm_stat",
        "count": len(_monitor.history),
        "data": _monitor.history_dicts(last_n=n),
    }


# -------------------------
# Request log endpoints
# -------------------------
@app.get("/requests/recent")
def requests_recent(n: int = 50) -> dict:
    """Return the last N request records."""
    if _rlog is None:
        return {"ok": False, "error": "RequestLog not available"}
    return {"ok": True, "count": min(n, _rlog.entry_count), "requests": _rlog.recent(n)}


@app.get("/requests/stats")
def requests_stats() -> dict:
    """Return aggregate request statistics (totals, error rate, p50/p95 latency & tok/s)."""
    if _rlog is None:
        return {"ok": False, "error": "RequestLog not available"}
    return {"ok": True, **_rlog.stats()}


@app.get("/model/info")
def model_info() -> dict:
    """Return metadata about the currently loaded model."""
    result = {
        "model_id": MODEL_ID,
        "model_dir": MODEL_DIR,
        "loaded": _model is not None,
    }
    if _mm is not None:
        result["slot"] = _mm.slot_info
        snap = _mm.snapshot()
        result["memory"] = {
            "model_size_gb": snap.model_size_gb,
            "active_gb": snap.active_gb,
            "pressure_pct": snap.pressure_pct,
        }
    return result


class ModelLoadReq(BaseModel):
    model_dir: str
    model_id: Optional[str] = None


@app.post("/model/unload")
async def model_unload() -> dict:
    """
    Unload the current model, releasing all GPU memory.

    This is the critical operation for preventing kernel panics when
    switching models or when memory pressure is high.

    Steps performed:
      1. Drain the request queue (reject new requests)
      2. Delete model + tokenizer Python references
      3. Run GC (3 passes for reference cycles)
      4. Clear MLX Metal buffer cache
      5. Reset peak memory tracker

    After unload, the server will reject inference requests until
    a new model is loaded via POST /model/load.
    """
    global _model, _tok

    if _model is None:
        return {
            "status": "no_model",
            "message": "No model is currently loaded",
            "memory": _mm.snapshot().to_dict() if _mm else {},
        }

    # Reject if there are queued requests
    if _queue.qsize() > 0:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot unload while {_queue.qsize()} request(s) are queued. "
            f"Wait for them to finish or restart the server.",
        )

    old_model_id = MODEL_ID

    if _mm is not None:
        # CRITICAL: Clear the global references BEFORE calling mm.unload_model().
        # The memory manager's unload does `del` on its internal slot references,
        # but if _model/_tok globals still point to the same objects, the Python
        # refcount stays > 0 and Metal buffers are NOT freed.
        # By setting globals to None first, the only remaining refs are inside
        # the MemoryManager's slot — when unload_model() deletes those, refcount
        # hits zero and Metal buffers are released immediately.
        _model = None
        _tok = None
        result = _mm.unload_model()
        result["memory_after"] = _mm.snapshot().to_dict()
        log.info(f"Model unloaded via API: {old_model_id}")
        return result
    else:
        # Manual unload without memory manager
        before = mx.get_active_memory() / (1024**3)
        _model = None
        _tok = None
        gc.collect()
        gc.collect()
        gc.collect()
        mx.clear_cache()
        time.sleep(0.05)
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()
        after = mx.get_active_memory() / (1024**3)
        return {
            "status": "unloaded",
            "model_id": old_model_id,
            "before_active_gb": round(before, 3),
            "after_active_gb": round(after, 3),
            "freed_gb": round(before - after, 3),
        }


@app.post("/model/load")
async def model_load(req: ModelLoadReq) -> dict:
    """
    Load a new model (unloading any existing one first).

    This is the safe way to switch models at runtime without
    restarting the server process.

    Body:
      { "model_dir": "/path/to/model", "model_id": "optional-name" }
    """
    global _model, _tok, MODEL_ID, MODEL_DIR

    model_dir = req.model_dir
    new_model_id = req.model_id or os.path.basename(model_dir.rstrip("/"))

    if not os.path.isdir(model_dir):
        raise HTTPException(
            status_code=400,
            detail=f"Model directory not found: {model_dir}",
        )

    # Reject if there are queued requests
    if _queue.qsize() > 0:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot load while {_queue.qsize()} request(s) are queued.",
        )

    log.info(f"API model load requested: {new_model_id} from {model_dir}")

    try:
        if _mm is not None:
            # Memory-safe load (unloads previous model automatically)
            model, tok = _mm.load_model(
                model_dir, world=_world, model_id=new_model_id, lazy=False
            )

            # Patch Qwen3 thinking if needed
            if _mm.should_disable_thinking(new_model_id):
                tok = _mm.patch_chat_template_no_thinking(tok)

            _model = model
            _tok = tok
            MODEL_ID = new_model_id
            MODEL_DIR = model_dir

            # Broadcast to workers that model changed
            # (workers need to load too — for now they must restart)

            snap = _mm.snapshot()
            return {
                "status": "loaded",
                "model_id": new_model_id,
                "model_dir": model_dir,
                "slot": _mm.slot_info,
                "memory": snap.to_dict(),
            }
        else:
            # Without memory manager — manual load
            # Unload first
            _model = None
            _tok = None
            gc.collect()
            gc.collect()
            mx.clear_cache()

            _model, _tok = sharded_load_with_fallback(model_dir)
            MODEL_ID = new_model_id
            MODEL_DIR = model_dir

            return {
                "status": "loaded",
                "model_id": new_model_id,
                "active_gb": round(mx.get_active_memory() / (1024**3), 3),
            }

    except MemoryPressureError as e:
        raise HTTPException(
            status_code=507,
            detail=f"Not enough memory to load {new_model_id}: {e}",
        )
    except Exception as e:
        log.error(f"Failed to load model {new_model_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {e}",
        )


@app.post("/model/gc")
async def model_gc() -> dict:
    """
    Force a garbage collection + Metal cache clear cycle.

    Use this to reclaim memory between heavy requests without
    unloading the model. Safe to call at any time.
    """
    if _mm is not None:
        result = _mm.gc_cycle(reason="API request")
        result["memory"] = _mm.snapshot().to_dict()
        return result
    else:
        before = mx.get_active_memory() / (1024**3)
        gc.collect()
        mx.clear_cache()
        after = mx.get_active_memory() / (1024**3)
        return {
            "before_active_gb": round(before, 3),
            "after_active_gb": round(after, 3),
            "freed_gb": round(before - after, 3),
        }


async def _stream_generator(chunk_queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    """Yield SSE chunks from the queue until None is received."""
    while True:
        chunk = await chunk_queue.get()
        if chunk is None:
            break
        yield chunk


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionsReq):
    if req.stream and stream_generate is None:
        raise HTTPException(
            status_code=400,
            detail="stream=true not supported (stream_generate not available)",
        )
    if req.model and req.model != MODEL_ID:
        raise HTTPException(
            status_code=400, detail=f"Only model '{MODEL_ID}' is served"
        )

    if _world.rank() != 0:
        raise HTTPException(status_code=500, detail="Rank != 0 received HTTP request")

    # Pre-request memory check — reject early if we're already in trouble
    if _mm is not None:
        try:
            _mm.check_pressure(context="pre-request chat")
        except MemoryPressureError as e:
            raise HTTPException(
                status_code=507,
                detail=f"Memory pressure too high to accept request: {e}",
            )

    prompt = _build_chat_prompt(req.messages)
    max_t = _safe_max_tokens(req.max_tokens or DEFAULT_MAX_TOKENS)

    if req.stream:
        # Streaming mode: return SSE response
        chunk_queue: asyncio.Queue = asyncio.Queue()
        try:
            _queue.put_nowait(("chat", prompt, max_t, chunk_queue, True))
        except asyncio.QueueFull:
            raise HTTPException(
                status_code=429, detail="Server busy (queue full). Try again later."
            )

        return StreamingResponse(
            _stream_generator(chunk_queue),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming mode: return JSON response
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        try:
            _queue.put_nowait(("chat", prompt, max_t, fut, False))
        except asyncio.QueueFull:
            raise HTTPException(
                status_code=429, detail="Server busy (queue full). Try again later."
            )

        try:
            return await asyncio.wait_for(fut, timeout=REQ_TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out")


@app.post("/v1/completions")
async def completions(req: CompletionsReq):
    # Pre-request memory check
    if _mm is not None:
        try:
            _mm.check_pressure(context="pre-request completions")
        except MemoryPressureError as e:
            raise HTTPException(
                status_code=507,
                detail=f"Memory pressure too high to accept request: {e}",
            )

    if req.stream and stream_generate is None:
        raise HTTPException(
            status_code=400,
            detail="stream=true not supported (stream_generate not available)",
        )
    if req.model and req.model != MODEL_ID:
        raise HTTPException(
            status_code=400, detail=f"Only model '{MODEL_ID}' is served"
        )

    if _world.rank() != 0:
        raise HTTPException(status_code=500, detail="Rank != 0 received HTTP request")

    if isinstance(req.prompt, list):
        # Keep it simple + safe for distributed mode: one prompt at a time.
        if len(req.prompt) != 1:
            raise HTTPException(
                status_code=400,
                detail="Only a single prompt string is supported (prompt must be a string, or a list of length 1).",
            )
        prompt = req.prompt[0]
    else:
        prompt = req.prompt

    max_t = req.max_tokens or DEFAULT_MAX_TOKENS

    if req.stream:
        # Streaming mode: return SSE response
        chunk_queue: asyncio.Queue = asyncio.Queue()
        try:
            _queue.put_nowait(("completions", prompt, max_t, chunk_queue, True))
        except asyncio.QueueFull:
            raise HTTPException(
                status_code=429, detail="Server busy (queue full). Try again later."
            )

        return StreamingResponse(
            _stream_generator(chunk_queue),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming mode: return JSON response
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        try:
            _queue.put_nowait(("completions", prompt, max_t, fut, False))
        except asyncio.QueueFull:
            raise HTTPException(
                status_code=429, detail="Server busy (queue full). Try again later."
            )

        try:
            return await asyncio.wait_for(fut, timeout=REQ_TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out")


# -------------------------
# Main
# -------------------------
def _graceful_shutdown(signum=None, frame=None) -> None:
    """Send shutdown to all workers, print exit banner, then exit."""
    sig_name = signal.Signals(signum).name if signum else "EXIT"
    D = "\033[2m"
    G = "\033[1;32m"
    R = "\033[0m"

    # Tell workers to exit
    with _worker_lock:
        for r, sock in _worker_socks.items():
            try:
                send_msg(sock, {"type": "shutdown"})
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass

    print(f"\n{D}  [{sig_name}] Shutting down...{R}", flush=True)
    print(f"{G}  ✓ Server stopped. GPU memory released on all nodes.{R}", flush=True)
    print(
        f"{D}  Tip: run 'make mem' to verify  |  'make kill-all' to force-clean{R}\n",
        flush=True,
    )

    # Exit without raising (avoids ugly traceback on Ctrl+C)
    os._exit(0)


def main() -> None:
    global _model, _tok, _world, _mm

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _world = mx.distributed.init()
    rank = _world.rank()

    # Initialize memory manager (if not already done at module level)
    if _MEMORY_MANAGER_AVAILABLE and _mm is None:
        _mm = init_manager()

    if _mm is not None and rank == 0:
        log.info("Memory safety enabled — kernel panic protection active")
        _mm.print_status()
    elif _mm is None and rank == 0:
        log.warning(
            "⚠ Memory manager NOT available — no protection against "
            "GPU memory exhaustion kernel panics!"
        )

    _model, _tok = sharded_load_with_fallback(MODEL_DIR)

    if _world.rank() == 0:
        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, _graceful_shutdown)
        signal.signal(signal.SIGTERM, _graceful_shutdown)

        # Mount dashboard before uvicorn starts (routes must be registered beforehand)
        if _DASHBOARD_AVAILABLE:
            _mount_dashboard_now()

        th = threading.Thread(
            target=rank0_accept_workers, args=(_world.size(),), daemon=True
        )
        th.start()

        if not rank0_wait_for_workers(_world.size(), timeout_s=60):
            raise RuntimeError("Workers did not connect to control-plane in time")

        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            log_level="warning",
            # Let our signal handler run instead of uvicorn's default
            # (uvicorn installs its own SIGINT handler that raises SystemExit)
        )
    else:
        # Workers: exit cleanly on SIGINT/SIGTERM too
        def _worker_exit(signum=None, frame=None):
            rank = _world.rank() if _world else "?"
            print(f"\n[worker {rank}] signal received — exiting.", flush=True)
            os._exit(0)

        signal.signal(signal.SIGINT, _worker_exit)
        signal.signal(signal.SIGTERM, _worker_exit)

        worker_loop(_world.rank())


if __name__ == "__main__":
    main()
