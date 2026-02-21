#!/usr/bin/env python3
import asyncio
import atexit
import json
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

# Dashboard (optional — only mounted on rank 0)
try:
    from dashboard import GenerationStats, metrics_store, mount_dashboard

    _DASHBOARD_AVAILABLE = True
except ImportError:
    _DASHBOARD_AVAILABLE = False
    metrics_store = None
    GenerationStats = None


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
    """
    model_path = Path(repo)
    world = mx.distributed.init()
    rank = world.rank()

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
# -------------------------
# Lifespan (replaces deprecated @app.on_event)
# -------------------------
@asynccontextmanager
async def _lifespan(application):
    """Startup/shutdown lifespan for FastAPI — runs the queue worker on rank0."""
    if _world and _world.rank() == 0:
        asyncio.create_task(_queue_worker())
        _print_ready_banner()
    yield
    # shutdown: nothing to clean up (daemon threads die with the process)


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
    # Prefer tokenizer chat template when available
    if hasattr(_tok, "apply_chat_template"):
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        return _tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    # Fallback: simple "ROLE: content" format
    parts = [f"{m.role.upper()}: {m.content}" for m in messages]
    parts.append("ASSISTANT:")
    return "\n".join(parts)


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
    """
    while True:
        item = await _queue.get()
        if item is None:
            _queue.task_done()
            continue

        kind, prompt, max_t, result_target, is_stream = (
            item  # kind: "chat" | "completions"
        )
        try:
            rank0_broadcast_task({"prompt": prompt, "max_tokens": max_t})

            if is_stream and stream_generate is not None:
                # Streaming mode: yield chunks via async queue
                chunk_queue: asyncio.Queue = result_target
                req_id = (
                    f"chatcmpl-{uuid.uuid4().hex[:24]}"
                    if kind == "chat"
                    else f"cmpl-{uuid.uuid4().hex[:24]}"
                )
                created = int(time.time())

                t0 = time.time()
                token_count = 0

                for response in stream_generate(_model, _tok, prompt, max_tokens=max_t):
                    token_count += 1
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
                                "finish_reason": "stop",
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
                                "finish_reason": "stop",
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

        except Exception as e:
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
    return {
        "ok": True,
        "world_size": _world.size(),
        "rank": _world.rank(),
        "model": MODEL_ID,
        "queue_max": QUEUE_MAX,
        "queue_size": _queue.qsize(),
    }


@app.get("/v1/models")
def list_models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.get("/queue")
def queue_status() -> dict:
    return {"size": _queue.qsize(), "max": QUEUE_MAX}


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

    prompt = _build_chat_prompt(req.messages)
    max_t = req.max_tokens or DEFAULT_MAX_TOKENS

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
    global _model, _tok, _world
    _world = mx.distributed.init()
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
