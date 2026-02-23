#!/usr/bin/env python3
import asyncio
import atexit
import gc
import json
import re
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
# OpenAI-ish schemas (+ tools MVP)
# -------------------------
class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON string (OpenAI-style)


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class ToolDefinitionFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: dict  # JSON Schema


class ToolDefinition(BaseModel):
    type: str = "function"
    function: ToolDefinitionFunction


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None

    # OpenAI tool calling fields (subset)
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class ChatCompletionsReq(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

    # Tool calling (MVP: non-streaming only)
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: Optional[Union[str, dict]] = None  # "auto" | "none" | {"type":"function","function":{"name":...}}


class CompletionsReq(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, list[str]]
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


# -------------------------
# Tool calling helpers (MVP)
# -------------------------
_TOOL_JSON_MODE_INSTRUCTION = """You are a tool-calling assistant.

You MUST respond with exactly one JSON object and nothing else.

Choose ONE of these shapes:

1) To call tools:
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "<tool_name>",
        "arguments": { ... JSON arguments ... }
      }
    }
  ]
}

2) To answer normally (no tool call):
{
  "content": "<final answer as a string>"
}

Rules:
- Do not wrap the JSON in markdown fences.
- If tool_choice is "none", you MUST return {"content": "..."}.
- If tool_choice forces a specific function name, you MUST call that function.
- If you call tools, arguments MUST be a JSON object (not a string).
"""

_MODEL_STOP_MARKERS = (
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|observation|>",
    "<|im_end|>",
    "<|endoftext|>",
    "[gMASK]",
    "[sMASK]",
    "[MASK]",
    "<sop>",
    "<eop>",
    "<|begin_of_image|>",
    "<|end_of_image|>",
    "<|begin_of_video|>",
    "<|end_of_video|>",
    "<|begin_of_audio|>",
    "<|end_of_audio|>",
    "<|begin_of_transcription|>",
    "<|end_of_transcription|>",
)

_STRIP_STRUCTURED_REASONING = (
    os.environ.get("MLX_STRIP_STRUCTURED_REASONING", "1") == "1"
)


def _strip_structured_reasoning_preamble(text: str) -> str:
    """
    Some models output "analysis steps + final answer" in plain text (without
    <think> tags), which Cherry cannot separate. Strip common structured
    reasoning preambles and keep the final answer part.
    """
    if not text:
        return text
    s = text.strip()

    # Chinese/English markers that usually indicate chain-of-thought blocks.
    has_reasoning_markers = any(
        k in s
        for k in (
            "分析用户输入",
            "识别核心实体",
            "确定事实",
            "检查约束条件",
            "构思回复",
            "最终润色",
            "Analyze the user's input",
            "Determine the intent",
            "Final Output",
        )
    )
    if not has_reasoning_markers:
        return s

    # Prefer content after explicit "final polish/output" anchors.
    for pat in (
        r"(?:最终润色|最终输出|Final Polish|Final Output Generation)\s*[:：]\s*",
        r"(?:最终答案|Final Answer)\s*[:：]\s*",
    ):
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            tail = s[m.end() :].strip()
            if tail:
                # If there are quoted candidates, keep the last quoted sentence.
                quoted = re.findall(r"[“\"]([^”\"]{2,300})[”\"]", tail)
                if quoted:
                    return quoted[-1].strip()
                return tail

    # Fallback: keep last non-empty line that looks like a final sentence.
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return s


def _sanitize_assistant_output(text: str) -> str:
    """
    Normalize model output for OpenAI-compatible clients:
    - remove reasoning blocks (<think>...</think>)
    - trim at role/control markers (<|user|> etc.)
    """
    if not text:
        return ""

    s = text
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"</?think>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"</?tool_response>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"</?tools>", "", s, flags=re.IGNORECASE)
    s = re.sub(
        r"^(?:\s*(?:<\|assistant\|>|<\|system\|>|<\|user\|>|<\|observation\|>))+",
        "",
        s,
    )
    s = re.sub(r"^(?:\s*</think>\s*)+", "", s, flags=re.IGNORECASE)

    # In streaming, model may emit an open <think> before the closing tag arrives.
    # Hide reasoning content until closure appears.
    low = s.lower()
    open_idx = low.rfind("<think>")
    close_idx = low.rfind("</think>")
    if open_idx != -1 and close_idx < open_idx:
        s = s[:open_idx]

    for marker in _MODEL_STOP_MARKERS:
        idx = s.find(marker)
        if idx != -1:
            s = s[:idx]
            break

    if _STRIP_STRUCTURED_REASONING:
        s = _strip_structured_reasoning_preamble(s)

    return s.strip()


def _contains_model_stop_marker(text: str) -> bool:
    return any(marker in text for marker in _MODEL_STOP_MARKERS)


def _extract_json_obj(text: str) -> dict:
    """Best-effort extraction of a JSON object from model output.

    Handles:
    - leading/trailing text
    - markdown fences
    - accidental prefix like 'json:' or '```json'
    """
    if not text:
        raise ValueError("empty model output")

    s = text.strip()

    # Strip markdown fences if present
    if s.startswith("```"):
        # remove first fence line
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s).strip()

    # If the output includes extra text, scan for first '{' and parse with brace matching
    start = s.find("{")
    if start == -1:
        raise ValueError("no '{' found in output")

    # brace matching to find a complete JSON object
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":  # escape
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    return json.loads(candidate)

    raise ValueError("unterminated JSON object")


def _inject_tool_calling_prompt(
    messages: list[ChatMessage],
    tools: list[ToolDefinition],
    tool_choice: Optional[Union[str, dict]],
) -> str:
    """Build a single text prompt that instructs the model to emit tool JSON."""
    # Tools block (as JSON schema-ish)
    tools_payload = [
        {
            "type": t.type,
            "function": {
                "name": t.function.name,
                "description": t.function.description,
                "parameters": t.function.parameters,
            },
        }
        for t in tools
    ]

    forced = tool_choice or "auto"
    # Normalize tool_choice: allow {"type":"function","function":{"name":...}}
    forced_name = None
    if isinstance(forced, dict):
        try:
            if forced.get("type") == "function":
                forced_name = forced.get("function", {}).get("name")
        except Exception:
            forced_name = None

    # Turn chat messages into a compact transcript.
    # We avoid relying on tokenizer chat templates so tool-role messages work everywhere.
    lines: list[str] = []
    for m in messages:
        role = (m.role or "").strip().lower()

        if role == "tool":
            tool_name = m.name or "tool"
            tcid = m.tool_call_id or "-"
            lines.append(f"TOOL[{tool_name} id={tcid}]: {m.content or ''}".strip())
            continue

        if role == "assistant" and m.tool_calls:
            # Previous tool call(s) issued by the assistant
            try:
                for tc in m.tool_calls:
                    args_obj = {}
                    try:
                        args_obj = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except Exception:
                        args_obj = {"_raw": tc.function.arguments}
                    lines.append(
                        f"ASSISTANT_TOOL_CALL[{tc.function.name} id={tc.id}]: {json.dumps(args_obj, ensure_ascii=False)}"
                    )
            except Exception:
                pass
            # Also include any assistant content (if present)
            if m.content:
                lines.append(f"ASSISTANT: {m.content}")
            continue

        # default: user/system/assistant content
        if role == "system":
            lines.append(f"SYSTEM: {m.content or ''}")
        elif role == "user":
            lines.append(f"USER: {m.content or ''}")
        elif role == "assistant":
            lines.append(f"ASSISTANT: {m.content or ''}")
        else:
            lines.append(f"{role.upper()}: {m.content or ''}")

    transcript = "\n".join(lines).strip()

    choice_line = f"tool_choice={json.dumps(forced, ensure_ascii=False)}"
    if forced_name:
        choice_line += f" (forced_name={forced_name})"

    prompt = (
        _TOOL_JSON_MODE_INSTRUCTION
        + "\n\nAvailable tools (JSON):\n"
        + json.dumps(tools_payload, ensure_ascii=False)
        + "\n\n"
        + choice_line
        + "\n\nConversation so far:\n"
        + transcript
        + "\n\nNow produce the JSON object: "
    )
    return prompt


def _tool_choice_instruction(tool_choice: Optional[Union[str, dict]]) -> str:
    if not tool_choice:
        return "tool_choice=auto"
    if isinstance(tool_choice, str):
        return f"tool_choice={tool_choice}"
    if isinstance(tool_choice, dict):
        try:
            if tool_choice.get("type") == "function":
                name = tool_choice.get("function", {}).get("name")
                if name:
                    return f"tool_choice=forced({name})"
        except Exception:
            pass
    return f"tool_choice={json.dumps(tool_choice, ensure_ascii=False)}"


def _use_glm_tool_template() -> bool:
    # Heuristic: GLM chat template contains <tool_call> markers
    try:
        tmpl = getattr(_tok, "chat_template", "") or ""
        return "<tool_call>" in tmpl
    except Exception:
        return False


def _build_glm_tool_prompt(
    messages: list[ChatMessage],
    tools: list[ToolDefinition],
    tool_choice: Optional[Union[str, dict]],
) -> str:
    # Prepend a system hint about tool_choice to bias the model
    tool_hint = _tool_choice_instruction(tool_choice)
    sys_msg = ChatMessage(role="system", content=f"Use tools when appropriate. {tool_hint}.")

    msgs = [sys_msg] + messages
    tools_payload = [
        {
            "type": t.type,
            "function": {
                "name": t.function.name,
                "description": t.function.description,
                "parameters": t.function.parameters,
            },
        }
        for t in tools
    ]

    if hasattr(_tok, "apply_chat_template"):
        msg_dicts = []
        for m in msgs:
            md = m.model_dump()
            # GLM template expects tool_calls[].function.arguments to be an object
            if md.get("role") == "assistant" and md.get("tool_calls"):
                for tc in md["tool_calls"]:
                    fn = tc.get("function", {})
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = json.loads(args)
                        except Exception:
                            fn["arguments"] = {"_raw": args}
            msg_dicts.append(md)
        return _tok.apply_chat_template(
            msg_dicts,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools_payload,
        )
    # Fallback to raw JSON-mode prompt if template is unavailable
    return _inject_tool_calling_prompt(messages, tools, tool_choice)


def _parse_tool_response(
    completion_text: str,
) -> tuple[Optional[list[ToolCall]], Optional[str]]:
    """Return (tool_calls, content). Exactly one will be non-None."""
    try:
        obj = _extract_json_obj(completion_text)
    except Exception:
        obj = None

    # GLM-style <tool_call> parsing (XML-ish)
    if obj is None and "<tool_call>" in completion_text:
        tool_calls: list[ToolCall] = []
        for block in re.findall(r"<tool_call>(.*?)</tool_call>", completion_text, re.DOTALL):
            block = block.strip()
            if not block:
                continue
            # First token before any <arg_key> is the function name
            fn_name = block.split("<arg_key>", 1)[0].strip()
            args: dict[str, str] = {}
            for k, v in re.findall(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", block, re.DOTALL):
                args[k.strip()] = v.strip().strip('"')
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=ToolCallFunction(
                        name=fn_name,
                        arguments=json.dumps(args, ensure_ascii=False),
                    ),
                )
            )
        if tool_calls:
            return tool_calls, None
        return None, completion_text.strip()

    # Heuristic fallback: "Tool: <name>" + "Parameters: {..}"
    if obj is None and "Tool:" in completion_text and "Parameters:" in completion_text:
        m = re.search(r"Tool:\s*([\w\-]+)\s*\nParameters:\s*(\{.*?\})", completion_text, re.DOTALL)
        if m:
            name = m.group(1).strip()
            args_raw = m.group(2).strip()
            try:
                args_obj = json.loads(args_raw)
            except Exception:
                args_obj = {"_raw": args_raw}
            return [
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=ToolCallFunction(
                        name=name,
                        arguments=json.dumps(args_obj, ensure_ascii=False),
                    ),
                )
            ], None

    if isinstance(obj, dict) and "tool_calls" in obj and obj["tool_calls"] is not None:
        calls_in = obj["tool_calls"]
        if not isinstance(calls_in, list):
            raise ValueError("tool_calls must be a list")

        tool_calls: list[ToolCall] = []
        for c in calls_in:
            if not isinstance(c, dict):
                continue
            fn = c.get("function") or {}
            args = fn.get("arguments", {})
            # OpenAI requires arguments to be a JSON string; we accept object and stringify it.
            if isinstance(args, (dict, list)):
                args_str = json.dumps(args, ensure_ascii=False)
            else:
                args_str = str(args)

            tool_calls.append(
                ToolCall(
                    id=str(c.get("id") or f"call_{uuid.uuid4().hex[:8]}"),
                    type=str(c.get("type") or "function"),
                    function=ToolCallFunction(
                        name=str(fn.get("name") or ""),
                        arguments=args_str,
                    ),
                )
            )

        return tool_calls, None

    # Normal answer
    if isinstance(obj, dict) and "content" in obj:
        return None, str(obj.get("content") or "")

    # Fallback: treat as content if JSON shape is unknown
    return None, completion_text.strip()


def _build_tool_calls_response(
    *,
    model_id: str,
    tool_calls: list[ToolCall],
    prompt_tokens: int,
    completion_tokens: int,
    timing: dict,
) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "timing": timing,
    }
def _build_chat_prompt(messages: list[ChatMessage], tool_ctx: Optional[dict] = None) -> str:
    # Tool calling MVP: build a raw-text prompt that requests a single JSON object.
    if tool_ctx and tool_ctx.get("tools"):
        if _use_glm_tool_template():
            return _build_glm_tool_prompt(
                messages,
                tools=tool_ctx["tools"],
                tool_choice=tool_ctx.get("tool_choice"),
            )
        return _inject_tool_calling_prompt(
            messages,
            tools=tool_ctx["tools"],
            tool_choice=tool_ctx.get("tool_choice"),
        )

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


def _infer_tool_call_from_user(
    messages: list[ChatMessage],
    tools: list[ToolDefinition],
    tool_choice: Optional[Union[str, dict]],
) -> Optional[list[ToolCall]]:
    # Honor explicit "none"
    if tool_choice == "none":
        return None

    # Find last user message
    last_user = None
    for m in reversed(messages):
        if (m.role or "").lower() == "user":
            last_user = m.content or ""
            break
    if not last_user:
        return None
    # Normalize escaped quotes if present
    last_user_norm = last_user.replace('\\"', '"')

    forced_name = None
    if isinstance(tool_choice, dict):
        try:
            if tool_choice.get("type") == "function":
                forced_name = tool_choice.get("function", {}).get("name")
        except Exception:
            forced_name = None

    tool_names = [t.function.name for t in tools if t and t.function and t.function.name]
    name = forced_name
    if not name and len(tool_names) == 1:
        name = tool_names[0]
    if not name:
        # Heuristic tool routing by keyword score (name + description + schema hints).
        text = last_user_norm.lower()
        best_name = None
        best_score = 0.0
        for t in tools:
            if not (t and t.function and t.function.name):
                continue
            tn = t.function.name
            score = 0.0
            tn_low = tn.lower()
            desc = (t.function.description or "").lower()

            # Direct name hit
            if tn_low in text:
                score += 10.0

            # Generic intent keywords
            if any(k in text for k in ("search", "搜索", "查一下", "查询", "新闻", "news")):
                if any(k in tn_low for k in ("search", "find", "query")) or any(
                    k in desc for k in ("search", "query", "新闻", "news")
                ):
                    score += 4.0
            if any(k in text for k in ("weather", "天气", "温度", "下雨")):
                if any(k in tn_low for k in ("weather", "forecast")) or any(
                    k in desc for k in ("weather", "forecast", "天气")
                ):
                    score += 4.0
            if any(k in text for k in ("time", "几点", "时间", "时区", "timezone")):
                if any(k in tn_low for k in ("time", "clock")) or any(
                    k in desc for k in ("time", "timezone", "时区", "时间")
                ):
                    score += 4.0

            # Schema hints
            try:
                params = t.function.parameters or {}
                props = (params.get("properties") or {}) if isinstance(params, dict) else {}
                prop_keys = " ".join(str(k).lower() for k in props.keys())
                if any(k in text for k in ("city", "城市", "北京", "上海")) and any(
                    k in prop_keys for k in ("city", "location")
                ):
                    score += 1.5
                if any(k in text for k in ("timezone", "时区", "tokyo", "asia/")) and "timezone" in prop_keys:
                    score += 1.5
                if any(k in text for k in ("新闻", "news", "搜索", "search")) and any(
                    k in prop_keys for k in ("query", "q", "keyword")
                ):
                    score += 1.5
            except Exception:
                pass

            if score > best_score:
                best_name = tn
                best_score = score

        # Avoid random tool calls on weak confidence.
        if best_score >= 2.0:
            name = best_name
    if not name:
        return None

    def _extract_keyvals(text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for k, v in re.findall(r"(\w+)\s*=\s*\"([^\"]+)\"", text):
            out[k] = v
        for k, v in re.findall(r"(\w+)\s*=\s*([^\s,;]+)", text):
            if k not in out:
                out[k] = v
        # Also accept JSON-like "key":"value"
        for k, v in re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*"([^"]+)"', text):
            if k not in out:
                out[k] = v
        return out

    def _extract_timezone(text: str) -> Optional[str]:
        m = re.search(r"\b([A-Za-z_]+/[A-Za-z_]+(?:/[A-Za-z_]+)?)\b", text)
        return m.group(1) if m else None

    def _extract_location_like(text: str) -> Optional[str]:
        # English: "... in Tokyo" / "for Berlin"
        m = re.search(
            r"\b(?:in|at|for|around)\s+([A-Za-z][A-Za-z ._-]{1,40})",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).strip(" .,!?")

        # Chinese: "...北京天气..." / "...上海..."
        m = re.search(r"([\u4e00-\u9fff]{2,8}(?:市|省|区|县)?)", text)
        if m:
            return m.group(1)
        return None

    args: dict[str, str] = _extract_keyvals(last_user_norm)

    # Fill required fields heuristically when model/tool parser leaves empty args.
    target_tool = next((t for t in tools if t and t.function and t.function.name == name), None)
    required: list[str] = []
    if target_tool is not None:
        try:
            params = target_tool.function.parameters or {}
            if isinstance(params, dict):
                required = [str(x) for x in (params.get("required") or [])]
        except Exception:
            required = []

    for req_key in required:
        if req_key in args and str(args[req_key]).strip():
            continue
        lk = req_key.lower()
        # First try explicit "key=..." style with the required key name.
        m = re.search(rf"{re.escape(req_key)}\s*=\s*\"([^\"]+)\"", last_user_norm, flags=re.IGNORECASE)
        if not m:
            m = re.search(rf"{re.escape(req_key)}\s*=\s*([^\s,;]+)", last_user_norm, flags=re.IGNORECASE)
        if m:
            args[req_key] = m.group(1).strip()
            continue

        if lk in ("timezone", "tz"):
            tz = _extract_timezone(last_user_norm)
            if tz:
                args[req_key] = tz
                continue

        if lk in ("city", "location", "place", "region"):
            loc = _extract_location_like(last_user_norm)
            if loc:
                args[req_key] = loc
                continue
        if lk in ("query", "q", "keyword", "keywords", "topic"):
            # keep user intent as search query (strip obvious command prefixes)
            q = last_user_norm
            q = re.sub(r"^\s*(请|帮我|麻烦你)?\s*(搜索|查询|查一下|查|search)\s*", "", q, flags=re.IGNORECASE)
            q = re.sub(r"^\s*(一下|一下子)\s*", "", q)
            q = q.strip("。.!?？")
            if q:
                args[req_key] = q

    return [
        ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            type="function",
            function=ToolCallFunction(
                name=name,
                arguments=json.dumps(args, ensure_ascii=False),
            ),
        )
    ]


def _infer_tool_call_from_text(
    text: str,
    tools: list[ToolDefinition],
    tool_choice: Optional[Union[str, dict]],
) -> Optional[list[ToolCall]]:
    if not text:
        return None
    return _infer_tool_call_from_user(
        [ChatMessage(role="user", content=text)], tools, tool_choice
    )


def _is_empty_tool_args(args_str: Optional[str]) -> bool:
    s = (args_str or "").strip().lower()
    return s in ("", "{}", "null")


def _last_user_text(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if (m.role or "").lower() == "user" and (m.content or "").strip():
            return m.content or ""
    return ""


def _patch_empty_tool_args(
    tool_calls: list[ToolCall],
    messages: list[ChatMessage],
    tools: list[ToolDefinition],
) -> list[ToolCall]:
    """
    For GLM outputs where tool_calls are emitted with empty arguments ("{}"),
    patch required fields from user text using lightweight extraction.
    """
    if not tool_calls or not messages:
        return tool_calls

    user_text = _last_user_text(messages)
    if not user_text:
        return tool_calls

    schema_by_name: dict[str, dict] = {}
    for t in tools or []:
        try:
            if t and t.function and t.function.name:
                schema_by_name[t.function.name] = t.function.parameters or {}
        except Exception:
            pass

    # Generic candidates from raw text
    kv: dict[str, str] = {}
    for k, v in re.findall(r"(\w+)\s*=\s*\"([^\"]+)\"", user_text):
        kv[k] = v
    for k, v in re.findall(r"(\w+)\s*=\s*([^\s,;]+)", user_text):
        if k not in kv:
            kv[k] = v
    for k, v in re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*"([^"]+)"', user_text):
        if k not in kv:
            kv[k] = v

    m_tz = re.search(r"\b([A-Za-z_]+/[A-Za-z_]+(?:/[A-Za-z_]+)?)\b", user_text)
    tz = m_tz.group(1) if m_tz else None

    m_en_loc = re.search(
        r"\b(?:in|at|for|around)\s+([A-Za-z][A-Za-z ._-]{1,40})",
        user_text,
        flags=re.IGNORECASE,
    )
    en_loc = m_en_loc.group(1).strip(" .,!?") if m_en_loc else None
    m_zh_loc = re.search(r"([\u4e00-\u9fff]{2,8}(?:市|省|区|县)?)", user_text)
    zh_loc = m_zh_loc.group(1) if m_zh_loc else None
    loc = en_loc or zh_loc

    for tc in tool_calls:
        if not _is_empty_tool_args(tc.function.arguments):
            continue

        fn_name = (tc.function.name or "").strip()
        schema = schema_by_name.get(fn_name, {})
        required = []
        if isinstance(schema, dict):
            required = [str(x) for x in (schema.get("required") or [])]

        patched: dict[str, str] = {}
        for req in required:
            if req in kv and str(kv[req]).strip():
                patched[req] = str(kv[req]).strip()
                continue
            lk = req.lower()
            if lk in ("timezone", "tz") and tz:
                patched[req] = tz
                continue
            if lk in ("city", "location", "place", "region") and loc:
                patched[req] = loc
                continue

        # Heuristic fallback by function name when required[] is absent/empty.
        if not patched:
            lfn = fn_name.lower()
            if ("weather" in lfn or "temperature" in lfn) and loc:
                patched["city"] = loc
            elif ("time" in lfn or "clock" in lfn) and tz:
                patched["timezone"] = tz

        if patched:
            tc.function.arguments = json.dumps(patched, ensure_ascii=False)

    return tool_calls


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


def rank0_wait_done(expected_world_size: int, timeout_s: Optional[float] = None) -> None:
    """
    Wait for {"type":"done"} from all workers.
    """
    if timeout_s is None:
        timeout_s = float(os.environ.get("CTRL_DONE_TIMEOUT", "25"))

    done: set[int] = set()
    t0 = time.time()
    while len(done) < (expected_world_size - 1):
        if time.time() - t0 > timeout_s:
            pending = [r for r in range(1, expected_world_size) if r not in done]
            raise TimeoutError(
                f"Timed out waiting worker done acks after {timeout_s:.1f}s; "
                f"pending ranks={pending}"
            )

        with _worker_lock:
            items = list(_worker_socks.items())
        for r, s in items:
            if r in done:
                continue
            s.settimeout(0.2)
            try:
                msg = recv_msg(s)
            except (socket.timeout, TimeoutError):
                # No data yet: worker may still be generating.
                continue
            except (ConnectionResetError, BrokenPipeError):
                raise ConnectionError(
                    f"Worker rank {r} control socket disconnected while waiting for done"
                )
            except OSError as e:
                # Distinguish transient timeout-style OSErrors from real disconnects.
                if "timed out" in str(e).lower():
                    continue
                raise ConnectionError(
                    f"Worker rank {r} control socket error while waiting for done: {e}"
                )
            finally:
                try:
                    s.settimeout(None)
                except Exception:
                    pass

            if msg is None:
                # EOF means peer closed the socket.
                raise ConnectionError(
                    f"Worker rank {r} control socket disconnected while waiting for done"
                )
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
    timeout_s = float(os.environ.get("CTRL_CONNECT_TIMEOUT", "60"))
    delay_s = float(os.environ.get("CTRL_CONNECT_DELAY", "0.2"))
    deadline = time.time() + timeout_s

    # Control-plane may start slightly after workers finish loading
    # (rank0 starts listener post-load). Retry until it comes up.
    s: Optional[socket.socket] = None
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((CTRL_HOST, CTRL_PORT))
            s.settimeout(None)
            break
        except (ConnectionRefusedError, TimeoutError, OSError):
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
            if time.time() >= deadline:
                raise
            time.sleep(delay_s)
            delay_s = min(delay_s * 1.5, 2.0)

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

        # kind: "chat" | "completions"
        if isinstance(item, tuple) and len(item) == 7:
            kind, prompt, max_t, result_target, is_stream, tool_ctx, req_messages = item
        elif isinstance(item, tuple) and len(item) == 6:
            kind, prompt, max_t, result_target, is_stream, tool_ctx = item
            req_messages = None
        else:
            kind, prompt, max_t, result_target, is_stream = item
            tool_ctx = None
            req_messages = None

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
                tool_streaming = (
                    kind == "chat"
                    and tool_ctx
                    and tool_ctx.get("tools")
                    and tool_ctx.get("tool_choice") != "none"
                )
                buffer = ""
                state = "tool" if tool_streaming else "unknown"
                tool_calls: Optional[list[ToolCall]] = None
                parsed_content: Optional[str] = None
                emitted_clean = ""
                tool_calls_emitted = False
                suppress_text_output = False

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

                    token_text = response.text

                    if tool_streaming:
                        buffer += token_text

                        # Try to parse tool call from buffer
                        try:
                            tool_calls, parsed_content = _parse_tool_response(
                                _sanitize_assistant_output(buffer)
                            )
                        except Exception:
                            tool_calls, parsed_content = None, None

                        if tool_calls:
                            # Prefer inferred args when model returns empty args
                            inferred = None
                            if req_messages:
                                inferred = _infer_tool_call_from_user(
                                    req_messages, tool_ctx["tools"], tool_ctx.get("tool_choice")
                                )
                            if not inferred:
                                inferred = _infer_tool_call_from_text(
                                    prompt, tool_ctx["tools"], tool_ctx.get("tool_choice")
                                )
                            if inferred:
                                empty_args = all(
                                    (tc.function.arguments or "").strip()
                                    in ("", "{}", "null")
                                    for tc in tool_calls
                                )
                                if empty_args:
                                    tool_calls = inferred
                            if req_messages:
                                tool_calls = _patch_empty_tool_args(
                                    tool_calls, req_messages, tool_ctx["tools"]
                                )

                            # Emit tool_calls delta once, then keep draining generation
                            # to completion so rank0 and worker collectives stay aligned.
                            if not tool_calls_emitted:
                                delta_calls = []
                                for i, tc in enumerate(tool_calls):
                                    delta_calls.append(
                                        {
                                            "index": i,
                                            "id": tc.id,
                                            "type": tc.type,
                                            "function": {
                                                "name": tc.function.name,
                                                "arguments": tc.function.arguments,
                                            },
                                        }
                                    )
                                chunk = {
                                    "id": req_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": MODEL_ID,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"tool_calls": delta_calls},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                await chunk_queue.put(f"data: {json.dumps(chunk)}\n\n")
                                tool_calls_emitted = True
                            continue

                        # If model produced content JSON, keep it as fallback
                        if parsed_content is not None:
                            state = "content"
                        continue

                    # Default streaming (no tools or tool_choice=none)
                    buffer += token_text
                    clean_text = _sanitize_assistant_output(buffer)
                    delta_text = ""
                    if clean_text.startswith(emitted_clean):
                        delta_text = clean_text[len(emitted_clean) :]
                    else:
                        # Sanitization removed/rewrote earlier text; emit the new tail only.
                        i = 0
                        max_i = min(len(clean_text), len(emitted_clean))
                        while i < max_i and clean_text[i] == emitted_clean[i]:
                            i += 1
                        delta_text = clean_text[i:]
                    emitted_clean = clean_text

                    if delta_text == "":
                        if _contains_model_stop_marker(buffer):
                            suppress_text_output = True
                        continue

                    if suppress_text_output:
                        continue

                    if kind == "chat":
                        chunk = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta_text},
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
                                    "text": delta_text,
                                    "finish_reason": None,
                                    "logprobs": None,
                                }
                            ],
                        }
                    await chunk_queue.put(f"data: {json.dumps(chunk)}\n\n")
                    if _contains_model_stop_marker(buffer):
                        suppress_text_output = True

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
                if tool_streaming and not _aborted:
                    if not tool_calls:
                        # Final fallback: infer tool call from prompt
                        inferred = None
                        if req_messages:
                            inferred = _infer_tool_call_from_user(
                                req_messages, tool_ctx["tools"], tool_ctx.get("tool_choice")
                            )
                        if not inferred:
                            inferred = _infer_tool_call_from_text(
                                prompt, tool_ctx["tools"], tool_ctx.get("tool_choice")
                            )
                        if inferred:
                            tool_calls = inferred
                            delta_calls = []
                            for i, tc in enumerate(tool_calls):
                                delta_calls.append(
                                    {
                                        "index": i,
                                        "id": tc.id,
                                        "type": tc.type,
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    }
                                )
                            chunk = {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": MODEL_ID,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"tool_calls": delta_calls},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            await chunk_queue.put(f"data: {json.dumps(chunk)}\n\n")
                        elif parsed_content:
                            # Fallback to content if no tool call could be inferred
                            chunk = {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": MODEL_ID,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": parsed_content},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            await chunk_queue.put(f"data: {json.dumps(chunk)}\n\n")
                    if tool_calls:
                        if req_messages:
                            tool_calls = _patch_empty_tool_args(
                                tool_calls, req_messages, tool_ctx["tools"]
                            )
                        _finish = "tool_calls"
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
                completion = _sanitize_assistant_output(completion)

                # Tool calling MVP: parse a single JSON object from the model output when tools are enabled
                parsed_tool_calls: Optional[list[ToolCall]] = None
                parsed_content: Optional[str] = None
                if kind == "chat" and tool_ctx and tool_ctx.get("tools"):
                    try:
                        parsed_tool_calls, parsed_content = _parse_tool_response(completion)
                    except Exception as e:
                        # If parsing fails, fall back to plain text completion
                        log.warning(f"tool parse failed; falling back to text: {e}")
                        parsed_tool_calls, parsed_content = None, completion

                    if parsed_content is not None:
                        parsed_content = _sanitize_assistant_output(parsed_content)

                    # Enforce tool_choice="none"
                    if tool_ctx.get("tool_choice") == "none":
                        parsed_tool_calls = None

                    # Heuristic fallback: infer tool call from user message if model didn't emit one
                    inferred = None
                    msg_list = req_messages or (tool_ctx.get("messages") if tool_ctx else []) or []
                    if parsed_tool_calls is None:
                        inferred = _infer_tool_call_from_user(
                            msg_list, tool_ctx["tools"], tool_ctx.get("tool_choice")
                        )
                        if inferred:
                            parsed_tool_calls = inferred
                            parsed_content = None
                    else:
                        inferred = _infer_tool_call_from_user(
                            msg_list, tool_ctx["tools"], tool_ctx.get("tool_choice")
                        )
                        if inferred:
                            # If model returned empty args, prefer inferred args
                            empty_args = all(
                                (tc.function.arguments or "").strip() in ("", "{}", "null")
                                for tc in parsed_tool_calls
                            )
                            if empty_args:
                                parsed_tool_calls = inferred
                                parsed_content = None
                        elif parsed_tool_calls:
                            # Fallback: try parsing from prompt text
                            empty_args = all(
                                (tc.function.arguments or "").strip() in ("", "{}", "null")
                                for tc in parsed_tool_calls
                            )
                            if empty_args:
                                inferred_from_prompt = _infer_tool_call_from_text(
                                    prompt, tool_ctx["tools"], tool_ctx.get("tool_choice")
                                )
                                if inferred_from_prompt:
                                    parsed_tool_calls = inferred_from_prompt
                                    parsed_content = None

                    if parsed_tool_calls is not None and req_messages:
                        parsed_tool_calls = _patch_empty_tool_args(
                            parsed_tool_calls, msg_list, tool_ctx["tools"]
                        )

                    # If we got a normal content answer, replace completion with it.
                    if parsed_content is not None:
                        completion = parsed_content
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
                    # If tools were enabled and the model requested tool calls, emit finish_reason="tool_calls"
                    if parsed_tool_calls:
                        resp = _build_tool_calls_response(
                            model_id=MODEL_ID,
                            tool_calls=parsed_tool_calls,
                            prompt_tokens=pt,
                            completion_tokens=ct,
                            timing=timing,
                        )
                    else:
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
            log.exception(
                "Queue worker request failed: kind=%s stream=%s req_id=%s",
                kind,
                is_stream,
                req_id,
            )
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

    tool_ctx = None
    if req.tools:
        # If tool results are already present, discourage additional tool calls
        has_tool_msg = any((m.role or "") == "tool" for m in req.messages)
        tool_choice = req.tool_choice
        if has_tool_msg and (tool_choice is None or tool_choice == "auto"):
            tool_choice = "none"

        tool_ctx = {
            "tools": req.tools,
            "tool_choice": tool_choice,
            "messages": req.messages,
        }

    prompt = _build_chat_prompt(req.messages, tool_ctx=tool_ctx)
    max_t = _safe_max_tokens(req.max_tokens or DEFAULT_MAX_TOKENS)

    if req.stream:
        # Streaming mode: return SSE response
        chunk_queue: asyncio.Queue = asyncio.Queue()
        # Fast-path: streaming tool_calls without model execution
        if tool_ctx and tool_ctx.get("tools") and tool_ctx.get("tool_choice") != "none":
            inferred = _infer_tool_call_from_user(
                req.messages, tool_ctx["tools"], tool_ctx.get("tool_choice")
            )
            if inferred:
                created = int(time.time())
                delta_calls = []
                for i, tc in enumerate(inferred):
                    delta_calls.append(
                        {
                            "index": i,
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    )

                async def _tool_stream():
                    chunk = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": delta_calls},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    final_chunk = {
                        "id": chunk["id"],
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                        ],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    _tool_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
        try:
            _queue.put_nowait(("chat", prompt, max_t, chunk_queue, True, tool_ctx, req.messages))
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
            _queue.put_nowait(("chat", prompt, max_t, fut, False, tool_ctx, req.messages))
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
            _queue.put_nowait(("completions", prompt, max_t, chunk_queue, True, None))
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
            _queue.put_nowait(("completions", prompt, max_t, fut, False, None))
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
