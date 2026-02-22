#!/usr/bin/env python3
"""
dashboard.py — Exo-style visual dashboard for mlx-jaccl-cluster.

Features:
  - Visual Mac Mini node cards with live hardware metrics (via macmon)
  - Animated RDMA / Thunderbolt link between nodes
  - Per-node GPU usage, temperature, power, RAM usage
  - Integrated chat panel with streaming support
  - Live metrics via SSE (Server-Sent Events)
  - Dark theme inspired by exo's dashboard

Mounts onto the existing FastAPI app:
    from dashboard import mount_dashboard
    mount_dashboard(app, ...)
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncGenerator, Callable, Optional

if TYPE_CHECKING:
    from memory_monitor import MemoryMonitor

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from starlette.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Metrics ring-buffer (updated by the queue worker in openai_cluster_server.py)
# ---------------------------------------------------------------------------


@dataclass
class GenerationStats:
    """One completed generation's stats."""

    timestamp: float
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    tokens_per_sec: float
    model_id: str
    kind: str  # "chat" | "completions"


class MetricsStore:
    """
    Thread-safe ring buffer of recent generation stats + running counters.
    Call record_generation() after each completed request.
    """

    def __init__(self, maxlen: int = 200):
        self._lock = asyncio.Lock()
        self._history: deque[GenerationStats] = deque(maxlen=maxlen)
        self._total_requests: int = 0
        self._total_tokens: int = 0
        self._total_prompt_tokens: int = 0
        self._error_count: int = 0
        self._server_start: float = time.time()

    async def record_generation(self, stats: GenerationStats) -> None:
        async with self._lock:
            self._history.append(stats)
            self._total_requests += 1
            self._total_tokens += stats.completion_tokens
            self._total_prompt_tokens += stats.prompt_tokens

    async def record_error(self) -> None:
        async with self._lock:
            self._error_count += 1

    async def snapshot(self) -> dict:
        async with self._lock:
            now = time.time()
            recent = [s for s in self._history if now - s.timestamp <= 60.0]
            if recent:
                avg_tps = sum(s.tokens_per_sec for s in recent) / len(recent)
                peak_tps = max(s.tokens_per_sec for s in recent)
                avg_latency = sum(s.elapsed_s for s in recent) / len(recent)
            else:
                avg_tps = peak_tps = avg_latency = 0.0

            return {
                "uptime_s": round(now - self._server_start),
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "total_prompt_tokens": self._total_prompt_tokens,
                "error_count": self._error_count,
                "recent_count": len(recent),
                "avg_tps_60s": round(avg_tps, 1),
                "peak_tps_60s": round(peak_tps, 1),
                "avg_latency_60s": round(avg_latency, 3),
                "history": [
                    {
                        "t": round(s.timestamp - self._server_start, 1),
                        "tps": round(s.tokens_per_sec, 1),
                        "latency": round(s.elapsed_s, 3),
                        "ctokens": s.completion_tokens,
                        "kind": s.kind,
                    }
                    for s in list(self._history)[-40:]
                ],
            }


# Singleton store
metrics_store = MetricsStore()


# ---------------------------------------------------------------------------
# Hardware metrics poller (macmon on all nodes)
# ---------------------------------------------------------------------------


class HardwarePoller:
    """
    Polls macmon on all cluster nodes (local + remote via SSH).
    Runs in a background daemon thread, stores latest metrics per host.
    """

    # Common Homebrew / Cargo paths where macmon may live
    _MACMON_SEARCH_PATHS = [
        "/opt/homebrew/bin/macmon",
        "/usr/local/bin/macmon",
        os.path.expanduser("~/.cargo/bin/macmon"),
    ]

    def __init__(
        self,
        hostfile: str = "",
        poll_interval: float = 2.0,
    ):
        self._hostfile = hostfile
        self._interval = poll_interval
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        self._hosts: list[dict] = []
        self._local_hostname = socket.gethostname()
        self._started = False

        # Resolve full path to macmon (bare "macmon" may not be in PATH
        # inside SSH sessions spawned by mlx.launch)
        self._macmon_bin = shutil.which("macmon") or ""
        if not self._macmon_bin:
            for p in self._MACMON_SEARCH_PATHS:
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    self._macmon_bin = p
                    break
        if self._macmon_bin:
            print(f"[hw-poller] macmon resolved → {self._macmon_bin}", flush=True)
        else:
            print(
                "[hw-poller] WARNING: macmon not found — hardware metrics will be unavailable",
                flush=True,
            )

        # Resolve the full path for remote nodes too (assume same install path)
        self._macmon_remote = self._macmon_bin or "macmon"

        # Parse hostfile
        if hostfile and os.path.isfile(hostfile):
            try:
                with open(hostfile) as f:
                    self._hosts = json.load(f)
            except Exception:
                pass

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()

    def _poll_loop(self) -> None:
        while True:
            self._poll_all()
            time.sleep(self._interval)

    def _is_local(self, ssh_host: str) -> bool:
        # Compare against local hostname variants (strip .local / .home suffixes)
        local = self._local_hostname.lower()
        target = ssh_host.lower()
        # Direct matches
        if target in (local, "localhost", "127.0.0.1"):
            return True
        # Match with common mDNS suffixes: "mac.home" == "mac", "mac.local" == "mac"
        local_base = local.split(".")[0]
        target_base = target.split(".")[0]
        return local_base == target_base

    def _poll_all(self) -> None:
        if not self._hosts:
            # No hostfile — just poll local
            self._poll_node(self._local_hostname, local=True)
            return

        for h in self._hosts:
            ssh = h.get("ssh", "")
            if not ssh:
                continue
            is_local = self._is_local(ssh)
            self._poll_node(ssh, local=is_local)

    def _poll_node(self, host: str, local: bool = False) -> None:
        try:
            if local:
                if not self._macmon_bin:
                    return
                result = subprocess.run(
                    [self._macmon_bin, "pipe", "-s", "1", "-i", "200"],
                    capture_output=True,
                    text=True,
                    timeout=4,
                )
            else:
                # Use full path on remote node — SSH sessions often lack
                # /opt/homebrew/bin in PATH
                remote_cmd = f"{self._macmon_remote} pipe -s 1 -i 200"
                result = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=3", host, remote_cmd],
                    capture_output=True,
                    text=True,
                    timeout=6,
                )

            if result.returncode != 0 or not result.stdout.strip():
                return

            # macmon may output multiple lines; take the last valid JSON
            lines = result.stdout.strip().split("\n")
            raw = None
            for line in reversed(lines):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        raw = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

            if not raw:
                return

            parsed = self._parse(raw)
            with self._lock:
                self._data[host] = parsed

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        except Exception as exc:
            # Log unexpected errors once to help debug
            import traceback

            traceback.print_exc()

    def _parse(self, data: dict) -> dict:
        gpu_usage_raw = data.get("gpu_usage", [0, 0])
        gpu_freq = gpu_usage_raw[0] if len(gpu_usage_raw) > 0 else 0
        # NOTE: gpu_usage[1] is freq/max_freq ratio (always ~1.0 when model loaded),
        # NOT actual GPU compute utilisation.  Apple Silicon doesn't expose a
        # direct "GPU busy %" — GPU power draw is the best real-time proxy.
        gpu_freq_ratio = gpu_usage_raw[1] if len(gpu_usage_raw) > 1 else 0

        mem = data.get("memory", {})
        ram_total = mem.get("ram_total", 0)
        ram_usage = mem.get("ram_usage", 0)

        temp = data.get("temp", {})

        gpu_power = round(data.get("gpu_power", 0), 1)
        # Derive a 0-100 "activity %" from GPU power.
        # M4 Pro GPU TDP is ~22 W; scale so 20 W → 100 %.
        GPU_POWER_CEIL = 20.0
        gpu_activity_pct = (
            min(100, round((gpu_power / GPU_POWER_CEIL) * 100)) if gpu_power > 0 else 0
        )

        return {
            "gpu_usage_pct": gpu_activity_pct,  # power-based activity
            "gpu_freq_ratio": round(gpu_freq_ratio * 100),  # clock % (for tooltip)
            "gpu_freq_mhz": gpu_freq,
            "gpu_power_w": gpu_power,
            "gpu_temp_c": round(temp.get("gpu_temp_avg", 0)),
            "cpu_temp_c": round(temp.get("cpu_temp_avg", 0)),
            "sys_power_w": round(data.get("sys_power", 0), 1),
            "cpu_power_w": round(data.get("cpu_power", 0), 1),
            "ram_used_gb": round(ram_usage / (1024**3), 1),
            "ram_total_gb": round(ram_total / (1024**3), 1),
            "ram_pct": round(ram_usage / ram_total * 100) if ram_total > 0 else 0,
            "timestamp": data.get("timestamp", ""),
        }

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)


# Singleton hardware poller (initialized in mount_dashboard)
hw_poller: Optional[HardwarePoller] = None


# ---------------------------------------------------------------------------
# HTML template — Exo-style visual dashboard
# ---------------------------------------------------------------------------


def _render_dashboard(
    model_id: str,
    world_size: int,
    rank: int,
    queue_max: int,
    rdma_devices: list[str],
    host: str,
    port: int,
    hostfile: str = "",
    model_config: Optional[dict] = None,
) -> str:
    api_base = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"

    # Parse hostfile for node info
    hosts_data = []
    if hostfile and os.path.isfile(hostfile):
        try:
            with open(hostfile) as f:
                hosts_data = json.load(f)
        except Exception:
            pass

    # Build node info for template
    nodes_json = []
    for i in range(world_size):
        h = hosts_data[i] if i < len(hosts_data) else {}
        ssh = h.get("ssh", f"node-{i}")
        rdma_devs = h.get("rdma", [])
        rdma = next(
            (d for d in rdma_devs if d),
            rdma_devices[i] if i < len(rdma_devices) else "—",
        )
        nodes_json.append(
            {
                "rank": i,
                "ssh": ssh,
                "role": "coordinator" if i == 0 else "worker",
                "rdma": rdma,
            }
        )

    # Model config info
    mcfg = model_config or {}
    model_arch = mcfg.get("model_type", "")
    model_hidden = mcfg.get("hidden_size", "")
    model_layers = mcfg.get("num_hidden_layers", "")
    model_quant = ""
    q = mcfg.get("quantization", {})
    if q:
        bits = q.get("bits", "")
        group = q.get("group_size", "")
        model_quant = f"{bits}-bit" + (f" g{group}" if group else "")

    nodes_js = json.dumps(nodes_json)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>JACCL — Distributed ML Dashboard</title>
  <script src="https://unpkg.com/htmx.org@1.9.12/dist/htmx.min.js"></script>
  <script src="https://unpkg.com/htmx.org@1.9.12/dist/ext/sse.js"></script>
  <style>
    :root {{
      --bg: #0a0a0a;
      --surface: #141414;
      --surface2: #1e1e1e;
      --surface3: #252525;
      --border: #2a2a2a;
      --border2: #333;
      --accent: #f5c542;
      --accent-dim: #b8942f;
      --green: #22c55e;
      --green-dim: #166534;
      --teal: #00d4aa;
      --blue: #5b9bf5;
      --red: #ef4444;
      --orange: #f59e0b;
      --text: #e8e8e8;
      --text2: #b0b0b0;
      --dim: #666;
      --font: "SF Mono", "JetBrains Mono", "Fira Code", "Menlo", monospace;
      --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro", "Helvetica Neue", sans-serif;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      font-size: 13px;
      min-height: 100vh;
      overflow-x: hidden;
    }}

    /* ═══════ TOP BAR ═══════ */
    .topbar {{
      display: flex; align-items: center; gap: 14px;
      padding: 10px 20px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      position: sticky; top: 0; z-index: 100;
    }}
    .topbar .logo {{
      font-family: var(--font);
      font-size: 18px; font-weight: 800;
      color: var(--accent);
      letter-spacing: 1px;
      text-transform: uppercase;
    }}
    .topbar .logo span {{ color: var(--dim); font-weight: 400; font-size: 12px; margin-left: 6px; }}
    .topbar .chip {{
      background: var(--surface2); border: 1px solid var(--border);
      border-radius: 6px; padding: 4px 10px; font-size: 11px; color: var(--text2);
    }}
    .topbar .chip.model {{ color: var(--accent); border-color: var(--accent-dim); }}
    .topbar .spacer {{ flex: 1; }}
    .topbar .live-indicator {{
      display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--dim);
    }}
    .topbar .live-dot {{
      width: 8px; height: 8px; border-radius: 50%; background: var(--green);
      box-shadow: 0 0 8px var(--green);
      animation: pulse 2s ease-in-out infinite;
    }}
    @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}
    .topbar a {{
      color: var(--dim); text-decoration: none; font-size: 11px;
      transition: color 0.2s;
    }}
    .topbar a:hover {{ color: var(--accent); }}

    /* ═══════ LAYOUT ═══════ */
    .layout {{
      display: grid;
      grid-template-columns: 180px 1fr 320px;
      grid-template-rows: 1fr;
      height: calc(100vh - 45px);
    }}

    /* ═══════ LEFT SIDEBAR — CONVERSATIONS ═══════ */
    .sidebar-left {{
      background: var(--surface);
      border-right: 1px solid var(--border);
      display: flex; flex-direction: column;
      padding: 12px;
    }}
    .new-chat-btn {{
      width: 100%; padding: 10px;
      background: var(--surface2); border: 1px solid var(--border);
      border-radius: 8px; color: var(--text);
      font-family: var(--font); font-size: 12px;
      cursor: pointer; transition: all 0.15s;
      display: flex; align-items: center; gap: 6px;
      justify-content: center;
    }}
    .new-chat-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
    .conversations-list {{
      flex: 1; margin-top: 12px; overflow-y: auto;
    }}
    .conv-empty {{
      text-align: center; color: var(--dim); margin-top: 40px; font-size: 12px;
    }}
    .conv-empty .icon {{ font-size: 28px; margin-bottom: 8px; opacity: 0.4; }}
    .sidebar-footer {{
      font-size: 10px; color: var(--dim); text-align: center;
      padding-top: 8px; border-top: 1px solid var(--border);
    }}

    /* ═══════ CENTER — NODES + CHAT ═══════ */
    .center {{
      display: flex; flex-direction: column;
      overflow: hidden;
    }}

    /* ── Node Topology ── */
    .topology {{
      padding: 20px 24px 12px;
      display: flex; flex-direction: column; align-items: center; gap: 0;
      flex-shrink: 0;
      background: var(--bg);
    }}
    .node-card {{
      display: flex; align-items: flex-start; gap: 16px;
      padding: 14px 16px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      min-width: 500px;
      transition: border-color 0.3s;
      position: relative;
    }}
    .node-card:hover {{ border-color: var(--border2); }}
    .node-card.coordinator {{ border-left: 3px solid var(--accent); }}
    .node-card.worker {{ border-left: 3px solid var(--teal); }}

    /* Mac Mini visual */
    .mac-mini-icon {{
      width: 80px; height: 54px;
      background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 85%, #111 100%);
      border-radius: 8px;
      border: 1.5px solid #444;
      position: relative;
      display: flex; align-items: center; justify-content: center;
      flex-shrink: 0;
      overflow: hidden;
    }}
    .mac-mini-icon .screen {{
      width: 60px; height: 34px;
      background: #111;
      border-radius: 3px;
      border: 1px solid #333;
      display: flex; align-items: center; justify-content: center;
      font-size: 9px; color: var(--dim);
      position: relative;
      overflow: hidden;
    }}
    .mac-mini-icon .screen .gpu-bar {{
      position: absolute; bottom: 0; left: 0;
      height: 100%; background: var(--accent);
      opacity: 0.15; transition: width 0.8s ease;
    }}
    .mac-mini-icon .screen .gpu-text {{
      position: relative; z-index: 1;
      font-weight: 700; font-size: 14px;
    }}
    .mac-mini-icon::after {{
      content: '';
      position: absolute; bottom: -1px; left: 50%; transform: translateX(-50%);
      width: 20px; height: 2px; background: #555; border-radius: 1px;
    }}

    .node-info {{
      flex: 1; min-width: 0;
    }}
    .node-hostname {{
      font-size: 14px; font-weight: 700; color: var(--text);
      margin-bottom: 2px;
      display: flex; align-items: center; gap: 8px;
    }}
    .node-hostname .role-badge {{
      font-size: 9px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.5px;
      padding: 2px 6px; border-radius: 4px;
    }}
    .role-badge.coord {{ background: #2a2210; color: var(--accent); border: 1px solid var(--accent-dim); }}
    .role-badge.worker {{ background: #0d2a22; color: var(--teal); border: 1px solid #1a4a35; }}
    .node-subtitle {{ font-size: 11px; color: var(--dim); margin-bottom: 6px; }}

    /* ── Metric tiles grid ── */
    .metric-tiles {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 6px;
      margin-bottom: 6px;
    }}
    .metric-tile {{
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 8px;
      display: flex; flex-direction: column;
      align-items: center;
      gap: 1px;
      transition: border-color 0.3s;
    }}
    .metric-tile:hover {{ border-color: var(--border2); }}
    .metric-tile .mt-label {{
      font-size: 8px;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--dim);
      white-space: nowrap;
    }}
    .metric-tile .mt-value {{
      font-size: 16px;
      font-weight: 800;
      line-height: 1.1;
      transition: color 0.3s;
    }}
    .metric-tile .mt-unit {{
      font-size: 9px;
      color: var(--dim);
    }}
    /* Tile color variants */
    .metric-tile.gpu-pw .mt-value {{ color: var(--accent); }}
    .metric-tile.temp .mt-value  {{ color: var(--green); }}
    .metric-tile.temp.hot .mt-value {{ color: var(--orange); }}
    .metric-tile.sys-pw .mt-value {{ color: var(--blue); }}
    .metric-tile.freq .mt-value  {{ color: var(--teal); }}

    /* ── Memory bar ── */
    .mem-bar-row {{
      display: flex; align-items: center; gap: 8px;
    }}
    .mem-bar-label {{ font-size: 10px; color: var(--dim); min-width: 80px; }}
    .mem-bar-wrap {{
      flex: 1; height: 6px;
      background: var(--surface3); border-radius: 3px;
      overflow: hidden;
    }}
    .mem-bar {{
      height: 100%; border-radius: 3px;
      transition: width 0.8s ease, background 0.3s;
    }}
    .mem-bar.low {{ background: var(--green); }}
    .mem-bar.mid {{ background: var(--orange); }}
    .mem-bar.high {{ background: var(--red); }}
    .mem-bar-pct {{ font-size: 10px; color: var(--text2); min-width: 30px; text-align: right; }}

    /* RDMA Link */
    .rdma-link {{
      display: flex; flex-direction: column; align-items: center;
      padding: 4px 0;
      position: relative;
    }}
    .rdma-line {{
      width: 2px; height: 32px;
      background: repeating-linear-gradient(
        180deg,
        var(--accent) 0px, var(--accent) 4px,
        transparent 4px, transparent 8px
      );
      background-size: 2px 8px;
      animation: rdma-flow 0.6s linear infinite;
      position: relative;
    }}
    @keyframes rdma-flow {{
      0% {{ background-position: 0 0; }}
      100% {{ background-position: 0 8px; }}
    }}
    .rdma-label {{
      display: flex; align-items: center; gap: 6px;
      font-size: 10px; color: var(--accent-dim);
      padding: 2px 10px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      margin: -2px 0;
      z-index: 1;
    }}
    .rdma-label .speed {{ color: var(--accent); font-weight: 700; }}

    /* ── Chat area ── */
    .chat-area {{
      flex: 1; display: flex; flex-direction: column;
      overflow: hidden;
      border-top: 1px solid var(--border);
    }}
    .chat-messages {{
      flex: 1; overflow-y: auto; padding: 16px 24px;
      display: flex; flex-direction: column; gap: 12px;
    }}
    .chat-placeholder {{
      text-align: center; margin: auto; color: var(--dim);
    }}
    .chat-placeholder .big {{ font-size: 32px; margin-bottom: 8px; }}
    .chat-placeholder .title {{ font-size: 14px; color: var(--text2); }}
    .chat-placeholder .sub {{ font-size: 11px; margin-top: 4px; opacity: 0.5; }}

    .msg {{ display: flex; gap: 10px; align-items: flex-start; animation: fadeIn 0.2s ease; }}
    @keyframes fadeIn {{ from {{ opacity:0; transform: translateY(4px); }} to {{ opacity:1; transform: none; }} }}
    .msg-avatar {{
      width: 28px; height: 28px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 13px; flex-shrink: 0; margin-top: 2px;
    }}
    .msg.user .msg-avatar {{ background: var(--surface3); border: 1px solid var(--border); }}
    .msg.assistant .msg-avatar {{ background: #1a1a0a; border: 1px solid var(--accent-dim); }}
    .msg-body {{ flex: 1; }}
    .msg-role {{ font-size: 10px; color: var(--dim); margin-bottom: 3px; text-transform: uppercase; letter-spacing: 0.7px; }}
    .msg-content {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 8px; padding: 10px 14px;
      line-height: 1.6; white-space: pre-wrap; word-break: break-word;
      font-size: 13px;
    }}
    .msg.user .msg-content {{ border-color: var(--border2); background: var(--surface2); }}
    .msg-content.streaming::after {{
      content: "▊"; animation: blink 0.7s step-end infinite; color: var(--accent);
    }}
    @keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0}} }}
    .msg-timing {{
      font-size: 10px; color: var(--dim); margin-top: 4px;
    }}

    /* Chat input */
    .chat-input-area {{
      padding: 12px 24px 16px;
      background: var(--surface);
      border-top: 1px solid var(--border);
    }}
    .model-select-row {{
      display: flex; align-items: center; gap: 8px;
      margin-bottom: 8px;
    }}
    .model-select-row label {{ font-size: 11px; color: var(--dim); text-transform: uppercase; letter-spacing: 0.5px; }}
    .model-select-row .model-name {{
      background: var(--surface2); border: 1px solid var(--border);
      border-radius: 6px; padding: 4px 10px; font-size: 12px; color: var(--accent);
      font-family: var(--font);
    }}
    .chat-form {{ display: flex; gap: 10px; align-items: flex-end; }}
    .chat-form textarea {{
      flex: 1; background: var(--surface2); border: 1px solid var(--border);
      border-radius: 8px; color: var(--text); font-family: var(--font);
      font-size: 13px; padding: 10px 14px; resize: none;
      outline: none; line-height: 1.5;
      transition: border-color 0.2s;
      min-height: 44px; max-height: 120px;
    }}
    .chat-form textarea:focus {{ border-color: var(--accent); }}
    .chat-form textarea::placeholder {{ color: var(--dim); }}
    .send-btn {{
      background: var(--accent); color: #000; border: none;
      border-radius: 8px; padding: 10px 18px; cursor: pointer;
      font-family: var(--font); font-size: 13px; font-weight: 700;
      transition: all 0.15s; height: 44px; white-space: nowrap;
      text-transform: uppercase; letter-spacing: 0.5px;
    }}
    .send-btn:hover {{ background: #ffd666; }}
    .send-btn:active {{ transform: scale(0.97); }}
    .send-btn:disabled {{ background: var(--dim); cursor: not-allowed; color: #999; }}
    .chat-meta {{
      font-size: 10px; color: var(--dim); margin-top: 6px;
      display: flex; gap: 14px; align-items: center;
    }}
    .chat-meta label {{ display: flex; align-items: center; gap: 4px; cursor: pointer; }}
    .chat-meta input[type="checkbox"] {{ accent-color: var(--accent); }}
    .chat-meta input[type="number"] {{
      width: 52px; background: var(--surface2); border: 1px solid var(--border);
      color: var(--text); border-radius: 4px; padding: 2px 5px;
      font-family: var(--font); font-size: 11px; outline: none;
    }}

    /* ═══════ RIGHT SIDEBAR ═══════ */
    .sidebar-right {{
      background: var(--surface);
      border-left: 1px solid var(--border);
      overflow-y: auto;
      padding: 14px;
      display: flex; flex-direction: column; gap: 12px;
    }}

    .panel {{
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
    }}
    .panel-title {{
      font-size: 10px; text-transform: uppercase; letter-spacing: 1px;
      color: var(--dim); margin-bottom: 12px;
      display: flex; align-items: center; gap: 6px;
    }}
    .panel-title .icon {{ font-size: 13px; }}

    /* Stats grid */
    .stats-2x2 {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
    }}
    .stat-card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 8px; padding: 10px;
    }}
    .stat-card .label {{ font-size: 9px; color: var(--dim); text-transform: uppercase; letter-spacing: 0.7px; }}
    .stat-card .value {{ font-size: 20px; font-weight: 700; margin-top: 2px; line-height: 1; }}
    .stat-card .unit {{ font-size: 10px; color: var(--dim); }}
    .stat-card.gold .value {{ color: var(--accent); }}
    .stat-card.green .value {{ color: var(--green); }}
    .stat-card.teal .value {{ color: var(--teal); }}
    .stat-card.orange .value {{ color: var(--orange); }}

    /* Queue bar */
    .queue-bar-outer {{
      margin-top: 10px; background: var(--surface);
      border-radius: 4px; overflow: hidden; height: 6px;
      border: 1px solid var(--border);
    }}
    .queue-bar-fill {{
      height: 100%; border-radius: 4px;
      transition: width 0.4s ease, background 0.3s;
      background: var(--accent);
    }}

    /* Sparkline */
    .sparkline-container {{ margin-top: 10px; height: 48px; position: relative; }}
    .sparkline-container svg {{ width: 100%; height: 100%; }}
    .sparkline-label {{ font-size: 9px; color: var(--dim); position: absolute; top: 0; right: 0; }}

    /* Model info */
    .model-info-row {{
      display: flex; justify-content: space-between;
      padding: 4px 0; font-size: 11px;
    }}
    .model-info-row .k {{ color: var(--dim); }}
    .model-info-row .v {{ color: var(--text2); font-weight: 600; }}

    /* RDMA info panel */
    .rdma-panel {{
      background: linear-gradient(135deg, #1a1a0a 0%, #0f1a14 100%);
      border: 1px solid #2a2a1a;
      border-radius: 10px; padding: 12px 14px;
      display: flex; align-items: center; gap: 12px;
    }}
    .rdma-panel .rdma-icon {{ font-size: 22px; }}
    .rdma-panel .rdma-text {{ flex: 1; }}
    .rdma-panel .rdma-title {{ color: var(--accent); font-weight: 700; font-size: 12px; }}
    .rdma-panel .rdma-sub {{ color: var(--dim); font-size: 10px; margin-top: 1px; }}
    .rdma-panel .rdma-bw {{ text-align: right; }}
    .rdma-panel .rdma-bw .big {{ font-size: 18px; font-weight: 700; color: var(--accent); }}
    .rdma-panel .rdma-bw .small {{ font-size: 9px; color: var(--dim); }}

    /* Uptime row */
    .info-row {{
      display: flex; justify-content: space-between;
      font-size: 11px; padding: 3px 0;
    }}
    .info-row .k {{ color: var(--dim); }}
    .info-row .v {{ color: var(--text2); }}

    /* Endpoint list */
    .endpoint-list {{ list-style: none; }}
    .endpoint-list li {{
      padding: 4px 0; font-size: 11px;
      display: flex; justify-content: space-between;
    }}
    .endpoint-list li .method {{ color: var(--green); font-weight: 700; font-size: 10px; min-width: 36px; }}
    .endpoint-list li .path {{ color: var(--text2); }}

    /* Error toast */
    #error-toast {{
      position: fixed; bottom: 24px; right: 24px;
      background: var(--red); color: #fff;
      border-radius: 8px; padding: 10px 18px;
      font-size: 13px; display: none; z-index: 999;
      box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}

    /* Responsive */
    @media (max-width: 1100px) {{
      .layout {{ grid-template-columns: 1fr 280px; }}
      .sidebar-left {{ display: none; }}
    }}
    @media (max-width: 760px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar-right {{ display: none; }}
    }}
  </style>
</head>
<body>

<!-- ═══════ TOP BAR ═══════ -->
<div class="topbar">
  <span class="logo">JACCL</span>
  <span class="chip model">{model_id}</span>
  <span class="chip">{world_size} nodes · TP</span>
  <span class="spacer"></span>
  <div class="live-indicator">
    <span class="live-dot" id="live-dot"></span>
    <span id="uptime-display">starting…</span>
  </div>
  <a href="/docs" target="_blank">API Docs ↗</a>
  <a href="{api_base}/v1" target="_blank">{api_base}/v1</a>
</div>

<!-- ═══════ MAIN LAYOUT ═══════ -->
<div class="layout">

  <!-- ═══════ LEFT SIDEBAR ═══════ -->
  <div class="sidebar-left">
    <button class="new-chat-btn" onclick="newChat()">＋ New Chat</button>
    <div class="conversations-list" id="conv-list">
      <div class="conv-empty">
        <div class="icon">💬</div>
        No conversations<br/>Start a new chat to begin
      </div>
    </div>
    <div class="sidebar-footer">{world_size} nodes · RDMA over Thunderbolt</div>
  </div>

  <!-- ═══════ CENTER ═══════ -->
  <div class="center">

    <!-- Node Topology -->
    <div class="topology" id="topology">
      <!-- Built dynamically by JS based on nodes data -->
    </div>

    <!-- Chat -->
    <div class="chat-area">
      <div class="chat-messages" id="chat-messages">
        <div class="chat-placeholder">
          <div class="big">⚡</div>
          <div class="title">JACCL — {world_size} nodes · {model_id}</div>
          <div class="sub">RDMA over Thunderbolt · Tensor Parallelism · {model_quant if model_quant else "MLX"}</div>
        </div>
      </div>
      <div class="chat-input-area">
        <div class="model-select-row">
          <label>Model:</label>
          <span class="model-name">{model_id}</span>
        </div>
        <div class="chat-form">
          <textarea id="chat-input" rows="1"
            placeholder="► Ask anything"
            onkeydown="handleKey(event)"></textarea>
          <button class="send-btn" id="send-btn" onclick="sendMessage()">Send</button>
        </div>
        <div class="chat-meta">
          <span style="opacity:0.5;">Shift+Enter — newline</span>
          <label>
            <input type="checkbox" id="stream-toggle" checked>
            Stream
          </label>
          <label>
            Max tokens
            <input type="number" id="max-tokens-input" value="512" min="64" max="4096" step="64">
          </label>
        </div>
      </div>
    </div>

  </div>

  <!-- ═══════ RIGHT SIDEBAR ═══════ -->
  <div class="sidebar-right"
       hx-ext="sse"
       sse-connect="/metrics/stream"
       sse-swap="message"
       hx-swap="none"
       id="metrics-sse-anchor">

    <!-- Live Metrics -->
    <div class="panel">
      <div class="panel-title"><span class="icon">📊</span> Live Metrics
        <span style="margin-left:auto;font-size:9px;color:var(--dim);" id="metrics-age"></span>
      </div>
      <div class="stats-2x2">
        <div class="stat-card green">
          <div class="label">Avg tok/s</div>
          <div class="value" id="m-avg-tps">—</div>
          <div class="unit">last 60s</div>
        </div>
        <div class="stat-card teal">
          <div class="label">Peak tok/s</div>
          <div class="value" id="m-peak-tps">—</div>
          <div class="unit">last 60s</div>
        </div>
        <div class="stat-card gold">
          <div class="label">Requests</div>
          <div class="value" id="m-total-req">0</div>
          <div class="unit">total</div>
        </div>
        <div class="stat-card orange">
          <div class="label">Latency</div>
          <div class="value" id="m-latency">—</div>
          <div class="unit">avg sec</div>
        </div>
      </div>

      <div style="margin-top:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span style="font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:0.7px;">Queue</span>
          <span style="font-size:11px;" id="q-label">0 / {queue_max}</span>
        </div>
        <div class="queue-bar-outer">
          <div class="queue-bar-fill" id="q-bar" style="width:0%"></div>
        </div>
      </div>

      <div class="sparkline-container">
        <span class="sparkline-label">tok/s</span>
        <svg id="spark-svg" viewBox="0 0 280 44" preserveAspectRatio="none">
          <polyline id="spark-line" fill="none" stroke="var(--accent)" stroke-width="1.5"
            stroke-linejoin="round" stroke-linecap="round" points=""/>
          <polyline id="spark-fill" fill="var(--accent)" fill-opacity="0.06" stroke="none" points=""/>
        </svg>
      </div>

      <div class="info-row" style="margin-top:8px;">
        <span class="k">Uptime</span><span class="v" id="m-uptime">—</span>
      </div>
      <div class="info-row">
        <span class="k">Total tokens</span><span class="v" id="m-total-tok">0</span>
      </div>
      <div class="info-row">
        <span class="k">Errors</span><span class="v" id="m-errors">0</span>
      </div>
    </div>

    <!-- Model Info -->
    <div class="panel">
      <div class="panel-title"><span class="icon">🧠</span> Model</div>
      <div class="model-info-row"><span class="k">Name</span><span class="v">{model_id}</span></div>
      <div class="model-info-row"><span class="k">Architecture</span><span class="v">{model_arch or "—"}</span></div>
      <div class="model-info-row"><span class="k">Hidden / Layers</span><span class="v">{model_hidden or "—"} / {model_layers or "—"}</span></div>
      <div class="model-info-row"><span class="k">Quantization</span><span class="v">{model_quant or "—"}</span></div>
      <div class="model-info-row"><span class="k">Parallelism</span><span class="v">Tensor × {world_size}</span></div>
      <div class="model-info-row"><span class="k">Backend</span><span class="v">JACCL · MLX RDMA</span></div>
    </div>

    <!-- RDMA -->
    <div class="rdma-panel">
      <span class="rdma-icon">🔗</span>
      <div class="rdma-text">
        <div class="rdma-title">RDMA / Thunderbolt</div>
        <div class="rdma-sub">JACCL · all_sum collective</div>
      </div>
      <div class="rdma-bw">
        <div class="big">~8 GB/s</div>
        <div class="small">peak bandwidth</div>
      </div>
    </div>

    <!-- API Endpoints -->
    <div class="panel">
      <div class="panel-title"><span class="icon">🔌</span> Endpoints</div>
      <ul class="endpoint-list">
        <li><span class="method">POST</span><span class="path">/v1/chat/completions</span></li>
        <li><span class="method">POST</span><span class="path">/v1/completions</span></li>
        <li><span class="method">GET</span><span class="path">/v1/models</span></li>
        <li><span class="method">GET</span><span class="path">/health</span></li>
        <li><span class="method">GET</span><span class="path">/metrics/snapshot</span></li>
        <li><span class="method">GET</span><span class="path">/queue</span></li>
      </ul>
    </div>

  </div><!-- /sidebar-right -->

</div><!-- /layout -->

<div id="error-toast"></div>

<script>
// ═══════ NODES DATA ═══════
const NODES = {nodes_js};
const QUEUE_MAX = {queue_max};

// ═══════ BUILD TOPOLOGY ═══════
function buildTopology() {{
  const container = document.getElementById('topology');
  container.innerHTML = '';

  NODES.forEach((node, idx) => {{
    // Node card
    const card = document.createElement('div');
    card.className = 'node-card ' + node.role;
    card.id = 'node-' + node.rank;

    const roleClass = node.role === 'coordinator' ? 'coord' : 'worker';
    card.innerHTML = `
      <div class="mac-mini-icon">
        <div class="screen">
          <div class="gpu-bar" id="gpu-bar-${{node.rank}}" style="width:0%"></div>
          <span class="gpu-text" id="gpu-text-${{node.rank}}">—W</span>
        </div>
      </div>
      <div class="node-info">
        <div class="node-hostname">
          ${{node.ssh}}
          <span class="role-badge ${{roleClass}}">${{node.role}}</span>
        </div>
        <div class="node-subtitle">rank ${{node.rank}} · rdma: ${{node.rdma}}</div>
        <div class="metric-tiles">
          <div class="metric-tile gpu-pw" id="tile-gpupw-${{node.rank}}">
            <span class="mt-label">GPU Power</span>
            <span class="mt-value" id="val-gpupw-${{node.rank}}">—</span>
            <span class="mt-unit">watts</span>
          </div>
          <div class="metric-tile temp" id="tile-temp-${{node.rank}}">
            <span class="mt-label">GPU Temp</span>
            <span class="mt-value" id="val-temp-${{node.rank}}">—</span>
            <span class="mt-unit">°C</span>
          </div>
          <div class="metric-tile sys-pw" id="tile-syspw-${{node.rank}}">
            <span class="mt-label">System</span>
            <span class="mt-value" id="val-syspw-${{node.rank}}">—</span>
            <span class="mt-unit">watts</span>
          </div>
          <div class="metric-tile freq" id="tile-freq-${{node.rank}}">
            <span class="mt-label">GPU Clock</span>
            <span class="mt-value" id="val-freq-${{node.rank}}">—</span>
            <span class="mt-unit">MHz</span>
          </div>
        </div>
        <div class="mem-bar-row">
          <span class="mem-bar-label" id="mem-label-${{node.rank}}">—/— GB</span>
          <div class="mem-bar-wrap">
            <div class="mem-bar low" id="mem-bar-${{node.rank}}" style="width:0%"></div>
          </div>
          <span class="mem-bar-pct" id="mem-pct-${{node.rank}}">—</span>
        </div>
      </div>
    `;
    container.appendChild(card);

    // RDMA link between nodes (not after the last)
    if (idx < NODES.length - 1) {{
      const link = document.createElement('div');
      link.className = 'rdma-link';
      link.innerHTML = `
        <div class="rdma-line"></div>
        <div class="rdma-label">
          <span>⚡</span> RDMA Thunderbolt <span class="speed">~8 GB/s</span>
        </div>
        <div class="rdma-line"></div>
      `;
      container.appendChild(link);
    }}
  }});
}}
buildTopology();

// ═══════ UPDATE NODE HARDWARE METRICS ═══════
function updateNodeHW(nodeSSH, rank, hw) {{
  if (!hw) return;

  // ── Mac Mini screen: GPU power bar + watt label ──
  const gpuBar = document.getElementById('gpu-bar-' + rank);
  const gpuText = document.getElementById('gpu-text-' + rank);
  if (gpuBar && gpuText) {{
    const actPct = hw.gpu_usage_pct || 0;
    const gpuW   = hw.gpu_power_w   ?? 0;
    gpuBar.style.width = actPct + '%';
    gpuBar.style.background = actPct > 60 ? 'var(--orange)' : 'var(--accent)';
    gpuText.textContent = gpuW + 'W';
  }}

  // ── Tile: GPU Power ──
  const gpuPwVal = document.getElementById('val-gpupw-' + rank);
  if (gpuPwVal && hw.gpu_power_w !== undefined) {{
    gpuPwVal.textContent = hw.gpu_power_w;
  }}

  // ── Tile: GPU Temp ──
  const tempVal = document.getElementById('val-temp-' + rank);
  const tempTile = document.getElementById('tile-temp-' + rank);
  if (tempVal && hw.gpu_temp_c !== undefined) {{
    tempVal.textContent = hw.gpu_temp_c;
    if (tempTile) {{
      tempTile.classList.toggle('hot', hw.gpu_temp_c > 70);
    }}
  }}

  // ── Tile: System Power ──
  const sysPwVal = document.getElementById('val-syspw-' + rank);
  if (sysPwVal && hw.sys_power_w !== undefined) {{
    sysPwVal.textContent = hw.sys_power_w;
  }}

  // ── Tile: GPU Clock Frequency ──
  const freqVal = document.getElementById('val-freq-' + rank);
  if (freqVal && hw.gpu_freq_mhz !== undefined) {{
    freqVal.textContent = hw.gpu_freq_mhz;
  }}

  // ── Memory bar ──
  const memLabel = document.getElementById('mem-label-' + rank);
  const memBar = document.getElementById('mem-bar-' + rank);
  const memPct = document.getElementById('mem-pct-' + rank);
  if (memLabel && memBar && memPct && hw.ram_used_gb !== undefined) {{
    const used = hw.ram_used_gb;
    const total = hw.ram_total_gb;
    const pct = hw.ram_pct || 0;
    memLabel.textContent = used + '/' + total + ' GB';
    memBar.style.width = pct + '%';
    memBar.className = 'mem-bar ' + (pct > 80 ? 'high' : pct > 50 ? 'mid' : 'low');
    memPct.textContent = pct + '%';
  }}
}}

// ═══════ CHAT STATE ═══════
let messages = [];
let generating = false;
let currentReader = null;

function handleKey(e) {{
  if (e.key === 'Enter' && !e.shiftKey) {{
    e.preventDefault();
    sendMessage();
  }}
  const ta = document.getElementById('chat-input');
  setTimeout(() => {{
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 120) + 'px';
  }}, 0);
}}

function escapeHtml(text) {{
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}}

function showError(msg) {{
  const t = document.getElementById('error-toast');
  t.textContent = msg;
  t.style.display = 'block';
  setTimeout(() => {{ t.style.display = 'none'; }}, 4000);
}}

function setGenerating(val) {{
  generating = val;
  const btn = document.getElementById('send-btn');
  btn.disabled = val;
  btn.textContent = val ? '…' : 'Send';
}}

function scrollToBottom() {{
  const el = document.getElementById('chat-messages');
  el.scrollTop = el.scrollHeight;
}}

function clearPlaceholder() {{
  const ph = document.querySelector('.chat-placeholder');
  if (ph) ph.remove();
}}

function newChat() {{
  messages = [];
  document.getElementById('chat-messages').innerHTML = `
    <div class="chat-placeholder">
      <div class="big">⚡</div>
      <div class="title">JACCL — {world_size} nodes · {model_id}</div>
      <div class="sub">RDMA over Thunderbolt · Tensor Parallelism</div>
    </div>`;
}}

function appendMessage(role, content, streaming) {{
  clearPlaceholder();
  const id = 'msg-' + Date.now() + '-' + Math.random().toString(36).slice(2);
  const avatar = role === 'user' ? '👤' : '⚡';
  const streamClass = streaming ? ' streaming' : '';
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  el.id = id;
  el.innerHTML = `
    <div class="msg-avatar">${{avatar}}</div>
    <div class="msg-body">
      <div class="msg-role">${{role}}</div>
      <div class="msg-content${{streamClass}}" id="${{id}}-content">${{escapeHtml(content)}}</div>
      <div class="msg-timing" id="${{id}}-timing"></div>
    </div>`;
  document.getElementById('chat-messages').appendChild(el);
  scrollToBottom();
  return id;
}}

function updateMessage(id, content, done) {{
  const el = document.getElementById(id + '-content');
  if (!el) return;
  el.textContent = content;
  if (done) el.classList.remove('streaming');
  scrollToBottom();
}}

function setTiming(id, text) {{
  const el = document.getElementById(id + '-timing');
  if (el) el.textContent = text;
}}

async function sendMessage() {{
  if (generating) return;
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  input.style.height = 'auto';

  const stream = document.getElementById('stream-toggle').checked;
  const maxTokens = parseInt(document.getElementById('max-tokens-input').value) || 512;

  messages.push({{ role: 'user', content: text }});
  appendMessage('user', text, false);
  setGenerating(true);

  const msgId = appendMessage('assistant', '', true);
  let fullText = '';
  const t0 = performance.now();

  try {{
    if (stream) {{
      const resp = await fetch('/v1/chat/completions', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ messages, max_tokens: maxTokens, stream: true }}),
      }});
      if (!resp.ok) throw new Error(`HTTP ${{resp.status}}: ${{await resp.text()}}`);

      const reader = resp.body.getReader();
      currentReader = reader;
      const decoder = new TextDecoder();
      let buf = '';

      while (true) {{
        const {{ done, value }} = await reader.read();
        if (done) break;
        buf += decoder.decode(value, {{ stream: true }});
        const lines = buf.split('\\n');
        buf = lines.pop();
        for (const line of lines) {{
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (data === '[DONE]') break;
          try {{
            const chunk = JSON.parse(data);
            if (chunk.error) throw new Error(chunk.error);
            const delta = chunk.choices?.[0]?.delta?.content;
            if (delta) {{
              fullText += delta;
              updateMessage(msgId, fullText, false);
            }}
          }} catch (e) {{
            if (!e.message.includes('JSON')) console.warn(e);
          }}
        }}
      }}
      updateMessage(msgId, fullText, true);
      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
      const toks = fullText.split(/\\s+/).length;  // rough estimate
      setTiming(msgId, `${{elapsed}}s`);

    }} else {{
      const resp = await fetch('/v1/chat/completions', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ messages, max_tokens: maxTokens, stream: false }}),
      }});
      if (!resp.ok) throw new Error(`HTTP ${{resp.status}}: ${{await resp.text()}}`);
      const data = await resp.json();
      fullText = data.choices?.[0]?.message?.content || '';
      updateMessage(msgId, fullText, true);
      const timing = data.timing;
      if (timing) {{
        setTiming(msgId, `${{timing.tokens_per_sec}} tok/s · ${{timing.seconds}}s`);
      }}
    }}

    if (fullText) messages.push({{ role: 'assistant', content: fullText }});

  }} catch (e) {{
    updateMessage(msgId, '⚠ ' + e.message, true);
    showError(e.message);
    messages.pop();
  }} finally {{
    setGenerating(false);
    currentReader = null;
  }}
}}

// ═══════ METRICS SSE ═══════
function formatUptime(secs) {{
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (h > 0) return h + 'h ' + m + 'm';
  if (m > 0) return m + 'm ' + s + 's';
  return s + 's';
}}

function renderSparkline(history) {{
  if (!history || history.length < 2) return;
  const vals = history.map(h => h.tps);
  const maxV = Math.max(...vals, 1);
  const W = 280, H = 44;
  const pts = vals.map((v, i) => {{
    const x = (i / (vals.length - 1)) * W;
    const y = H - (v / maxV) * (H - 4) - 2;
    return x.toFixed(1) + ',' + y.toFixed(1);
  }});
  const lineStr = pts.join(' ');
  const fillStr = '0,' + H + ' ' + lineStr + ' ' + W + ',' + H;
  document.getElementById('spark-line').setAttribute('points', lineStr);
  document.getElementById('spark-fill').setAttribute('points', fillStr);
}}

document.body.addEventListener('htmx:sseMessage', function(evt) {{
  try {{
    const m = JSON.parse(evt.detail.data);

    // Update metrics panel
    document.getElementById('m-avg-tps').textContent = m.avg_tps_60s > 0 ? m.avg_tps_60s : '—';
    document.getElementById('m-peak-tps').textContent = m.peak_tps_60s > 0 ? m.peak_tps_60s : '—';
    document.getElementById('m-total-req').textContent = m.total_requests;
    document.getElementById('m-latency').textContent = m.avg_latency_60s > 0 ? m.avg_latency_60s : '—';
    document.getElementById('m-uptime').textContent = formatUptime(m.uptime_s);
    document.getElementById('m-total-tok').textContent =
      m.total_tokens > 999 ? (m.total_tokens / 1000).toFixed(1) + 'k' : m.total_tokens;
    document.getElementById('m-errors').textContent = m.error_count || 0;

    // Uptime in topbar
    document.getElementById('uptime-display').textContent = formatUptime(m.uptime_s);

    // Queue
    const qSize = m.queue_size || 0;
    document.getElementById('q-label').textContent = qSize + ' / ' + QUEUE_MAX;
    const pct = Math.min(100, (qSize / QUEUE_MAX) * 100);
    const bar = document.getElementById('q-bar');
    bar.style.width = pct + '%';
    bar.style.background = pct > 75 ? 'var(--red)' : pct > 40 ? 'var(--orange)' : 'var(--accent)';

    // Sparkline
    if (m.history) renderSparkline(m.history);

    // Timestamp
    document.getElementById('metrics-age').textContent = new Date().toLocaleTimeString();

    // Pulse live dot
    const dot = document.getElementById('live-dot');
    dot.style.opacity = '0.3';
    setTimeout(() => {{ dot.style.opacity = '1'; }}, 200);

    // ── Hardware metrics per node ──
    if (m.hardware) {{
      NODES.forEach(node => {{
        const hw = m.hardware[node.ssh];
        if (hw) updateNodeHW(node.ssh, node.rank, hw);
      }});
    }}

  }} catch(e) {{ /* ignore */ }}
}});

// Fallback poll
async function pollMetrics() {{
  try {{
    const r = await fetch('/metrics/snapshot');
    if (!r.ok) return;
    const m = await r.json();
    document.body.dispatchEvent(new CustomEvent('htmx:sseMessage', {{
      detail: {{ data: JSON.stringify(m) }}
    }}));
  }} catch(e) {{}}
}}
setInterval(pollMetrics, 3000);
pollMetrics();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# SSE event generator
# ---------------------------------------------------------------------------


async def _metrics_event_generator(
    get_queue_info: Callable[[], dict],
    monitor: Optional["MemoryMonitor"] = None,
    interval: float = 2.0,
) -> AsyncGenerator[str, None]:
    """Yields SSE events with merged metrics + queue info + hardware every `interval` seconds.

    When a MemoryMonitor is provided, its macmon data is used for the local
    node (avoiding a duplicate macmon subprocess).  The HardwarePoller still
    handles remote nodes via SSH.
    """
    while True:
        try:
            snap = await metrics_store.snapshot()
            qi = get_queue_info()
            snap.update(qi)

            # Include hardware metrics — prefer monitor for local node
            if hw_poller is not None:
                hw_data = hw_poller.snapshot()

                # If monitor is running, overlay its macmon data for the local node
                # (fresher, and avoids a duplicate macmon subprocess)
                if monitor is not None and monitor.running:
                    local_hw = monitor.hardware_dict()
                    if local_hw is not None:
                        local_hostname = socket.gethostname()
                        # Find the local node key in hw_data and replace it
                        replaced = False
                        for key in list(hw_data.keys()):
                            key_base = key.lower().split(".")[0]
                            local_base = local_hostname.lower().split(".")[0]
                            if key_base == local_base or key in (
                                "localhost",
                                "127.0.0.1",
                                local_hostname,
                            ):
                                hw_data[key] = local_hw
                                replaced = True
                                break
                        if not replaced and local_hostname not in hw_data:
                            hw_data[local_hostname] = local_hw

                snap["hardware"] = hw_data

            # Include live memory monitor summary if available
            if monitor is not None and monitor.running:
                snap["memory_monitor"] = {
                    "source": "macmon" if monitor.using_macmon else "vm_stat",
                    "peak_pressure_1m": round(monitor.peak_pressure_1m, 1),
                    "avg_pressure_1m": round(monitor.avg_pressure_1m, 1),
                    "threshold": monitor.memory_threshold,
                }
                mlx = monitor.latest_mlx
                if mlx is not None:
                    snap["memory_monitor"]["mlx_active_gb"] = round(mlx.active_gb, 3)
                    snap["memory_monitor"]["mlx_pressure_pct"] = round(
                        mlx.pressure_pct, 1
                    )

            yield f"data: {json.dumps(snap)}\n\n"
        except asyncio.CancelledError:
            break
        except Exception:
            pass
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Mount dashboard onto FastAPI app
# ---------------------------------------------------------------------------


def mount_dashboard(
    app: FastAPI,
    *,
    get_state: Callable[[], dict],
    get_queue_info: Callable[[], dict],
    model_id: str,
    world_size: int,
    rank: int,
    queue_max: int,
    rdma_devices: Optional[list[str]] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    hostfile: str = "",
    memory_monitor: Optional["MemoryMonitor"] = None,
) -> None:
    """
    Mount dashboard routes onto an existing FastAPI app.

    Parameters
    ----------
    app            : the FastAPI instance
    get_state      : callable returning current server state dict
    get_queue_info : callable returning {"queue_size": int, "queue_max": int}
    model_id       : model name/id string
    world_size     : total number of distributed ranks
    rank           : rank of this node
    queue_max      : maximum queue depth
    rdma_devices   : list of RDMA device names per rank
    host           : bind host
    port           : HTTP port
    hostfile       : path to hostfile for node info + hw polling
    memory_monitor : optional MemoryMonitor instance — when provided, the
                     dashboard uses its macmon data for the local node instead
                     of spawning its own macmon subprocess (avoids duplicate
                     macmon processes, matches exo's single-source pattern)
    """
    global hw_poller

    if rdma_devices is None:
        rdma_devices = ["rdma_en4"] * world_size

    # Load model config for display
    model_config = None
    model_dir = os.environ.get("MODEL_DIR", "")
    if model_dir:
        cfg_path = os.path.join(model_dir, "config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path) as f:
                    model_config = json.load(f)
            except Exception:
                pass

    # Start hardware poller
    hw_poller = HardwarePoller(hostfile=hostfile, poll_interval=2.5)
    hw_poller.start()

    # Pre-render HTML
    _html = _render_dashboard(
        model_id=model_id,
        world_size=world_size,
        rank=rank,
        queue_max=queue_max,
        rdma_devices=rdma_devices,
        host=host,
        port=port,
        hostfile=hostfile,
        model_config=model_config,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard_root():
        return HTMLResponse(content=_html)

    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard_page():
        return HTMLResponse(content=_html)

    @app.get("/metrics/stream", include_in_schema=False)
    async def metrics_stream(request: Request):
        """SSE endpoint — pushes metrics + hardware JSON every 2.5s."""

        async def event_gen():
            async for event in _metrics_event_generator(
                get_queue_info, monitor=memory_monitor
            ):
                if await request.is_disconnected():
                    break
                yield event

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/metrics/snapshot", include_in_schema=False)
    async def metrics_snapshot():
        """Non-SSE fallback — returns current metrics as JSON."""
        snap = await metrics_store.snapshot()
        qi = get_queue_info()
        snap.update(qi)
        if hw_poller is not None:
            snap["hardware"] = hw_poller.snapshot()
        if memory_monitor is not None and memory_monitor.running:
            snap["memory_monitor"] = {
                "source": "macmon" if memory_monitor.using_macmon else "vm_stat",
                "peak_pressure_1m": round(memory_monitor.peak_pressure_1m, 1),
                "avg_pressure_1m": round(memory_monitor.avg_pressure_1m, 1),
                "threshold": memory_monitor.memory_threshold,
            }
            mlx = memory_monitor.latest_mlx
            if mlx is not None:
                snap["memory_monitor"]["mlx_active_gb"] = round(mlx.active_gb, 3)
                snap["memory_monitor"]["mlx_pressure_pct"] = round(mlx.pressure_pct, 1)
        return snap
