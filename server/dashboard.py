#!/usr/bin/env python3
"""
dashboard.py — Self-contained HTMX + SSE dashboard for mlx-jaccl-cluster.

Mounts onto the existing FastAPI app:
    from dashboard import mount_dashboard
    mount_dashboard(app, get_state)

Where get_state() returns a DashboardState-compatible dict.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Optional

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
            # Recent window: last 60 s
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
                    for s in list(self._history)[-40:]  # last 40 for sparkline
                ],
            }


# Singleton store — import this in openai_cluster_server.py to record stats
metrics_store = MetricsStore()


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------


def _render_dashboard(
    model_id: str,
    world_size: int,
    rank: int,
    queue_max: int,
    rdma_devices: list[str],
    host: str,
    port: int,
) -> str:
    node_rows = ""
    for i, dev in enumerate(rdma_devices):
        role = "coordinator" if i == 0 else "worker"
        role_badge = (
            '<span class="badge badge-coord">coordinator</span>'
            if i == 0
            else '<span class="badge badge-worker">worker</span>'
        )
        dev_label = dev if dev else "<span class='dim'>—</span>"
        node_rows += f"""
        <tr>
          <td><span class="rank-circle">{i}</span></td>
          <td>{role_badge}</td>
          <td><code>{dev_label}</code></td>
          <td><span class="dot-green" title="active"></span> active</td>
        </tr>"""

    api_base = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>mlx-jaccl-cluster — Dashboard</title>
  <script src="https://unpkg.com/htmx.org@1.9.12/dist/htmx.min.js"></script>
  <script src="https://unpkg.com/htmx.org@1.9.12/dist/ext/sse.js"></script>
  <style>
    :root {{
      --bg: #0f1117;
      --surface: #1a1d27;
      --surface2: #22263a;
      --border: #2e3352;
      --accent: #5b6af0;
      --accent2: #00d4aa;
      --warn: #f0a030;
      --danger: #e05555;
      --text: #e2e4f0;
      --dim: #6b7294;
      --green: #22c55e;
      --radius: 10px;
      --font: "SF Mono", "JetBrains Mono", "Fira Code", monospace;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      font-size: 13px;
      min-height: 100vh;
    }}

    /* ---- Layout ---- */
    .topbar {{
      display: flex; align-items: center; gap: 12px;
      padding: 12px 24px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      position: sticky; top: 0; z-index: 100;
    }}
    .topbar .logo {{ font-size: 16px; font-weight: 700; color: var(--accent); letter-spacing: 0.5px; }}
    .topbar .model-badge {{
      background: var(--surface2); border: 1px solid var(--border);
      border-radius: 6px; padding: 3px 10px; font-size: 12px; color: var(--accent2);
    }}
    .topbar .spacer {{ flex: 1; }}
    .topbar .status-dot {{
      width: 8px; height: 8px; border-radius: 50%; background: var(--green);
      display: inline-block; margin-right: 6px;
      box-shadow: 0 0 6px var(--green);
    }}
    .topbar .api-link {{ color: var(--dim); text-decoration: none; font-size: 11px; }}
    .topbar .api-link:hover {{ color: var(--accent2); }}

    .main-grid {{
      display: grid;
      grid-template-columns: 320px 1fr;
      grid-template-rows: auto 1fr;
      gap: 16px;
      padding: 16px 24px 24px;
      max-width: 1400px;
      margin: 0 auto;
    }}

    /* ---- Cards ---- */
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
    }}
    .card-title {{
      font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
      color: var(--dim); margin-bottom: 14px;
      display: flex; align-items: center; gap: 8px;
    }}
    .card-title .icon {{ font-size: 14px; }}

    /* ---- Cluster nodes table ---- */
    .node-table {{ width: 100%; border-collapse: collapse; }}
    .node-table th {{
      font-size: 10px; color: var(--dim); text-transform: uppercase;
      letter-spacing: 0.8px; text-align: left; padding: 4px 8px 8px;
    }}
    .node-table td {{ padding: 6px 8px; border-top: 1px solid var(--border); vertical-align: middle; }}
    .rank-circle {{
      display: inline-flex; align-items: center; justify-content: center;
      width: 22px; height: 22px; border-radius: 50%;
      background: var(--surface2); border: 1px solid var(--border);
      font-size: 11px; font-weight: 700;
    }}
    .badge {{ border-radius: 4px; padding: 2px 7px; font-size: 10px; font-weight: 600; }}
    .badge-coord {{ background: #1e264f; color: var(--accent); border: 1px solid var(--accent); }}
    .badge-worker {{ background: #1a2f2a; color: var(--accent2); border: 1px solid var(--accent2); }}
    .dot-green {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--green); margin-right: 5px; }}
    .dim {{ color: var(--dim); }}
    code {{ color: var(--accent2); font-family: var(--font); }}

    /* ---- Stats grid ---- */
    .stats-grid {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
      margin-top: 4px;
    }}
    .stat-box {{
      background: var(--surface2); border-radius: 8px; padding: 10px 12px;
      border: 1px solid var(--border);
    }}
    .stat-box .label {{ font-size: 10px; color: var(--dim); text-transform: uppercase; letter-spacing: 0.7px; }}
    .stat-box .value {{
      font-size: 22px; font-weight: 700; color: var(--text);
      margin-top: 3px; line-height: 1;
    }}
    .stat-box .unit {{ font-size: 11px; color: var(--dim); }}
    .stat-box.accent .value {{ color: var(--accent); }}
    .stat-box.green  .value {{ color: var(--green); }}
    .stat-box.teal   .value {{ color: var(--accent2); }}
    .stat-box.warn   .value {{ color: var(--warn); }}

    /* ---- Queue bar ---- */
    .queue-bar-wrap {{
      margin-top: 12px; background: var(--surface2);
      border-radius: 6px; overflow: hidden; height: 8px;
      border: 1px solid var(--border);
    }}
    .queue-bar {{
      height: 100%; background: var(--accent);
      transition: width 0.4s ease;
      border-radius: 6px;
    }}
    .queue-label {{ font-size: 11px; color: var(--dim); margin-top: 5px; }}

    /* ---- Sparkline ---- */
    .sparkline-wrap {{ margin-top: 12px; position: relative; height: 56px; }}
    .sparkline-wrap svg {{ width: 100%; height: 100%; }}
    .spark-label {{
      font-size: 10px; color: var(--dim); position: absolute;
      top: 0; right: 0;
    }}

    /* ---- Chat panel ---- */
    .chat-panel {{
      display: flex; flex-direction: column;
      grid-column: 2; grid-row: 1 / span 2;
    }}
    .chat-messages {{
      flex: 1; overflow-y: auto; padding: 16px;
      display: flex; flex-direction: column; gap: 12px;
      min-height: 400px; max-height: calc(100vh - 260px);
    }}
    .msg {{
      display: flex; gap: 10px; align-items: flex-start;
      animation: fadeIn 0.2s ease;
    }}
    @keyframes fadeIn {{ from {{ opacity:0; transform: translateY(4px); }} to {{ opacity:1; transform: none; }} }}
    .msg-avatar {{
      width: 28px; height: 28px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 13px; flex-shrink: 0; margin-top: 2px;
    }}
    .msg.user .msg-avatar {{ background: var(--accent); }}
    .msg.assistant .msg-avatar {{ background: var(--surface2); border: 1px solid var(--border); }}
    .msg-body {{ flex: 1; }}
    .msg-role {{ font-size: 10px; color: var(--dim); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.7px; }}
    .msg-content {{
      background: var(--surface2); border: 1px solid var(--border);
      border-radius: 8px; padding: 10px 14px;
      line-height: 1.6; white-space: pre-wrap; word-break: break-word;
    }}
    .msg.user .msg-content {{ border-color: #2d355a; background: #1e2240; }}
    .msg.assistant .msg-content {{ border-color: var(--border); }}
    .msg-content.streaming::after {{
      content: "▊"; animation: blink 0.7s step-end infinite;
      color: var(--accent2);
    }}
    @keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0}} }}

    /* ---- Chat form ---- */
    .chat-form-wrap {{
      padding: 14px 16px;
      border-top: 1px solid var(--border);
      background: var(--surface);
    }}
    .chat-form {{ display: flex; gap: 10px; align-items: flex-end; }}
    .chat-form textarea {{
      flex: 1; background: var(--surface2); border: 1px solid var(--border);
      border-radius: 8px; color: var(--text); font-family: var(--font);
      font-size: 13px; padding: 10px 14px; resize: none;
      outline: none; line-height: 1.5;
      transition: border-color 0.2s;
      min-height: 44px; max-height: 160px;
    }}
    .chat-form textarea:focus {{ border-color: var(--accent); }}
    .chat-form textarea::placeholder {{ color: var(--dim); }}
    .send-btn {{
      background: var(--accent); color: #fff; border: none;
      border-radius: 8px; padding: 10px 18px; cursor: pointer;
      font-family: var(--font); font-size: 13px; font-weight: 600;
      transition: background 0.15s, transform 0.1s;
      height: 44px; white-space: nowrap;
    }}
    .send-btn:hover {{ background: #6b7cf5; }}
    .send-btn:active {{ transform: scale(0.97); }}
    .send-btn:disabled {{ background: var(--dim); cursor: not-allowed; }}
    .chat-meta {{
      font-size: 11px; color: var(--dim); margin-top: 7px;
      display: flex; gap: 16px; align-items: center;
    }}
    .chat-meta .shortcut {{ opacity: 0.6; }}

    /* ---- Left column ---- */
    .left-col {{
      display: flex; flex-direction: column; gap: 14px;
      grid-column: 1; grid-row: 1 / span 2;
    }}

    /* ---- Uptime / misc ---- */
    .uptime-row {{ display: flex; justify-content: space-between; margin-top: 8px; }}
    .uptime-row span {{ font-size: 11px; color: var(--dim); }}
    .uptime-row .val {{ color: var(--text); }}

    /* ---- RDMA badge ---- */
    .rdma-banner {{
      background: linear-gradient(135deg, #0d2a22 0%, #0f2035 100%);
      border: 1px solid #1a4a35;
      border-radius: 8px; padding: 10px 14px;
      display: flex; align-items: center; gap: 10px;
      margin-top: 4px;
    }}
    .rdma-banner .rdma-icon {{ font-size: 20px; }}
    .rdma-banner .rdma-text {{ flex: 1; }}
    .rdma-banner .rdma-title {{ color: var(--accent2); font-weight: 700; font-size: 13px; }}
    .rdma-banner .rdma-sub {{ color: var(--dim); font-size: 11px; margin-top: 2px; }}
    .rdma-banner .rdma-speed {{
      font-size: 18px; font-weight: 700; color: var(--accent2);
      text-align: right;
    }}
    .rdma-banner .rdma-speed-sub {{ font-size: 10px; color: var(--dim); text-align: right; }}

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: var(--surface); }}
    ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

    /* ---- Error toast ---- */
    #error-toast {{
      position: fixed; bottom: 24px; right: 24px;
      background: var(--danger); color: #fff;
      border-radius: 8px; padding: 10px 18px;
      font-size: 13px; display: none; z-index: 999;
      box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    }}

    /* ---- Responsive ---- */
    @media (max-width: 900px) {{
      .main-grid {{ grid-template-columns: 1fr; }}
      .chat-panel {{ grid-column: 1; grid-row: auto; }}
      .left-col {{ grid-column: 1; grid-row: auto; }}
    }}
  </style>
</head>
<body>

<div class="topbar">
  <span class="logo">⚡ mlx-jaccl</span>
  <span class="model-badge">{model_id}</span>
  <span style="color:var(--dim);font-size:11px;">{world_size}-node cluster</span>
  <span class="spacer"></span>
  <span><span class="status-dot" id="live-dot"></span><span style="font-size:11px;color:var(--dim);">live</span></span>
  <a class="api-link" href="/docs" target="_blank">API docs ↗</a>
  <a class="api-link" href="{api_base}/v1" target="_blank" style="margin-left:8px;">{api_base}/v1 ↗</a>
</div>

<div class="main-grid">

  <!-- ============ LEFT COLUMN ============ -->
  <div class="left-col">

    <!-- Cluster Nodes -->
    <div class="card">
      <div class="card-title"><span class="icon">🖥</span> Cluster Nodes</div>
      <table class="node-table">
        <thead>
          <tr>
            <th>Rank</th><th>Role</th><th>RDMA device</th><th>Status</th>
          </tr>
        </thead>
        <tbody>{node_rows}
        </tbody>
      </table>

      <div class="rdma-banner" style="margin-top:14px;">
        <span class="rdma-icon">🔗</span>
        <div class="rdma-text">
          <div class="rdma-title">RDMA / Thunderbolt 5</div>
          <div class="rdma-sub">JACCL backend · all_sum collective</div>
        </div>
        <div>
          <div class="rdma-speed">~8 GB/s</div>
          <div class="rdma-speed-sub">peak bandwidth</div>
        </div>
      </div>
    </div>

    <!-- Live Metrics (SSE) -->
    <div class="card"
         hx-ext="sse"
         sse-connect="/metrics/stream"
         sse-swap="message"
         hx-swap="none"
         id="metrics-sse-anchor">
      <div class="card-title"><span class="icon">📊</span> Live Metrics <span id="metrics-age" style="color:var(--dim);font-size:10px;margin-left:auto;"></span></div>

      <div class="stats-grid" id="stats-grid">
        <div class="stat-box green">
          <div class="label">Avg tok/s</div>
          <div class="value" id="m-avg-tps">—</div>
          <div class="unit">last 60 s</div>
        </div>
        <div class="stat-box teal">
          <div class="label">Peak tok/s</div>
          <div class="value" id="m-peak-tps">—</div>
          <div class="unit">last 60 s</div>
        </div>
        <div class="stat-box accent">
          <div class="label">Requests</div>
          <div class="value" id="m-total-req">0</div>
          <div class="unit">total</div>
        </div>
        <div class="stat-box warn">
          <div class="label">Latency</div>
          <div class="value" id="m-latency">—</div>
          <div class="unit">avg s</div>
        </div>
      </div>

      <!-- Queue depth bar -->
      <div style="margin-top:14px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span style="font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:0.7px;">Queue depth</span>
          <span style="font-size:11px;" id="q-label">0 / {queue_max}</span>
        </div>
        <div class="queue-bar-wrap">
          <div class="queue-bar" id="q-bar" style="width:0%"></div>
        </div>
      </div>

      <!-- Sparkline (tok/s) -->
      <div class="sparkline-wrap" style="margin-top:14px;">
        <span class="spark-label">tok/s</span>
        <svg id="spark-svg" viewBox="0 0 280 50" preserveAspectRatio="none">
          <polyline id="spark-line"
            fill="none" stroke="var(--accent2)" stroke-width="1.5"
            stroke-linejoin="round" stroke-linecap="round"
            points=""/>
          <polyline id="spark-fill"
            fill="var(--accent2)" fill-opacity="0.08" stroke="none"
            points=""/>
        </svg>
      </div>

      <div class="uptime-row">
        <span>uptime</span><span class="val" id="m-uptime">—</span>
        <span>total tokens</span><span class="val" id="m-total-tok">0</span>
      </div>
    </div>

  </div><!-- /left-col -->

  <!-- ============ CHAT PANEL ============ -->
  <div class="card chat-panel">
    <div class="card-title" style="padding:0 0 12px;border-bottom:1px solid var(--border);">
      <span class="icon">💬</span> Chat
      <span style="margin-left:auto;font-size:10px;color:var(--dim);" id="gen-status"></span>
    </div>

    <div class="chat-messages" id="chat-messages">
      <div style="text-align:center;margin:auto;color:var(--dim);">
        <div style="font-size:28px;margin-bottom:8px;">⚡</div>
        <div style="font-size:14px;">mlx-jaccl — {world_size} nodes · {model_id}</div>
        <div style="font-size:11px;margin-top:6px;opacity:0.6;">RDMA over Thunderbolt 5 · JACCL backend</div>
      </div>
    </div>

    <div class="chat-form-wrap">
      <div class="chat-form" id="chat-form">
        <textarea id="chat-input" rows="1"
          placeholder="Send a message… (Shift+Enter for newline)"
          onkeydown="handleKey(event)"></textarea>
        <button class="send-btn" id="send-btn" onclick="sendMessage()">Send ↵</button>
      </div>
      <div class="chat-meta">
        <span class="shortcut">Shift+Enter — newline</span>
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
          <input type="checkbox" id="stream-toggle" checked style="accent-color:var(--accent);">
          <span>Stream tokens</span>
        </label>
        <label style="display:flex;align-items:center;gap:6px;">
          <span style="color:var(--dim);">Max tokens</span>
          <input type="number" id="max-tokens-input" value="512" min="64" max="4096" step="64"
            style="width:60px;background:var(--surface2);border:1px solid var(--border);color:var(--text);
                   border-radius:5px;padding:2px 6px;font-family:var(--font);font-size:12px;outline:none;">
        </label>
      </div>
    </div>
  </div><!-- /chat-panel -->

</div><!-- /main-grid -->

<div id="error-toast"></div>

<script>
// ---- Chat state ----
let messages = [];   // [{{role, content}}]
let generating = false;
let currentReader = null;

function handleKey(e) {{
  if (e.key === 'Enter' && !e.shiftKey) {{
    e.preventDefault();
    sendMessage();
  }}
  // Auto-resize textarea
  const ta = document.getElementById('chat-input');
  setTimeout(() => {{
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
  }}, 0);
}}

function escapeHtml(text) {{
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
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
  const status = document.getElementById('gen-status');
  btn.disabled = val;
  btn.textContent = val ? 'Generating…' : 'Send ↵';
  status.textContent = val ? '⏳ generating…' : '';
}}

function scrollToBottom() {{
  const el = document.getElementById('chat-messages');
  el.scrollTop = el.scrollHeight;
}}

function clearPlaceholder() {{
  const ph = document.querySelector('#chat-messages > div[style*="text-align:center"]');
  if (ph) ph.remove();
}}

function appendMessage(role, content, streaming) {{
  clearPlaceholder();
  const id = 'msg-' + Date.now() + '-' + Math.random().toString(36).slice(2);
  const avatar = role === 'user' ? '👤' : '🤖';
  const streamClass = streaming ? ' streaming' : '';
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  el.id = id;
  el.innerHTML = `
    <div class="msg-avatar">${{avatar}}</div>
    <div class="msg-body">
      <div class="msg-role">${{role}}</div>
      <div class="msg-content${{streamClass}}" id="${{id}}-content">${{escapeHtml(content)}}</div>
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

  try {{
    if (stream) {{
      const resp = await fetch('/v1/chat/completions', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          messages: messages,
          max_tokens: maxTokens,
          stream: true,
        }}),
      }});

      if (!resp.ok) {{
        const err = await resp.text();
        throw new Error(`HTTP ${{resp.status}}: ${{err}}`);
      }}

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
            if (e.message !== 'Unexpected end of JSON input') console.warn('parse err', e);
          }}
        }}
      }}
      updateMessage(msgId, fullText, true);

    }} else {{
      // Non-streaming
      const resp = await fetch('/v1/chat/completions', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          messages: messages,
          max_tokens: maxTokens,
          stream: false,
        }}),
      }});
      if (!resp.ok) {{
        const err = await resp.text();
        throw new Error(`HTTP ${{resp.status}}: ${{err}}`);
      }}
      const data = await resp.json();
      fullText = data.choices?.[0]?.message?.content || '';
      const timing = data.timing;
      if (timing) {{
        const info = ` [done · ${{timing.tokens_per_sec}} tok/s · ${{timing.seconds}}s]`;
        updateMessage(msgId, fullText + '\\n' + info, true);
      }} else {{
        updateMessage(msgId, fullText, true);
      }}
    }}

    if (fullText) messages.push({{ role: 'assistant', content: fullText }});

  }} catch (e) {{
    updateMessage(msgId, '⚠ ' + e.message, true);
    showError(e.message);
    messages.pop();  // remove user message if generation failed
  }} finally {{
    setGenerating(false);
    currentReader = null;
  }}
}}

// ---- Metrics SSE ----
function formatUptime(secs) {{
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (h > 0) return `${{h}}h ${{m}}m`;
  if (m > 0) return `${{m}}m ${{s}}s`;
  return `${{s}}s`;
}}

function renderSparkline(history) {{
  if (!history || history.length < 2) return;
  const vals = history.map(h => h.tps);
  const maxV = Math.max(...vals, 1);
  const W = 280, H = 50;
  const pts = vals.map((v, i) => {{
    const x = (i / (vals.length - 1)) * W;
    const y = H - (v / maxV) * (H - 4) - 2;
    return `${{x.toFixed(1)}},${{y.toFixed(1)}}`;
  }});
  const lineStr = pts.join(' ');
  const fillStr = `0,${{H}} ` + lineStr + ` ${{W}},${{H}}`;
  document.getElementById('spark-line').setAttribute('points', lineStr);
  document.getElementById('spark-fill').setAttribute('points', fillStr);
}}

// Listen for SSE metrics updates
document.body.addEventListener('htmx:sseMessage', function(evt) {{
  try {{
    const m = JSON.parse(evt.detail.data);

    document.getElementById('m-avg-tps').textContent =
      m.avg_tps_60s > 0 ? m.avg_tps_60s : '—';
    document.getElementById('m-peak-tps').textContent =
      m.peak_tps_60s > 0 ? m.peak_tps_60s : '—';
    document.getElementById('m-total-req').textContent = m.total_requests;
    document.getElementById('m-latency').textContent =
      m.avg_latency_60s > 0 ? m.avg_latency_60s : '—';
    document.getElementById('m-uptime').textContent = formatUptime(m.uptime_s);
    document.getElementById('m-total-tok').textContent =
      m.total_tokens > 999 ? (m.total_tokens / 1000).toFixed(1) + 'k' : m.total_tokens;

    // Queue bar
    const qSize = m.queue_size || 0;
    const qMax = m.queue_max || {queue_max};
    document.getElementById('q-label').textContent = `${{qSize}} / ${{qMax}}`;
    const pct = Math.min(100, (qSize / qMax) * 100);
    const bar = document.getElementById('q-bar');
    bar.style.width = pct + '%';
    bar.style.background = pct > 75 ? 'var(--danger)' : pct > 40 ? 'var(--warn)' : 'var(--accent)';

    // Sparkline
    if (m.history) renderSparkline(m.history);

    // Age label
    document.getElementById('metrics-age').textContent = 'updated ' + new Date().toLocaleTimeString();

    // Live dot pulse
    const dot = document.getElementById('live-dot');
    dot.style.opacity = '0.3';
    setTimeout(() => {{ dot.style.opacity = '1'; }}, 200);

  }} catch(e) {{ /* ignore parse errors */ }}
}});

// Fallback poll if SSE isn't available (older browsers / non-HTMX path)
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
// Poll every 3s as backup
setInterval(pollMetrics, 3000);
pollMetrics();  // initial load
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------


async def _metrics_event_generator(
    get_queue_info: Callable[[], dict],
    interval: float = 2.0,
) -> AsyncGenerator[str, None]:
    """Yields SSE events with merged metrics + queue info every `interval` seconds."""
    while True:
        try:
            snap = await metrics_store.snapshot()
            qi = get_queue_info()
            snap.update(qi)
            yield f"data: {json.dumps(snap)}\n\n"
        except asyncio.CancelledError:
            break
        except Exception:
            pass
        await asyncio.sleep(interval)


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
) -> None:
    """
    Mount dashboard routes onto an existing FastAPI app.

    Parameters
    ----------
    app            : the FastAPI instance from openai_cluster_server.py
    get_state      : callable returning current server state dict (can be lambda: {})
    get_queue_info : callable returning {"queue_size": int, "queue_max": int}
    model_id       : model name/id string
    world_size     : total number of distributed ranks
    rank           : rank of this node (dashboard only served on rank 0)
    queue_max      : maximum queue depth (for display)
    rdma_devices   : list of RDMA device names per rank (e.g. ["rdma_en4", "rdma_en4"])
    host           : bind host for the HTTP server
    port           : HTTP port
    """
    if rdma_devices is None:
        rdma_devices = [f"rdma_en4" for _ in range(world_size)]

    # Pre-render the HTML once (it's static except for live metrics)
    _html = _render_dashboard(
        model_id=model_id,
        world_size=world_size,
        rank=rank,
        queue_max=queue_max,
        rdma_devices=rdma_devices,
        host=host,
        port=port,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard_root():
        return HTMLResponse(content=_html)

    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard_page():
        return HTMLResponse(content=_html)

    @app.get("/metrics/stream", include_in_schema=False)
    async def metrics_stream(request: Request):
        """SSE endpoint — pushes metrics JSON every 2 s."""

        async def event_gen():
            async for event in _metrics_event_generator(get_queue_info):
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
        return snap
