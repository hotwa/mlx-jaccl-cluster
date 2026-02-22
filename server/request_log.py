#!/usr/bin/env python3
"""
request_log.py — Per-request logging with JSONL persistence and aggregate stats.

Provides:
  1. RequestRecord dataclass capturing full request lifecycle metadata
  2. RequestLog with dual storage:
     - In-memory ring buffer (deque) for fast recent-query access
     - Append-only JSONL file for persistent history
  3. Aggregate statistics: total, error rate, avg/p50/p95 tok/s and latency
  4. Thread-safe writes (single lock for append to both stores)

Usage:
    from request_log import RequestLog, RequestRecord

    rlog = RequestLog()                         # logs/requests.jsonl by default
    rlog.record(RequestRecord(
        request_id="chatcmpl-abc123",
        kind="chat",
        model_id="Qwen3-8B-4bit",
        prompt_tokens=42,
        generated_tokens=128,
        ...
    ))

    recent = rlog.recent(20)                    # last 20 records
    stats  = rlog.stats()                       # aggregate dict
    rlog.clear()                                # wipe in-memory + optionally file

Integration with openai_cluster_server.py:
    - _queue_worker() creates a RequestRecord before/after each generation
    - Endpoints: GET /requests/recent, GET /requests/stats
"""

import json
import logging
import os
import threading
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

log = logging.getLogger("request_log")

# ============================================================================
#  Constants
# ============================================================================

_DEFAULT_LOG_PATH = os.environ.get("REQUEST_LOG_PATH", "logs/requests.jsonl")
_DEFAULT_MAX_ENTRIES = int(os.environ.get("REQUEST_LOG_MAX", "10000"))

# Valid status values
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_TIMEOUT = "timeout"
STATUS_CANCELLED = "cancelled"
STATUS_PRESSURE_ABORT = "pressure_abort"


# ============================================================================
#  RequestRecord
# ============================================================================


@dataclass
class RequestRecord:
    """
    Complete lifecycle record for a single API request.

    Captured by the queue worker in openai_cluster_server.py and written
    to both the in-memory ring buffer and the JSONL log file.
    """

    # --- Identity ---
    request_id: str = ""
    timestamp: float = 0.0  # unix epoch when request entered the queue worker
    kind: str = ""  # "chat" | "completions"
    model_id: str = ""

    # --- Token counts ---
    prompt_tokens: int = 0
    generated_tokens: int = 0
    max_tokens_requested: int = 0

    # --- Timing ---
    wall_time_s: float = 0.0  # total wall-clock time for the request
    tokens_per_second: float = 0.0

    # --- Memory (before/after generation) ---
    memory_before_gb: float = 0.0
    memory_after_gb: float = 0.0
    memory_delta_gb: float = 0.0  # after - before (positive = grew)

    # --- Cache ---
    kv_cache_hit: bool = False  # future: populated by KV prefix cache (Phase S2)

    # --- Outcome ---
    status: str = STATUS_OK  # ok | error | timeout | cancelled | pressure_abort
    error_message: Optional[str] = None
    finish_reason: str = ""  # stop | length | memory_pressure | error

    # --- Streaming ---
    is_stream: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict, suitable for JSON serialization."""
        d = asdict(self)
        # Round floats for cleaner output
        for key in (
            "timestamp",
            "wall_time_s",
            "tokens_per_second",
            "memory_before_gb",
            "memory_after_gb",
            "memory_delta_gb",
        ):
            if key in d and isinstance(d[key], float):
                d[key] = round(d[key], 4)
        return d

    def to_json_line(self) -> str:
        """Serialize to a single JSON line (for JSONL file)."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


# ============================================================================
#  RequestLog
# ============================================================================


class RequestLog:
    """
    Dual-store request logger: in-memory ring buffer + JSONL file.

    Thread-safe: all writes go through a single lock.
    Reads from the deque are safe without the lock in CPython (deque iteration
    is atomic in CPython), but we hold the lock for stats computation to get
    a consistent snapshot.

    Parameters:
        path:         JSONL file path (parent dirs created automatically)
        max_entries:  max entries in the in-memory ring buffer
        persist:      if False, skip file writes (useful for testing)
    """

    def __init__(
        self,
        path: str = _DEFAULT_LOG_PATH,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        persist: bool = True,
    ):
        self._path = Path(path)
        self._persist = persist
        self._lock = threading.Lock()
        self._buffer: Deque[RequestRecord] = deque(maxlen=max_entries)

        # Running counters (avoid re-scanning the full buffer for basic stats)
        self._total_count: int = 0
        self._error_count: int = 0
        self._timeout_count: int = 0
        self._cancelled_count: int = 0
        self._pressure_abort_count: int = 0
        self._total_generated_tokens: int = 0
        self._total_wall_time_s: float = 0.0

        # Ensure log directory exists
        if self._persist:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                log.warning(
                    f"Could not create log directory {self._path.parent}: {e} — "
                    f"file logging disabled"
                )
                self._persist = False

        # Load existing entry count (don't load full history into memory)
        if self._persist and self._path.exists():
            try:
                with open(self._path) as f:
                    existing = sum(1 for _ in f)
                if existing > 0:
                    log.info(
                        f"RequestLog: found {existing} existing entries in {self._path}"
                    )
            except Exception:
                pass

        log.info(
            f"RequestLog initialized: path={self._path}, "
            f"max_entries={max_entries}, persist={self._persist}"
        )

    # ------------------------------------------------------------------
    #  Write
    # ------------------------------------------------------------------

    def record(self, entry: RequestRecord) -> None:
        """
        Record a completed request.

        Appends to both the in-memory ring buffer and the JSONL file.
        Thread-safe.
        """
        with self._lock:
            self._buffer.append(entry)

            # Update running counters
            self._total_count += 1
            self._total_generated_tokens += entry.generated_tokens
            self._total_wall_time_s += entry.wall_time_s

            if entry.status == STATUS_ERROR:
                self._error_count += 1
            elif entry.status == STATUS_TIMEOUT:
                self._timeout_count += 1
            elif entry.status == STATUS_CANCELLED:
                self._cancelled_count += 1
            elif entry.status == STATUS_PRESSURE_ABORT:
                self._pressure_abort_count += 1

        # File write outside the lock (IO can be slow, don't block reads)
        if self._persist:
            try:
                with open(self._path, "a") as f:
                    f.write(entry.to_json_line() + "\n")
            except Exception as e:
                log.warning(f"Failed to write request log entry: {e}")

    # ------------------------------------------------------------------
    #  Read
    # ------------------------------------------------------------------

    def recent(self, n: int = 50) -> List[Dict[str, Any]]:
        """
        Return the last N request records as dicts.

        Thread-safe (deque slicing is safe in CPython).
        """
        records = list(self._buffer)
        return [r.to_dict() for r in records[-n:]]

    def recent_records(self, n: int = 50) -> List[RequestRecord]:
        """Return the last N RequestRecord objects."""
        records = list(self._buffer)
        return records[-n:]

    def stats(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics over the in-memory buffer.

        Returns a dict with:
          - total, ok, errors, timeouts, cancelled, pressure_aborts (counts)
          - error_rate (fraction)
          - avg_tps, p50_tps, p95_tps (tokens per second)
          - avg_latency_s, p50_latency_s, p95_latency_s (wall time)
          - total_generated_tokens, total_wall_time_s
          - avg_memory_delta_gb
          - kv_cache_hit_rate (when available)
        """
        with self._lock:
            records = list(self._buffer)
            total = self._total_count
            errors = self._error_count
            timeouts = self._timeout_count
            cancelled = self._cancelled_count
            pressure_aborts = self._pressure_abort_count
            total_tokens = self._total_generated_tokens
            total_wall = self._total_wall_time_s

        ok_count = total - errors - timeouts - cancelled - pressure_aborts

        result: Dict[str, Any] = {
            "total_requests": total,
            "ok": ok_count,
            "errors": errors,
            "timeouts": timeouts,
            "cancelled": cancelled,
            "pressure_aborts": pressure_aborts,
            "error_rate": round(errors / max(total, 1), 4),
            "total_generated_tokens": total_tokens,
            "total_wall_time_s": round(total_wall, 2),
            "buffer_size": len(records),
        }

        # --- Percentile calculations over the buffer ---
        if records:
            # Filter to successful requests for perf stats
            ok_records = [r for r in records if r.status == STATUS_OK]

            if ok_records:
                tps_values = sorted(
                    r.tokens_per_second for r in ok_records if r.tokens_per_second > 0
                )
                latency_values = sorted(
                    r.wall_time_s for r in ok_records if r.wall_time_s > 0
                )
                memory_deltas = [r.memory_delta_gb for r in ok_records]

                if tps_values:
                    result["avg_tps"] = round(sum(tps_values) / len(tps_values), 2)
                    result["p50_tps"] = round(_percentile(tps_values, 0.50), 2)
                    result["p95_tps"] = round(_percentile(tps_values, 0.95), 2)

                if latency_values:
                    result["avg_latency_s"] = round(
                        sum(latency_values) / len(latency_values), 3
                    )
                    result["p50_latency_s"] = round(
                        _percentile(latency_values, 0.50), 3
                    )
                    result["p95_latency_s"] = round(
                        _percentile(latency_values, 0.95), 3
                    )

                if memory_deltas:
                    result["avg_memory_delta_gb"] = round(
                        sum(memory_deltas) / len(memory_deltas), 4
                    )

            # KV cache hit rate (across all records, not just OK ones)
            cache_checked = [r for r in records if r.status == STATUS_OK]
            if cache_checked:
                hits = sum(1 for r in cache_checked if r.kv_cache_hit)
                result["kv_cache_hit_rate"] = round(hits / len(cache_checked), 4)

        return result

    # ------------------------------------------------------------------
    #  Management
    # ------------------------------------------------------------------

    def clear(self, clear_file: bool = False) -> Dict[str, int]:
        """
        Clear the in-memory buffer and optionally truncate the log file.

        Returns a dict with the number of entries cleared.
        """
        with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            cleared_counters = self._total_count
            self._total_count = 0
            self._error_count = 0
            self._timeout_count = 0
            self._cancelled_count = 0
            self._pressure_abort_count = 0
            self._total_generated_tokens = 0
            self._total_wall_time_s = 0.0

        if clear_file and self._persist:
            try:
                with open(self._path, "w") as f:
                    pass  # truncate
                log.info(f"RequestLog file truncated: {self._path}")
            except Exception as e:
                log.warning(f"Failed to truncate log file: {e}")

        log.info(
            f"RequestLog cleared: {count} buffer entries, "
            f"{cleared_counters} total counter"
        )
        return {
            "buffer_entries_cleared": count,
            "counter_total_cleared": cleared_counters,
        }

    @property
    def log_path(self) -> str:
        """Return the path to the JSONL log file."""
        return str(self._path)

    @property
    def entry_count(self) -> int:
        """Return the total number of requests recorded since init (or last clear)."""
        return self._total_count

    def __repr__(self) -> str:
        return (
            f"RequestLog("
            f"entries={self._total_count}, "
            f"buffer={len(self._buffer)}, "
            f"errors={self._error_count}, "
            f"path={self._path})"
        )


# ============================================================================
#  Helpers
# ============================================================================


def _percentile(sorted_values: List[float], pct: float) -> float:
    """
    Compute a percentile from a pre-sorted list of values.

    Uses the "nearest rank" method for simplicity.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = int(pct * (n - 1))
    idx = max(0, min(idx, n - 1))
    return sorted_values[idx]


# ============================================================================
#  Module-level singleton
# ============================================================================

_instance: Optional[RequestLog] = None


def get_request_log() -> RequestLog:
    """Get or create the global RequestLog singleton."""
    global _instance
    if _instance is None:
        _instance = RequestLog()
    return _instance


def init_request_log(**kwargs: Any) -> RequestLog:
    """Initialize the global RequestLog with custom settings."""
    global _instance
    _instance = RequestLog(**kwargs)
    return _instance
