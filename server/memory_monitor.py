#!/usr/bin/env python3
"""
memory_monitor.py — Continuous background memory & hardware monitoring.

Architecture (follows exo's InfoGatherer pattern):
  ┌─────────────────────────────────────────────────────────────┐
  │                    MemoryMonitor                            │
  │                                                             │
  │  ┌──────────────┐   primary    ┌────────────────────────┐  │
  │  │ macmon pipe   │────────────▶│  MacmonSnapshot        │  │
  │  │ (persistent   │  JSON lines │  ram, gpu, temp, power │  │
  │  │  subprocess)  │             └────────────────────────┘  │
  │  └──────────────┘                                          │
  │         ▲ fallback if macmon not found                      │
  │  ┌──────────────┐                                          │
  │  │ vm_stat /    │─────────────▶ system memory only         │
  │  │ psutil poll  │                                          │
  │  └──────────────┘                                          │
  │                                                             │
  │  ┌──────────────┐   always     ┌────────────────────────┐  │
  │  │ MLX poll     │────────────▶│  MLX MemorySnapshot    │  │
  │  │ (mm.snapshot)│  every tick  │  active, peak, cache,  │  │
  │  └──────────────┘              │  pressure, model info  │  │
  │                                └────────────────────────┘  │
  │                                                             │
  │  Consumers:                                                 │
  │    • MemoryManager  (pressure checks, limit decisions)      │
  │    • Dashboard SSE  (replaces HardwarePoller for local)     │
  │    • /memory/live   (pressure history graph)                │
  │    • KV cache S2    (eviction threshold)                    │
  └─────────────────────────────────────────────────────────────┘

Key design decisions:
  - macmon runs as a PERSISTENT streaming subprocess (like exo), not one-shot.
    Exo: `macmon pipe --interval 1000` → reads JSON lines from stdout.
    Us:  `macmon pipe -i <ms>`          → same approach, threaded not async.
  - Two daemon threads:
      1. _macmon_thread   — reads lines from macmon stdout (or fallback poll)
      2. _mlx_poll_thread — polls MemoryManager.snapshot() at fixed interval
    Separated because macmon is IO-bound (subprocess read) while MLX queries
    are CPU-bound (Metal API calls that release GIL).
  - Thread-safe reads: dataclass snapshots assigned atomically, deque(maxlen=N)
    is thread-safe for append/iter in CPython.
  - Falls back gracefully: macmon not found → vm_stat → psutil → zeros.
  - Tiered thresholds match exo's _default_memory_threshold() scheme.

Reference:
  - exo: refs/exo/src/exo/utils/info_gatherer/info_gatherer.py  _monitor_macmon()
  - exo: refs/exo/src/exo/utils/info_gatherer/macmon.py          MacmonMetrics
  - exo: refs/exo/src/exo/worker/engines/mlx/cache.py            _default_memory_threshold()
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional

if TYPE_CHECKING:
    from memory_manager import MemoryManager, MemorySnapshot

log = logging.getLogger("memory_monitor")

# ============================================================================
#  Constants
# ============================================================================

_GiB = 1024**3

# Macmon persistent pipe interval in milliseconds.
# exo uses 1000 (1 s); we default to 1500 to match our overall poll cadence.
_MACMON_INTERVAL_MS = int(os.environ.get("MACMON_INTERVAL_MS", "1500"))

# MLX snapshot poll interval (seconds).  Independent of macmon cadence.
_MLX_POLL_INTERVAL = float(os.environ.get("MEMORY_POLL_INTERVAL", "1.5"))

# How many records to keep in the rolling history.
# At 1.5 s interval, 60 entries ≈ 90 s of history.
_HISTORY_SIZE = 60

# Fallback vm_stat / psutil poll interval when macmon is not available.
_FALLBACK_POLL_INTERVAL = float(os.environ.get("MEMORY_FALLBACK_INTERVAL", "1.5"))

# Environment variable to override reported available memory (for testing).
# Same name as exo uses: OVERRIDE_MEMORY_MB
_OVERRIDE_MEMORY_MB_ENV = "OVERRIDE_MEMORY_MB"

# Common paths where macmon may live (Homebrew, Cargo)
_MACMON_SEARCH_PATHS = [
    "/opt/homebrew/bin/macmon",
    "/usr/local/bin/macmon",
    os.path.expanduser("~/.cargo/bin/macmon"),
]


# ============================================================================
#  Tiered memory thresholds  (matches exo cache.py lines 30-38)
# ============================================================================


def default_memory_threshold(total_ram_gb: float) -> float:
    """
    Default memory pressure threshold based on total system RAM.

    Smaller machines need lower (more aggressive) thresholds because they
    have less absolute headroom.  15 % of 32 GB = 4.8 GB, while
    15 % of 192 GB = 28.8 GB.

    Directly mirrors exo's ``_default_memory_threshold()`` from
    ``refs/exo/src/exo/worker/engines/mlx/cache.py`` lines 30-38.

    Returns a fraction 0.0–1.0.  Usage above this level is "high pressure"
    and should trigger eviction / cleanup.
    """
    if total_ram_gb >= 128:
        return 0.85
    if total_ram_gb >= 64:
        return 0.80
    if total_ram_gb >= 32:
        return 0.75
    return 0.70


def tiered_pressure_thresholds(total_ram_gb: float) -> Dict[str, float]:
    """
    Warn and critical MLX-pressure thresholds (fraction of MLX limit),
    tiered by machine size.

    Returns dict with keys ``"warn"`` and ``"critical"``, values 0.0–1.0.
    """
    if total_ram_gb >= 128:
        return {"warn": 0.85, "critical": 0.95}
    if total_ram_gb >= 64:
        return {"warn": 0.82, "critical": 0.93}
    if total_ram_gb >= 32:
        return {"warn": 0.78, "critical": 0.90}
    return {"warn": 0.72, "critical": 0.88}


# ============================================================================
#  Override helpers
# ============================================================================


def get_override_memory_bytes() -> Optional[int]:
    """
    Read ``OVERRIDE_MEMORY_MB`` env var and return value in bytes, or ``None``.

    Allows simulating constrained environments::

        OVERRIDE_MEMORY_MB=16384   →  pretend 16 GB available

    Same env var name that exo uses in ``info_gatherer.py``.
    """
    raw = os.environ.get(_OVERRIDE_MEMORY_MB_ENV)
    if raw is None:
        return None
    try:
        mb = int(raw)
        if mb <= 0:
            log.warning(f"{_OVERRIDE_MEMORY_MB_ENV}={raw} is not positive — ignoring")
            return None
        log.info(
            f"{_OVERRIDE_MEMORY_MB_ENV}={mb} MB → simulating "
            f"{mb / 1024:.1f} GB available memory"
        )
        return mb * (1024**2)
    except ValueError:
        log.warning(
            f"{_OVERRIDE_MEMORY_MB_ENV}={raw!r} is not a valid integer — ignoring"
        )
        return None


# ============================================================================
#  MacmonSnapshot — parsed macmon JSON line
# ============================================================================


@dataclass(frozen=True)
class MacmonSnapshot:
    """
    One parsed macmon JSON line.

    Mirrors the fields that exo's ``RawMacmonMetrics`` / ``MacmonMetrics``
    extract from ``macmon pipe`` output, plus a few extras we already use in
    our ``HardwarePoller._parse()``.
    """

    timestamp: float  # time.time() when we received the line

    # ── Memory ────────────────────────────────────────────────────────
    ram_total_bytes: int = 0
    ram_usage_bytes: int = 0
    ram_available_bytes: int = 0  # total - usage
    swap_total_bytes: int = 0
    swap_usage_bytes: int = 0

    # Convenience GB
    ram_total_gb: float = 0.0
    ram_usage_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_pct: float = 0.0  # usage / total × 100

    # ── GPU ────────────────────────────────────────────────────────────
    gpu_freq_mhz: int = 0
    gpu_freq_ratio: float = 0.0  # freq / max freq (0-1)
    gpu_power_w: float = 0.0

    # ── CPU ────────────────────────────────────────────────────────────
    ecpu_freq_mhz: int = 0
    ecpu_usage: float = 0.0
    pcpu_freq_mhz: int = 0
    pcpu_usage: float = 0.0
    cpu_power_w: float = 0.0

    # ── Temp & Power ──────────────────────────────────────────────────
    gpu_temp_c: float = 0.0
    cpu_temp_c: float = 0.0
    sys_power_w: float = 0.0
    ane_power_w: float = 0.0

    # ── Derived (matches HardwarePoller._parse output) ────────────────
    gpu_activity_pct: int = 0  # power-derived 0-100

    @classmethod
    def from_macmon_json(
        cls, raw: dict, ts: Optional[float] = None
    ) -> "MacmonSnapshot":
        """Parse a raw macmon JSON dict into a snapshot.

        Mirrors exo ``MacmonMetrics.from_raw`` + our ``HardwarePoller._parse``.
        """
        now = ts if ts is not None else time.time()

        mem = raw.get("memory", {})
        ram_total = mem.get("ram_total", 0)
        ram_usage = mem.get("ram_usage", 0)
        ram_avail = max(0, ram_total - ram_usage)
        swap_total = mem.get("swap_total", 0)
        swap_usage = mem.get("swap_usage", 0)

        gpu_raw = raw.get("gpu_usage", [0, 0.0])
        gpu_freq = gpu_raw[0] if len(gpu_raw) > 0 else 0
        gpu_ratio = gpu_raw[1] if len(gpu_raw) > 1 else 0.0

        ecpu_raw = raw.get("ecpu_usage", [0, 0.0])
        pcpu_raw = raw.get("pcpu_usage", [0, 0.0])

        temp = raw.get("temp", {})

        gpu_power = raw.get("gpu_power", 0.0)
        # Derive 0-100 "activity %" from GPU power.
        # M4 Pro GPU TDP ≈ 22 W; scale so 20 W → 100 %.
        GPU_POWER_CEIL = 20.0
        gpu_activity = (
            min(100, round((gpu_power / GPU_POWER_CEIL) * 100)) if gpu_power > 0 else 0
        )

        return cls(
            timestamp=now,
            ram_total_bytes=ram_total,
            ram_usage_bytes=ram_usage,
            ram_available_bytes=ram_avail,
            swap_total_bytes=swap_total,
            swap_usage_bytes=swap_usage,
            ram_total_gb=round(ram_total / _GiB, 2),
            ram_usage_gb=round(ram_usage / _GiB, 2),
            ram_available_gb=round(ram_avail / _GiB, 3),
            ram_pct=round(ram_usage / ram_total * 100, 1) if ram_total > 0 else 0.0,
            gpu_freq_mhz=int(gpu_freq),
            gpu_freq_ratio=round(float(gpu_ratio), 4),
            gpu_power_w=round(gpu_power, 2),
            ecpu_freq_mhz=int(ecpu_raw[0]) if len(ecpu_raw) > 0 else 0,
            ecpu_usage=round(float(ecpu_raw[1]), 4) if len(ecpu_raw) > 1 else 0.0,
            pcpu_freq_mhz=int(pcpu_raw[0]) if len(pcpu_raw) > 0 else 0,
            pcpu_usage=round(float(pcpu_raw[1]), 4) if len(pcpu_raw) > 1 else 0.0,
            cpu_power_w=round(raw.get("cpu_power", 0.0), 2),
            gpu_temp_c=round(temp.get("gpu_temp_avg", 0.0), 1),
            cpu_temp_c=round(temp.get("cpu_temp_avg", 0.0), 1),
            sys_power_w=round(raw.get("sys_power", 0.0), 2),
            ane_power_w=round(raw.get("ane_power", 0.0), 2),
            gpu_activity_pct=gpu_activity,
        )

    def to_hardware_dict(self) -> Dict[str, Any]:
        """
        Return a dict matching the format that ``HardwarePoller._parse()``
        produces, so the dashboard can consume monitor data directly without
        changes.
        """
        return {
            "gpu_usage_pct": self.gpu_activity_pct,
            "gpu_freq_ratio": round(self.gpu_freq_ratio * 100),
            "gpu_freq_mhz": self.gpu_freq_mhz,
            "gpu_power_w": self.gpu_power_w,
            "gpu_temp_c": round(self.gpu_temp_c),
            "cpu_temp_c": round(self.cpu_temp_c),
            "sys_power_w": self.sys_power_w,
            "cpu_power_w": self.cpu_power_w,
            "ram_used_gb": round(self.ram_usage_gb, 1),
            "ram_total_gb": round(self.ram_total_gb, 1),
            "ram_pct": round(self.ram_pct),
            "timestamp": "",  # macmon timestamp string not stored; dashboard can ignore
        }

    def to_memory_dict(self) -> Dict[str, float]:
        """
        Return a dict matching the format ``MemoryManager._get_local_memory_usage()``
        returns, so it can be used as a drop-in data source.
        """
        return {
            "total_gb": self.ram_total_gb,
            "used_gb": self.ram_usage_gb,
            "free_gb": self.ram_available_gb,
            "available_gb": self.ram_available_gb,
            "wired_gb": 0.0,  # macmon doesn't break out wired
            "compressed_gb": 0.0,  # macmon doesn't break out compressed
        }


# ============================================================================
#  PressureRecord — lightweight sample for the rolling history deque
# ============================================================================


@dataclass(frozen=True)
class PressureRecord:
    """Lightweight record for the rolling pressure history."""

    timestamp: float
    # MLX state
    mlx_active_gb: float
    mlx_pressure_pct: float  # active / mlx_limit × 100
    # System state (from macmon or fallback)
    system_used_gb: float
    system_available_gb: float
    system_pressure_pct: float  # ram_usage / ram_total × 100
    model_loaded: bool
    # Hardware highlights
    gpu_power_w: float = 0.0
    gpu_temp_c: float = 0.0


# ============================================================================
#  Fallback system memory probe (when macmon is unavailable)
# ============================================================================


def _probe_system_memory_vmstat() -> Dict[str, float]:
    """
    Quick system memory probe using ``vm_stat`` + ``sysctl`` (macOS).

    Used ONLY as a fallback when macmon is not installed.  macmon gives us
    the same data (ram_total, ram_usage) and much more.
    """
    result: Dict[str, float] = {
        "total_gb": 0.0,
        "used_gb": 0.0,
        "free_gb": 0.0,
        "available_gb": 0.0,
        "wired_gb": 0.0,
        "compressed_gb": 0.0,
        "pressure_pct": 0.0,
    }

    # Total RAM via sysctl
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0:
            result["total_gb"] = int(out.stdout.strip()) / _GiB
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        try:
            import psutil

            result["total_gb"] = psutil.virtual_memory().total / _GiB
        except ImportError:
            pass

    # vm_stat page counts
    page_size = 16384
    pages: Dict[str, int] = {}
    try:
        out = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                if "page size" in line.lower():
                    try:
                        page_size = int(
                            line.split("page size of")[1].strip().split()[0]
                        )
                    except (IndexError, ValueError):
                        pass
                    continue
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                try:
                    pages[key.strip().lower()] = int(val.strip().rstrip("."))
                except ValueError:
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if pages:

        def _p2g(key: str) -> float:
            return pages.get(key, 0) * page_size / _GiB

        free = _p2g("pages free")
        inactive = _p2g("pages inactive")
        purgeable = _p2g("pages purgeable")
        wired = _p2g("pages wired down")
        compressed = _p2g("pages occupied by compressor")

        available = free + inactive + purgeable
        result["free_gb"] = round(free, 3)
        result["wired_gb"] = round(wired, 3)
        result["compressed_gb"] = round(compressed, 3)
        result["available_gb"] = round(available, 3)
        result["used_gb"] = round(max(0, result["total_gb"] - available), 3)
        if result["total_gb"] > 0:
            result["pressure_pct"] = round(
                (1.0 - available / result["total_gb"]) * 100, 1
            )
    elif result["total_gb"] > 0:
        try:
            import psutil

            vm = psutil.virtual_memory()
            result["available_gb"] = round(vm.available / _GiB, 3)
            result["used_gb"] = round(vm.used / _GiB, 3)
            result["free_gb"] = round(vm.free / _GiB, 3)
            result["pressure_pct"] = round(vm.percent, 1)
        except ImportError:
            pass

    # Apply override if set
    override = get_override_memory_bytes()
    if override is not None:
        override_gb = override / _GiB
        result["available_gb"] = round(override_gb, 3)
        result["used_gb"] = round(max(0, result["total_gb"] - override_gb), 3)
        if result["total_gb"] > 0:
            result["pressure_pct"] = round(
                (1.0 - override_gb / result["total_gb"]) * 100, 1
            )

    return result


# ============================================================================
#  Resolve macmon binary
# ============================================================================


def _find_macmon() -> Optional[str]:
    """Find macmon binary.  Returns full path or None."""
    found = shutil.which("macmon")
    if found:
        return found
    for p in _MACMON_SEARCH_PATHS:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


# ============================================================================
#  MemoryMonitor
# ============================================================================


class MemoryMonitor:
    """
    Background memory & hardware monitoring daemon.

    Primary data source: **macmon** running as a persistent streaming
    subprocess (``macmon pipe -i <ms>``), exactly like exo's
    ``InfoGatherer._monitor_macmon()``.

    Fallback: ``vm_stat`` / ``psutil`` polling when macmon is unavailable.

    Also polls MLX-specific state via ``MemoryManager.snapshot()`` on a
    separate timer.

    Exposes:
        latest_macmon     — most recent MacmonSnapshot (hardware + system RAM)
        latest_mlx        — most recent MemorySnapshot (MLX allocator state)
        history           — rolling deque of PressureRecords
        summary()         — JSON-ready dict with current + rolling stats
        hardware_dict()   — dict matching HardwarePoller._parse() output

    Usage::

        monitor = MemoryMonitor(manager=mm)
        monitor.start()

        snap = monitor.latest_macmon       # MacmonSnapshot or None
        mlx  = monitor.latest_mlx          # MemorySnapshot or None
        hist = monitor.history_dicts()     # list[dict] for JSON/SSE

        monitor.stop()

    Thread safety:
        Frozen dataclass snapshots are assigned atomically (safe in CPython).
        ``deque(maxlen=N)`` append + iteration are thread-safe in CPython.
    """

    def __init__(
        self,
        manager: "MemoryManager",
        macmon_interval_ms: int = _MACMON_INTERVAL_MS,
        mlx_poll_interval: float = _MLX_POLL_INTERVAL,
        history_size: int = _HISTORY_SIZE,
        on_critical: Optional[Callable[["PressureRecord"], None]] = None,
    ):
        self._manager = manager
        self._macmon_interval_ms = macmon_interval_ms
        self._mlx_poll_interval = mlx_poll_interval
        self._on_critical = on_critical

        # ── Public state (read from any thread) ──────────────────────
        self.latest_macmon: Optional[MacmonSnapshot] = None
        self.latest_mlx: Optional["MemorySnapshot"] = None
        self.history: Deque[PressureRecord] = deque(maxlen=history_size)

        # ── macmon state ─────────────────────────────────────────────
        self._macmon_bin = _find_macmon()
        self._macmon_proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        self._using_macmon: bool = False

        # ── Threads ──────────────────────────────────────────────────
        self._macmon_thread: Optional[threading.Thread] = None
        self._mlx_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False

        # ── Tiered thresholds ────────────────────────────────────────
        total_gb = manager._total_ram / _GiB
        self.memory_threshold = float(
            os.environ.get("MEMORY_THRESHOLD", default_memory_threshold(total_gb))
        )
        self._pressure_thresholds = tiered_pressure_thresholds(total_gb)
        self._critical_pct = self._pressure_thresholds["critical"] * 100
        self._consecutive_critical = 0
        self._CRITICAL_CALLBACK_AFTER = 2

        # ── Override tracking ────────────────────────────────────────
        self._override_bytes = get_override_memory_bytes()

        log.info(
            f"MemoryMonitor configured: "
            f"macmon={'✓ ' + self._macmon_bin if self._macmon_bin else '✗ (fallback to vm_stat)'}, "
            f"macmon_interval={self._macmon_interval_ms}ms, "
            f"mlx_poll={self._mlx_poll_interval}s, "
            f"threshold={self.memory_threshold:.0%}, "
            f"warn={self._pressure_thresholds['warn']:.0%}, "
            f"critical={self._pressure_thresholds['critical']:.0%}, "
            f"total_ram={total_gb:.0f} GB"
            + (
                f", override={self._override_bytes // (1024**2)} MB"
                if self._override_bytes
                else ""
            )
        )

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background monitoring threads."""
        if self._started:
            log.warning("MemoryMonitor already started — ignoring")
            return

        self._stop_event.clear()
        self._started = True

        # Thread 1: macmon pipe (or vm_stat fallback)
        if self._macmon_bin:
            self._macmon_thread = threading.Thread(
                target=self._macmon_pipe_loop,
                name="monitor-macmon",
                daemon=True,
            )
            self._using_macmon = True
            log.info(f"Starting persistent macmon pipe: {self._macmon_bin}")
        else:
            self._macmon_thread = threading.Thread(
                target=self._fallback_poll_loop,
                name="monitor-fallback",
                daemon=True,
            )
            self._using_macmon = False
            log.warning(
                "macmon not found — falling back to vm_stat/psutil polling "
                "(install macmon for GPU/temp/power metrics)"
            )
        self._macmon_thread.start()

        # Thread 2: MLX allocator snapshot poll
        self._mlx_thread = threading.Thread(
            target=self._mlx_poll_loop,
            name="monitor-mlx",
            daemon=True,
        )
        self._mlx_thread.start()

        log.info("MemoryMonitor started (2 background threads)")

    def stop(self) -> None:
        """Stop background threads and kill macmon subprocess."""
        if not self._started:
            return
        self._stop_event.set()

        # Kill macmon subprocess
        if self._macmon_proc is not None:
            try:
                self._macmon_proc.terminate()
                self._macmon_proc.wait(timeout=2)
            except Exception:
                try:
                    self._macmon_proc.kill()
                except Exception:
                    pass
            self._macmon_proc = None

        # Join threads
        for t in (self._macmon_thread, self._mlx_thread):
            if t is not None and t.is_alive():
                t.join(timeout=3)

        self._started = False
        log.info("MemoryMonitor stopped")

    @property
    def running(self) -> bool:
        return self._started and not self._stop_event.is_set()

    @property
    def using_macmon(self) -> bool:
        return self._using_macmon

    # ------------------------------------------------------------------
    #  Thread 1a: macmon persistent pipe  (primary, like exo)
    # ------------------------------------------------------------------

    def _macmon_pipe_loop(self) -> None:
        """
        Run ``macmon pipe -i <interval_ms>`` as a persistent subprocess and
        read JSON lines from its stdout.

        This is the exo pattern — see
        ``refs/exo/src/exo/utils/info_gatherer/info_gatherer.py`` L528-563.

        On pipe break / process exit, we restart macmon with a brief delay.
        """
        while not self._stop_event.is_set():
            try:
                cmd = [self._macmon_bin, "pipe", "-i", str(self._macmon_interval_ms)]
                log.debug(f"Spawning macmon: {' '.join(cmd)}")

                self._macmon_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # line-buffered
                )

                # Read lines until process exits or we're told to stop
                for line in self._macmon_proc.stdout:  # type: ignore[union-attr]
                    if self._stop_event.is_set():
                        break

                    line = line.strip()
                    if not line or not line.startswith("{"):
                        continue

                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        log.debug(f"macmon: invalid JSON line (skipped)")
                        continue

                    snap = MacmonSnapshot.from_macmon_json(raw)

                    # Apply OVERRIDE_MEMORY_MB if set
                    if self._override_bytes is not None:
                        override_avail = self._override_bytes
                        snap = MacmonSnapshot(
                            timestamp=snap.timestamp,
                            ram_total_bytes=snap.ram_total_bytes,
                            ram_usage_bytes=snap.ram_total_bytes - override_avail,
                            ram_available_bytes=override_avail,
                            swap_total_bytes=snap.swap_total_bytes,
                            swap_usage_bytes=snap.swap_usage_bytes,
                            ram_total_gb=snap.ram_total_gb,
                            ram_usage_gb=round(
                                (snap.ram_total_bytes - override_avail) / _GiB, 2
                            ),
                            ram_available_gb=round(override_avail / _GiB, 3),
                            ram_pct=round(
                                (snap.ram_total_bytes - override_avail)
                                / snap.ram_total_bytes
                                * 100,
                                1,
                            )
                            if snap.ram_total_bytes > 0
                            else 0.0,
                            gpu_freq_mhz=snap.gpu_freq_mhz,
                            gpu_freq_ratio=snap.gpu_freq_ratio,
                            gpu_power_w=snap.gpu_power_w,
                            ecpu_freq_mhz=snap.ecpu_freq_mhz,
                            ecpu_usage=snap.ecpu_usage,
                            pcpu_freq_mhz=snap.pcpu_freq_mhz,
                            pcpu_usage=snap.pcpu_usage,
                            cpu_power_w=snap.cpu_power_w,
                            gpu_temp_c=snap.gpu_temp_c,
                            cpu_temp_c=snap.cpu_temp_c,
                            sys_power_w=snap.sys_power_w,
                            ane_power_w=snap.ane_power_w,
                            gpu_activity_pct=snap.gpu_activity_pct,
                        )

                    self.latest_macmon = snap  # atomic assignment

                # Process exited — check why
                rc = self._macmon_proc.wait()
                if rc != 0 and not self._stop_event.is_set():
                    stderr = ""
                    if self._macmon_proc.stderr:
                        stderr = self._macmon_proc.stderr.read().strip()
                    log.warning(
                        f"macmon exited with code {rc}"
                        + (f": {stderr}" if stderr else "")
                        + " — restarting in 2s"
                    )

            except Exception as e:
                if not self._stop_event.is_set():
                    log.warning(f"macmon pipe error: {e} — restarting in 2s")

            # Wait before restart (unless we're shutting down)
            self._stop_event.wait(timeout=2.0)

    # ------------------------------------------------------------------
    #  Thread 1b: fallback vm_stat/psutil poll  (when macmon unavailable)
    # ------------------------------------------------------------------

    def _fallback_poll_loop(self) -> None:
        """Poll vm_stat/psutil when macmon is not installed."""
        log.debug("Fallback memory poll loop entering")

        while not self._stop_event.is_set():
            try:
                sys_mem = _probe_system_memory_vmstat()

                # Build a minimal MacmonSnapshot from vm_stat data
                # (no GPU/temp/power — those require macmon)
                total_bytes = int(sys_mem.get("total_gb", 0) * _GiB)
                used_bytes = int(sys_mem.get("used_gb", 0) * _GiB)
                avail_bytes = int(sys_mem.get("available_gb", 0) * _GiB)

                snap = MacmonSnapshot(
                    timestamp=time.time(),
                    ram_total_bytes=total_bytes,
                    ram_usage_bytes=used_bytes,
                    ram_available_bytes=avail_bytes,
                    ram_total_gb=round(total_bytes / _GiB, 2),
                    ram_usage_gb=round(used_bytes / _GiB, 2),
                    ram_available_gb=round(avail_bytes / _GiB, 3),
                    ram_pct=round(sys_mem.get("pressure_pct", 0), 1),
                )
                self.latest_macmon = snap

            except Exception as e:
                log.warning(f"Fallback memory poll error: {e}")

            self._stop_event.wait(timeout=_FALLBACK_POLL_INTERVAL)

    # ------------------------------------------------------------------
    #  Thread 2: MLX allocator poll
    # ------------------------------------------------------------------

    def _mlx_poll_loop(self) -> None:
        """Poll MemoryManager.snapshot() and build history records."""
        log.debug("MLX poll loop entering")

        while not self._stop_event.is_set():
            try:
                # MLX snapshot (calls mx.get_active_memory etc.)
                mlx_snap = self._manager.snapshot()
                self.latest_mlx = mlx_snap  # atomic assignment

                # Build combined pressure record from macmon + MLX
                hw = self.latest_macmon  # may be None in first few ms
                record = PressureRecord(
                    timestamp=time.time(),
                    mlx_active_gb=mlx_snap.active_gb,
                    mlx_pressure_pct=mlx_snap.pressure_pct,
                    system_used_gb=hw.ram_usage_gb if hw else 0.0,
                    system_available_gb=hw.ram_available_gb if hw else 0.0,
                    system_pressure_pct=hw.ram_pct if hw else 0.0,
                    model_loaded=mlx_snap.model_loaded,
                    gpu_power_w=hw.gpu_power_w if hw else 0.0,
                    gpu_temp_c=hw.gpu_temp_c if hw else 0.0,
                )
                self.history.append(record)

                # Check critical pressure threshold
                self._check_critical(record)

            except Exception as e:
                log.warning(f"MLX poll error (will retry): {e}")

            self._stop_event.wait(timeout=self._mlx_poll_interval)

    def _check_critical(self, record: PressureRecord) -> None:
        """Invoke on_critical callback after consecutive critical readings."""
        if record.mlx_pressure_pct >= self._critical_pct:
            self._consecutive_critical += 1
            if (
                self._consecutive_critical >= self._CRITICAL_CALLBACK_AFTER
                and self._on_critical is not None
            ):
                try:
                    self._on_critical(record)
                except Exception as e:
                    log.error(f"on_critical callback failed: {e}")
        else:
            self._consecutive_critical = 0

    # ------------------------------------------------------------------
    #  Public queries
    # ------------------------------------------------------------------

    @property
    def peak_pressure_1m(self) -> float:
        """Max MLX pressure % in the history window."""
        if not self.history:
            return 0.0
        return max(r.mlx_pressure_pct for r in self.history)

    @property
    def avg_pressure_1m(self) -> float:
        """Average MLX pressure % in the history window."""
        if not self.history:
            return 0.0
        return sum(r.mlx_pressure_pct for r in self.history) / len(self.history)

    @property
    def peak_system_pressure_1m(self) -> float:
        """Max system RAM usage % in the history window."""
        if not self.history:
            return 0.0
        return max(r.system_pressure_pct for r in self.history)

    @property
    def min_available_gb_1m(self) -> float:
        """Min system available GB in the history window."""
        if not self.history:
            return 0.0
        return min(r.system_available_gb for r in self.history)

    def hardware_dict(self) -> Optional[Dict[str, Any]]:
        """
        Return the latest macmon data in the same dict format that
        ``HardwarePoller._parse()`` produces.

        Returns ``None`` if no macmon data is available yet.
        The dashboard can call this instead of running its own macmon
        subprocess for the local node.
        """
        hw = self.latest_macmon
        if hw is None:
            return None
        return hw.to_hardware_dict()

    def history_dicts(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return history as a list of JSON-serializable dicts."""
        records = list(self.history)
        if last_n is not None:
            records = records[-last_n:]
        return [
            {
                "timestamp": round(r.timestamp, 3),
                "mlx_active_gb": round(r.mlx_active_gb, 3),
                "mlx_pressure_pct": round(r.mlx_pressure_pct, 1),
                "system_used_gb": round(r.system_used_gb, 2),
                "system_available_gb": round(r.system_available_gb, 3),
                "system_pressure_pct": round(r.system_pressure_pct, 1),
                "model_loaded": r.model_loaded,
                "gpu_power_w": round(r.gpu_power_w, 1),
                "gpu_temp_c": round(r.gpu_temp_c),
            }
            for r in records
        ]

    def summary(self) -> Dict[str, Any]:
        """
        JSON-ready dict with current state + rolling stats.

        Suitable for ``GET /memory/live`` and SSE events.
        """
        mlx = self.latest_mlx
        hw = self.latest_macmon

        result: Dict[str, Any] = {
            "monitor_running": self.running,
            "source": "macmon" if self._using_macmon else "vm_stat",
            "macmon_interval_ms": self._macmon_interval_ms
            if self._using_macmon
            else None,
            "mlx_poll_interval_s": self._mlx_poll_interval,
            "history_size": len(self.history),
            "memory_threshold": self.memory_threshold,
            "thresholds": self._pressure_thresholds,
        }

        if mlx is not None:
            result["mlx"] = {
                "active_gb": round(mlx.active_gb, 3),
                "peak_gb": round(mlx.peak_gb, 3),
                "cache_gb": round(mlx.cache_gb, 3),
                "limit_gb": round(mlx.limit_gb, 2),
                "pressure_pct": round(mlx.pressure_pct, 1),
                "model_loaded": mlx.model_loaded,
                "model_id": mlx.model_id,
            }

        if hw is not None:
            result["system"] = {
                "ram_total_gb": hw.ram_total_gb,
                "ram_used_gb": hw.ram_usage_gb,
                "ram_available_gb": hw.ram_available_gb,
                "ram_pct": hw.ram_pct,
                "gpu_power_w": hw.gpu_power_w,
                "gpu_temp_c": hw.gpu_temp_c,
                "cpu_temp_c": hw.cpu_temp_c,
                "sys_power_w": hw.sys_power_w,
                "gpu_activity_pct": hw.gpu_activity_pct,
            }

        result["rolling"] = {
            "peak_mlx_pressure_pct": round(self.peak_pressure_1m, 1),
            "avg_mlx_pressure_pct": round(self.avg_pressure_1m, 1),
            "peak_system_pressure_pct": round(self.peak_system_pressure_1m, 1),
            "min_available_gb": round(self.min_available_gb_1m, 3),
        }

        override = os.environ.get(_OVERRIDE_MEMORY_MB_ENV)
        if override:
            result["override_memory_mb"] = int(override)

        return result

    # ------------------------------------------------------------------
    #  Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        mlx = self.latest_mlx
        hw = self.latest_macmon
        parts = [f"running={self.running}"]
        parts.append(f"source={'macmon' if self._using_macmon else 'vm_stat'}")
        if mlx:
            parts.append(
                f"mlx={mlx.active_gb:.2f}/{mlx.limit_gb:.1f}GB "
                f"({mlx.pressure_pct:.1f}%)"
            )
        if hw:
            parts.append(
                f"ram={hw.ram_usage_gb:.1f}/{hw.ram_total_gb:.0f}GB ({hw.ram_pct:.0f}%)"
            )
            parts.append(f"gpu={hw.gpu_power_w:.1f}W/{hw.gpu_temp_c:.0f}°C")
        parts.append(f"peak_1m={self.peak_pressure_1m:.1f}%")
        return f"MemoryMonitor({', '.join(parts)})"
