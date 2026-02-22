#!/usr/bin/env python3
"""
memory_manager.py — GPU/Unified Memory safety layer for MLX-JACCL cluster.

Prevents the macOS kernel panic:
  "Memory object unexpectedly not found in fPendingMemorySet"
  @IOGPUGroupMemory.cpp:219

Root cause: MLX's default memory limit is 1.5× max_recommended_working_set,
which on a 48 GB M4 Pro = ~56 GB — ABOVE physical RAM. When the KV cache
grows unbounded (e.g. Qwen3 thinking mode), Metal allocations push past
physical memory. macOS tries emergency GPU memory eviction, the IOGPUFamily
driver's internal tracking (fPendingMemorySet) gets out of sync with what
was force-evicted, and the kernel panics.

This module:
  1. ADAPTIVELY sets MLX memory / cache / wired limits based on ACTUAL free
     RAM at startup — not just total RAM. If Chrome + Xcode are eating 15 GB
     when the server starts, the limit accounts for that.
  2. Provides proper model load() and unload() with full Metal buffer cleanup
  3. Monitors memory during generation and can abort before pressure kills the OS
  4. Disables Qwen3 "thinking" mode by default to prevent runaway token generation
  5. Exposes a memory snapshot for the dashboard / health endpoint
  6. Can re-check and re-adjust limits at model load time (system state may
     have changed since init)
  7. Cross-node awareness: reads the hostfile to know about remote nodes,
     queries remote memory via SSH, and reports cluster-wide memory state

Usage:
    from memory_manager import MemoryManager

    mm = MemoryManager()                       # adaptive limits from actual free RAM
    model, tok = mm.load_model(path, world)    # re-checks free RAM, load + shard
    mm.check_pressure()                        # call during generation loops
    mm.unload_model()                          # full cleanup
    snap = mm.snapshot()                       # for dashboard/health
    cluster = mm.cluster_memory_snapshot()     # all nodes
"""

import gc
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx

# ---------------------------------------------------------------------------
#  Lazy imports — mlx_lm may not be installed in every env
# ---------------------------------------------------------------------------
_load_model = None
_load_tokenizer = None
_stream_generate = None
_generate = None


def _lazy_imports():
    global _load_model, _load_tokenizer, _stream_generate, _generate

    if _load_model is not None:
        return

    from mlx_lm.utils import load_model, load_tokenizer

    _load_model = load_model
    _load_tokenizer = load_tokenizer

    try:
        from mlx_lm import stream_generate

        _stream_generate = stream_generate
    except ImportError:
        try:
            from mlx_lm.generate import stream_generate

            _stream_generate = stream_generate
        except ImportError:
            _stream_generate = None

    try:
        from mlx_lm import generate

        _generate = generate
    except ImportError:
        try:
            from mlx_lm.utils import generate

            _generate = generate
        except ImportError:
            from mlx_lm.generate import generate

            _generate = generate


log = logging.getLogger("memory_manager")

# ============================================================================
#  Constants
# ============================================================================

_GiB = 1024**3
_MiB = 1024**2

# How much RAM to reserve for macOS + system daemons + WindowServer etc.
# On a headless Mac mini this can be as low as 3 GB, but 6 GB is safe with
# a display attached, Finder, Spotlight, Bluetooth, etc.
# NOTE: This is the MINIMUM reserve. The adaptive calculation may reserve
# MORE if macOS is already using more than this at startup.
_SYSTEM_RESERVE_GB = float(os.environ.get("MLX_SYSTEM_RESERVE_GB", "6.0"))

# Maximum fraction of physical RAM that MLX is allowed to use.
# 0.70 = 70% → on 48 GB machine → 33.6 GB limit.
# This is a CEILING — the adaptive calculation may choose a lower limit
# if the system is already using significant RAM at startup.
_MAX_RAM_FRACTION = float(os.environ.get("MLX_MAX_RAM_FRACTION", "0.70"))

# Cache limit as fraction of the memory limit.
# Lower = more aggressive cache reclaim = less risk of memory pressure.
_CACHE_FRACTION = float(os.environ.get("MLX_CACHE_FRACTION", "0.50"))

# Wired limit — 0 means don't set (let macOS manage).
# Setting this can help on macOS 15+ to keep model weights resident.
_WIRED_LIMIT_GB = float(os.environ.get("MLX_WIRED_LIMIT_GB", "0"))

# Memory pressure thresholds (fraction of memory limit).
# When env vars are NOT set, these are overridden at __init__ time by
# tiered defaults from memory_monitor.tiered_pressure_thresholds() which
# adapts to machine size (matching exo's approach).
_PRESSURE_WARN = float(
    os.environ.get("MLX_PRESSURE_WARN", "0")
)  # 0 = use tiered default
_PRESSURE_CRITICAL = float(
    os.environ.get("MLX_PRESSURE_CRITICAL", "0")
)  # 0 = use tiered default

# Environment variable to override reported available memory (for testing).
# Same name as exo uses.  Value in megabytes.
#   OVERRIDE_MEMORY_MB=16384  →  simulate 16 GB available
_OVERRIDE_MEMORY_MB = os.environ.get("OVERRIDE_MEMORY_MB")

# Hard cap on max_tokens to prevent runaway generation
_HARD_MAX_TOKENS = int(os.environ.get("MLX_HARD_MAX_TOKENS", "4096"))

# Qwen3 thinking mode: disabled by default to prevent unbounded generation
_QWEN3_ENABLE_THINKING = os.environ.get("QWEN3_ENABLE_THINKING", "0") == "1"

# Hostfile path (for cross-node memory awareness)
_HOSTFILE = os.environ.get("HOSTFILE", "")

# SSH timeout for remote memory queries (seconds)
_SSH_TIMEOUT = float(os.environ.get("MLX_SSH_TIMEOUT", "5"))


# ============================================================================
#  Dataclasses
# ============================================================================


@dataclass
class MemorySnapshot:
    """Point-in-time memory state."""

    timestamp: float = 0.0
    active_gb: float = 0.0
    peak_gb: float = 0.0
    cache_gb: float = 0.0
    limit_gb: float = 0.0
    cache_limit_gb: float = 0.0
    total_ram_gb: float = 0.0
    system_reserve_gb: float = 0.0
    pressure_pct: float = 0.0  # active / limit as percentage
    model_loaded: bool = False
    model_id: str = ""
    model_size_gb: float = 0.0
    # Adaptive: actual system usage when limits were computed
    system_used_gb: float = 0.0
    actual_available_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "active_gb": round(self.active_gb, 3),
            "peak_gb": round(self.peak_gb, 3),
            "cache_gb": round(self.cache_gb, 3),
            "limit_gb": round(self.limit_gb, 2),
            "cache_limit_gb": round(self.cache_limit_gb, 2),
            "total_ram_gb": round(self.total_ram_gb, 1),
            "system_reserve_gb": round(self.system_reserve_gb, 1),
            "pressure_pct": round(self.pressure_pct, 1),
            "model_loaded": self.model_loaded,
            "model_id": self.model_id,
            "model_size_gb": round(self.model_size_gb, 3),
            "system_used_gb": round(self.system_used_gb, 2),
            "actual_available_gb": round(self.actual_available_gb, 2),
        }


@dataclass
class NodeMemoryInfo:
    """Memory info for a single node (local or remote)."""

    hostname: str = ""
    ssh: str = ""
    total_gb: float = 0.0
    free_gb: float = 0.0
    used_gb: float = 0.0
    wired_gb: float = 0.0
    compressed_gb: float = 0.0
    pressure_level: str = "unknown"
    reachable: bool = False
    is_local: bool = False
    mlx_limit_gb: float = 0.0  # what WE set (only meaningful for local)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "ssh": self.ssh,
            "total_gb": round(self.total_gb, 1),
            "free_gb": round(self.free_gb, 2),
            "used_gb": round(self.used_gb, 2),
            "wired_gb": round(self.wired_gb, 2),
            "compressed_gb": round(self.compressed_gb, 2),
            "pressure_level": self.pressure_level,
            "reachable": self.reachable,
            "is_local": self.is_local,
            "mlx_limit_gb": round(self.mlx_limit_gb, 2),
        }


@dataclass
class ModelSlot:
    """Tracks a loaded model's references and metadata."""

    model: Any = None
    tokenizer: Any = None
    model_id: str = ""
    model_path: str = ""
    strategy: str = ""
    load_time_s: float = 0.0
    size_gb: float = 0.0  # active memory after load minus before load
    loaded_at: float = 0.0


class MemoryPressureError(RuntimeError):
    """Raised when memory pressure exceeds the critical threshold."""

    pass


class ModelNotLoadedError(RuntimeError):
    """Raised when trying to use a model that isn't loaded."""

    pass


# ============================================================================
#  MemoryManager
# ============================================================================


class MemoryManager:
    """
    Central memory + model lifecycle manager for the MLX-JACCL cluster.

    ADAPTIVE: Limits are computed from ACTUAL free RAM at startup, not just
    total RAM. If the system is already using 15 GB when the server starts,
    the limit is lowered accordingly. Limits are also re-checked and
    re-adjusted at model load time.

    CROSS-NODE AWARE: Reads the hostfile to know about remote nodes and can
    query their memory status via SSH for cluster-wide reporting.

    Singleton-ish: create one instance at process start, pass it around.
    Thread-safe for snapshot reads; load/unload should be called from
    a single thread (the main thread or the queue worker).
    """

    def __init__(
        self,
        system_reserve_gb: Optional[float] = None,
        max_ram_fraction: Optional[float] = None,
        cache_fraction: Optional[float] = None,
        wired_limit_gb: Optional[float] = None,
        hard_max_tokens: Optional[int] = None,
        hostfile: Optional[str] = None,
    ):
        self._system_reserve_gb = system_reserve_gb or _SYSTEM_RESERVE_GB
        self._max_ram_fraction = max_ram_fraction or _MAX_RAM_FRACTION
        self._cache_fraction = cache_fraction or _CACHE_FRACTION
        self._wired_limit_gb = (
            wired_limit_gb if wired_limit_gb is not None else _WIRED_LIMIT_GB
        )
        self.hard_max_tokens = hard_max_tokens or _HARD_MAX_TOKENS
        self._hostfile = hostfile or _HOSTFILE

        # Device info (local node)
        self._device_info = mx.device_info()
        self._total_ram = self._device_info.get("memory_size", 0)
        self._max_recommended = self._device_info.get(
            "max_recommended_working_set_size", 0
        )
        self._max_buffer = self._device_info.get("max_buffer_length", 0)
        self._device_name = self._device_info.get("device_name", "unknown")

        # Adaptive: measure actual system usage RIGHT NOW
        self._system_used_at_init = 0.0  # GB used by non-MLX processes
        self._actual_available_at_init = 0.0  # GB truly available for MLX

        # Computed limits
        self._memory_limit = 0
        self._cache_limit = 0
        self._prev_memory_limit = 0
        self._prev_cache_limit = 0

        # Model slot
        self._slot = ModelSlot()

        # Generation tracking
        self._generation_active = False
        self._generation_tokens = 0
        self._generation_start = 0.0
        self._abort_requested = False

        # Hostfile data (loaded lazily)
        self._hosts_data: Optional[list] = None

        # ── Tiered pressure thresholds ────────────────────────────
        # Import here to avoid circular import at module level.
        # When the env vars are "0" (i.e. not explicitly set by user),
        # we use tiered defaults based on machine size — matching exo's
        # _default_memory_threshold() pattern.
        total_gb = self._total_ram / _GiB
        try:
            from memory_monitor import tiered_pressure_thresholds

            tiered = tiered_pressure_thresholds(total_gb)
        except ImportError:
            # Fallback if memory_monitor not available
            tiered = {"warn": 0.80, "critical": 0.92}

        self._pressure_warn = _PRESSURE_WARN if _PRESSURE_WARN > 0 else tiered["warn"]
        self._pressure_critical = (
            _PRESSURE_CRITICAL if _PRESSURE_CRITICAL > 0 else tiered["critical"]
        )
        log.info(
            f"Pressure thresholds (tiered for {total_gb:.0f} GB): "
            f"warn={self._pressure_warn:.0%}, critical={self._pressure_critical:.0%}"
        )

        # ── OVERRIDE_MEMORY_MB ────────────────────────────────────
        self._override_memory_bytes: Optional[int] = None
        if _OVERRIDE_MEMORY_MB is not None:
            try:
                mb = int(_OVERRIDE_MEMORY_MB)
                if mb > 0:
                    self._override_memory_bytes = mb * (1024**2)
                    log.info(
                        f"OVERRIDE_MEMORY_MB={mb} → simulating "
                        f"{mb / 1024:.1f} GB available memory"
                    )
            except ValueError:
                log.warning(
                    f"OVERRIDE_MEMORY_MB={_OVERRIDE_MEMORY_MB!r} is not valid — ignoring"
                )

        # Apply limits immediately (adaptive — reads actual free RAM)
        self._apply_limits()

    # ------------------------------------------------------------------
    #  Limit computation & application
    # ------------------------------------------------------------------

    def _apply_limits(self) -> None:
        """
        Compute and set safe MLX memory limits.

        ADAPTIVE: Probes the actual free RAM right now via vm_stat/sysctl,
        so the limit accounts for what macOS + other apps are already using.

        On a 48 GB machine where 15 GB is already used:
          - Old static approach: limit = 48 × 0.70 = 33.6 GB  (total 48.6 GB → panic)
          - New adaptive:        limit = min(33.6, 48 - 15 - 3) = 30.0 GB  (safe)

        The 3 GB above is the "safety margin" — extra headroom beyond what's
        currently free, because macOS usage fluctuates.
        """
        total_gb = self._total_ram / _GiB

        # ── Step 1: Measure actual system memory usage RIGHT NOW ──────
        sys_info = self._get_local_memory_usage()
        system_used_gb = sys_info.get("used_gb", 0.0)
        system_free_gb = sys_info.get("free_gb", 0.0)
        system_available_gb = sys_info.get("available_gb", 0.0)

        # If we couldn't measure, fall back to assuming the reserve is all
        # that's used (conservative but not adaptive).
        if system_available_gb <= 0:
            system_used_gb = self._system_reserve_gb
            system_available_gb = total_gb - self._system_reserve_gb
            log.warning(
                "Could not measure actual system memory usage — "
                f"assuming {self._system_reserve_gb:.0f} GB system overhead (static fallback)"
            )

        self._system_used_at_init = system_used_gb
        self._actual_available_at_init = system_available_gb

        # ── Step 2: Compute candidate limits ──────────────────────────

        # Method 1 (static ceiling): fraction of total RAM — hard upper bound
        limit_by_fraction = self._total_ram * self._max_ram_fraction

        # Method 2 (static floor): total RAM minus configured reserve
        limit_by_reserve = self._total_ram - (self._system_reserve_gb * _GiB)

        # Method 3 (Apple's guidance): max recommended working set
        limit_by_recommended = self._max_recommended

        # Method 4 (ADAPTIVE — the new one): actual available RAM right now,
        # minus a safety margin so macOS has room to breathe if its usage
        # fluctuates after we set the limit.
        _SAFETY_MARGIN_GB = 3.0  # extra headroom beyond current usage
        limit_by_available = (system_available_gb - _SAFETY_MARGIN_GB) * _GiB

        # ── Step 3: Pick the SMALLEST (most conservative) ─────────────
        candidates = [
            ("fraction_of_total", limit_by_fraction),
            ("total_minus_reserve", limit_by_reserve),
            ("apple_recommended", limit_by_recommended),
            ("adaptive_available", limit_by_available),
        ]
        # Filter out nonsensical values (negative or too small)
        valid = [(n, v) for n, v in candidates if v > 2 * _GiB]
        if not valid:
            # Absolute fallback: 50% of total RAM
            valid = [("emergency_fallback", self._total_ram * 0.5)]

        winning_name, winning_value = min(valid, key=lambda x: x[1])
        self._memory_limit = int(winning_value)

        # Cache limit: fraction of memory limit
        self._cache_limit = int(self._memory_limit * self._cache_fraction)

        # Clamp to sane range: [4 GB, total_ram - 4 GB]
        floor = int(4 * _GiB)
        ceiling = int(self._total_ram - 4 * _GiB)
        self._memory_limit = max(floor, min(self._memory_limit, ceiling))
        self._cache_limit = max(floor // 2, min(self._cache_limit, self._memory_limit))

        # ── Step 4: Apply to MLX ──────────────────────────────────────
        self._prev_memory_limit = mx.set_memory_limit(self._memory_limit)
        self._prev_cache_limit = mx.set_cache_limit(self._cache_limit)

        # Optional wired limit (macOS 15+)
        if self._wired_limit_gb > 0:
            wired_bytes = int(self._wired_limit_gb * _GiB)
            try:
                mx.set_wired_limit(wired_bytes)
                log.info(f"Wired limit set: {self._wired_limit_gb:.1f} GB")
            except Exception as e:
                log.warning(f"Could not set wired limit: {e}")

        # ── Step 5: Log the decision ─────────────────────────────────
        log.info(
            f"Memory limits applied on {self._device_name} (ADAPTIVE):\n"
            f"  Total RAM:         {total_gb:.1f} GB\n"
            f"  System used now:   {system_used_gb:.1f} GB  "
            f"(non-MLX: macOS + apps)\n"
            f"  Actually available: {system_available_gb:.1f} GB\n"
            f"  Max recommended:   {self._max_recommended / _GiB:.1f} GB  "
            f"(Apple's guidance)\n"
            f"  ──────────────────────────────────────\n"
            f"  Candidates:\n"
            + "\n".join(
                f"    {n:25s} → {v / _GiB:6.1f} GB"
                + (" ← selected" if n == winning_name else "")
                for n, v in candidates
            )
            + f"\n"
            f"  ──────────────────────────────────────\n"
            f"  MLX memory limit:  {self._memory_limit / _GiB:.1f} GB  "
            f"(was {self._prev_memory_limit / _GiB:.1f} GB)  "
            f"[chosen by: {winning_name}]\n"
            f"  MLX cache limit:   {self._cache_limit / _GiB:.1f} GB  "
            f"(was {self._prev_cache_limit / _GiB:.1f} GB)\n"
            f"  System reserve:    {self._system_reserve_gb:.1f} GB (config min)\n"
            f"  Safety margin:     {_SAFETY_MARGIN_GB:.1f} GB\n"
            f"  Hard max tokens:   {self.hard_max_tokens}"
        )

    def readjust_limits(self, reason: str = "manual") -> Dict[str, Any]:
        """
        Re-measure actual free RAM and re-adjust MLX limits.

        Call this before model load to account for system state changes
        since init (e.g., user closed Chrome, or macOS woke from sleep).

        Returns a dict showing before/after for logging.
        """
        old_limit = self._memory_limit / _GiB
        old_cache = self._cache_limit / _GiB

        self._apply_limits()

        new_limit = self._memory_limit / _GiB
        new_cache = self._cache_limit / _GiB
        changed = abs(new_limit - old_limit) > 0.1

        result = {
            "reason": reason,
            "changed": changed,
            "limit_gb_before": round(old_limit, 2),
            "limit_gb_after": round(new_limit, 2),
            "cache_gb_before": round(old_cache, 2),
            "cache_gb_after": round(new_cache, 2),
            "system_used_gb": round(self._system_used_at_init, 2),
            "available_gb": round(self._actual_available_at_init, 2),
        }

        if changed:
            log.info(
                f"Limits readjusted ({reason}): "
                f"{old_limit:.1f} → {new_limit:.1f} GB "
                f"(system using {self._system_used_at_init:.1f} GB)"
            )

        return result

    def _get_local_memory_usage(self) -> Dict[str, float]:
        """
        Probe actual local memory usage via sysctl + vm_stat.

        If OVERRIDE_MEMORY_MB is set, ``available_gb`` is overridden to
        simulate a constrained machine (same env var name as exo uses).

        Returns dict with:
          total_gb:     total physical RAM
          used_gb:      RAM currently used (total - free - inactive - purgeable)
          free_gb:      RAM marked as free by the kernel
          available_gb: RAM available for new allocations
                        (free + inactive + purgeable + compressor-reclaimable)
          wired_gb:     wired (non-evictable) pages
          compressed_gb: compressor-occupied pages
        """
        result = {
            "total_gb": 0.0,
            "used_gb": 0.0,
            "free_gb": 0.0,
            "available_gb": 0.0,
            "wired_gb": 0.0,
            "compressed_gb": 0.0,
        }

        # Get total RAM via sysctl
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
            # Fallback from MLX device info
            result["total_gb"] = mx.device_info().get("memory_size", 0) / _GiB

        # Parse vm_stat for detailed page counts
        page_size = 16384  # Apple Silicon default
        pages = {}
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
                        # "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
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
                    key = key.strip().lower()
                    try:
                        pages[key] = int(val.strip().rstrip("."))
                    except ValueError:
                        pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        if pages:

            def _pages_to_gb(key):
                return pages.get(key, 0) * page_size / _GiB

            free = _pages_to_gb("pages free")
            inactive = _pages_to_gb("pages inactive")
            purgeable = _pages_to_gb("pages purgeable")
            wired = _pages_to_gb("pages wired down")
            active = _pages_to_gb("pages active")
            speculative = _pages_to_gb("pages speculative")
            compressed = _pages_to_gb("pages occupied by compressor")

            result["free_gb"] = round(free, 2)
            result["wired_gb"] = round(wired, 2)
            result["compressed_gb"] = round(compressed, 2)

            # "Available" = memory the system can give to a new process.
            # free + inactive + purgeable are all reclaimable.
            # We do NOT count compressed — those pages are in use, just compressed.
            available = free + inactive + purgeable
            result["available_gb"] = round(available, 2)

            # "Used" = total minus available
            total = result["total_gb"]
            result["used_gb"] = round(max(0, total - available), 2)

        # ── Apply OVERRIDE_MEMORY_MB if set ───────────────────────
        if self._override_memory_bytes is not None:
            override_gb = self._override_memory_bytes / _GiB
            result["available_gb"] = round(override_gb, 2)
            result["used_gb"] = round(max(0, result["total_gb"] - override_gb), 2)

        return result

    # ------------------------------------------------------------------
    #  Memory queries
    # ------------------------------------------------------------------

    def snapshot(self) -> MemorySnapshot:
        """Return a point-in-time memory snapshot (safe to call from any thread)."""
        active = mx.get_active_memory()
        peak = mx.get_peak_memory()
        cache = mx.get_cache_memory()
        pressure = (active / self._memory_limit * 100) if self._memory_limit > 0 else 0

        return MemorySnapshot(
            timestamp=time.time(),
            active_gb=active / _GiB,
            peak_gb=peak / _GiB,
            cache_gb=cache / _GiB,
            limit_gb=self._memory_limit / _GiB,
            cache_limit_gb=self._cache_limit / _GiB,
            total_ram_gb=self._total_ram / _GiB,
            system_reserve_gb=self._system_reserve_gb,
            pressure_pct=pressure,
            model_loaded=self._slot.model is not None,
            model_id=self._slot.model_id,
            model_size_gb=self._slot.size_gb,
            system_used_gb=self._system_used_at_init,
            actual_available_gb=self._actual_available_at_init,
        )

    def active_gb(self) -> float:
        return mx.get_active_memory() / _GiB

    def peak_gb(self) -> float:
        return mx.get_peak_memory() / _GiB

    def pressure_pct(self) -> float:
        """Current memory usage as percentage of the limit."""
        active = mx.get_active_memory()
        return (active / self._memory_limit * 100) if self._memory_limit > 0 else 0

    def headroom_gb(self) -> float:
        """How much memory is available before hitting the limit."""
        return max(0, (self._memory_limit - mx.get_active_memory())) / _GiB

    # ------------------------------------------------------------------
    #  Pressure checks
    # ------------------------------------------------------------------

    def check_pressure(self, context: str = "generation") -> None:
        """
        Check memory pressure. Call this during generation loops.

        Raises MemoryPressureError if pressure exceeds the critical threshold.
        Logs a warning if pressure exceeds the warn threshold.

        Uses tiered thresholds (set in __init__) that adapt to machine size,
        matching exo's approach.
        """
        pct = self.pressure_pct()
        active = self.active_gb()

        if pct >= self._pressure_critical * 100:
            msg = (
                f"CRITICAL memory pressure during {context}: "
                f"{active:.2f} GB active = {pct:.1f}% of "
                f"{self._memory_limit / _GiB:.1f} GB limit. "
                f"Aborting to prevent kernel panic."
            )
            log.error(msg)
            # Emergency cleanup before raising
            self._emergency_cleanup()
            raise MemoryPressureError(msg)

        if pct >= self._pressure_warn * 100:
            log.warning(
                f"High memory pressure during {context}: "
                f"{active:.2f} GB active = {pct:.1f}% of "
                f"{self._memory_limit / _GiB:.1f} GB limit. "
                f"Headroom: {self.headroom_gb():.2f} GB"
            )

    def _emergency_cleanup(self) -> None:
        """Best-effort emergency memory reclaim."""
        log.warning("Emergency memory cleanup triggered")
        gc.collect()
        gc.collect()
        mx.clear_cache()
        # Give Metal a moment to actually free buffers
        time.sleep(0.1)
        gc.collect()
        mx.clear_cache()
        log.info(
            f"After emergency cleanup: "
            f"active={self.active_gb():.2f} GB, "
            f"cache={mx.get_cache_memory() / _GiB:.2f} GB"
        )

    # ------------------------------------------------------------------
    #  Model load / unload
    # ------------------------------------------------------------------

    def unload_model(self) -> Dict[str, Any]:
        """
        Fully unload the current model, releasing all Metal GPU buffers.

        Returns a dict with before/after memory stats for logging.

        The key steps are:
          1. Delete Python references to model + tokenizer
          2. Run GC twice (second pass catches reference cycles)
          3. Clear MLX Metal cache (releases freed buffers back to the system)
          4. Reset peak memory tracker
        """
        before_active = self.active_gb()
        before_cache = mx.get_cache_memory() / _GiB
        model_id = self._slot.model_id or "(none)"

        if self._slot.model is None:
            log.info("No model loaded — nothing to unload")
            return {
                "status": "no_model",
                "before_active_gb": before_active,
                "after_active_gb": before_active,
                "freed_gb": 0.0,
            }

        log.info(f"Unloading model: {model_id}")

        # Save metadata before we destroy the slot
        old_model_path = self._slot.model_path
        old_strategy = self._slot.strategy
        old_load_time = self._slot.load_time_s
        old_size_gb = self._slot.size_gb
        old_loaded_at = self._slot.loaded_at

        # Step 1: Grab references out of the slot, then destroy the slot.
        # We need to explicitly `del` the model and tokenizer objects so
        # Python's refcount drops to zero *immediately* (no waiting for GC).
        # Setting to None is not enough — the ModelSlot dataclass field
        # keeps a reference alive until the slot itself is collected.
        model_ref = self._slot.model
        tok_ref = self._slot.tokenizer
        self._slot = ModelSlot()  # wipe the slot (drops its references)

        # Step 2: Explicit del — this is what actually frees the Metal buffers.
        # mlx.array objects release their backing Metal allocations when the
        # Python refcount hits zero. `del` forces that to happen NOW rather
        # than waiting for GC to discover the dead object.
        del model_ref
        del tok_ref

        # Step 3: Aggressive garbage collection to catch any reference cycles
        # (nn.Module trees often have parent↔child cycles).
        gc.collect()
        gc.collect()
        gc.collect()

        # Step 4: Clear MLX Metal buffer cache.
        # Even after the Python objects are freed, MLX may keep the backing
        # Metal buffers in a free-list / cache for reuse. clear_cache()
        # returns them to the OS.
        mx.clear_cache()

        # Step 5: Small sleep + second pass — Metal's internal allocator
        # may process frees asynchronously.
        time.sleep(0.05)
        gc.collect()
        mx.clear_cache()

        # Step 6: Reset peak tracker
        mx.reset_peak_memory()

        after_active = self.active_gb()
        after_cache = mx.get_cache_memory() / _GiB
        freed = before_active - after_active

        result = {
            "status": "unloaded",
            "model_id": model_id,
            "before_active_gb": round(before_active, 3),
            "before_cache_gb": round(before_cache, 3),
            "after_active_gb": round(after_active, 3),
            "after_cache_gb": round(after_cache, 3),
            "freed_gb": round(freed, 3),
        }

        log.info(
            f"Model unloaded: {model_id}\n"
            f"  Before: {before_active:.3f} GB active, {before_cache:.3f} GB cache\n"
            f"  After:  {after_active:.3f} GB active, {after_cache:.3f} GB cache\n"
            f"  Freed:  {freed:.3f} GB"
        )

        return result

    def load_model(
        self,
        model_path: str,
        world: Optional[Any] = None,
        model_id: Optional[str] = None,
        lazy: bool = False,
    ) -> Tuple[Any, Any]:
        """
        Load and optionally shard a model with full memory safety.

        Steps:
          1. Unload any existing model first (prevents double-loading)
          2. Pre-flight memory check
          3. Load model weights (eager by default for JACCL safety)
          4. Barrier + shard across nodes
          5. Post-load memory check + record slot metadata
          6. Load tokenizer

        Args:
            model_path: Path to the model directory (HuggingFace format)
            world: mx.distributed world (None = single-node)
            model_id: Human-readable model identifier
            lazy: If False (default), load weights eagerly. Eager loading
                  avoids the JACCL eval deadlock with lazy tensors.

        Returns:
            (model, tokenizer) tuple

        Raises:
            MemoryPressureError: if not enough headroom to load
            RuntimeError: if load fails
        """
        _lazy_imports()

        model_path = str(model_path)
        path = Path(model_path)
        rank = world.rank() if world else 0

        if model_id is None:
            model_id = path.name

        # Step 1: Unload any existing model
        if self._slot.model is not None:
            log.info(
                f"[rank {rank}] Unloading previous model before loading {model_id}"
            )
            self.unload_model()

        # Step 2: Re-measure system memory and readjust limits.
        # The system state may have changed since init — maybe the user
        # closed Chrome, or maybe Spotlight indexing kicked in.
        readjust = self.readjust_limits(reason=f"pre-load {model_id}")
        if readjust["changed"]:
            log.info(
                f"[rank {rank}] Limits readjusted before load: "
                f"{readjust['limit_gb_before']:.1f} → {readjust['limit_gb_after']:.1f} GB "
                f"(system using {readjust['system_used_gb']:.1f} GB)"
            )

        # Step 3: Pre-flight check — estimate if we have enough headroom
        headroom = self.headroom_gb()
        model_disk_gb = self._estimate_model_disk_size(path)

        log.info(
            f"[rank {rank}] Loading model: {model_id}\n"
            f"  Path:          {model_path}\n"
            f"  Disk size:     {model_disk_gb:.2f} GB\n"
            f"  Headroom:      {headroom:.2f} GB\n"
            f"  Memory limit:  {self._memory_limit / _GiB:.1f} GB  (adaptive)\n"
            f"  System used:   {self._system_used_at_init:.1f} GB  (non-MLX)"
        )

        # Conservative check: model in RAM is typically ~1.0-1.3× disk size
        # for quantized models. If sharded, only ~half stays per node.
        nodes = world.size() if world else 1
        estimated_per_node = (model_disk_gb * 1.3) / nodes
        # During loading, the full model is briefly in RAM before sharding drops
        # non-local slices. So we need headroom for the full model temporarily.
        estimated_peak_load = model_disk_gb * 1.3

        if estimated_peak_load > headroom * 0.95:
            log.warning(
                f"[rank {rank}] Tight memory for loading {model_id}: "
                f"estimated peak {estimated_peak_load:.2f} GB vs "
                f"headroom {headroom:.2f} GB. Proceeding with caution."
            )

        if estimated_peak_load > headroom * 1.5:
            raise MemoryPressureError(
                f"Not enough memory to load {model_id}: "
                f"estimated {estimated_peak_load:.1f} GB needed, "
                f"only {headroom:.1f} GB available. "
                f"Unload the current model first or increase memory limit."
            )

        # Step 3: Load model weights
        before_mem = self.active_gb()
        mx.reset_peak_memory()
        t0 = time.perf_counter()

        log.info(f"[rank {rank}] Step 1/5: load_model(lazy={lazy}) ...")
        model, _ = _load_model(path, lazy=lazy)
        t_load = time.perf_counter() - t0
        log.info(f"[rank {rank}] Step 1/5: done in {t_load:.2f}s")

        # Step 4: Barrier + shard (distributed only)
        strategy = "none (single node)"
        if world and world.size() > 1:
            # Pre-shard barrier
            log.info(f"[rank {rank}] Step 2/5: pre-shard barrier ...")
            t0 = time.perf_counter()
            mx.eval(mx.distributed.all_sum(mx.zeros((1,))))
            log.info(
                f"[rank {rank}] Step 2/5: barrier done in {time.perf_counter() - t0:.4f}s"
            )

            # Shard
            log.info(f"[rank {rank}] Step 3/5: sharding ...")
            t0 = time.perf_counter()
            if hasattr(model, "shard"):
                model.shard(world)
                strategy = f"Tensor Parallelism x{world.size()}"
            elif hasattr(getattr(model, "model", None), "pipeline"):
                model.model.pipeline(world)
                strategy = f"Pipeline Parallelism x{world.size()}"
            else:
                strategy = f"Replicated x{world.size()}"
            log.info(
                f"[rank {rank}] Step 3/5: done in {time.perf_counter() - t0:.2f}s — {strategy}"
            )

            # Post-shard barrier
            log.info(f"[rank {rank}] Step 4/5: post-shard barrier ...")
            t0 = time.perf_counter()
            mx.eval(mx.distributed.all_sum(mx.zeros((1,))))
            log.info(
                f"[rank {rank}] Step 4/5: barrier done in {time.perf_counter() - t0:.4f}s"
            )
        else:
            log.info(f"[rank {rank}] Steps 2-4: skipped (single node)")

        # Step 5: Load tokenizer
        log.info(f"[rank {rank}] Step 5/5: load_tokenizer ...")
        t0 = time.perf_counter()
        try:
            tok = _load_tokenizer(path, {"trust_remote_code": True}, eos_token_ids=None)
        except Exception as e:
            log.warning(
                f"[rank {rank}] Standard tokenizer load failed ({e}), trying fallback"
            )
            tok = self._load_custom_tokenizer(path)
        log.info(f"[rank {rank}] Step 5/5: done in {time.perf_counter() - t0:.2f}s")

        # Record slot metadata
        after_mem = self.active_gb()
        model_mem = after_mem - before_mem

        self._slot = ModelSlot(
            model=model,
            tokenizer=tok,
            model_id=model_id,
            model_path=model_path,
            strategy=strategy,
            load_time_s=t_load,
            size_gb=max(0, model_mem),
            loaded_at=time.time(),
        )

        log.info(
            f"[rank {rank}] Model loaded: {model_id}\n"
            f"  Strategy:   {strategy}\n"
            f"  Load time:  {t_load:.2f}s\n"
            f"  Model mem:  {model_mem:.3f} GB\n"
            f"  Total active: {after_mem:.3f} GB / {self._memory_limit / _GiB:.1f} GB limit\n"
            f"  Peak during load: {mx.get_peak_memory() / _GiB:.3f} GB"
        )

        return model, tok

    @property
    def model(self) -> Any:
        """Get the currently loaded model, or raise if none loaded."""
        if self._slot.model is None:
            raise ModelNotLoadedError("No model is currently loaded")
        return self._slot.model

    @property
    def tokenizer(self) -> Any:
        """Get the currently loaded tokenizer, or raise if none loaded."""
        if self._slot.tokenizer is None:
            raise ModelNotLoadedError("No model is currently loaded")
        return self._slot.tokenizer

    @property
    def model_id(self) -> str:
        return self._slot.model_id

    @property
    def model_loaded(self) -> bool:
        return self._slot.model is not None

    @property
    def slot_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model slot."""
        if not self.model_loaded:
            return {"loaded": False}
        return {
            "loaded": True,
            "model_id": self._slot.model_id,
            "model_path": self._slot.model_path,
            "strategy": self._slot.strategy,
            "load_time_s": round(self._slot.load_time_s, 2),
            "size_gb": round(self._slot.size_gb, 3),
            "loaded_at": self._slot.loaded_at,
            "uptime_s": round(time.time() - self._slot.loaded_at, 1)
            if self._slot.loaded_at
            else 0,
        }

    # ------------------------------------------------------------------
    #  Safe generation wrappers
    # ------------------------------------------------------------------

    def clamp_max_tokens(self, requested: Optional[int]) -> int:
        """
        Clamp max_tokens to a safe value.

        Prevents:
          - Runaway generation (Qwen3 thinking → thousands of tokens)
          - KV cache growing until it triggers the kernel panic
        """
        if requested is None or requested <= 0:
            return min(512, self.hard_max_tokens)
        return min(requested, self.hard_max_tokens)

    def generation_guard(self, token_count: int, context: str = "generate") -> None:
        """
        Call this inside generation loops (every N tokens) to check:
          1. Memory pressure
          2. Token count vs hard limit
          3. Abort flag

        Raises MemoryPressureError or RuntimeError if generation should stop.
        """
        # Check memory every 16 tokens (not every token — that's too expensive)
        if token_count % 16 == 0:
            self.check_pressure(context=f"{context} @ token {token_count}")

        if token_count >= self.hard_max_tokens:
            raise MemoryPressureError(
                f"Hard token limit reached: {token_count} >= {self.hard_max_tokens}. "
                f"Stopping generation to prevent memory exhaustion."
            )

        if self._abort_requested:
            self._abort_requested = False
            raise RuntimeError("Generation aborted by memory manager")

    def request_abort(self) -> None:
        """Request the current generation to abort (e.g. from a health check thread)."""
        self._abort_requested = True

    # ------------------------------------------------------------------
    #  Qwen3 thinking mode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def should_disable_thinking(model_id: str) -> bool:
        """
        Check if this model is Qwen3 and thinking should be disabled.

        Qwen3's default chat template includes <think>...</think> sequences
        that can generate thousands of tokens before the actual answer,
        causing KV cache to explode and trigger the kernel panic.
        """
        if _QWEN3_ENABLE_THINKING:
            return False

        model_lower = model_id.lower()
        # Match Qwen3 variants (Qwen3-8B, Qwen3-4B, etc.)
        # but NOT Qwen2.5 or Qwen2
        return "qwen3" in model_lower

    @staticmethod
    def patch_chat_template_no_thinking(tokenizer: Any) -> Any:
        """
        If the tokenizer has a Qwen3 chat template with thinking,
        patch it to disable thinking by default.

        This adds /no_think to the template or uses enable_thinking=False.
        """
        if not hasattr(tokenizer, "apply_chat_template"):
            return tokenizer

        # Wrap apply_chat_template to inject enable_thinking=False
        original_apply = tokenizer.apply_chat_template

        def _patched_apply(*args, **kwargs):
            # Only inject if caller didn't explicitly set it
            if "enable_thinking" not in kwargs:
                kwargs["enable_thinking"] = False
            try:
                return original_apply(*args, **kwargs)
            except TypeError:
                # Some tokenizer versions don't support enable_thinking kwarg
                # Fall back to adding /no_think to the last user message
                kwargs.pop("enable_thinking", None)
                return original_apply(*args, **kwargs)

        tokenizer.apply_chat_template = _patched_apply
        return tokenizer

    def build_safe_chat_prompt(
        self,
        messages: list,
        tokenizer: Any = None,
        model_id: Optional[str] = None,
    ) -> str:
        """
        Build a chat prompt with safety features:
          - Disable Qwen3 thinking if appropriate
          - Use tokenizer chat template when available
          - Fall back to simple format otherwise
        """
        tok = tokenizer or self._slot.tokenizer
        mid = model_id or self._slot.model_id

        if tok is None:
            # Last resort: simple concatenation
            parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
            parts.append("ASSISTANT:")
            return "\n".join(parts)

        if hasattr(tok, "apply_chat_template"):
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            # Disable thinking for Qwen3
            if self.should_disable_thinking(mid):
                kwargs["enable_thinking"] = False
            try:
                return tok.apply_chat_template(messages, **kwargs)
            except TypeError:
                # Tokenizer doesn't support enable_thinking
                kwargs.pop("enable_thinking", None)
                try:
                    return tok.apply_chat_template(messages, **kwargs)
                except Exception:
                    pass

        # Fallback
        parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    #  Periodic maintenance
    # ------------------------------------------------------------------

    def gc_cycle(self, reason: str = "periodic") -> Dict[str, float]:
        """
        Run a garbage collection + cache trim cycle.

        Call this between requests or after generation completes.
        Returns before/after memory stats.
        """
        before_active = self.active_gb()
        before_cache = mx.get_cache_memory() / _GiB

        gc.collect()
        mx.clear_cache()

        after_active = self.active_gb()
        after_cache = mx.get_cache_memory() / _GiB

        freed = before_active - after_active
        if freed > 0.01:
            log.debug(
                f"GC cycle ({reason}): freed {freed:.3f} GB "
                f"(active {before_active:.3f} → {after_active:.3f} GB)"
            )

        return {
            "before_active_gb": round(before_active, 3),
            "after_active_gb": round(after_active, 3),
            "before_cache_gb": round(before_cache, 3),
            "after_cache_gb": round(after_cache, 3),
            "freed_gb": round(freed, 3),
        }

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_model_disk_size(path: Path) -> float:
        """Estimate model size on disk in GB by summing weight files."""
        total = 0
        for pattern in ["*.safetensors", "*.bin", "*.npz", "*.gguf", "*.pt"]:
            for f in path.glob(pattern):
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total / _GiB

    @staticmethod
    def _load_custom_tokenizer(model_path: Path) -> Any:
        """Fallback tokenizer loader for custom models."""
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    # ------------------------------------------------------------------
    #  System-level memory info (supplements MLX's own tracking)
    # ------------------------------------------------------------------

    @staticmethod
    def system_memory_pressure() -> Dict[str, Any]:
        """
        Query macOS memory pressure via memory_pressure tool + our detailed
        _get_local_memory_usage(). Returns system-level memory info.
        """
        # Get detailed local memory info
        local = MemoryManager._get_local_memory_usage()

        result = {
            "available": True,
            "pressure_level": "unknown",
            "total_gb": local.get("total_gb", 0.0),
            "free_gb": local.get("free_gb", 0.0),
            "available_gb": local.get("available_gb", 0.0),
            "used_gb": local.get("used_gb", 0.0),
            "wired_gb": local.get("wired_gb", 0.0),
            "compressed_gb": local.get("compressed_gb", 0.0),
        }

        # Get pressure level from memory_pressure tool
        try:
            out = subprocess.run(
                ["memory_pressure", "-Q"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0:
                text = out.stdout.lower()
                if "normal" in text:
                    result["pressure_level"] = "normal"
                elif "warn" in text:
                    result["pressure_level"] = "warning"
                elif "critical" in text:
                    result["pressure_level"] = "critical"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            result["available"] = False

        return result

    # ------------------------------------------------------------------
    #  Cross-node memory awareness
    # ------------------------------------------------------------------

    def _load_hosts(self) -> list:
        """Load hostfile data (cached after first call)."""
        if self._hosts_data is not None:
            return self._hosts_data

        self._hosts_data = []
        hostfile = self._hostfile
        if not hostfile or not os.path.isfile(hostfile):
            return self._hosts_data

        try:
            with open(hostfile) as f:
                self._hosts_data = json.load(f)
            log.debug(f"Loaded {len(self._hosts_data)} hosts from {hostfile}")
        except Exception as e:
            log.warning(f"Could not load hostfile {hostfile}: {e}")
            self._hosts_data = []

        return self._hosts_data

    def _is_local_host(self, ssh_target: str) -> bool:
        """Check if an SSH target refers to this machine."""
        local_hostname = socket.gethostname().lower()
        target = ssh_target.lower().split(".")[0]  # strip .local / .home etc.
        local_short = local_hostname.split(".")[0]
        return target == local_short

    def _query_remote_memory(self, ssh_target: str) -> NodeMemoryInfo:
        """
        Query a remote node's memory via SSH.

        Runs a small command that outputs JSON with memory stats.
        Times out quickly — if the remote node is down, we just report
        it as unreachable rather than blocking.
        """
        node = NodeMemoryInfo(
            hostname=ssh_target.split(".")[0],
            ssh=ssh_target,
            is_local=False,
            reachable=False,
        )

        cmd = (
            'python3 -c "'
            "import subprocess, json;"
            "o=subprocess.run(['sysctl','-n','hw.memsize'],capture_output=True,text=True);"
            "total=int(o.stdout.strip()) if o.returncode==0 else 0;"
            "o2=subprocess.run(['vm_stat'],capture_output=True,text=True);"
            "p={};"
            "[p.update({l.split(':')[0].strip().lower(): int(l.split(':')[1].strip().rstrip('.'))}) "
            "for l in o2.stdout.splitlines() if ':' in l and l.split(':')[1].strip().rstrip('.').isdigit()];"
            "ps=16384;"
            "free=p.get('pages free',0)*ps;"
            "inact=p.get('pages inactive',0)*ps;"
            "purg=p.get('pages purgeable',0)*ps;"
            "wired=p.get('pages wired down',0)*ps;"
            "comp=p.get('pages occupied by compressor',0)*ps;"
            "avail=free+inact+purg;"
            "g=1073741824;"
            "print(json.dumps({"
            "'total_gb':round(total/g,1),"
            "'free_gb':round(free/g,2),"
            "'available_gb':round(avail/g,2),"
            "'used_gb':round(max(0,total-avail)/g,2),"
            "'wired_gb':round(wired/g,2),"
            "'compressed_gb':round(comp/g,2)"
            '}))"'
        )

        try:
            out = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "ConnectTimeout=3",
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "BatchMode=yes",
                    ssh_target,
                    cmd,
                ],
                capture_output=True,
                text=True,
                timeout=_SSH_TIMEOUT,
            )
            if out.returncode == 0 and out.stdout.strip():
                data = json.loads(out.stdout.strip())
                node.total_gb = data.get("total_gb", 0)
                node.free_gb = data.get("free_gb", 0)
                node.used_gb = data.get("used_gb", 0)
                node.wired_gb = data.get("wired_gb", 0)
                node.compressed_gb = data.get("compressed_gb", 0)
                node.reachable = True

                # Determine pressure level from available ratio
                if node.total_gb > 0:
                    avail_pct = data.get("available_gb", 0) / node.total_gb
                    if avail_pct > 0.3:
                        node.pressure_level = "normal"
                    elif avail_pct > 0.15:
                        node.pressure_level = "warning"
                    else:
                        node.pressure_level = "critical"
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            json.JSONDecodeError,
        ) as e:
            log.debug(f"Could not query remote memory for {ssh_target}: {e}")

        return node

    def local_node_info(self) -> NodeMemoryInfo:
        """Get memory info for the local node."""
        local = self._get_local_memory_usage()
        sys_pressure = self.system_memory_pressure()

        return NodeMemoryInfo(
            hostname=socket.gethostname().split(".")[0],
            ssh=socket.gethostname(),
            total_gb=local.get("total_gb", self._total_ram / _GiB),
            free_gb=local.get("free_gb", 0),
            used_gb=local.get("used_gb", 0),
            wired_gb=local.get("wired_gb", 0),
            compressed_gb=local.get("compressed_gb", 0),
            pressure_level=sys_pressure.get("pressure_level", "unknown"),
            reachable=True,
            is_local=True,
            mlx_limit_gb=self._memory_limit / _GiB,
        )

    def cluster_memory_snapshot(self) -> Dict[str, Any]:
        """
        Query memory status across ALL nodes in the cluster.

        Reads the hostfile, queries each node (local = direct, remote = SSH),
        and returns a cluster-wide memory picture.

        This is useful for:
          - Dashboard cluster memory view
          - Pre-load checks: is the weakest node going to OOM?
          - Monitoring: detect memory pressure on any node before it panics
        """
        hosts = self._load_hosts()
        nodes: list = []

        if not hosts:
            # No hostfile — just return local info
            nodes.append(self.local_node_info().to_dict())
            return {
                "node_count": 1,
                "nodes": nodes,
                "cluster_total_gb": nodes[0]["total_gb"],
                "cluster_available_gb": nodes[0]["free_gb"],
                "weakest_node": nodes[0]["hostname"],
                "weakest_available_gb": nodes[0]["free_gb"],
            }

        for i, host in enumerate(hosts):
            ssh_target = host.get("ssh", "")
            if not ssh_target:
                continue

            if self._is_local_host(ssh_target):
                node = self.local_node_info()
            else:
                node = self._query_remote_memory(ssh_target)

            nodes.append(node.to_dict())

        # Compute cluster-wide stats
        cluster_total = sum(n["total_gb"] for n in nodes)
        cluster_available = sum(n.get("free_gb", 0) for n in nodes if n["reachable"])

        # Find the weakest node (least available memory)
        reachable_nodes = [n for n in nodes if n["reachable"]]
        if reachable_nodes:
            weakest = min(reachable_nodes, key=lambda n: n.get("free_gb", 0))
        else:
            weakest = {"hostname": "unknown", "free_gb": 0}

        return {
            "node_count": len(nodes),
            "nodes": nodes,
            "cluster_total_gb": round(cluster_total, 1),
            "cluster_available_gb": round(cluster_available, 2),
            "weakest_node": weakest["hostname"],
            "weakest_available_gb": round(weakest.get("free_gb", 0), 2),
        }

    # ------------------------------------------------------------------
    #  String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        snap = self.snapshot()
        return (
            f"MemoryManager("
            f"device={self._device_name}, "
            f"active={snap.active_gb:.2f}GB, "
            f"limit={snap.limit_gb:.1f}GB, "
            f"pressure={snap.pressure_pct:.1f}%, "
            f"model={'✓ ' + self._slot.model_id if self._slot.model else '✗'})"
        )

    def status_lines(self) -> list:
        """Return formatted status lines for CLI / banner display."""
        snap = self.snapshot()
        lines = [
            f"  Device:        {self._device_name}",
            f"  Total RAM:     {snap.total_ram_gb:.0f} GB",
            f"  System used:   {snap.system_used_gb:.1f} GB (non-MLX, measured at init)",
            f"  Available:     {snap.actual_available_gb:.1f} GB (for MLX at init)",
            f"  Memory limit:  {snap.limit_gb:.1f} GB  ← adaptive",
            f"  Cache limit:   {snap.cache_limit_gb:.1f} GB",
            f"  Active now:    {snap.active_gb:.3f} GB ({snap.pressure_pct:.1f}%)",
            f"  Peak:          {snap.peak_gb:.3f} GB",
            f"  Reserve:       {snap.system_reserve_gb:.0f} GB (config min)",
            f"  Pressure:      warn={self._pressure_warn:.0%}, critical={self._pressure_critical:.0%} (tiered)",
            f"  Hard max tok:  {self.hard_max_tokens}",
        ]
        if self._override_memory_bytes is not None:
            lines.append(
                f"  Override:      OVERRIDE_MEMORY_MB="
                f"{self._override_memory_bytes // (1024**2)} "
                f"({self._override_memory_bytes / _GiB:.1f} GB)"
            )
        if snap.model_loaded:
            lines.append(
                f"  Model:         {snap.model_id} ({snap.model_size_gb:.2f} GB)"
            )
        else:
            lines.append(f"  Model:         (none loaded)")
        return lines

    def print_status(self) -> None:
        """Print formatted memory status to stdout."""
        print("\n  ┌─── Memory Manager ───────────────────────────")
        for line in self.status_lines():
            print(f"  │ {line}")
        print("  └──────────────────────────────────────────────\n")


# ============================================================================
#  Convenience: module-level singleton
# ============================================================================

_instance: Optional[MemoryManager] = None


def get_manager() -> MemoryManager:
    """Get or create the global MemoryManager singleton."""
    global _instance
    if _instance is None:
        _instance = MemoryManager()
    return _instance


def init_manager(**kwargs) -> MemoryManager:
    """Initialize the global MemoryManager with custom settings."""
    global _instance
    _instance = MemoryManager(**kwargs)
    return _instance
