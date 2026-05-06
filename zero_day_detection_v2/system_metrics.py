"""
system_metrics.py
=================
Cross-Layer System Metrics Collector
Part of: AI-Powered Cross-Layer Network Intrusion Detection System

Collects CPU usage, RAM usage, IRQ interrupt counts, and network interface
interrupt frequency to provide system-level context alongside network anomalies.

Compatible with: Linux, macOS, Windows (graceful fallback for /proc filesystems)
"""

import os
import re
import time
import logging
import threading
import platform
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import psutil
import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
ROLLING_WINDOW   = 60        # seconds of history kept
SAMPLE_INTERVAL  = 1.0       # seconds between samples
IRQ_SPIKE_SIGMA  = 3.0       # std deviations above mean to flag a spike
PROC_INTERRUPTS  = "/proc/interrupts"
PROC_STAT        = "/proc/stat"
IS_LINUX         = platform.system() == "Linux"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  IRQ Reader
# ══════════════════════════════════════════════════════════════════════════════

def _read_proc_interrupts() -> Dict[str, int]:
    """
    Read /proc/interrupts and return a dict of {irq_name: total_count}.
    Falls back to empty dict on non-Linux platforms.
    """
    if not IS_LINUX:
        return {}

    counts: Dict[str, int] = {}
    try:
        with open(PROC_INTERRUPTS, "r") as fh:
            lines = fh.readlines()

        # First line is CPU header; skip it
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 2:
                continue
            irq_id   = parts[0].rstrip(":")
            cpu_vals = []
            idx = 1
            while idx < len(parts):
                try:
                    cpu_vals.append(int(parts[idx]))
                    idx += 1
                except ValueError:
                    break
            counts[irq_id] = sum(cpu_vals)
    except Exception as exc:
        logger.warning(f"Could not read {PROC_INTERRUPTS}: {exc}")

    return counts


def _get_network_irq_names() -> List[str]:
    """
    Identify IRQ names likely related to network interfaces by cross-referencing
    /proc/interrupts device name column with known NIC patterns.
    """
    network_keywords = ["eth", "ens", "enp", "wlan", "wlp", "virtio", "vmxnet",
                        "e1000", "igb", "ixgbe", "mlx", "i40e", "bnxt", "tg3"]
    names: List[str] = []
    if not IS_LINUX:
        return names

    try:
        with open(PROC_INTERRUPTS, "r") as fh:
            for line in fh.readlines()[1:]:
                # Device description is everything after the last numeric column
                parts = line.split()
                if not parts:
                    continue
                irq_id = parts[0].rstrip(":")
                desc   = " ".join(parts[1:]).lower()
                if any(kw in desc for kw in network_keywords):
                    names.append(irq_id)
    except Exception:
        pass

    return names


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Single-shot Metric Readers
# ══════════════════════════════════════════════════════════════════════════════

def get_cpu_usage() -> Dict[str, float]:
    """
    Return per-core and overall CPU utilisation (%).

    Returns:
        {
            "overall": float,          # e.g. 34.2
            "per_core": [float, ...],  # one value per logical core
        }
    """
    overall   = psutil.cpu_percent(interval=None)
    per_core  = psutil.cpu_percent(interval=None, percpu=True)
    return {"overall": overall, "per_core": per_core}


def get_ram_usage() -> Dict[str, float]:
    """
    Return RAM statistics in MB and percentage.

    Returns:
        {
            "total_mb": float,
            "used_mb":  float,
            "free_mb":  float,
            "percent":  float,
        }
    """
    vm = psutil.virtual_memory()
    return {
        "total_mb": round(vm.total   / (1024 ** 2), 2),
        "used_mb":  round(vm.used    / (1024 ** 2), 2),
        "free_mb":  round(vm.free    / (1024 ** 2), 2),
        "percent":  vm.percent,
    }


def get_irq_counts() -> Dict[str, int]:
    """
    Return current cumulative IRQ counts per interrupt line.
    On non-Linux platforms returns synthetic simulated values.
    """
    if IS_LINUX:
        return _read_proc_interrupts()

    # ── Simulation for macOS / Windows ──────────────────────────────────────
    base = getattr(get_irq_counts, "_base", None)
    if base is None:
        base = {"NET0": 1_000_000, "NET1": 500_000, "USB0": 200_000, "TIMER": 10_000_000}
        get_irq_counts._base = base

    # Increment by random amounts to simulate activity
    rng = np.random.default_rng()
    for k in base:
        base[k] += int(rng.integers(0, 300))

    return dict(base)


def get_network_irq_frequency(
    prev_counts: Dict[str, int],
    curr_counts: Dict[str, int],
    elapsed_sec: float,
) -> Dict[str, float]:
    """
    Compute per-IRQ interrupt frequency (interrupts/second) between two samples.

    Args:
        prev_counts: IRQ counts from previous sample
        curr_counts: IRQ counts from current sample
        elapsed_sec: Seconds elapsed between samples

    Returns:
        Dict mapping IRQ name → frequency (irq/s)
    """
    if elapsed_sec <= 0:
        return {}

    freq: Dict[str, float] = {}
    for irq, curr_val in curr_counts.items():
        prev_val = prev_counts.get(irq, curr_val)
        delta    = max(0, curr_val - prev_val)
        freq[irq] = round(delta / elapsed_sec, 3)

    return freq


def get_process_load(top_n: int = 5) -> List[Dict]:
    """
    Return top N processes ranked by CPU usage.

    Returns:
        List of {pid, name, cpu_percent, memory_mb}
    """
    procs = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
        try:
            info = proc.info
            procs.append({
                "pid":       info["pid"],
                "name":      info["name"],
                "cpu_pct":   info["cpu_percent"] or 0.0,
                "mem_mb":    round((info["memory_info"].rss if info["memory_info"] else 0) / (1024 ** 2), 2),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    procs.sort(key=lambda x: x["cpu_pct"], reverse=True)
    return procs[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Spike Detector
# ══════════════════════════════════════════════════════════════════════════════

def detect_irq_spike(history: deque, current_value: float, sigma: float = IRQ_SPIKE_SIGMA) -> Tuple[bool, float]:
    """
    Determine whether the current IRQ frequency is a statistical spike.

    Args:
        history:       deque of recent frequency values
        current_value: latest measurement
        sigma:         Z-score threshold

    Returns:
        (is_spike: bool, z_score: float)
    """
    arr = np.array(list(history))
    if len(arr) < 5:
        return False, 0.0

    mean = arr.mean()
    std  = arr.std()
    if std < 1e-9:
        return False, 0.0

    z = (current_value - mean) / std
    return (z > sigma), round(float(z), 3)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Background Collector (Thread-safe ring-buffer time series)
# ══════════════════════════════════════════════════════════════════════════════

class SystemMetricsCollector:
    """
    Background thread that continuously samples system metrics and stores
    a rolling window of history for use by the detection engine and dashboard.

    Usage:
        collector = SystemMetricsCollector()
        collector.start()
        ...
        snapshot = collector.get_snapshot()
        collector.stop()
    """

    def __init__(self, window: int = ROLLING_WINDOW, interval: float = SAMPLE_INTERVAL):
        self._window   = window
        self._interval = interval
        self._lock     = threading.Lock()
        self._stop_evt = threading.Event()

        # Time-series history (max `window` entries)
        maxlen = window
        self._cpu_history:         deque = deque(maxlen=maxlen)
        self._ram_history:         deque = deque(maxlen=maxlen)
        self._irq_freq_history:    deque = deque(maxlen=maxlen)   # total net IRQ freq
        self._timestamps:          deque = deque(maxlen=maxlen)

        # IRQ state
        self._prev_irq_counts: Dict[str, int] = {}
        self._prev_sample_ts:  float           = time.time()
        self._net_irq_names:   List[str]       = _get_network_irq_names()

        # Latest single snapshot
        self._latest: Dict = {}

        # Spike flag
        self.irq_spike_active: bool  = False
        self.cpu_spike_active:  bool  = False

        self._thread = threading.Thread(target=self._run, daemon=True, name="SysMetricsCollector")

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self):
        """Start the background collection thread."""
        # Prime the CPU % counter (needs two calls)
        psutil.cpu_percent(interval=None)
        self._thread.start()
        logger.info("SystemMetricsCollector started.")

    def stop(self):
        """Signal the collection thread to stop."""
        self._stop_evt.set()
        self._thread.join(timeout=5)
        logger.info("SystemMetricsCollector stopped.")

    # ── Collection Loop ───────────────────────────────────────────────────────

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                self._collect()
            except Exception as exc:
                logger.error(f"Metrics collection error: {exc}")
            self._stop_evt.wait(timeout=self._interval)

    def _collect(self):
        now = datetime.utcnow()
        ts  = time.time()

        # ── CPU ──────────────────────────────────────────────────────────────
        cpu = get_cpu_usage()

        # ── RAM ──────────────────────────────────────────────────────────────
        ram = get_ram_usage()

        # ── IRQ ──────────────────────────────────────────────────────────────
        curr_irq     = get_irq_counts()
        elapsed      = ts - self._prev_sample_ts
        irq_freqs    = get_network_irq_frequency(self._prev_irq_counts, curr_irq, elapsed)
        self._prev_irq_counts = curr_irq
        self._prev_sample_ts  = ts

        # Sum of network-related IRQ frequencies
        if self._net_irq_names:
            net_irq_total = sum(irq_freqs.get(n, 0.0) for n in self._net_irq_names)
        else:
            # On non-Linux, use total of all simulated IRQs
            net_irq_total = sum(irq_freqs.values())

        # ── Spike detection ──────────────────────────────────────────────────
        irq_spike, irq_z   = detect_irq_spike(self._irq_freq_history, net_irq_total)
        cpu_spike, cpu_z   = detect_irq_spike(self._cpu_history, cpu["overall"], sigma=2.5)

        # ── Store ─────────────────────────────────────────────────────────────
        with self._lock:
            self._timestamps.append(now.isoformat())
            self._cpu_history.append(cpu["overall"])
            self._ram_history.append(ram["percent"])
            self._irq_freq_history.append(net_irq_total)

            self.irq_spike_active = irq_spike
            self.cpu_spike_active  = cpu_spike

            self._latest = {
                "timestamp":      now.isoformat(),
                "cpu":            cpu,
                "ram":            ram,
                "net_irq_freq":   round(net_irq_total, 3),
                "irq_spike":      irq_spike,
                "irq_z_score":    irq_z,
                "cpu_spike":      cpu_spike,
                "cpu_z_score":    cpu_z,
                "top_processes":  get_process_load(top_n=5),
            }

    # ── Public API ────────────────────────────────────────────────────────────

    def get_snapshot(self) -> Dict:
        """Return the most recent metrics snapshot (thread-safe)."""
        with self._lock:
            return dict(self._latest)

    def get_history(self) -> Dict[str, list]:
        """
        Return the full rolling history as lists.
        Suitable for feeding Plotly line charts.
        """
        with self._lock:
            return {
                "timestamps":    list(self._timestamps),
                "cpu_percent":   list(self._cpu_history),
                "ram_percent":   list(self._ram_history),
                "net_irq_freq":  list(self._irq_freq_history),
            }

    def get_fused_features(self) -> np.ndarray:
        """
        Return a 1-D numpy array of normalised system metrics for feature fusion.
        Dimensions: [cpu_overall, ram_percent, net_irq_freq_norm, irq_spike_flag, cpu_spike_flag]
        """
        snap = self.get_snapshot()
        if not snap:
            return np.zeros(5, dtype=np.float32)

        return np.array([
            snap.get("cpu", {}).get("overall", 0.0) / 100.0,
            snap.get("ram", {}).get("percent",  0.0) / 100.0,
            min(snap.get("net_irq_freq", 0.0) / 10_000.0, 1.0),   # normalise by 10K irq/s
            float(snap.get("irq_spike",  False)),
            float(snap.get("cpu_spike",   False)),
        ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Stand-alone smoke-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  System Metrics — Live Test  ")
    print("=" * 60)

    collector = SystemMetricsCollector(window=10, interval=1.0)
    collector.start()

    for i in range(6):
        time.sleep(1.5)
        snap = collector.get_snapshot()
        print(f"\n[Sample {i+1}]")
        print(f"  CPU  : {snap.get('cpu', {}).get('overall', 'N/A')} %")
        print(f"  RAM  : {snap.get('ram', {}).get('percent', 'N/A')} %")
        print(f"  NET IRQ Freq : {snap.get('net_irq_freq', 'N/A')} irq/s")
        print(f"  IRQ Spike : {snap.get('irq_spike', False)}")
        print(f"  CPU Spike : {snap.get('cpu_spike',  False)}")

    history = collector.get_history()
    print(f"\nHistory lengths → cpu:{len(history['cpu_percent'])}  "
          f"ram:{len(history['ram_percent'])}  "
          f"irq:{len(history['net_irq_freq'])}")

    fused = collector.get_fused_features()
    print(f"\nFused feature vector (5-D): {fused}")

    collector.stop()
    print("\nDone.")
