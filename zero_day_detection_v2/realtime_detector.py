"""
realtime_detector.py
====================
Real-Time Cross-Layer Network Packet Sniffer & Detector
Part of: AI-Powered Cross-Layer Network Intrusion Detection System

Enhancements over original detect_realtime.py:
  - Async packet processing (ThreadPoolExecutor)
  - Fused feature vector (79 packet + 5 system = 84-D)
  - Adaptive threshold (rolling window)
  - Encrypted/stealth traffic heuristics
  - Severity classification
  - Cross-layer IRQ/CPU correlation
  - Full JSON anomaly log with attack type and severity
"""

import os
import sys
import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logging.warning("Scapy not installed — packet capture unavailable. Use demo_mode.py.")

import joblib

from system_metrics import SystemMetricsCollector
from hybrid_detector import HybridDetector
from alerts import get_global_dispatcher, get_global_history

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
LOGS_DIR   = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "realtime_detector.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_WORKERS           = 4      # async thread pool size
PACKET_BUFFER_SIZE    = 200    # max queued packets
STATS_PRINT_INTERVAL  = 100    # print stats every N packets


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Feature Extraction (preserved + extended from original detect_realtime.py)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(packet) -> Optional[np.ndarray]:
    """
    Extract a 79-dimensional feature vector from a Scapy packet.
    (Preserved from original detect_realtime.py with minor hardening)

    Returns None if extraction fails.
    """
    features = np.zeros(79, dtype=np.float32)

    try:
        features[0] = len(packet)   # total packet length

        if IP in packet:
            ip = packet[IP]
            features[1] = ip.ttl
            features[2] = ip.proto
            features[3] = len(ip)
            features[4] = int(ip.flags)   if hasattr(ip, "flags") else 0
            features[5] = int(ip.frag)    if hasattr(ip, "frag")  else 0
            features[6] = int(ip.flags & 0x1) if hasattr(ip, "flags") else 0   # MF
            features[7] = int(ip.flags & 0x2) if hasattr(ip, "flags") else 0   # DF

            if TCP in packet:
                tcp = packet[TCP]
                features[8]  = tcp.sport
                features[9]  = tcp.dport
                features[10] = tcp.seq
                features[11] = tcp.ack
                features[12] = tcp.dataofs
                features[13] = int(tcp.flags)
                features[14] = tcp.window
                features[15] = len(tcp)
                features[16] = int(tcp.flags & 0x01)   # FIN
                features[17] = int(tcp.flags & 0x02)   # SYN
                features[18] = int(tcp.flags & 0x04)   # RST
                features[19] = int(tcp.flags & 0x08)   # PSH
                features[20] = int(tcp.flags & 0x10)   # ACK
                features[21] = int(tcp.flags & 0x20)   # URG

            elif UDP in packet:
                udp = packet[UDP]
                features[22] = udp.sport
                features[23] = udp.dport
                features[24] = len(udp)

            elif ICMP in packet:
                icmp = packet[ICMP]
                features[25] = icmp.type
                features[26] = icmp.code
                features[27] = len(icmp)

        features[28] = float(packet.time)
        features[29] = len(packet) / 1500.0   # normalised size

        eps = np.finfo(float).eps
        pkt_len_f = float(len(packet)) + eps
        if TCP in packet:
            features[30] = len(packet[TCP]) / pkt_len_f
        if UDP in packet:
            features[31] = len(packet[UDP]) / pkt_len_f
        if ICMP in packet:
            features[32] = len(packet[ICMP]) / pkt_len_f

        if IP in packet:
            src = packet[IP].src
            features[33] = 1.0 if src.startswith(("192.168.", "10.", "172.16.")) else 0.0

        # ── Stealth / encrypted traffic heuristics ────────────────────────────
        if TCP in packet:
            tcp = packet[TCP]
            # Near-MTU packet on TLS port → possible covert channel
            if tcp.dport in (443, 8443, 4433) and len(packet) > 1400:
                features[34] = 1.0    # stealth_tls_burst flag

        # Ensure no NaN / inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return None

    return features


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Real-Time Detection Engine
# ══════════════════════════════════════════════════════════════════════════════

class RealtimeDetector:
    """
    Wraps HybridDetector + SystemMetricsCollector for continuous
    live packet-level intrusion detection.
    """

    def __init__(self):
        # ── System metrics collector ───────────────────────────────────────────
        self._sys_collector = SystemMetricsCollector(window=60, interval=1.0)
        self._sys_collector.start()

        # ── Hybrid detector (LSTM + RF + XAI + Alerts) ────────────────────────
        self._detector = HybridDetector(
            dispatcher    = get_global_dispatcher(),
            sys_collector = self._sys_collector,
            log_anomalies = True,
        )

        # ── Async processing pool ─────────────────────────────────────────────
        self._executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        # ── Statistics ─────────────────────────────────────────────────────────
        self._total    = 0
        self._anomaly  = 0
        self._lock     = threading.Lock()

        logger.info("RealtimeDetector initialized.")

    # ── Packet Callback ────────────────────────────────────────────────────────

    def _on_packet(self, packet):
        """Scapy callback — submit packet to async processing pool."""
        self._executor.submit(self._process, packet)

    def _process(self, packet):
        """Process a single packet: extract features → hybrid detect → log."""
        features = extract_features(packet)
        if features is None:
            return

        src_ip   = packet[IP].src  if IP in packet else "0.0.0.0"
        dst_ip   = packet[IP].dst  if IP in packet else "0.0.0.0"
        proto    = ("TCP" if TCP in packet
                    else "UDP" if UDP in packet
                    else "ICMP" if ICMP in packet else "Other")

        result = self._detector.predict(
            features,
            src_ip   = src_ip,
            dst_ip   = dst_ip,
            protocol = proto,
        )

        with self._lock:
            self._total += 1
            if result["is_anomaly"]:
                self._anomaly += 1

            # Print stats periodically
            if self._total % STATS_PRINT_INTERVAL == 0:
                self._print_stats()

    def _print_stats(self):
        sys_snap = self._sys_collector.get_snapshot()
        det_stats = self._detector.get_stats()
        logger.info(
            f"[STATS] packets={self._total}  anomalies={self._anomaly}  "
            f"cpu={sys_snap.get('cpu', {}).get('overall', 'N/A')}%  "
            f"threshold={det_stats['current_threshold']:.6f}"
        )

    # ── Start / Stop ───────────────────────────────────────────────────────────

    def start(self, interface: str = None, pkt_filter: str = "ip"):
        """
        Begin packet capture on the given interface.

        Args:
            interface:  Network interface name (None = default)
            pkt_filter: BPF filter string
        """
        if not SCAPY_AVAILABLE:
            logger.error("Scapy is not installed. Cannot capture live packets.")
            return

        logger.info(f"Starting packet capture (interface={interface or 'default'}, "
                    f"filter='{pkt_filter}')")
        logger.info("Press Ctrl+C to stop.")

        try:
            sniff(
                iface  = interface,
                filter = pkt_filter,
                prn    = self._on_packet,
                store  = 0,         # do not store packets in memory
            )
        except PermissionError:
            logger.error(
                "Permission denied — run as root/Administrator for packet capture.\n"
                "  macOS:   sudo python realtime_detector.py\n"
                "  Linux:   sudo python realtime_detector.py\n"
                "  Windows: Run PowerShell as Administrator"
            )
        except KeyboardInterrupt:
            logger.info("Capture stopped by user.")
        finally:
            self.stop()

    def stop(self):
        self._sys_collector.stop()
        self._executor.shutdown(wait=False)
        logger.info(f"Detector stopped. Total: {self._total}, Anomalies: {self._anomaly}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI-Powered Cross-Layer Real-Time IDS")
    parser.add_argument("--iface",  default=None,
                        help="Network interface (e.g. eth0, en0). Defaults to system default.")
    parser.add_argument("--filter", default="ip",
                        help="BPF filter for packet capture (default: 'ip')")
    args = parser.parse_args()

    if os.name == "nt":
        # Windows: check admin
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        if not is_admin:
            logger.warning("Not running as Administrator — packet capture may fail.")

    engine = RealtimeDetector()
    engine.start(interface=args.iface, pkt_filter=args.filter)
