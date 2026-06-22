"""
tshark_detector.py
==================
TShark-based Real-Time Cross-Layer Network Packet Sniffer & Detector
Part of: AI-Powered Cross-Layer Network Intrusion Detection System
"""

import os
import sys
import json
import logging
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from wireshark_integration import extract_features_from_tshark_packet, _get_ip_from_pkt, TSHARK_PATH
from system_metrics import SystemMetricsCollector
from hybrid_detector import HybridDetector
from alerts import get_global_dispatcher

BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "tshark_detector.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

MAX_WORKERS = 4
STATS_PRINT_INTERVAL = 100

class TSharkRealtimeDetector:
    def __init__(self):
        self._sys_collector = SystemMetricsCollector(window=60, interval=1.0)
        self._sys_collector.start()

        self._detector = HybridDetector(
            dispatcher=get_global_dispatcher(),
            sys_collector=self._sys_collector,
            log_anomalies=True,
        )

        self._executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self._total = 0
        self._anomaly = 0
        self._lock = threading.Lock()
        self._running = False
        self._process = None

        logger.info("TSharkRealtimeDetector initialized.")

    def _process_packet(self, data):
        try:
            pkt = {"_source": {"layers": data.get("layers", {})}}
            features = extract_features_from_tshark_packet(pkt)
            if features is None:
                return

            src_ip, dst_ip, proto = _get_ip_from_pkt(pkt)

            result = self._detector.predict(
                features[:79],
                src_ip=src_ip,
                dst_ip=dst_ip,
                protocol=proto,
            )

            with self._lock:
                self._total += 1
                if result["is_anomaly"]:
                    self._anomaly += 1

                if self._total % STATS_PRINT_INTERVAL == 0:
                    self._print_stats()
        except Exception as e:
            logger.debug(f"Error processing packet: {e}")

    def _print_stats(self):
        sys_snap = self._sys_collector.get_snapshot()
        det_stats = self._detector.get_stats()
        logger.info(
            f"[TSHARK STATS] packets={self._total} anomalies={self._anomaly} "
            f"cpu={sys_snap.get('cpu', {}).get('overall', 'N/A')}% "
            f"threshold={det_stats['current_threshold']:.6f}"
        )

    def start(self, interface=None):
        if not TSHARK_PATH:
            logger.error("TShark is not installed. Cannot capture live packets.")
            return

        if interface is None and sys.platform == "darwin":
            interface = "en0"

        cmd = [TSHARK_PATH, "-l", "-T", "ek"]
        if interface:
            cmd.extend(["-i", interface])

        logger.info(f"Starting TShark capture (interface={interface or 'default'})")
        logger.info("Press Ctrl+C to stop.")

        self._running = True
        try:
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            for line in self._process.stdout:
                if not self._running:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "layers" in data:
                        self._executor.submit(self._process_packet, data)
                except Exception:
                    pass
        except PermissionError:
            logger.error("Permission denied. Run with sudo.")
        except KeyboardInterrupt:
            logger.info("Capture stopped by user.")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        if self._process:
            self._process.terminate()
            self._process = None
        self._sys_collector.stop()
        self._executor.shutdown(wait=False)
        logger.info(f"Detector stopped. Total: {self._total}, Anomalies: {self._anomaly}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI-Powered Cross-Layer TShark Real-Time IDS")
    parser.add_argument("--iface", default=None, help="Network interface")
    args = parser.parse_args()

    engine = TSharkRealtimeDetector()
    engine.start(interface=args.iface)
