"""
demo_mode.py
============
Demo Attack Simulator
Part of: AI-Powered Cross-Layer Network Intrusion Detection System

Generates synthetic attack and normal traffic without requiring:
  - Real network access
  - Admin/root privileges
  - Enterprise dataset download

Simulated attack patterns:
  1. Normal baseline traffic
  2. DoS / DDoS flood
  3. Port scan sweep
  4. Brute force login attempts
  5. Encrypted stealth / covert channel
  6. ICMP fragmentation storm

Use this to demonstrate the full dashboard and alert pipeline.
"""

import json
import logging
import random
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
LOGS_DIR   = BASE_DIR / "logs"
DEMO_DIR   = BASE_DIR / "demo_data"
LOGS_DIR.mkdir(exist_ok=True)
DEMO_DIR.mkdir(exist_ok=True)

# ── Feature index reference (same as xai_engine.py) ──────────────────────────
FI = {
    "pkt_len": 0,  "ip_ttl": 1,   "ip_proto": 2,  "ip_flags": 4,
    "ip_frag": 5,  "mf_flag": 6,  "df_flag": 7,
    "tcp_sport": 8, "tcp_dport": 9, "tcp_seq": 10, "tcp_ack": 11,
    "tcp_dataofs": 12, "tcp_flags": 13, "tcp_window": 14, "tcp_len": 15,
    "tcp_fin": 16,  "tcp_syn": 17, "tcp_rst": 18,
    "tcp_psh": 19,  "tcp_ack_f": 20, "tcp_urg": 21,
    "udp_sport": 22, "udp_dport": 23, "udp_len": 24,
    "icmp_type": 25, "icmp_code": 26, "icmp_len": 27,
    "timestamp": 28, "pkt_norm": 29,
}

# ── Sample IP pools ───────────────────────────────────────────────────────────
INTERNAL_IPS = [f"192.168.1.{i}" for i in range(1, 20)]
EXTERNAL_IPS = [
    "45.33.32.156", "185.220.101.1", "91.108.4.1",
    "203.0.113.5",  "198.51.100.7", "172.217.14.196",
    "104.16.0.1",   "52.94.133.2",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Feature Vector Generators
# ══════════════════════════════════════════════════════════════════════════════

def _base_features(rng: np.random.Generator) -> np.ndarray:
    """79-D zero-filled feature vector with random noise."""
    return rng.normal(0.0, 0.05, 79).astype(np.float32)


def gen_normal_packet(rng: np.random.Generator) -> Dict:
    """Simulate a benign HTTP/HTTPS packet."""
    feat = _base_features(rng)
    feat[FI["pkt_len"]]   = float(rng.integers(200, 1400))
    feat[FI["ip_ttl"]]    = float(rng.choice([64, 128]))
    feat[FI["ip_proto"]]  = 6.0   # TCP
    feat[FI["tcp_sport"]] = float(rng.integers(1024, 60000))
    feat[FI["tcp_dport"]] = float(rng.choice([80, 443, 8080]))
    feat[FI["tcp_flags"]] = 0x18  # PSH+ACK
    feat[FI["tcp_ack_f"]] = 1.0
    feat[FI["pkt_norm"]]  = feat[FI["pkt_len"]] / 1500.0

    return {
        "features":    feat,
        "attack_type": "Normal",
        "src_ip":      random.choice(INTERNAL_IPS),
        "dst_ip":      random.choice(INTERNAL_IPS),
        "protocol":    "TCP",
        "severity":    "LOW",
        "hint":        "Normal traffic",
    }


def gen_dos_packet(rng: np.random.Generator) -> Dict:
    """Simulate a SYN-flood DoS packet."""
    feat = _base_features(rng)
    feat[FI["pkt_len"]]   = float(rng.integers(40, 80))   # tiny packet
    feat[FI["ip_ttl"]]    = float(rng.integers(1, 20))    # suspicious TTL
    feat[FI["ip_proto"]]  = 6.0
    feat[FI["tcp_sport"]] = float(rng.integers(1024, 65535))
    feat[FI["tcp_dport"]] = 80.0
    feat[FI["tcp_syn"]]   = 1.0
    feat[FI["tcp_ack_f"]] = 0.0   # SYN without ACK → flood
    feat[FI["tcp_flags"]] = 0x02
    feat[FI["pkt_norm"]]  = feat[FI["pkt_len"]] / 1500.0

    # Add noise to push reconstruction error above threshold
    feat += rng.normal(2.0, 0.5, 79).astype(np.float32)

    return {
        "features":    feat,
        "attack_type": "DoS",
        "src_ip":      random.choice(EXTERNAL_IPS),
        "dst_ip":      random.choice(INTERNAL_IPS),
        "protocol":    "TCP",
        "severity":    "CRITICAL",
        "hint":        "SYN flood; Unusual TTL; Unusual port usage",
    }


def gen_port_scan_packet(rng: np.random.Generator, scan_port: int = None) -> Dict:
    """Simulate a port scan probe."""
    feat = _base_features(rng)
    port = scan_port or int(rng.integers(1, 65535))
    feat[FI["pkt_len"]]   = float(rng.integers(40, 60))   # minimal
    feat[FI["ip_ttl"]]    = 64.0
    feat[FI["ip_proto"]]  = 6.0
    feat[FI["tcp_sport"]] = float(rng.integers(40000, 60000))
    feat[FI["tcp_dport"]] = float(port)
    feat[FI["tcp_syn"]]   = 1.0
    feat[FI["tcp_rst"]]   = 0.0
    feat[FI["tcp_flags"]] = 0x02
    feat[FI["pkt_norm"]]  = feat[FI["pkt_len"]] / 1500.0

    feat += rng.normal(1.5, 0.3, 79).astype(np.float32)

    return {
        "features":    feat,
        "attack_type": "PortScan",
        "src_ip":      random.choice(EXTERNAL_IPS),
        "dst_ip":      random.choice(INTERNAL_IPS),
        "protocol":    "TCP",
        "severity":    "HIGH",
        "hint":        f"Port scan on port {port}; Unusual port usage",
    }


def gen_brute_force_packet(rng: np.random.Generator) -> Dict:
    """Simulate a brute-force SSH/RDP attempt."""
    feat = _base_features(rng)
    target_port = random.choice([22, 3389, 21, 23])
    feat[FI["pkt_len"]]   = float(rng.integers(80, 300))
    feat[FI["ip_ttl"]]    = float(rng.choice([64, 128]))
    feat[FI["ip_proto"]]  = 6.0
    feat[FI["tcp_sport"]] = float(rng.integers(40000, 60000))
    feat[FI["tcp_dport"]] = float(target_port)
    feat[FI["tcp_flags"]] = 0x18   # PSH+ACK (sending credentials)
    feat[FI["tcp_ack_f"]] = 1.0
    feat[FI["pkt_norm"]]  = feat[FI["pkt_len"]] / 1500.0

    feat += rng.normal(1.8, 0.4, 79).astype(np.float32)

    return {
        "features":    feat,
        "attack_type": "BruteForce",
        "src_ip":      random.choice(EXTERNAL_IPS),
        "dst_ip":      random.choice(INTERNAL_IPS),
        "protocol":    "TCP",
        "severity":    "HIGH",
        "hint":        f"Repeated auth attempts on port {target_port}",
    }


def gen_stealth_packet(rng: np.random.Generator) -> Dict:
    """Simulate encrypted stealth / covert channel traffic."""
    feat = _base_features(rng)
    feat[FI["pkt_len"]]   = float(rng.integers(1400, 1500))  # near-MTU
    feat[FI["ip_ttl"]]    = float(rng.integers(100, 128))
    feat[FI["ip_proto"]]  = 6.0
    feat[FI["tcp_sport"]] = float(rng.integers(40000, 60000))
    feat[FI["tcp_dport"]] = float(random.choice([443, 8443, 4433]))
    feat[FI["tcp_flags"]] = 0x18
    feat[FI["tcp_ack_f"]] = 1.0
    feat[FI["pkt_norm"]]  = 0.99   # near-MTU hint

    feat += rng.normal(1.2, 0.3, 79).astype(np.float32)

    return {
        "features":    feat,
        "attack_type": "Probe",
        "src_ip":      random.choice(EXTERNAL_IPS),
        "dst_ip":      random.choice(INTERNAL_IPS),
        "protocol":    "TCP",
        "severity":    "MEDIUM",
        "hint":        "Maximum-size HTTPS packet; possible covert channel",
    }


def gen_icmp_fragment_packet(rng: np.random.Generator) -> Dict:
    """Simulate ICMP fragmentation storm."""
    feat = _base_features(rng)
    feat[FI["pkt_len"]]   = float(rng.integers(500, 1500))
    feat[FI["ip_ttl"]]    = float(rng.integers(1, 30))
    feat[FI["ip_proto"]]  = 1.0    # ICMP
    feat[FI["mf_flag"]]   = 1.0    # More Fragments
    feat[FI["ip_frag"]]   = float(rng.integers(1, 100))
    feat[FI["icmp_type"]] = float(rng.integers(0, 255))
    feat[FI["pkt_norm"]]  = feat[FI["pkt_len"]] / 1500.0

    feat += rng.normal(2.2, 0.5, 79).astype(np.float32)

    return {
        "features":    feat,
        "attack_type": "DoS",
        "src_ip":      random.choice(EXTERNAL_IPS),
        "dst_ip":      random.choice(INTERNAL_IPS),
        "protocol":    "ICMP",
        "severity":    "CRITICAL",
        "hint":        "IP fragmentation storm; Unusual TTL; ICMP anomaly",
    }


# ── Pattern registry ──────────────────────────────────────────────────────────
ATTACK_GENERATORS = {
    "Normal":    gen_normal_packet,
    "DoS":       gen_dos_packet,
    "PortScan":  gen_port_scan_packet,
    "BruteForce": gen_brute_force_packet,
    "Probe":     gen_stealth_packet,
    "ICMP":      gen_icmp_fragment_packet,
}

# Probability weights for each packet type in demo traffic
TRAFFIC_MIX = {
    "Normal":    0.50,
    "DoS":       0.15,
    "PortScan":  0.12,
    "BruteForce":0.10,
    "Probe":     0.08,
    "ICMP":      0.05,
}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Demo Log Writer  (writes to anomalies.log in the expected format)
# ══════════════════════════════════════════════════════════════════════════════

def write_demo_log(packet_info: Dict, reconstruction_error: float, threshold: float):
    """
    Write a single synthetic anomaly to logs/anomalies.log in the same
    JSON-line format as the live detector.
    """
    if packet_info["attack_type"] == "Normal":
        return   # Only log anomalies

    record = {
        "timestamp":           datetime.now().isoformat(),
        "src_ip":              packet_info["src_ip"],
        "dst_ip":              packet_info["dst_ip"],
        "protocol":            packet_info["protocol"],
        "reconstruction_error": round(reconstruction_error, 6),
        "threshold":           round(threshold, 6),
        "is_anomaly":          True,
        "attack_type":         packet_info["attack_type"],
        "severity":            packet_info["severity"],
        "confidence":          round(random.uniform(0.70, 0.98), 3),
        "risk_score":          round(random.uniform(0.4, 0.99), 3),
        "hint":                packet_info["hint"],
    }
    log_path = LOGS_DIR / "anomalies.log"
    with open(log_path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Demo Simulator Class
# ══════════════════════════════════════════════════════════════════════════════

class DemoAttackSimulator:
    """
    Continuously generates synthetic network packets and writes anomalies
    to the standard log file so the dashboard updates in real time.

    Usage:
        sim = DemoAttackSimulator(rate=2.0)
        sim.start()
        # ... dashboard runs...
        sim.stop()
    """

    def __init__(
        self,
        rate:           float = 2.0,   # packets per second
        callback:       Optional[Callable[[Dict], None]] = None,
        detector=None,                  # optional HybridDetector reference
    ):
        self._rate     = rate
        self._callback = callback
        self._detector = detector
        self._rng      = np.random.default_rng(int(time.time()))
        self._stop_evt = threading.Event()
        self._thread   = threading.Thread(
            target=self._run, daemon=True, name="DemoSimulator"
        )
        self._types     = list(TRAFFIC_MIX.keys())
        self._weights   = list(TRAFFIC_MIX.values())
        self._threshold = 0.05   # default demo threshold

        self.packets_sent    = 0
        self.anomalies_sent  = 0

    def start(self):
        self._thread.start()
        logger.info(f"DemoAttackSimulator started at {self._rate} pkt/s")

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=3)
        logger.info("DemoAttackSimulator stopped")

    def _run(self):
        interval = 1.0 / max(self._rate, 0.1)
        while not self._stop_evt.is_set():
            try:
                self._emit_packet()
            except Exception as e:
                logger.error(f"Demo simulator error: {e}")
            self._stop_evt.wait(timeout=interval)

    def _emit_packet(self):
        pkt_type = random.choices(self._types, weights=self._weights, k=1)[0]
        gen_fn   = ATTACK_GENERATORS[pkt_type]
        pkt_info = gen_fn(self._rng)
        self.packets_sent += 1

        # Compute a synthetic reconstruction error
        feat = pkt_info["features"]
        base_error = float(np.mean(feat ** 2))   # surrogate MSE

        if pkt_type != "Normal":
            self.anomalies_sent += 1
            write_demo_log(pkt_info, base_error, self._threshold)

        # Optional: pass through real HybridDetector
        if self._detector is not None:
            result = self._detector.predict(
                feat,
                src_ip   = pkt_info["src_ip"],
                dst_ip   = pkt_info["dst_ip"],
                protocol = pkt_info["protocol"],
            )
            pkt_info["detector_result"] = result

        # Optional callback (e.g. for Streamlit live updates)
        if self._callback:
            self._callback(pkt_info)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Pre-seed the log file with historical demo data
# ══════════════════════════════════════════════════════════════════════════════

def seed_demo_log(n_entries: int = 200, clear_existing: bool = False):
    """
    Populate logs/anomalies.log with N synthetic attack records
    so the dashboard has data to display immediately on first launch.
    """
    log_path = LOGS_DIR / "anomalies.log"

    if clear_existing and log_path.exists():
        log_path.unlink()
        logger.info("Cleared existing anomaly log")

    rng = np.random.default_rng(42)
    types = ["DoS", "PortScan", "BruteForce", "Probe", "ICMP"]
    thresh = 0.05

    written = 0
    for i in range(n_entries):
        pkt_type = random.choice(types)
        gen_fn   = ATTACK_GENERATORS[pkt_type]
        pkt_info = gen_fn(rng)

        # Spread timestamps over the last 2 hours
        ago_seconds = random.randint(0, 7200)
        ts = datetime.fromtimestamp(time.time() - ago_seconds).isoformat()

        feat  = pkt_info["features"]
        error = float(np.mean(feat ** 2)) + random.uniform(0.05, 0.5)

        record = {
            "timestamp":           ts,
            "src_ip":              pkt_info["src_ip"],
            "dst_ip":              pkt_info["dst_ip"],
            "protocol":            pkt_info["protocol"],
            "reconstruction_error": round(error, 6),
            "threshold":           round(thresh, 6),
            "is_anomaly":          True,
            "attack_type":         pkt_type,
            "severity":            pkt_info["severity"],
            "confidence":          round(random.uniform(0.70, 0.98), 3),
            "risk_score":          round(random.uniform(0.4, 0.99), 3),
            "hint":                pkt_info["hint"],
        }
        with open(log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        written += 1

    logger.info(f"Seeded {written} demo anomaly records to {log_path}")
    return written


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CLI Entry
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Demo Attack Simulator")
    parser.add_argument("--seed",    type=int,   default=200,
                        help="Pre-seed N historical records to anomalies.log")
    parser.add_argument("--rate",    type=float, default=1.0,
                        help="Live simulation rate (packets/second)")
    parser.add_argument("--runtime", type=int,   default=30,
                        help="Seconds to run live simulation")
    parser.add_argument("--clear",   action="store_true",
                        help="Clear existing anomaly log before seeding")
    args = parser.parse_args()

    # Seed historical data
    n = seed_demo_log(args.seed, clear_existing=args.clear)
    print(f"✅ Seeded {n} historical anomaly records")

    # Live simulation
    sim = DemoAttackSimulator(rate=args.rate)
    sim.start()
    print(f"🚀 Live simulation running for {args.runtime}s at {args.rate} pkt/s...")
    time.sleep(args.runtime)
    sim.stop()

    print(f"\n📊 Summary:")
    print(f"   Packets sent   : {sim.packets_sent}")
    print(f"   Anomalies sent : {sim.anomalies_sent}")
    print(f"   Log file       : {LOGS_DIR / 'anomalies.log'}")
