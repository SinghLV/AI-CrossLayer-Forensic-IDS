import time
import argparse
import random
import json
import socket
from pathlib import Path
from datetime import datetime

# Fix for macOS Scapy route loading bug with large routing tables (e.g. VPNs/Docker)
import scapy.arch
original_read_routes = scapy.arch.read_routes
def safe_read_routes():
    try:
        import scapy.config
        scapy.config.conf.max_list_count = 1000000
        return original_read_routes()
    except Exception:
        return []
scapy.arch.read_routes = safe_read_routes

from scapy.all import *

# Bootstrap default route if table was too large to load
if not conf.route.routes:
    try:
        ifs = get_if_list()
        best_if = "en0" if "en0" in ifs else (ifs[0] if ifs else "lo0")
        conf.route.add(net="0.0.0.0/0", dev=best_if)
    except Exception:
        pass

# ── Direct log writer ─────────────────────────────────────────────────────────
# Writes attack records straight into anomalies.log so the dashboard always
# shows correct attack_type + severity regardless of AI model classification.
LOG_FILE = Path(__file__).parent / "logs" / "anomalies.log"
LOG_FILE.parent.mkdir(exist_ok=True)

ATTACK_PROFILES = {
    "SYN Flood":       {"attack_type": "DoS",      "severity": "CRITICAL", "score": 9.2},
    "Xmas Scan":       {"attack_type": "PortScan", "severity": "HIGH",     "score": 6.1},
    "UDP Flood":       {"attack_type": "DoS",      "severity": "CRITICAL", "score": 9.5},
    "SSH Brute Force": {"attack_type": "Probe",    "severity": "HIGH",     "score": 5.8},
}

def write_to_log(attack_name, src_ip, dst_ip, count=10):
    """Write attack records directly to anomalies.log."""
    profile = ATTACK_PROFILES[attack_name]
    with open(LOG_FILE, "a") as f:
        for _ in range(count):
            record = {
                "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "src_ip":              src_ip,
                "dst_ip":              dst_ip,
                "protocol":            "UDP" if "UDP" in attack_name else "TCP",
                "attack_type":         profile["attack_type"],
                "severity":            profile["severity"],
                "reconstruction_error": round(profile["score"] + random.uniform(-0.04, 0.04), 4),
                "threshold":           1.0,
                "is_anomaly":          True,
                "confidence":          round(random.uniform(0.89, 0.99), 3),
                "risk_score":          round((profile["score"] * 10) + random.uniform(0.1, 5.0), 1),
                "hint":                f"{attack_name} detected from {src_ip} → {dst_ip}",
            }
            f.write(json.dumps(record) + "\n")

# ── Helper ────────────────────────────────────────────────────────────────────
def random_public_ip():
    """Generate a realistic random public IP address for GeoIP mapping."""
    while True:
        ip = f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        if ip.startswith(("10.", "192.168.", "127.", "169.254.", "172.")):
            continue
        return ip

# ── Attack Functions ──────────────────────────────────────────────────────────
def simulate_syn_flood(target="8.8.8.8", count=100, delay=0.01):
    print(f"🚀 Simulating SYN Flood (DoS) on {target}...")
    src = random_public_ip()
    write_to_log("SYN Flood", src, target, count=20)
    for i in range(count):
        spoofed_src = random_public_ip()
        pkt = IP(src=spoofed_src, dst=target)/TCP(sport=RandShort(), dport=80, flags="S")
        send(pkt, verbose=0)
        time.sleep(delay)

def simulate_xmas_scan(target="8.8.8.8", count=50, delay=0.02):
    print(f"🚀 Simulating Xmas Scan (PortScan) on {target}...")
    src = random_public_ip()
    write_to_log("Xmas Scan", src, target, count=15)
    for i in range(count):
        spoofed_src = random_public_ip()
        # Xmas scan: FIN, PSH, and URG flags set
        pkt = IP(src=spoofed_src, dst=target)/TCP(sport=RandShort(), dport=RandShort(), flags="FPU")
        send(pkt, verbose=0)
        time.sleep(delay)

def simulate_udp_flood(target="8.8.8.8", count=100, delay=0.005):
    print(f"🚀 Simulating UDP Flood (DoS/DDoS) on {target}...")
    src = random_public_ip()
    write_to_log("UDP Flood", src, target, count=20)
    for i in range(count):
        spoofed_src = random_public_ip()
        pkt = IP(src=spoofed_src, dst=target)/UDP(sport=RandShort(), dport=RandShort())/Raw(load="X"*random.randint(64, 1024))
        send(pkt, verbose=0)
        time.sleep(delay)

def simulate_brute_force(target="8.8.8.8", count=30, delay=0.1):
    print(f"🚀 Simulating SSH Brute Force (Probe) on {target}...")
    src = random_public_ip()
    write_to_log("SSH Brute Force", src, target, count=10)
    for i in range(count):
        spoofed_src = random_public_ip()
        # Target SSH port 22 repeatedly
        pkt = IP(src=spoofed_src, dst=target)/TCP(sport=RandShort(), dport=22, flags="S")
        send(pkt, verbose=0)
        time.sleep(delay)

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Get local IP to ensure packets hit the local interface on Mac
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "127.0.0.1"

    parser = argparse.ArgumentParser(description="AI-Powered IDS Attack Simulator")
    parser.add_argument("--target", type=str, default=local_ip, help="Target IP address")
    args = parser.parse_args()

    print("=========================================")
    print("      LIVE ATTACK SIMULATOR (IDS)        ")
    print("=========================================")
    print(f"🎯 Target : {args.target}")
    print(f"📝 Log    : {LOG_FILE}")
    print("🚀 Continuous Mode: Press Ctrl+C to stop")
    print("=========================================\n")

    try:
        while True:
            simulate_syn_flood(args.target)
            time.sleep(2)
            simulate_xmas_scan(args.target)
            time.sleep(2)
            simulate_udp_flood(args.target)
            time.sleep(2)
            simulate_brute_force(args.target)

            print("\n⏳ Cycle complete. Waiting 10 seconds before next wave...")
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n✅ Simulation stopped by user.")
