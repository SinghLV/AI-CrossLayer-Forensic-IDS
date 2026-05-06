"""
wireshark_integration.py
========================
TShark/Wireshark integration layer for the AI-Powered Cross-Layer IDS.

Role: Supplementary forensic analysis engine (Scapy remains primary realtime detector).
Provides:
  - PCAP parsing via tshark CLI (subprocess)
  - Protocol extraction & metadata
  - 84-D feature vector from tshark JSON
  - Forensic analysis: top talkers, TCP stats, bandwidth
  - PCAP replay through HybridDetector
  - Live Scapy vs TShark drift validation
  - Graceful fallback if tshark not installed
"""

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Availability Check ────────────────────────────────────────────────────────

def _find_tshark() -> Optional[str]:
    """Return tshark binary path or None."""
    path = shutil.which("tshark")
    if path:
        return path
    # Common macOS install via Wireshark.app
    candidates = [
        "/Applications/Wireshark.app/Contents/MacOS/tshark",
        "/usr/local/bin/tshark",
        "/usr/bin/tshark",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None

TSHARK_PATH: Optional[str] = _find_tshark()
TSHARK_AVAILABLE: bool = TSHARK_PATH is not None


def get_tshark_version() -> str:
    if not TSHARK_AVAILABLE:
        return "Not installed"
    try:
        out = subprocess.check_output([TSHARK_PATH, "--version"], stderr=subprocess.DEVNULL, text=True)
        return out.splitlines()[0]
    except Exception:
        return "Unknown"


# ── TShark Raw Parsing ────────────────────────────────────────────────────────

def _run_tshark(args: list[str], timeout: int = 120) -> str:
    """Run tshark command and return stdout."""
    cmd = [TSHARK_PATH] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout


def parse_pcap_json(pcap_path: str, max_packets: int = 10000) -> list[dict]:
    """
    Parse a PCAP/PCAPNG file via tshark -T json.
    Returns a list of packet dicts (tshark JSON format).
    """
    if not TSHARK_AVAILABLE:
        raise RuntimeError("TShark not available. Install Wireshark: https://www.wireshark.org/download.html")

    raw = _run_tshark([
        "-r", pcap_path,
        "-T", "json",
        "-c", str(max_packets),
        "--no-duplicate-keys",
    ])
    if not raw.strip():
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"TShark JSON parse error: {e}. Attempting line-by-line fallback.")
        packets = []
        for line in raw.splitlines():
            try:
                packets.append(json.loads(line))
            except Exception:
                pass
        return packets


def get_protocol_summary(pcap_path: str) -> dict:
    """
    Returns per-protocol packet counts using tshark -z io,phs.
    """
    if not TSHARK_AVAILABLE:
        return {}
    raw = _run_tshark(["-r", pcap_path, "-q", "-z", "io,phs"])
    protocols = {}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("eth") or "frames:" not in line:
            continue
        parts = line.split()
        # Format: <protocol> <frames:N> <bytes:N>
        proto = parts[0].split(".")[-1].upper() if parts else "?"
        for p in parts:
            if p.startswith("frames:"):
                try:
                    protocols[proto] = int(p.split(":")[1])
                except Exception:
                    pass
    return protocols


def get_io_stats(pcap_path: str) -> dict:
    """Returns overall IO statistics from tshark -z io,stat,0."""
    if not TSHARK_AVAILABLE:
        return {}
    raw = _run_tshark(["-r", pcap_path, "-q", "-z", "io,stat,0"])
    stats = {}
    for line in raw.splitlines():
        if "Frames:" in line and "Bytes:" in line:
            parts = line.split("|")
            for part in parts:
                part = part.strip()
                if part.startswith("Frames:"):
                    try:
                        stats["total_frames"] = int(part.split(":")[1].strip())
                    except Exception:
                        pass
                elif part.startswith("Bytes:"):
                    try:
                        stats["total_bytes"] = int(part.split(":")[1].strip())
                    except Exception:
                        pass
    return stats


# ── Feature Extraction ────────────────────────────────────────────────────────

def extract_features_from_tshark_packet(pkt: dict) -> Optional[np.ndarray]:
    """
    Extract an 84-dimensional feature vector from a tshark JSON packet.
    Slots 0-34 mirror the Scapy extract_features() mapping.
    Slots 35-39 are TShark-exclusive (delta_time, retransmission, TLS, DNS, malformed).
    Slots 40-83 are zeroed (system metrics added at HybridDetector level).
    """
    features = np.zeros(84, dtype=np.float32)
    try:
        layers = pkt.get("_source", {}).get("layers", {})

        # Frame
        frame = layers.get("frame", {})
        features[0]  = float(frame.get("frame.len", 0))
        features[35] = float(frame.get("frame.time_delta", 0))

        # IP
        ip = layers.get("ip", {})
        if ip:
            features[1] = float(ip.get("ip.ttl", 0))
            features[2] = float(ip.get("ip.proto", 0))
            features[3] = float(ip.get("ip.len", 0))
            flags_hex = ip.get("ip.flags", "0x0")
            try:
                flag_val = int(flags_hex, 16)
                features[4] = float(flag_val)
                features[6] = float(flag_val & 0x1)  # MF
                features[7] = float(flag_val & 0x2)  # DF
            except Exception:
                pass

        # TCP
        tcp = layers.get("tcp", {})
        if tcp:
            features[8]  = float(tcp.get("tcp.srcport", 0))
            features[9]  = float(tcp.get("tcp.dstport", 0))
            features[10] = float(tcp.get("tcp.seq", 0))
            features[11] = float(tcp.get("tcp.ack", 0))
            features[12] = float(tcp.get("tcp.hdr_len", 0))
            features[14] = float(tcp.get("tcp.window_size", 0))
            # TCP Flags
            flags_hex = tcp.get("tcp.flags", "0x0")
            try:
                tf = int(flags_hex, 16)
                features[13] = float(tf)
                features[16] = float(tf & 0x01)  # FIN
                features[17] = float(tf & 0x02)  # SYN
                features[18] = float(tf & 0x04)  # RST
                features[19] = float(tf & 0x08)  # PSH
                features[20] = float(tf & 0x10)  # ACK
                features[21] = float(tf & 0x20)  # URG
            except Exception:
                pass
            # TShark-exclusive: TCP retransmission
            analysis = tcp.get("tcp.analysis", {})
            if isinstance(analysis, dict):
                features[36] = 1.0 if "tcp.analysis.retransmission" in analysis else 0.0
            # Stealth TLS burst
            dport = int(tcp.get("tcp.dstport", 0))
            if dport in (443, 8443, 4433) and features[0] > 1400:
                features[34] = 1.0

        # UDP
        udp = layers.get("udp", {})
        if udp:
            features[22] = float(udp.get("udp.srcport", 0))
            features[23] = float(udp.get("udp.dstport", 0))
            features[24] = float(udp.get("udp.length", 0))

        # ICMP
        icmp = layers.get("icmp", {})
        if icmp:
            features[25] = float(icmp.get("icmp.type", 0))
            features[26] = float(icmp.get("icmp.code", 0))

        # TLS handshake
        tls = layers.get("tls", {})
        if tls:
            hs = tls.get("tls.handshake", {})
            features[37] = float(hs.get("tls.handshake.type", 0)) if isinstance(hs, dict) else 0.0

        # DNS
        dns = layers.get("dns", {})
        if dns:
            features[38] = float(dns.get("dns.count.queries", 0))

        # Malformed
        ws = layers.get("_ws.malformed", None)
        features[39] = 1.0 if ws is not None else 0.0

        # Private IP check
        src_ip = ip.get("ip.src", "") if ip else ""
        features[33] = 1.0 if src_ip.startswith(("192.168.", "10.", "172.16.")) else 0.0

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        logger.error(f"TShark feature extraction error: {e}")
        return None
    return features


def _get_ip_from_pkt(pkt: dict) -> tuple[str, str, str]:
    """Return (src_ip, dst_ip, protocol) from a tshark packet dict."""
    layers = pkt.get("_source", {}).get("layers", {})
    ip = layers.get("ip", {})
    src = ip.get("ip.src", "0.0.0.0")
    dst = ip.get("ip.dst", "0.0.0.0")
    if "tcp" in layers:
        proto = "TCP"
    elif "udp" in layers:
        proto = "UDP"
    elif "icmp" in layers:
        proto = "ICMP"
    else:
        proto = "Other"
    return src, dst, proto


# ── Top Talkers ───────────────────────────────────────────────────────────────

def get_top_talkers(packets: list[dict], top_n: int = 10) -> pd.DataFrame:
    """Count packets per source IP and return top N."""
    counts: dict[str, int] = {}
    for pkt in packets:
        layers = pkt.get("_source", {}).get("layers", {})
        ip = layers.get("ip", {})
        src = ip.get("ip.src", "Unknown")
        counts[src] = counts.get(src, 0) + 1
    df = pd.DataFrame(list(counts.items()), columns=["Source IP", "Packet Count"])
    return df.sort_values("Packet Count", ascending=False).head(top_n).reset_index(drop=True)


# ── Suspicious Flow Detection ─────────────────────────────────────────────────

def get_suspicious_flows(packets: list[dict]) -> pd.DataFrame:
    """
    Flag packets with:
      - TCP RST floods
      - Malformed packets
      - Unusual flag combinations (e.g. SYN+FIN, URG+FIN)
      - Retransmissions
    """
    rows = []
    for pkt in packets:
        layers = pkt.get("_source", {}).get("layers", {})
        frame = layers.get("frame", {})
        ip    = layers.get("ip", {})
        tcp   = layers.get("tcp", {})

        reason = []
        flags_hex = tcp.get("tcp.flags", "0x0") if tcp else "0x0"
        try:
            tf = int(flags_hex, 16)
            if (tf & 0x02) and (tf & 0x01):  # SYN+FIN (illegal)
                reason.append("SYN+FIN")
            if (tf & 0x04) and (tf & 0x02):  # RST+SYN
                reason.append("RST+SYN")
            if tf == 0x00:                    # NULL scan
                reason.append("NULL Scan")
            if tf == 0x3F:                    # XMAS scan
                reason.append("XMAS Scan")
        except Exception:
            pass

        if layers.get("_ws.malformed"):
            reason.append("Malformed")
        if tcp and isinstance(tcp.get("tcp.analysis"), dict):
            if "tcp.analysis.retransmission" in tcp["tcp.analysis"]:
                reason.append("Retransmission")

        if reason:
            rows.append({
                "Time":       frame.get("frame.time_relative", "?"),
                "Src IP":     ip.get("ip.src", "?") if ip else "?",
                "Dst IP":     ip.get("ip.dst", "?") if ip else "?",
                "Dst Port":   tcp.get("tcp.dstport", "?") if tcp else "?",
                "Flags":      flags_hex,
                "Reason":     ", ".join(reason),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Time", "Src IP", "Dst IP", "Dst Port", "Flags", "Reason"]
    )


# ── Protocol Frequency ────────────────────────────────────────────────────────

def get_protocol_distribution(packets: list[dict]) -> pd.DataFrame:
    """Count packets per L7 protocol."""
    counts: dict[str, int] = {}
    proto_keys = {
        "http": "HTTP", "http2": "HTTP/2", "tls": "TLS/HTTPS",
        "dns": "DNS", "ftp": "FTP", "ssh": "SSH",
        "icmp": "ICMP", "arp": "ARP", "tcp": "TCP (Other)", "udp": "UDP (Other)",
    }
    for pkt in packets:
        layers = pkt.get("_source", {}).get("layers", {})
        matched = False
        for key, label in proto_keys.items():
            if key in layers:
                counts[label] = counts.get(label, 0) + 1
                matched = True
                break
        if not matched:
            counts["Other"] = counts.get("Other", 0) + 1
    df = pd.DataFrame(list(counts.items()), columns=["Protocol", "Count"])
    return df.sort_values("Count", ascending=False).reset_index(drop=True)


# ── Bandwidth Timeline ────────────────────────────────────────────────────────

def get_bandwidth_timeline(packets: list[dict], bucket_size: int = 100) -> pd.DataFrame:
    """Aggregate packet sizes over sequential buckets to approximate bandwidth."""
    rows = []
    for i in range(0, len(packets), bucket_size):
        bucket = packets[i:i + bucket_size]
        total_bytes = sum(
            int(p.get("_source", {}).get("layers", {}).get("frame", {}).get("frame.len", 0))
            for p in bucket
        )
        rows.append({"Packet#": i + bucket_size // 2, "Bytes": total_bytes})
    return pd.DataFrame(rows)


# ── PCAP Replay Mode ──────────────────────────────────────────────────────────

def replay_pcap(pcap_path: str, detector, max_packets: int = 5000,
                progress_callback=None) -> list[dict]:
    """
    Parse PCAP and feed each packet's feature vector into HybridDetector.
    Returns list of result dicts (same format as anomalies.log).

    Args:
        pcap_path:         Path to .pcap or .pcapng file
        detector:          HybridDetector instance
        max_packets:       Cap to avoid OOM
        progress_callback: Optional callable(current, total) for UI progress
    """
    packets = parse_pcap_json(pcap_path, max_packets=max_packets)
    results = []
    total = len(packets)
    for i, pkt in enumerate(packets):
        features = extract_features_from_tshark_packet(pkt)
        if features is None:
            continue
        src, dst, proto = _get_ip_from_pkt(pkt)
        try:
            result = detector.predict(features[:79], src_ip=src, dst_ip=dst, protocol=proto)
            results.append(result)
        except Exception as e:
            logger.debug(f"Replay predict error at pkt {i}: {e}")
        if progress_callback and i % 50 == 0:
            progress_callback(i, total)
    logger.info(f"Replay complete: {len(results)} packets processed from {pcap_path}")
    return results


# ── Scapy vs TShark Drift Validation ─────────────────────────────────────────

def validate_scapy_vs_tshark(scapy_features: np.ndarray, pcap_path: str) -> dict:
    """
    Write a single-packet pcap (via scapy dump) and compare TShark extraction.
    Returns a dict with drift metrics.
    NOTE: Requires scapy to be available.
    """
    if not TSHARK_AVAILABLE:
        return {"error": "TShark not available"}
    try:
        packets = parse_pcap_json(pcap_path, max_packets=1)
        if not packets:
            return {"error": "No packets in PCAP"}
        tshark_features = extract_features_from_tshark_packet(packets[0])
        if tshark_features is None:
            return {"error": "Feature extraction failed"}

        # Compare first 35 slots (shared between Scapy and TShark)
        s = scapy_features[:35].astype(float)
        t = tshark_features[:35].astype(float)
        diff = np.abs(s - t)
        mismatch_indices = np.where(diff > 1e-3)[0].tolist()
        return {
            "max_drift":        float(diff.max()),
            "mean_drift":       float(diff.mean()),
            "mismatch_indices": mismatch_indices,
            "drift_ok":         len(mismatch_indices) == 0,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Full Forensic Report ──────────────────────────────────────────────────────

@dataclass
class ForensicReport:
    pcap_path:   str
    total_packets: int             = 0
    total_bytes:   int             = 0
    duration_s:    float           = 0.0
    protocol_dist: pd.DataFrame    = field(default_factory=pd.DataFrame)
    top_talkers:   pd.DataFrame    = field(default_factory=pd.DataFrame)
    suspicious:    pd.DataFrame    = field(default_factory=pd.DataFrame)
    bandwidth:     pd.DataFrame    = field(default_factory=pd.DataFrame)
    io_stats:      dict            = field(default_factory=dict)
    raw_packets:   list            = field(default_factory=list)
    ai_results:    list            = field(default_factory=list)


def analyze_pcap(pcap_path: str, detector=None,
                 max_packets: int = 10000,
                 progress_callback=None) -> ForensicReport:
    """
    Full forensic analysis of a PCAP file.
    Optionally runs AI detection if detector is provided.
    """
    report = ForensicReport(pcap_path=pcap_path)

    if not TSHARK_AVAILABLE:
        logger.warning("TShark not available — forensic analysis disabled.")
        return report

    packets = parse_pcap_json(pcap_path, max_packets=max_packets)
    report.raw_packets   = packets
    report.total_packets = len(packets)

    # IO stats from tshark -z
    io = get_io_stats(pcap_path)
    report.io_stats    = io
    report.total_bytes = io.get("total_bytes", 0)

    # Compute duration from first/last packet timestamps
    try:
        times = [
            float(p.get("_source", {}).get("layers", {})
                   .get("frame", {}).get("frame.time_epoch", 0))
            for p in packets
        ]
        times = [t for t in times if t > 0]
        if len(times) >= 2:
            report.duration_s = max(times) - min(times)
    except Exception:
        pass

    report.protocol_dist = get_protocol_distribution(packets)
    report.top_talkers   = get_top_talkers(packets)
    report.suspicious    = get_suspicious_flows(packets)
    report.bandwidth     = get_bandwidth_timeline(packets)

    # Optional AI replay
    if detector is not None:
        report.ai_results = replay_pcap(
            pcap_path, detector,
            max_packets=max_packets,
            progress_callback=progress_callback,
        )

    return report


# ── CLI Self-Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TShark PCAP Forensic Analyzer")
    parser.add_argument("--pcap", required=True, help="Path to .pcap / .pcapng file")
    parser.add_argument("--max", type=int, default=1000, help="Max packets to analyze")
    args = parser.parse_args()

    print(f"TShark available : {TSHARK_AVAILABLE}")
    print(f"TShark version   : {get_tshark_version()}")
    if not TSHARK_AVAILABLE:
        print("Install: brew install wireshark  (macOS)")
        print("         sudo apt install tshark  (Linux)")
        raise SystemExit(1)

    print(f"\nAnalyzing: {args.pcap}")
    report = analyze_pcap(args.pcap, max_packets=args.max)
    print(f"  Packets  : {report.total_packets:,}")
    print(f"  Bytes    : {report.total_bytes:,}")
    print(f"  Duration : {report.duration_s:.2f}s")
    print(f"\nProtocol Distribution:\n{report.protocol_dist}")
    print(f"\nTop Talkers:\n{report.top_talkers}")
    print(f"\nSuspicious Flows ({len(report.suspicious)}):")
    print(report.suspicious.head(10))
