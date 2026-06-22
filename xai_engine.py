"""
xai_engine.py
=============
Explainable AI (XAI) Engine
Part of: AI-Powered Cross-Layer Network Intrusion Detection System

Generates human-readable explanations for every detected anomaly using:
  - Rule-based checks on 79 packet features
  - System metric correlations (CPU/IRQ spikes)
  - Attack-type-specific reasoning
  - Severity scoring
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Severity tiers ────────────────────────────────────────────────────────────
SEVERITY_LOW      = "LOW"
SEVERITY_MEDIUM   = "MEDIUM"
SEVERITY_HIGH     = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"

# ── Feature index map (matches detect_realtime.py / realtime_detector.py) ─────
FI = {
    "pkt_len":      0,   # total packet length
    "ip_ttl":       1,   # IP TTL
    "ip_proto":     2,   # IP protocol number
    "ip_len":       3,   # IP layer length
    "ip_flags":     4,   # IP flags
    "ip_frag":      5,   # fragmentation offset
    "mf_flag":      6,   # more-fragments flag
    "df_flag":      7,   # don't-fragment flag
    "tcp_sport":    8,   # TCP source port
    "tcp_dport":    9,   # TCP destination port
    "tcp_seq":      10,
    "tcp_ack":      11,
    "tcp_dataofs":  12,
    "tcp_flags":    13,  # TCP flags (int)
    "tcp_window":   14,
    "tcp_len":      15,
    "tcp_fin":      16,
    "tcp_syn":      17,
    "tcp_rst":      18,
    "tcp_psh":      19,
    "tcp_ack_f":    20,
    "tcp_urg":      21,
    "udp_sport":    22,
    "udp_dport":    23,
    "udp_len":      24,
    "icmp_type":    25,
    "icmp_code":    26,
    "icmp_len":     27,
    "timestamp":    28,
    "pkt_norm":     29,  # normalised packet size
    "tcp_ratio":    30,
    "udp_ratio":    31,
    "icmp_ratio":   32,
    "direction":    33,  # 0=incoming, 1=outgoing
    # 34-78: zero-padded / fused system metrics at inference time
    "cpu_usage":    34,  # fused from system_metrics
    "ram_usage":    35,
    "net_irq":      36,
    "irq_spike":    37,
    "cpu_spike":    38,
}

# ── Thresholds for rule-based checks ─────────────────────────────────────────
EPHEMERAL_PORT_THRESHOLD = 49151
HIGH_PORT_THRESHOLD      = 32768
MIN_NORMAL_PACKET        = 60
MAX_NORMAL_PACKET        = 1500
NORMAL_TTL_RANGE         = range(32, 129)
COMMON_TCP_FLAGS         = {0x02, 0x12, 0x10, 0x18, 0x11, 0x04}  # SYN,SYN-ACK,ACK,PSH-ACK,FIN-ACK,RST
NORMAL_PROTOCOLS         = {1, 6, 17}   # ICMP, TCP, UDP
HIGH_CPU_THRESHOLD       = 0.80         # 80% (normalised 0-1)
HIGH_IRQ_THRESHOLD       = 0.50         # 50% of 10K irq/s normalisation


@dataclass
class AnomalyExplanation:
    """Structured output of the XAI engine for a single anomaly."""
    reconstruction_error: float
    attack_type:          str
    severity:             str
    reasons:              List[str] = field(default_factory=list)
    risk_score:           float = 0.0   # 0.0 – 1.0

    def to_string(self) -> str:
        reason_lines = "\n  - ".join(self.reasons) if self.reasons else "No specific pattern identified"
        return (
            f"🔴 Anomaly Detected\n"
            f"   Attack Type  : {self.attack_type}\n"
            f"   Severity     : {self.severity}\n"
            f"   Risk Score   : {self.risk_score:.2f}\n"
            f"   LSTM Error   : {self.reconstruction_error:.6f}\n"
            f"   Reasons:\n  - {reason_lines}"
        )

    def to_dict(self) -> Dict:
        return {
            "attack_type":          self.attack_type,
            "severity":             self.severity,
            "risk_score":           round(self.risk_score, 3),
            "reconstruction_error": round(self.reconstruction_error, 6),
            "reasons":              self.reasons,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Rule Checkers
# ══════════════════════════════════════════════════════════════════════════════

def _check_packet_size(feat: np.ndarray, reasons: List[str]) -> int:
    """Returns severity contribution (0-2)."""
    pkt_len = feat[FI["pkt_len"]]
    if pkt_len < MIN_NORMAL_PACKET:
        reasons.append(f"Abnormally small packet ({int(pkt_len)} bytes < {MIN_NORMAL_PACKET})")
        return 1
    if pkt_len > MAX_NORMAL_PACKET:
        reasons.append(f"Oversized packet ({int(pkt_len)} bytes > {MAX_NORMAL_PACKET} MTU)")
        return 2
    return 0


def _check_ttl(feat: np.ndarray, reasons: List[str]) -> int:
    ttl = int(feat[FI["ip_ttl"]])
    if ttl not in NORMAL_TTL_RANGE:
        reasons.append(f"Abnormal TTL ({ttl}) — may indicate spoofed origin or OS fingerprinting")
        return 1
    return 0


def _check_protocol(feat: np.ndarray, reasons: List[str]) -> int:
    proto = int(feat[FI["ip_proto"]])
    if proto not in NORMAL_PROTOCOLS:
        reasons.append(f"Unusual IP protocol ({proto}) — not ICMP/TCP/UDP")
        return 2
    return 0


def _check_tcp_flags(feat: np.ndarray, reasons: List[str]) -> int:
    """Detect SYN flood, NULL scan, XMAS scan, RST flood."""
    score   = 0
    tcp_flags = int(feat[FI["tcp_flags"]])
    syn       = int(feat[FI["tcp_syn"]])
    ack_f     = int(feat[FI["tcp_ack_f"]])
    rst       = int(feat[FI["tcp_rst"]])
    fin       = int(feat[FI["tcp_fin"]])
    urg       = int(feat[FI["tcp_urg"]])
    psh       = int(feat[FI["tcp_psh"]])

    # SYN with no ACK → potential SYN flood
    if syn == 1 and ack_f == 0:
        reasons.append("SYN packet with no ACK — possible SYN flood attempt")
        score += 2

    # NULL scan (no flags set)
    if tcp_flags == 0:
        reasons.append("NULL scan detected (all TCP flags = 0)")
        score += 2

    # XMAS scan (FIN+URG+PSH set)
    if fin == 1 and urg == 1 and psh == 1:
        reasons.append("XMAS scan detected (FIN+URG+PSH flags)")
        score += 2

    # RST flood
    if rst == 1:
        reasons.append("TCP RST flag set — possible RST injection or port scan")
        score += 1

    if tcp_flags not in COMMON_TCP_FLAGS and tcp_flags != 0:
        reasons.append(f"Uncommon TCP flag combination (0x{tcp_flags:02x})")
        score += 1

    return score


def _check_ports(feat: np.ndarray, reasons: List[str]) -> int:
    score = 0
    tcp_sport = int(feat[FI["tcp_sport"]])
    tcp_dport = int(feat[FI["tcp_dport"]])
    udp_sport = int(feat[FI["udp_sport"]])
    udp_dport = int(feat[FI["udp_dport"]])

    # Only flag DESTINATION ports in ephemeral range (servers usually shouldn't listen here)
    for port_name, port_val in [("TCP dst", tcp_dport), ("UDP dst", udp_dport)]:
        if port_val > EPHEMERAL_PORT_THRESHOLD:
            reasons.append(f"{port_name} port {port_val} is in the ephemeral/dynamic range (unusual for target)")
            score += 1
            
    # Common attack target ports
    for port_name, port_val in [("TCP src", tcp_sport), ("TCP dst", tcp_dport),
                                ("UDP src", udp_sport), ("UDP dst", udp_dport)]:
        if port_val in {22, 23, 3389, 445, 1433, 3306, 27017}:
            reasons.append(f"{port_name} port {port_val} is a high-value attack target")
            score += 2

    return score


def _check_fragmentation(feat: np.ndarray, reasons: List[str]) -> int:
    mf_flag = int(feat[FI["mf_flag"]])
    frag    = int(feat[FI["ip_frag"]])
    if mf_flag == 1:
        reasons.append("IP 'More Fragments' flag set — possible fragmentation attack")
        return 2
    if frag > 0:
        reasons.append(f"Non-zero fragment offset ({frag}) — possible evasion technique")
        return 1
    return 0


def _check_encrypted_stealth(feat: np.ndarray, reasons: List[str]) -> int:
    """
    Heuristic detection of suspicious encrypted / stealth traffic
    without decrypting packets.
    """
    score = 0
    tcp_dport = int(feat[FI["tcp_dport"]])
    pkt_norm  = feat[FI["pkt_norm"]]

    # TLS/HTTPS on unusual ports
    if tcp_dport in {443, 8443, 4433, 8080} and pkt_norm > 0.8:
        reasons.append(f"Large HTTPS-like packet on port {tcp_dport} — possible covert channel")
        score += 1

    # Consistently maximum-size packets → possible data exfiltration
    if pkt_norm >= 0.99:
        reasons.append("Maximum-size packet (near MTU) — possible data exfiltration burst")
        score += 1

    return score


def _check_system_metrics(feat: np.ndarray, reasons: List[str]) -> int:
    """Cross-layer: correlate packet anomaly with CPU/IRQ spike."""
    score = 0
    if len(feat) > FI["cpu_usage"] and feat[FI["cpu_usage"]] > HIGH_CPU_THRESHOLD:
        reasons.append(f"CPU usage spike ({feat[FI['cpu_usage']]*100:.1f}%) detected during network event")
        score += 1

    if len(feat) > FI["net_irq"] and feat[FI["net_irq"]] > HIGH_IRQ_THRESHOLD:
        reasons.append("Elevated network IRQ frequency detected")
        score += 1

    if len(feat) > FI["irq_spike"] and feat[FI["irq_spike"]] > 0.5:
        reasons.append("IRQ spike flag active — kernel interrupt storm correlates with attack")
        score += 1

    if len(feat) > FI["cpu_spike"] and feat[FI["cpu_spike"]] > 0.5:
        reasons.append("CPU spike flag active — processing anomaly in kernel space")
        score += 1

    return score


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Attack-type specific reasoning
# ══════════════════════════════════════════════════════════════════════════════

_ATTACK_REASONS: Dict[str, str] = {
    "DoS":        "High-volume packet pattern consistent with Denial-of-Service flood",
    "PortScan":   "Sequential/random port probe pattern detected — reconnaissance activity",
    "BruteForce": "Repeated connection attempts to auth-sensitive port — brute force likely",
    "Probe":      "Low-rate systematic scan — network mapping or vulnerability probe",
    "Unknown":    "Unclassified threat pattern — manual review recommended",
    "Normal":     "Traffic within normal parameters — false positive possible",
}


def _get_attack_reason(attack_type: str) -> str:
    return _ATTACK_REASONS.get(attack_type, "Anomalous behaviour detected")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Severity Classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_severity(
    reconstruction_error: float,
    threshold:            float,
    rule_score:           int,
    attack_type:          str,
) -> Tuple[str, float]:
    """
    Assign a severity level and risk score (0–1).

    Factors:
      - How far above the threshold the error is
      - Cumulative rule score from XAI checks
      - Known high-risk attack types
    """
    # Error ratio above threshold (1.0 = exactly at threshold)
    ratio = reconstruction_error / max(threshold, 1e-9)

    # Base risk from error ratio
    base_risk = min(1.0, (ratio - 1.0) / 5.0) if ratio > 1.0 else 0.0

    # Add rule score contribution (each rule point = 5% extra risk)
    rule_risk = min(0.4, rule_score * 0.05)

    # Attack type multiplier
    atk_mult = {"DoS": 1.3, "BruteForce": 1.2, "PortScan": 1.0,
                 "Probe": 0.9, "Unknown": 1.1, "Normal": 0.3}
    mult = atk_mult.get(attack_type, 1.0)

    risk_score = min(1.0, (base_risk + rule_risk) * mult)

    if risk_score >= 0.75:
        severity = SEVERITY_CRITICAL
    elif risk_score >= 0.50:
        severity = SEVERITY_HIGH
    elif risk_score >= 0.25:
        severity = SEVERITY_MEDIUM
    else:
        severity = SEVERITY_LOW

    return severity, round(risk_score, 3)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Main XAI Engine
# ══════════════════════════════════════════════════════════════════════════════

class XAIEngine:
    """
    Explainable AI engine for the Cross-Layer IDS.

    Usage:
        engine = XAIEngine()
        explanation = engine.explain(features, recon_error, threshold, attack_type)
        print(explanation.to_string())
    """

    def explain(
        self,
        features:             np.ndarray,
        reconstruction_error: float,
        threshold:            float,
        attack_type:          str = "Unknown",
    ) -> AnomalyExplanation:
        """
        Generate a full explanation for a detected anomaly.

        Args:
            features:             79+ dimensional feature vector
            reconstruction_error: LSTM reconstruction MSE
            threshold:            Current adaptive threshold
            attack_type:          Attack class from Random Forest

        Returns:
            AnomalyExplanation with reasons, severity, and risk score
        """
        reasons    = []
        rule_score = 0

        # ── Run all rule checkers ──────────────────────────────────────────────
        rule_score += _check_packet_size(features, reasons)
        rule_score += _check_ttl(features, reasons)
        rule_score += _check_protocol(features, reasons)
        rule_score += _check_tcp_flags(features, reasons)
        rule_score += _check_ports(features, reasons)
        rule_score += _check_fragmentation(features, reasons)
        rule_score += _check_encrypted_stealth(features, reasons)
        rule_score += _check_system_metrics(features, reasons)

        # ── Add attack-type context ────────────────────────────────────────────
        atk_reason = _get_attack_reason(attack_type)
        if atk_reason:
            reasons.insert(0, atk_reason)

        # ── If no specific rules fired, add generic reason ────────────────────
        if len(reasons) <= 1 and attack_type != "Normal":
            reasons.append(
                f"LSTM reconstruction error ({reconstruction_error:.6f}) "
                f"exceeds adaptive threshold ({threshold:.6f}) by "
                f"{((reconstruction_error/max(threshold,1e-9)-1)*100):.1f}%"
            )
            
        # ── Handle true "Normal" traffic cleanly ──────────────────────────────
        if attack_type == "Normal" and reconstruction_error < threshold:
            reasons = ["Traffic conforms to expected baseline behavior."]
            rule_score = 0

        # ── Severity & risk score ──────────────────────────────────────────────
        severity, risk_score = classify_severity(
            reconstruction_error, threshold, rule_score, attack_type
        )

        return AnomalyExplanation(
            reconstruction_error = reconstruction_error,
            attack_type          = attack_type,
            severity             = severity,
            reasons              = reasons,
            risk_score           = risk_score,
        )

    def batch_explain(
        self,
        feature_matrix:       np.ndarray,
        reconstruction_errors: np.ndarray,
        threshold:             float,
        attack_types:          List[str],
    ) -> List[AnomalyExplanation]:
        """Generate explanations for a batch of anomalies."""
        return [
            self.explain(feature_matrix[i], reconstruction_errors[i], threshold, attack_types[i])
            for i in range(len(feature_matrix))
        ]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Smoke Test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np

    rng    = np.random.default_rng(42)
    engine = XAIEngine()

    # ── Simulate a DoS attack packet ─────────────────────────────────────────
    feat = np.zeros(79, dtype=np.float32)
    feat[FI["pkt_len"]]  = 64          # small packet
    feat[FI["ip_ttl"]]   = 10          # suspicious TTL
    feat[FI["ip_proto"]] = 6           # TCP
    feat[FI["tcp_syn"]]  = 1           # SYN set
    feat[FI["tcp_ack_f"]]= 0           # no ACK → SYN flood
    feat[FI["tcp_dport"]]= 80          # HTTP
    feat[FI["cpu_usage"]]= 0.92        # 92% CPU
    feat[FI["irq_spike"]]= 1.0         # IRQ spike

    exp = engine.explain(
        features=feat,
        reconstruction_error=0.45,
        threshold=0.05,
        attack_type="DoS",
    )

    print(exp.to_string())
    print("\nDict output:")
    import json
    print(json.dumps(exp.to_dict(), indent=2))
