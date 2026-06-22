"""
alerts.py
=========
Multi-Channel Alert System
Part of: AI-Powered Cross-Layer Network Intrusion Detection System

Provides:
  - Colored terminal alerts (ANSI + colorama)
  - Streamlit-native alert banners
  - In-memory alert history ring buffer
  - Sound alert (cross-platform optional beep)

Severity Colors:
  GREEN  (#00ff88) = LOW
  YELLOW (#ffff00) = MEDIUM
  ORANGE (#ff8800) = HIGH
  RED    (#ff0033) = CRITICAL
"""

import sys
import time
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ── Try colorama for Windows terminal colors ──────────────────────────────────
try:
    import colorama
    colorama.init(autoreset=True)
    _COLORAMA_AVAILABLE = True
except ImportError:
    _COLORAMA_AVAILABLE = False

# ── ANSI color codes ──────────────────────────────────────────────────────────
ANSI = {
    "RESET":    "\033[0m",
    "BOLD":     "\033[1m",
    "RED":      "\033[91m",
    "ORANGE":   "\033[38;5;208m",
    "YELLOW":   "\033[93m",
    "GREEN":    "\033[92m",
    "CYAN":     "\033[96m",
    "WHITE":    "\033[97m",
    "DIM":      "\033[2m",
}

# ── Severity → display config ─────────────────────────────────────────────────
SEVERITY_CONFIG = {
    "LOW": {
        "ansi":      ANSI["GREEN"],
        "emoji":     "🟢",
        "hex":       "#00ff88",
        "bg_hex":    "#003322",
        "st_type":   "success",
        "sound_hz":  400,
        "sound_ms":  100,
    },
    "MEDIUM": {
        "ansi":      ANSI["YELLOW"],
        "emoji":     "🟡",
        "hex":       "#ffff00",
        "bg_hex":    "#333300",
        "st_type":   "warning",
        "sound_hz":  600,
        "sound_ms":  200,
    },
    "HIGH": {
        "ansi":      ANSI["ORANGE"],
        "emoji":     "🟠",
        "hex":       "#ff8800",
        "bg_hex":    "#331a00",
        "st_type":   "error",
        "sound_hz":  800,
        "sound_ms":  300,
    },
    "CRITICAL": {
        "ansi":      ANSI["RED"],
        "emoji":     "🔴",
        "hex":       "#ff0033",
        "bg_hex":    "#220011",
        "st_type":   "error",
        "sound_hz":  1200,
        "sound_ms":  500,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Alert Data Model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Alert:
    """A single security alert record."""
    timestamp:   str
    severity:    str
    attack_type: str
    src_ip:      str
    dst_ip:      str
    score:       float
    reasons:     List[str]
    error:       float
    protocol:    str = "Unknown"
    alert_id:    str = field(default_factory=lambda: str(time.time_ns()))

    @property
    def title(self) -> str:
        cfg = SEVERITY_CONFIG.get(self.severity, SEVERITY_CONFIG["LOW"])
        return f"{cfg['emoji']} [{self.severity}] {self.attack_type} — {self.src_ip} → {self.dst_ip}"

    def to_dict(self) -> Dict:
        return {
            "id":          self.alert_id,
            "timestamp":   self.timestamp,
            "severity":    self.severity,
            "attack_type": self.attack_type,
            "src_ip":      self.src_ip,
            "dst_ip":      self.dst_ip,
            "risk_score":  round(self.score, 3),
            "recon_error": round(self.error, 6),
            "protocol":    self.protocol,
            "reasons":     self.reasons,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Terminal Alert
# ══════════════════════════════════════════════════════════════════════════════

class TerminalAlert:
    """Prints beautifully colored alerts to the terminal."""

    @staticmethod
    def fire(alert: Alert):
        cfg   = SEVERITY_CONFIG.get(alert.severity, SEVERITY_CONFIG["LOW"])
        color = cfg["ansi"]
        bold  = ANSI["BOLD"]
        reset = ANSI["RESET"]
        dim   = ANSI["DIM"]

        border = "═" * 70
        print(f"\n{color}{bold}{border}{reset}")
        print(f"{color}{bold}  {cfg['emoji']}  ALERT: {alert.attack_type.upper():<20}  SEVERITY: {alert.severity}{reset}")
        print(f"{color}{border}{reset}")
        print(f"  {bold}Time     :{reset} {alert.timestamp}")
        print(f"  {bold}Source   :{reset} {color}{alert.src_ip}{reset}")
        print(f"  {bold}Target   :{reset} {alert.dst_ip}")
        print(f"  {bold}Protocol :{reset} {alert.protocol}")
        print(f"  {bold}Risk     :{reset} {color}{alert.score:.2f}/1.0{reset}")
        print(f"  {bold}LSTM Err :{reset} {alert.error:.6f}")
        if alert.reasons:
            print(f"  {bold}Reasons  :{reset}")
            for reason in alert.reasons[:4]:   # show max 4
                print(f"    {dim}• {reason}{reset}")
        print(f"{color}{border}{reset}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Streamlit Alert Banner (call from within st context)
# ══════════════════════════════════════════════════════════════════════════════

def streamlit_alert_banner(alert: Alert, container=None):
    """
    Render a styled alert banner inside Streamlit.

    Args:
        alert:     The Alert to display
        container: Optional Streamlit container (column, expander, etc.)
                   Defaults to using st directly.
    """
    try:
        import streamlit as st
        target = container if container is not None else st
        cfg    = SEVERITY_CONFIG.get(alert.severity, SEVERITY_CONFIG["LOW"])

        html = f"""
        <div style="
            background-color: {cfg['bg_hex']};
            border-left: 4px solid {cfg['hex']};
            border-radius: 6px;
            padding: 12px 16px;
            margin: 6px 0;
            font-family: 'Courier New', monospace;
        ">
            <div style="color:{cfg['hex']}; font-weight:bold; font-size:15px;">
                {cfg['emoji']} {alert.attack_type} — {alert.severity}
            </div>
            <div style="color:#cccccc; font-size:13px; margin-top:4px;">
                <b>🕒</b> {alert.timestamp} &nbsp;|&nbsp;
                <b>⚡</b> {alert.src_ip} → {alert.dst_ip} &nbsp;|&nbsp;
                <b>🎯</b> Risk: {alert.score:.2f}
            </div>
            <div style="color:#aaaaaa; font-size:12px; margin-top:6px;">
                {' &nbsp;•&nbsp; '.join(alert.reasons[:3])}
            </div>
        </div>
        """
        target.markdown(html, unsafe_allow_html=True)
    except ImportError:
        logger.warning("Streamlit not available; cannot render alert banner")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Sound Alert (Optional)
# ══════════════════════════════════════════════════════════════════════════════

def sound_alert(severity: str):
    """
    Play a non-blocking terminal beep scaled to severity.
    Only works on platforms with `winsound` (Windows) or `afplay` (macOS).
    """
    cfg = SEVERITY_CONFIG.get(severity, SEVERITY_CONFIG["LOW"])

    def _beep():
        try:
            import platform
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(cfg["sound_hz"], cfg["sound_ms"])
            elif platform.system() == "Darwin":
                import subprocess
                # macOS: use say command for an audible alert
                subprocess.run(["say", f"Alert: {severity}"],
                               capture_output=True, timeout=2)
            else:
                # Linux: write BEL character to terminal
                print("\a", end="", flush=True)
        except Exception:
            pass   # Sound is optional; never crash over it

    threading.Thread(target=_beep, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Alert History Ring Buffer
# ══════════════════════════════════════════════════════════════════════════════

class AlertHistory:
    """
    Thread-safe in-memory ring buffer of recent alerts.
    Used by the Streamlit dashboard for live feed display.
    """

    def __init__(self, maxlen: int = 500):
        self._buffer: deque = deque(maxlen=maxlen)
        self._lock   = threading.Lock()
        self._counts: Dict[str, int] = {
            "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0,
            "DoS": 0, "PortScan": 0, "BruteForce": 0, "Probe": 0, "Unknown": 0,
        }

    def add(self, alert: Alert):
        with self._lock:
            self._buffer.append(alert)
            sev = alert.severity
            atk = alert.attack_type
            self._counts[sev]  = self._counts.get(sev, 0) + 1
            self._counts[atk]  = self._counts.get(atk, 0) + 1

    def get_recent(self, n: int = 50) -> List[Alert]:
        with self._lock:
            alerts = list(self._buffer)
        return list(reversed(alerts[-n:]))

    def get_counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def to_dataframe(self):
        """Convert history to pandas DataFrame for dashboard charts."""
        try:
            import pandas as pd
            with self._lock:
                rows = [a.to_dict() for a in list(self._buffer)]
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows)
        except ImportError:
            return []

    def clear(self):
        with self._lock:
            self._buffer.clear()
            for k in self._counts:
                self._counts[k] = 0


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Unified Alert Dispatcher
# ══════════════════════════════════════════════════════════════════════════════

class AlertDispatcher:
    """
    Central dispatcher that routes an Alert to all configured channels.

    Channels (all optional):
      - terminal_alert (always enabled)
      - history        (AlertHistory instance)
      - sound          (disabled by default)
      - streamlit      (only if called within a Streamlit context)
    """

    def __init__(
        self,
        history:       Optional[AlertHistory] = None,
        enable_sound:  bool = False,
        min_severity:  str  = "LOW",
    ):
        self.history      = history or AlertHistory()
        self.enable_sound = enable_sound
        self._severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        self._min_rank    = self._severity_rank.get(min_severity, 0)
        self._terminal    = TerminalAlert()

    def dispatch(self, alert: Alert):
        """Route alert to all configured channels."""
        rank = self._severity_rank.get(alert.severity, 0)
        if rank < self._min_rank:
            return   # Filtered below minimum threshold

        # Terminal
        self._terminal.fire(alert)

        # History buffer
        self.history.add(alert)

        # Sound (non-blocking)
        if self.enable_sound:
            sound_alert(alert.severity)

        logger.info(
            f"[{alert.severity}] {alert.attack_type} | "
            f"{alert.src_ip}→{alert.dst_ip} | score={alert.score:.2f}"
        )

    def create_and_dispatch(
        self,
        severity:    str,
        attack_type: str,
        src_ip:      str,
        dst_ip:      str,
        score:       float,
        error:       float,
        reasons:     List[str],
        protocol:    str = "Unknown",
    ) -> Alert:
        """Convenience: build an Alert and dispatch it in one call."""
        alert = Alert(
            timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            severity    = severity,
            attack_type = attack_type,
            src_ip      = src_ip,
            dst_ip      = dst_ip,
            score       = score,
            error       = error,
            reasons     = reasons,
            protocol    = protocol,
        )
        self.dispatch(alert)
        return alert


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Global singleton (for use across modules)
# ══════════════════════════════════════════════════════════════════════════════

_global_history    = AlertHistory(maxlen=1000)
_global_dispatcher = AlertDispatcher(history=_global_history, enable_sound=False)


def get_global_dispatcher() -> AlertDispatcher:
    return _global_dispatcher

def get_global_history() -> AlertHistory:
    return _global_history


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Smoke Test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    disp = AlertDispatcher(enable_sound=False)

    test_cases = [
        ("LOW",      "Normal",     "192.168.1.5",  "192.168.1.1",   0.10, 0.001),
        ("MEDIUM",   "Probe",      "10.0.0.99",    "10.0.0.1",      0.35, 0.045),
        ("HIGH",     "PortScan",   "172.16.0.50",  "172.16.255.255",0.65, 0.120),
        ("CRITICAL", "DoS",        "45.33.32.156", "192.168.0.1",   0.95, 0.480),
    ]

    for severity, atk, src, dst, score, error in test_cases:
        disp.create_and_dispatch(
            severity    = severity,
            attack_type = atk,
            src_ip      = src,
            dst_ip      = dst,
            score       = score,
            error       = error,
            reasons     = ["Test reason 1", "Test reason 2"],
            protocol    = "TCP",
        )
        time.sleep(0.3)

    print(f"\nHistory counts: {disp.history.get_counts()}")
    print(f"Recent alerts  : {len(disp.history.get_recent())}")
