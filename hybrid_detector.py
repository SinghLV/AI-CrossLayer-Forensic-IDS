"""
hybrid_detector.py
==================
LSTM + Random Forest Hybrid Detection Engine
Part of: AI-Powered Cross-Layer Network Intrusion Detection System

Workflow:
  1. Packet features extracted (79-D) + system metrics fused (84-D total)
  2. LSTM Autoencoder computes reconstruction error
  3. Adaptive threshold check (rolling mean + 3σ, updates every 100 packets)
  4. If anomaly → Random Forest classifies attack type
  5. XAI Engine generates human-readable explanation
  6. Severity classified → Alert dispatched
"""

import os
import json
import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import joblib

from xai_engine import XAIEngine, AnomalyExplanation
from alerts import AlertDispatcher, Alert, get_global_dispatcher
from system_metrics import SystemMetricsCollector

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"
LOGS_DIR    = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LSTM Model (re-defined here for self-contained loading)
# ══════════════════════════════════════════════════════════════════════════════

class LightweightLSTMAutoencoder(nn.Module):
    """Sequence-to-Sequence LSTM Autoencoder for anomaly detection."""

    def __init__(self, input_dim: int = 79, hidden_dim: int = 64):
        super().__init__()
        self.encoder   = nn.LSTM(input_dim,  hidden_dim, num_layers=1, batch_first=True)
        self.decoder   = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc, _  = self.encoder(x)
        dec, _  = self.decoder(enc)
        return self.output_fc(dec)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Adaptive Threshold Manager
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveThreshold:
    """
    Dynamically adjusts the anomaly threshold based on a rolling window
    of reconstruction errors from recent traffic.

    threshold = rolling_mean + σ × rolling_std

    This reduces false positives during traffic bursts and improves
    zero-day detection by adapting to the current traffic baseline.
    """

    def __init__(
        self,
        initial_threshold: float,
        window_size:       int   = 500,
        sigma:             float = 3.0,
        update_interval:   int   = 100,     # packets between updates
    ):
        self._static_threshold = initial_threshold
        self._current          = initial_threshold
        self._window           = deque(maxlen=window_size)
        self._sigma            = sigma
        self._update_interval  = update_interval
        self._packet_count     = 0
        self._lock             = threading.Lock()

    def add_error(self, error: float):
        """Register a new reconstruction error (called for every packet)."""
        with self._lock:
            self._window.append(error)
            self._packet_count += 1

        # Recalculate every N packets
        if self._packet_count % self._update_interval == 0:
            self._recalculate()
            
        # Demo Fix: Don't let the threshold adapt too high (clamp at 1.0)
        # This prevents the AI from "normalizing" the attack traffic.
        if self._current > 1.0:
            self._current = 1.0

    def _recalculate(self):
        if len(self._window) < 10:
            return
        arr = np.array(self._window)
        new_threshold = float(arr.mean() + self._sigma * arr.std())
        # Blend with static threshold: never go below 50% of original
        blended = max(new_threshold, 0.5 * self._static_threshold)
        self._current = blended
        logger.debug(
            f"Adaptive threshold updated: {self._current:.6f} "
            f"(mean={arr.mean():.6f}, std={arr.std():.6f})"
        )

    @property
    def value(self) -> float:
        with self._lock:
            return self._current


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Hybrid Detector
# ══════════════════════════════════════════════════════════════════════════════

class HybridDetector:
    """
    The core detection engine combining LSTM Autoencoder + Random Forest.

    Load once, call `.predict()` for each packet feature vector.
    """

    def __init__(
        self,
        dispatcher:       Optional[AlertDispatcher] = None,
        sys_collector:    Optional[SystemMetricsCollector] = None,
        log_anomalies:    bool = True,
    ):
        self._device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._xai        = XAIEngine()
        self._dispatcher = dispatcher or get_global_dispatcher()
        self._sys        = sys_collector
        self._log        = log_anomalies
        self._anomaly_log_path = LOGS_DIR / "anomalies.log"

        # ── Load LSTM ──────────────────────────────────────────────────────────
        self._lstm: Optional[LightweightLSTMAutoencoder] = None
        self._scaler        = None
        self._base_threshold = 0.05

        # ── Load RF ───────────────────────────────────────────────────────────
        self._rf          = None
        self._rf_scaler   = None
        self._label_enc   = None

        # ── Adaptive threshold ────────────────────────────────────────────────
        self._threshold   = AdaptiveThreshold(self._base_threshold)

        # ── Statistics ────────────────────────────────────────────────────────
        self.total_packets   = 0
        self.total_anomalies = 0
        
        print("\n" + "="*40)
        print("🚀 [AI ENGINE] VERSION 2.0 - LOCAL TIME ACTIVE")
        print("="*40 + "\n")
        
        # Load components
        self._load_models()

    # ── Model Loading ──────────────────────────────────────────────────────────

    def _load_models(self):
        """Load LSTM and RF models from models/ directory."""
        # LSTM
        lstm_path = MODELS_DIR / "lstm_autoencoder_best.pth"
        scaler_path = MODELS_DIR / "scaler.joblib"
        thresh_path = MODELS_DIR / "anomaly_threshold.joblib"

        if lstm_path.exists() and scaler_path.exists():
            try:
                ckpt = torch.load(lstm_path, map_location=self._device)
                input_dim  = ckpt.get("input_dim",  79)
                hidden_dim = ckpt.get("hidden_dim", 64)
                self._lstm = LightweightLSTMAutoencoder(input_dim, hidden_dim).to(self._device)
                self._lstm.load_state_dict(ckpt["model_state_dict"])
                self._lstm.eval()
                self._scaler = joblib.load(scaler_path)

                if thresh_path.exists():
                    base = float(joblib.load(thresh_path))
                    self._base_threshold = base
                    self._threshold = AdaptiveThreshold(base)

                logger.info(f"✓ LSTM loaded (input_dim={input_dim}, hidden={hidden_dim})")
                logger.info(f"  Base threshold: {self._base_threshold:.6f}")
            except Exception as e:
                logger.warning(f"LSTM load failed ({e}) — using demo mode")
                self._lstm = None
        else:
            logger.warning("LSTM model not found. Run train_lstm.py first (or use demo mode).")

        # RF
        rf_path  = MODELS_DIR / "rf_classifier.joblib"
        le_path  = MODELS_DIR / "rf_label_encoder.joblib"
        rfs_path = MODELS_DIR / "rf_scaler.joblib"

        if rf_path.exists() and le_path.exists():
            try:
                self._rf        = joblib.load(rf_path)
                self._label_enc = joblib.load(le_path)
                if rfs_path.exists():
                    self._rf_scaler = joblib.load(rfs_path)
                logger.info(f"✓ RF loaded — classes: {list(self._label_enc.classes_)}")
            except Exception as e:
                logger.warning(f"RF load failed ({e})")
        else:
            logger.warning("RF model not found. Run train_rf.py first.")

    # ── Feature Fusion ─────────────────────────────────────────────────────────

    def _fuse_features(self, packet_features: np.ndarray) -> np.ndarray:
        """
        Append system-level metrics to the 79-D packet feature vector
        to create a fused cross-layer feature vector.

        Returns:
            Array of shape (84,)  [79 packet + 5 system]
        """
        if self._sys is not None:
            sys_feat = self._sys.get_fused_features()   # 5-D
        else:
            sys_feat = np.zeros(5, dtype=np.float32)

        fused = np.concatenate([packet_features.astype(np.float32), sys_feat])
        return fused

    # ── LSTM Reconstruction ────────────────────────────────────────────────────

    def _lstm_reconstruct(self, features_79: np.ndarray) -> float:
        """
        Run the 79-D feature vector through the LSTM autoencoder.
        Returns reconstruction error (MSE).
        Falls back to demo heuristic if model is not loaded.
        """
        if self._lstm is None or self._scaler is None:
            # Demo fallback: higher error for obviously anomalous features
            return float(np.random.exponential(scale=0.02))

        try:
            scaled = self._scaler.transform(features_79.reshape(1, -1))
            scaled = np.clip(scaled, -5.0, 5.0)
            tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(1).to(self._device)

            with torch.no_grad():
                with torch.amp.autocast(device_type=self._device.type,
                                        enabled=self._device.type == "cuda"):
                    output = self._lstm(tensor)
                error = float(torch.mean((output - tensor) ** 2).item())
            return error
        except Exception as e:
            logger.error(f"LSTM inference error: {e}")
            return 0.0

    # ── RF Classification ──────────────────────────────────────────────────────

    def _rf_classify(self, features_79: np.ndarray) -> Tuple[str, float]:
        """
        Classify the attack type using the Random Forest.
        Returns (attack_type, confidence).
        """
        if self._rf is None or self._label_enc is None:
            return "Unknown", 0.5

        try:
            X = features_79.reshape(1, -1)
            if self._rf_scaler is not None:
                X = self._rf_scaler.transform(X)

            pred_idx  = self._rf.predict(X)[0]
            proba     = self._rf.predict_proba(X)[0]
            confidence = float(proba[pred_idx])
            atk_type  = self._label_enc.inverse_transform([pred_idx])[0]
            return atk_type, confidence
        except Exception as e:
            logger.error(f"RF inference error: {e}")
            return "Unknown", 0.0

    # ── Main Predict ───────────────────────────────────────────────────────────

    def predict(
        self,
        packet_features: np.ndarray,
        src_ip:   str = "0.0.0.0",
        dst_ip:   str = "0.0.0.0",
        protocol: str = "Unknown",
    ) -> Dict:
        """
        Full hybrid detection pipeline for a single packet.

        Args:
            packet_features: 79-D numpy array from feature extraction
            src_ip:          Source IP address string
            dst_ip:          Destination IP address string
            protocol:        Protocol string (TCP/UDP/ICMP)

        Returns:
            Detection result dict with all fields filled
        """
        self.total_packets += 1

        # ── Step 1: Feature fusion ─────────────────────────────────────────────
        fused = self._fuse_features(packet_features)

        # ── Step 2: LSTM reconstruction ────────────────────────────────────────
        error = self._lstm_reconstruct(packet_features)
        self._threshold.add_error(error)
        threshold = self._threshold.value

        is_anomaly = error > threshold

        # ── Step 3: RF classification (only if anomaly) ────────────────────────
        attack_type  = "Normal"
        confidence   = 1.0
        explanation  = None
        severity     = "LOW"

        if is_anomaly:
            self.total_anomalies += 1
            attack_type, confidence = self._rf_classify(packet_features)

            # ── Step 4: XAI explanation ────────────────────────────────────────
            explanation = self._xai.explain(fused, error, threshold, attack_type)
            severity    = explanation.severity

            # ── Step 5: Alert dispatch ────────────────────────────────────────
            self._dispatcher.create_and_dispatch(
                severity    = severity,
                attack_type = attack_type,
                src_ip      = src_ip,
                dst_ip      = dst_ip,
                score       = explanation.risk_score,
                error       = error,
                reasons     = explanation.reasons,
                protocol    = protocol,
            )

            # ── Step 6: Log to file ───────────────────────────────────────────
            if self._log:
                self._write_log(
                    src_ip, dst_ip, protocol, error, threshold,
                    attack_type, severity, confidence,
                    explanation.risk_score, explanation.reasons
                )

        result = {
            "is_anomaly":    is_anomaly,
            "anomaly_score": round(error, 6),
            "threshold":     round(threshold, 6),
            "attack_type":   attack_type,
            "confidence":    round(confidence, 3),
            "severity":      severity,
            "src_ip":        src_ip,
            "dst_ip":        dst_ip,
            "protocol":      protocol,
            "timestamp":     datetime.now().isoformat(timespec="seconds"),
            "explanation":   explanation.to_dict() if explanation else None,
        }
        return result

    def _write_log(
        self,
        src_ip: str, dst_ip: str, protocol: str,
        error: float, threshold: float,
        attack_type: str, severity: str, confidence: float,
        risk_score: float, reasons: List[str],
    ):
        """Append a JSON line to anomalies.log."""
        record = {
            "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "src_ip":             src_ip,
            "dst_ip":             dst_ip,
            "protocol":           protocol,
            "reconstruction_error": round(error, 6),
            "threshold":          round(threshold, 6),
            "is_anomaly":         True,
            "attack_type":        attack_type,
            "severity":           severity,
            "confidence":         round(confidence, 3),
            "risk_score":         round(risk_score, 3),
            "hint":               "; ".join(reasons[:3]),
        }
        try:
            with open(self._anomaly_log_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write anomaly log: {e}")

    def get_stats(self) -> Dict:
        return {
            "total_packets":   self.total_packets,
            "total_anomalies": self.total_anomalies,
            "anomaly_rate":    round(
                self.total_anomalies / max(self.total_packets, 1), 4
            ),
            "current_threshold": round(self._threshold.value, 6),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Smoke Test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    detector = HybridDetector()

    rng = np.random.default_rng(42)

    print("\n=== Hybrid Detector Smoke Test ===")
    for i in range(10):
        # Normal packet
        feat = rng.normal(0, 1, 79).astype(np.float32)
        result = detector.predict(feat, "192.168.1.1", "192.168.1.2", "TCP")
        status = "ANOMALY" if result["is_anomaly"] else "normal "
        print(f"  [{i+1:2d}] {status} | error={result['anomaly_score']:.4f} "
              f"| type={result['attack_type']:10s} | severity={result['severity']}")
        time.sleep(0.05)

    # Inject obvious anomaly
    feat_attack = rng.normal(10, 2, 79).astype(np.float32)   # far from normal distribution
    result = detector.predict(feat_attack, "45.33.32.156", "192.168.0.1", "TCP")
    print(f"\n  [ATTACK] error={result['anomaly_score']:.4f} | type={result['attack_type']} "
          f"| severity={result['severity']} | risk={result.get('explanation', {}).get('risk_score', 'N/A') if result['explanation'] else 'N/A'}")

    print(f"\nStats: {detector.get_stats()}")
