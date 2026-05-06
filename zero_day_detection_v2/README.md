# Intelligent Threat Detector — AI-Powered Cross-Layer IDS

## 🚀 Overview
The **Intelligent Threat Detector (v2.0)** is a professional-grade research platform for Zero-Day Network Intrusion Detection. It fuses **Deep Learning (LSTM Autoencoders)** with **System-Level Telemetry (CPU/RAM/IRQ)** to identify sophisticated attacks that traditional IDSs miss.

### ✨ Key Features
1. **True Cross-Layer Analysis:** Correlates network traffic spikes with hardware-level metrics (CPU load, RAM usage, and IRQ/Interrupt spikes) to detect stealthy mining or ransomware.
2. **Wireshark/TShark Forensics:** New **Packet Forensics** engine for offline PCAP analysis, protocol deep-dives, and AI-powered traffic replay.
3. **Hybrid AI Engine:**
   - **LSTM Autoencoder:** Unsupervised anomaly detection for novel (zero-day) attacks.
   - **Random Forest:** Supervised classification for known attack vectors.
4. **Explainable AI (XAI):** Provides human-readable hints (e.g., "Suspicious TCP flag combo detected") for every anomaly.
5. **SOC-Grade Dashboard:** Enterprise-level UI with adaptive Dark/Light themes, live graphs, and threat visualization.

---

## ⚙️ System Requirements

### 1. Python Environment
```bash
pip install -r requirements.txt
```

### 2. Wireshark/TShark (New in v2.0)
For the **Packet Forensics** tab to work, TShark must be installed on your system:
- **macOS:** `brew install wireshark`
- **Linux:** `sudo apt install tshark`
- **Windows:** Install from [wireshark.org](https://www.wireshark.org/download.html)

---

---

## 🏃 Execution Guide

### 1. Training Phase
The system will generate synthetic training data if no dataset is found.
```bash
python train_lstm.py  # Train Anomaly Detection
python train_rf.py    # Train Attack Classifier
```

### 2. Live Monitoring
**Terminal 1 (Dashboard):**
```bash
streamlit run dashboard.py
```

**Terminal 2 (Real-time Sniffer):**
*Requires sudo/admin for packet capture.*
```bash
# macOS/Linux
sudo python realtime_detector.py --iface en0

# Windows (Admin PowerShell)
python realtime_detector.py
```

### 3. Forensic Analysis
1. Open the dashboard.
2. Navigate to the **📡 Packet Forensics** tab.
3. Upload any `.pcap` or `.pcapng` file.
4. (Optional) Toggle **"Run AI anomaly detection"** to score the PCAP using the live models.

---

## 🏗️ Architecture & Modules

- **`dashboard.py`**: Enterprise Streamlit UI with 6 specialized tabs.
- **`wireshark_integration.py`**: TShark CLI wrapper for JSON parsing and forensic metadata extraction.
- **`realtime_detector.py`**: Multi-threaded Scapy sniffer for real-time traffic analysis.
- **`system_metrics.py`**: Monitors `/proc/interrupts` and `psutil` for cross-layer correlation.
- **`hybrid_detector.py`**: The "Brain" — integrates LSTM, RF, and XAI logic.
- **`xai_engine.py`**: Translates 84-D feature vectors into security insights.
- **`report_generator.py`**: Generates SOC-ready PDF reports with embedded charts.
- **`neo4j_visualizer.py`**: (Optional) Graph intelligence for complex threat actors.

---

## 📜 Academic Reference
**Topic:** *AI-Powered Cross-Layer Network Intrusion Detection System using Network Traffic and CPU Interrupt Analysis.*

This project implements a multi-layer defense strategy by monitoring both the **Network Layer** (Packets) and the **Hardware Layer** (CPU/Interrupts) to identify anomalies through temporal patterns.
