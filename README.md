# 🔍 AI-Powered Cross-Layer Intrusion Detection System (X-IDS)

A professional-grade research platform for **Zero-Day Network Intrusion Detection**. This system implements a multi-layer defense strategy by correlating **Network Traffic Analysis** with **System-Level Telemetry** (CPU Load, RAM, and Interrupt Spikes) to identify sophisticated threats that traditional IDSs miss.

---

## 🚀 System Highlights

### 1. True Cross-Layer Correlation
Unlike standard IDS solutions, X-IDS monitors both the **Packet Layer** and the **Hardware Layer**. By observing `/proc/interrupts` and IRQ spikes, it can detect stealthy attacks like cryptomining or background data shredding that don't always show up in packet headers.

### 2. Advanced Packet Forensics (TShark Engine)
Integrated a high-performance forensic layer using the **TShark CLI**. This allows for:
*   **Deep Protocol Inspection**: Automated metadata extraction for HTTP, DNS, TLS, SSH, and FTP.
*   **Historical AI Replay**: Score historical PCAP traffic through the AI pipeline to identify missed threats.
*   **Flow Anomaly Detection**: Identifying suspicious TCP flag combinations and transmission patterns.

### 3. Hybrid AI Architecture
The "Brain" of the system uses a dual-model approach:
*   **LSTM Autoencoder (Unsupervised)**: Trained on normal traffic patterns to identify "Zero-Day" anomalies based on reconstruction error.
*   **Random Forest (Supervised)**: Provides rapid classification of known attack vectors (DDoS, Brute Force, Scans).

### 4. Explainable AI (XAI)
Every detection is accompanied by a human-readable insight. The XAI engine translates complex 84-dimensional feature vectors into clear security hints (e.g., *"Suspiciously high entropy in packet payload detected"*).

---

## 🛠️ Technical Stack
*   **Core**: Python 3.9+
*   **Traffic Engine**: Scapy & TShark
*   **AI/ML**: PyTorch (LSTM), Scikit-learn (Random Forest)
*   **Dashboard**: Streamlit (SOC-Grade UI)
*   **Metrics**: Psutil & System-level Telemetry

---

## ⚙️ Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```
*Note: Ensure TShark is installed on your system PATH to enable forensic features.*

### 2. Launch the SOC Dashboard
```bash
streamlit run dashboard.py
```

### 3. Start Real-time Detection
```bash
# Requires admin/sudo for packet capture
sudo python realtime_detector.py --iface en0
```

---

## 📊 Performance & Metrics
The system provides a comprehensive set of metrics including:
*   **Reconstruction Error Timeline**: Visualizing anomaly severity over time.
*   **Protocol Distribution**: Real-time breakdown of network traffic.
*   **System Health**: Monitoring CPU/RAM correlation with network spikes.
*   **Forensic Reports**: Auto-generated PDF reports for security audits.

---

## 📄 License & Usage
This platform is developed for advanced network security research. Use only on networks where you have explicit authorization to monitor traffic.
