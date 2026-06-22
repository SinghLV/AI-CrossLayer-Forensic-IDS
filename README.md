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

### 3. Hybrid AI Engine (Unsupervised & Supervised)
The "Brain" of the system uses a sophisticated dual-model approach:
*   **LSTM Autoencoder (Deep Learning)**: Trained on normal traffic patterns to identify "Zero-Day" anomalies based on reconstruction error.
*   **Random Forest (Classifier)**: Provides rapid classification of known attack vectors (DDoS, Brute Force, Scans) using an 84-dimensional feature vector.

### 4. Knowledge Graph & Relationship Mapping (Neo4j)
Integrates a graph-based visualizer to map relationships between IPs, MAC addresses, and protocols. This allows security analysts to see "islands" of communication and identify lateral movement within a network.

### 5. Explainable AI (XAI) & Insights
Every detection is accompanied by a human-readable insight. The XAI engine translates complex feature vectors into clear security hints, explaining *why* the AI flagged a specific event.

### 6. SOC-Grade Reporting
Includes an automated **PDF Report Generator** that compiles anomaly timelines, feature importance graphs, and protocol distributions into a professional security audit document.

---

## 🛠️ Technical Stack
*   **Core**: Python 3.9+
*   **Traffic Engine**: Scapy & TShark
*   **AI/ML**: PyTorch (LSTM), Scikit-learn (Random Forest)
*   **Graph Engine**: Neo4j (Relationship Mapping)
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
*   **Knowledge Graph**: Dynamic visualization of network relationships.

---

## 📄 License & Usage
This platform is developed for advanced network security research. Use only on networks where you have explicit authorization to monitor traffic.
