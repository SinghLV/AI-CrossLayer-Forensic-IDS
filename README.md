# 🛡️ AI-Powered Cross-Layer Intrusion Detection System

A professional-grade research platform for **Zero-Day Network Intrusion Detection**. This system fuses **Deep Learning (LSTM Autoencoders)** with **System-Level Telemetry (CPU/RAM/IRQ)** and **Advanced TShark Forensics** to identify sophisticated attacks.

---

## 🚀 Key Features
*   **True Cross-Layer Analysis:** Correlates network traffic spikes with hardware-level metrics (CPU load, RAM usage, and IRQ spikes).
*   **Wireshark/TShark Forensics:** Advanced engine for offline PCAP analysis and AI-powered traffic replay.
*   **Explainable AI (XAI):** Translates 84-D feature vectors into human-readable security hints.
*   **Academic Ready:** Phased toggle system (1–4) for incremental faculty demonstrations.

---

## ⚙️ Quick Start

### 1. Requirements
```bash
pip install -r requirements.txt
```
*Note: Install TShark (`brew install wireshark` or `apt install tshark`) for Forensic features.*

### 2. Run Dashboard
```bash
streamlit run dashboard.py
```

### 3. Run Real-time Sniffer
```bash
# Requires admin/sudo
sudo python realtime_detector.py --iface en0
```

---

## 📜 Academic Reference & Strategy
For students/researchers, please check:
*   **`academic_progress_phases.md`**: Detailed presentation script and phase-by-phase strategy.
*   **`config.py`**: Switch between phases (1 to 4) to control UI feature visibility.

---

## 📄 License
This project is for educational and research purposes. Use only on networks you have explicit permission to monitor.
