# 🛡️ AI-Powered Cross-Layer Intrusion Detection System (v2.0)

A professional-grade research platform for **Zero-Day Network Intrusion Detection**. This system fuses **Deep Learning (LSTM Autoencoders)** with **System-Level Telemetry (CPU/RAM/IRQ)** and **Advanced TShark Forensics** to identify sophisticated attacks.

---

## 🏛️ Project Structure
This repository contains two versions of the detection system:

1.  **[zero_day_detection_v2](./zero_day_detection_v2)** (Recommended)
    *   **New:** Packet Forensics engine (Wireshark/TShark).
    *   **New:** Phased Presentation Framework for academic demos.
    *   **New:** 84-D feature vector extraction matching academic standards.
    *   **New:** Integrated PDF Report Generator.
2.  **[zero_day_detection_project_complete](./zero_day_detection_project_complete)** (Legacy)
    *   Original research version with Neo4j graph integration.

---

## 🚀 Key Features (v2.0)
*   **True Cross-Layer Analysis:** Correlates network traffic spikes with hardware-level metrics (CPU load, RAM usage, and IRQ spikes).
*   **Wireshark/TShark Forensics:** Advanced engine for offline PCAP analysis and AI-powered traffic replay.
*   **Explainable AI (XAI):** Translates 84-D feature vectors into human-readable security hints.
*   **Academic Ready:** Phased toggle system (1–4) for incremental faculty demonstrations.

---

## ⚙️ Quick Start (v2.0)

### 1. Requirements
```bash
cd zero_day_detection_v2
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
*   **`zero_day_detection_v2/academic_progress_phases.md`**: Detailed presentation script and phase-by-phase strategy.
*   **`zero_day_detection_v2/config.py`**: Switch between phases (1 to 4) to control UI feature visibility.

---

## 🔍 Legacy Project Overview (v1.0)
The original version of this system focused on LSTM Autoencoders and Neo4j relationship mapping.

### Legacy Workflow:
1. **Train**: `python src/train_model.py`
2. **Detect**: `python src/detect_realtime.py`
3. **Graph**: Uses Neo4j on `localhost:7687` for network relationship analysis.

---

## 📄 License
This project is for educational and research purposes. Use only on networks you have explicit permission to monitor.
