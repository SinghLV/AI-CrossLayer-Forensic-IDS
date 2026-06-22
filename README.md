# AI-Powered Cross-Layer Intrusion Detection System (X-IDS)

A professional-grade, enterprise-ready research platform for **Zero-Day Network Intrusion Detection**. This system implements a multi-layer defense strategy by correlating **Network Traffic Analysis** with **System-Level Telemetry** to identify sophisticated threats that traditional IDSs miss. 

It features a cutting-edge **Enterprise SOC Dashboard** built with Streamlit, providing real-time Threat Intelligence and Explainable AI insights.

---

## 🚀 System Highlights

### 1. Enterprise SOC Dashboard
The entire user interface is designed to mimic a high-end Security Operations Center (SOC). It features:
*   **Advanced Threat Visualization**: Dynamic telemetry updates and responsive metrics.
*   **Live Threat Intelligence Engine**: A SIEM-style query bar to search and filter live packet anomalies in real-time.
*   **AI Co-Pilot**: An integrated chat interface to ask questions about current network threats and receive automated mitigation strategies.

### 2. Hybrid AI Engine (Unsupervised & Supervised)
The "Brain" of the system uses a sophisticated dual-model approach:
*   **LSTM Autoencoder (Deep Learning)**: Trained on normal traffic patterns to identify "Zero-Day" anomalies based on reconstruction error.
*   **Random Forest (Classifier)**: Provides rapid classification of known attack vectors (DDoS, Brute Force, Scans) using an 84-dimensional feature vector.

### 3. Advanced Packet Forensics (TShark Engine)
Integrated a high-performance forensic layer using the **TShark CLI**. This allows for:
*   **Deep Protocol Inspection**: Automated metadata extraction for HTTP, DNS, TLS, SSH, and FTP.
*   **Flow Anomaly Detection**: Identifying suspicious TCP flag combinations and transmission patterns.

### 4. True Cross-Layer Correlation
Unlike standard IDS solutions, X-IDS monitors both the **Packet Layer** and the **Hardware Layer**. By observing system interrupt anomalies and CPU/RAM spikes, it detects stealthy attacks like cryptomining or background data exfiltration that evade packet-only inspection.

### 5. Explainable AI (XAI) & Insights
Every detection is accompanied by a human-readable insight. The XAI engine translates complex feature vectors into clear security hints, explaining *why* the AI flagged a specific event.

---

## 📂 Project Structure

Here is a breakdown of where everything exists in the repository:

*   **`dashboard.py`**: The main entry point for the frontend. Contains the entire Streamlit SOC interface, including the SIEM query engine and interactive Plotly graphs.
*   **`realtime_detector.py`**: The background packet sniffing engine using Scapy. Captures live traffic and pushes it to the AI for inference.
*   **`tshark_detector.py`**: An advanced alternative to `realtime_detector.py` that utilizes Wireshark's TShark engine for deep packet inspection.
*   **`hybrid_detector.py`**: The core AI logic. Loads the pre-trained machine learning models (LSTM, Random Forest) and executes the inference pipeline on incoming packets.
*   **`xai_engine.py`**: The Explainable AI module. Analyzes the outputs of the hybrid detector and generates human-readable "Hints" and risk scores.
*   **`config.py`**: Central configuration file managing system thresholds, file paths, and presentation phases.
*   **`demo_mode.py`**: A simulator that injects fake threat telemetry into the logs for UI testing and demonstration purposes without requiring a live attack.
*   **`logs/`**: Directory where the `anomalies.log` file is stored. The dashboard reads directly from this log in real-time.

---

## ⚙️ How to Install and Run

Follow these instructions to clone, setup, and run the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/SinghLV/AI-CrossLayer-Forensic-IDS.git
cd AI-CrossLayer-Forensic-IDS
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: Ensure `TShark` (Wireshark command-line tool) is installed on your system PATH to enable the advanced forensic features.*

### 4. Launch the Enterprise SOC Dashboard
Run the following command to start the user interface:
```bash
streamlit run dashboard.py
```
This will open the dashboard in your default web browser (usually at `http://localhost:8501`).

### 5. Start the Real-Time Detection Engine (Optional)
If you want to monitor actual live network traffic (instead of using the Demo Mode simulator in the UI), open a **new terminal window**, activate your virtual environment, and run the packet sniffer:

```bash
# Note: Packet sniffing usually requires Administrator or root privileges.
sudo python realtime_detector.py
```
*Note: You can edit `config.py` to target a specific network interface (e.g., `en0` or `eth0`).*

---

## 📊 Using the Dashboard

1. **Live Monitor**: View the incoming stream of anomalies. Use the **SIEM Search** bar to dynamically filter logs (e.g., `severity=CRITICAL` or `attack_type=DoS`).
2. **Demo Mode**: If you aren't capturing live traffic, simply toggle **"DEMO MODE"** in the left sidebar to simulate a live network attack feed.
3. **AI Co-Pilot**: Use the chat window in the sidebar to ask for mitigation strategies regarding specific alerts.

---

## 📄 License & Disclaimer
This platform is developed for advanced network security research and educational purposes. Use only on networks where you have explicit authorization to monitor traffic.
