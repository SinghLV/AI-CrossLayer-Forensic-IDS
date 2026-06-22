"""
Configuration settings for the X-IDS Presentation.
Modify PROJECT_PHASE to control feature visibility during faculty demonstrations.
"""

# Current Project Phase (1 to 4)
# Phase 1: Foundation (Dashboard & System Health)
# Phase 2: Intelligent Detection (AI Anomaly & Reconstruction)
# Phase 3: Forensic Deep-Dive (TShark Analysis & Top Talkers)
# Phase 4: Full Suite (Explainable AI & Professional Reporting)
PROJECT_PHASE = 4 

# Feature Visibilit Mapping
PHASE_FEATURES = {
    1: ["Overview Dashboard", "Real-time Packet Monitoring", "CPU/RAM/IRQ Metrics"],
    2: ["LSTM Anomaly Detection", "Reconstruction Error Timeline", "Dynamic Thresholding"],
    3: ["TShark Forensic Engine", "Protocol Distribution", "Network Relationship Graph"],
    4: ["Explainable AI (XAI)", "PDF Audit Reports", "Forensic Replay Engine"]
}
