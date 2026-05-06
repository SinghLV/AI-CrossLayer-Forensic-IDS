"""
config.py
=========
Global configuration for the Intelligent Threat Detector.
Used to control feature visibility for phased faculty presentations.
"""

# --- Phased Presentation Levels ---
# 1: Basic (Live Monitoring Only)
# 2: AI Core (Live + System Metrics + Attack Analytics)
# 3: XAI & Reports (Live + System + Analytics + Reports)
# 4: Full (All tabs including Packet Forensics)

PROJECT_PHASE = 2  # Set this to 1, 2, 3, or 4 based on your presentation progress

# --- Other Global Settings ---
DEBUG_MODE = False
DEFAULT_MAX_ROWS = 1000
DEFAULT_REFRESH_RATE = 5
