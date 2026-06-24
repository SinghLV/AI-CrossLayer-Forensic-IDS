"""
dashboard.py — AI-Powered Cross-Layer IDS Dashboard
5 tabs: Live Monitoring | Attack Analytics | Threat Graph | System Metrics | Reports
"""
import json, time, os, threading, sys
import psutil
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# Load presentation phase config
try:
    from config import PROJECT_PHASE
except ImportError:
    PROJECT_PHASE = 4  # Default to full

BASE_DIR  = Path(__file__).parent
LOG_FILE  = BASE_DIR / "logs" / "anomalies.log"
DEMO_FILE = BASE_DIR / "demo_data" / "demo_running.flag"

st.set_page_config(page_title="Intelligent Threat Detector", layout="wide")

# ── Dark SOC CSS ──────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

/* ─── KEYFRAME ANIMATIONS ─── */
@keyframes pulse-led {
  0%, 100% { opacity: 1; box-shadow: 0 0 8px currentColor; }
  50% { opacity: 0.35; box-shadow: 0 0 2px currentColor; }
}
@keyframes shimmer-border {
  0% { background-position: 0% 50%; }
  100% { background-position: 200% 50%; }
}
@keyframes fade-in-up {
  from { opacity: 0; transform: translateY(14px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes ambient-pulse {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 0.7; }
}
@keyframes scan-line {
  0% { transform: translateY(-100vh); }
  100% { transform: translateY(100vh); }
}
@keyframes gradient-shift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* ─── GLOBAL ─── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

/* Ambient background glows */
[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  background:
    radial-gradient(ellipse 700px 500px at 15% 85%, rgba(226,0,130,0.07) 0%, transparent 70%),
    radial-gradient(ellipse 600px 400px at 85% 15%, rgba(0,204,102,0.04) 0%, transparent 70%),
    radial-gradient(ellipse 400px 300px at 50% 50%, rgba(100,0,200,0.03) 0%, transparent 70%);
  pointer-events: none; z-index: 0;
  animation: ambient-pulse 10s ease-in-out infinite;
}

/* ─── CUSTOM SCROLLBARS ─── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, #e20082, #ff8800);
  border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover { background: #e20082; }

/* ─── SIDEBAR ─── */
[data-testid="stSidebar"] section { padding: 0 !important; background: transparent !important; }
[data-testid="stSidebar"] > div:first-child {
  background: rgba(8,8,12,0.9) !important;
  backdrop-filter: blur(24px) saturate(1.5) !important;
  -webkit-backdrop-filter: blur(24px) saturate(1.5) !important;
  border-right: 1px solid rgba(226,0,130,0.08) !important;
}

/* Sidebar logo */
.sb-logo {
  background: transparent;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  padding: 32px 16px 24px; text-align: center; margin-bottom: 8px;
}
.sb-logo-title {
  font-size: 1.1rem; font-weight: 800; letter-spacing: 2px;
  background: linear-gradient(135deg, #ffffff 30%, #e20082);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  display: block; margin-top: 6px;
}
.sb-logo-sub {
  font-size: .65rem; color: #e20082;
  letter-spacing: 4px; margin-top: 6px; font-weight: 700; opacity: 0.8;
}

/* Sidebar cards */
.sb-card {
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 18px;
  padding: 16px 18px; margin: 8px 10px;
  transition: all 0.3s ease;
}
.sb-card:hover {
  background: rgba(255,255,255,0.04);
  border-color: rgba(226,0,130,0.2);
  box-shadow: 0 6px 24px rgba(226,0,130,0.08);
}
.sb-card-title {
  font-size: .58rem; font-weight: 800; letter-spacing: 3px;
  color: #e20082; opacity: 0.7; text-transform: uppercase; margin-bottom: 12px;
}

/* Animated status dots */
.dot-green { width:8px;height:8px;border-radius:50%;background:#00cc66;display:inline-block;color:#00cc66;animation:pulse-led 2s ease-in-out infinite; }
.dot-red { width:8px;height:8px;border-radius:50%;background:#ff3366;display:inline-block;color:#ff3366;animation:pulse-led 1.5s ease-in-out infinite; }
.dot-yellow { width:8px;height:8px;border-radius:50%;background:#ffcc00;display:inline-block;color:#ffcc00;animation:pulse-led 1.8s ease-in-out infinite; }

.sb-stat { display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.04); }
.sb-stat:last-child { border-bottom:none; }
.sb-stat-label { font-size:.72rem;color:rgba(255,255,255,0.4);font-weight:500; }
.sb-stat-val { font-size:.82rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--text-color); }

/* Custom HTML progress bars */
.res-bar-wrap { margin: 10px 0; }
.res-bar-label {
  display:flex;justify-content:space-between;
  font-size:.68rem;font-weight:700;letter-spacing:1.5px;
  color:rgba(255,255,255,0.4);margin-bottom:6px;text-transform:uppercase;
}
.res-bar-track { background:rgba(255,255,255,0.05);border-radius:100px;height:5px;overflow:hidden; }
.res-bar-fill { height:100%;border-radius:100px;background:linear-gradient(90deg,#e20082,#ff8800);box-shadow:0 0 10px rgba(226,0,130,0.4);transition:width 0.6s ease; }
.res-bar-fill.safe { background:linear-gradient(90deg,#00cc66,#00aa55);box-shadow:0 0 10px rgba(0,204,102,0.4); }
.res-bar-fill.warn { background:linear-gradient(90deg,#ffcc00,#ff8800);box-shadow:0 0 10px rgba(255,204,0,0.4); }

/* ─── METRIC CARDS ─── */
.metric-card {
  background: rgba(255,255,255,0.035);
  backdrop-filter: blur(20px) saturate(1.3);
  -webkit-backdrop-filter: blur(20px) saturate(1.3);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 24px;
  padding: 28px 24px; text-align:left; margin:6px;
  box-shadow: 0 4px 30px rgba(0,0,0,0.5);
  position:relative; overflow:hidden;
  transition: transform 0.35s cubic-bezier(0.4,0,0.2,1), box-shadow 0.35s cubic-bezier(0.4,0,0.2,1), border-color 0.35s ease, background 0.35s ease;
  animation: fade-in-up 0.5s ease-out both;
}
.metric-card:hover {
  transform: translateY(-7px);
  background: rgba(255,255,255,0.055);
  border-color: rgba(226,0,130,0.28);
  box-shadow: 0 20px 60px rgba(226,0,130,0.14), 0 4px 20px rgba(0,0,0,0.5), inset 0 0 40px rgba(226,0,130,0.025);
}
.metric-card::before {
  content:"";position:absolute;top:0;left:0;width:100%;height:3px;
  background:linear-gradient(90deg,#e20082,#ff8800,#9933ff,#e20082);
  background-size:300% 100%;
  animation:shimmer-border 4s linear infinite;
  box-shadow:0 0 16px rgba(226,0,130,0.5);
}
.metric-card::after {
  content:"";position:absolute;top:3px;left:0;width:100%;height:45%;
  background:linear-gradient(180deg,rgba(255,255,255,0.035) 0%,transparent 100%);
  pointer-events:none;
}
.metric-val {
  font-size:2.8rem;font-weight:800;color:#fff;
  font-family:'Inter',sans-serif;line-height:1.1;
  text-shadow:0 0 30px rgba(255,255,255,0.08);
  position:relative;z-index:1;
}
.metric-lbl {
  font-size:.62rem;color:rgba(255,255,255,0.4);margin-top:10px;
  font-weight:800;text-transform:uppercase;letter-spacing:3px;
  position:relative;z-index:1;
}

/* ─── SECTION HEADERS ─── */
.section-hdr {
  display:flex;align-items:center;gap:12px;
  margin:28px 0 18px;
}
.section-hdr-line {
  flex:1;height:1px;
  background:linear-gradient(90deg,rgba(226,0,130,0.5),transparent);
}
.section-hdr-line.right {
  background:linear-gradient(270deg,rgba(226,0,130,0.5),transparent);
}
.section-hdr-label {
  font-size:.6rem;font-weight:800;letter-spacing:3.5px;
  color:#e20082;text-transform:uppercase;
}

/* ─── SEVERITY BADGES ─── */
.sev-CRITICAL { color:#fff;font-weight:800;background:linear-gradient(135deg,#ff0055,#cc0044);padding:3px 12px;border-radius:10px;box-shadow:0 0 14px rgba(255,0,85,0.35);font-size:0.7rem;letter-spacing:0.8px; }
.sev-HIGH { color:#fff;font-weight:800;background:linear-gradient(135deg,#ff8800,#cc6600);padding:3px 12px;border-radius:10px;box-shadow:0 0 14px rgba(255,136,0,0.35);font-size:0.7rem;letter-spacing:0.8px; }
.sev-MEDIUM { color:#000;font-weight:800;background:linear-gradient(135deg,#ffcc00,#cca300);padding:3px 12px;border-radius:10px;box-shadow:0 0 14px rgba(255,204,0,0.35);font-size:0.7rem;letter-spacing:0.8px; }
.sev-LOW { color:#fff;font-weight:800;background:linear-gradient(135deg,#00cc66,#00994d);padding:3px 12px;border-radius:10px;box-shadow:0 0 14px rgba(0,204,102,0.35);font-size:0.7rem;letter-spacing:0.8px; }

/* ─── GLASS TABS ─── */
div[data-testid="stTabs"] {
  background:rgba(255,255,255,0.018);
  backdrop-filter:blur(12px);
  -webkit-backdrop-filter:blur(12px);
  border:1px solid rgba(255,255,255,0.05);
  border-radius:16px;
  padding:4px 8px !important;
  margin-bottom:20px;
}
div[data-testid="stTabs"] button {
  color:var(--text-color) !important;opacity:0.4;
  border-bottom:2px solid transparent !important;
  font-size:.78rem !important;font-weight:700 !important;
  letter-spacing:1px !important;
  padding-bottom:14px !important;margin-right:22px !important;
  transition:all 0.3s ease !important;
}
div[data-testid="stTabs"] button:hover { opacity:0.75 !important; }
div[data-testid="stTabs"] button[aria-selected="true"] {
  opacity:1 !important;
  border-bottom:2px solid #e20082 !important;
  text-shadow:0 0 20px rgba(226,0,130,0.6) !important;
}

h1,h2,h3 { color:var(--text-color);font-weight:700; }
h1 { letter-spacing:4px !important;text-shadow:0 0 40px rgba(226,0,130,0.07); }
h2 { letter-spacing:1px !important;opacity:0.9; }
h3 { opacity:0.85; }

/* ─── DATAFRAMES ─── */
.stDataFrame {
  border:1px solid rgba(255,255,255,0.05) !important;
  border-radius:20px !important;
  background:rgba(255,255,255,0.025) !important;
  backdrop-filter:blur(10px) !important;
  overflow:hidden !important;
}

/* ─── CYBER BUTTONS ─── */
.stButton > button {
  background:rgba(255,255,255,0.025) !important;
  backdrop-filter:blur(8px) !important;
  -webkit-backdrop-filter:blur(8px) !important;
  border:1px solid rgba(226,0,130,0.25) !important;
  color:rgba(255,255,255,0.75) !important;
  border-radius:14px !important;
  font-weight:700 !important;font-size:0.78rem !important;
  letter-spacing:0.8px !important;padding:10px 22px !important;
  transition:all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
}
.stButton > button:hover {
  background:rgba(226,0,130,0.1) !important;
  border-color:rgba(226,0,130,0.55) !important;
  color:#fff !important;
  box-shadow:0 0 24px rgba(226,0,130,0.18),inset 0 0 20px rgba(226,0,130,0.04) !important;
  transform:translateY(-2px) !important;
}
.stButton > button:active {
  transform:translateY(0) !important;
  box-shadow:0 0 10px rgba(226,0,130,0.4) !important;
}

/* ─── TEXT INPUTS ─── */
.stTextInput > div > div > input {
  border-radius:14px !important;
  border:1px solid rgba(255,255,255,0.07) !important;
  background:rgba(0,0,0,0.45) !important;
  backdrop-filter:blur(8px) !important;
  -webkit-backdrop-filter:blur(8px) !important;
  transition:border-color 0.3s ease,box-shadow 0.3s ease !important;
  font-family:'JetBrains Mono',monospace !important;
  font-size:0.82rem !important;
}
.stTextInput > div > div > input:focus {
  border-color:rgba(226,0,130,0.4) !important;
  box-shadow:0 0 20px rgba(226,0,130,0.1) !important;
}

/* ─── SELECTS ─── */
.stSelectbox > div > div, .stMultiSelect > div > div {
  border-radius:14px !important;border-color:rgba(255,255,255,0.07) !important;
  background:rgba(255,255,255,0.025) !important;
}

/* ─── PLOTLY ─── */
.stPlotlyChart {
  border-radius:22px !important;overflow:hidden !important;
  border:1px solid rgba(255,255,255,0.05) !important;
  box-shadow:0 4px 20px rgba(0,0,0,0.3) !important;
}

/* ─── EXPANDERS ─── */
.streamlit-expanderHeader {
  background: rgba(255,255,255,0.025) !important;
  border-radius: 12px !important;
}

/* ─── HR DIVIDERS ─── */
hr {
  border-color: rgba(255,255,255,0.04) !important;
  margin: 20px 0 !important;
}

/* ─── RESPONSIVE: prevent overflow on smaller windows ─── */
html, body { overflow-x: hidden !important; }
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {
  max-width: calc(100vw - 260px) !important;
  overflow-x: hidden !important;
}
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"] {
  display: none !important;
}

</style>""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
SEV_COLOR = {"CRITICAL":"#ff0055","HIGH":"#ff8800","MEDIUM":"#ffcc00","LOW":"#00cc66"}
ATK_COLOR = {"DoS":"#ff0055","PortScan":"#ff8800","BruteForce":"#9933ff",
             "Probe":"#ffcc00","Normal":"#00cc66","Unknown":"#8b949e"}

def _theme_adaptive(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E0E0E0",
        margin=dict(l=10,r=10,t=30,b=10)
    )
    return fig

def metric_card(label, value, col, highlight=False):
    style = 'border-left: 3px solid #ff3366;' if highlight else ''
    col.markdown(f'<div class="metric-card" style="{style}"><div class="metric-val">{value}</div>'
                 f'<div class="metric-lbl">{label}</div></div>', unsafe_allow_html=True)

@st.cache_data(ttl=1) # 1 second cache for ultra-fast updates
def load_df(n=2000):
    if not LOG_FILE.exists(): return pd.DataFrame()
    rows=[]
    with open(LOG_FILE) as f:
        lines=f.readlines()[-n:]
    for l in lines:
        l=l.strip()
        if l:
            try: rows.append(json.loads(l))
            except: pass
    if not rows: return pd.DataFrame()
    df=pd.DataFrame(rows)
    if "timestamp" in df.columns:
        # Support both ISO and custom local time formats
        df["timestamp"]=pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["reconstruction_error","risk_score","confidence"]:
        if c in df.columns:
            df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
    # Sort to show newest first
    if not df.empty and "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False)
    return df

@st.cache_data(max_entries=1000)
def fetch_geoip(ip):
    if not ip or ip == "0.0.0.0":
        return None, None, "Unknown", "Unknown"
        
    import ipaddress
    import hashlib
    
    try:
        is_priv = ipaddress.ip_address(ip).is_private
    except:
        is_priv = False
        
    if is_priv:
        # For demo purposes, we map private IPs (like 172.17.x.x) to deterministic global cities
        # so the Threat Map looks realistically populated instead of empty or clustered at (0,0).
        locations = [
            (40.7128, -74.0060, "New York", "USA"),
            (51.5074, -0.1278, "London", "UK"),
            (35.6762, 139.6503, "Tokyo", "Japan"),
            (-33.8688, 151.2093, "Sydney", "Australia"),
            (37.7749, -122.4194, "San Francisco", "USA"),
            (52.5200, 13.4050, "Berlin", "Germany"),
            (1.3521, 103.8198, "Singapore", "Singapore"),
            (-23.5505, -46.6333, "Sao Paulo", "Brazil"),
            (55.7558, 37.6173, "Moscow", "Russia"),
            (28.6139, 77.2090, "New Delhi", "India"),
            (48.8566, 2.3522, "Paris", "France"),
            (31.2304, 121.4737, "Shanghai", "China")
        ]
        # Hash the IP string to always get the same city for the same IP
        idx = int(hashlib.md5(ip.encode()).hexdigest(), 16) % len(locations)
        return locations[idx]

    # Otherwise, query the specific external IP
    import requests
    try:
        r = requests.get(f"http://ip-api.com/json/{ip}", timeout=2).json()
        if r.get("status") == "success":
            return r.get("lat"), r.get("lon"), r.get("city"), r.get("country")
    except:
        pass
    return None, None, "Unknown", "Unknown"

def get_sys_metrics():
    try:
        import psutil
        # Calling with interval=None twice with a tiny sleep forces a real delta calculation
        # This is the "Gold Standard" for real-time CPU tracking on macOS
        psutil.cpu_percent(interval=None) 
        time.sleep(0.05)
        cpu = psutil.cpu_percent(interval=None)
        
        vmem = psutil.virtual_memory()
        return cpu, vmem.percent
    except: 
        return 0.0, 0.0

# ── Auto-Start Real-Time Engine ────────────────────────────────────────────────
@st.cache_resource(ttl=3600) # Force refresh every hour
def start_live_engine(engine_type):
    """
    Initialize and start the background detection thread.
    Uses st.cache_resource to ensure only one instance runs.
    """
    # Force a unique ID for the engine to break cache
    engine_id = f"{engine_type}_v2_0" 
    print(f"\n[DASHBOARD] Booting AI Engine: {engine_id}")
    # Clear logs exactly ONCE per server lifecycle for a clean live demo
    if LOG_FILE.exists():
        try: LOG_FILE.unlink()
        except Exception: pass
        
    try:
        if engine_type == "SCAPY":
            from realtime_detector import RealtimeDetector
            engine = RealtimeDetector()
            t = threading.Thread(target=engine.start, daemon=True)
            t.start()
            return t
        else:
            from tshark_detector import TSharkRealtimeDetector
            engine = TSharkRealtimeDetector()
            t = threading.Thread(target=engine.start, daemon=True)
            t.start()
            return t
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to start {engine_type} engine: {e}")
        return None

# Determine engine type based on phase
engine_type = "SCAPY" if PROJECT_PHASE <= 2 else "TSHARK"
st.session_state.current_engine_type = engine_type

# Start the engine
start_live_engine(st.session_state.current_engine_type)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Logo Banner ──
    st.markdown("""
    <div class="sb-logo">
      <span class="sb-logo-title">THREAT<br>INTELLIGENCE</span>
    </div>
    """, unsafe_allow_html=True)

    # ── System Status Card ──
    cpu, ram = get_sys_metrics()
    cpu_dot  = "dot-red" if cpu > 80 else ("dot-yellow" if cpu > 50 else "dot-green")
    ram_dot  = "dot-red" if ram > 85 else ("dot-yellow" if ram > 65 else "dot-green")
    df_rows  = 0
    if LOG_FILE.exists():
        try: df_rows = sum(1 for _ in open(LOG_FILE))
        except: pass
    
    engine_type = st.session_state.get('current_engine_type', 'UNKNOWN')
    
    st.markdown(f"""
    <div class="sb-card">
      <div class="sb-card-title">SYSTEM STATUS</div>
      <div class="sb-stat">
        <span class="sb-stat-label">ENGINE TYPE</span>
        <span><span class="dot-green"></span>&nbsp;<span style="color:#00ff88;font-size:.72rem;font-weight:bold;">{engine_type}</span></span>
      </div>
      <div class="sb-stat">
        <span class="sb-stat-label">CPU LOAD</span>
        <span><span class="{cpu_dot}"></span>&nbsp;<span class="sb-stat-val">{cpu:.1f}%</span></span>
      </div>
      <div class="sb-stat">
        <span class="sb-stat-label">RAM USAGE</span>
        <span><span class="{ram_dot}"></span>&nbsp;<span class="sb-stat-val">{round((psutil.virtual_memory().total - psutil.virtual_memory().available)/(1024**3), 1)} GB</span></span>
      </div>
      <div class="sb-stat">
        <span class="sb-stat-label">LOG ENTRIES</span>
        <span class="sb-stat-val">{df_rows:,}</span>
      </div>
      <div class="sb-stat">
        <span class="sb-stat-label">NEO4J</span>
        <span><span class="dot-yellow"></span>&nbsp;<span style="color:#ffcc00;font-size:.72rem">FALLBACK</span></span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,rgba(226,0,130,0.2),transparent);margin:8px 10px'></div>", unsafe_allow_html=True)

    # ── CPU / RAM Progress ──
    cpu_cls = "safe" if cpu < 50 else ("warn" if cpu < 80 else "")
    ram_cls = "safe" if ram < 65 else ("warn" if ram < 85 else "")
    st.markdown(f"""
    <div class="sb-card">
      <div class="sb-card-title">RESOURCE MONITOR</div>
      <div class="res-bar-wrap">
        <div class="res-bar-label"><span>CPU</span><span>{cpu:.1f}%</span></div>
        <div class="res-bar-track"><div class="res-bar-fill {cpu_cls}" style="width:{min(cpu,100):.1f}%"></div></div>
      </div>
      <div class="res-bar-wrap">
        <div class="res-bar-label"><span>RAM</span><span>{ram:.1f}%</span></div>
        <div class="res-bar-track"><div class="res-bar-fill {ram_cls}" style="width:{min(ram,100):.1f}%"></div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:2px;background:linear-gradient(90deg,transparent,#0f3060,transparent);margin:6px 10px'></div>", unsafe_allow_html=True)

    # ── CONTROLS ──
    st.markdown("""
    <div class="sb-card">
      <div class="sb-card-title">CONTROLS</div>
    </div>
    """, unsafe_allow_html=True)
    demo_on  = st.toggle("DEMO MODE", value=False)
    refresh  = st.slider("Refresh (s)", 1, 30, 2) # Now allows 1 second refresh!
    max_rows = st.slider("Max rows", 100, 5000, 1000)

    # ── ADMIN HELP ──
    with st.expander("ADMIN COMMAND HELP"):
        if sys.platform == "win32":
            st.code("python -m streamlit run dashboard.py", language="bash")
            st.caption("Note: Run PowerShell as Administrator")
        else:
            st.code("sudo ./venv/bin/python -m streamlit run dashboard.py", language="bash")
            st.caption("Note: Root privileges required for Scapy")

    st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,rgba(226,0,130,0.2),transparent);margin:8px 10px'></div>", unsafe_allow_html=True)

    # ── Actions ──
    st.markdown("""
    <div class="sb-card">
      <div class="sb-card-title">ACTIONS</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("CLEAR LOGS"):
        if LOG_FILE.exists():
            try: LOG_FILE.unlink()
            except Exception: pass
        st.rerun()
    if st.button("SEED DEMO DATA"):
        try:
            from demo_mode import seed_demo_log
            n = seed_demo_log(300, clear_existing=True)
            st.success(f"Seeded {n} records")
            load_df.clear()
        except Exception as e: st.error(str(e))
    if st.button("REFRESH NOW"): load_df.clear(); st.rerun()

    # ── AI CO-PILOT (Charlotte-Style) ──
    st.markdown("""
    <div class="sb-card" style="margin-top: 15px;">
      <div class="sb-card-title">AI CO-PILOT</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "I'm your AI Analyst. How can I help?"}
        ]
        
    # Container for chat messages to keep sidebar tidy
    chat_container = st.container(height=300, border=False)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask about IPs, threats..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    time.sleep(1) # Fake processing delay
                    
                    if "192.168.1.5" in prompt or "critical" in prompt.lower():
                        response = "I see multiple Critical anomalies. `192.168.1.5` is executing a high-volume DoS. Recommend immediate isolation."
                    elif "mitigate" in prompt.lower() or "action" in prompt.lower():
                        response = "Actions: 1. Apply rate-limiting. 2. Blacklist offending IPs. 3. Verify CPU spikes."
                    else:
                        response = f"I've analyzed the recent telemetry. The system is actively isolating threats."
                    
                    st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # ── Footer ──
    st.markdown("""<br>""", unsafe_allow_html=True)

# ── Start demo simulator ───────────────────────────────────────────────────────
if demo_on:
    if "demo_sim" not in st.session_state:
        try:
            from demo_mode import DemoAttackSimulator
            sim=DemoAttackSimulator(rate=1.5)
            sim.start()
            st.session_state["demo_sim"]=sim
        except: pass
else:
    if "demo_sim" in st.session_state:
        try: st.session_state["demo_sim"].stop()
        except: pass
        del st.session_state["demo_sim"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 32px 0 24px; position:relative;">
  <h1 style="font-size:2.6rem; font-weight:900; letter-spacing:5px; margin:0; background:linear-gradient(135deg,#ffffff 40%,#e20082 70%,#ff8800 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; background-size:200% auto; text-shadow:none;">INTELLIGENT THREAT DETECTOR</h1>
</div>
""", unsafe_allow_html=True)

df=load_df(max_rows)

# ── Tabs ──────────────────────────────────────────────────────────────────────
# ── Tabs Configuration (Phased Presentation) ──────────────────────────────────
tab_map = {
    1: ["LIVE MONITOR", "SYSTEM METRICS"],
    2: ["LIVE MONITOR", "ATTACK ANALYTICS", "SYSTEM METRICS"],
    3: ["LIVE MONITOR", "ATTACK ANALYTICS", "SYSTEM METRICS", "PACKET FORENSICS"],
    4: ["LIVE MONITOR", "ATTACK ANALYTICS", "GEO-IP MAP", "THREAT GRAPH", "SYSTEM METRICS", "PACKET FORENSICS"]
}
current_tabs = tab_map.get(PROJECT_PHASE, tab_map[4])
tabs = st.tabs(current_tabs)

# Assign tabs to variables based on their names
tab_dict = dict(zip(current_tabs, tabs))
t1 = tab_dict.get("LIVE MONITOR")
t2 = tab_dict.get("ATTACK ANALYTICS")
t3 = tab_dict.get("THREAT GRAPH")
t4 = tab_dict.get("SYSTEM METRICS")
t5 = tab_dict.get("AI CO-PILOT")
t6 = tab_dict.get("PACKET FORENSICS")
t7 = tab_dict.get("GEO-IP MAP")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    if df.empty:
        st.info("No data. Use sidebar to seed demo data or start realtime_detector.py")
    else:
        # Quick Filter Buttons
        qf1, qf2, qf3, qf4 = st.columns(4)
        
        if qf1.button("Critical DoS", use_container_width=True):
            st.session_state.siem_query_input = "index=main | search severity=CRITICAL attack_type=DoS"
            st.rerun()
            
        if qf2.button("Port Scans", use_container_width=True):
            st.session_state.siem_query_input = "attack_type=PortScan"
            st.rerun()
            
        if qf3.button("High Risk TCP", use_container_width=True):
            st.session_state.siem_query_input = "severity=HIGH protocol=TCP"
            st.rerun()
            
        if qf4.button("Clear Search", use_container_width=True):
            st.session_state.siem_query_input = ""
            st.rerun()
            
        st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

        # --- SIEM SEARCH HEADER ---
        st.markdown("""
        <div class="section-hdr">
          <div class="section-hdr-line"></div>
          <div class="section-hdr-label">SIEM QUERY ENGINE</div>
          <div class="section-hdr-line right"></div>
        </div>
        """, unsafe_allow_html=True)
        
        sc1, sc2 = st.columns([8, 1.5])
        with sc1:
            siem_query = st.text_input("SIEM Search", key="siem_query_input", label_visibility="collapsed", placeholder='index=main sourcetype=ids | search severity=CRITICAL attack_type=DoS', help="Filter live data dynamically. Use key=value syntax.")
        with sc2:
            st.button("Search", use_container_width=True)
            
        st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)
        
        if siem_query:
            try:
                tokens = siem_query.split()
                for q in tokens:
                    if "=" in q:
                        k, v = q.split("=", 1)
                        if k in df.columns:
                            # Exact match or contains
                            if df[k].astype(str).str.fullmatch(v, case=False).any():
                                df = df[df[k].astype(str).str.fullmatch(v, case=False, na=False)]
                            else:
                                df = df[df[k].astype(str).str.contains(v, case=False, na=False)]
            except Exception as e:
                st.error(f"Query parsing error: {e}")
        # --------------------------------
        c1,c2,c3,c4,c5=st.columns(5)
        total=len(df)
        crit=len(df[df.get("severity","")=="CRITICAL"]) if "severity" in df else 0
        hi  =len(df[df.get("severity","")=="HIGH"])    if "severity" in df else 0
        recent=len(df[df["timestamp"]>datetime.utcnow()-timedelta(minutes=5)]) if "timestamp" in df else 0
        avg_e=df["reconstruction_error"].mean() if "reconstruction_error" in df else 0
        metric_card("Total Anomalies",f"{total:,}",c1)
        metric_card("Critical",crit,c2)
        metric_card("High",hi,c3)
        metric_card("Last 5 min",recent,c4)
        metric_card("Avg Score",f"{avg_e:,.2f}",c5)
        st.markdown("---")

        # Live timeline — always show attacks even if buried by Normal entries
        if "timestamp" in df.columns and "reconstruction_error" in df.columns:
            # Always include ALL critical/high rows + sample of normal
            if "severity" in df.columns:
                atk_df    = df[df["severity"].isin(["CRITICAL","HIGH","MEDIUM"])]
                normal_df = df[df["severity"]=="LOW"].sort_values("timestamp").tail(200)
                plot_df   = pd.concat([normal_df, atk_df]).sort_values("timestamp")
            else:
                plot_df = df.sort_values("timestamp").tail(300)
            fig=px.scatter(plot_df,x="timestamp",y="reconstruction_error",
                color="severity" if "severity" in df.columns else None,
                color_discrete_map=SEV_COLOR,
                hover_data=["src_ip","dst_ip","attack_type"] if "src_ip" in df.columns else None,
                title="Anomaly Score Timeline")
            st.plotly_chart(_theme_adaptive(fig),use_container_width=True)

        # Live feed
        st.subheader("Live Anomaly Feed")
        cols=["timestamp","src_ip","dst_ip","attack_type","severity","reconstruction_error","hint"]
        if PROJECT_PHASE < 4 and "hint" in cols:
            cols.remove("hint")
        show_cols=[c for c in cols if c in df.columns]
        # Show attacks first, then most-recent Normal — never bury attacks under Normal
        if "severity" in df.columns:
            atk_rows    = df[df["severity"].isin(["CRITICAL","HIGH","MEDIUM"])].sort_values("timestamp",ascending=False).head(10)
            normal_rows = df[df["severity"]=="LOW"].sort_values("timestamp",ascending=False).head(10)
            recent_df   = pd.concat([atk_rows, normal_rows])[show_cols]
        else:
            recent_df = df.sort_values("timestamp",ascending=False).head(20)[show_cols]

        # --- SPLUNK RAW EVENT LOG VIEW ---
        log_html = "<div style='background:rgba(255,255,255,0.03); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); border: 1px solid rgba(255,255,255,0.08); border-radius:24px; padding:20px; font-family:\"JetBrains Mono\", monospace; max-height: 600px; overflow-y: auto; box-shadow: 0 4px 30px rgba(0,0,0,0.3);'>"
        for _, row in recent_df.iterrows():
            ts = row.get("timestamp", "Unknown time")
            sev = row.get("severity", "LOW")
            sev_color = SEV_COLOR.get(sev, "#00cc66")
            
            # Create key=value raw log string
            kv_pairs = []
            for k, v in row.items():
                if k not in ["timestamp", "severity"]:
                    kv_pairs.append(f'<span style="color:#00e6e6;">{k}</span>=<span style="color:#e0e0e0;">"{v}"</span>')
            log_str = " ".join(kv_pairs)
            
            log_html += f"<div style='border-bottom: 1px solid rgba(255,255,255,0.05); padding: 8px 4px; display:flex; gap:16px; align-items:flex-start;'><div style='min-width: 180px; color:{sev_color}; font-size:0.8rem; font-weight:600;'>▶ {ts}</div><div style='font-size:0.8rem; word-break: break-all; line-height: 1.4;'><span style='background:{sev_color}; color:#000; font-weight:bold; padding:2px 6px; border-radius:3px; font-size:0.7rem; margin-right:8px;'>{sev}</span> {log_str}</div></div>"
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)

        # Attack counters
        if "attack_type" in df.columns:
            st.subheader("Attack Counters")
            cts=df["attack_type"].value_counts()
            if len(cts) > 0:
                cols2=st.columns(len(cts))
                for i,(k,v) in enumerate(cts.items()):
                    metric_card(k,v,cols2[i])
            else:
                st.info("No attacks recorded yet.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ATTACK ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
if t2:
    with t2:
        if df.empty:
            st.info("No data yet.")
        else:
            # --- EXPLAINABLE AI SECTION ---
            st.markdown("<h3 style='font-size:1.1rem; margin-top:10px; margin-bottom:15px;'>EXPLAINABLE AI INSIGHTS</h3>", unsafe_allow_html=True)
            if "hint" in df.columns:
                xai_df = df.dropna(subset=["hint"])
                xai_df = xai_df[xai_df["hint"] != ""]
                if not xai_df.empty:
                    # Prioritize attacks over Normal traffic, and deduplicate so we don't see the same attack 3 times
                    if "severity" in xai_df.columns:
                        xai_atk = xai_df[xai_df["severity"].isin(["CRITICAL","HIGH","MEDIUM"])].sort_values("timestamp", ascending=False).drop_duplicates(subset=["attack_type", "src_ip"])
                        xai_norm = xai_df[xai_df["severity"] == "LOW"].sort_values("timestamp", ascending=False).drop_duplicates(subset=["hint"])
                        latest_xai = pd.concat([xai_atk, xai_norm]).head(3)
                    else:
                        latest_xai = xai_df.sort_values("timestamp", ascending=False).drop_duplicates(subset=["hint"]).head(3)
                        
                    for _, row in latest_xai.iterrows():
                        color = SEV_COLOR.get(row.get('severity','LOW'), '#00cc66')
                        score = row.get('risk_score', row.get('reconstruction_error', 0))
                        hints = str(row['hint']).split(';')
                        hint_html = "".join([f"<li style='margin-bottom:4px;'>{h.strip()}</li>" for h in hints if h.strip()])
                        
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.03); backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); border: 1px solid rgba(255,255,255,0.08); border-left: 4px solid {color}; padding: 18px 22px; margin-bottom: 14px; border-radius: 20px; box-shadow: 0 4px 24px rgba(0,0,0,0.3);">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                                <span style="font-weight:600; color:var(--text-title); font-size:1.0rem;">{row.get('attack_type', 'Anomaly')} Detected</span>
                                <span style="font-family:'JetBrains Mono', monospace; color:{color}; font-size:0.85rem; background:var(--bg-main); padding:4px 10px; border-radius:4px;">IP: {row.get('src_ip','Unknown')} | Risk: {score:.2f}</span>
                            </div>
                            <div style="color:var(--text-muted); font-size:0.75rem; font-weight:600; letter-spacing:0.5px; margin-bottom:8px; text-transform:uppercase;">Reasoning Engine Output</div>
                            <ul style="color:var(--text-main); font-size:0.85rem; margin-top:0; margin-bottom:0; padding-left:20px;">
                                {hint_html}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No explainability hints available yet.")
            else:
                st.info("XAI Engine is initialising...")
            
            st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 25px 0;'>", unsafe_allow_html=True)
            # --- END EXPLAINABLE AI ---

            r1c1,r1c2=st.columns(2)
            # Severity donut
            if "severity" in df.columns:
                sev_c=df["severity"].value_counts().reset_index()
                sev_c.columns=["severity","count"]
                fig=px.pie(sev_c,names="severity",values="count",hole=0.55,
                    color="severity",color_discrete_map=SEV_COLOR,title="Severity Distribution")
                r1c1.plotly_chart(_theme_adaptive(fig),use_container_width=True)

            # Attack type bar
            if "attack_type" in df.columns:
                atk_c=df["attack_type"].value_counts().reset_index()
                atk_c.columns=["attack_type","count"]
                fig=px.bar(atk_c,x="count",y="attack_type",orientation="h",
                    color="attack_type",color_discrete_map=ATK_COLOR,title="Attack Types")
                r1c2.plotly_chart(_theme_adaptive(fig),use_container_width=True)

            r2c1,r2c2=st.columns(2)
            # Top source IPs
            if "src_ip" in df.columns:
                top_src=df["src_ip"].value_counts().head(10).reset_index()
                top_src.columns=["ip","count"]
                fig=px.bar(top_src,x="count",y="ip",orientation="h",
                    color="count",color_continuous_scale="Reds",title="Top 10 Attacker IPs")
                r2c1.plotly_chart(_theme_adaptive(fig),use_container_width=True)

            # Protocol pie
            if "protocol" in df.columns:
                proto_c=df["protocol"].value_counts().reset_index()
                proto_c.columns=["protocol","count"]
                fig=px.pie(proto_c,names="protocol",values="count",title="Protocol Distribution",
                    color_discrete_sequence=["#00ff88","#4488ff","#ff8800","#ff4499"])
                r2c2.plotly_chart(_theme_adaptive(fig),use_container_width=True)

            # Anomaly score histogram
            if "reconstruction_error" in df.columns:
                fig=px.histogram(df,x="reconstruction_error",nbins=50,
                    color_discrete_sequence=["#00ff88"],title="Anomaly Score Distribution")
                st.plotly_chart(_theme_adaptive(fig),use_container_width=True)

            # Heatmap (hour x attack type)
            if "timestamp" in df.columns and "attack_type" in df.columns:
                st.subheader("Attack Heatmap (Hour × Type)")
                df2=df.copy()
                df2["hour"]=df2["timestamp"].dt.hour
                heat=df2.groupby(["hour","attack_type"]).size().unstack(fill_value=0)
                fig=px.imshow(heat,color_continuous_scale="Reds",
                    labels=dict(x="Attack Type",y="Hour",color="Count"),title="")
                st.plotly_chart(_theme_adaptive(fig),use_container_width=True)

            st.markdown("---")
            # Model metrics
            st.subheader("Model Performance Metrics")
            mc1,mc2=st.columns(2)
            rf_mets_path=BASE_DIR/"logs"/"rf_evaluation.json"
            lstm_mets_path=BASE_DIR/"logs"/"training_metrics.json"
            if rf_mets_path.exists():
                rf_m=json.loads(rf_mets_path.read_text())
                mc1.markdown("**Random Forest**")
                for k in ["accuracy","precision","recall","f1_score","roc_auc"]:
                    if k in rf_m: mc1.metric(k.replace("_"," ").title(), f"{rf_m[k]:.4f}")
            else:
                mc1.info("Run train_rf.py to see RF metrics")

            if lstm_mets_path.exists():
                lstm_m=json.loads(lstm_mets_path.read_text())
                mc2.markdown("**LSTM Autoencoder**")
                mc2.metric("Best Val Loss",f"{lstm_m.get('best_val_loss',0):.6f}")
                mc2.metric("Threshold",f"{lstm_m.get('threshold',0):.6f}")
                # Loss curve
                if "train_loss" in lstm_m and "val_loss" in lstm_m:
                    epochs=list(range(1,len(lstm_m["train_loss"])+1))
                    fig6=go.Figure()
                    fig6.add_trace(go.Scatter(x=epochs,y=lstm_m["train_loss"],
                        name="Train Loss",line=dict(color="#00ff88")))
                    fig6.add_trace(go.Scatter(x=epochs,y=lstm_m["val_loss"],
                        name="Val Loss",line=dict(color="#4488ff",dash="dash")))
                    fig6.update_layout(title="LSTM Training Curves",
                        xaxis_title="Epoch",yaxis_title="Loss")
                    mc2.plotly_chart(_theme_adaptive(fig6),use_container_width=True)
            else:
                mc2.info("Run train_lstm.py to see LSTM metrics")

            # Attack summary table
            st.subheader("Attack Summary")
            if "attack_type" in df.columns:
                summary=df.groupby("attack_type").agg(
                    count=("attack_type","count"),
                    avg_score=("reconstruction_error","mean"),
                    max_score=("reconstruction_error","max"),
                ).round(4).reset_index()
                st.dataframe(summary,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — THREAT GRAPH
# ══════════════════════════════════════════════════════════════════════════════
if t3:
    with t3:
        if df.empty:
            st.info("No data yet.")
        else:
            st.subheader("THREAT INTELLIGENCE GRAPH")
            
            if "src_ip" in df.columns and "dst_ip" in df.columns:
                edges = df.groupby(["src_ip", "dst_ip"]).size().reset_index(name="weight")
                G = nx.from_pandas_edgelist(edges, 'src_ip', 'dst_ip', ['weight'], create_using=nx.DiGraph())
                # Increase k to spread nodes out further and reduce overlap
                pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
                
                thin_edge_x, thin_edge_y = [], []
                thick_edge_x, thick_edge_y = [], []
                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    w = edge[2].get('weight', 1)
                    if w > 5:
                        thick_edge_x.extend([x0, x1, None])
                        thick_edge_y.extend([y0, y1, None])
                    else:
                        thin_edge_x.extend([x0, x1, None])
                        thin_edge_y.extend([y0, y1, None])
                    
                edge_trace_thin = go.Scatter(
                    x=thin_edge_x, y=thin_edge_y,
                    line=dict(width=1, color='rgba(255, 255, 255, 0.1)'),
                    hoverinfo='none',
                    mode='lines')
                    
                edge_trace_thick = go.Scatter(
                    x=thick_edge_x, y=thick_edge_y,
                    line=dict(width=3, color='rgba(255, 0, 85, 0.7)'),
                    hoverinfo='none',
                    mode='lines')
                    
                node_x = []
                node_y = []
                node_text = []
                node_colors = []
                node_sizes = []
                
                src_ips = df['src_ip'].unique() if 'src_ip' in df.columns else []
                dst_ips = df['dst_ip'].unique() if 'dst_ip' in df.columns else []
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(str(node))
                    
                    if node in src_ips:
                        sevs = df[df['src_ip'] == node]['severity'].unique() if 'severity' in df.columns else []
                        if 'CRITICAL' in sevs:
                            node_colors.append('#ff0055')
                            node_sizes.append(24)
                        elif 'HIGH' in sevs:
                            node_colors.append('#ff8800')
                            node_sizes.append(20)
                        elif 'MEDIUM' in sevs:
                            node_colors.append('#ffcc00')
                            node_sizes.append(16)
                        else:
                            node_colors.append('#00cc66')
                            node_sizes.append(14)
                    else:
                        # Target Nodes
                        node_colors.append('#00e6e6')
                        node_sizes.append(30)
                    
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers', # Remove static text to prevent overlapping
                    hoverinfo='text',
                    text=node_text,
                    marker=dict(
                        showscale=False,
                        color=node_colors,
                        size=node_sizes,
                        line_width=2,
                        line_color='#15171e'))
                        
                fig = go.Figure(data=[edge_trace_thin, edge_trace_thick, node_trace],
                             layout=go.Layout(
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=20),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                )
                st.plotly_chart(_theme_adaptive(fig), use_container_width=True)
            else:
                st.warning("Graph unavailable: 'src_ip' or 'dst_ip' not found in data.")

            # Cluster table
            if "src_ip" in df.columns and "dst_ip" in df.columns:
                st.subheader("Attack Clusters (Multiple Attackers → Same Target)")
                cluster=df.groupby("dst_ip")["src_ip"].nunique().reset_index()
                cluster.columns=["target_ip","unique_attackers"]
                cluster=cluster[cluster["unique_attackers"]>=2].sort_values("unique_attackers",ascending=False)
                if not cluster.empty:
                    st.dataframe(cluster,use_container_width=True)
                else:
                    st.info("No clusters detected yet.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SYSTEM METRICS
# ══════════════════════════════════════════════════════════════════════════════
if t4:
    with t4:
        st.subheader("CROSS-LAYER SYSTEM METRICS")
        try:
            import psutil
            # cpu variable is already acquired from the sidebar, doing it again without interval yields 0.0!
            ram_obj=psutil.virtual_memory()
            # Non-blocking per-core metrics
            cores=psutil.cpu_percent(interval=None, percpu=True)
            
            # Calculate actual GB for "11 GB to 3 GB" style monitoring
            used_gb = (ram_obj.total - ram_obj.available) / (1024**3)
            total_gb = ram_obj.total / (1024**3)

            m1,m2,m3,m4=st.columns(4)
            metric_card("CPU Usage",f"{cpu:.1f}%",m1, highlight=True)
            metric_card("RAM Used",f"{used_gb:.1f} GB",m2)
            metric_card("RAM Available",f"{ram_obj.available/(1024**3):.1f} GB",m3)
            metric_card("CPU Cores",len(cores),m4)

            # CPU gauge
            fig=go.Figure(go.Indicator(mode="gauge+number",value=cpu,
                title={"text":"CPU %","font":{"color":"#00ff88"}},
                gauge={"axis":{"range":[0,100]},"bar":{"color":"#00ff88"},
                       "steps":[{"range":[0,50],"color":"#0d1120"},
                                 {"range":[50,80],"color":"#1a2a00"},
                                 {"range":[80,100],"color":"#3a0000"}],
                       "threshold":{"line":{"color":"red","width":4},"value":85}}))
            fig.update_layout(height=250)
            
            fig2=go.Figure(go.Indicator(mode="gauge+number",value=ram_obj.percent,
                title={"text":"RAM %","font":{"color":"#4488ff"}},
                gauge={"axis":{"range":[0,100]},"bar":{"color":"#4488ff"},
                       "steps":[{"range":[0,60],"color":"#0d1120"},
                                 {"range":[60,85],"color":"#001a3a"},
                                 {"range":[85,100],"color":"#3a0000"}]}))
            fig2.update_layout(height=250)

            gc1,gc2=st.columns(2)
            gc1.plotly_chart(_theme_adaptive(fig),use_container_width=True)
            gc2.plotly_chart(_theme_adaptive(fig2),use_container_width=True)

            # Per-core bars
            st.subheader("Per-Core CPU Usage")
            core_df=pd.DataFrame({"core":[f"Core {i}" for i in range(len(cores))],"usage":cores})
            fig3=px.bar(core_df,x="core",y="usage",color="usage",
                color_continuous_scale=["#00ff88","#ff8800","#ff0033"],
                title="Per-Core Utilization")
            st.plotly_chart(_theme_adaptive(fig3),use_container_width=True)

            # Top processes
            st.subheader("Top Processes by CPU")
            procs=[]
            for p in psutil.process_iter(["pid","name","cpu_percent","memory_info"]):
                try:
                    cpu_p = p.info.get("cpu_percent")
                    cpu_p = float(cpu_p) if cpu_p is not None else 0.0
                    mem_rss = p.info["memory_info"].rss if p.info.get("memory_info") else 0
                    procs.append({
                        "PID": p.info["pid"],
                        "Name": p.info["name"],
                        "CPU%": cpu_p,
                        "RAM MB": round(mem_rss / 1024**2, 1)
                    })
                except: pass
            procs=sorted(procs,key=lambda x:x.get("CPU%", 0.0) or 0.0,reverse=True)[:10]
            st.dataframe(pd.DataFrame(procs),use_container_width=True)

            # IRQ simulation (if Linux /proc/interrupts exists)
            irq_path=Path("/proc/interrupts")
            if irq_path.exists():
                st.subheader("Network IRQ Activity")
                try:
                    irq_lines=irq_path.read_text().split("\n")[1:20]
                    irq_data=[]
                    for l in irq_lines:
                        parts=l.split()
                        if len(parts)>2:
                            try: irq_data.append({"IRQ":parts[0].rstrip(":"),"Count":sum(int(x) for x in parts[1:] if x.isdigit())})
                            except: pass
                    if irq_data:
                        irq_df=pd.DataFrame(irq_data).sort_values("Count",ascending=False).head(10)
                        fig4=px.bar(irq_df,x="IRQ",y="Count",color="Count",
                            color_continuous_scale="Greens",title="Top IRQ Counts")
                        st.plotly_chart(_theme_adaptive(fig4),use_container_width=True)
                except: st.info("IRQ data unavailable")
            else:
                # macOS/Windows fallback: LIVE Network I/O instead of hardcoded demo data
                st.subheader("Network I/O Activity (Live)")
                net_io = psutil.net_io_counters(pernic=True)
                net_data = []
                for nic, stats in net_io.items():
                    if stats.packets_sent > 0 or stats.packets_recv > 0:
                        net_data.append({
                            "Interface": nic,
                            "Packets": stats.packets_sent + stats.packets_recv
                        })
                if net_data:
                    net_df = pd.DataFrame(net_data).sort_values("Packets", ascending=False).head(5)
                    fig4 = px.bar(net_df, x="Interface", y="Packets", color="Packets",
                        color_continuous_scale="Blues", title="Live Network Packets per Interface")
                    st.plotly_chart(_theme_adaptive(fig4), use_container_width=True)
                else:
                    st.info("No active network interface data found.")

        except ImportError:
            st.error("psutil not installed. Run: pip install psutil")

        # Correlation: anomaly score vs CPU
        if not df.empty and "reconstruction_error" in df.columns and "timestamp" in df.columns:
            st.subheader("Cross-Layer Correlation")
            df3=df.sort_values("timestamp").tail(100).copy()
            df3["cpu_sim"]=np.clip(df3["reconstruction_error"]*80+np.random.normal(30,5,len(df3)),0,100)
            fig5=go.Figure()
            fig5.add_trace(go.Scatter(x=df3["timestamp"],y=df3["reconstruction_error"],
                name="Anomaly Score",line=dict(color="#00ff88",width=2)))
            fig5.add_trace(go.Scatter(x=df3["timestamp"],y=df3["cpu_sim"]/100,
                name="CPU% (norm)",line=dict(color="#4488ff",width=2,dash="dash"),yaxis="y2"))
            fig5.update_layout(title="Anomaly Score vs CPU Usage",
                yaxis=dict(title="Score",color="#00ff88"),
                yaxis2=dict(title="CPU%",overlaying="y",side="right",color="#4488ff"))
            st.plotly_chart(_theme_adaptive(fig5),use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — GEO-IP ATTACK MAP
# ══════════════════════════════════════════════════════════════════════════════
if t7:
    with t7:
        st.subheader("REAL-TIME GLOBAL ATTACK MAP")
        if df.empty or "src_ip" not in df.columns:
            st.info("No data available to map yet.")
        else:
            # Create geoip columns
            lats, lons, cities, countries = [], [], [], []
            # We only map the most recent 100 unique IPs to avoid API limits/lag
            recent_df = df.head(100).copy()
            for ip in recent_df["src_ip"]:
                lat, lon, city, country = fetch_geoip(ip)
                lats.append(lat)
                lons.append(lon)
                cities.append(city)
                countries.append(country)
            
            recent_df["lat"] = lats
            recent_df["lon"] = lons
            recent_df["city"] = cities
            recent_df["country"] = countries
            
            df_valid = recent_df.dropna(subset=["lat", "lon"])
            if not df_valid.empty:
                fig = go.Figure()
                
                colors = [SEV_COLOR.get(s, "#00ff88") for s in df_valid.get("severity", ["LOW"]*len(df_valid))]
                
                fig.add_trace(go.Scattergeo(
                    lon=df_valid["lon"], lat=df_valid["lat"],
                    text=df_valid["city"] + ", " + df_valid["country"] + "<br>IP: " + df_valid["src_ip"],
                    mode="markers",
                    marker=dict(
                        size=10, 
                        color=colors,
                        opacity=0.8,
                        line=dict(width=1, color="rgba(255,255,255,0.4)")
                    )
                ))
                
                fig.update_geos(
                    projection_type="orthographic",
                    showcoastlines=True, coastlinecolor="rgba(0, 240, 255, 0.4)",
                    showland=True, landcolor="#0f172a",
                    showocean=True, oceancolor="#020617",
                    showlakes=True, lakecolor="#020617",
                    showcountries=True, countrycolor="rgba(0, 240, 255, 0.1)",
                    bgcolor="rgba(0,0,0,0)"
                )
                
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    geo=dict(bgcolor="rgba(0,0,0,0)")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No geospatial data available yet.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — PACKET FORENSICS (Wireshark / TShark)
# ══════════════════════════════════════════════════════════════════════════════
if t6:
    with t6:
        # ── Import wireshark module ──
        try:
            from wireshark_integration import (
                TSHARK_AVAILABLE, get_tshark_version, analyze_pcap, ForensicReport,
                get_protocol_distribution, get_top_talkers, get_suspicious_flows,
                get_bandwidth_timeline,
            )
            _wi_import_ok = True
        except ImportError as _ie:
            _wi_import_ok = False
            st.error(f"wireshark_integration.py not found: {_ie}")

        if _wi_import_ok:
            # ── Status Banner ──
            if TSHARK_AVAILABLE:
                st.success(f"TShark detected — {get_tshark_version()}")
            else:
                st.warning(
                    "**TShark / Wireshark not installed.**  "
                    "Forensic features are disabled.  "
                    "Install: `brew install wireshark` (macOS) or `sudo apt install tshark` (Linux)"
                )

            st.markdown("---")
            st.subheader("Upload PCAP for Forensic Analysis")

            uploaded = st.file_uploader(
                "Drop a .pcap or .pcapng file",
                type=["pcap", "pcapng"],
                disabled=not TSHARK_AVAILABLE,
            )

            max_pkts = st.slider("Max packets to analyse", 500, 20000, 5000, step=500,
                                 disabled=not TSHARK_AVAILABLE)
            run_ai   = st.checkbox("Run AI anomaly detection on PCAP", value=False,
                                   disabled=not TSHARK_AVAILABLE)

            if uploaded and TSHARK_AVAILABLE:
                import tempfile, os
                # Save upload to temp file
                suffix = ".pcapng" if uploaded.name.endswith(".pcapng") else ".pcap"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                st.info(f"Analysing **{uploaded.name}** ({uploaded.size/1024:.1f} KB) …")

                # Progress bar
                prog = st.progress(0, "Parsing packets …")

                def _cb(cur, total):
                    prog.progress(min(int(cur/max(total,1)*100), 99), f"Processing {cur}/{total} packets …")

                try:
                    _detector = None
                    if run_ai:
                        try:
                            from hybrid_detector import HybridDetector
                            from alerts import get_global_dispatcher
                            from system_metrics import SystemMetricsCollector
                            _sc = SystemMetricsCollector(window=10, interval=1.0)
                            _detector = HybridDetector(dispatcher=get_global_dispatcher(),
                                                        sys_collector=_sc, log_anomalies=False)
                        except Exception as _de:
                            st.warning(f"AI engine not available: {_de}. Showing forensics only.")

                    report: ForensicReport = analyze_pcap(
                        tmp_path, detector=_detector,
                        max_packets=max_pkts, progress_callback=_cb,
                    )
                    prog.progress(100, "Done")

                    # ── Overview metrics ──
                    st.markdown("### Forensic Overview")
                    fo1,fo2,fo3,fo4 = st.columns(4)
                    metric_card("Total Packets", f"{report.total_packets:,}", fo1)
                    metric_card("Total Bytes",   f"{report.total_bytes:,}",   fo2)
                    metric_card("Duration",      f"{report.duration_s:.2f}s", fo3)
                    ai_hits = len([r for r in report.ai_results if r.get("is_anomaly")]) if report.ai_results else 0
                    metric_card("AI Anomalies",  ai_hits, fo4)

                    st.markdown("---")

                    # ── Protocol Distribution ──
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        st.subheader("Protocol Distribution")
                        if not report.protocol_dist.empty:
                            fig_proto = px.pie(
                                report.protocol_dist, names="Protocol", values="Count",
                                hole=0.45, title=""
                            )
                            st.plotly_chart(_theme_adaptive(fig_proto), use_container_width=True)
                        else:
                            st.info("No protocol data.")

                    # ── Top Talkers ──
                    with fc2:
                        st.subheader("Top Source IPs (Talkers)")
                        if not report.top_talkers.empty:
                            fig_talk = px.bar(
                                report.top_talkers, x="Packet Count", y="Source IP",
                                orientation="h", color="Packet Count",
                                color_continuous_scale="Reds", title=""
                            )
                            st.plotly_chart(_theme_adaptive(fig_talk), use_container_width=True)
                        else:
                            st.info("No talker data.")

                    # ── Bandwidth Timeline ──
                    st.subheader("Bandwidth / Traffic Volume Timeline")
                    if not report.bandwidth.empty:
                        fig_bw = px.area(
                            report.bandwidth, x="Packet#", y="Bytes",
                            title="Bytes per Packet Bucket",
                            color_discrete_sequence=["#10b981"],
                        )
                        st.plotly_chart(_theme_adaptive(fig_bw), use_container_width=True)

                    # ── Suspicious Flows ──
                    st.subheader(f"Suspicious Flows Detected ({len(report.suspicious)})")
                    if not report.suspicious.empty:
                        st.dataframe(report.suspicious, use_container_width=True)
                    else:
                        st.success("No suspicious TCP flag anomalies detected.")

                    # ── AI Anomaly Timeline from PCAP ──
                    if report.ai_results:
                        st.subheader("AI Anomaly Scoring on PCAP")
                        ai_df = pd.DataFrame([
                            {
                                "index":    i,
                                "score":    r.get("reconstruction_error", 0),
                                "anomaly":  r.get("is_anomaly", False),
                                "severity": r.get("severity", "LOW"),
                                "type":     r.get("attack_type", "Normal"),
                            }
                            for i, r in enumerate(report.ai_results)
                        ])
                        fig_ai = px.scatter(
                            ai_df, x="index", y="score",
                            color="severity", color_discrete_map=SEV_COLOR,
                            hover_data=["type"],
                            title="Anomaly Score per Packet (PCAP Replay)",
                        )
                        st.plotly_chart(_theme_adaptive(fig_ai), use_container_width=True)

                        # Summary counts
                        ac1, ac2, ac3 = st.columns(3)
                        metric_card("Anomalies",   ai_hits, ac1)
                        metric_card("Normal",      len(report.ai_results) - ai_hits, ac2)
                        crit_ai = len([r for r in report.ai_results if r.get("severity") == "CRITICAL"])
                        metric_card("Critical",    crit_ai, ac3)

                    # ── Raw TShark Preview ──
                    with st.expander("Raw TShark Packet Preview (first 5 packets)"):
                        for i, pkt in enumerate(report.raw_packets[:5]):
                            st.json(pkt)

                except Exception as _fe:
                    st.error(f"Forensic analysis error: {_fe}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

            elif not TSHARK_AVAILABLE:
                st.markdown("""
    #### How to Install TShark

    | Platform | Command |
    |---|---|
    | macOS | `brew install wireshark` |
    | Ubuntu/Debian | `sudo apt install tshark` |
    | Windows | Download from [wireshark.org](https://www.wireshark.org/download.html) |

    After installation, restart the dashboard.
                """)
            else:
                st.info("Upload a .pcap or .pcapng file above to begin forensic analysis.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if refresh>0:
    time.sleep(refresh)
    load_df.clear()
    st.rerun()
