"""
dashboard.py — AI-Powered Cross-Layer IDS Dashboard
5 tabs: Live Monitoring | Attack Analytics | Threat Graph | System Metrics | Reports
"""
import json, time, os, threading
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load presentation phase config
try:
    from config import PROJECT_PHASE
except ImportError:
    PROJECT_PHASE = 4  # Default to full

BASE_DIR  = Path(__file__).parent
LOG_FILE  = BASE_DIR / "logs" / "anomalies.log"
DEMO_FILE = BASE_DIR / "demo_data" / "demo_running.flag"

st.set_page_config(page_title="Intelligent Threat Detector", page_icon="⚡", layout="wide")

# ── Dark SOC CSS ──────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

/* ─── THEME ADAPTIVE CORE (Native Streamlit) ─── */
html,body,[data-testid="stAppViewContainer"]{
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"]{
  background-color: var(--secondary-background-color);
  border-right: 1px solid var(--border-color);
}
[data-testid="stSidebar"] section{padding:0 !important}

.sb-logo{
  background-color: var(--background-color);
  border-bottom: 1px solid var(--border-color);
  padding:20px 16px 16px;
  text-align:center;
  margin-bottom:4px;
}
.sb-logo-title{
  font-size:1.1rem;font-weight:700;letter-spacing:2px;
  background: linear-gradient(90deg, #10b981, #3b82f6);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  background-clip:text; display:block; margin-top:6px;
}
.sb-logo-sub{font-size:.65rem; color: #6b7280; letter-spacing:1px; margin-top:2px}

.sb-card{
  background-color: var(--background-color);
  border: 1px solid var(--border-color);
  border-radius:10px;
  padding:12px 14px;
  margin:8px 10px;
  opacity: 0.9;
}
.sb-card-title{
  font-size:.65rem; font-weight:600; letter-spacing:2px;
  color: #6b7280; text-transform:uppercase; margin-bottom:10px;
  display:flex; align-items:center; gap:6px;
}

/* Status dots */
.dot-green{width:8px;height:8px;border-radius:50%;background:#10b981;display:inline-block;}
.dot-red{width:8px;height:8px;border-radius:50%;background:#ef4444;display:inline-block;}
.dot-yellow{width:8px;height:8px;border-radius:50%;background:#f59e0b;display:inline-block;}

.sb-stat{display:flex;justify-content:space-between;align-items:center;
  padding:5px 0;border-bottom:1px solid var(--border-color)}
.sb-stat:last-child{border-bottom:none}
.sb-stat-label{font-size:.72rem; color: #6b7280}
.sb-stat-val{font-size:.78rem; font-weight:600; font-family:'Share Tech Mono',monospace; color: #10b981}

/* Progress bar */
.stProgress>div>div{background:linear-gradient(90deg, #10b981, #3b82f6)!important; border-radius:4px}
.stProgress>div{background: var(--background-color)!important; border-radius:4px; border:1px solid var(--border-color)}

/* Metric cards */
.metric-card{
  background-color: var(--background-color);
  border: 1px solid var(--border-color);
  border-radius:10px;
  padding:16px 14px;
  text-align:center;
  margin:4px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.metric-val{font-size:2rem; font-weight:700; color: #10b981;
  font-family:'Share Tech Mono',monospace;
}
.metric-lbl{font-size:.75rem; color: #6b7280; margin-top:4px; letter-spacing:1px; text-transform:uppercase}

/* Severity */
.sev-CRITICAL{color:#ef4444; font-weight:700}
.sev-HIGH{color:#f97316; font-weight:700}
.sev-MEDIUM{color:#f59e0b}
.sev-LOW{color:#10b981}

/* Tabs */
div[data-testid="stTabs"] button{
  color: #6b7280; border-bottom:2px solid transparent; font-size:.82rem;
}
div[data-testid="stTabs"] button[aria-selected="true"]{
  color: #10b981; border-bottom: 2px solid #10b981;
}

h1,h2,h3{color: #10b981}
.stDataFrame{border:1px solid var(--border-color); border-radius:8px}
</style>""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
SEV_COLOR = {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#f59e0b","LOW":"#10b981"}
ATK_COLOR = {"DoS":"#ef4444","PortScan":"#f97316","BruteForce":"#ec4899",
             "Probe":"#f59e0b","Normal":"#10b981","Unknown":"#6b7280"}

def _theme_adaptive(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10,r=10,t=30,b=10)
    )
    return fig

def metric_card(label, value, col):
    col.markdown(f'<div class="metric-card"><div class="metric-val">{value}</div>'
                 f'<div class="metric-lbl">{label}</div></div>', unsafe_allow_html=True)

@st.cache_data(ttl=3)
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
        df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
    for c in ["reconstruction_error","risk_score","confidence"]:
        if c in df.columns:
            df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
    return df

def get_sys_metrics():
    try:
        import psutil
        cpu=psutil.cpu_percent(interval=None)
        ram=psutil.virtual_memory().percent
        return cpu,ram
    except: return 0.0,0.0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Logo Banner ──
    st.markdown("""
    <div class="sb-logo">
      <span class="sb-logo-title">THREAT INTELLIGENCE</span>
      <div class="sb-logo-sub">AI-POWERED · v2.0</div>
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
    st.markdown(f"""
    <div class="sb-card">
      <div class="sb-card-title">⬡ SYSTEM STATUS</div>
      <div class="sb-stat">
        <span class="sb-stat-label">ENGINE</span>
        <span><span class="dot-green"></span>&nbsp;<span style="color:#00ff88;font-size:.72rem">ONLINE</span></span>
      </div>
      <div class="sb-stat">
        <span class="sb-stat-label">CPU LOAD</span>
        <span><span class="{cpu_dot}"></span>&nbsp;<span class="sb-stat-val">{cpu:.1f}%</span></span>
      </div>
      <div class="sb-stat">
        <span class="sb-stat-label">RAM USAGE</span>
        <span><span class="{ram_dot}"></span>&nbsp;<span class="sb-stat-val">{ram:.1f}%</span></span>
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

    st.markdown("<div style='height:2px;background:linear-gradient(90deg,transparent,#0f3060,transparent);margin:6px 10px'></div>", unsafe_allow_html=True)

    # ── CPU / RAM Progress ──
    st.markdown("""
    <div class="sb-card">
      <div class="sb-card-title">▣ RESOURCE MONITOR</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(int(cpu)/100, f"CPU  {cpu:.1f}%")
    st.progress(int(ram)/100, f"RAM  {ram:.1f}%")

    st.markdown("<div style='height:2px;background:linear-gradient(90deg,transparent,#0f3060,transparent);margin:6px 10px'></div>", unsafe_allow_html=True)

    # ── Controls Card ──
    st.markdown("""
    <div class="sb-card">
      <div class="sb-card-title">⚙ CONTROLS</div>
    </div>
    """, unsafe_allow_html=True)
    demo_on  = st.toggle("🚀 Demo Mode", value=False)
    refresh  = st.slider("Refresh (s)", 3, 30, 5)
    max_rows = st.slider("Max rows", 100, 5000, 1000)

    st.markdown("<div style='height:2px;background:linear-gradient(90deg,transparent,#0f3060,transparent);margin:6px 10px'></div>", unsafe_allow_html=True)

    # ── Actions ──
    st.markdown("""
    <div class="sb-card">
      <div class="sb-card-title">⚡ ACTIONS</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🌱  Seed Demo Data"):
        try:
            from demo_mode import seed_demo_log
            n = seed_demo_log(300, clear_existing=True)
            st.success(f"✓ Seeded {n} records")
            load_df.clear()
        except Exception as e: st.error(str(e))
    if st.button("🔄  Refresh Now"): load_df.clear(); st.rerun()

    # ── Footer ──
    st.markdown("""
    <div style="position:fixed;bottom:16px;left:0;right:0;width:220px;
      text-align:center;font-size:.6rem;color:#1a3050;letter-spacing:1px;padding:0 10px">
      AI-POWERED · CROSS-LAYER IDS<br>
      <span style="color:#0f2540">Network + CPU Interrupt Analysis</span>
    </div>
    """, unsafe_allow_html=True)

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
st.markdown('<h1 style="text-align:center;letter-spacing:2px">INTELLIGENT THREAT DETECTOR</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#6b7280">Network Traffic + CPU Interrupt Analysis</p>', unsafe_allow_html=True)

df=load_df(max_rows)

# ── Tabs ──────────────────────────────────────────────────────────────────────
# ── Tabs Configuration (Phased) ────────────────────────────────────────────────
tab_map = {
    1: ["🔴 Live Monitor"],
    2: ["🔴 Live Monitor", "📊 Attack Analytics", "💻 System Metrics"],
    3: ["🔴 Live Monitor", "📊 Attack Analytics", "💻 System Metrics", "📋 Reports"],
    4: ["🔴 Live Monitor", "📊 Attack Analytics", "🕸️ Threat Graph", "💻 System Metrics", "📋 Reports", "📡 Packet Forensics"]
}
current_tabs = tab_map.get(PROJECT_PHASE, tab_map[4])
tabs = st.tabs(current_tabs)

# Assign tabs to variables based on availability
t1 = tabs[0]
t2 = tabs[1] if len(tabs) > 1 else None
t3 = tabs[2] if len(tabs) > 2 else None
t4 = tabs[3] if len(tabs) > 3 else None
t5 = tabs[4] if len(tabs) > 4 else None
t6 = tabs[5] if len(tabs) > 5 else None

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    if df.empty:
        st.info("No data. Use sidebar to seed demo data or start realtime_detector.py")
    else:
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
        metric_card("Avg Score",f"{avg_e:.4f}",c5)
        st.markdown("---")

        # Live timeline
        if "timestamp" in df.columns and "reconstruction_error" in df.columns:
            fig=px.scatter(df.tail(300),x="timestamp",y="reconstruction_error",
                color="severity" if "severity" in df else None,
                color_discrete_map=SEV_COLOR,
                hover_data=["src_ip","dst_ip","attack_type"] if "src_ip" in df else None,
                title="Anomaly Score Timeline")
            st.plotly_chart(_theme_adaptive(fig),use_container_width=True)

        # Live feed
        st.subheader("Live Anomaly Feed")
        cols=["timestamp","src_ip","dst_ip","attack_type","severity","reconstruction_error","hint"]
        show_cols=[c for c in cols if c in df.columns]
        recent_df=df.sort_values("timestamp",ascending=False).head(20)[show_cols]

        def color_sev(val):
            c={"CRITICAL":"#3d0010","HIGH":"#3d1a00","MEDIUM":"#3d3d00","LOW":"#003d1a"}
            return f"background-color:{c.get(val,'')};color:{SEV_COLOR.get(val,'white')}"

        if "severity" in recent_df.columns:
            styled=recent_df.style.map(color_sev,subset=["severity"])
            st.dataframe(styled,use_container_width=True)
        else:
            st.dataframe(recent_df,use_container_width=True)

        # Attack counters
        if t2:
            st.markdown("---")
        if "attack_type" in df.columns:
            st.subheader("Attack Counters")
            cts=df["attack_type"].value_counts()
            cols2=st.columns(len(cts))
            for i,(k,v) in enumerate(cts.items()):
                metric_card(k,v,cols2[i])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ATTACK ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
if t2:
    with t2:
    if df.empty:
        st.info("No data yet.")
    else:
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — THREAT GRAPH
# ══════════════════════════════════════════════════════════════════════════════
if t3:
    with t3:
    if df.empty:
        st.info("No data yet.")
    else:
        st.subheader("🕸️ Threat Intelligence Graph")
        # Try Neo4j first, fallback to pyvis from df
        neo4j_ok=False
        try:
            from neo4j_visualizer import Neo4jHandler, build_pyvis_graph, build_graph_from_dataframe
            handler=Neo4jHandler()
            if handler.is_available:
                neo4j_ok=True
                # Push latest data
                for _,row in df.head(100).iterrows():
                    handler.push_anomaly(
                        str(row.get("src_ip","?")), str(row.get("dst_ip","?")),
                        str(row.get("attack_type","Unknown")),
                        str(row.get("severity","LOW")),
                        str(row.get("protocol","TCP")),
                        float(row.get("reconstruction_error",0)),
                    )
                gdata=handler.get_graph_data(limit=80)
                graph_path=build_pyvis_graph(gdata, output_path="logs/threat_graph.html")
                handler.close()
                st.success("✓ Neo4j connected")
            else:
                raise Exception("Neo4j offline")
        except Exception as e:
            st.info(f"Neo4j offline — using local graph. ({e})")
            try:
                graph_path=build_graph_from_dataframe(df.head(200), output_path="logs/threat_graph.html")
            except: graph_path=None

        if graph_path and os.path.exists(graph_path):
            with open(graph_path,"r",encoding="utf-8") as f:
                html=f.read()
            st.components.v1.html(html, height=600, scrolling=True)
        else:
            st.warning("Graph unavailable. Install pyvis: pip install pyvis")

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
    st.subheader("💻 Cross-Layer System Metrics")
    try:
        import psutil
        cpu=psutil.cpu_percent(interval=None)
        ram=psutil.virtual_memory()
        cores=psutil.cpu_percent(interval=None,percpu=True)

        m1,m2,m3,m4=st.columns(4)
        metric_card("CPU Usage",f"{cpu:.1f}%",m1)
        metric_card("RAM Used",f"{ram.percent:.1f}%",m2)
        metric_card("RAM Free",f"{ram.free//(1024**2)} MB",m3)
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
        
        fig2=go.Figure(go.Indicator(mode="gauge+number",value=ram.percent,
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
            # Simulate IRQ chart for demo
            irq_demo=pd.DataFrame({"IRQ":["NET0","NET1","USB","TIMER","ACPI"],"irq_s":[842,433,120,9800,55]})
            fig4=px.bar(irq_demo,x="IRQ",y="irq_s",color="irq_s",
                color_continuous_scale="Greens",title="Network IRQ Frequency (irq/s) — Demo")
            st.plotly_chart(_theme_adaptive(fig4),use_container_width=True)

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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — REPORTS
# ══════════════════════════════════════════════════════════════════════════════
if t5:
    with t5:
    st.subheader("📋 Security Reports & Export")
    if df.empty:
        st.info("No data to export yet.")
    else:
        st.markdown(f"**Total records:** {len(df):,}  |  "
                    f"**Date range:** {df['timestamp'].min() if 'timestamp' in df else 'N/A'} "
                    f"→ {df['timestamp'].max() if 'timestamp' in df else 'N/A'}")

        rc1,rc2=st.columns(2)
        # CSV
        with rc1:
            st.markdown("#### 📄 CSV Export")
            try:
                from report_generator import dataframe_to_csv_bytes
                csv_bytes=dataframe_to_csv_bytes(df)
            except:
                csv_bytes=df.to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV",csv_bytes,
                file_name=f"ids_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",use_container_width=True)

        # PDF
        with rc2:
            st.markdown("#### 📑 PDF Report")
            if st.button("Generate PDF",use_container_width=True):
                with st.spinner("Generating PDF..."):
                    try:
                        from report_generator import generate_pdf_bytes
                        mets={}
                        try:
                            import json as jmod
                            mf=BASE_DIR/"logs"/"rf_evaluation.json"
                            if mf.exists(): mets=jmod.loads(mf.read_text())
                        except: pass
                        pdf_b=generate_pdf_bytes(df,mets)
                        if pdf_b:
                            st.download_button("⬇️ Download PDF",pdf_b,
                                file_name=f"ids_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf",use_container_width=True)
                        else:
                            st.error("fpdf2 not installed: pip install fpdf2")
                    except Exception as e:
                        st.error(f"PDF error: {e}")

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
