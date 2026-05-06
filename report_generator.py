"""
report_generator.py
===================
PDF & CSV Report Generator
Part of: AI-Powered Cross-Layer Network Intrusion Detection System

Generates:
  - CSV export of full anomaly history
  - Professional PDF report with summary stats, charts, and top-IP tables
"""

import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR    = BASE_DIR / "logs"
REPORTS_DIR.mkdir(exist_ok=True)

# ── Try fpdf2 ─────────────────────────────────────────────────────────────────
try:
    from fpdf import FPDF, XPos, YPos
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    logger.warning("fpdf2 not installed — PDF reports unavailable. Run: pip install fpdf2")

# ── Try matplotlib for embedded charts ────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CSV Export
# ══════════════════════════════════════════════════════════════════════════════

def generate_csv_report(df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """
    Export the anomaly DataFrame to a CSV file.

    Args:
        df:          Anomaly DataFrame
        output_path: Optional explicit path. Defaults to reports/anomaly_report_<ts>.csv

    Returns:
        Path to the saved CSV file.
    """
    if output_path is None:
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"anomaly_report_{ts}.csv"
    else:
        path = Path(output_path)

    df.to_csv(path, index=False)
    logger.info(f"CSV report saved: {path}")
    return str(path)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return CSV as bytes (for Streamlit download button)."""
    return df.to_csv(index=False).encode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Chart Helpers (for embedding in PDF)
# ══════════════════════════════════════════════════════════════════════════════

def _save_severity_pie(df: pd.DataFrame, path: str):
    """Save a severity distribution pie chart."""
    if not MATPLOTLIB_AVAILABLE or "severity" not in df.columns:
        return

    counts = df["severity"].value_counts()
    colors = {"CRITICAL": "#ff0033", "HIGH": "#ff8800",
               "MEDIUM": "#ffff00", "LOW": "#00ff88"}
    clrs   = [colors.get(k, "#aaaaaa") for k in counts.index]

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0a0e1a")
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index,
        colors=clrs, autopct="%1.0f%%",
        textprops={"color": "white"},
    )
    ax.set_title("Severity Distribution", color="white", fontsize=12)
    plt.tight_layout()
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_attack_bar(df: pd.DataFrame, path: str):
    """Save an attack type bar chart."""
    if not MATPLOTLIB_AVAILABLE or "attack_type" not in df.columns:
        return

    counts = df["attack_type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0a0e1a")
    ax.barh(counts.index, counts.values, color="#00ff88", edgecolor="none")
    ax.set_xlabel("Count", color="white")
    ax.set_title("Attack Type Distribution", color="white", fontsize=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    plt.tight_layout()
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_timeline(df: pd.DataFrame, path: str):
    """Save an anomaly score timeline."""
    if not MATPLOTLIB_AVAILABLE or "reconstruction_error" not in df.columns:
        return
    if "timestamp" not in df.columns:
        return

    df2 = df.sort_values("timestamp").tail(200)
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0a0e1a")
    ax.scatter(range(len(df2)), df2["reconstruction_error"],
               c=df2["reconstruction_error"], cmap="RdYlGn_r",
               s=15, alpha=0.8)
    ax.set_xlabel("Event #", color="white")
    ax.set_ylabel("Reconstruction Error", color="white")
    ax.set_title("Anomaly Score Timeline (last 200 events)", color="white")
    ax.tick_params(colors="white")
    plt.tight_layout()
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PDF Report
# ══════════════════════════════════════════════════════════════════════════════

class IDSReportPDF(FPDF if FPDF_AVAILABLE else object):
    """Custom FPDF subclass with header and footer."""

    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(0, 200, 100)
        self.cell(0, 8, "AI-Powered Cross-Layer IDS - Security Report", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(0, 200, 100)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()} | Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", align="C")


def generate_pdf_report(
    df:           pd.DataFrame,
    metrics:      Optional[Dict] = None,
    output_path:  Optional[str]  = None,
) -> Optional[str]:
    """
    Generate a professional PDF security report.

    Args:
        df:          Anomaly DataFrame
        metrics:     Optional dict of model performance metrics
        output_path: Save path. Defaults to reports/report_<ts>.pdf

    Returns:
        Path to saved PDF, or None if fpdf2 not available.
    """
    if not FPDF_AVAILABLE:
        logger.error("fpdf2 not installed. Run: pip install fpdf2")
        return None

    if output_path is None:
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"ids_report_{ts}.pdf"
    else:
        path = Path(output_path)

    # ── Generate chart images ─────────────────────────────────────────────────
    sev_chart   = str(REPORTS_DIR / "_tmp_sev.png")
    atk_chart   = str(REPORTS_DIR / "_tmp_atk.png")
    time_chart  = str(REPORTS_DIR / "_tmp_tl.png")

    _save_severity_pie(df, sev_chart)
    _save_attack_bar(df, atk_chart)
    _save_timeline(df, time_chart)

    # ── PDF Construction ──────────────────────────────────────────────────────
    pdf = IDSReportPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    # ── Title ─────────────────────────────────────────────────────────────────
    pdf.set_fill_color(10, 14, 26)
    pdf.rect(10, 30, 190, 28, "F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 255, 136)
    pdf.set_xy(10, 34)
    pdf.cell(190, 10, "Intrusion Detection System", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(180, 180, 180)
    pdf.cell(190, 8, "AI-Powered Cross-Layer Network Security Report", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    # ── Executive Summary ──────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 255, 136)
    pdf.cell(0, 8, "Executive Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(50, 50, 50)
    pdf.set_font("Helvetica", "", 10)

    total   = len(df)
    by_sev  = df["severity"].value_counts().to_dict()   if "severity"    in df.columns else {}
    by_atk  = df["attack_type"].value_counts().to_dict() if "attack_type" in df.columns else {}
    avg_err = df["reconstruction_error"].mean() if "reconstruction_error" in df.columns else 0
    top_src = df["src_ip"].value_counts().idxmax() if "src_ip" in df.columns and not df.empty else "N/A"

    summary_lines = [
        f"Report Period     : {df['timestamp'].min()} - {df['timestamp'].max()}" if "timestamp" in df.columns else "",
        f"Total Anomalies   : {total:,}",
        f"Critical Events   : {by_sev.get('CRITICAL', 0)}",
        f"High Severity     : {by_sev.get('HIGH', 0)}",
        f"Avg Anomaly Score : {avg_err:.6f}",
        f"Top Attacker IP   : {top_src}",
        f"Most Common Attack: {max(by_atk, key=by_atk.get) if by_atk else 'N/A'}",
    ]
    for line in summary_lines:
        if line:
            pdf.cell(0, 6, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── Charts ────────────────────────────────────────────────────────────────
    import os
    if os.path.exists(sev_chart) and os.path.exists(atk_chart):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(0, 255, 136)
        pdf.cell(0, 8, "Attack Distribution", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.image(sev_chart, x=10,  y=None, w=88)
        pdf.set_xy(105, pdf.get_y() - 42)
        pdf.image(atk_chart, x=105, y=None, w=95)
        pdf.ln(4)

    if os.path.exists(time_chart):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(0, 255, 136)
        pdf.cell(0, 8, "Anomaly Timeline", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.image(time_chart, x=10, y=None, w=185)
        pdf.ln(4)

    # ── Top 10 Suspicious IPs ─────────────────────────────────────────────────
    if "src_ip" in df.columns:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(0, 255, 136)
        pdf.cell(0, 8, "Top 10 Suspicious Source IPs", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(30, 30, 30)

        top_ips = df["src_ip"].value_counts().head(10)
        col_w   = [90, 30, 50, 20]
        headers = ["IP Address", "Attacks", "Primary Type", "Sev."]
        pdf.set_fill_color(20, 40, 60)
        pdf.set_text_color(255, 255, 255)
        for h, w in zip(headers, col_w):
            pdf.cell(w, 7, h, border=1, fill=True)
        pdf.ln()
        pdf.set_text_color(30, 30, 30)

        for ip, count in top_ips.items():
            ip_df = df[df["src_ip"] == ip]
            atk   = ip_df["attack_type"].value_counts().idxmax() if "attack_type" in ip_df.columns else "N/A"
            sev   = ip_df["severity"].value_counts().idxmax()    if "severity"    in ip_df.columns else "N/A"
            row   = [str(ip), str(count), atk, sev]
            fill  = sev in ("CRITICAL", "HIGH")
            pdf.set_fill_color(255, 220, 220) if fill else pdf.set_fill_color(255, 255, 255)
            for cell, w in zip(row, col_w):
                pdf.cell(w, 6, str(cell)[:30], border=1, fill=True)
            pdf.ln()

    # ── Model Metrics ─────────────────────────────────────────────────────────
    if metrics:
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(0, 255, 136)
        pdf.cell(0, 8, "Model Performance Metrics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                pdf.cell(0, 6, f"  {k}: {v}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── Save ──────────────────────────────────────────────────────────────────
    pdf.output(str(path))
    logger.info(f"PDF report saved: {path}")

    # Cleanup temp images
    for tmp in [sev_chart, atk_chart, time_chart]:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

    return str(path)


def generate_pdf_bytes(df: pd.DataFrame, metrics: Optional[Dict] = None) -> Optional[bytes]:
    """Generate PDF and return as bytes for Streamlit download."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    result = generate_pdf_report(df, metrics, output_path=tmp_path)
    if result is None:
        return None

    with open(tmp_path, "rb") as fh:
        data = fh.read()
    os.unlink(tmp_path)
    return data


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate IDS Security Report")
    parser.add_argument("--log",    default="logs/anomalies.log",
                        help="Path to anomalies.log")
    parser.add_argument("--format", choices=["csv", "pdf", "both"], default="both")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        print("Run demo_mode.py --seed 200 first to generate sample data.")
        exit(1)

    # Load anomalies
    records = []
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    print(f"Loaded {len(df)} anomaly records")

    if args.format in ("csv", "both"):
        csv_path = generate_csv_report(df)
        print(f"CSV  → {csv_path}")

    if args.format in ("pdf", "both"):
        pdf_path = generate_pdf_report(df)
        if pdf_path:
            print(f"PDF  → {pdf_path}")
        else:
            print("PDF generation failed (is fpdf2 installed?)")
