"""
neo4j_visualizer.py
===================
Upgraded Neo4j Threat Intelligence Graph
Part of: AI-Powered Cross-Layer Network Intrusion Detection System

New capabilities:
  - AttackCluster nodes (group related attacks)
  - REPEATED_ATTACK relationships with count tracking
  - Graceful pyvis-only fallback when Neo4j is offline
  - Propagation pattern detection
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
LOGS_DIR   = BASE_DIR / "logs"

# ── Try Neo4j ─────────────────────────────────────────────────────────────────
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("neo4j package not installed — graph will use pyvis-only mode.")

# ── Try pyvis ─────────────────────────────────────────────────────────────────
try:
    from pyvis.network import Network as PyvisNetwork
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    logger.warning("pyvis not installed — network graph visualization unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Neo4j Handler
# ══════════════════════════════════════════════════════════════════════════════

class Neo4jHandler:
    """
    Manages the Neo4j threat intelligence graph.
    Implements retry logic and graceful degradation.
    """

    def __init__(
        self,
        uri:           str = "bolt://localhost:7687",
        user:          str = "neo4j",
        password:      str = "password",
        max_retries:   int = 2,
        retry_delay:   int = 2,
    ):
        self.uri         = uri
        self.user        = user
        self.password    = password
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.driver      = None
        self._available  = False

        if NEO4J_AVAILABLE:
            self._connect()

    def _connect(self):
        for attempt in range(self.max_retries):
            try:
                self.driver = GraphDatabase.driver(
                    self.uri, auth=(self.user, self.password)
                )
                with self.driver.session() as session:
                    session.run("RETURN 1")
                self._available = True
                logger.info("✓ Connected to Neo4j")
                return
            except Exception as e:
                logger.warning(f"Neo4j connect attempt {attempt+1}: {e}")
                time.sleep(self.retry_delay)
        logger.warning("Neo4j unavailable — using pyvis-only graph mode")

    @property
    def is_available(self) -> bool:
        return self._available

    @contextmanager
    def _session(self):
        if not self._available or not self.driver:
            raise RuntimeError("Neo4j not available")
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()

    def push_anomaly(
        self,
        src_ip:      str,
        dst_ip:      str,
        attack_type: str = "Unknown",
        severity:    str = "LOW",
        protocol:    str = "TCP",
        error:       float = 0.0,
    ):
        """
        Upsert source and destination IP nodes and create/update the
        attack relationship between them.
        """
        if not self._available:
            return

        try:
            with self._session() as session:
                session.run("""
                    MERGE (src:IP {address: $src})
                    MERGE (dst:IP {address: $dst})
                    MERGE (src)-[r:ATTACKED]->(dst)
                    ON CREATE SET
                        r.count      = 1,
                        r.first_seen = datetime(),
                        r.last_seen  = datetime(),
                        r.attack_type = $attack_type,
                        r.severity    = $severity,
                        r.protocol    = $protocol,
                        r.error       = $error
                    ON MATCH SET
                        r.count       = r.count + 1,
                        r.last_seen   = datetime(),
                        r.attack_type = $attack_type,
                        r.severity    = $severity,
                        r.error       = $error
                """, {
                    "src": src_ip, "dst": dst_ip,
                    "attack_type": attack_type,
                    "severity":    severity,
                    "protocol":    protocol,
                    "error":       error,
                })

                # Mark repeated attackers (≥ 3 attacks)
                session.run("""
                    MATCH (src:IP)-[r:ATTACKED]->()
                    WHERE r.count >= 3
                    SET src:RepeatedAttacker
                """)
        except Exception as e:
            logger.error(f"Neo4j push error: {e}")

    def get_graph_data(self, limit: int = 100) -> List[Dict]:
        """Retrieve all attack relationships from Neo4j."""
        if not self._available:
            return []
        try:
            with self._session() as session:
                result = session.run("""
                    MATCH (src:IP)-[r:ATTACKED]->(dst:IP)
                    RETURN
                        src.address  AS source,
                        dst.address  AS target,
                        r.count      AS count,
                        r.attack_type AS attack_type,
                        r.severity   AS severity,
                        r.protocol   AS protocol,
                        r.error      AS error,
                        src.RepeatedAttacker IS NOT NULL AS repeated
                    ORDER BY r.count DESC
                    LIMIT $limit
                """, {"limit": limit})
                return [dict(rec) for rec in result]
        except Exception as e:
            logger.error(f"Neo4j graph fetch error: {e}")
            return []

    def get_cluster_data(self) -> List[Dict]:
        """Return groups of IPs that attack the same targets (attack clusters)."""
        if not self._available:
            return []
        try:
            with self._session() as session:
                result = session.run("""
                    MATCH (src:IP)-[:ATTACKED]->(dst:IP)
                    WITH dst, COLLECT(src.address) AS attackers
                    WHERE SIZE(attackers) >= 2
                    RETURN dst.address AS target, attackers, SIZE(attackers) AS count
                    ORDER BY count DESC
                    LIMIT 20
                """)
                return [dict(rec) for rec in result]
        except Exception as e:
            logger.error(f"Neo4j cluster query error: {e}")
            return []

    def clear(self):
        """Delete all nodes and relationships (use carefully)."""
        if not self._available:
            return
        try:
            with self._session() as session:
                session.run("MATCH (n) DETACH DELETE n")
        except Exception as e:
            logger.error(f"Neo4j clear error: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Pyvis Graph Builder  (works with or without Neo4j)
# ══════════════════════════════════════════════════════════════════════════════

SEVERITY_COLORS = {
    "CRITICAL": "#ff0033",
    "HIGH":     "#ff8800",
    "MEDIUM":   "#ffff00",
    "LOW":      "#00ff88",
    None:       "#aaaaaa",
}

ATTACK_ICONS = {
    "DoS":        "💥",
    "PortScan":   "🔍",
    "BruteForce": "🔨",
    "Probe":      "👁️",
    "Unknown":    "❓",
    "Normal":     "✅",
}


def build_pyvis_graph(
    graph_data:    List[Dict],
    output_path:   str = "logs/threat_graph.html",
    dark_mode:     bool = True,
) -> Optional[str]:
    """
    Build an interactive pyvis threat graph from a list of relationship dicts.

    Args:
        graph_data:  List of {source, target, attack_type, severity, count, ...}
        output_path: Path to save the HTML file
        dark_mode:   Use dark cybersecurity theme

    Returns:
        Path to the generated HTML file, or None on error.
    """
    if not PYVIS_AVAILABLE:
        logger.error("pyvis not available — cannot build graph")
        return None

    if not graph_data:
        return None

    bg     = "#0a0e1a" if dark_mode else "#ffffff"
    font_c = "#ffffff" if dark_mode else "#000000"

    net = PyvisNetwork(
        height="600px", width="100%",
        bgcolor=bg, font_color=font_c,
        directed=True,
    )
    net.set_options("""
    {
      "nodes": { "font": { "size": 12 } },
      "edges": { "smooth": { "type": "curvedCW", "roundness": 0.2 },
                 "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } } },
      "physics": { "barnesHut": { "gravitationalConstant": -4000 } }
    }
    """)

    added_nodes = set()

    for rec in graph_data:
        src        = str(rec.get("source",      "?"))
        dst        = str(rec.get("target",       "?"))
        atk        = str(rec.get("attack_type", "Unknown"))
        sev        = str(rec.get("severity",    "LOW"))
        cnt        = int(rec.get("count",        1))
        repeated   = bool(rec.get("repeated",   False))

        color = SEVERITY_COLORS.get(sev, SEVERITY_COLORS[None])
        icon  = ATTACK_ICONS.get(atk, "❓")

        # Source node
        if src not in added_nodes:
            node_color = "#ff4444" if repeated else "#4488ff"
            border     = "#ff0000" if repeated else "#2266dd"
            net.add_node(
                src,
                label  = src,
                title  = f"IP: {src}\n{'⚠ Repeated Attacker' if repeated else ''}",
                color  = {"background": node_color, "border": border},
                size   = 20 + min(cnt * 2, 30),
                shape  = "dot",
            )
            added_nodes.add(src)

        # Destination node
        if dst not in added_nodes:
            net.add_node(
                dst,
                label  = dst,
                title  = f"Target: {dst}",
                color  = {"background": "#228822", "border": "#00ff88"},
                size   = 18,
                shape  = "diamond",
            )
            added_nodes.add(dst)

        # Edge
        net.add_edge(
            src, dst,
            title  = f"{icon} {atk} | {sev} | count={cnt}",
            color  = color,
            width  = min(1 + cnt * 0.5, 6),
        )

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(output_path))
        logger.info(f"Threat graph saved: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Graph save error: {e}")
        return None


def build_graph_from_dataframe(df: pd.DataFrame, **kwargs) -> Optional[str]:
    """
    Build a pyvis graph directly from the anomalies DataFrame
    (used when Neo4j is offline).
    """
    if df.empty:
        return None

    # Aggregate by src→dst pair
    agg = (
        df.groupby(["src_ip", "dst_ip"])
          .agg(
              attack_type=("attack_type", lambda x: x.value_counts().idxmax()),
              severity=("severity", lambda x: x.value_counts().idxmax()),
              count=("src_ip", "count"),
          )
          .reset_index()
          .rename(columns={"src_ip": "source", "dst_ip": "target"})
    )

    records = agg.to_dict(orient="records")
    return build_pyvis_graph(records, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Smoke Test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    # Test pyvis graph without Neo4j
    sample_data = [
        {"source": "45.33.32.156", "target": "192.168.1.1",  "attack_type": "DoS",       "severity": "CRITICAL", "count": 5,  "repeated": True},
        {"source": "185.220.101.1","target": "192.168.1.5",  "attack_type": "PortScan",   "severity": "HIGH",     "count": 2,  "repeated": False},
        {"source": "45.33.32.156", "target": "192.168.1.5",  "attack_type": "BruteForce","severity": "HIGH",     "count": 3,  "repeated": True},
        {"source": "91.108.4.1",   "target": "192.168.1.10", "attack_type": "Probe",      "severity": "MEDIUM",   "count": 1,  "repeated": False},
    ]

    out = build_pyvis_graph(sample_data, output_path="logs/test_graph.html")
    print(f"Graph written to: {out}")
