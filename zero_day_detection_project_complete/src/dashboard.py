import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import time
import logging
from typing import Optional, Dict, Any
import os
from neo4j import GraphDatabase
import networkx as nx
from pyvis.network import Network

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure the page
st.set_page_config(
    page_title="Zero-Day Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

class Neo4jConnection:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test the connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def close(self):
        if hasattr(self, 'driver'):
            self.driver.close()

    def create_anomaly_graph(self, anomalies_df):
        """Create or update the anomaly graph in Neo4j."""
        try:
            with self.driver.session() as session:
                # Clear existing data
                session.run("MATCH (n) DETACH DELETE n")
                
                # Create constraints if they don't exist
                try:
                    # Check if constraints exist first
                    result = session.run("SHOW CONSTRAINTS")
                    existing_constraints = [record["name"] for record in result]
                    
                    if "ip_address_unique" not in existing_constraints:
                        session.run("CREATE CONSTRAINT ip_address_unique FOR (i:IP) REQUIRE i.address IS UNIQUE")
                    
                    if "anomaly_id_unique" not in existing_constraints:
                        session.run("CREATE CONSTRAINT anomaly_id_unique FOR (a:Anomaly) REQUIRE a.id IS UNIQUE")
                except Exception as e:
                    logger.warning(f"Could not create constraints: {str(e)}")
                
                # Create nodes and relationships
                for _, row in anomalies_df.iterrows():
                    try:
                        session.run("""
                            MERGE (src:IP {address: $src_ip})
                            MERGE (dst:IP {address: $dst_ip})
                            CREATE (a:Anomaly {
                                id: $id,
                                timestamp: $timestamp,
                                error: $error,
                                protocol: $protocol
                            })
                            CREATE (src)-[:SENT]->(a)
                            CREATE (a)-[:TARGETED]->(dst)
                        """, {
                            'src_ip': row['src_ip'],
                            'dst_ip': row['dst_ip'],
                            'id': f"{row['timestamp']}_{row['src_ip']}_{row['dst_ip']}",
                            'timestamp': row['timestamp'].isoformat(),
                            'error': float(row['reconstruction_error']),
                            'protocol': row['protocol']
                        })
                    except Exception as e:
                        logger.error(f"Error creating graph for row: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"Error in create_anomaly_graph: {str(e)}")
            raise

    def get_graph_data(self):
        """Retrieve graph data from Neo4j."""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (src:IP)-[r1:SENT]->(a:Anomaly)-[r2:TARGETED]->(dst:IP)
                    RETURN src.address as source, dst.address as target, 
                           a.error as error, a.protocol as protocol,
                           a.timestamp as timestamp
                    LIMIT 100
                """)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error in get_graph_data: {str(e)}")
            return []

def create_network_graph(graph_data):
    """Create an interactive network graph using pyvis."""
    try:
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        
        # Add nodes and edges
        for record in graph_data:
            # Add source node
            net.add_node(record['source'], 
                        title=f"Source IP: {record['source']}",
                        color="#ff9999")
            
            # Add target node
            net.add_node(record['target'],
                        title=f"Target IP: {record['target']}",
                        color="#99ff99")
            
            # Add edge
            net.add_edge(record['source'], 
                        record['target'],
                        title=f"Protocol: {record['protocol']}\nError: {record['error']:.4f}",
                        value=record['error'])
        
        # Save the graph
        net.save_graph("temp_graph.html")
        return "temp_graph.html"
    except Exception as e:
        logger.error(f"Error creating network graph: {str(e)}")
        return None

@st.cache_data(ttl=5)  # Reduced cache time to 5 seconds
def load_anomalies(max_entries: int = 1000) -> pd.DataFrame:
    """
    Load and parse anomalies from the log file with improved error handling and validation.
    
    Args:
        max_entries (int): Maximum number of entries to load
        
    Returns:
        pd.DataFrame: DataFrame containing anomaly data
    """
    try:
        anomalies = []
        if not os.path.exists('anomalies.log'):
            logger.warning("anomalies.log file not found")
            return pd.DataFrame()
            
        # Get the current file size
        current_size = os.path.getsize('anomalies.log')
        
        # Read the file in reverse to get the most recent entries first
        with open('anomalies.log', 'r') as f:
            # Read all lines and reverse them
            lines = f.readlines()
            lines.reverse()
            
            for line_num, line in enumerate(lines, 1):
                if len(anomalies) >= max_entries:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # Try parsing as JSON first
                    anomaly = json.loads(line)
                    if not isinstance(anomaly, dict):
                        continue
                        
                    # Validate required fields
                    required_fields = ['timestamp', 'src_ip', 'dst_ip', 'reconstruction_error']
                    if not all(field in anomaly for field in required_fields):
                        continue
                        
                    # Convert is_anomaly to boolean
                    if 'is_anomaly' in anomaly:
                        anomaly['is_anomaly'] = str(anomaly['is_anomaly']).lower() == 'true'
                        
                    anomalies.append(anomaly)
                except json.JSONDecodeError:
                    try:
                        # Parse text format
                        parts = line.split(' - ')
                        if len(parts) >= 3:
                            timestamp = parts[0]
                            mse_part = parts[2].split(',')[0]
                            mse_value = float(mse_part.split(':')[1].strip())
                            
                            src_ip = None
                            dst_ip = None
                            for part in parts[2].split(','):
                                if 'Source:' in part:
                                    src_ip = part.split(':')[1].strip()
                                elif 'Destination:' in part:
                                    dst_ip = part.split(':')[1].strip()
                            
                            if src_ip and dst_ip:
                                anomaly = {
                                    "timestamp": timestamp,
                                    "src_ip": src_ip,
                                    "dst_ip": dst_ip,
                                    "reconstruction_error": mse_value,
                                    "protocol": "Unknown",
                                    "is_anomaly": True,
                                    "hint": "None"
                                }
                                anomalies.append(anomaly)
                    except Exception as e:
                        logger.error(f"Error parsing line {line_num}: {str(e)}")
                        continue
                        
        df = pd.DataFrame(anomalies)
        if not df.empty:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                logger.error(f"Error converting timestamps: {str(e)}")
                return pd.DataFrame()
                
        return df
    except Exception as e:
        logger.error(f"Unexpected error in load_anomalies: {str(e)}")
        return pd.DataFrame()

def update_dashboard():
    """Update the dashboard with latest data."""
    try:
        # Load anomalies
        df = load_anomalies()
        
        if df.empty:
            st.warning("No anomalies detected yet. Waiting for data...")
            return
        
        # Calculate metrics
        total_anomalies = len(df)
        recent_anomalies = len(df[df['timestamp'] > datetime.now() - timedelta(minutes=5)])
        avg_error = df['reconstruction_error'].mean()
        max_error = df['reconstruction_error'].max()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Anomalies", total_anomalies)
        with col2:
            st.metric("Recent Anomalies (5min)", recent_anomalies)
        with col3:
            st.metric("Average Error", f"{avg_error:.4f}")
        with col4:
            st.metric("Max Error", f"{max_error:.4f}")
        
        # Anomaly Timeline
        st.subheader("Anomaly Timeline")
        fig_timeline = px.scatter(
            df,
            x='timestamp',
            y='reconstruction_error',
            color='protocol',
            hover_data=['src_ip', 'dst_ip', 'hint'],
            title="Anomaly Detection Timeline"
        )
        fig_timeline.update_layout(
            xaxis_title="Time",
            yaxis_title="Reconstruction Error",
            showlegend=True
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Neo4j Graph Visualization
        st.subheader("Network Graph Analysis")
        
        try:
            # Initialize Neo4j connection
            neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            
            # Update graph with latest data
            neo4j_conn.create_anomaly_graph(df)
            
            # Get graph data and create visualization
            graph_data = neo4j_conn.get_graph_data()
            if graph_data:
                graph_file = create_network_graph(graph_data)
                if graph_file:
                    with open(graph_file, 'r', encoding='utf-8') as f:
                        html = f.read()
                    st.components.v1.html(html, height=600)
                else:
                    st.error("Failed to create network graph visualization")
            else:
                st.info("No graph data available yet.")
                
            neo4j_conn.close()
        except Exception as e:
            st.error(f"Could not connect to Neo4j database: {str(e)}")
            st.info("""
            To fix this:
            1. Make sure Neo4j is installed and running
            2. Default password is 'password'
            3. Ensure Neo4j is running on bolt://localhost:7687
            """)
        
        # IP Analysis
        st.subheader("IP Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Source IP Distribution
            source_ip_counts = df['src_ip'].value_counts().head(10)
            fig_source = px.bar(
                x=source_ip_counts.index,
                y=source_ip_counts.values,
                title="Top 10 Source IPs",
                labels={'x': 'Source IP', 'y': 'Number of Anomalies'}
            )
            st.plotly_chart(fig_source, use_container_width=True)
        
        with col2:
            # Destination IP Distribution
            dest_ip_counts = df['dst_ip'].value_counts().head(10)
            fig_dest = px.bar(
                x=dest_ip_counts.index,
                y=dest_ip_counts.values,
                title="Top 10 Destination IPs",
                labels={'x': 'Destination IP', 'y': 'Number of Anomalies'}
            )
            st.plotly_chart(fig_dest, use_container_width=True)
        
        # Protocol Analysis
        st.subheader("Protocol Analysis")
        protocol_counts = df['protocol'].value_counts()
        fig_protocol = px.pie(
            values=protocol_counts.values,
            names=protocol_counts.index,
            title="Anomaly Distribution by Protocol"
        )
        st.plotly_chart(fig_protocol, use_container_width=True)
        
        # Zero-Day Hints Analysis
        st.subheader("Zero-Day Attack Hints")
        all_hints = []
        for hints in df['hint']:
            if hints != 'None':
                all_hints.extend(hints.split(', '))
        hint_counts = pd.Series(all_hints).value_counts()
        
        if not hint_counts.empty:
            fig_hints = px.bar(
                x=hint_counts.index,
                y=hint_counts.values,
                title="Zero-Day Attack Characteristics",
                labels={'x': 'Characteristic', 'y': 'Frequency'}
            )
            st.plotly_chart(fig_hints, use_container_width=True)
        else:
            st.info("No zero-day attack hints detected yet.")
        
        # Recent Anomalies Table
        st.subheader("Recent Anomalies")
        recent_df = df.sort_values('timestamp', ascending=False).head(10)
        st.dataframe(
            recent_df[['timestamp', 'src_ip', 'dst_ip', 'protocol', 'reconstruction_error', 'hint']],
            use_container_width=True
        )
    except Exception as e:
        logger.error(f"Error updating dashboard: {str(e)}")
        st.error("An error occurred while updating the dashboard. Please check the logs for details.")

def main():
    st.title("🔍 Zero-Day Attack Detection Dashboard")
    
    # Update dashboard
    update_dashboard()
    
    # Add refresh button
    if st.button("🔄 Refresh Now"):
        st.rerun()

if __name__ == "__main__":
    main() 