# 🔍 Zero-Day Attack Detection System

A comprehensive machine learning-based system for detecting zero-day attacks in network traffic using LSTM Autoencoders, real-time packet analysis, and interactive visualizations.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow](#workflow)
- [Components](#components)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This project implements an advanced zero-day attack detection system that combines:

- **LSTM Autoencoder** for anomaly detection
- **Real-time packet capture** and analysis
- **Interactive dashboard** with visualizations
- **Neo4j graph database** for network relationship analysis
- **Machine learning pipeline** for continuous monitoring

## ✨ Features

- 🔥 **Real-time Detection**: Monitor network traffic in real-time
- 🧠 **ML-Powered**: LSTM Autoencoder for sophisticated anomaly detection
- 📊 **Interactive Dashboard**: Beautiful Streamlit-based visualizations
- 🌐 **Network Graph Analysis**: Neo4j integration for relationship mapping
- 📈 **Comprehensive Metrics**: Detailed analytics and reporting
- 🔄 **Auto-refresh**: Continuous monitoring with automatic updates
- 🎯 **Zero-day Focus**: Specifically designed for unknown attack patterns

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Network       │    │   LSTM          │    │   Dashboard     │
│   Traffic       │───▶│   Autoencoder   │───▶│   (Streamlit)   │
│   (Scapy)       │    │   (PyTorch)     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Packet        │    │   Anomaly       │    │   Neo4j         │
│   Processing    │    │   Detection     │    │   Graph DB      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

### System Requirements
- **Python 3.8+**
- **Windows 10/11** (or Linux/macOS)
- **4GB+ RAM** (recommended)
- **Network interface** with packet capture permissions

### Software Dependencies
- **Neo4j Database** (optional, for graph visualization)
- **Wireshark** (optional, for packet analysis)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd zero_day_detection_project_complete
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Optional: Install Neo4j
For full graph visualization functionality:

1. Download [Neo4j Desktop](https://neo4j.com/download/) or [Neo4j Community Edition](https://neo4j.com/download-center/#community)
2. Create a new database
3. Set password to `password` (or update configuration)
4. Start the database service

## 📖 Usage

### Quick Start

1. **Train the Model** (First time only):
```bash
python src/train_model.py
```

2. **Start Real-time Detection**:
```bash
python src/detect_realtime.py
```

3. **Launch Dashboard**:
```bash
streamlit run src/dashboard.py
```

4. **Access Dashboard**: Open `http://localhost:8501` in your browser

### Complete Workflow

#### Step 1: Data Preparation
```bash
# Preprocess training data
python src/preprocess.py
```

#### Step 2: Model Training
```bash
# Train the LSTM Autoencoder
python src/train_model.py
```

#### Step 3: Real-time Detection
```bash
# Start monitoring network traffic
python src/detect_realtime.py
```

#### Step 4: Visualization
```bash
# Launch the interactive dashboard
streamlit run src/dashboard.py
```

## 🔄 Workflow

### 1. **Data Processing Pipeline**
```
Raw Network Data → Feature Extraction → Preprocessing → Training Data
```

### 2. **Model Training Pipeline**
```
Training Data → LSTM Autoencoder → Model Validation → Model Save
```

### 3. **Real-time Detection Pipeline**
```
Live Packets → Feature Extraction → Model Inference → Anomaly Detection → Logging
```

### 4. **Visualization Pipeline**
```
Anomaly Logs → Dashboard Processing → Interactive Charts → Real-time Updates
```

## 🧩 Components

### Core Modules

#### `src/train_model.py`
- **Purpose**: Train the LSTM Autoencoder model
- **Input**: Network traffic dataset
- **Output**: Trained model files (`*.pth`, `*.joblib`)

#### `src/detect_realtime.py`
- **Purpose**: Real-time packet capture and anomaly detection
- **Input**: Live network traffic
- **Output**: Anomaly logs (`anomalies.log`)

#### `src/dashboard.py`
- **Purpose**: Interactive web dashboard
- **Input**: Anomaly logs and Neo4j data
- **Output**: Real-time visualizations

#### `src/preprocess.py`
- **Purpose**: Data cleaning and preprocessing
- **Input**: Raw network data
- **Output**: Cleaned, scaled data

#### `src/neo4j_visualizer.py`
- **Purpose**: Graph database operations
- **Input**: Anomaly data
- **Output**: Network relationship graphs

### Data Files

- `data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: Training dataset
- `models/`: Directory containing trained models
- `anomalies.log`: Real-time anomaly detection logs
- `temp_graph.html`: Generated network graphs

## ⚙️ Configuration

### Model Configuration
```python
# In train_model.py
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
HIDDEN_DIM = 32
```

### Detection Configuration
```python
# In detect_realtime.py
ANOMALY_THRESHOLD = 0.1
PACKET_BUFFER_SIZE = 100
DETECTION_INTERVAL = 1.0  # seconds
```

### Neo4j Configuration
```python
# In dashboard.py and neo4j_visualizer.py
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
```

## 📊 Dashboard Features

### Real-time Metrics
- Total anomalies detected
- Recent anomalies (5-minute window)
- Average reconstruction error
- Maximum error threshold

### Interactive Visualizations
- **Anomaly Timeline**: Scatter plot of detection events
- **Network Graph**: Interactive Neo4j-powered network relationships
- **IP Analysis**: Source and destination IP distributions
- **Protocol Analysis**: Attack distribution by protocol
- **Zero-day Hints**: Characteristics of detected anomalies

### Data Tables
- Recent anomalies with detailed information
- Export capabilities for analysis

## 🔧 Troubleshooting

### Common Issues

#### 1. **Permission Denied for Packet Capture**
```bash
# Windows: Run as Administrator
# Linux: Use sudo
sudo python src/detect_realtime.py
```

#### 2. **Neo4j Connection Failed**
- Ensure Neo4j is running on `localhost:7687`
- Verify password is set to `password`
- Check firewall settings

#### 3. **Model Loading Errors**
```bash
# Retrain the model
python src/train_model.py
```

#### 4. **Dashboard Not Loading**
```bash
# Check if Streamlit is installed
pip install streamlit

# Run with explicit port
streamlit run src/dashboard.py --server.port 8501
```

#### 5. **Memory Issues**
- Reduce `BATCH_SIZE` in training
- Lower `PACKET_BUFFER_SIZE` in detection
- Close unnecessary applications

### Log Files
- `training.log`: Model training progress
- `detection.log`: Real-time detection logs
- `dashboard.log`: Dashboard operation logs
- `anomalies.log`: Detected anomalies

## 🛠️ Advanced Usage

### Custom Model Training
```python
# Modify hyperparameters in train_model.py
HIDDEN_DIM = 64  # Increase for more complex patterns
NUM_EPOCHS = 10  # More training iterations
LEARNING_RATE = 0.0005  # Lower learning rate
```

### Custom Detection Rules
```python
# Add custom rules in detect_realtime.py
def custom_anomaly_rule(packet, error):
    # Your custom logic here
    return True if error > 0.15 else False
```

### Dashboard Customization
```python
# Modify dashboard.py for custom visualizations
# Add new charts, metrics, or data sources
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints where appropriate
- Write meaningful commit messages

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ISCX Dataset**: Network traffic data
- **PyTorch**: Deep learning framework
- **Streamlit**: Dashboard framework
- **Neo4j**: Graph database
- **Scapy**: Packet manipulation

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the log files
3. Create an issue on GitHub
4. Contact the development team

---

**⚠️ Important Notes:**
- This system requires network packet capture permissions
- Run detection scripts with appropriate privileges
- Monitor system resources during operation
- Keep models updated with new training data
- Regularly backup anomaly logs and models

**🔒 Security Considerations:**
- This tool is for educational and research purposes
- Use only on networks you own or have permission to monitor
- Follow local laws and regulations regarding network monitoring
- Implement appropriate access controls for production use 
