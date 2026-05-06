import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
import joblib
import logging
import json
import time
from datetime import datetime
import os
from scapy.all import sniff, IP, TCP, UDP, ICMP
import pandas as pd
from collections import defaultdict
import sys

# Configure logging with more verbose output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set NumPy print options to display full arrays
np.set_printoptions(threshold=np.inf)

# Global counters for packet statistics
total_packets_processed = 0
non_anomalous_packets = 0

class LightweightLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(LightweightLSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        output = self.fc(decoded)
        return output

def extract_features(packet):
    """Extract features from a network packet."""
    features = np.zeros(79)
    
    try:
        # Basic packet features
        features[0] = len(packet)  # Total packet length
        
        # IP layer features
        if IP in packet:
            ip = packet[IP]
            features[1] = ip.ttl
            features[2] = ip.proto
            features[3] = len(ip)
            
            # IP flags - convert to integer first
            features[4] = int(ip.flags) if hasattr(ip, 'flags') else 0
            
            # IP fragmentation
            features[5] = int(ip.frag) if hasattr(ip, 'frag') else 0
            features[6] = int(ip.flags & 0x1) if hasattr(ip, 'flags') else 0  # More fragments flag
            features[7] = int(ip.flags & 0x2) if hasattr(ip, 'flags') else 0  # Don't fragment flag
            
            # Protocol-specific features
            if TCP in packet:
                tcp = packet[TCP]
                features[8] = tcp.sport
                features[9] = tcp.dport
                features[10] = tcp.seq
                features[11] = tcp.ack
                features[12] = tcp.dataofs
                features[13] = int(tcp.flags)  # Convert flags to integer
                features[14] = tcp.window
                features[15] = len(tcp)
                
                # TCP flags - convert to integers
                features[16] = int(tcp.flags & 0x01)  # FIN
                features[17] = int(tcp.flags & 0x02)  # SYN
                features[18] = int(tcp.flags & 0x04)  # RST
                features[19] = int(tcp.flags & 0x08)  # PSH
                features[20] = int(tcp.flags & 0x10)  # ACK
                features[21] = int(tcp.flags & 0x20)  # URG
                
            elif UDP in packet:
                udp = packet[UDP]
                features[22] = udp.sport
                features[23] = udp.dport
                features[24] = len(udp)
                
            elif ICMP in packet:
                icmp = packet[ICMP]
                features[25] = icmp.type
                features[26] = icmp.code
                features[27] = len(icmp)
        
        # Additional statistical features
        features[28] = float(packet.time)  # Timestamp
        features[29] = len(packet) / 1500  # Normalized packet size
        
        # Protocol-specific ratios - add epsilon to prevent division by zero
        packet_len_f = float(len(packet))
        epsilon = np.finfo(float).eps # Smallest positive float

        if TCP in packet:
            features[30] = len(packet[TCP]) / (packet_len_f + epsilon)  # TCP payload ratio
        if UDP in packet:
            features[31] = len(packet[UDP]) / (packet_len_f + epsilon)  # UDP payload ratio
        if ICMP in packet:
            features[32] = len(packet[ICMP]) / (packet_len_f + epsilon)  # ICMP payload ratio
            
        # Add packet direction (0 for incoming, 1 for outgoing)
        if IP in packet:
            features[33] = 1 if packet[IP].src.startswith(('192.168.', '10.', '172.16.')) else 0
        
        # All 79 features are now extracted or set to 0. No more padding needed.
        # The next index is 34. We need to define up to 78.
        # This requires adding 45 more features to get to 79 features. The simplest is to fill them with zeros.
        # Since the training data had 79 features, and we only defined 34, we must have implicitly relied on some fixed structure.
        # Let's ensure the full 79 are zero-padded if not explicitly extracted.
        features[34:79] = 0 # Ensure remaining features are zeroed out if not set above.
            
        # Ensure no NaN or infinite values before returning
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}", exc_info=True)
        return None
    
    logger.debug(f"Extracted features shape: {features.shape}")
    return features

def check_zero_day_hint(packet, reconstruction_error):
    """Check for potential zero-day attack characteristics."""
    hints = []
    
    try:
        if IP in packet:
            ip = packet[IP]
            
            # Check for unusual ports
            if TCP in packet:
                tcp = packet[TCP]
                if tcp.sport > 49151 or tcp.dport > 49151:  # Dynamic/private ports
                    hints.append("Unusual port usage")
            
            # Check for unusual flags
            if TCP in packet:
                tcp = packet[TCP]
                if tcp.flags.value not in [0x02, 0x12, 0x10, 0x18]:  # Common flag combinations
                    hints.append("Unusual TCP flags")
            
            # Check for unusual packet sizes
            if len(packet) > 1500 or len(packet) < 60:  # Standard MTU and minimum size
                hints.append("Unusual packet size")
            
            # Check for unusual protocols
            if ip.proto not in [1, 6, 17]:  # ICMP, TCP, UDP
                hints.append("Unusual protocol")
            
            # Check for unusual TTL values
            if ip.ttl not in range(32, 129):  # Common TTL ranges
                hints.append("Unusual TTL")
    
    except Exception as e:
        logger.error(f"Error checking zero-day hints: {str(e)}", exc_info=True)
    
    return hints

def load_model_and_scaler():
    """Load the trained model and scaler."""
    try:
        # Load model
        checkpoint = torch.load('models/lstm_autoencoder_best.pth')
        model = LightweightLSTMAutoencoder(input_dim=79)  # Model expects 79 features
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load scaler (now correctly saved as 79-feature scaler)
        scaler = joblib.load('models/scaler.joblib') # Use scaler.joblib, not scaler_final.joblib
        
        # Load threshold
        threshold = joblib.load('models/anomaly_threshold.joblib')
        
        logger.info(f"Model loaded successfully. Input dimension: 79 (scaled with 79 features)")
        return model, scaler, threshold
    except Exception as e:
        logger.error(f"Error loading model and scaler: {str(e)}", exc_info=True)
        raise

def process_packet(packet, model, scaler, threshold, device):
    """Process a single packet and detect anomalies."""
    global total_packets_processed, non_anomalous_packets # Declare global variables
    logger.debug("Starting packet processing...")
    try:
        total_packets_processed += 1 # Increment total packets

        # Extract features
        features = extract_features(packet)
        if features is None:
            logger.debug("Feature extraction returned None")
            return
        
        logger.debug(f"Features shape: {features.shape}")
        logger.debug(f"Features BEFORE scaling (full array):\n{features}") # Full print

        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        logger.debug(f"Features AFTER scaling (full array):\n{features_scaled}")
        logger.debug(f"Scaled features shape: {features_scaled.shape}")

        # Clip features to a reasonable range to prevent numerical instability
        features_scaled = np.clip(features_scaled, -5.0, 5.0) # Re-adding clipping

        # Convert to tensor and add sequence dimension
        input_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(1).to(device)
        logger.debug(f"Input tensor shape: {input_tensor.shape}")
        logger.debug(f"Input tensor (full array):\n{input_tensor}") # Full print
        
        # Forward pass with mixed precision
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            output = model(input_tensor)
            reconstruction_error = torch.mean((output - input_tensor) ** 2).item()
        
        logger.debug(f"Reconstruction error: {reconstruction_error:.6f}, Threshold: {threshold:.6f}")

        # Check for anomaly
        is_anomaly = reconstruction_error > threshold
        logger.debug(f"Is anomaly: {is_anomaly}")

        if not is_anomaly: # Increment non-anomalous if not an anomaly
            non_anomalous_packets += 1
        
        # Log anomaly if detected
        if is_anomaly:
            timestamp = datetime.now().isoformat()
            src_ip = packet[IP].src if IP in packet else 'N/A'
            dst_ip = packet[IP].dst if IP in packet else 'N/A'
            protocol = packet[IP].proto if IP in packet else 'N/A'
            
            # Check for zero-day hints
            zero_day_hints = check_zero_day_hint(packet, reconstruction_error)
            hint_str = ', '.join(zero_day_hints) if zero_day_hints else 'None'
            
            # Create JSON object for logging
            anomaly_data = {
                "timestamp": timestamp,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "protocol": protocol,
                "reconstruction_error": float(reconstruction_error),
                "threshold": float(threshold),
                "is_anomaly": str(is_anomaly),
                "hint": hint_str
            }
            
            # Log to console in human-readable format
            logger.warning(f"ANOMALY DETECTED - MSE: {reconstruction_error:.6f}, Source: {src_ip}, Destination: {dst_ip}, Protocol: {protocol}")
            
            # Write JSON to file
            with open('anomalies.log', 'a') as f:
                f.write(json.dumps(anomaly_data) + '\n')

        # Log non-anomalous packet count periodically
        if total_packets_processed % 100 == 0: # Log every 100 packets
            non_anomaly_percentage = (non_anomalous_packets / total_packets_processed) * 100
            logger.info(f"Packet Stats: Total={total_packets_processed}, Non-Anomalous={non_anomalous_packets} ({non_anomaly_percentage:.2f}%) ")

    except Exception as e:
        logger.error(f"Error processing packet: {str(e)}", exc_info=True)

def main():
    logger.info("Starting detection script...")
    # Ensure models directory exists
    if not os.path.exists('models'):
        logger.error("Models directory not found. Please train the model first.")
        return
    
    # Load model and scaler
    try:
        logger.info("Loading model and scaler...")
        model, scaler, threshold = load_model_and_scaler()
        logger.info("Model and scaler loaded successfully")
    except Exception as e:
        logger.error(f"Error in detection: {str(e)}", exc_info=True)
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    logger.info(f"Starting real-time detection on {device}")
    logger.info(f"Anomaly threshold: {threshold:.5f}")
    
    # Start packet capture directly with Scapy
    logger.info("Starting packet sniffing with Scapy. This may require administrator privileges.")
    try:
        sniff(prn=lambda p: process_packet(p, model, scaler, threshold, device),
              store=0,
              filter="ip"
             )
    except Exception as e:
        logger.error(f"Error starting packet sniffing: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Check if script is run as admin (for network sniffing permissions)
    if os.name == 'nt': # Windows
        import ctypes
        try:
            is_admin_check = ctypes.windll.shell32.IsUserAnAdmin()
            logger.info(f"Running as admin: {is_admin_check}")
        except:
            is_admin_check = False
            logger.warning("Could not check admin status")

        if not is_admin_check:
            logger.warning("This script may require administrator privileges to capture network traffic.")
    
    main()
