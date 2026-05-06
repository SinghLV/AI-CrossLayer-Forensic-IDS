import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import logging
import os
from datetime import datetime
import sys # Import sys for StreamHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout) # Explicitly direct to stdout
    ]
)
logger = logging.getLogger(__name__)

class LightweightLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):  # Reduced hidden_dim for lighter model
        super(LightweightLSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers=1,  # Single layer for efficiency
            batch_first=True,
            bidirectional=False  # Unidirectional for efficiency
        )
        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, input_dim)
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        # Encoder
        encoded, _ = self.encoder(x)
        
        # Decoder
        decoded, _ = self.decoder(encoded)
        
        # Final projection
        output = self.fc(decoded)
        return output

def load_and_preprocess_data(filepath='data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'):
    logger.info(f"Loading data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Original DataFrame shape: {df.shape}")
        
        # Drop any columns that are entirely NaN
        df.dropna(axis=1, how='all', inplace=True)
        
        # Convert object columns to numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Handle infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Handle outliers using IQR method
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        # Fill remaining NaNs with median
        df.fillna(df.median(), inplace=True)
        
        logger.info(f"Final DataFrame shape after preprocessing: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error during data loading and preprocessing: {str(e)}")
        raise

def train_model(data_df, batch_size=64, num_epochs=5, learning_rate=0.001): # Reduced num_epochs
    # Initialize scaler
    scaler_obj = StandardScaler() # Renamed to avoid conflict with GradScaler
    scaled_data = scaler_obj.fit_transform(data_df)
    
    # Reshape data for LSTM (samples, sequence_length, features)
    X = scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1])
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightLSTMAutoencoder(input_dim=X.shape[2])
    model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Initialize gradient scaler for mixed precision training
    amp_scaler = GradScaler() # Renamed to amp_scaler
    
    logger.info(f"Starting training on {device}")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in dataloader:
            inputs = batch[0].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda'): # Updated autocast usage
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
            
            # Backward pass with gradient scaling
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'models/lstm_autoencoder_best.pth')
            joblib.dump(scaler_obj, 'models/scaler.joblib') # Correctly saving StandardScaler
            logger.info(f"Model and scaler saved. Best Loss: {best_loss:.6f}")
    
    # Calculate reconstruction errors for thresholding
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            with torch.amp.autocast(device_type='cuda'): # Updated autocast usage
                outputs = model(inputs)
                # Calculate per-sample MSE for thresholding
                per_sample_error = torch.mean((outputs - inputs) ** 2, dim=[1, 2]) # Keep dim=[1,2] for per-sample MSE
                reconstruction_errors.extend(per_sample_error.cpu().numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    threshold = np.mean(reconstruction_errors) + 3 * np.std(reconstruction_errors)
    
    # Save threshold
    joblib.dump(threshold, 'models/anomaly_threshold.joblib')
    logger.info(f"Training complete. Anomaly threshold: {threshold:.6f}")
    
    return model, scaler_obj, threshold # Return scaler_obj

if __name__ == "__main__":
    try:
        # Load and preprocess data
        data_df = load_and_preprocess_data()
        
        # Train the model
        model, scaler, threshold = train_model(data_df, num_epochs=5) # Pass num_epochs
        logger.info("Training finished successfully")
        
    except Exception as e:
        logger.error(f"An error occurred during the training process: {str(e)}")
