"""
train_lstm.py
=============
LSTM Autoencoder Training with Adaptive Threshold
Part of: AI-Powered Cross-Layer Network Intrusion Detection System

Enhancements over original train_model.py:
  - Adaptive anomaly threshold (rolling mean + 3σ, not fixed)
  - Validation loss curve saved as JSON
  - Cross-platform torch.amp.autocast (no deprecation warnings)
  - Supports both real dataset and synthetic demo data
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

from preprocess import (
    load_cicids2017, generate_synthetic_data,
    prepare_lstm_data, PACKET_FEAT_DIM
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR   = BASE_DIR / "logs"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "training_lstm.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
BATCH_SIZE    = 64
NUM_EPOCHS    = 10
LEARNING_RATE = 0.001
HIDDEN_DIM    = 64      # increased from original 32 for better representation
THRESHOLD_SIGMA = 3.0   # adaptive threshold = mean + σ * std


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Model Definition (preserved + documented from original)
# ══════════════════════════════════════════════════════════════════════════════

class LightweightLSTMAutoencoder(nn.Module):
    """
    Sequence-to-Sequence LSTM Autoencoder for anomaly detection.

    Architecture:
      Encoder: LSTM(input_dim → hidden_dim)
      Decoder: LSTM(hidden_dim → hidden_dim)
      Output:  Linear(hidden_dim → input_dim)

    A high reconstruction error indicates an anomaly.
    """

    def __init__(self, input_dim: int = PACKET_FEAT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(input_dim,  hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, input_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _  = self.encoder(x)
        decoded, _  = self.decoder(encoded)
        return self.output_fc(decoded)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Adaptive Threshold Computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_adaptive_threshold(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    sigma: float = THRESHOLD_SIGMA,
) -> float:
    """
    Compute an adaptive anomaly threshold based on reconstruction errors
    of the training set (normal traffic).

    Threshold = mean(errors) + sigma * std(errors)

    This avoids a fixed threshold and adapts to the data distribution.
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for (batch,) in dataloader:
            batch = batch.to(device)
            out   = model(batch)
            mse   = torch.mean((out - batch) ** 2, dim=[1, 2])
            errors.extend(mse.cpu().numpy())

    arr       = np.array(errors)
    threshold = float(arr.mean() + sigma * arr.std())
    logger.info(
        f"Adaptive threshold: {threshold:.6f}  "
        f"(mean={arr.mean():.6f}, std={arr.std():.6f}, σ={sigma})"
    )
    return threshold


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    X_scaled:      np.ndarray,
    batch_size:    int   = BATCH_SIZE,
    num_epochs:    int   = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    hidden_dim:    int   = HIDDEN_DIM,
) -> dict:
    """
    Train the LSTM Autoencoder and return model artifacts.

    Args:
        X_scaled:    Scaled feature array (n_samples, n_features)
        batch_size:  Mini-batch size
        num_epochs:  Training epochs
        learning_rate: Adam optimizer LR
        hidden_dim:  LSTM hidden state dimension

    Returns:
        dict with keys: model, threshold, train_loss_curve, val_loss_curve
    """
    input_dim = X_scaled.shape[1]

    # ── Dataset ──────────────────────────────────────────────────────────────
    X_tensor = torch.tensor(
        X_scaled.reshape(-1, 1, input_dim),
        dtype=torch.float32,
    )
    # 80/20 train-val split
    n_val     = int(0.2 * len(X_tensor))
    X_train_t = X_tensor[n_val:]
    X_val_t   = X_tensor[:n_val]

    train_ds = TensorDataset(X_train_t)
    val_ds   = TensorDataset(X_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    model     = LightweightLSTMAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # ── Training pass ─────────────────────────────────────────────────────
        model.train()
        epoch_train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                out  = model(batch)
                loss = criterion(out, batch)

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            epoch_train_loss += loss.item()

        avg_train = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── Validation pass ───────────────────────────────────────────────────
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                    out  = model(batch)
                    loss = criterion(out, batch)
                epoch_val_loss += loss.item()

        avg_val = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val)

        logger.info(f"Epoch {epoch+1}/{num_epochs}  train_loss={avg_train:.6f}  val_loss={avg_val:.6f}")

        # ── Save best ─────────────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss":                 best_val_loss,
                    "input_dim":            input_dim,
                    "hidden_dim":           hidden_dim,
                },
                MODELS_DIR / "lstm_autoencoder_best.pth",
            )
            logger.info(f"  ✓ Best model saved (val_loss={best_val_loss:.6f})")

    # ── Adaptive threshold on the full training set ───────────────────────────
    full_loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    threshold   = compute_adaptive_threshold(model, full_loader, device)

    # ── Persist artifacts ──────────────────────────────────────────────────────
    joblib.dump(threshold, MODELS_DIR / "anomaly_threshold.joblib")

    # Save loss curves
    metrics = {
        "train_loss": train_losses,
        "val_loss":   val_losses,
        "best_val_loss": best_val_loss,
        "threshold":  threshold,
        "trained_at": datetime.utcnow().isoformat(),
    }
    with open(LOGS_DIR / "training_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info(f"Training complete. Threshold: {threshold:.6f}")
    return {"model": model, "threshold": threshold,
            "train_loss": train_losses, "val_loss": val_losses}


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder")
    parser.add_argument("--dataset", default="",
                        help="Path to CSV dataset (CICIDS2017). Leave blank for synthetic demo data.")
    parser.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument("--hidden-dim", type=int,   default=HIDDEN_DIM)
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    if args.dataset and os.path.exists(args.dataset):
        logger.info(f"Loading dataset: {args.dataset}")
        df, _ = load_cicids2017(args.dataset)
        if df is None:
            logger.error("Failed to load dataset. Falling back to synthetic data.")
            df, _ = generate_synthetic_data()
    else:
        logger.info("No dataset provided — using synthetic demo data.")
        df, _ = generate_synthetic_data()

    X_scaled, scaler = prepare_lstm_data(df)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    logger.info(f"Scaler saved. Feature shape: {X_scaled.shape}")

    # ── Train ──────────────────────────────────────────────────────────────────
    result = train_model(
        X_scaled,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
    )

    print(f"\n✅ LSTM training complete!")
    print(f"   Best val loss : {min(result['val_loss']):.6f}")
    print(f"   Threshold     : {result['threshold']:.6f}")
    print(f"   Models saved  : {MODELS_DIR}")
