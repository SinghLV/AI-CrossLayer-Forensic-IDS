"""
preprocess.py
=============
Multi-Dataset Preprocessing Pipeline
Part of: AI-Powered Cross-Layer Network Intrusion Detection System

Supports:
  - CICIDS2017  (CSV format from CIC)
  - NSL-KDD     (KDDTrain+.txt / KDDTest+.txt)
  - UNSW-NB15   (CSV format)
  - Raw packet feature arrays (79-feature vectors from realtime_detector)

All datasets are normalised to a unified feature space before training.
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR   = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.2
PACKET_FEAT_DIM = 79        # feature dimension used by realtime_detector


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Generic Cleaning Utilities
# ══════════════════════════════════════════════════════════════════════════════

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard cleaning pipeline:
    1. Drop all-NaN columns
    2. Convert object → numeric
    3. Handle inf values
    4. IQR-clip outliers
    5. Fill remaining NaN with median
    """
    logger.info(f"Cleaning dataframe: shape={df.shape}")

    # Drop entirely-NaN columns
    df.dropna(axis=1, how="all", inplace=True)

    # Convert object columns to numeric where possible
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0.5 * len(df):    # keep if >50% convertible
            df[col] = converted
        # else leave as string (label column)

    # Inf → NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # IQR clipping for numeric columns
    num_cols = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr     = q3 - q1
        df[col]  = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

    # Fill NaN with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    logger.info(f"Cleaned dataframe: shape={df.shape}")
    return df


def select_features(X: np.ndarray, variance_thresh: float = 0.01) -> Tuple[np.ndarray, object]:
    """
    Remove near-zero-variance features.

    Returns:
        (X_selected, selector) — selector can be used to transform test sets
    """
    selector = VarianceThreshold(threshold=variance_thresh)
    X_sel    = selector.fit_transform(X)
    n_removed = X.shape[1] - X_sel.shape[1]
    logger.info(f"Feature selection: {X.shape[1]} → {X_sel.shape[1]} features ({n_removed} removed)")
    return X_sel, selector


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CICIDS2017 Loader
# ══════════════════════════════════════════════════════════════════════════════

# Canonical label mapping for CICIDS2017
CICIDS_LABEL_MAP = {
    "BENIGN":                     "Normal",
    "DoS Hulk":                   "DoS",
    "DoS GoldenEye":              "DoS",
    "DoS slowloris":              "DoS",
    "DoS Slowhttptest":           "DoS",
    "DDoS":                       "DoS",
    "PortScan":                   "PortScan",
    "FTP-Patator":                "BruteForce",
    "SSH-Patator":                "BruteForce",
    "Bot":                        "BruteForce",
    "Infiltration":               "Probe",
    "Web Attack – Brute Force":   "BruteForce",
    "Web Attack – XSS":           "BruteForce",
    "Web Attack – Sql Injection": "BruteForce",
    "Heartbleed":                 "Probe",
}


def load_cicids2017(path: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Load a CICIDS2017 CSV file.

    Args:
        path: Path to any CICIDS2017 CSV (e.g. Friday-WorkingHours-Afternoon-DDos...)

    Returns:
        (X_df, y_series) where y is a Series of normalised attack labels.
        Returns (None, None) if file is not found.
    """
    if not os.path.exists(path):
        logger.warning(f"CICIDS2017 file not found: {path}")
        return None, None

    logger.info(f"Loading CICIDS2017: {path}")
    df = pd.read_csv(path, low_memory=False)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Identify label column (case-insensitive)
    label_col = None
    for col in df.columns:
        if col.lower() in ("label", " label"):
            label_col = col
            break

    if label_col is None:
        logger.warning("No label column found in CICIDS2017 file — treating as unlabelled")
        df = clean_dataframe(df)
        return df, None

    # Extract and map labels
    y_raw = df[label_col].astype(str).str.strip()
    y     = y_raw.map(lambda x: CICIDS_LABEL_MAP.get(x, "Unknown"))

    df.drop(columns=[label_col], inplace=True)
    df = clean_dataframe(df)

    logger.info(f"CICIDS2017 loaded: {df.shape}, label distribution:\n{y.value_counts()}")
    return df, y


# ══════════════════════════════════════════════════════════════════════════════
# 3.  NSL-KDD Loader
# ══════════════════════════════════════════════════════════════════════════════

NSL_KDD_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty_level",
]

NSL_KDD_LABEL_MAP = {
    "normal": "Normal",
    "dos":    "DoS",   "neptune": "DoS",  "teardrop": "DoS",  "land": "DoS",
    "back":   "DoS",   "pod": "DoS",      "smurf": "DoS",     "apache2": "DoS",
    "portsweep": "PortScan", "satan": "PortScan", "ipsweep": "PortScan", "nmap": "PortScan",
    "ftp_write": "BruteForce", "guess_passwd": "BruteForce", "imap": "BruteForce",
    "warezmaster": "BruteForce", "rootkit": "BruteForce",
    "spy": "Probe", "warezclient": "Probe",
}


def load_nsl_kdd(path: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Load NSL-KDD dataset (KDDTrain+.txt or KDDTest+.txt).

    Returns:
        (X_df, y_series)
    """
    if not os.path.exists(path):
        logger.warning(f"NSL-KDD file not found: {path}")
        return None, None

    logger.info(f"Loading NSL-KDD: {path}")
    df = pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS, low_memory=False)

    y = df["label"].str.lower().map(
        lambda x: NSL_KDD_LABEL_MAP.get(x, "Unknown")
    )
    df.drop(columns=["label", "difficulty_level"], inplace=True)

    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols)

    df = clean_dataframe(df)
    logger.info(f"NSL-KDD loaded: {df.shape}")
    return df, y


# ══════════════════════════════════════════════════════════════════════════════
# 4.  UNSW-NB15 Loader
# ══════════════════════════════════════════════════════════════════════════════

UNSW_LABEL_MAP = {
    "Normal":       "Normal",
    "Generic":      "DoS",
    "Exploits":     "BruteForce",
    "Fuzzers":      "Probe",
    "DoS":          "DoS",
    "Reconnaissance": "PortScan",
    "Analysis":     "Probe",
    "Backdoor":     "BruteForce",
    "Shellcode":    "BruteForce",
    "Worms":        "DoS",
}


def load_unsw_nb15(path: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Load UNSW-NB15 CSV file.

    Returns:
        (X_df, y_series)
    """
    if not os.path.exists(path):
        logger.warning(f"UNSW-NB15 file not found: {path}")
        return None, None

    logger.info(f"Loading UNSW-NB15: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    label_col = None
    for candidate in ("attack_cat", "label", "category"):
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        logger.warning("UNSW-NB15: no label column found, treating as unlabelled")
        df = clean_dataframe(df)
        return df, None

    y = df[label_col].astype(str).str.strip().map(
        lambda x: UNSW_LABEL_MAP.get(x, "Unknown")
    )
    df.drop(columns=[label_col], inplace=True)

    # Drop string columns that don't encode well
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() > 50:
            df.drop(columns=[col], inplace=True)

    df = pd.get_dummies(df, drop_first=True)
    df = clean_dataframe(df)
    logger.info(f"UNSW-NB15 loaded: {df.shape}")
    return df, y


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Unified Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def prepare_lstm_data(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    target_dim: int = PACKET_FEAT_DIM,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Prepare data for LSTM Autoencoder training.
    Pads or truncates feature dimension to `target_dim`.

    Returns:
        (X_scaled, fitted_scaler)
    """
    X = df.values.astype(np.float32)

    # Pad or truncate to target_dim
    if X.shape[1] < target_dim:
        pad = np.zeros((X.shape[0], target_dim - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    elif X.shape[1] > target_dim:
        X = X[:, :target_dim]

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


def prepare_rf_data(
    df: pd.DataFrame,
    y: pd.Series,
    target_dim: int = PACKET_FEAT_DIM,
    test_size: float = TEST_SIZE,
) -> Dict:
    """
    Prepare labelled data for Random Forest training.

    Returns:
        {
            X_train, X_test, y_train, y_test,
            scaler, label_encoder, feature_names
        }
    """
    X = df.values.astype(np.float32)

    # Pad or truncate
    if X.shape[1] < target_dim:
        pad = np.zeros((X.shape[0], target_dim - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    elif X.shape[1] > target_dim:
        X = X[:, :target_dim]

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y.fillna("Unknown").astype(str))

    # Scale
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_enc, test_size=test_size,
        random_state=RANDOM_STATE, stratify=y_enc
    )

    logger.info(f"RF data split — train:{X_tr.shape}, test:{X_te.shape}")
    logger.info(f"Classes: {list(le.classes_)}")

    return {
        "X_train":       X_tr,
        "X_test":        X_te,
        "y_train":       y_tr,
        "y_test":        y_te,
        "scaler":        scaler,
        "label_encoder": le,
        "feature_names": [f"feat_{i}" for i in range(target_dim)],
    }


def generate_synthetic_data(n_normal: int = 5000, n_attack: int = 2000) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic network feature data when no real dataset is available.
    Used for demo mode and smoke-testing the RF classifier.

    Returns:
        (X_df, y_series)
    """
    rng = np.random.default_rng(RANDOM_STATE)

    attack_labels = ["DoS", "PortScan", "BruteForce", "Probe"]
    rows, labels  = [], []

    # Normal traffic
    for _ in range(n_normal):
        feat = rng.normal(loc=0.0, scale=1.0, size=PACKET_FEAT_DIM).astype(np.float32)
        rows.append(feat)
        labels.append("Normal")

    # Attack traffic (each category gets n_attack/4 samples)
    per_class = max(1, n_attack // len(attack_labels))
    for atk in attack_labels:
        offset = rng.uniform(2.0, 5.0)
        for _ in range(per_class):
            feat = rng.normal(loc=offset, scale=1.5, size=PACKET_FEAT_DIM).astype(np.float32)
            rows.append(feat)
            labels.append(atk)

    X_df = pd.DataFrame(np.array(rows), columns=[f"feat_{i}" for i in range(PACKET_FEAT_DIM)])
    y    = pd.Series(labels)

    logger.info(f"Generated synthetic data: {X_df.shape}, labels: {y.value_counts().to_dict()}")
    return X_df, y


# ══════════════════════════════════════════════════════════════════════════════
# 6.  CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-dataset Preprocessing CLI")
    parser.add_argument("--dataset", choices=["cicids", "nslkdd", "unsw", "synthetic"],
                        default="synthetic", help="Dataset to load and preprocess")
    parser.add_argument("--path", default="", help="Path to dataset file (not needed for synthetic)")
    parser.add_argument("--mode", choices=["lstm", "rf"], default="lstm",
                        help="Prepare for LSTM (unlabelled) or RF (labelled) training")
    args = parser.parse_args()

    if args.dataset == "synthetic":
        logger.info("Using synthetic data (demo mode)")
        df, y = generate_synthetic_data()
    elif args.dataset == "cicids":
        df, y = load_cicids2017(args.path)
    elif args.dataset == "nslkdd":
        df, y = load_nsl_kdd(args.path)
    elif args.dataset == "unsw":
        df, y = load_unsw_nb15(args.path)
    else:
        raise ValueError("Unknown dataset")

    if df is None:
        print("Dataset not found. Use --path to specify location.")
        exit(1)

    if args.mode == "lstm":
        X, scaler = prepare_lstm_data(df)
        joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
        print(f"LSTM data ready: {X.shape}. Scaler saved.")
    else:
        if y is None:
            print("Labels required for RF mode. Dataset has no label column.")
            exit(1)
        result = prepare_rf_data(df, y)
        print(f"RF data ready — X_train:{result['X_train'].shape}, classes:{list(result['label_encoder'].classes_)}")
