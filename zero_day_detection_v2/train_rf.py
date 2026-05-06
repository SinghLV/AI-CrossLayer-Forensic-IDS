"""
train_rf.py
===========
Random Forest Attack Classifier Training
Part of: AI-Powered Cross-Layer Network Intrusion Detection System

Trains a Random Forest to classify the *type* of attack detected by the
LSTM Autoencoder anomaly detector.

Attack classes:
  Normal | DoS | PortScan | BruteForce | Probe | Unknown

Outputs:
  models/rf_classifier.joblib
  models/rf_label_encoder.joblib
  logs/rf_evaluation.json
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from preprocess import (
    load_cicids2017, load_nsl_kdd, load_unsw_nb15,
    generate_synthetic_data, prepare_rf_data
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR   = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "training_rf.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── RF Hyper-parameters ───────────────────────────────────────────────────────
N_ESTIMATORS  = 200
MAX_DEPTH     = 20
MIN_SAMPLES   = 2
N_JOBS        = -1         # use all CPU cores
RANDOM_STATE  = 42
CLASS_WEIGHT  = "balanced" # handle class imbalance automatically


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Model Training
# ══════════════════════════════════════════════════════════════════════════════

def train_random_forest(data: dict) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on labelled network features.

    Args:
        data: dict from prepare_rf_data() with X_train, y_train, etc.

    Returns:
        Fitted RandomForestClassifier
    """
    logger.info(f"Training Random Forest — n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}")

    rf = RandomForestClassifier(
        n_estimators  = N_ESTIMATORS,
        max_depth     = MAX_DEPTH,
        min_samples_leaf = MIN_SAMPLES,
        class_weight  = CLASS_WEIGHT,
        n_jobs        = N_JOBS,
        random_state  = RANDOM_STATE,
        oob_score     = True,
    )
    rf.fit(data["X_train"], data["y_train"])
    logger.info(f"OOB score: {rf.oob_score_:.4f}")

    return rf


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_rf(rf: RandomForestClassifier, data: dict) -> dict:
    """
    Evaluate the trained RF on the held-out test set.
    Generates confusion matrix and ROC curve plots.

    Returns:
        metrics dict
    """
    le       = data["label_encoder"]
    X_test   = data["X_test"]
    y_test   = data["y_test"]
    classes  = list(le.classes_)

    y_pred   = rf.predict(X_test)
    y_proba  = rf.predict_proba(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    logger.info(f"\n{'='*50}")
    logger.info(f"  Random Forest Evaluation")
    logger.info(f"{'='*50}")
    logger.info(f"  Accuracy  : {acc:.4f}")
    logger.info(f"  Precision : {prec:.4f}")
    logger.info(f"  Recall    : {rec:.4f}")
    logger.info(f"  F1-Score  : {f1:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=classes, zero_division=0)}")

    # ── Confusion Matrix Plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax
    )
    ax.set_title("Random Forest — Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved: {cm_path}")

    # ── ROC Curve (one-vs-rest, macro) ────────────────────────────────────────
    y_bin = label_binarize(y_test, classes=list(range(len(classes))))
    if y_bin.shape[1] > 1:
        try:
            auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
            logger.info(f"  ROC AUC (macro): {auc:.4f}")

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_facecolor("#1a1a2e")
            fig.patch.set_facecolor("#0f0f23")
            for idx, cls_name in enumerate(classes):
                if idx < y_bin.shape[1] and idx < y_proba.shape[1]:
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(y_bin[:, idx], y_proba[:, idx])
                    ax.plot(fpr, tpr, label=cls_name, linewidth=2)
            ax.plot([0, 1], [0, 1], "w--", alpha=0.4)
            ax.set_xlabel("False Positive Rate", color="white")
            ax.set_ylabel("True Positive Rate", color="white")
            ax.set_title("ROC Curves (One-vs-Rest)", color="white", fontsize=14)
            ax.legend(facecolor="#1a1a2e", labelcolor="white")
            ax.tick_params(colors="white")
            plt.tight_layout()
            roc_path = REPORTS_DIR / "roc_curve.png"
            fig.savefig(roc_path, dpi=150)
            plt.close(fig)
            logger.info(f"ROC curve saved: {roc_path}")
        except Exception as e:
            auc = None
            logger.warning(f"ROC AUC calculation failed: {e}")
    else:
        auc = None

    # ── Feature Importance Plot ───────────────────────────────────────────────
    importances = rf.feature_importances_
    top_n       = 20
    indices     = np.argsort(importances)[::-1][:top_n]
    feat_names  = data.get("feature_names", [f"feat_{i}" for i in range(len(importances))])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f23")
    ax.barh(
        range(top_n),
        importances[indices][::-1],
        color="#00ff88",
        edgecolor="none",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feat_names[i] for i in indices[::-1]], color="white", fontsize=9)
    ax.set_xlabel("Feature Importance", color="white")
    ax.set_title(f"Top {top_n} Most Important Features", color="white", fontsize=13)
    ax.tick_params(axis="x", colors="white")
    plt.tight_layout()
    fi_path = REPORTS_DIR / "feature_importance.png"
    fig.savefig(fi_path, dpi=150)
    plt.close(fig)
    logger.info(f"Feature importance saved: {fi_path}")

    metrics = {
        "accuracy":   round(acc,  4),
        "precision":  round(prec, 4),
        "recall":     round(rec,  4),
        "f1_score":   round(f1,   4),
        "roc_auc":    round(float(auc), 4) if auc is not None else None,
        "oob_score":  round(rf.oob_score_, 4),
        "classes":    classes,
        "evaluated_at": datetime.utcnow().isoformat(),
    }
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Random Forest Attack Classifier")
    parser.add_argument("--dataset", choices=["cicids", "nslkdd", "unsw", "synthetic"],
                        default="synthetic")
    parser.add_argument("--path", default="", help="Path to dataset CSV")
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    parser.add_argument("--max-depth",    type=int, default=MAX_DEPTH)
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    if args.dataset == "synthetic" or not args.path:
        logger.info("Using synthetic demo data for RF training")
        df, y = generate_synthetic_data(n_normal=8000, n_attack=4000)
    elif args.dataset == "cicids":
        df, y = load_cicids2017(args.path)
    elif args.dataset == "nslkdd":
        df, y = load_nsl_kdd(args.path)
    elif args.dataset == "unsw":
        df, y = load_unsw_nb15(args.path)
    else:
        df, y = generate_synthetic_data()

    if df is None or y is None:
        logger.error("Failed to load dataset with labels. Exiting.")
        sys.exit(1)

    # ── Prepare ────────────────────────────────────────────────────────────────
    data = prepare_rf_data(df, y)

    # ── Train ──────────────────────────────────────────────────────────────────
    N_ESTIMATORS = args.n_estimators
    MAX_DEPTH    = args.max_depth
    rf = train_random_forest(data)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    metrics = evaluate_rf(rf, data)

    # ── Save ───────────────────────────────────────────────────────────────────
    joblib.dump(rf,                    MODELS_DIR / "rf_classifier.joblib")
    joblib.dump(data["label_encoder"], MODELS_DIR / "rf_label_encoder.joblib")
    joblib.dump(data["scaler"],        MODELS_DIR / "rf_scaler.joblib")

    with open(LOGS_DIR / "rf_evaluation.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\n✅ Random Forest training complete!")
    print(f"   Accuracy  : {metrics['accuracy']:.4f}")
    print(f"   F1-Score  : {metrics['f1_score']:.4f}")
    print(f"   Classes   : {metrics['classes']}")
    print(f"   Models saved to: {MODELS_DIR}")
    print(f"   Plots saved to : {REPORTS_DIR}")
