from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

# ============================================================
# ADIS — M2.1 Intra-Day Anomaly Detection (Tuesday Split)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "cicids2017"
RESULTS_DIR = PROJECT_ROOT / "results" / "m21"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "Bruteforce-Tuesday-no-metadata.parquet"


def log(message=""):
    print(f"[ADIS] {message}")


def prepare_tuesday_split(df, train_benign_ratio=0.75, random_state=42):
    """
    Split Tuesday data:
    - Train: train_benign_ratio of Benign traffic only
    - Test:  Remaining Benign + ALL Attacks
    """
    feature_columns = [c for c in df.columns if c != "Label"]

    # Binary label mapping
    if pd.api.types.is_numeric_dtype(df["Label"]):
        labels = (df["Label"] != 0).astype(int)
    else:
        labels = df["Label"].astype(str).str.strip().str.lower().ne("benign").astype(int)

    benign_df = df[labels == 0]
    attack_df = df[labels == 1]

    log(f"Total Tuesday Benign:  {len(benign_df):,}")
    log(f"Total Tuesday Attacks: {len(attack_df):,}")

    # Split Benign only
    train_benign, test_benign = train_test_split(
        benign_df,
        train_size=train_benign_ratio,
        random_state=random_state,
        shuffle=True
    )

    # Test set = remaining Benign + all Attacks
    test_combined = pd.concat([test_benign, attack_df], axis=0).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)

    X_train = train_benign[feature_columns].copy()
    X_test = test_combined[feature_columns].copy()

    if pd.api.types.is_numeric_dtype(test_combined["Label"]):
        y_test = (test_combined["Label"] != 0).astype(int).to_numpy()
    else:
        y_test = test_combined["Label"].astype(str).str.strip().str.lower().ne("benign").astype(int).to_numpy()

    return X_train, X_test, y_test, feature_columns


def clean_features(X_train, X_test):
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    return X_train, X_test


def main():
    print("=" * 70)
    print("[ADIS] M2.1 — Tuesday Intra-Day Isolation Forest")
    print("=" * 70)

    df = pd.read_parquet(DATA_FILE)
    log(f"Loaded Tuesday dataset: {len(df):,} rows × {len(df.columns)} columns")

    X_train, X_test, y_test, feature_columns = prepare_tuesday_split(df, train_benign_ratio=0.75)
    X_train, X_test = clean_features(X_train, X_test)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    log(f"Training shape (Benign only): {X_train.shape}")
    log(f"Testing shape (Benign + Attacks): {X_test.shape}")

    # Train Isolation Forest
    log("Training Isolation Forest...")
    start = time.perf_counter()
    model = IsolationForest(
        n_estimators=200,
        max_samples=256,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)
    log(f"Training finished in {time.perf_counter() - start:.2f} seconds")

    # Evaluate
    log("Evaluating...")
    raw_decision = model.decision_function(X_test)
    anomaly_scores = -raw_decision

    roc_auc = roc_auc_score(y_test, anomaly_scores)
    pr_auc = average_precision_score(y_test, anomaly_scores)

    # Threshold Tuning for Best F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, anomaly_scores)
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0,
    )
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_idx]

    y_pred_tuned = (anomaly_scores >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_tuned)

    print()
    print("=" * 70)
    print(f"ROC-AUC Score:         {roc_auc:.4f}")
    print(f"PR-AUC (Avg Prec):     {pr_auc:.4f}")
    print(f"Optimized F1:          {best_f1:.4f}")
    print(f"Precision:             {precision_score(y_test, y_pred_tuned, zero_division=0):.4f}")
    print(f"Recall:                {recall_score(y_test, y_pred_tuned, zero_division=0):.4f}")
    print()
    print("Confusion Matrix:")
    print(f"TN: {cm[0][0]:,} | FP: {cm[0][1]:,}")
    print(f"FN: {cm[1][0]:,} | TP: {cm[1][1]:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()