from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
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
# ADIS — M2.2 Deep Autoencoder vs Isolation Forest
# Benchmark: CIC-IDS2017 Tuesday (Patator Anomaly Detection)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "cicids2017"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "m22"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "Bruteforce-Tuesday-no-metadata.parquet"

# Hardware Acceleration Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(message=""):
    print(f"[ADIS] {message}")


# ------------------------------------------------------------
# 1. Deep Autoencoder Architecture
# ------------------------------------------------------------
class NetworkAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 8),  # Latent Space
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),  # يضمن بقاء النواتج محصورة بين 0 و 1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


# ------------------------------------------------------------
# 2. Data Preparation & Scaling
# ------------------------------------------------------------

def load_and_prepare_data(train_benign_ratio=0.75, random_state=42):
    log(f"Loading dataset: {DATA_FILE.name}")
    df = pd.read_parquet(DATA_FILE)

    # 1. تنظيف القيم الشاذة والمفقودة
    df = df.replace([np.inf, -np.inf], np.nan)
    medians = df.median(numeric_only=True)
    df = df.fillna(medians)

    feature_cols = [c for c in df.columns if c != "Label"]

    # 2. استخراج الـ Labels
    if pd.api.types.is_numeric_dtype(df["Label"]):
        labels = (df["Label"] != 0).astype(int)
    else:
        labels = df["Label"].astype(str).str.strip().str.lower().ne("benign").astype(int)

    benign_df = df[labels == 0]
    attack_df = df[labels == 1]

    train_benign, test_benign = train_test_split(
        benign_df,
        train_size=train_benign_ratio,
        random_state=random_state,
        shuffle=True,
    )

    test_combined = pd.concat([test_benign, attack_df], axis=0).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)

    X_train = train_benign[feature_cols].copy()
    X_test = test_combined[feature_cols].copy()

    if pd.api.types.is_numeric_dtype(test_combined["Label"]):
        y_test = (test_combined["Label"] != 0).astype(int).to_numpy()
    else:
        y_test = test_combined["Label"].astype(str).str.strip().str.lower().ne("benign").astype(int).to_numpy()

    # 3. Log-Transformation لضغط القيم المليونية
    log("Applying Log1p transformation to handle heavy-tailed network features...")
    X_train_log = np.log1p(np.maximum(0, X_train.to_numpy()))
    X_test_log = np.log1p(np.maximum(0, X_test.to_numpy()))

    # 4. MinMaxScaler لحصر القيم بين [0, 1]
    log("Fitting MinMaxScaler on Benign training data...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_log).astype(np.float32)
    # Clip test values to [0, 1] to avoid out-of-range gradients
    X_test_scaled = np.clip(scaler.transform(X_test_log), 0.0, 1.0).astype(np.float32)

    return (
        X_train_scaled,
        X_test_scaled,
        y_test,
        feature_cols,
        scaler,
        X_train,
        X_test,
    )


# ------------------------------------------------------------
# 3. Model Training (Autoencoder)
# ------------------------------------------------------------
def train_autoencoder(X_train: np.ndarray, epochs=8, batch_size=512, lr=1e-3):
    input_dim = X_train.shape[1]
    model = NetworkAutoencoder(input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = TensorDataset(torch.from_numpy(X_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    log(f"Training Autoencoder on {DEVICE} ({epochs} epochs, batch_size={batch_size})...")
    start = time.perf_counter()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        epoch_loss = total_loss / len(X_train)
        log(f"  Epoch [{epoch:02d}/{epochs:02d}] — Reconstruction Loss (MSE): {epoch_loss:.6f}")

    elapsed = time.perf_counter() - start
    log(f"Autoencoder training finished in {elapsed:.2f} seconds")
    return model, elapsed


# ------------------------------------------------------------
# 4. Evaluation & Metric Calculation
# ------------------------------------------------------------
def compute_metrics(y_test, anomaly_scores, model_name=""):
    roc_auc = roc_auc_score(y_test, anomaly_scores)
    pr_auc = average_precision_score(y_test, anomaly_scores)

    precisions, recalls, thresholds = precision_recall_curve(y_test, anomaly_scores)
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0,
    )

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_idx]

    y_pred = (anomaly_scores >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "model": model_name,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "best_f1": float(best_f1),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "threshold": float(best_threshold),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def evaluate_autoencoder(model, X_test, y_test, batch_size=1024):
    log("Computing Autoencoder reconstruction errors on test data...")
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X_test))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    errors = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            reconstructed = model(batch_x)
            # Sample-wise MSE
            sample_mse = torch.mean((batch_x - reconstructed) ** 2, dim=1)
            errors.append(sample_mse.cpu().numpy())

    anomaly_scores = np.concatenate(errors)
    return compute_metrics(y_test, anomaly_scores, model_name="Deep Autoencoder")


def evaluate_isolation_forest(X_train, X_test, y_test):
    log("Training Baseline Isolation Forest for direct comparison...")
    start = time.perf_counter()
    iso = IsolationForest(
        n_estimators=200,
        max_samples=256,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train)
    train_time = time.perf_counter() - start

    raw_scores = iso.decision_function(X_test)
    anomaly_scores = -raw_scores

    metrics = compute_metrics(y_test, anomaly_scores, model_name="Isolation Forest")
    metrics["training_time_seconds"] = train_time
    return iso, metrics


# ------------------------------------------------------------
# 5. Reporting & Serialization
# ------------------------------------------------------------
def print_comparison_table(ae_res, iso_res):
    print("\n" + "=" * 78)
    print("[ADIS] M2.2 — Direct Benchmark: Isolation Forest vs Deep Autoencoder")
    print("=" * 78)
    header = f"{'Metric':<25} | {'Isolation Forest':<22} | {'Deep Autoencoder':<22}"
    print(header)
    print("-" * 78)

    rows = [
        ("ROC-AUC Score", f"{iso_res['roc_auc']:.4f}", f"{ae_res['roc_auc']:.4f}"),
        ("PR-AUC (Avg Precision)", f"{iso_res['pr_auc']:.4f}", f"{ae_res['pr_auc']:.4f}"),
        ("Optimized F1 Score", f"{iso_res['best_f1']:.4f}", f"{ae_res['best_f1']:.4f}"),
        ("Precision", f"{iso_res['precision']:.4f}", f"{ae_res['precision']:.4f}"),
        ("Recall / Detection", f"{iso_res['recall']:.4f}", f"{ae_res['recall']:.4f}"),
        ("True Positives (TP)", f"{iso_res['tp']:,}", f"{ae_res['tp']:,}"),
        ("False Positives (FP)", f"{iso_res['fp']:,}", f"{ae_res['fp']:,}"),
        ("False Negatives (FN)", f"{iso_res['fn']:,}", f"{ae_res['fn']:,}"),
    ]

    for label, v1, v2 in rows:
        print(f"{label:<25} | {v1:<22} | {v2:<22}")
    print("=" * 78 + "\n")


def main():
    print("=" * 70)
    print("[ADIS] M2.2 — Deep Autoencoder Anomaly Detection Engine")
    print("=" * 70)

    (
        X_train_scaled,
        X_test_scaled,
        y_test,
        feature_cols,
        scaler,
        X_train_raw,
        X_test_raw,
    ) = load_and_prepare_data(train_benign_ratio=0.75)

    # 1. Train and Evaluate Autoencoder
    ae_model, ae_train_time = train_autoencoder(X_train_scaled, epochs=8, batch_size=512)
    ae_metrics = evaluate_autoencoder(ae_model, X_test_scaled, y_test)
    ae_metrics["training_time_seconds"] = ae_train_time

    # 2. Train and Evaluate Baseline Isolation Forest
    iso_model, iso_metrics = evaluate_isolation_forest(X_train_raw, X_test_raw, y_test)

    # 3. Print Comparison
    print_comparison_table(ae_metrics, iso_metrics)

    # 4. Save Models & Artifacts
    ae_artifact_path = MODEL_DIR / "m22_autoencoder.pt"
    torch.save(
        {
            "state_dict": ae_model.state_dict(),
            "input_dim": len(feature_cols),
            "features": feature_cols,
            "threshold": ae_metrics["threshold"],
        },
        ae_artifact_path,
    )

    scaler_path = MODEL_DIR / "m22_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    # 5. Save Experiment Benchmark Results to JSON
    results_payload = {
        "experiment": "M2.2",
        "dataset": DATA_FILE.name,
        "features_count": len(feature_cols),
        "autoencoder": ae_metrics,
        "isolation_forest": iso_metrics,
    }

    results_file = RESULTS_DIR / "m22_autoencoder_vs_isolation_forest.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=4)

    log(f"Autoencoder model saved to: {ae_artifact_path}")
    log(f"Scaler saved to:            {scaler_path}")
    log(f"Benchmark results saved to: {results_file}")
    log("M2.2 pipeline executed successfully.")


if __name__ == "__main__":
    main()