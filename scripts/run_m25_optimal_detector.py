from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
# ADIS — M2.5 Optimal Deep Autoencoder Engine
# Pure Reconstruction Tuning on Clean Manifold
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "cicids2017"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "m25"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "Bruteforce-Tuesday-no-metadata.parquet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PureAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.1),
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(0.1),
            nn.Linear(24, 12),  # Bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 24),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(0.1),
            nn.Linear(24, 48),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.1),
            nn.Linear(48, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def load_data(train_benign_ratio=0.75, random_state=42):
    print(f"[ADIS] Loading {DATA_FILE.name}...")
    df = pd.read_parquet(DATA_FILE)
    df = df.replace([np.inf, -np.inf], np.nan)
    medians = df.median(numeric_only=True)
    df = df.fillna(medians)

    feature_cols = [c for c in df.columns if c != "Label"]

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

    y_test = (
        (test_combined["Label"] != 0).astype(int).to_numpy()
        if pd.api.types.is_numeric_dtype(test_combined["Label"])
        else test_combined["Label"].astype(str).str.strip().str.lower().ne("benign").astype(int).to_numpy()
    )

    # Scaling
    X_train_log = np.log1p(np.maximum(0, X_train.to_numpy(dtype=np.float32)))
    X_test_log = np.log1p(np.maximum(0, X_test.to_numpy(dtype=np.float32)))

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_log).astype(np.float32)
    X_test_scaled = np.clip(scaler.transform(X_test_log), 0.0, 1.0).astype(np.float32)

    return X_train_scaled, X_test_scaled, y_test, feature_cols, scaler


def train_engine(X_train: np.ndarray, epochs=10, batch_size=512, lr=1.5e-3):
    input_dim = X_train.shape[1]
    model = PureAutoencoder(input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    dataset = TensorDataset(torch.from_numpy(X_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"[ADIS] Training Pure Autoencoder ({epochs} epochs)...")
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
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
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] — Loss: {epoch_loss:.6f}")

    print(f"[ADIS] Training finished in {time.perf_counter() - start:.2f}s")
    return model


def evaluate_pure(model, X_test, y_test):
    print("[ADIS] Running inference...")
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X_test))
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    errors = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            reconstructed = model(batch_x)
            # Sample-wise squared error across features
            mse = torch.mean((batch_x - reconstructed) ** 2, dim=1)
            errors.append(mse.cpu().numpy())

    anomaly_scores = np.concatenate(errors)

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
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_idx]

    y_pred = (anomaly_scores >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "best_f1": float(best_f1),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "threshold": float(best_thresh),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def main():
    print("=" * 70)
    print("[ADIS] M2.5 — Pure Deep Autoencoder Benchmark")
    print("=" * 70)

    X_train, X_test, y_test, feature_cols, scaler = load_data()
    model = train_engine(X_train, epochs=10, batch_size=512)
    results = evaluate_pure(model, X_test, y_test)

    print("\n" + "=" * 70)
    print(f"ROC-AUC Score:          {results['roc_auc']:.4f}")
    print(f"PR-AUC (Avg Precision): {results['pr_auc']:.4f}")
    print(f"Optimized F1 Score:     {results['best_f1']:.4f}")
    print(f"Precision:              {results['precision']:.4f}")
    print(f"Recall / Detection:     {results['recall']:.4f}")
    print(f"Optimal Threshold:      {results['threshold']:.6f}")
    print("\nConfusion Matrix:")
    print(f"TN: {results['tn']:,} | FP: {results['fp']:,}")
    print(f"FN: {results['fn']:,}  | TP: {results['tp']:,}")
    print("=" * 70)

    # Save stable checkpoint
    torch.save(
        {
            "state_dict": model.state_dict(),
            "threshold": results["threshold"],
            "features": feature_cols,
        },
        MODEL_DIR / "m25_pure_autoencoder.pt",
    )
    joblib.dump(scaler, MODEL_DIR / "m25_scaler.joblib")
    print("[ADIS] Model and Scaler saved successfully.")


if __name__ == "__main__":
    main()