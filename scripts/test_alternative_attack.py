from pathlib import Path
import time
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
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

# ============================================================
# ADIS — Validation on Volumetric Anomaly (Portscan / DDoS)
# ============================================================

RAW_DIR = Path("data/raw/cicids2017")
# يمكنك تجربة Portscan-Friday أو DDoS-Friday
TARGET_FILE = RAW_DIR / "Portscan-Friday-no-metadata.parquet"
if not TARGET_FILE.exists():
    TARGET_FILE = RAW_DIR / "DDoS-Friday-no-metadata.parquet"

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
            nn.Linear(24, 12),
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
        return self.decoder(self.encoder(x))


def main():
    print("=" * 70)
    print(f"[ADIS] Benchmark Engine on: {TARGET_FILE.name}")
    print("=" * 70)

    df = pd.read_parquet(TARGET_FILE)
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

    print(f"[ADIS] Benign samples: {len(benign_df):,} | Attack samples: {len(attack_df):,}")

    train_benign, test_benign = train_test_split(
        benign_df, train_size=0.75, random_state=42, shuffle=True
    )

    test_combined = pd.concat([test_benign, attack_df], axis=0).sample(
        frac=1.0, random_state=42
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

    # Train
    model = PureAutoencoder(len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_scaled)),
        batch_size=512,
        shuffle=True,
        drop_last=True,
    )

    print("[ADIS] Training on Benign baseline (8 epochs)...")
    for epoch in range(1, 9):
        model.train()
        total_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        print(f"  Epoch [{epoch:02d}/08] — Loss: {total_loss / len(X_train_scaled):.6f}")

    # Inference
    print("[ADIS] Evaluating anomaly detection...")
    model.eval()
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_scaled)), batch_size=1024, shuffle=False)
    errors = []
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(DEVICE)
            mse = torch.mean((batch_x - model(batch_x)) ** 2, dim=1)
            errors.append(mse.cpu().numpy())

    scores = np.concatenate(errors)

    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)

    precisions, recalls, thresholds = precision_recall_curve(y_test, scores)
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0,
    )
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_idx]

    y_pred = (scores >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 70)
    print(f"RESULTS FOR {TARGET_FILE.name}:")
    print("=" * 70)
    print(f"ROC-AUC Score:          {roc_auc:.4f}")
    print(f"PR-AUC:                 {pr_auc:.4f}")
    print(f"Optimized F1 Score:     {best_f1:.4f}")
    print(f"Precision:              {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall / Detection:     {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0][0]:,} | FP: {cm[0][1]:,}")
    print(f"FN: {cm[1][0]:,} | TP: {cm[1][1]:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()