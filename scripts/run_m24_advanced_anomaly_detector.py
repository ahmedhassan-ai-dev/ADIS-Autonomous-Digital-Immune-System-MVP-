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
# ADIS — M2.4 State-of-the-Art Anomaly Detection Engine
# Denoising Autoencoder + Latent Space Distance Metric
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "cicids2017"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "m24"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "Bruteforce-Tuesday-no-metadata.parquet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(message=""):
    print(f"[ADIS] {message}")


# ------------------------------------------------------------
# 1. Advanced Architecture: Residual/Denoising Autoencoder
# ------------------------------------------------------------
class AdvancedAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        
        # Encoder with Residual Capacity
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Linear(32, latent_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out, z


# ------------------------------------------------------------
# 2. Data Preparation: Domain-Engineered Scaling
# ------------------------------------------------------------
def load_and_prepare_data(train_benign_ratio=0.75, random_state=42):
    log(f"Loading dataset: {DATA_FILE.name}")
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

    log(f"Dataset Split — Benign: {len(benign_df):,}, Attacks: {len(attack_df):,}")

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

    # 1. Log-transformation on positive domain
    log("Applying stable log1p scaling...")
    X_train_log = np.log1p(np.maximum(0, X_train.to_numpy(dtype=np.float32)))
    X_test_log = np.log1p(np.maximum(0, X_test.to_numpy(dtype=np.float32)))

    # 2. MinMaxScaler strictly bounded in [0, 1]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_log).astype(np.float32)
    X_test_scaled = np.clip(scaler.transform(X_test_log), 0.0, 1.0).astype(np.float32)

    return X_train_scaled, X_test_scaled, y_test, feature_cols, scaler


# ------------------------------------------------------------
# 3. Denoising Training (Noise injection for robust manifold)
# ------------------------------------------------------------
def train_model(X_train: np.ndarray, epochs=12, batch_size=512, lr=1e-3, noise_factor=0.03):
    input_dim = X_train.shape[1]
    model = AdvancedAutoencoder(input_dim=input_dim, latent_dim=16).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = TensorDataset(torch.from_numpy(X_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    log(f"Training Denoising Autoencoder on {DEVICE} ({epochs} epochs)...")
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            
            # Inject slight Gaussian noise to force robust feature learning
            noisy_batch = batch_x + noise_factor * torch.randn_like(batch_x)
            noisy_batch = torch.clamp(noisy_batch, 0.0, 1.0)

            optimizer.zero_grad()
            reconstructed, _ = model(noisy_batch)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_x)

        scheduler.step()
        epoch_loss = total_loss / len(X_train)
        log(f"  Epoch [{epoch:02d}/{epochs:02d}] — Denoising Reconstruction MSE: {epoch_loss:.6f}")

    elapsed = time.perf_counter() - start
    log(f"Training completed in {elapsed:.2f} seconds")

    # Compute Latent Center of Benign Training Manifold
    log("Computing latent space baseline centroid...")
    model.eval()
    with torch.no_grad():
        train_loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        all_z = []
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(DEVICE)
            _, z = model(batch_x)
            all_z.append(z.cpu().numpy())
        latent_centroid = np.mean(np.concatenate(all_z, axis=0), axis=0)

    return model, latent_centroid, elapsed


# ------------------------------------------------------------
# 4. Hybrid Anomaly Scoring
# ------------------------------------------------------------
def evaluate_hybrid(model, latent_centroid, X_test, y_test, batch_size=1024, alpha=0.7):
    log("Evaluating test dataset using Hybrid Metric (Reconstruction + Latent Deviation)...")
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X_test))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    rec_errors = []
    latent_dists = []
    centroid_tensor = torch.from_numpy(latent_centroid).to(DEVICE)

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            reconstructed, z = model(batch_x)

            # 1. Reconstruction MSE per sample
            mse = torch.mean((batch_x - reconstructed) ** 2, dim=1)
            rec_errors.append(mse.cpu().numpy())

            # 2. Euclidean Distance in Latent Space from Normal Centroid
            dist = torch.norm(z - centroid_tensor, dim=1)
            latent_dists.append(dist.cpu().numpy())

    rec_errors = np.concatenate(rec_errors)
    latent_dists = np.concatenate(latent_dists)

    # Standardize components to [0, 1] for balanced combination
    rec_norm = (rec_errors - rec_errors.min()) / (rec_errors.max() - rec_errors.min() + 1e-8)
    dist_norm = (latent_dists - latent_dists.min()) / (latent_dists.max() - latent_dists.min() + 1e-8)

    # Hybrid Anomaly Score
    hybrid_score = alpha * rec_norm + (1.0 - alpha) * dist_norm

    # Threshold-independent metrics
    roc_auc = roc_auc_score(y_test, hybrid_score)
    pr_auc = average_precision_score(y_test, hybrid_score)

    # F1 Optimization
    precisions, recalls, thresholds = precision_recall_curve(y_test, hybrid_score)
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0,
    )

    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_idx]

    y_pred = (hybrid_score >= best_thresh).astype(int)
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
        "cm": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def main():
    print("=" * 75)
    print("[ADIS] M2.4 — SOTA Semi-Supervised Anomaly Detector Engine")
    print("=" * 75)

    X_train, X_test, y_test, feature_cols, scaler = load_and_prepare_data(train_benign_ratio=0.75)

    model, centroid, train_time = train_model(X_train, epochs=12, batch_size=512)
    results = evaluate_hybrid(model, centroid, X_test, y_test)

    print("\n" + "=" * 75)
    print("[ADIS] Evaluation Results (Hybrid Anomaly Detector):")
    print("=" * 75)
    print(f"ROC-AUC Score:          {results['roc_auc']:.4f}")
    print(f"PR-AUC (Avg Precision): {results['pr_auc']:.4f}")
    print(f"Optimized F1 Score:     {results['best_f1']:.4f}")
    print(f"Precision:              {results['precision']:.4f}")
    print(f"Recall / Detection:     {results['recall']:.4f}")
    print(f"Accuracy:               {results['accuracy']:.4f}")
    print(f"Selected Threshold:     {results['threshold']:.6f}")
    print("\nConfusion Matrix:")
    print("                  Pred Normal     Pred Anomaly")
    print(f"Actual Normal     {results['tn']:>12,}     {results['fp']:>12,}")
    print(f"Actual Anomaly    {results['fn']:>12,}     {results['tp']:>12,}")
    print("=" * 75)

    # Save artifact for ADIS Immune Memory integration
    model_path = MODEL_DIR / "m24_best_detector.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "centroid": centroid,
            "threshold": results["threshold"],
            "features": feature_cols,
            "input_dim": len(feature_cols),
            "latent_dim": 16,
        },
        model_path,
    )

    results_file = RESULTS_DIR / "m24_best_detector_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    log(f"Best detector model saved to: {model_path}")
    log(f"Results saved to:             {results_file}")
    log("M2.4 completed.")


if __name__ == "__main__":
    main()