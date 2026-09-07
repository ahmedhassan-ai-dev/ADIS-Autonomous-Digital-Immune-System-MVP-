from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

# ============================================================
# ADIS — Comprehensive Multi-Attack Anomaly Benchmark
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017"
RESULTS_DIR = PROJECT_ROOT / "results" / "comprehensive_benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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


def evaluate_dataset(file_path: Path, model: nn.Module, scaler: MinMaxScaler, feature_cols: list):
    print(f"\n[ADIS] Evaluating on: {file_path.name}")
    df = pd.read_parquet(file_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    medians = df.median(numeric_only=True)
    df = df.fillna(medians)

    # Label extraction
    if pd.api.types.is_numeric_dtype(df["Label"]):
        y = (df["Label"] != 0).astype(int).to_numpy()
    else:
        y = df["Label"].astype(str).str.strip().str.lower().ne("benign").astype(int).to_numpy()

    total_samples = len(y)
    attack_count = int(np.sum(y))
    benign_count = total_samples - attack_count

    print(f"       Total: {total_samples:,} | Benign: {benign_count:,} | Attacks: {attack_count:,}")

    if attack_count == 0:
        print("       [Skip] Dataset contains no attacks (Pure Benign Baseline).")
        return None

    # Preprocessing
    X_raw = df[feature_cols].copy()
    X_log = np.log1p(np.maximum(0, X_raw.to_numpy(dtype=np.float32)))
    X_scaled = np.clip(scaler.transform(X_log), 0.0, 1.0).astype(np.float32)

    # Inference (MSE Reconstruction Error)
    model.eval()
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_scaled)), batch_size=1024, shuffle=False)
    errors = []
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(DEVICE)
            mse = torch.mean((batch_x - model(batch_x)) ** 2, dim=1)
            errors.append(mse.cpu().numpy())

    scores = np.concatenate(errors)

    # Metrics
    roc_auc = roc_auc_score(y, scores)
    pr_auc = average_precision_score(y, scores)

    precisions, recalls, thresholds = precision_recall_curve(y, scores)
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
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)

    return {
        "dataset": file_path.name.replace("-no-metadata.parquet", ""),
        "total": total_samples,
        "attacks": attack_count,
        "benign": benign_count,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(best_f1),
        "precision": float(prec),
        "recall": float(rec),
        "tp": int(tp),
        "fp": int(fp),
        "threshold": float(best_thresh),
    }


def main():
    print("=" * 80)
    print("[ADIS] Multi-Attack Comprehensive Anomaly Detection Benchmark")
    print("=" * 80)

    # 1. التدريب الأساسي على Benign-Monday (أو Benign Portscan/Baseline)
    # نستخدم Benign-Monday كـ True Baseline أو Portscan benign
    train_file = RAW_DIR / "Benign-Monday-no-metadata.parquet"
    if not train_file.exists():
        train_file = RAW_DIR / "Portscan-Friday-no-metadata.parquet"

    print(f"[ADIS] Training Base Autoencoder on: {train_file.name} (Benign Only)...")
    df_train = pd.read_parquet(train_file)
    df_train = df_train.replace([np.inf, -np.inf], np.nan).fillna(df_train.median(numeric_only=True))

    feature_cols = [c for c in df_train.columns if c != "Label"]
    
    # استخراج Benign فقط
    if pd.api.types.is_numeric_dtype(df_train["Label"]):
        benign_mask = (df_train["Label"] == 0)
    else:
        benign_mask = df_train["Label"].astype(str).str.strip().str.lower().eq("benign")
    
    benign_train = df_train[benign_mask][feature_cols].copy()
    if len(benign_train) > 150000:
        benign_train = benign_train.sample(n=150000, random_state=42)

    X_train_log = np.log1p(np.maximum(0, benign_train.to_numpy(dtype=np.float32)))
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_log).astype(np.float32)

    # Train
    model = PureAutoencoder(len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(torch.from_numpy(X_train_scaled)), batch_size=512, shuffle=True, drop_last=True)
    model.train()
    for epoch in range(1, 9):
        total_loss = 0.0
        for (bx,) in loader:
            bx = bx.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, bx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        print(f"  Epoch [{epoch:02d}/08] — Loss: {total_loss / len(X_train_scaled):.6f}")

    # 2. حلقة التقييم على كل الملفات المتاحة
    target_files = sorted(list(RAW_DIR.glob("*.parquet")))
    all_results = []

    for f in target_files:
        res = evaluate_dataset(f, model, scaler, feature_cols)
        if res is not None:
            all_results.append(res)

    # 3. عرض جدول المقارنة الشامل
    print("\n" + "=" * 90)
    print(f"{'Attack Dataset':<25} | {'ROC-AUC':<9} | {'PR-AUC':<8} | {'F1-Score':<9} | {'Precision':<10} | {'Recall':<8}")
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['dataset']:<25} | {r['roc_auc']:<9.4f} | {r['pr_auc']:<8.4f} | "
            f"{r['f1']:<9.4f} | {r['precision']:<10.4f} | {r['recall']:<8.4f}"
        )
    print("=" * 90)

    # حفظ النتائج
    output_file = RESULTS_DIR / "multi_attack_benchmark.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n[ADIS] Benchmark report saved to: {output_file}")


if __name__ == "__main__":
    main()