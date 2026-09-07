from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# ============================================================
# ADIS — M2.3 Threshold Sensitivity & Operating Point Analysis
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "cicids2017"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "m23"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "Bruteforce-Tuesday-no-metadata.parquet"
MODEL_PATH = MODEL_DIR / "m22_autoencoder.pt"
SCALER_PATH = MODEL_DIR / "m22_scaler.joblib"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NetworkAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # مطابقة تامة للمعمارية المخزنة داخل الـ checkpoint
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def load_test_pipeline(train_benign_ratio=0.75, random_state=42):
    print("[ADIS] Loading test dataset & saved scaler...")
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

    _, test_benign = train_test_split(
        benign_df,
        train_size=train_benign_ratio,
        random_state=random_state,
        shuffle=True,
    )

    test_combined = pd.concat([test_benign, attack_df], axis=0).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)

    X_test_raw = test_combined[feature_cols].copy()
    y_test = (
        (test_combined["Label"] != 0).astype(int).to_numpy()
        if pd.api.types.is_numeric_dtype(test_combined["Label"])
        else test_combined["Label"].astype(str).str.strip().str.lower().ne("benign").astype(int).to_numpy()
    )

    scaler = joblib.load(SCALER_PATH)
    X_test_log = np.log1p(np.maximum(0, X_test_raw.to_numpy(dtype=np.float32)))
    X_test_scaled = np.clip(scaler.transform(X_test_log), 0.0, 1.0).astype(np.float32)

    return X_test_scaled, y_test, feature_cols


def extract_reconstruction_errors(model, X_test_scaled, batch_size=1024):
    print("[ADIS] Calculating reconstruction errors (anomaly scores)...")
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X_test_scaled))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    errors = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            reconstructed = model(batch_x)
            sample_mse = torch.mean((batch_x - reconstructed) ** 2, dim=1)
            errors.append(sample_mse.cpu().numpy())

    return np.concatenate(errors)


def find_operating_points(y_test, anomaly_scores):
    fpr_array, tpr_array, roc_thresholds = roc_curve(y_test, anomaly_scores)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, anomaly_scores)

    # 1. Best F1 Threshold
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0,
    )
    best_f1_idx = np.argmax(f1_scores)
    best_f1_thresh = pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else pr_thresholds[-1]

    # 2. Strict Low-FPR Profile (FPR <= 1%)
    fpr_1pct_indices = np.where(fpr_array <= 0.01)[0]
    idx_1pct = fpr_1pct_indices[-1] if len(fpr_1pct_indices) > 0 else 0
    strict_thresh = roc_thresholds[idx_1pct]

    # 3. High-Recall Profile (Target: Recall >= 90%)
    high_rec_indices = np.where(tpr_array >= 0.90)[0]
    idx_high_rec = high_rec_indices[0] if len(high_rec_indices) > 0 else len(tpr_array) - 1
    high_rec_thresh = roc_thresholds[idx_high_rec]

    profiles = {
        "High-F1 (Balanced Engine)": float(best_f1_thresh),
        "Strict Policy (FPR <= 1%)": float(strict_thresh),
        "High-Recall (Recall >= 90%)": float(high_rec_thresh),
    }

    detailed_results = {}
    for name, thresh in profiles.items():
        y_pred = (anomaly_scores >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        detailed_results[name] = {
            "threshold": thresh,
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "fpr": float(fpr),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    return detailed_results, fpr_array, tpr_array, precisions, recalls


def plot_sensitivity_curves(fpr_array, tpr_array, precisions, recalls, detailed_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. ROC Curve (Recall vs FPR)
    ax1.plot(fpr_array, tpr_array, color="#1f77b4", lw=2, label="ROC Curve")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--")

    for name, data in detailed_results.items():
        ax1.scatter(data["fpr"], data["recall"], s=80, label=f"{name} (TPR={data['recall']:.2f}, FPR={data['fpr']:.3f})")

    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax1.set_ylabel("True Positive Rate / Recall", fontsize=11)
    ax1.set_title("ADIS Operating Points on ROC Curve", fontsize=13)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)

    # 2. Precision-Recall Curve
    ax2.plot(recalls, precisions, color="#2ca02c", lw=2, label="PR Curve")
    for name, data in detailed_results.items():
        ax2.scatter(data["recall"], data["precision"], s=80, label=f"{name} (Prec={data['precision']:.2f}, Rec={data['recall']:.2f})")

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall Curve", fontsize=13)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plot_file = RESULTS_DIR / "m23_threshold_sensitivity.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"[ADIS] Sensitivity curves saved to: {plot_file}")


def main():
    print("=" * 75)
    print("[ADIS] M2.3 — Autoencoder Threshold Sensitivity Analysis")
    print("=" * 75)

    X_test_scaled, y_test, feature_cols = load_test_pipeline()

    # تحميل الـ Checkpoint وتطبيق الأوزان
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = NetworkAutoencoder(input_dim=len(feature_cols)).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    anomaly_scores = extract_reconstruction_errors(model, X_test_scaled)
    detailed_results, fpr, tpr, prec, rec = find_operating_points(y_test, anomaly_scores)

    roc_auc = roc_auc_score(y_test, anomaly_scores)
    pr_auc = average_precision_score(y_test, anomaly_scores)

    print("\n" + "=" * 85)
    print(f"{'Operational Profile':<28} | {'Threshold':<11} | {'Recall':<8} | {'FPR':<8} | {'Precision':<9} | {'F1':<8} | {'FP Count':<8}")
    print("-" * 85)
    for name, data in detailed_results.items():
        print(
            f"{name:<28} | {data['threshold']:<11.6f} | {data['recall']:<8.4f} | {data['fpr']:<8.4f} | "
            f"{data['precision']:<9.4f} | {data['f1']:<8.4f} | {data['fp']:<8,}"
        )
    print("=" * 85 + "\n")

    plot_sensitivity_curves(fpr, tpr, prec, rec, detailed_results)

    # حفظ النتائج
    report_file = RESULTS_DIR / "m23_operating_points.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc),
                "profiles": detailed_results,
            },
            f,
            indent=4,
        )
    print(f"[ADIS] Detailed report saved to: {report_file}")
    print("[ADIS] M2.3 executed successfully.")


if __name__ == "__main__":
    main()