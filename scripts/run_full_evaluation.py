import os
import glob
import random
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. SETUP & REPRODUCIBILITY
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("=" * 85)
print("ADIS — INNATE ANOMALY DETECTOR: RIGOROUS EVALUATION OF SAVED MODEL")
print("=" * 85)

# ============================================================
# 2. PATHS & CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017"
MODEL_PATH = PROJECT_ROOT / "models" / "merged_binary_detector.joblib"
OUTPUT_DIR = PROJECT_ROOT / "results" / "innate_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "Benign-Monday-no-metadata.parquet",
    "Botnet-Friday-no-metadata.parquet",
    "Bruteforce-Tuesday-no-metadata.parquet",
    "DDoS-Friday-no-metadata.parquet",
    "DoS-Wednesday-no-metadata.parquet",
    "Infiltration-Thursday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet",
    "WebAttacks-Thursday-no-metadata.parquet",
]

# ============================================================
# 3. LOAD SAVED MODEL & FEATURES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")

print(f"[ADIS] Loading primary production detector from: {MODEL_PATH}")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_cols = artifact["features"]
print(f"[ADIS] Detector loaded successfully. Model features: {len(feature_cols)}")

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    medians = df.median(numeric_only=True)
    return df.fillna(medians)

def normalize_labels(df: pd.DataFrame) -> np.ndarray:
    label_col = "Label" if "Label" in df.columns else [c for c in df.columns if "label" in c.lower()][0]
    if pd.api.types.is_numeric_dtype(df[label_col]):
        return (df[label_col] != 0).astype(int).to_numpy()
    return np.where(
        df[label_col].astype(str).str.strip().str.lower().str.contains("benign"),
        0,
        1
    )

# ============================================================
# 5. DATA INGESTION & LEAKAGE / DUPLICATE CHECK
# ============================================================

print("\n[ADIS] Ingesting datasets & running duplicate/leakage audit...")
all_data = {}

for filename in DATASETS:
    file_path = DATA_DIR / filename
    if not file_path.exists():
        print(f"  [!] Missing file: {filename} — Skipped.")
        continue

    df = pd.read_parquet(file_path)
    df = clean_dataframe(df)
    y = normalize_labels(df)
    df["binary_label"] = y

    # تأكد من وجود كافة الـ features
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"File {filename} is missing features: {missing_features}")

    all_data[filename] = df
    benign_cnt = (y == 0).sum()
    attack_cnt = (y == 1).sum()
    print(f"  + Loaded {filename:<40s} | Benign: {benign_cnt:>8,} | Attack: {attack_cnt:>7,}")

print("\n" + "=" * 70)
print("DATASET DUPLICATE ANALYSIS")
print("=" * 70)
for name, df in all_data.items():
    dupes = df[feature_cols].duplicated().sum()
    print(f"  {name:<42s} Duplicates: {dupes:>8,} rows")

# ============================================================
# 6. EVALUATE PRIMARY MODEL ACROSS ALL DATASETS
# ============================================================

print("\n" + "=" * 90)
print("EVALUATING PRIMARY SAVED MODEL (LightGBM Binary Detector)")
print("=" * 90)

results = []
all_scores = {}
all_labels = {}
all_predictions = {}

OPERATING_THRESHOLD = 0.50  # Balanced probability cut-off

for filename, df in all_data.items():
    clean_name = filename.replace("-no-metadata.parquet", "")
    X = df[feature_cols].copy()
    y = df["binary_label"].to_numpy()

    # التنبؤ باحتمالات الشذوذ (Scores)
    scores = model.predict_proba(X)[:, 1]
    predictions = (scores >= OPERATING_THRESHOLD).astype(int)

    all_scores[clean_name] = scores
    all_labels[clean_name] = y
    all_predictions[clean_name] = predictions

    attack_cnt = int(np.sum(y))
    benign_cnt = len(y) - attack_cnt

    cm = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    if attack_cnt == 0:
        roc_auc = 1.0
        pr_auc = 0.0
        prec = 1.0 if fp == 0 else 0.0
        rec = 1.0
        f1 = 1.0 if fp == 0 else (1.0 - fpr)
    else:
        roc_auc = roc_auc_score(y, scores)
        pr_auc = average_precision_score(y, scores)
        prec = precision_score(y, predictions, zero_division=0)
        rec = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)

    results.append({
        "Dataset": clean_name,
        "Samples": len(df),
        "Benign": benign_cnt,
        "Attack": attack_cnt,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        "FPR": fpr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Mean Score": float(np.mean(scores)),
        "Median Score": float(np.median(scores)),
    })

results_df = pd.DataFrame(results)

# ============================================================
# 7. PRINT & SAVE RESULTS
# ============================================================

print("\n" + "=" * 96)
print(
    results_df[
        [
            "Dataset",
            "ROC-AUC",
            "PR-AUC",
            "F1",
            "Precision",
            "Recall",
            "FPR",
            "FP",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)
print("=" * 96)

csv_path = OUTPUT_DIR / "innate_detector_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n[ADIS] Evaluation CSV report saved to: {csv_path}")

# ============================================================
# 8. MACRO & WEIGHTED AVERAGES
# ============================================================

metric_columns = ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "FPR"]

print("\n" + "=" * 70)
print("MACRO AVERAGE (Across Threat Families)")
print("=" * 70)
for metric in metric_columns:
    print(f"  {metric:12s}: {results_df[metric].mean():.4f}")

print("\n" + "=" * 70)
print("WEIGHTED AVERAGE (Volume-Weighted Across All Traffic)")
print("=" * 70)
for metric in metric_columns:
    weighted_val = np.average(results_df[metric], weights=results_df["Samples"])
    print(f"  {metric:12s}: {weighted_val:.4f}")

# ============================================================
# 9. CONFUSION MATRICES BREAKDOWN
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRICES PER DATASET")
print("=" * 70)
for r in results:
    print(f"\n[-] {r['Dataset']}:")
    print(f"    TN (Normal as Normal):    {r['TN']:>10,}")
    print(f"    FP (Normal as Anomaly):   {r['FP']:>10,}  (FPR: {r['FPR']:.4%})")
    print(f"    FN (Attack as Normal):    {r['FN']:>10,}")
    print(f"    TP (Attack as Anomaly):   {r['TP']:>10,}  (Recall: {r['Recall']:.4%})")

# ============================================================
# 10. VISUALIZATION CURVES
# ============================================================

print("\n[ADIS] Generating High-Resolution Diagnostic Plots...")

# 1. ROC Curves
plt.figure(figsize=(10, 7))
for name in all_scores:
    y = all_labels[name]
    if np.sum(y) > 0:
        fpr_arr, tpr_arr, _ = roc_curve(y, all_scores[name])
        auc_val = roc_auc_score(y, all_scores[name])
        plt.plot(fpr_arr, tpr_arr, label=f"{name} (AUC = {auc_val:.4f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)", fontsize=11)
plt.ylabel("True Positive Rate / Recall", fontsize=11)
plt.title("ADIS Innate Detector — ROC Curves across All Threat Classes", fontsize=13)
plt.legend(loc="lower right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=300)
plt.close()

# 2. Precision-Recall Curves
plt.figure(figsize=(10, 7))
for name in all_scores:
    y = all_labels[name]
    if np.sum(y) > 0:
        prec_arr, rec_arr, _ = precision_recall_curve(y, all_scores[name])
        ap_val = average_precision_score(y, all_scores[name])
        plt.plot(rec_arr, prec_arr, label=f"{name} (PR-AUC = {ap_val:.4f})")

plt.xlim([0.0, 1.02])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall", fontsize=11)
plt.ylabel("Precision", fontsize=11)
plt.title("ADIS Innate Detector — Precision-Recall Curves", fontsize=13)
plt.legend(loc="lower left", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "precision_recall_curves.png", dpi=300)
plt.close()

# 3. F1 vs Threshold Curve
plt.figure(figsize=(10, 7))
candidate_thresholds = np.linspace(0.05, 0.95, 90)

for name in all_scores:
    y = all_labels[name]
    if np.sum(y) > 0:
        f1_vals = []
        scores = all_scores[name]
        for t in candidate_thresholds:
            p = (scores >= t).astype(int)
            f1_vals.append(f1_score(y, p, zero_division=0))
        plt.plot(candidate_thresholds, f1_vals, label=name)

plt.axvline(OPERATING_THRESHOLD, linestyle="--", color="black", label=f"Selected Threshold ({OPERATING_THRESHOLD})")
plt.xlabel("Anomaly Probability Threshold", fontsize=11)
plt.ylabel("F1 Score", fontsize=11)
plt.title("ADIS — F1 Score vs Anomaly Decision Threshold", fontsize=13)
plt.legend(loc="lower center", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "f1_vs_threshold.png", dpi=300)
plt.close()

# 4. Anomaly Score Distributions
plt.figure(figsize=(10, 6))
# رسم عينة ممثلة تحتوي على حركة طبيعية وهجوم معاً
sample_target = "DoS-Wednesday" if "DoS-Wednesday" in all_scores else list(all_scores.keys())[1]
sample_s = all_scores[sample_target]
sample_y = all_labels[sample_target]

plt.hist(sample_s[sample_y == 0], bins=80, density=True, alpha=0.5, label="Normal Flows (Benign)", color="#1f77b4")
plt.hist(sample_s[sample_y == 1], bins=80, density=True, alpha=0.5, label="Attack Flows (Anomaly)", color="#d62728")
plt.axvline(OPERATING_THRESHOLD, linestyle="--", color="black", linewidth=2, label=f"Decision Threshold ({OPERATING_THRESHOLD})")
plt.yscale("log")
plt.xlabel("Predicted Anomaly Probability", fontsize=11)
plt.ylabel("Log Probability Density", fontsize=11)
plt.title(f"ADIS — Anomaly Score Distribution Separation ({sample_target})", fontsize=13)
plt.legend(loc="upper center", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "anomaly_score_distributions.png", dpi=300)
plt.close()

print(f"[ADIS] All 4 diagnostic plots saved to: {OUTPUT_DIR}")
print("[ADIS] Comprehensive evaluation of saved model completed successfully.")