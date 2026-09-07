from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# ============================================================
# ADIS — Comprehensive Test for Merged Binary Detector
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017"
MODEL_PATH = PROJECT_ROOT / "models" / "merged_binary_detector.joblib"
RESULTS_DIR = PROJECT_ROOT / "results" / "merged_benchmark"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    medians = df.median(numeric_only=True)
    return df.fillna(medians)


def evaluate_file(file_path: Path, model, feature_cols: list):
    print(f"\n[ADIS] Loading and testing: {file_path.name}")
    start = time.perf_counter()
    df = pd.read_parquet(file_path)
    df = clean_dataframe(df)

    # استخراج الـ Labels
    if pd.api.types.is_numeric_dtype(df["Label"]):
        y_true = (df["Label"] != 0).astype(int).to_numpy()
    else:
        y_true = df["Label"].astype(str).str.strip().str.lower().ne("benign").astype(int).to_numpy()

    total_samples = len(y_true)
    attack_count = int(np.sum(y_true))
    benign_count = total_samples - attack_count

    print(f"       Rows: {total_samples:,} | Benign: {benign_count:,} | Attacks: {attack_count:,}")

    # التنبؤ
    X = df[feature_cols].copy()
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    elapsed = time.perf_counter() - start

    # حساب المقاييس
    cm = confusion_matrix(y_true, y_pred)
    
    if attack_count == 0:
        # ملف Benign-Monday فقط
        tn = cm[0][0]
        fp = cm[0][1] if len(cm[0]) > 1 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        print(f"       [Benign Pure Baseline] FP: {fp:,} / {total_samples:,} (FPR: {fpr:.4%}) | Time: {elapsed:.2f}s")
        return {
            "dataset": file_path.name.replace("-no-metadata.parquet", ""),
            "total": total_samples,
            "attacks": 0,
            "benign": total_samples,
            "roc_auc": 1.0,
            "pr_auc": 0.0,
            "f1": 1.0 if fp == 0 else (1 - fpr),
            "precision": 1.0 if fp == 0 else 0.0,
            "recall": 1.0,
            "tp": 0,
            "fp": int(fp),
            "tn": int(tn),
            "fn": 0,
            "elapsed_seconds": elapsed,
        }

    tn, fp, fn, tp = cm.ravel()
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"       ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Time: {elapsed:.2f}s")

    return {
        "dataset": file_path.name.replace("-no-metadata.parquet", ""),
        "total": total_samples,
        "attacks": attack_count,
        "benign": benign_count,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "elapsed_seconds": elapsed,
    }


def main():
    print("=" * 85)
    print("[ADIS] Multi-Attack Comprehensive Benchmark — Merged Binary Detector")
    print("=" * 85)

    print(f"[ADIS] Loading saved model from: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["features"]

    target_files = sorted(list(RAW_DIR.glob("*.parquet")))
    all_results = []

    for f in target_files:
        res = evaluate_file(f, model, feature_cols)
        all_results.append(res)

    print("\n" + "=" * 96)
    print(f"{'Attack Dataset':<25} | {'ROC-AUC':<9} | {'PR-AUC':<8} | {'F1-Score':<9} | {'Precision':<10} | {'Recall':<8} | {'FP Count':<8}")
    print("-" * 96)
    for r in all_results:
        print(
            f"{r['dataset']:<25} | {r['roc_auc']:<9.4f} | {r['pr_auc']:<8.4f} | "
            f"{r['f1']:<9.4f} | {r['precision']:<10.4f} | {r['recall']:<8.4f} | {r['fp']:<8,}"
        )
    print("=" * 96)

    # حفظ النتائج
    report_file = RESULTS_DIR / "full_data_benchmark_results.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n[ADIS] Benchmark report saved to: {report_file}")


if __name__ == "__main__":
    main()