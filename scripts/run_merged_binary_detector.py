from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import joblib

# ============================================================
# ADIS — Merged Multi-Attack Binary Classifier (Normal vs Anomaly)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "merged_benchmark"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# عدد العينات المأخوذة من كل فئة داخل كل ملف للسرعة والتوازن
SAMPLES_PER_CLASS_PER_FILE = 25000


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    medians = df.median(numeric_only=True)
    return df.fillna(medians)


def load_merged_dataset():
    print("[ADIS] Gathering balanced slices from all CIC-IDS2017 datasets...")
    files = sorted(list(RAW_DIR.glob("*.parquet")))
    
    collected_dfs = []

    for file_path in files:
        df = pd.read_parquet(file_path)
        df = clean_dataframe(df)

        # تحويل التصنيف إلى Binary
        if pd.api.types.is_numeric_dtype(df["Label"]):
            df["Binary_Label"] = (df["Label"] != 0).astype(int)
        else:
            df["Binary_Label"] = df["Label"].astype(str).str.strip().str.lower().ne("benign").astype(int)

        # أخذ عينة ممثلة من Benign ومن Attacks داخل كل ملف
        benign_sub = df[df["Binary_Label"] == 0]
        attack_sub = df[df["Binary_Label"] == 1]

        n_benign = min(len(benign_sub), SAMPLES_PER_CLASS_PER_FILE)
        n_attack = min(len(attack_sub), SAMPLES_PER_CLASS_PER_FILE)

        sampled_benign = benign_sub.sample(n=n_benign, random_state=42) if n_benign > 0 else benign_sub
        sampled_attack = attack_sub.sample(n=n_attack, random_state=42) if n_attack > 0 else attack_sub

        merged_slice = pd.concat([sampled_benign, sampled_attack], axis=0)
        collected_dfs.append(merged_slice)
        print(f"  + {file_path.name:<40} | Benign: {len(sampled_benign):>6,} | Attacks: {len(sampled_attack):>6,}")

    full_df = pd.concat(collected_dfs, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"\n[ADIS] Final Merged Dataset Size: {len(full_df):,} flows")
    print(f"       Normal (0):  {(full_df['Binary_Label'] == 0).sum():,}")
    print(f"       Anomaly (1): {(full_df['Binary_Label'] == 1).sum():,}")

    return full_df


def main():
    print("=" * 75)
    print("[ADIS] Merged Supervised Binary Detector Training")
    print("=" * 75)

    df = load_merged_dataset()

    ignore_cols = ["Label", "Binary_Label"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feature_cols].copy()
    y = df["Binary_Label"].to_numpy()

    # تقسيم البيانات (80% تدريب و 20% اختبار لتقييم واقعي)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\n[ADIS] Training set: {X_train.shape[0]:,} flows | Test set: {X_test.shape[0]:,} flows")

    # تدريب مصنف LightGBM السريع والفعال
    print("[ADIS] Training Binary Anomaly Classifier (LightGBM)...")
    start = time.perf_counter()

    clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.08,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    print(f"[ADIS] Training completed in {elapsed:.2f} seconds.")

    # تقييم النموذج
    print("[ADIS] Evaluating model on Holdout Test Set...")
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 70)
    print("[ADIS] Overall Evaluation Results (Merged Binary Detector):")
    print("=" * 70)
    print(f"ROC-AUC Score:          {roc_auc:.4f}")
    print(f"PR-AUC (Avg Precision): {pr_auc:.4f}")
    print(f"Precision:              {prec:.4f}")
    print(f"Recall / Detection:     {rec:.4f}")
    print(f"F1-Score:               {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"Normal as Normal (TN):  {cm[0][0]:>10,}")
    print(f"Normal as Anomaly (FP): {cm[0][1]:>10,}")
    print(f"Anomaly as Normal (FN): {cm[1][0]:>10,}")
    print(f"Anomaly as Anomaly (TP):{cm[1][1]:>10,}")
    print("=" * 70)

    # حفظ النموذج
    model_path = MODELS_DIR / "merged_binary_detector.joblib"
    joblib.dump({"model": clf, "features": feature_cols}, model_path)
    print(f"[ADIS] Model saved to: {model_path}")


if __name__ == "__main__":
    main()