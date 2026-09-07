from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
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
# ADIS — M2.1 Feature Selection & Anomaly Detection
# Target: Filter top 12 features for Brute Force (Patator)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "cicids2017"
DATA_FILE = DATA_DIR / "Bruteforce-Tuesday-no-metadata.parquet"


def log(message=""):
    print(f"[ADIS] {message}")


def clean_dataframe(df):
    """Clean infinite and missing values."""
    df = df.replace([np.inf, -np.inf], np.nan)
    medians = df.median(numeric_only=True)
    df = df.fillna(medians)
    return df


def select_top_features(X_sample, y_sample, k=12):
    """Select top K features using Mutual Information."""
    log(f"Computing Mutual Information on sample ({len(X_sample):,} rows)...")
    start = time.perf_counter()

    mi_scores = mutual_info_classif(
        X_sample,
        y_sample,
        discrete_features='auto',
        random_state=42
    )

    mi_series = pd.Series(mi_scores, index=X_sample.columns).sort_values(ascending=False)
    selected_features = mi_series.head(k).index.tolist()

    elapsed = time.perf_counter() - start
    log(f"Feature ranking completed in {elapsed:.2f} seconds")

    print("\n" + "=" * 60)
    print(f"[ADIS] Top {k} Features Selected by Importance:")
    print("=" * 60)
    for rank, (feat, score) in enumerate(mi_series.head(k).items(), 1):
        print(f"{rank:2d}. {feat:<35} | Score: {score:.5f}")
    print("=" * 60 + "\n")

    return selected_features


def prepare_data(df, selected_features=None, train_benign_ratio=0.75):
    """Split into Train (Benign only) and Test (Benign + Attacks)."""
    # Label extraction
    if pd.api.types.is_numeric_dtype(df["Label"]):
        labels = (df["Label"] != 0).astype(int)
    else:
        labels = df["Label"].astype(str).str.strip().str.lower().ne("benign").astype(int)

    features = selected_features if selected_features else [c for c in df.columns if c != "Label"]

    benign_df = df[labels == 0]
    attack_df = df[labels == 1]

    train_benign, test_benign = train_test_split(
        benign_df,
        train_size=train_benign_ratio,
        random_state=42,
        shuffle=True,
    )

    test_combined = pd.concat([test_benign, attack_df], axis=0).sample(
        frac=1.0, random_state=42
    ).reset_index(drop=True)

    X_train = train_benign[features].copy()
    X_test = test_combined[features].copy()

    if pd.api.types.is_numeric_dtype(test_combined["Label"]):
        y_test = (test_combined["Label"] != 0).astype(int).to_numpy()
    else:
        y_test = test_combined["Label"].astype(str).str.strip().str.lower().ne("benign").astype(int).to_numpy()

    return X_train, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate Isolation Forest and perform F1-threshold tuning."""
    raw_scores = model.decision_function(X_test)
    anomaly_scores = -raw_scores

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

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "threshold": best_threshold,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "cm": cm,
        "y_pred": y_pred,
    }


def main():
    print("=" * 70)
    print("[ADIS] M2.1 — Feature Selection & Anomaly Detection Pipeline")
    print("=" * 70)

    log("Loading dataset...")
    df = pd.read_parquet(DATA_FILE)
    df = clean_dataframe(df)

    feature_cols = [c for c in df.columns if c != "Label"]

    # 1. Feature Selection on stratified sample (50,000 rows for speed and accuracy)
    sample_df = df.groupby("Label", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 25000), random_state=42)
    )
    X_sample = sample_df[feature_cols]
    if pd.api.types.is_numeric_dtype(sample_df["Label"]):
        y_sample = (sample_df["Label"] != 0).astype(int)
    else:
        y_sample = sample_df["Label"].astype(str).str.strip().str.lower().ne("benign").astype(int)

    top_12_features = select_top_features(X_sample, y_sample, k=12)

    # 2. Prepare Data with Top 12 Features only
    X_train, X_test, y_test = prepare_data(df, selected_features=top_12_features)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    log(f"Training on Benign data with {len(top_12_features)} features: {X_train.shape}")
    log(f"Testing on Mixed data: {X_test.shape}")

    # 3. Train Isolation Forest
    log("Training Isolation Forest on selected features...")
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

    # 4. Evaluation
    results = evaluate_model(model, X_test, y_test)
    cm = results["cm"]

    print("\n" + "=" * 70)
    print(f"[ADIS] Evaluation Results with Top 12 Features:")
    print("=" * 70)
    print(f"ROC-AUC Score:         {results['roc_auc']:.4f}")
    print(f"PR-AUC (Avg Prec):     {results['pr_auc']:.4f}")
    print(f"Optimized F1 Score:    {results['best_f1']:.4f}")
    print(f"Precision:             {results['precision']:.4f}")
    print(f"Recall / Detection:    {results['recall']:.4f}")
    print(f"Optimal Threshold:     {results['threshold']:.6f}")
    print("\nConfusion Matrix:")
    print("                 Pred Normal    Pred Anomaly")
    print(f"Actual Normal    {cm[0][0]:>12,}    {cm[0][1]:>12,}")
    print(f"Actual Anomaly   {cm[1][0]:>12,}    {cm[1][1]:>12,}")
    print("=" * 70)


if __name__ == "__main__":
    main()