import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

from src.detection.m21_detector_adapter import (
    M21IsolationForestAdapter,
)


DATA_PATH = (
    "data/raw/cicids2017/"
    "DDoS-Friday-no-metadata.parquet"
)

SAMPLE_PER_CLASS = 5000
RANDOM_STATE = 42


def main():
    print("=" * 80)
    print("[ADIS] M2.1 — Real DDoS/Benign Benchmark")
    print("=" * 80)

    df = pd.read_parquet(DATA_PATH)

    benign = df[
        df["Label"].astype(str).str.strip().str.lower()
        == "benign"
    ]

    ddos = df[
        df["Label"].astype(str).str.strip().str.lower()
        == "ddos"
    ]

    n = min(SAMPLE_PER_CLASS, len(benign), len(ddos))

    benign = benign.sample(
        n=n,
        random_state=RANDOM_STATE,
    )

    ddos = ddos.sample(
        n=n,
        random_state=RANDOM_STATE,
    )

    test_df = pd.concat(
        [benign, ddos],
        ignore_index=True,
    ).sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    y_true = (
        test_df["Label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("ddos")
        .astype(int)
        .to_numpy()
    )

    detector = M21IsolationForestAdapter()

    # Adapter preprocessing is vectorized here by using its
    # persisted feature list and calling the underlying model.
    X = test_df[detector.features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        raise ValueError("Benchmark contains NaN/Inf values.")

    X = X.astype(np.float32)

    anomaly_scores = -detector.model.decision_function(X)
    predictions = (
        anomaly_scores >= detector.threshold
    ).astype(int)

    benign_scores = anomaly_scores[y_true == 0]
    ddos_scores = anomaly_scores[y_true == 1]

    benign_detection_rate = float(
        (benign_scores >= detector.threshold).mean()
    )

    ddos_detection_rate = float(
        (ddos_scores >= detector.threshold).mean()
    )

    roc_auc = roc_auc_score(
        y_true,
        anomaly_scores,
    )

    pr_auc = average_precision_score(
        y_true,
        anomaly_scores,
    )

    print()
    print(f"Samples per class: {n:,}")
    print(f"Total samples:     {len(test_df):,}")
    print(f"Threshold:         {detector.threshold:.10f}")

    print()
    print("Anomaly Score Statistics")
    print("-" * 80)

    print(
        f"Benign | mean={benign_scores.mean():.6f} "
        f"median={np.median(benign_scores):.6f} "
        f"max={benign_scores.max():.6f}"
    )

    print(
        f"DDoS   | mean={ddos_scores.mean():.6f} "
        f"median={np.median(ddos_scores):.6f} "
        f"max={ddos_scores.max():.6f}"
    )

    print()
    print("Detection")
    print("-" * 80)

    print(
        f"DDoS detection rate:   "
        f"{ddos_detection_rate:.4f}"
    )

    print(
        f"Benign false alarm rate: "
        f"{benign_detection_rate:.4f}"
    )

    print()
    print("Ranking Metrics")
    print("-" * 80)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    print()
    print("Prediction Counts")
    print("-" * 80)

    print(
        f"Predicted normal:  {(predictions == 0).sum():,}"
    )

    print(
        f"Predicted anomaly: {(predictions == 1).sum():,}"
    )

    print()
    print("[ADIS] M2.1 Real Benchmark Completed")


if __name__ == "__main__":
    main()