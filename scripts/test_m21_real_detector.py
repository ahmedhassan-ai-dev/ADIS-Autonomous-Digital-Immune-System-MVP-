import pandas as pd

from src.detection.m21_detector_adapter import (
    M21IsolationForestAdapter,
)


DATA_PATH = (
    "data/raw/cicids2017/"
    "DDoS-Friday-no-metadata.parquet"
)


def main():
    print("=" * 80)
    print("[ADIS] M2.1 — Real CIC-IDS2017 Detector Test")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load real CIC-IDS2017 data
    # ---------------------------------------------------------

    df = pd.read_parquet(DATA_PATH)

    print(f"Dataset shape: {df.shape}")
    print(f"Labels: {df['Label'].value_counts().to_dict()}")

    # ---------------------------------------------------------
    # 2. Create real detector
    # ---------------------------------------------------------

    detector = M21IsolationForestAdapter()

    print()
    print(f"Detector features: {len(detector.features)}")
    print(f"Detector threshold: {detector.threshold:.10f}")

    # ---------------------------------------------------------
    # 3. Select one real DDoS flow
    # ---------------------------------------------------------

    ddos = df[
        df["Label"].astype(str).str.strip().str.lower()
        == "ddos"
    ]

    if len(ddos) == 0:
        raise RuntimeError("No DDoS samples found.")

    flow = ddos.iloc[[0]]

    print()
    print("Selected label:", flow["Label"].iloc[0])

    # ---------------------------------------------------------
    # 4. Run detector
    # ---------------------------------------------------------

    result = detector.detect(flow)

    print()
    print("Detector Result:")
    print("-" * 80)

    for key, value in result.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # 5. Basic validation
    # ---------------------------------------------------------

    assert result["feature_count"] == 59
    assert result["anomaly_score"] >= 0.0
    assert result["threshold"] > 0.0

    print()
    print("[ADIS] M2.1 Real Detector Test PASSED")


if __name__ == "__main__":
    main()