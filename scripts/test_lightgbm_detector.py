import pandas as pd

from src.detection.lightgbm_detector import LightGBMDetector


DATA_PATH = (
    "data/raw/cicids2017/"
    "Bruteforce-Tuesday-no-metadata.parquet"
)


def main() -> None:
    print("=" * 70)
    print("ADIS — LightGBM Detector Test")
    print("=" * 70)

    detector = LightGBMDetector()

    print("\n[1] Detector Info")
    print(detector.info())

    df = pd.read_parquet(DATA_PATH)

    print("\n[2] Dataset")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    sample = df.iloc[:10].copy()

    print("\n[3] Running Detection")

    results = detector.detect(sample)

    for i, result in enumerate(results):
        print(
            f"Flow {i}: "
            f"score={result['anomaly_score']:.6f} | "
            f"decision={result['decision']}"
        )

    print("\n[4] Assertions")

    assert detector.n_features == 77
    assert len(results) == 10

    for result in results:
        assert 0.0 <= result["anomaly_score"] <= 1.0
        assert result["decision"] in {"BENIGN", "ATTACK"}

    print("✓ Feature count = 77")
    print("✓ Prediction count = 10")
    print("✓ Scores are within [0, 1]")
    print("✓ Decisions are valid")

    print("\n" + "=" * 70)
    print("LIGHTGBM DETECTOR TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()