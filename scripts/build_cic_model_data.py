from pathlib import Path
import pandas as pd
from src.data.cic_adapter import load_and_prepare

# مسار القراءة (Raw)
RAW_DATA_DIR = Path("data") / "raw" / "cicids2017"

# مسار الحفظ (Processed)
PROCESSED_DATA_DIR = Path("data") / "processed" / "cicids2017"

DATASETS = [
    "Benign-Monday-no-metadata.parquet",
    "Bruteforce-Tuesday-no-metadata.parquet",
]


def process_and_save_dataset(filename):
    print("\n" + "=" * 70)
    print(f"[ADIS] Processing: {filename}")
    print("=" * 70)

    raw_path = RAW_DATA_DIR / filename

    # تجهيز البيانات عبر الـ Adapter
    X, y, metadata = load_and_prepare(raw_path)

    print("\n[ADIS] Processing result")
    print("-" * 70)
    print(f"Rows:                 {len(X):,}")
    print(f"Features:             {X.shape[1]:,}")
    print("\nLabel distribution:")
    print(f"  Normal:             {(y == 0).sum():,}")
    print(f"  Anomaly:            {(y == 1).sum():,}")
    print(f"  Missing values:     {X.isna().sum().sum():,}")

    # دمج الـ Features مع الـ Label إن لزم، أو حفظ الـ DataFrame المعالج
    processed_df = X.copy()
    if y is not None:
        processed_df["Label"] = y

    # التأكد من وجود مجلد processed وحفظ الملف
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / filename

    processed_df.to_parquet(output_path, index=False)
    print(f"\n[ADIS] Saved processed file to: {output_path}")

    return X, y


def main():
    print("=" * 70)
    print("[ADIS] M1.4 — CIC-IDS2017 Data Adapter & Processor")
    print("=" * 70)

    for filename in DATASETS:
        process_and_save_dataset(filename)

    print("\n" + "=" * 70)
    print("[ADIS] M1.4 completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()