from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw/cicids2017")

DATASETS = [
    "Benign-Monday-no-metadata.parquet",
    "Bruteforce-Tuesday-no-metadata.parquet",
    "DoS-Wednesday-no-metadata.parquet",
    "Infiltration-Thursday-no-metadata.parquet",
    "WebAttacks-Thursday-no-metadata.parquet",
    "Botnet-Friday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet",
    "DDoS-Friday-no-metadata.parquet",
]


def audit_dataset(file_path: Path):
    print("\n" + "=" * 70)
    print(f"[ADIS] Auditing: {file_path.name}")
    print("=" * 70)

    df = pd.read_parquet(file_path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    if "Label" in df.columns:
        print("\nLabel distribution:")

        label_counts = df["Label"].value_counts(dropna=False)

        for label, count in label_counts.items():
            percentage = count / len(df) * 100
            print(f"  {str(label):30s} {count:10,} ({percentage:6.2f}%)")

    else:
        print("\n[WARNING] Label column not found.")

    # ---------------------------------------------------------
    # Data types
    # ---------------------------------------------------------

    print("\nData types:")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    non_numeric_columns = [
        col for col in df.columns
        if col not in numeric_columns
    ]

    print(f"Numeric columns:     {len(numeric_columns)}")
    print(f"Non-numeric columns: {len(non_numeric_columns)}")

    if non_numeric_columns:
        print("\nNon-numeric columns:")
        for col in non_numeric_columns:
            print(f"  - {col}")

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    missing_count = int(df.isna().sum().sum())

    print(f"\nMissing cells: {missing_count:,}")

    # ---------------------------------------------------------
    # Infinity
    # ---------------------------------------------------------

    numeric_df = df[numeric_columns]

    infinity_count = int(
        numeric_df.isin([float("inf"), float("-inf")]).sum().sum()
    )

    print(f"Infinity cells: {infinity_count:,}")

    if infinity_count > 0:
        print("\nColumns containing infinity:")

        for col in numeric_columns:
            count = int(
                df[col].isin([float("inf"), float("-inf")]).sum()
            )

            if count > 0:
                print(f"  {col}: {count:,}")

    # ---------------------------------------------------------
    # Constant columns
    # ---------------------------------------------------------

    constant_columns = []

    for col in numeric_columns:
        if df[col].nunique(dropna=False) <= 1:
            constant_columns.append(col)

    print(f"\nConstant numeric columns: {len(constant_columns)}")

    if constant_columns:
        for col in constant_columns:
            print(f"  - {col}")

    # ---------------------------------------------------------
    # Duplicate rows
    # ---------------------------------------------------------

    duplicate_count = int(df.duplicated().sum())

    print(f"\nDuplicate rows: {duplicate_count:,}")

    # ---------------------------------------------------------
    # Basic statistics
    # ---------------------------------------------------------

    print("\nSample statistics:")

    preview_columns = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Packet Length Mean",
        "SYN Flag Count",
        "ACK Flag Count",
    ]

    available_preview = [
        col for col in preview_columns
        if col in df.columns
    ]

    if available_preview:
        print(df[available_preview].describe().T)

    return {
        "file": file_path.name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": missing_count,
        "infinity_cells": infinity_count,
        "constant_columns": len(constant_columns),
        "duplicate_rows": duplicate_count,
    }


def main():
    print("[ADIS] CIC-IDS2017 Dataset Audit")
    print("[ADIS] Starting...\n")

    results = []

    for filename in DATASETS:

        file_path = DATA_DIR / filename

        if not file_path.exists():
            print(f"[WARNING] File not found: {file_path}")
            continue

        result = audit_dataset(file_path)
        results.append(result)

    if not results:
        print("\n[ERROR] No datasets were found.")
        print(f"Expected directory: {DATA_DIR}")
        return

    # ---------------------------------------------------------
    # Save audit summary
    # ---------------------------------------------------------

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_df = pd.DataFrame(results)

    output_path = output_dir / "cic_dataset_audit.csv"

    audit_df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("[ADIS] FINAL AUDIT SUMMARY")
    print("=" * 70)

    print(audit_df.to_string(index=False))

    print(f"\n[ADIS] Audit saved to: {output_path}")


if __name__ == "__main__":
    main()