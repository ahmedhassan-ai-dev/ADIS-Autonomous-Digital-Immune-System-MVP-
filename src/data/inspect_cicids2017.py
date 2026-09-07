from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

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


# ============================================================
# Helpers
# ============================================================

def inspect_dataset(path: Path):
    print("\n" + "=" * 80)
    print(f"DATASET: {path.name}")
    print("=" * 80)

    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return None

    # Load one dataset at a time to avoid unnecessary RAM usage
    df = pd.read_parquet(path)

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print("\n[1] BASIC INFORMATION")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")
    print(f"Memory  : {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    print("\n[2] COLUMNS")

    for i, column in enumerate(df.columns, start=1):
        print(f"{i:3}. {column}")

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    print("\n[3] DATA TYPES")

    dtype_summary = (
        df.dtypes
        .value_counts()
        .rename_axis("dtype")
        .reset_index(name="count")
    )

    print(dtype_summary.to_string(index=False))

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\n[4] MISSING VALUES")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.to_string())

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    print("\n[5] INFINITE VALUES")

    numeric_columns = df.select_dtypes(include=np.number).columns

    if len(numeric_columns) > 0:
        inf_counts = np.isinf(df[numeric_columns]).sum()
        inf_counts = inf_counts[inf_counts > 0].sort_values(ascending=False)

        if inf_counts.empty:
            print("No infinite values found.")
        else:
            print(inf_counts.to_string())
    else:
        print("No numeric columns found.")

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------

    print("\n[6] DUPLICATES")

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_count:,}")

    if len(df) > 0:
        duplicate_percentage = (duplicate_count / len(df)) * 100
        print(f"Duplicate %   : {duplicate_percentage:.2f}%")

    # --------------------------------------------------------
    # Label information
    # --------------------------------------------------------

    print("\n[7] LABEL INFORMATION")

    label_columns = [
        column for column in df.columns
        if column.strip().lower() == "label"
    ]

    if label_columns:
        label_column = label_columns[0]

        print(f"Label column: {label_column}")
        print(f"Unique labels: {df[label_column].nunique()}")

        print("\nLabel distribution:")
        print(
            df[label_column]
            .value_counts(dropna=False)
            .to_string()
        )

        print("\nLabel percentage:")
        print(
            (df[label_column].value_counts(normalize=True, dropna=False) * 100)
            .round(2)
            .astype(str)
            .add("%")
            .to_string()
        )

    else:
        print("WARNING: No Label column found.")

    # --------------------------------------------------------
    # Constant columns
    # --------------------------------------------------------

    print("\n[8] CONSTANT COLUMNS")

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        for column in constant_columns:
            print(f"- {column}")
    else:
        print("No constant columns found.")

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    print("\n[9] SAMPLE")

    print(df.head(3).to_string())

    return {
        "file": path.name,
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(
            df.memory_usage(deep=True).sum() / (1024 ** 2), 2
        ),
        "duplicate_rows": int(duplicate_count),
        "missing_cells": int(df.isna().sum().sum()),
        "infinite_cells": (
            int(np.isinf(df[numeric_columns]).sum().sum())
            if len(numeric_columns) > 0
            else 0
        ),
        "constant_columns": constant_columns,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("ADIS — CIC-IDS2017 DATASET INSPECTOR")
    print("=" * 80)

    if not DATA_DIR.exists():
        print(f"\n[ERROR] Dataset directory not found:")
        print(DATA_DIR)
        return

    results = []

    for dataset_name in DATASETS:
        dataset_path = DATA_DIR / dataset_name

        result = inspect_dataset(dataset_path)

        if result is not None:
            results.append(result)

    # ========================================================
    # Final Summary
    # ========================================================

    print("\n\n" + "=" * 80)
    print("FINAL DATASET SUMMARY")
    print("=" * 80)

    if not results:
        print("No datasets were inspected.")
        return

    summary = pd.DataFrame(results)

    print(
        summary[
            [
                "file",
                "rows",
                "columns",
                "memory_mb",
                "duplicate_rows",
                "missing_cells",
                "infinite_cells",
            ]
        ].to_string(index=False)
    )

    print("\n[ADIS] Dataset inspection completed successfully.")


if __name__ == "__main__":
    main()