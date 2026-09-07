from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# ADIS — CIC-IDS2017 Data Adapter
# ============================================================

LABEL_COLUMN = "Label"

# Features that were constant in the CIC-IDS2017 audit
CONSTANT_FEATURES = {
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
}


# Behavioral network-flow features selected for the first ADIS benchmark.
#
# We intentionally avoid:
# - Flow ID
# - Source IP
# - Destination IP
# - Source Port
# - Destination Port
# - Timestamp
#
# The cleaned dataset supplied for this benchmark already removed
# those metadata fields.
BEHAVIORAL_FEATURES = [
    # Flow intensity
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",

    # Packet volume
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",
    "Bwd Packets Length Total",

    # Packet characteristics
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Packet Length Min",
    "Packet Length Max",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",

    # Inter-arrival timing
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",

    # TCP behavior
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "ECE Flag Count",

    # Direction / segment behavior
    "Down/Up Ratio",
    "Avg Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",

    # Subflow behavior
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",

    # TCP window / activity behavior
    "Init Fwd Win Bytes",
    "Init Bwd Win Bytes",
    "Fwd Act Data Packets",
    "Fwd Seg Size Min",

    # Active / idle behavior
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]


def load_parquet(path):
    """
    Load one CIC-IDS2017 parquet file.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"[ADIS] Dataset not found: {path}"
        )

    if path.suffix.lower() != ".parquet":
        raise ValueError(
            f"[ADIS] Expected a parquet file, got: {path.suffix}"
        )

    print(f"[ADIS] Loading: {path.name}")

    df = pd.read_parquet(path)

    print(
        f"[ADIS] Loaded {len(df):,} rows × {len(df.columns)} columns"
    )

    return df


def validate_dataset(df):
    """
    Validate the basic structure of a CIC-IDS2017 dataframe.
    """

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"[ADIS] Missing required label column: {LABEL_COLUMN}"
        )

    missing_features = [
        feature
        for feature in BEHAVIORAL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "[ADIS] Missing behavioral features:\n"
            + "\n".join(f"  - {feature}" for feature in missing_features)
        )

    if df.empty:
        raise ValueError("[ADIS] Dataset is empty.")

    print("[ADIS] Dataset validation passed.")


def remove_constant_features(X):
    """
    Remove features known to be constant in the CIC audit.
    """

    existing_constants = [
        feature
        for feature in CONSTANT_FEATURES
        if feature in X.columns
    ]

    if existing_constants:
        X = X.drop(columns=existing_constants)

    return X, existing_constants


def clean_numeric_values(X):
    """
    Handle invalid numeric values.

    Inf/-Inf are converted to NaN.

    Missing values are intentionally NOT imputed here.
    Imputation must be fitted using training data only to avoid leakage.
    """

    X = X.copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    return X


def extract_labels(df):
    """
    Convert CIC labels into binary ADIS anomaly labels.

    0 = normal / benign
    1 = anomalous / attack
    """

    labels = (
        df[LABEL_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    y = (labels != "benign").astype(np.int8)

    return y


def prepare_features(df):
    """
    Extract and prepare behavioral features.

    Returns:
        X:
            Numeric behavioral feature matrix.
        removed_constants:
            List of removed constant features.
    """

    validate_dataset(df)

    X = df[BEHAVIORAL_FEATURES].copy()

    # Make sure every selected feature is numeric.
    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    X = clean_numeric_values(X)

    X, removed_constants = remove_constant_features(X)

    return X, removed_constants


def prepare_dataset(df):
    """
    Complete CIC-IDS2017 preprocessing.

    Returns:
        X: behavioral features
        y: binary anomaly labels
        metadata: preprocessing information
    """

    X, removed_constants = prepare_features(df)

    y = extract_labels(df)

    metadata = {
        "original_rows": len(df),
        "original_columns": len(df.columns),
        "selected_features": list(X.columns),
        "removed_constant_features": removed_constants,
        "missing_values": int(X.isna().sum().sum()),
        "infinite_values": 0,
        "normal_samples": int((y == 0).sum()),
        "anomaly_samples": int((y == 1).sum()),
    }

    return X, y, metadata


def load_and_prepare(path):
    """
    Convenience function:
        parquet → dataframe → X/y/metadata
    """

    df = load_parquet(path)

    X, y, metadata = prepare_dataset(df)

    return X, y, metadata


if __name__ == "__main__":

    print("=" * 70)
    print("[ADIS] CIC-IDS2017 Adapter Test")
    print("=" * 70)

    dataset_path = (
        Path("data")
        / "raw"
        / "cic-ids2017"
        / "Benign-Monday-no-metadata.parquet"
    )

    X, y, metadata = load_and_prepare(dataset_path)

    print("\n" + "=" * 70)
    print("[ADIS] Adapter Summary")
    print("=" * 70)

    print(f"Rows:                {len(X):,}")
    print(f"Features:            {X.shape[1]}")
    print(f"Normal samples:      {(y == 0).sum():,}")
    print(f"Anomaly samples:     {(y == 1).sum():,}")
    print(f"Missing values:      {X.isna().sum().sum():,}")

    print("\nRemoved constant features:")

    if metadata["removed_constant_features"]:
        for feature in metadata["removed_constant_features"]:
            print(f"  - {feature}")
    else:
        print("  None")

    print("\nSelected features:")

    for feature in X.columns:
        print(f"  - {feature}")

    print("\nFeature preview:")
    print(X.head())

    print("\n[ADIS] Adapter test completed successfully.")