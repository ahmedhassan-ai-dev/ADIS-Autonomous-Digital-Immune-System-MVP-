import os

import pandas as pd


# Features used by the anomaly detection model
FEATURE_COLUMNS = [
    "cpu_usage",
    "memory_usage",
    "network_connection",
    "destination_port",
    "failed_logins",
    "file_operations",
    "process_spawn_count",
]


def validate_telemetry(df):
    """
    Validate that the telemetry DataFrame contains
    all required columns.
    """

    required_columns = FEATURE_COLUMNS + [
        "timestamp",
        "process_name",
        "parent_process",
        "label",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return True


def extract_features(df):
    """
    Extract numerical behavioral features for ML models.

    Returns:
        pandas.DataFrame: Feature matrix
    """

    validate_telemetry(df)

    features = df[FEATURE_COLUMNS].copy()

    return features


def validate_features(features):
    """
    Validate the extracted ML feature matrix.
    """

    # Check required columns
    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in features.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    # Check for missing values
    if features.isnull().any().any():
        raise ValueError(
            "Feature matrix contains missing values."
        )

    # Check that all features are numeric
    non_numeric_columns = features.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    if non_numeric_columns:
        raise ValueError(
            f"Non-numeric feature columns found: "
            f"{non_numeric_columns}"
        )

    return True


def process_telemetry(df):
    """
    Complete telemetry preprocessing pipeline.

    Returns:
        features: ML-ready feature matrix
        labels: Labels reserved for evaluation
    """

    validate_telemetry(df)

    features = extract_features(df)
    labels = df["label"].copy()

    validate_features(features)

    return features, labels


def save_telemetry(df, path="data/raw/synthetic_telemetry.csv"):
    """
    Save raw telemetry to disk.
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    df.to_csv(path, index=False)

    print(f"[ADIS] Raw telemetry saved to: {path}")


def save_processed_data(
    features,
    labels,
    path="data/processed/telemetry_features.csv",
):
    """
    Save processed ML features and labels.

    Labels are retained for evaluation only.
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    processed_df = features.copy()
    processed_df["label"] = labels.values

    processed_df.to_csv(path, index=False)

    print(f"[ADIS] Processed telemetry saved to: {path}")


if __name__ == "__main__":
    from generator import generate_dataset

    print("[ADIS] Generating synthetic telemetry...")

    
    # 1. Generate raw telemetry
    df = generate_dataset()

    print(f"Raw dataset shape: {df.shape}")

    
    # 2. Save raw telemetry
    save_telemetry(df)


    # 3. Process telemetry
    features, labels = process_telemetry(df)

    print(f"Feature matrix shape: {features.shape}")


    # 4. Validate processed data
    print("\n[ADIS] Feature validation passed.")

    print("\nSelected features:")
    for feature in FEATURE_COLUMNS:
        print(f"- {feature}")

    print("\nFeature preview:")
    print(features.head())

    print("\nLabels:")
    print(labels.value_counts())

    
    # 5. Save processed data
    save_processed_data(features, labels)

    print("\n[ADIS] Telemetry processing completed successfully.")
