from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017"
MODELS_DIR = PROJECT_ROOT / "models"

OUTPUT_PATH = MODELS_DIR / "behavioral_encoder.joblib"

SAMPLES_PER_FILE = 10_000

EMBEDDING_DIM = 32

RANDOM_STATE = 42

ENCODER_VERSION = "adis-behavioral-encoder-v2"


# ============================================================
# Schema
# ============================================================

IGNORE_COLUMNS = {
    "Label",
}


def schema_hash(features):

    payload = json.dumps(
        list(features),
        separators=(",", ":"),
        ensure_ascii=True,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


# ============================================================
# Dataset cleaning
# ============================================================

def clean_dataframe(df):

    df = df.copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df


# ============================================================
# Select model features
# ============================================================

def get_feature_columns(df):

    return [
        column
        for column in df.columns
        if column not in IGNORE_COLUMNS
    ]


# ============================================================
# Load benign reference data
# ============================================================

def load_benign_reference():

    files = sorted(
        RAW_DIR.glob("*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {RAW_DIR}"
        )

    collected = []

    print("[ADIS] Loading benign reference data...")

    for file_path in files:

        print(
            f"    {file_path.name}"
        )

        df = pd.read_parquet(
            file_path
        )

        df = clean_dataframe(
            df
        )

        labels = (
            df["Label"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        benign = df[
            labels == "benign"
        ]

        if len(benign) == 0:
            print(
                "      No benign rows."
            )
            continue

        n = min(
            SAMPLES_PER_FILE,
            len(benign),
        )

        sample = benign.sample(
            n=n,
            random_state=RANDOM_STATE,
        )

        collected.append(
            sample
        )

        print(
            f"      Benign samples: {n:,}"
        )

    if not collected:
        raise RuntimeError(
            "No benign samples were collected."
        )

    reference = pd.concat(
        collected,
        axis=0,
        ignore_index=True,
    )

    return reference


# ============================================================
# Numeric conversion
# ============================================================

def prepare_numeric_matrix(
    df,
    features,
    medians=None,
):

    X = df[features].copy()

    for column in features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if medians is None:

        medians = (
            X.median(
                numeric_only=True
            )
        )

    X = X.fillna(
        medians
    )

    return X, medians


# ============================================================
# Determine skewed positive features
# ============================================================

def select_log_features(X):

    log_features = []

    for column in X.columns:

        values = X[column].to_numpy(
            dtype=np.float64
        )

        if len(values) == 0:
            continue

        if not np.all(
            np.isfinite(values)
        ):
            continue

        # Protocol / binary / very-low-cardinality
        # features should not be log transformed.
        unique_count = (
            pd.Series(values)
            .nunique()
        )

        if unique_count <= 8:
            continue

        minimum = np.min(values)

        if minimum < 0:
            continue

        series = pd.Series(values)

        skewness = series.skew()

        if (
            np.isfinite(skewness)
            and skewness > 2.0
        ):
            log_features.append(
                column
            )

    return log_features


# ============================================================
# Apply log transform
# ============================================================

def apply_log_transform(
    X,
    log_features,
):

    X = X.copy()

    for column in log_features:

        values = X[column].to_numpy(
            dtype=np.float64
        )

        values = np.maximum(
            values,
            0.0,
        )

        X[column] = np.log1p(
            values
        )

    return X


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print(
        "[ADIS] M3 — Building Behavioral Representation Encoder v2"
    )
    print("=" * 80)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load benign reference
    # --------------------------------------------------------

    reference = (
        load_benign_reference()
    )

    features = get_feature_columns(
        reference
    )

    print(
        f"\n[ADIS] Production feature space: "
        f"{len(features)} features"
    )

    # --------------------------------------------------------
    # Prepare matrix
    # --------------------------------------------------------

    X, medians = (
        prepare_numeric_matrix(
            reference,
            features,
        )
    )

    print(
        f"[ADIS] Reference matrix: "
        f"{X.shape}"
    )

    # --------------------------------------------------------
    # Log transform
    # --------------------------------------------------------

    log_features = (
        select_log_features(
            X
        )
    )

    print(
        f"\n[ADIS] Selected log-transformed features: "
        f"{len(log_features)}"
    )

    for column in log_features:

        print(
            f"    log1p -> {column}"
        )

    X_transformed = (
        apply_log_transform(
            X,
            log_features,
        )
    )

    # --------------------------------------------------------
    # Robust scaling
    # --------------------------------------------------------

    print(
        "\n[ADIS] Fitting RobustScaler..."
    )

    scaler = RobustScaler(
        quantile_range=(25.0, 75.0)
    )

    X_scaled = scaler.fit_transform(
        X_transformed
    )

    # --------------------------------------------------------
    # PCA with whitening
    # --------------------------------------------------------

    print(
        "\n[ADIS] Fitting PCA..."
    )

    pca = PCA(
        n_components=EMBEDDING_DIM,
        whiten=True,
        random_state=RANDOM_STATE,
    )

    X_embedding = pca.fit_transform(
        X_scaled
    )

    explained_variance = (
        float(
            np.sum(
                pca.explained_variance_ratio_
            )
        )
    )

    print(
        f"[ADIS] PCA explained variance "
        f"({EMBEDDING_DIM}D): "
        f"{explained_variance:.6f}"
    )

    # --------------------------------------------------------
    # L2 normalization
    # --------------------------------------------------------

    norms = np.linalg.norm(
        X_embedding,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1.0

    X_embedding = (
        X_embedding / norms
    )

    # --------------------------------------------------------
    # Embedding diagnostics
    # --------------------------------------------------------

    embedding_std = np.std(
        X_embedding,
        axis=0,
    )

    print(
        "\n[ADIS] Embedding diagnostics:"
    )

    print(
        f"    Mean std across dimensions: "
        f"{np.mean(embedding_std):.6f}"
    )

    print(
        f"    Min dimension std: "
        f"{np.min(embedding_std):.6f}"
    )

    print(
        f"    Max dimension std: "
        f"{np.max(embedding_std):.6f}"
    )

    # --------------------------------------------------------
    # Artifact
    # --------------------------------------------------------

    artifact = {

        "encoder_version":
            ENCODER_VERSION,

        "features":
            features,

        "feature_schema_hash":
            schema_hash(
                features
            ),

        "medians":
            medians.to_dict(),

        "log_features":
            log_features,

        "scaler":
            scaler,

        "pca":
            pca,

        "embedding_dim":
            EMBEDDING_DIM,

        "whiten":
            True,

        "normalization":
            "l2",

        "reference_samples":
            int(len(reference)),

        "explained_variance":
            explained_variance,

        "random_state":
            RANDOM_STATE,
    }

    joblib.dump(
        artifact,
        OUTPUT_PATH,
    )

    print(
        "\n[ADIS] Behavioral encoder saved:"
    )

    print(
        f"    {OUTPUT_PATH}"
    )

    print(
        "\n[ADIS] M3 encoder v2 completed successfully."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()