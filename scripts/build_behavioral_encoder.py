from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


# ============================================================
# ADIS — Build Behavioral Representation Encoder
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "merged_binary_detector.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "behavioral_encoder.joblib"
)

RANDOM_STATE = 42

NORMAL_SAMPLES_PER_FILE = 10000

EMBEDDING_DIM = 32


def clean_dataframe(df):

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    medians = df[numeric_cols].median()

    df[numeric_cols] = df[numeric_cols].fillna(
        medians
    )

    return df


def main():

    print("=" * 75)
    print("[ADIS] M3 — Building Behavioral Representation Encoder")
    print("=" * 75)

    # --------------------------------------------------------
    # Load production feature schema
    # --------------------------------------------------------

    artifact = joblib.load(MODEL_PATH)

    features = artifact["features"]

    print(
        f"[ADIS] Production feature space: "
        f"{len(features)} features"
    )

    # --------------------------------------------------------
    # Collect benign behavioral baseline
    # --------------------------------------------------------

    collected = []

    files = sorted(
        RAW_DIR.glob("*.parquet")
    )

    for file_path in files:

        print(
            f"[ADIS] Loading "
            f"{file_path.name}"
        )

        df = pd.read_parquet(
            file_path
        )

        df = clean_dataframe(df)

        if "Label" in df.columns:

            benign = df[
                df["Label"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("benign")
            ]

        else:
            benign = df

        n = min(
            NORMAL_SAMPLES_PER_FILE,
            len(benign),
        )

        sample = benign.sample(
            n=n,
            random_state=RANDOM_STATE,
        )

        collected.append(
            sample[features]
        )

        print(
            f"       Benign samples: "
            f"{n:,}"
        )

    X = pd.concat(
        collected,
        axis=0,
        ignore_index=True,
    )

    print()
    print(
        f"[ADIS] Encoder training matrix: "
        f"{X.shape}"
    )

    # --------------------------------------------------------
    # Robust Scaling
    # --------------------------------------------------------

    print(
        "[ADIS] Fitting RobustScaler..."
    )

    scaler = RobustScaler(
        quantile_range=(5, 95)
    )

    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    print(
        f"[ADIS] Fitting PCA "
        f"({EMBEDDING_DIM} dimensions)..."
    )

    pca = PCA(
        n_components=EMBEDDING_DIM,
        random_state=RANDOM_STATE,
    )

    pca.fit(X_scaled)

    explained_variance = (
        pca.explained_variance_ratio_.sum()
    )

    print()
    print(
        "[ADIS] PCA explained variance: "
        f"{explained_variance:.4f}"
    )

    # --------------------------------------------------------
    # Save encoder
    # --------------------------------------------------------

    artifact = {
        "scaler": scaler,
        "pca": pca,
        "features": features,
        "embedding_dim": EMBEDDING_DIM,
        "random_state": RANDOM_STATE,
    }

    joblib.dump(
        artifact,
        OUTPUT_PATH,
    )

    print()
    print(
        f"[ADIS] Behavioral encoder saved to:"
    )

    print(
        f"       {OUTPUT_PATH}"
    )

    print()
    print("=" * 75)
    print("[ADIS] M3 encoder completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
    