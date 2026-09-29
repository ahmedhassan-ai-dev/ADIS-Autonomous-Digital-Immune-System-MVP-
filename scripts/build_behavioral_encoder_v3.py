from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.analysis.supervised_projector import SupervisedBehavioralProjector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017"
MODELS_DIR = PROJECT_ROOT / "models"

OUTPUT_PATH = MODELS_DIR / "behavioral_encoder.joblib"
BACKUP_PATH = MODELS_DIR / "behavioral_encoder_v2_backup.joblib"

SAMPLES_PER_FAMILY = 10000
TEST_SIZE = 0.20
RANDOM_STATE = 42

EMBEDDING_DIM = 32
ENCODER_VERSION = "adis-behavioral-encoder-v3-supervised"

KNOWN_FAMILIES = {
    "benign": "Benign-Monday-no-metadata.parquet",
    "bruteforce": "Bruteforce-Tuesday-no-metadata.parquet",
    "dos": "DoS-Wednesday-no-metadata.parquet",
    "webattacks": "WebAttacks-Thursday-no-metadata.parquet",
    "botnet": "Botnet-Friday-no-metadata.parquet",
    "portscan": "Portscan-Friday-no-metadata.parquet",
    "ddos": "DDoS-Friday-no-metadata.parquet",
}

# Novel family is deliberately excluded from M3 training.
NOVEL_FAMILY = "Infiltration-Thursday-no-metadata.parquet"

IGNORE_COLUMNS = {"Label"}


def schema_hash(features: list[str]) -> str:
    payload = json.dumps(
        list(features),
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def row_fingerprint(row: pd.Series, features: list[str]) -> str:
    payload = []
    for f in features:
        value = row[f]
        if pd.isna(value):
            payload.append(None)
        elif isinstance(value, (np.floating, float)):
            payload.append(float(value))
        elif isinstance(value, (np.integer, int)):
            payload.append(int(value))
        else:
            payload.append(str(value))
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def clean_numeric(df: pd.DataFrame, features: list[str], medians=None):
    x = df[features].copy().replace([np.inf, -np.inf], np.nan)

    for c in features:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    if medians is None:
        medians = x.median(numeric_only=True)

    x = x.fillna(medians)
    return x, medians


def select_log_features(x: pd.DataFrame) -> list[str]:
    result = []

    for c in x.columns:
        values = x[c].to_numpy(dtype=np.float64)

        if len(values) == 0 or not np.all(np.isfinite(values)):
            continue

        if pd.Series(values).nunique() <= 8:
            continue

        if np.min(values) < 0:
            continue

        skewness = pd.Series(values).skew()
        if np.isfinite(skewness) and skewness > 2.0:
            result.append(c)

    return result


def apply_log(x: pd.DataFrame, log_features: list[str]) -> pd.DataFrame:
    x = x.copy()
    for c in log_features:
        values = np.clip(
            x[c].to_numpy(dtype=np.float64),
            0.0,
            None,
        )
        x[c] = np.log1p(values)
    return x


def load_family(name: str, filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path).replace([np.inf, -np.inf], np.nan)

    labels = df["Label"].astype(str).str.strip().str.lower()

    if name == "benign":
        selected = df[labels == "benign"].copy()
    else:
        selected = df[labels != "benign"].copy()

    n = min(SAMPLES_PER_FAMILY, len(selected))
    return selected.sample(
        n=n,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)


def main() -> None:
    print("=" * 88)
    print("[ADIS] M3 v3 — Supervised Behavioral Representation Encoder")
    print("=" * 88)
    print("[INFO] Infiltration is excluded from training.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve the known-good v2 artifact before replacing it.
    if OUTPUT_PATH.exists():
        if not BACKUP_PATH.exists():
            BACKUP_PATH.write_bytes(OUTPUT_PATH.read_bytes())
            print(f"[OK] v2 backup: {BACKUP_PATH}")

    frames = []
    labels = []
    fingerprints_source = {}

    print("\n[1/7] Loading training families...")
    for family, filename in KNOWN_FAMILIES.items():
        df = load_family(family, filename)
        frames.append(df)

        y = np.full(len(df), family, dtype=object)
        labels.append(y)

        print(f"  {family:<12} {len(df):>6} rows")

    data = pd.concat(frames, ignore_index=True)
    y = np.concatenate(labels)

    features = [
        c for c in data.columns
        if c not in IGNORE_COLUMNS
    ]

    print(f"\n[INFO] Features: {len(features)}")
    print(f"[INFO] Training rows: {len(data)}")

    # One fixed stratified split. Validation is never used for fitting.
    train_idx, val_idx = train_test_split(
        np.arange(len(data)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = data.iloc[train_idx].reset_index(drop=True)
    val_df = data.iloc[val_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_val = y[val_idx]

    print(
        f"[2/7] Leakage-safe split: "
        f"train={len(train_df)}, validation={len(val_df)}"
    )

    # Fit preprocessing ONLY on training data.
    x_train, medians = clean_numeric(
        train_df,
        features,
    )

    x_val, _ = clean_numeric(
        val_df,
        features,
        medians=medians,
    )

    log_features = select_log_features(x_train)

    x_train = apply_log(x_train, log_features)
    x_val = apply_log(x_val, log_features)

    print(f"[3/7] Log features: {len(log_features)}")

    scaler = RobustScaler(
        quantile_range=(25.0, 75.0),
    )

    z_train = scaler.fit_transform(x_train)
    z_val = scaler.transform(x_val)

    # Supervised projector.
    print("[4/7] Fitting supervised projector...")

    projector = SupervisedBehavioralProjector(
        pca_components=26,
        lda_components=6,
        lda_weight=2.0,
        random_state=RANDOM_STATE,
    )

    projector.fit(z_train, y_train)

    e_train = projector.transform(z_train)
    e_val = projector.transform(z_val)

    # L2 normalize exactly as the production encoder contract expects.
    def l2(e):
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        return e / np.maximum(norms, 1e-12)

    e_train = l2(e_train)
    e_val = l2(e_val)

    print(f"[5/7] Embedding shape: {e_train.shape}")

    train_std = np.std(e_train, axis=0)
    val_std = np.std(e_val, axis=0)

    print(
        f"[INFO] Train mean/std: "
        f"{np.mean(e_train):.6f} / {np.mean(train_std):.6f}"
    )
    print(
        f"[INFO] Validation mean/std: "
        f"{np.mean(e_val):.6f} / {np.mean(val_std):.6f}"
    )

    # Store row fingerprints of training records for future leakage-safe
    # benchmark filtering.
    print("[6/7] Building leakage fingerprints...")

    training_fingerprints = set()
    for _, row in train_df.iterrows():
        training_fingerprints.add(
            row_fingerprint(row, features)
        )

    # The production adapter expects artifact["pca"].transform(z).
    # The custom projector therefore deliberately occupies the pca slot.
    artifact = {
        "encoder_version": ENCODER_VERSION,
        "features": features,
        "feature_schema_hash": schema_hash(features),
        "medians": medians.to_dict(),
        "log_features": log_features,
        "scaler": scaler,
        "pca": projector,
        "embedding_dim": EMBEDDING_DIM,
        "whiten": True,
        "normalization": "l2",

        # New v3 metadata.
        "representation_type": "supervised_lda_plus_pca",
        "projector": "SupervisedBehavioralProjector",
        "lda_components": projector.lda_components,
        "pca_components": projector.pca_components,
        "lda_weight": projector.lda_weight,
        "training_families": sorted(KNOWN_FAMILIES.keys()),
        "novel_family_excluded": "Infiltration",
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "training_fingerprints": sorted(training_fingerprints),
        "random_state": RANDOM_STATE,

        # Useful diagnostics.
        "train_embedding_std_mean": float(np.mean(train_std)),
        "validation_embedding_std_mean": float(np.mean(val_std)),
    }

    # Save a dedicated v3 artifact first.
    v3_path = MODELS_DIR / "behavioral_encoder_v3.joblib"
    joblib.dump(artifact, v3_path)

    print(f"[OK] v3 artifact saved: {v3_path}")

    # Replace production artifact only after successful serialization.
    joblib.dump(artifact, OUTPUT_PATH)

    print(f"[OK] Production artifact replaced: {OUTPUT_PATH}")
    print("\n[7/7] M3 v3 completed successfully.")
    print("=" * 88)


if __name__ == "__main__":
    main()
