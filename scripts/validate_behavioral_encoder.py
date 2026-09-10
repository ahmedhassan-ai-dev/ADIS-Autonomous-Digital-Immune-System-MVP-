from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017"

ENCODER_PATH = (
    PROJECT_ROOT
    / "models"
    / "behavioral_encoder.joblib"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "behavioral_encoder_validation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SAMPLES_PER_FILE = 1000

RANDOM_STATE = 42


ATTACK_FILES = {
    "Bruteforce":
        "Bruteforce-Tuesday-no-metadata.parquet",

    "DoS":
        "DoS-Wednesday-no-metadata.parquet",

    "Infiltration":
        "Infiltration-Thursday-no-metadata.parquet",

    "WebAttacks":
        "WebAttacks-Thursday-no-metadata.parquet",

    "Botnet":
        "Botnet-Friday-no-metadata.parquet",

    "Portscan":
        "Portscan-Friday-no-metadata.parquet",

    "DDoS":
        "DDoS-Friday-no-metadata.parquet",
}


def prepare_dataframe(
    df,
    features,
    medians,
    log_features,
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

    for column in features:

        X[column] = X[column].fillna(
            medians[column]
        )

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


def encode(
    df,
    features,
    medians,
    log_features,
    scaler,
    pca,
):

    X = prepare_dataframe(
        df,
        features,
        medians,
        log_features,
    )

    # IMPORTANT:
    # Keep DataFrame feature names.
    X_scaled = scaler.transform(
        X
    )

    embedding = pca.transform(
        X_scaled
    ).astype(
        np.float32
    )

    norms = np.linalg.norm(
        embedding,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1.0

    embedding /= norms

    return embedding


def cosine_matrix(A, B):

    return A @ B.T


def stats(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def main():

    print("=" * 80)
    print(
        "[ADIS] M3 — Behavioral Encoder Validation v2"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    print(
        "\n[1] Loading encoder..."
    )

    artifact = joblib.load(
        ENCODER_PATH
    )

    features = list(
        artifact["features"]
    )

    medians = artifact["medians"]

    medians = pd.Series(
        medians,
        dtype=np.float64,
    )

    log_features = list(
        artifact.get(
            "log_features",
            [],
        )
    )

    scaler = artifact["scaler"]

    pca = artifact["pca"]

    print(
        f"    Version: "
        f"{artifact.get('encoder_version')}"
    )

    print(
        f"    Features: "
        f"{len(features)}"
    )

    print(
        f"    Dimensions: "
        f"{artifact.get('embedding_dim')}"
    )

    print(
        f"    Log features: "
        f"{len(log_features)}"
    )

    print(
        f"    Whitening: "
        f"{artifact.get('whiten')}"
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print(
        "\n[2] Loading datasets..."
    )

    datasets = {}

    benign_path = (
        RAW_DIR
        / "Benign-Monday-no-metadata.parquet"
    )

    benign = pd.read_parquet(
        benign_path
    )

    benign_labels = (
        benign["Label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    benign = benign[
        benign_labels == "benign"
    ]

    benign = benign.sample(
        n=min(
            SAMPLES_PER_FILE,
            len(benign),
        ),
        random_state=RANDOM_STATE,
    )

    datasets["Benign"] = benign

    print(
        f"    Benign: {len(benign):,}"
    )

    for family, filename in ATTACK_FILES.items():

        path = RAW_DIR / filename

        df = pd.read_parquet(
            path
        )

        labels = (
            df["Label"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df = df[
            labels != "benign"
        ]

        if len(df) == 0:
            continue

        df = df.sample(
            n=min(
                SAMPLES_PER_FILE,
                len(df),
            ),
            random_state=RANDOM_STATE,
        )

        datasets[family] = df

        print(
            f"    {family}: {len(df):,}"
        )

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    print(
        "\n[3] Encoding..."
    )

    embeddings = {}

    for name, df in datasets.items():

        start = time.perf_counter()

        embeddings[name] = encode(
            df,
            features,
            medians,
            log_features,
            scaler,
            pca,
        )

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000

        print(
            f"    {name:15s}: "
            f"{embeddings[name].shape} "
            f"| {elapsed:.2f} ms"
        )

    # ========================================================
    # Same-family
    # ========================================================

    print(
        "\n[4] Same-family similarity..."
    )

    same_values = []

    for family in ATTACK_FILES:

        if family not in embeddings:
            continue

        E = embeddings[family]

        rng = np.random.default_rng(
            RANDOM_STATE
        )

        n = len(E)

        pair_count = min(
            5000,
            n * (n - 1) // 2,
        )

        i = rng.integers(
            0,
            n,
            pair_count,
        )

        j = rng.integers(
            0,
            n,
            pair_count,
        )

        mask = i != j

        sims = np.sum(
            E[i[mask]]
            * E[j[mask]],
            axis=1,
        )

        same_values.extend(
            sims.tolist()
        )

        s = stats(sims)

        print(
            f"    {family:15s}: "
            f"median={s['median']:.4f} "
            f"p05={s['p05']:.4f} "
            f"p95={s['p95']:.4f}"
        )

    # ========================================================
    # Cross-family
    # ========================================================

    print(
        "\n[5] Cross-family similarity..."
    )

    cross_values = []

    families = [
        x
        for x in ATTACK_FILES
        if x in embeddings
    ]

    cross_stats = {}

    for i in range(len(families)):

        for j in range(i + 1, len(families)):

            a = families[i]
            b = families[j]

            A = embeddings[a]
            B = embeddings[b]

            rng = np.random.default_rng(
                RANDOM_STATE
            )

            A = A[
                rng.choice(
                    len(A),
                    min(200, len(A)),
                    replace=False,
                )
            ]

            B = B[
                rng.choice(
                    len(B),
                    min(200, len(B)),
                    replace=False,
                )
            ]

            similarities = cosine_matrix(
                A,
                B,
            ).reshape(-1)

            cross_values.extend(
                similarities.tolist()
            )

            key = f"{a}_vs_{b}"

            cross_stats[key] = (
                stats(similarities)
            )

            print(
                f"    {key:35s}: "
                f"median="
                f"{np.median(similarities):.4f}"
            )

    # ========================================================
    # Attack → Benign
    # ========================================================

    print(
        "\n[6] Attack → Benign similarity..."
    )

    attack_benign_values = []

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    benign_embeddings = embeddings[
        "Benign"
    ]

    benign_subset = benign_embeddings[
        rng.choice(
            len(benign_embeddings),
            min(
                500,
                len(benign_embeddings),
            ),
            replace=False,
        )
    ]

    attack_benign_stats = {}

    for family in ATTACK_FILES:

        if family not in embeddings:
            continue

        attack = embeddings[
            family
        ]

        attack_subset = attack[
            rng.choice(
                len(attack),
                min(200, len(attack)),
                replace=False,
            )
        ]

        similarities = cosine_matrix(
            attack_subset,
            benign_subset,
        ).reshape(-1)

        attack_benign_values.extend(
            similarities.tolist()
        )

        attack_benign_stats[family] = (
            stats(similarities)
        )

        s = stats(similarities)

        print(
            f"    {family:15s}: "
            f"median={s['median']:.4f} "
            f"p95={s['p95']:.4f}"
        )

    # ========================================================
    # Threshold sweep
    # ========================================================

    print(
        "\n[7] Threshold analysis..."
    )

    same_values = np.asarray(
        same_values
    )

    cross_values = np.asarray(
        cross_values
    )

    attack_benign_values = np.asarray(
        attack_benign_values
    )

    rows = []

    for threshold in np.arange(
        0.50,
        1.001,
        0.01,
    ):

        same_recall = np.mean(
            same_values >= threshold
        )

        cross_fmr = np.mean(
            cross_values >= threshold
        )

        benign_fmr = np.mean(
            attack_benign_values >= threshold
        )

        rows.append(
            {
                "threshold":
                    float(threshold),

                "same_family_recall":
                    float(same_recall),

                "cross_family_false_match_rate":
                    float(cross_fmr),

                "attack_benign_false_match_rate":
                    float(benign_fmr),
            }
        )

    threshold_df = pd.DataFrame(
        rows
    )

    threshold_df.to_csv(
        RESULTS_DIR
        / "threshold_sweep.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Conservative threshold
    # --------------------------------------------------------

    eligible = threshold_df[
        threshold_df[
            "cross_family_false_match_rate"
        ] <= 0.01
    ]

    if len(eligible):

        best = eligible.loc[
            eligible[
                "same_family_recall"
            ].idxmax()
        ]

        recommended = float(
            best["threshold"]
        )

    else:

        recommended = None

    # ========================================================
    # Summary
    # ========================================================

    same_stats = stats(
        same_values
    )

    cross_stats_all = stats(
        cross_values
    )

    benign_stats = stats(
        attack_benign_values
    )

    print(
        "\n[8] SUMMARY"
    )

    print(
        f"    Same-family median: "
        f"{same_stats['median']:.4f}"
    )

    print(
        f"    Cross-family median: "
        f"{cross_stats_all['median']:.4f}"
    )

    print(
        f"    Attack→Benign median: "
        f"{benign_stats['median']:.4f}"
    )

    print(
        f"    Recommended threshold: "
        f"{recommended}"
    )

    report = {

        "encoder_version":
            artifact.get(
                "encoder_version"
            ),

        "feature_count":
            len(features),

        "embedding_dimension":
            artifact.get(
                "embedding_dim"
            ),

        "whiten":
            artifact.get(
                "whiten"
            ),

        "same_family":
            same_stats,

        "cross_family":
            cross_stats_all,

        "attack_to_benign":
            benign_stats,

        "recommended_threshold":
            recommended,

        "cross_family_pairs":
            cross_stats,

        "attack_to_benign_by_family":
            attack_benign_stats,
    }

    with open(
        RESULTS_DIR
        / "validation_report.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print(
        "\n[ADIS] Validation completed."
    )

    print(
        f"[ADIS] Results: "
        f"{RESULTS_DIR}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()