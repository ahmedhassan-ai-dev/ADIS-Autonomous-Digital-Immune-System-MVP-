from pathlib import Path
import json
import time
import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# ADIS — Leave-One-Attack-Family-Out Evaluation
#
# Purpose:
#   Evaluate whether the Innate Detector generalizes to an
#   attack family that was completely excluded from training.
#
# Important:
#   This is NOT a random train/test split.
#   The held-out attack family is never used during training.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cicids2017"
MODEL_PATH = PROJECT_ROOT / "models" / "merged_binary_detector.joblib"

RESULTS_DIR = PROJECT_ROOT / "results" / "unseen_attack_evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Dataset mapping
# ------------------------------------------------------------

DATASETS = {
    "Benign-Monday": "Benign-Monday-no-metadata.parquet",
    "Botnet-Friday": "Botnet-Friday-no-metadata.parquet",
    "Bruteforce-Tuesday": "Bruteforce-Tuesday-no-metadata.parquet",
    "DDoS-Friday": "DDoS-Friday-no-metadata.parquet",
    "DoS-Wednesday": "DoS-Wednesday-no-metadata.parquet",
    "Infiltration-Thursday": "Infiltration-Thursday-no-metadata.parquet",
    "Portscan-Friday": "Portscan-Friday-no-metadata.parquet",
    "WebAttacks-Thursday": "WebAttacks-Thursday-no-metadata.parquet",
}


# ------------------------------------------------------------
# Attack family definition
# ------------------------------------------------------------

ATTACK_FAMILIES = {
    "Botnet": "Botnet-Friday",
    "Bruteforce": "Bruteforce-Tuesday",
    "DDoS": "DDoS-Friday",
    "DoS": "DoS-Wednesday",
    "Infiltration": "Infiltration-Thursday",
    "Portscan": "Portscan-Friday",
    "WebAttacks": "WebAttacks-Thursday",
}


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Maximum number of benign samples used for each experiment.
# We don't need hundreds of thousands of benign flows for this
# experiment.
MAX_BENIGN_TRAIN_PER_DATASET = 25000

# Maximum number of attack samples from the held-out family.
MAX_HELDOUT_ATTACKS = 25000

RANDOM_STATE = 42

# Decision threshold used by the saved production detector.
DEFAULT_THRESHOLD = 0.50


# ============================================================
# Utilities
# ============================================================

def load_dataset(dataset_name: str) -> pd.DataFrame:
    filename = DATASETS[dataset_name]
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    print(f"[ADIS] Loading {filename}...")
    df = pd.read_parquet(path)

    print(
        f"       Rows: {len(df):,} | "
        f"Columns: {df.shape[1]}"
    )

    return df


def convert_binary_label(df: pd.DataFrame) -> pd.Series:
    """
    Convert CIC-IDS2017 Label into:

        0 = Benign
        1 = Attack
    """

    labels = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return labels.ne("benign").astype(np.int8)


def clean_features(df: pd.DataFrame, feature_columns):
    """
    Prepare exactly the feature set expected by the saved
    LightGBM model.

    IMPORTANT:
    No fitting or statistics are computed here.
    The saved model defines the feature space.
    """

    missing = [
        c for c in feature_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Dataset is missing model features:\n"
            + "\n".join(missing)
        )

    X = df[feature_columns].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    # LightGBM supports missing values.
    # We intentionally do NOT calculate medians here.
    #
    # This prevents evaluation-time preprocessing statistics
    # from influencing the evaluation.
    return X


def evaluate_predictions(y_true, y_prob, threshold=0.50):
    y_pred = (y_prob >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(
            average_precision_score(y_true, y_prob)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        ),
    }

    return metrics


# ============================================================
# Build training data
# ============================================================

def build_training_data(
    held_out_dataset: str,
    feature_columns,
):
    """
    Build training data while completely excluding the
    held-out attack family.

    Training contains:
        - Benign traffic from all datasets
        - Attack traffic from every OTHER attack family

    The held-out attack family is NOT included.
    """

    train_parts = []

    for dataset_name in DATASETS:

        df = load_dataset(dataset_name)

        y = convert_binary_label(df)

        benign = df[y == 0]

        if len(benign) > MAX_BENIGN_TRAIN_PER_DATASET:
            benign = benign.sample(
                n=MAX_BENIGN_TRAIN_PER_DATASET,
                random_state=RANDOM_STATE,
            )

        X_benign = clean_features(
            benign,
            feature_columns,
        )

        benign_part = X_benign.copy()
        benign_part["__target__"] = 0

        train_parts.append(benign_part)

        # Do not include attacks from the held-out family.
        if dataset_name == held_out_dataset:
            print(
                f"       [HOLDOUT] Excluding all attacks "
                f"from {dataset_name}"
            )
            continue

        attacks = df[y == 1]

        if len(attacks) == 0:
            continue

        if len(attacks) > MAX_BENIGN_TRAIN_PER_DATASET:
            attacks = attacks.sample(
                n=MAX_BENIGN_TRAIN_PER_DATASET,
                random_state=RANDOM_STATE,
            )

        X_attacks = clean_features(
            attacks,
            feature_columns,
        )

        attack_part = X_attacks.copy()
        attack_part["__target__"] = 1

        train_parts.append(attack_part)

    train_df = pd.concat(
        train_parts,
        axis=0,
        ignore_index=True,
    )

    train_df = train_df.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    y_train = train_df.pop("__target__").to_numpy(
        dtype=np.int8
    )

    X_train = train_df

    return X_train, y_train


# ============================================================
# Build held-out evaluation data
# ============================================================

def build_heldout_test_data(
    held_out_dataset: str,
    feature_columns,
):
    """
    Test set:

        Benign traffic
        +
        ONLY attacks from the held-out family
    """

    df = load_dataset(held_out_dataset)

    y = convert_binary_label(df)

    benign = df[y == 0]
    attacks = df[y == 1]

    if len(benign) > MAX_BENIGN_TRAIN_PER_DATASET:
        benign = benign.sample(
            n=MAX_BENIGN_TRAIN_PER_DATASET,
            random_state=RANDOM_STATE,
        )

    if len(attacks) > MAX_HELDOUT_ATTACKS:
        attacks = attacks.sample(
            n=MAX_HELDOUT_ATTACKS,
            random_state=RANDOM_STATE,
        )

    X_benign = clean_features(
        benign,
        feature_columns,
    )

    X_attack = clean_features(
        attacks,
        feature_columns,
    )

    X_test = pd.concat(
        [X_benign, X_attack],
        axis=0,
        ignore_index=True,
    )

    y_test = np.concatenate(
        [
            np.zeros(len(X_benign), dtype=np.int8),
            np.ones(len(X_attack), dtype=np.int8),
        ]
    )

    shuffled_indices = np.random.default_rng(
        RANDOM_STATE
    ).permutation(len(X_test))

    X_test = X_test.iloc[
        shuffled_indices
    ].reset_index(drop=True)

    y_test = y_test[shuffled_indices]

    return X_test, y_test


# ============================================================
# Main experiment
# ============================================================

def main():

    print("=" * 90)
    print("[ADIS] LEAVE-ONE-ATTACK-FAMILY-OUT EVALUATION")
    print("=" * 90)

    # --------------------------------------------------------
    # Load saved detector
    # --------------------------------------------------------

    print("\n[ADIS] Loading saved production detector...")

    artifact = joblib.load(MODEL_PATH)

    if isinstance(artifact, dict):
        model = artifact["model"]
        feature_columns = artifact["features"]
    else:
        model = artifact
        feature_columns = list(model.feature_name_)

    print(
        f"[ADIS] Model loaded successfully."
    )

    print(
        f"[ADIS] Expected model features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We CANNOT use the saved model directly for the
    # leave-one-family-out experiment.
    #
    # The saved model has already seen all families.
    #
    # Therefore we train a fresh LightGBM model for each
    # experiment.
    # --------------------------------------------------------

    print(
        "\n[ADIS] NOTE:"
        "\n       The saved production model is used only"
        "\n       as the feature-space reference."
        "\n       A fresh LightGBM model will be trained for"
        "\n       every held-out attack family."
    )

    results = []

    for family_name, held_out_dataset in ATTACK_FAMILIES.items():

        print("\n" + "=" * 90)
        print(
            f"[ADIS] HOLDING OUT ATTACK FAMILY: "
            f"{family_name}"
        )
        print(
            f"[ADIS] Held-out dataset: "
            f"{held_out_dataset}"
        )
        print("=" * 90)

        # ----------------------------------------------------
        # Build train data
        # ----------------------------------------------------

        print("\n[1/4] Building training data...")

        X_train, y_train = build_training_data(
            held_out_dataset,
            feature_columns,
        )

        print(
            f"[ADIS] Training shape: "
            f"{X_train.shape}"
        )

        print(
            f"[ADIS] Training normal: "
            f"{(y_train == 0).sum():,}"
        )

        print(
            f"[ADIS] Training attacks: "
            f"{(y_train == 1).sum():,}"
        )

        # ----------------------------------------------------
        # Build held-out test data
        # ----------------------------------------------------

        print("\n[2/4] Building held-out test set...")

        X_test, y_test = build_heldout_test_data(
            held_out_dataset,
            feature_columns,
        )

        print(
            f"[ADIS] Test shape: "
            f"{X_test.shape}"
        )

        print(
            f"[ADIS] Test normal: "
            f"{(y_test == 0).sum():,}"
        )

        print(
            f"[ADIS] Test held-out attacks: "
            f"{(y_test == 1).sum():,}"
        )

        # ----------------------------------------------------
        # Train fresh detector
        # ----------------------------------------------------

        print("\n[3/4] Training fresh LightGBM detector...")

        start = time.perf_counter()

        clf = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.08,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
            verbosity=-1,
        )

        clf.fit(
            X_train,
            y_train,
        )

        training_time = time.perf_counter() - start

        print(
            f"[ADIS] Training completed in "
            f"{training_time:.2f} seconds."
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        print("\n[4/4] Evaluating unseen attack family...")

        y_prob = clf.predict_proba(X_test)[:, 1]

        metrics = evaluate_predictions(
            y_test,
            y_prob,
            threshold=DEFAULT_THRESHOLD,
        )

        metrics["attack_family"] = family_name
        metrics["held_out_dataset"] = held_out_dataset
        metrics["training_rows"] = len(X_train)
        metrics["test_rows"] = len(X_test)
        metrics["training_time_seconds"] = training_time

        results.append(metrics)

        print("\n" + "-" * 75)
        print(f"RESULTS — {family_name}")
        print("-" * 75)

        print(
            f"ROC-AUC   : {metrics['roc_auc']:.4f}"
        )
        print(
            f"PR-AUC    : {metrics['pr_auc']:.4f}"
        )
        print(
            f"Precision : {metrics['precision']:.4f}"
        )
        print(
            f"Recall    : {metrics['recall']:.4f}"
        )
        print(
            f"F1        : {metrics['f1']:.4f}"
        )
        print(
            f"FPR       : {metrics['fpr']:.4f}"
        )

        print("\nConfusion Matrix:")
        print(
            f"TN: {metrics['tn']:,}"
        )
        print(
            f"FP: {metrics['fp']:,}"
        )
        print(
            f"FN: {metrics['fn']:,}"
        )
        print(
            f"TP: {metrics['tp']:,}"
        )

    # ========================================================
    # Save results
    # ========================================================

    results_df = pd.DataFrame(results)

    columns = [
        "attack_family",
        "held_out_dataset",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "fpr",
        "tn",
        "fp",
        "fn",
        "tp",
        "training_rows",
        "test_rows",
        "training_time_seconds",
    ]

    results_df = results_df[columns]

    csv_path = (
        RESULTS_DIR
        / "leave_one_attack_family_out.csv"
    )

    json_path = (
        RESULTS_DIR
        / "leave_one_attack_family_out.json"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 90)
    print("[ADIS] UNSEEN ATTACK FAMILY SUMMARY")
    print("=" * 90)

    print(
        results_df[
            [
                "attack_family",
                "roc_auc",
                "pr_auc",
                "f1",
                "precision",
                "recall",
                "fpr",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "-" * 90)

    print(
        f"Mean ROC-AUC : "
        f"{results_df['roc_auc'].mean():.4f}"
    )

    print(
        f"Mean PR-AUC  : "
        f"{results_df['pr_auc'].mean():.4f}"
    )

    print(
        f"Mean F1      : "
        f"{results_df['f1'].mean():.4f}"
    )

    print(
        f"Mean Recall  : "
        f"{results_df['recall'].mean():.4f}"
    )

    print(
        f"Mean FPR     : "
        f"{results_df['fpr'].mean():.4f}"
    )

    print("\n[ADIS] Results saved to:")
    print(f"       {csv_path}")
    print(f"       {json_path}")

    print("\n" + "=" * 90)
    print(
        "[ADIS] Leave-one-attack-family-out evaluation completed."
    )
    print("=" * 90)


if __name__ == "__main__":
    main()