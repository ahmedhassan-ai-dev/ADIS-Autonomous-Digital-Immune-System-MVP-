from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.analysis.threat_analyzer import BehavioralThreatAnalyzer


# =============================================================================
# CONFIG
# =============================================================================

MAX_SAMPLES_PER_LABEL_PER_FILE = 300
RANDOM_STATE = 42

DATA_ROOT = Path("data")
OUTPUT_DIR = Path("results/m63_investigation_benchmark")

LABEL_CANDIDATES = [
    "Label",
    "label",
    "LABEL",
]

BENIGN_LABELS = {
    "BENIGN",
    "BENIGN_TRAFFIC",
    "0",
    "0.0",
}

# Attack family inferred from a dataset filename when the file contains
# binary labels (0/1) instead of original CIC attack names.
FILENAME_FAMILY_MAP = {
    "bruteforce": "Bruteforce",
    "ddos": "DDoS",
    "dos": "DoS",
    "botnet": "Botnet",
    "portscan": "Portscan",
    "webattacks": "WebAttacks",
    "webattack": "WebAttacks",
    "infiltration": "Infiltration",
    "heartbleed": "Heartbleed",
}

FAMILY_ORDER = [
    "Benign",
    "Bruteforce",
    "DoS",
    "DDoS",
    "Botnet",
    "Portscan",
    "WebAttacks",
    "Infiltration",
    "Heartbleed",
]


# =============================================================================
# HELPERS
# =============================================================================

def find_label_column(df: pd.DataFrame) -> str:
    for candidate in LABEL_CANDIDATES:
        if candidate in df.columns:
            return candidate

    lowered = {str(c).strip().lower(): c for c in df.columns}

    for candidate in LABEL_CANDIDATES:
        key = candidate.lower()

        if key in lowered:
            return lowered[key]

    raise ValueError(
        f"Could not find label column. Available columns: {list(df.columns)}"
    )


def normalize_text_label(value) -> str:
    if pd.isna(value):
        return "UNKNOWN"

    return str(value).strip()


def infer_family_from_filename(path: Path) -> str | None:
    name = path.stem.lower()

    for key, family in FILENAME_FAMILY_MAP.items():
        if key in name:
            return family

    return None


def normalize_ground_truth(raw_label, source_file: Path) -> str:
    """
    Convert raw CIC labels into the same family-level label space
    used by the behavioral analyzer.

    Important:
    - 0 is treated as BENIGN.
    - 1 is treated as an attack belonging to the family encoded
      by the dataset filename.
    - Original textual CIC attack labels are normalized directly.
    """

    label = normalize_text_label(raw_label)
    upper = label.upper()

    # -------------------------------------------------------------------------
    # Binary labels
    # -------------------------------------------------------------------------

    if upper in BENIGN_LABELS:
        return "Benign"

    if upper in {"1", "1.0"}:
        filename_family = infer_family_from_filename(source_file)

        if filename_family is not None:
            return filename_family

        # We do NOT invent a family when the filename gives us no evidence.
        return "UnknownAttackFamily"

    # -------------------------------------------------------------------------
    # Original CIC attack labels
    # -------------------------------------------------------------------------

    if upper == "BENIGN":
        return "Benign"

    if "FTP-PATATOR" in upper or "SSH-PATATOR" in upper:
        return "Bruteforce"

    if "PATATOR" in upper:
        return "Bruteforce"

    if "DDOS" in upper:
        return "DDoS"

    if "DOS" in upper:
        return "DoS"

    if "BOT" in upper:
        return "Botnet"

    if "PORTSCAN" in upper or "PORT SCAN" in upper:
        return "Portscan"

    if "WEB ATTACK" in upper:
        return "WebAttacks"

    if "INFILTRATION" in upper:
        return "Infiltration"

    if "HEARTBLEED" in upper:
        return "Heartbleed"

    # Fallback: preserve the original label.
    return label


def discover_datasets() -> list[Path]:
    """
    Discover only source CIC parquet files.

    We intentionally exclude:
    - data/runtime
    - benchmark outputs
    - generated runtime test datasets
    """

    datasets = []

    for path in DATA_ROOT.rglob("*.parquet"):

        normalized = str(path).replace("\\", "/").lower()

        if "/runtime/" in normalized:
            continue

        if "/results/" in normalized:
            continue

        datasets.append(path)

    return sorted(set(datasets))


def load_sampled_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    label_col = find_label_column(df)

    sampled_parts = []

    raw_labels = df[label_col].dropna().unique()

    for raw_label in raw_labels:

        subset = df[df[label_col] == raw_label]

        if len(subset) > MAX_SAMPLES_PER_LABEL_PER_FILE:
            subset = subset.sample(
                n=MAX_SAMPLES_PER_LABEL_PER_FILE,
                random_state=RANDOM_STATE,
            )

        subset = subset.copy()

        subset["_adis_ground_truth_raw"] = normalize_text_label(raw_label)

        subset["_adis_ground_truth_family"] = subset[
            label_col
        ].apply(
            lambda x: normalize_ground_truth(x, path)
        )

        subset["_adis_source_file"] = str(path)

        sampled_parts.append(subset)

    if not sampled_parts:
        return pd.DataFrame()

    return pd.concat(sampled_parts, ignore_index=True)


def remove_benchmark_metadata(flow: dict) -> dict:
    """
    Prevent benchmark ground truth from reaching the analyzer.
    """

    forbidden = {
        "_adis_ground_truth_raw",
        "_adis_ground_truth_family",
        "_adis_source_file",
    }

    return {
        key: value
        for key, value in flow.items()
        if key not in forbidden
    }


def safe_metric(func, y_true, y_pred, **kwargs) -> float:
    try:
        return float(
            func(
                y_true,
                y_pred,
                zero_division=0,
                **kwargs,
            )
        )
    except Exception:
        return 0.0


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def main():

    print("=" * 90)
    print("M6.3 v3 — FAMILY-WISE INVESTIGATION BENCHMARK")
    print("=" * 90)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Analyzer
    # -------------------------------------------------------------------------

    print("\n[1/8] Initializing BehavioralThreatAnalyzer...")

    analyzer = BehavioralThreatAnalyzer()

    print("[OK] Analyzer initialized.")

    # -------------------------------------------------------------------------
    # 2. Discover datasets
    # -------------------------------------------------------------------------

    print("\n[2/8] Discovering CIC parquet datasets...")

    datasets = discover_datasets()

    if not datasets:
        raise RuntimeError("No parquet datasets found under data/")

    print(f"[OK] Found {len(datasets)} dataset(s).")

    for path in datasets:
        print(f"     - {path}")

    # -------------------------------------------------------------------------
    # 3. Load + sample
    # -------------------------------------------------------------------------

    print("\n[3/8] Loading and sampling datasets...")

    all_parts = []

    for path in datasets:

        print(f"\n[LOAD] {path}")

        try:
            sampled = load_sampled_dataset(path)

            if sampled.empty:
                print("       [SKIP] Empty dataset.")
                continue

            all_parts.append(sampled)

            print(f"       Rows used: {len(sampled):,}")

        except Exception as exc:
            print(f"       [SKIP] Failed: {exc}")

    if not all_parts:
        raise RuntimeError("No usable datasets were loaded.")

    benchmark_df = pd.concat(
        all_parts,
        ignore_index=True,
    )

    print(
        f"\n[OK] Total sampled flows: "
        f"{len(benchmark_df):,}"
    )

    # -------------------------------------------------------------------------
    # Ground truth audit
    # -------------------------------------------------------------------------

    print("\n[GROUND TRUTH FAMILY DISTRIBUTION]")

    gt_counts = (
        benchmark_df["_adis_ground_truth_family"]
        .value_counts()
        .sort_index()
    )

    for family, count in gt_counts.items():
        print(f"  {family:<25} {count:,}")

    # -------------------------------------------------------------------------
    # 4. Behavioral analysis
    # -------------------------------------------------------------------------

    print("\n[4/8] Running metadata-free behavioral analysis...")

    y_true = []
    y_pred = []
    prediction_records = []

    start_time = time.perf_counter()

    for idx, row in benchmark_df.iterrows():

        gt_family = row["_adis_ground_truth_family"]
        source_file = row["_adis_source_file"]
        raw_label = row["_adis_ground_truth_raw"]

        flow = row.to_dict()

        # Remove benchmark-only metadata.
        flow = remove_benchmark_metadata(flow)

        try:
            result = analyzer._infer_behavior_family(flow)

            predicted_family = result.get(
                "behavior_family",
                "UNKNOWN",
            )

            confidence = result.get(
                "classification_confidence",
                None,
            )

            reason = result.get(
                "classification_reason",
                None,
            )

        except Exception as exc:

            predicted_family = "ERROR"
            confidence = None
            reason = str(exc)

        y_true.append(gt_family)
        y_pred.append(predicted_family)

        prediction_records.append(
            {
                "index": idx,
                "source_file": source_file,
                "raw_label": raw_label,
                "ground_truth_family": gt_family,
                "predicted_family": predicted_family,
                "correct": gt_family == predicted_family,
                "classification_confidence": confidence,
                "classification_reason": reason,
            }
        )

    elapsed = time.perf_counter() - start_time

    print(
        f"[OK] Analysis completed in "
        f"{elapsed:.2f} seconds."
    )

    if elapsed > 0:
        print(
            f"[INFO] Throughput: "
            f"{len(benchmark_df) / elapsed:.2f} flows/sec"
        )

    # -------------------------------------------------------------------------
    # 5. Validate label space
    # -------------------------------------------------------------------------

    print("\n[5/8] Validating benchmark label space...")

    print(
        f"[INFO] Ground-truth classes: "
        f"{sorted(set(y_true))}"
    )

    print(
        f"[INFO] Predicted classes: "
        f"{sorted(set(y_pred))}"
    )

    if set(y_true) <= {"0", "1"}:
        raise RuntimeError(
            "Ground truth is still binary-only. "
            "Family-wise benchmark cannot be trusted."
        )

    # -------------------------------------------------------------------------
    # 6. Metrics
    # -------------------------------------------------------------------------

    print("\n[6/8] Calculating metrics...")

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_precision = safe_metric(
        precision_score,
        y_true,
        y_pred,
        average="macro",
    )

    macro_recall = safe_metric(
        recall_score,
        y_true,
        y_pred,
        average="macro",
    )

    macro_f1 = safe_metric(
        f1_score,
        y_true,
        y_pred,
        average="macro",
    )

    weighted_f1 = safe_metric(
        f1_score,
        y_true,
        y_pred,
        average="weighted",
    )

    error_count = sum(
        truth != pred
        for truth, pred in zip(y_true, y_pred)
    )

    error_rate = (
        error_count / len(y_true)
        if y_true
        else 0.0
    )

    # -------------------------------------------------------------------------
    # Classification report
    # -------------------------------------------------------------------------

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    # -------------------------------------------------------------------------
    # Family-wise metrics
    # -------------------------------------------------------------------------

    family_rows = []

    for family in labels:

        family_true_count = sum(
            x == family
            for x in y_true
        )

        family_pred_count = sum(
            x == family
            for x in y_pred
        )

        correct = sum(
            truth == family and pred == family
            for truth, pred in zip(y_true, y_pred)
        )

        metrics = report.get(
            family,
            {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
            },
        )

        family_rows.append(
            {
                "family": family,
                "samples": family_true_count,
                "predicted_as_family": family_pred_count,
                "correct": correct,
                "accuracy_within_family": (
                    correct / family_true_count
                    if family_true_count
                    else 0.0
                ),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1-score"],
            }
        )

    family_df = pd.DataFrame(
        family_rows
    ).sort_values(
        "family"
    )

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    # -------------------------------------------------------------------------
    # Prediction dataframe
    # -------------------------------------------------------------------------

    predictions_df = pd.DataFrame(
        prediction_records
    )

    errors_df = predictions_df[
        ~predictions_df["correct"]
    ].copy()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = {
        "benchmark": "M6.3 v3",
        "total_samples": len(y_true),
        "total_errors": error_count,
        "error_rate": error_rate,
        "accuracy": float(accuracy),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "elapsed_seconds": elapsed,
        "throughput_flows_per_second": (
            len(y_true) / elapsed
            if elapsed > 0
            else None
        ),
        "ground_truth_classes": labels,
        "predicted_classes": sorted(
            set(y_pred)
        ),
        "dataset_count": len(datasets),
    }

    # -------------------------------------------------------------------------
    # 7. Print report
    # -------------------------------------------------------------------------

    print("\n[7/8] Benchmark Summary")

    print("-" * 90)

    print(
        f"Accuracy           : {accuracy:.4f}"
    )

    print(
        f"Macro Precision    : {macro_precision:.4f}"
    )

    print(
        f"Macro Recall       : {macro_recall:.4f}"
    )

    print(
        f"Macro F1           : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1        : {weighted_f1:.4f}"
    )

    print(
        f"Total Samples      : {len(y_true):,}"
    )

    print(
        f"Total Errors       : {error_count:,}"
    )

    print(
        f"Error Rate         : {error_rate:.4f}"
    )

    print("-" * 90)

    print("\n[FAMILY-WISE RESULTS]")

    print(
        family_df.to_string(
            index=False
        )
    )

    print("\n[TOP CONFUSIONS]")

    confusion_counter = Counter(
        (truth, pred)
        for truth, pred in zip(
            y_true,
            y_pred,
        )
        if truth != pred
    )

    for (truth, pred), count in (
        confusion_counter.most_common(20)
    ):
        print(
            f"  {truth:<25} "
            f"-> {pred:<30} : {count}"
        )

    # -------------------------------------------------------------------------
    # 8. Save artifacts
    # -------------------------------------------------------------------------

    print("\n[8/8] Saving benchmark artifacts...")

    cm_df.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    family_df.to_csv(
        OUTPUT_DIR / "family_wise_metrics.csv",
        index=False,
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False,
    )

    errors_df.to_csv(
        OUTPUT_DIR / "errors.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "benchmark_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n[OK] Saved:")

    print(
        f"  {OUTPUT_DIR / 'confusion_matrix.csv'}"
    )

    print(
        f"  {OUTPUT_DIR / 'family_wise_metrics.csv'}"
    )

    print(
        f"  {OUTPUT_DIR / 'predictions.csv'}"
    )

    print(
        f"  {OUTPUT_DIR / 'errors.csv'}"
    )

    print(
        f"  {OUTPUT_DIR / 'benchmark_summary.json'}"
    )

    print("\n" + "=" * 90)
    print("[PASS] M6.3 v3 BENCHMARK COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    main()