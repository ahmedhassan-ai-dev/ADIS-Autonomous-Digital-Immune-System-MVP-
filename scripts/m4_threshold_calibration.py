"""
ADIS M4 — Automatic Recognition Threshold Calibration v2

Purpose
-------
Calibrate the operational KNOWN similarity threshold for
ADIS immune memory.

Evaluation groups
-----------------
1. Known enrolled attack families
2. Benign traffic
3. Held-out novel attack family

Pipeline
--------
CICIDS2017
    |
    v
Behavioral Encoder
    |
    v
32-D L2-normalized embedding
    |
    v
Offline Immune Memory
    |
    v
Cosine similarity
    |
    v
Threshold sweep
    |
    v
Operational KNOWN threshold

This script DOES NOT modify production configuration automatically.
"""

from __future__ import annotations

import json
import math
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.analysis.threat_analyzer import (
    BehavioralThreatAnalyzer,
    ThreatAntigen,
)
from src.memory.immune_memory import ImmuneMemory


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "behavioral_encoder.joblib"
)

DATA_DIR = PROJECT_ROOT / "data"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "m4_threshold_calibration"
)

MEMORY_PATH = (
    RESULTS_DIR
    / "calibration_memory"
)


# ------------------------------------------------------------
# Enrolled attack families
# ------------------------------------------------------------

ENROLLED_FAMILIES = [
    "Bruteforce",
    "DoS",
    "WebAttacks",
    "Botnet",
    "Portscan",
    "DDoS",
]


# Held-out family
NOVEL_FAMILY = "Infiltration"


# ------------------------------------------------------------
# Sample counts
# ------------------------------------------------------------

BENIGN_SAMPLES = 1000

KNOWN_SAMPLES_PER_FAMILY = 300

NOVEL_SAMPLES = 36


# ------------------------------------------------------------
# Threshold sweep
# ------------------------------------------------------------

THRESHOLDS = [
    0.70,
    0.72,
    0.74,
    0.75,
    0.76,
    0.78,
    0.80,
    0.82,
    0.84,
    0.85,
    0.86,
    0.88,
    0.90,
    0.91,
    0.92,
    0.93,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
]


# ============================================================
# Label normalization
# ============================================================

def normalize_label(value: object) -> str:
    """
    Normalize CICIDS-style labels into broad attack families.
    """

    text = str(value).strip()

    if not text:
        return ""

    lower = text.lower()

    # Handling binary artifacts if present
    if lower in ("0", "0.0"):
        return "Benign"
    if lower in ("1", "1.0"):
        return "UNKNOWN_DROP"

    mappings = {
        # ----------------------------------------------------
        # Benign
        # ----------------------------------------------------
        "benign": "Benign",

        # ----------------------------------------------------
        # Botnet (Maps both "bot" and "botnet")
        # ----------------------------------------------------
        "bot": "Botnet",
        "botnet": "Botnet",

        # ----------------------------------------------------
        # Bruteforce
        # ----------------------------------------------------
        "bruteforce": "Bruteforce",
        "ftp-patator": "Bruteforce",
        "ssh-patator": "Bruteforce",

        # ----------------------------------------------------
        # DoS
        # ----------------------------------------------------
        "dos": "DoS",
        "dos hulk": "DoS",
        "dos goldeneye": "DoS",
        "dos slowloris": "DoS",
        "dos slowhttptest": "DoS",
        "heartbleed": "DoS",

        # ----------------------------------------------------
        # Web Attacks
        # ----------------------------------------------------
        "webattacks": "WebAttacks",
        "web attack": "WebAttacks",
        "web attack - brute force": "WebAttacks",
        "web attack – brute force": "WebAttacks",
        "web attack - xss": "WebAttacks",
        "web attack – xss": "WebAttacks",
        "web attack - sql injection": "WebAttacks",
        "web attack – sql injection": "WebAttacks",

        # ----------------------------------------------------
        # Other families
        # ----------------------------------------------------
        "portscan": "Portscan",
        "port scan": "Portscan",
        "ddos": "DDoS",
        "infiltration": "Infiltration",
    }

    if lower in mappings:
        return mappings[lower]

    # Partial matching for composite label names
    for key, mapped in mappings.items():
        if key in lower:
            return mapped

    return text


# ============================================================
# Dataset discovery
# ============================================================

def find_dataset_files() -> List[Path]:
    """
    Discover parquet datasets recursively and eliminate duplicates.
    """

    # Using set of resolved paths to avoid duplicate loading of the same physical file
    unique_files = sorted({p.resolve() for p in DATA_DIR.rglob("*.parquet")})

    if not unique_files:
        raise FileNotFoundError(
            f"No parquet files found under:\n{DATA_DIR}"
        )

    return unique_files


# ============================================================
# Dataset loading
# ============================================================

def load_all_data() -> pd.DataFrame:
    """
    Load all parquet datasets containing a Label column.
    """

    files = find_dataset_files()

    print(
        f"    Found unique parquet files: {len(files)}"
    )

    frames = []

    for path in files:

        print(
            f"    Loading: {path.name}"
        )

        df = pd.read_parquet(path)

        if "Label" not in df.columns:

            print(
                "      → skipped: no Label column"
            )

            continue

        df = df.copy()

        df["_normalized_label"] = (
            df["Label"].map(normalize_label)
        )

        # Drop unknown artifacts if any
        df = df[df["_normalized_label"] != "UNKNOWN_DROP"].copy()

        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No usable parquet dataset containing "
            "a Label column was found."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    print(
        f"\n    Combined rows: {len(combined):,}"
    )

    print("\n    Label distribution:")

    counts = (
        combined["_normalized_label"]
        .value_counts()
    )

    for label, count in counts.items():

        print(
            f"      {str(label):20s} {int(count):,}"
        )

    return combined


# ============================================================
# Deterministic sampling
# ============================================================

def sample_deterministic(
    df: pd.DataFrame,
    n: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Deterministic sampling without replacement.
    """

    if len(df) < n:

        raise ValueError(
            f"Requested {n} rows but only "
            f"{len(df)} are available."
        )

    return df.sample(
        n=n,
        random_state=random_state,
        replace=False,
    ).copy()


# ============================================================
# Build antigen
# ============================================================

def make_calibration_antigen(
    analyzer: BehavioralThreatAnalyzer,
    row: pd.Series,
    family: str,
) -> ThreatAntigen:
    """
    Convert a dataset row into a ThreatAntigen suitable
    for ImmuneMemory.commit_antigen().
    """

    feature_row = (
        row.drop(
            labels=[
                "Label",
                "_normalized_label",
            ],
            errors="ignore",
        )
    )

    embedding = analyzer.encode(
        feature_row.to_frame().T
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    metadata = {
        "behavior_family": family,
        "representation_version": (
            analyzer.encoder_version
        ),
        "feature_schema_hash": (
            analyzer.feature_schema_hash
        ),
        "embedding_dimension": (
            analyzer.embedding_dim
        ),
        "source": "m4_calibration",
    }

    return ThreatAntigen(
        antigen_id=str(uuid.uuid4()),

        technique_id="UNMAPPED",

        technique_name=(
            "Calibration behavioral family"
        ),

        tactic="UNKNOWN",

        severity="HIGH",

        embedding=embedding.tolist(),

        metadata=metadata,
    )


# ============================================================
# Build calibration immune memory
# ============================================================

def build_calibration_memory(
    analyzer: BehavioralThreatAnalyzer,
    data: pd.DataFrame,
) -> ImmuneMemory:
    """
    Build deterministic offline immune memory.

    Memory contains representative samples from all
    enrolled attack families.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Qdrant local mode expects a DIRECTORY, not a JSON file.
    # --------------------------------------------------------

    if MEMORY_PATH.exists():

        print(
            "\n    Removing previous calibration memory..."
        )

        shutil.rmtree(
            MEMORY_PATH
        )

    MEMORY_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    memory = ImmuneMemory(
        storage_path=MEMORY_PATH,

        embedding_dim=(
            analyzer.embedding_dim
        ),

        # We only use the similarity returned
        # by recognize_threat() during calibration.
        known_threshold=0.92,

        near_threshold=0.75,

        encoder_version=(
            analyzer.encoder_version
        ),

        feature_schema_hash=(
            analyzer.feature_schema_hash
        ),
    )

    print(
        "\n[3] Building calibration immune memory..."
    )

    total = 0

    for family_index, family in enumerate(
        ENROLLED_FAMILIES
    ):

        family_df = data[
            data["_normalized_label"] == family
        ]

        if family_df.empty:

            raise RuntimeError(
                f"No samples found for enrolled family: "
                f"{family}"
            )

        samples = sample_deterministic(
            family_df,
            KNOWN_SAMPLES_PER_FAMILY,
            random_state=(
                1000 + family_index
            ),
        )

        print(
            f"    {family:12s}: "
            f"{len(samples):4d} memory samples"
        )

        for _, row in samples.iterrows():

            antigen = make_calibration_antigen(
                analyzer,
                row,
                family,
            )

            memory.commit_antigen(
                antigen=antigen,
                source_dataset="CICIDS2017",
                validated=True,
            )

            total += 1

    print(
        f"\n    Total memory cells: "
        f"{memory.count_memories():,}"
    )

    print(
        f"    Expected memory cells: "
        f"{total:,}"
    )

    return memory


# ============================================================
# Similarity
# ============================================================

def get_best_similarity(
    memory: ImmuneMemory,
    embedding: np.ndarray,
) -> float:
    """
    Return maximum cosine similarity against
    compatible immune memory.
    """

    result = memory.recognize_threat(
        query_vector=embedding,
        top_k=1,
    )

    similarity = result.get(
        "similarity",
        0.0,
    )

    try:
        similarity = float(
            similarity
        )
    except Exception:

        similarity = 0.0

    if not math.isfinite(
        similarity
    ):

        similarity = 0.0

    return similarity


# ============================================================
# Collect similarity scores
# ============================================================

def collect_scores(
    analyzer: BehavioralThreatAnalyzer,
    memory: ImmuneMemory,
    data: pd.DataFrame,
) -> Tuple[
    List[float],
    List[float],
    List[float],
]:
    """
    Collect:

        known_scores
        benign_scores
        novel_scores
    """

    print(
        "\n[4] Collecting similarity scores..."
    )

    known_scores = []

    benign_scores = []

    novel_scores = []

    # ========================================================
    # KNOWN
    # ========================================================

    print(
        "\n    [Known attack families]"
    )

    for family_index, family in enumerate(
        ENROLLED_FAMILIES
    ):

        family_df = data[
            data["_normalized_label"] == family
        ]

        if len(family_df) < (
            KNOWN_SAMPLES_PER_FAMILY * 2
        ):

            raise ValueError(
                f"Not enough samples for "
                f"known evaluation family: {family}"
            )

        # IMPORTANT:
        #
        # The memory was created from one deterministic
        # sample. The evaluation samples are DIFFERENT.
        #
        # This prevents measuring trivial self-similarity.
        #

        memory_sample = sample_deterministic(
            family_df,
            KNOWN_SAMPLES_PER_FAMILY,
            random_state=(
                1000 + family_index
            ),
        )

        remaining = family_df.drop(
            index=memory_sample.index
        )

        evaluation = sample_deterministic(
            remaining,
            KNOWN_SAMPLES_PER_FAMILY,
            random_state=(
                2000 + family_index
            ),
        )

        family_scores = []

        for _, row in evaluation.iterrows():

            feature_row = row.drop(
                labels=[
                    "Label",
                    "_normalized_label",
                ],
                errors="ignore",
            )

            embedding = analyzer.encode(
                feature_row.to_frame().T
            )

            score = get_best_similarity(
                memory,
                embedding,
            )

            known_scores.append(
                score
            )

            family_scores.append(
                score
            )

        print(
            f"      {family:12s} | "
            f"n={len(family_scores):3d} | "
            f"median={np.median(family_scores):.4f} | "
            f"p05={np.percentile(family_scores, 5):.4f} | "
            f"p95={np.percentile(family_scores, 95):.4f}"
        )

    # ========================================================
    # BENIGN
    # ========================================================

    print(
        "\n    [Benign traffic]"
    )

    benign_df = data[
        data["_normalized_label"] == "Benign"
    ]

    benign_eval = sample_deterministic(
        benign_df,
        BENIGN_SAMPLES,
        random_state=3000,
    )

    for _, row in benign_eval.iterrows():

        feature_row = row.drop(
            labels=[
                "Label",
                "_normalized_label",
            ],
            errors="ignore",
        )

        embedding = analyzer.encode(
            feature_row.to_frame().T
        )

        score = get_best_similarity(
            memory,
            embedding,
        )

        benign_scores.append(
            score
        )

    print(
        f"      n={len(benign_scores)} | "
        f"median={np.median(benign_scores):.4f} | "
        f"p95={np.percentile(benign_scores, 95):.4f} | "
        f"max={np.max(benign_scores):.4f}"
    )

    # ========================================================
    # NOVEL
    # ========================================================

    print(
        f"\n    [Novel family: {NOVEL_FAMILY}]"
    )

    novel_df = data[
        data["_normalized_label"] == NOVEL_FAMILY
    ]

    novel_eval = sample_deterministic(
        novel_df,
        NOVEL_SAMPLES,
        random_state=4000,
    )

    for _, row in novel_eval.iterrows():

        feature_row = row.drop(
            labels=[
                "Label",
                "_normalized_label",
            ],
            errors="ignore",
        )

        embedding = analyzer.encode(
            feature_row.to_frame().T
        )

        score = get_best_similarity(
            memory,
            embedding,
        )

        novel_scores.append(
            score
        )

    print(
        f"      n={len(novel_scores)} | "
        f"median={np.median(novel_scores):.4f} | "
        f"p95={np.percentile(novel_scores, 95):.4f} | "
        f"max={np.max(novel_scores):.4f}"
    )

    return (
        known_scores,
        benign_scores,
        novel_scores,
    )


# ============================================================
# Threshold evaluation
# ============================================================

def evaluate_threshold(
    threshold: float,
    known_scores: List[float],
    benign_scores: List[float],
    novel_scores: List[float],
) -> Dict[str, float]:
    """
    Evaluate a single KNOWN threshold.
    """

    known = np.asarray(
        known_scores,
        dtype=float,
    )

    benign = np.asarray(
        benign_scores,
        dtype=float,
    )

    novel = np.asarray(
        novel_scores,
        dtype=float,
    )

    # --------------------------------------------------------
    # Known acceptance
    # --------------------------------------------------------

    known_accept = (
        known >= threshold
    )

    known_acceptance_rate = float(
        np.mean(known_accept)
    )

    # --------------------------------------------------------
    # Benign false match
    # --------------------------------------------------------

    benign_false_match = (
        benign >= threshold
    )

    benign_false_match_rate = float(
        np.mean(benign_false_match)
    )

    benign_specificity = (
        1.0
        - benign_false_match_rate
    )

    # --------------------------------------------------------
    # Novel rejection
    # --------------------------------------------------------

    novel_rejected = (
        novel < threshold
    )

    novel_rejection_rate = float(
        np.mean(novel_rejected)
    )

    # --------------------------------------------------------
    # Binary classification metrics
    #
    # Positive = known attack
    # Negative = benign/novel
    # --------------------------------------------------------

    tp = int(
        np.sum(known >= threshold)
    )

    fn = int(
        np.sum(known < threshold)
    )

    fp_benign = int(
        np.sum(benign >= threshold)
    )

    fp_novel = int(
        np.sum(novel >= threshold)
    )

    fp = fp_benign + fp_novel

    tn = int(
        np.sum(benign < threshold)
        +
        np.sum(novel < threshold)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    tnr = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    balanced_accuracy = (
        (recall + tnr) / 2.0
    )

    # --------------------------------------------------------
    # Security-oriented score
    #
    # We want:
    #   high known acceptance
    #   high benign specificity
    #   high novel rejection
    # --------------------------------------------------------

    security_score = (
        0.40 * known_acceptance_rate
        +
        0.30 * benign_specificity
        +
        0.30 * novel_rejection_rate
    )

    return {
        "threshold": float(threshold),

        "known_acceptance_rate": (
            known_acceptance_rate
        ),

        "benign_false_match_rate": (
            benign_false_match_rate
        ),

        "benign_specificity": (
            benign_specificity
        ),

        "novel_rejection_rate": (
            novel_rejection_rate
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "balanced_accuracy": float(
            balanced_accuracy
        ),

        "security_score": float(
            security_score
        ),

        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


# ============================================================
# Threshold recommendation
# ============================================================

def select_best_threshold(
    results: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Security-first threshold selection.

    Requirements:
        Novel rejection >= 95%
        Benign FAR <= 10%

    Among eligible thresholds:
        maximize security score
        then known acceptance
    """

    eligible = [
        result
        for result in results
        if (
            result[
                "novel_rejection_rate"
            ] >= 0.95
            and
            result[
                "benign_false_match_rate"
            ] <= 0.10
        )
    ]

    if eligible:

        return max(
            eligible,
            key=lambda x: (
                x["security_score"],
                x["known_acceptance_rate"],
            ),
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return max(
        results,
        key=lambda x: (
            x["benign_specificity"]
            +
            x["novel_rejection_rate"],
            x["known_acceptance_rate"],
        ),
    )


# ============================================================
# Save results
# ============================================================

def save_results(
    analyzer: BehavioralThreatAnalyzer,
    memory: ImmuneMemory,
    results: List[Dict[str, float]],
    best: Dict[str, float],
    known_scores: List[float],
    benign_scores: List[float],
    novel_scores: List[float],
) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = (
        RESULTS_DIR
        / "threshold_calibration.csv"
    )

    pd.DataFrame(
        results
    ).to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------------
    # Complete JSON
    # --------------------------------------------------------

    json_path = (
        RESULTS_DIR
        / "threshold_calibration.json"
    )

    payload = {
        "calibration_version":
            "adis-m4-threshold-calibration-v2",

        "encoder_version":
            analyzer.encoder_version,

        "feature_count":
            len(analyzer.features),

        "embedding_dimension":
            analyzer.embedding_dim,

        "log_feature_count":
            len(analyzer.log_features),

        "feature_schema_hash":
            analyzer.feature_schema_hash,

        "enrolled_families":
            ENROLLED_FAMILIES,

        "novel_family":
            NOVEL_FAMILY,

        "memory_cells":
            memory.count_memories(),

        "known_samples_per_family":
            KNOWN_SAMPLES_PER_FAMILY,

        "benign_samples":
            BENIGN_SAMPLES,

        "novel_samples":
            NOVEL_SAMPLES,

        "recommended_threshold":
            best["threshold"],

        "recommended_threshold_metrics":
            best,

        "thresholds":
            results,

        "score_summary": {
            "known": {
                "count": len(known_scores),
                "median": float(
                    np.median(
                        known_scores
                    )
                ),
                "p05": float(
                    np.percentile(
                        known_scores,
                        5,
                    )
                ),
                "p95": float(
                    np.percentile(
                        known_scores,
                        95,
                    )
                ),
                "min": float(
                    np.min(
                        known_scores
                    )
                ),
                "max": float(
                    np.max(
                        known_scores
                    )
                ),
            },

            "benign": {
                "count": len(benign_scores),
                "median": float(
                    np.median(
                        benign_scores
                    )
                ),
                "p95": float(
                    np.percentile(
                        benign_scores,
                        95,
                    )
                ),
                "max": float(
                    np.max(
                        benign_scores
                    )
                ),
            },

            "novel": {
                "count": len(novel_scores),
                "median": float(
                    np.median(
                        novel_scores
                    )
                ),
                "p95": float(
                    np.percentile(
                        novel_scores,
                        95,
                    )
                ),
                "max": float(
                    np.max(
                        novel_scores
                    )
                ),
            },
        },
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Small operational summary
    # --------------------------------------------------------

    summary_path = (
        RESULTS_DIR
        / "calibration_summary.json"
    )

    summary = {
        "calibration_version":
            "adis-m4-threshold-calibration-v2",

        "recommended_known_threshold":
            best["threshold"],

        "known_acceptance_rate":
            best["known_acceptance_rate"],

        "benign_false_match_rate":
            best["benign_false_match_rate"],

        "novel_rejection_rate":
            best["novel_rejection_rate"],

        "precision":
            best["precision"],

        "recall":
            best["recall"],

        "f1":
            best["f1"],

        "balanced_accuracy":
            best["balanced_accuracy"],

        "security_score":
            best["security_score"],

        "encoder_version":
            analyzer.encoder_version,

        "feature_schema_hash":
            analyzer.feature_schema_hash,

        "embedding_dimension":
            analyzer.embedding_dim,

        "memory_cells":
            memory.count_memories(),
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        "\n[8] Results saved:"
    )

    print(
        f"    {csv_path}"
    )

    print(
        f"    {json_path}"
    )

    print(
        f"    {summary_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    start_time = (
        time.perf_counter()
    )

    print("=" * 80)

    print(
        "[ADIS] M4 — Automatic Recognition "
        "Threshold Calibration v2"
    )

    print("=" * 80)

    # ========================================================
    # 1. Analyzer
    # ========================================================

    print(
        "\n[1] Loading behavioral analyzer..."
    )

    analyzer = BehavioralThreatAnalyzer(
        encoder_path=MODEL_PATH,
    )

    print(
        f"    Encoder version: "
        f"{analyzer.encoder_version}"
    )

    print(
        f"    Features: "
        f"{len(analyzer.features)}"
    )

    print(
        f"    Embedding dimension: "
        f"{analyzer.embedding_dim}"
    )

    print(
        f"    Log features: "
        f"{len(analyzer.log_features)}"
    )

    # ========================================================
    # 2. Dataset
    # ========================================================

    print(
        "\n[2] Loading datasets..."
    )

    data = load_all_data()

    # Validate required groups

    print(
        "\n    Required families:"
    )

    required = (
        ENROLLED_FAMILIES
        + [
            NOVEL_FAMILY,
            "Benign",
        ]
    )

    for family in required:

        count = int(
            np.sum(
                data["_normalized_label"]
                == family
            )
        )

        print(
            f"      {family:12s}: "
            f"{count:,}"
        )

        if count == 0:

            raise RuntimeError(
                f"Required family is missing: "
                f"{family}"
            )

    # ========================================================
    # 3. Memory
    # ========================================================

    memory = build_calibration_memory(
        analyzer,
        data,
    )

    # ========================================================
    # 4. Similarities
    # ========================================================

    known_scores, benign_scores, novel_scores = (
        collect_scores(
            analyzer,
            memory,
            data,
        )
    )

    # ========================================================
    # 5. Threshold sweep
    # ========================================================

    print(
        "\n[5] Evaluating thresholds..."
    )

    results = []

    for threshold in THRESHOLDS:

        result = evaluate_threshold(
            threshold,
            known_scores,
            benign_scores,
            novel_scores,
        )

        results.append(
            result
        )

        print(
            f"    {threshold:.2f} | "
            f"Known={result['known_acceptance_rate']:.3f} | "
            f"Benign FAR={result['benign_false_match_rate']:.3f} | "
            f"Novel Reject={result['novel_rejection_rate']:.3f} | "
            f"F1={result['f1']:.3f} | "
            f"Security={result['security_score']:.3f}"
        )

    # ========================================================
    # 6. Select threshold
    # ========================================================

    print(
        "\n[6] Selecting operational threshold..."
    )

    best = select_best_threshold(
        results
    )

    print(
        "\n    ====================================="
    )

    print(
        "    RECOMMENDED OPERATIONAL THRESHOLD"
    )

    print(
        "    ====================================="
    )

    print(
        f"      KNOWN_THRESHOLD = "
        f"{best['threshold']:.2f}"
    )

    print(
        f"      Known acceptance = "
        f"{best['known_acceptance_rate']:.3f}"
    )

    print(
        f"      Benign FAR = "
        f"{best['benign_false_match_rate']:.3f}"
    )

    print(
        f"      Novel rejection = "
        f"{best['novel_rejection_rate']:.3f}"
    )

    print(
        f"      Precision = "
        f"{best['precision']:.3f}"
    )

    print(
        f"      Recall = "
        f"{best['recall']:.3f}"
    )

    print(
        f"      F1 = "
        f"{best['f1']:.3f}"
    )

    print(
        f"      Balanced accuracy = "
        f"{best['balanced_accuracy']:.3f}"
    )

    print(
        f"      Security score = "
        f"{best['security_score']:.3f}"
    )

    # ========================================================
    # 7. Memory info
    # ========================================================

    print(
        "\n[7] Calibration memory:"
    )

    info = memory.get_collection_info()

    print(
        f"    Collection: "
        f"{info['collection']}"
    )

    print(
        f"    Memory cells: "
        f"{info['memory_count']:,}"
    )

    print(
        f"    Embedding dimension: "
        f"{info['embedding_dim']}"
    )

    print(
        f"    Distance: "
        f"{info['distance']}"
    )

    print(
        f"    Encoder version: "
        f"{info['encoder_version']}"
    )

    print(
        f"    Schema hash: "
        f"{info['feature_schema_hash']}"
    )

    # ========================================================
    # 8. Save
    # ========================================================

    save_results(
        analyzer,
        memory,
        results,
        best,
        known_scores,
        benign_scores,
        novel_scores,
    )

    # ========================================================
    # Cleanup
    # ========================================================

    memory.close()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "[ADIS] M4 threshold calibration completed."
    )

    print(
        f"    Runtime: {elapsed:.2f} seconds"
    )

    print(
        f"    Recommended threshold: "
        f"{best['threshold']:.2f}"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()