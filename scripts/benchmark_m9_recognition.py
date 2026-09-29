from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from src.pipeline.memory_adapter import ImmuneMemoryAdapter
from src.pipeline.encoder_adapter import BehavioralEncoderAdapter


ROOT = Path(__file__).resolve().parents[1]

THRESHOLDS = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.92,
    0.95,
]


def parse_args():
    p = argparse.ArgumentParser(
        description="M9 — Large-Scale Immune Memory Recognition Benchmark"
    )

    p.add_argument(
        "--max-per-file",
        type=int,
        default=1000,
        help="Maximum sampled flows per parquet file.",
    )

    p.add_argument(
        "--memory-path",
        default="models/immune_memory_m9_benchmark",
    )

    p.add_argument(
        "--output",
        default="data/results/m9_recognition_benchmark.json",
    )

    return p.parse_args()


def discover_datasets():
    """
    Discover benchmark datasets.

    IMPORTANT:
    Only raw CICIDS datasets are used here.
    Processed datasets may have a reduced feature schema and therefore
    are not suitable for the production behavioral encoder.
    """

    data_root = ROOT / "data"

    files = []

    for path in data_root.rglob("*.parquet"):
        relative = path.relative_to(ROOT).as_posix().lower()

        # Runtime / generated benchmark data must never enter M9.
        if "/runtime/" in relative:
            continue

        if "/results/" in relative:
            continue

        # M9 requires the full encoder feature schema.
        if "/processed/" in relative:
            continue

        files.append(path)

    return sorted(files)


def normalize_label(value):
    if pd.isna(value):
        return "UNKNOWN"

    s = str(value).strip()

    if s in {"0", "0.0", "BENIGN", "Benign", "benign"}:
        return "BENIGN"

    if s in {"1", "1.0"}:
        return "ATTACK"

    return s


def family_from_label(label):
    s = label.lower()

    if s == "benign":
        return "BENIGN"

    if "ftp-patator" in s or "ssh-patator" in s:
        return "BRUTEFORCE"

    if "bruteforce" in s:
        return "BRUTEFORCE"

    if "ddos" in s:
        return "DDOS"

    if "dos" in s:
        return "DOS"

    if "bot" in s:
        return "BOTNET"

    if "portscan" in s or "port scan" in s:
        return "PORTSCAN"

    if "web" in s:
        return "WEBATTACK"

    if "heartbleed" in s:
        return "HEARTBLEED"

    if "infiltration" in s:
        return "INFILTRATION"

    if s == "attack":
        return "ATTACK"

    return s.upper()


def infer_family_from_filename(path):
    name = path.stem.lower()

    if "infiltration" in name:
        return "INFILTRATION"

    if "botnet" in name:
        return "BOTNET"

    if "ddos" in name:
        return "DDOS"

    if "dos" in name:
        return "DOS"

    if "portscan" in name or "port_scan" in name:
        return "PORTSCAN"

    if "web" in name:
        return "WEBATTACK"

    if "heartbleed" in name:
        return "HEARTBLEED"

    if "bruteforce" in name or "patator" in name:
        return "BRUTEFORCE"

    return "UNKNOWN"


def normalize_dataset(df, path):
    label_col = None

    for col in df.columns:
        if str(col).strip().lower() in {
            "label",
            "class",
            "target",
        }:
            label_col = col
            break

    if label_col is None:
        raise ValueError(f"No label column found in {path}")

    filename_family = infer_family_from_filename(path)

    labels = []

    for value in df[label_col]:
        normalized = normalize_label(value)

        if normalized == "ATTACK" and filename_family != "UNKNOWN":
            normalized = filename_family

        labels.append(family_from_label(normalized))

    df = df.copy()
    df["_m9_family"] = labels

    return df


def get_encoder_feature_names(encoder):
    feature_names = list(
        getattr(
            encoder,
            "feature_names",
            getattr(
                encoder,
                "features",
                [],
            ),
        )
    )

    if not feature_names:
        raise RuntimeError(
            "Could not determine encoder feature names."
        )

    return feature_names


def select_features(df, encoder):
    feature_names = get_encoder_feature_names(encoder)

    missing = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Dataset missing encoder features: {missing[:10]}"
        )

    return df[feature_names].copy()


def encode_one_flow(encoder, flow):
    """
    Encode exactly one flow.

    BehavioralEncoderAdapter currently expects a single flow,
    so M9 intentionally performs row-by-row encoding.
    """

    if not isinstance(flow, pd.DataFrame):
        flow = pd.DataFrame([flow])

    if len(flow) != 1:
        raise RuntimeError(
            "encode_one_flow expects exactly one flow."
        )

    if hasattr(encoder, "transform"):
        embedding = encoder.transform(flow)

    elif hasattr(encoder, "encode"):
        embedding = encoder.encode(flow)

    else:
        raise RuntimeError(
            "Encoder does not expose transform() or encode()."
        )

    embedding = np.asarray(embedding, dtype=np.float32)

    # Adapter should return one embedding for one flow.
    if embedding.ndim == 2:
        if embedding.shape[0] != 1:
            raise RuntimeError(
                f"Expected one embedding, got shape {embedding.shape}"
            )
        embedding = embedding[0]

    if embedding.ndim != 1:
        raise RuntimeError(
            f"Expected 1D embedding, got shape {embedding.shape}"
        )

    return embedding


def encode_dataframe(encoder, X):
    """
    Encode a dataframe flow-by-flow while preserving row order.
    """

    embeddings = []

    for index in range(len(X)):
        flow = X.iloc[[index]]
        embedding = encode_one_flow(encoder, flow)
        embeddings.append(embedding)

    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)

    return np.vstack(embeddings).astype(np.float32)


def safe_close(obj):
    try:
        if hasattr(obj, "close"):
            obj.close()
    except Exception:
        pass


def main():
    args = parse_args()

    print("=" * 90)
    print("M9 — LARGE-SCALE IMMUNE MEMORY RECOGNITION BENCHMARK")
    print("=" * 90)

    memory_path = ROOT / args.memory_path

    if memory_path.exists():
        raise RuntimeError(
            f"Benchmark memory already exists:\n{memory_path}\n\n"
            "Use a fresh --memory-path."
        )

    datasets = discover_datasets()

    print()
    print(f"Datasets discovered: {len(datasets)}")

    for path in datasets:
        print(f"  - {path.relative_to(ROOT)}")

    if not datasets:
        raise RuntimeError(
            "No raw benchmark datasets were discovered."
        )

    encoder = BehavioralEncoderAdapter()

    memory = ImmuneMemoryAdapter(
        storage_path=str(memory_path)
    )

    records = []

    try:
        # ------------------------------------------------------------
        # Phase A — Load + encode
        # ------------------------------------------------------------
        print()
        print("[1/5] Loading datasets and generating embeddings...")

        for path in datasets:
            try:
                df = pd.read_parquet(path)

                if df.empty:
                    print(
                        f"  [SKIP] {path.name}: empty dataset"
                    )
                    continue

                df = normalize_dataset(df, path)

                if len(df) > args.max_per_file:
                    df = df.sample(
                        n=args.max_per_file,
                        random_state=42,
                    ).reset_index(drop=True)
                else:
                    df = df.reset_index(drop=True)

                X = select_features(df, encoder)

                print(
                    f"  Encoding {path.name} "
                    f"({len(X)} flows)..."
                )

                embeddings = encode_dataframe(
                    encoder,
                    X,
                )

                if len(embeddings) != len(df):
                    raise RuntimeError(
                        "Embedding count does not match dataframe rows."
                    )

                for idx, (_, row) in enumerate(df.iterrows()):
                    records.append(
                        {
                            "dataset": str(
                                path.relative_to(ROOT)
                            ),
                            "row": int(idx),
                            "family": row["_m9_family"],
                            "embedding": embeddings[idx],
                        }
                    )

                print(
                    f"  [OK] {path.name:<50} "
                    f"{len(df):>5} flows"
                )

            except Exception as exc:
                print(
                    f"[WARN] Skipping {path.name}: {exc}"
                )

        print()
        print(f"Total benchmark flows: {len(records)}")

        if not records:
            raise RuntimeError(
                "No benchmark records were generated."
            )

        # ------------------------------------------------------------
        # Phase B — Build enrollment memory
        # ------------------------------------------------------------
        print()
        print("[2/5] Building leakage-safe enrollment memory...")

        # IMPORTANT:
        # Only known attack families are enrolled.
        # BENIGN and INFILTRATION remain outside enrollment.
        #
        # One representative prototype per known family is used
        # for this policy benchmark.

        enrollment = {}

        for record in records:
            family = record["family"]

            if family in {
                "BENIGN",
                "INFILTRATION",
                "UNKNOWN",
            }:
                continue

            if family not in enrollment:
                enrollment[family] = record["embedding"]

        print(
            f"Known families enrolled: {len(enrollment)}"
        )

        for family in sorted(enrollment):
            result = memory.enroll_embedding(
                embedding=np.asarray(
                    enrollment[family],
                    dtype=np.float32,
                ),
                behavior_family=family,
                source_dataset="M9_BENCHMARK_ENROLLMENT",
            )

            print(
                f"  Enrolled {family:<12} "
                f"-> {result.get('status')}"
            )

        print(
            f"Memory count: {memory.count_memories()}"
        )

        # ------------------------------------------------------------
        # Phase C — Similarity evaluation
        # ------------------------------------------------------------
        print()
        print("[3/5] Running recognition benchmark...")

        rows = []

        for index, record in enumerate(records, start=1):
            family = record["family"]

            result = memory.search_candidates(
                np.asarray(
                    record["embedding"],
                    dtype=np.float32,
                )
            )

            similarity = float(
                result.get("similarity", 0.0)
            )

            similarity = max(
                -1.0,
                min(1.0, similarity),
            )

            matched_family = "NOVEL"

            candidates = result.get(
                "candidates",
                [],
            )

            if candidates:
                top = candidates[0]

                matched_family = top.get(
                    "behavior_family",
                    "NOVEL",
                )

            rows.append(
                {
                    "true_family": family,
                    "matched_family": matched_family,
                    "similarity": similarity,
                }
            )

            if index % 1000 == 0:
                print(
                    f"  Evaluated "
                    f"{index}/{len(records)} flows..."
                )

        # ------------------------------------------------------------
        # Phase D — Threshold analysis
        # ------------------------------------------------------------
        print()
        print("[4/5] Threshold analysis...")

        threshold_results = []

        for threshold in THRESHOLDS:
            known = 0
            benign_false_matches = 0
            novel_rejected = 0

            known_total = 0
            benign_total = 0
            novel_total = 0

            for row in rows:
                true_family = row["true_family"]
                similarity = row["similarity"]

                recognized = similarity >= threshold

                if true_family == "BENIGN":
                    benign_total += 1

                    if recognized:
                        benign_false_matches += 1

                elif true_family == "INFILTRATION":
                    novel_total += 1

                    if not recognized:
                        novel_rejected += 1

                elif true_family != "UNKNOWN":
                    known_total += 1

                    if recognized:
                        known += 1

            known_rate = (
                known / known_total
                if known_total
                else 0.0
            )

            benign_far = (
                benign_false_matches / benign_total
                if benign_total
                else 0.0
            )

            novel_rejection = (
                novel_rejected / novel_total
                if novel_total
                else 0.0
            )

            threshold_results.append(
                {
                    "threshold": threshold,
                    "known_recognition_rate": known_rate,
                    "benign_far": benign_far,
                    "novel_rejection_rate": novel_rejection,
                    "known_total": known_total,
                    "benign_total": benign_total,
                    "novel_total": novel_total,
                }
            )

        # ------------------------------------------------------------
        # Phase E — Summary
        # ------------------------------------------------------------
        print()
        print("[5/5] Generating report...")

        report = {
            "benchmark": "M9",
            "description": (
                "Large-scale persistent immune memory "
                "recognition benchmark"
            ),
            "total_flows": len(rows),
            "known_families_enrolled": sorted(
                enrollment.keys()
            ),
            "family_distribution": dict(
                Counter(
                    row["true_family"]
                    for row in rows
                )
            ),
            "threshold_analysis": threshold_results,
            "notes": [
                "Only raw CICIDS datasets are used.",
                "Processed datasets are excluded because they may not contain the full encoder schema.",
                "Infiltration is never enrolled.",
                "Benign flows are never enrolled.",
                "Enrollment uses one prototype per known family.",
                "Threshold metrics are evaluated without changing M4.",
                "This benchmark evaluates recognition policy, not M3 family classification.",
            ],
        }

        output_path = ROOT / args.output

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 90)
        print("M9 THRESHOLD RESULTS")
        print("=" * 90)

        print(
            f"{'Threshold':<12}"
            f"{'Known Rec.':<15}"
            f"{'Benign FAR':<15}"
            f"{'Novel Reject':<15}"
        )

        print("-" * 60)

        for result in threshold_results:
            print(
                f"{result['threshold']:<12.2f}"
                f"{result['known_recognition_rate']:<15.4f}"
                f"{result['benign_far']:<15.4f}"
                f"{result['novel_rejection_rate']:<15.4f}"
            )

        print()
        print(f"Report: {output_path}")

        print()
        print("[PASS] M9 BENCHMARK COMPLETED")

    finally:
        safe_close(memory)


if __name__ == "__main__":
    main()