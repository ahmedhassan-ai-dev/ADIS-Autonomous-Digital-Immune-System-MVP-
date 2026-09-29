from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

SEED = 42
RNG = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "cicids2017"
MODEL_PATH = ROOT / "models" / "behavioral_encoder.joblib"
OUT_DIR = ROOT / "results" / "m3_m4_recognition_benchmark_v3"

LABEL_COL = "Label"
BENIGN = "Benign"
NOVEL = "Infiltration"

FILES = {
    "Benign": "Benign-Monday-no-metadata.parquet",
    "Bruteforce": "Bruteforce-Tuesday-no-metadata.parquet",
    "DoS": "DoS-Wednesday-no-metadata.parquet",
    "Infiltration": "Infiltration-Thursday-no-metadata.parquet",
    "WebAttacks": "WebAttacks-Thursday-no-metadata.parquet",
    "Botnet": "Botnet-Friday-no-metadata.parquet",
    "Portscan": "Portscan-Friday-no-metadata.parquet",
    "DDoS": "DDoS-Friday-no-metadata.parquet",
}

ENROLL_PER_FAMILY = 50
TEST_PER_FAMILY = 250
BENIGN_TEST = 1000
PAIR_SAMPLES = 10000


def row_fingerprint(row: pd.Series, features: list[str]) -> str:
    import hashlib

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


def encode(df: pd.DataFrame, artifact: dict) -> np.ndarray:
    features = artifact["features"]
    x = df[features].copy().replace([np.inf, -np.inf], np.nan)

    medians = artifact["medians"]

    for c in features:
        x[c] = pd.to_numeric(x[c], errors="coerce")
        x[c] = x[c].fillna(float(medians[c]))

    for c in artifact.get("log_features", []):
        x[c] = np.log1p(
            np.clip(
                x[c].to_numpy(dtype=np.float64),
                0,
                None,
            )
        )

    z = artifact["scaler"].transform(x)
    e = artifact["pca"].transform(z)

    e = e / np.maximum(
        np.linalg.norm(e, axis=1, keepdims=True),
        1e-12,
    )

    return e.astype(np.float32)


def sims(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


def nearest_memory(query, memory):
    best_score = np.full(len(query), -np.inf, dtype=np.float32)
    best_family = np.empty(len(query), dtype=object)

    for family, cells in memory.items():
        s = sims(query, cells).max(axis=1)
        mask = s > best_score
        best_score[mask] = s[mask]
        best_family[mask] = family

    return best_score, best_family


def load_family(name, path, n, seed, training_fingerprints, features):
    df = pd.read_parquet(path)

    if name == BENIGN:
        selected = df
    else:
        labels = df[LABEL_COL].astype(str).str.strip().str.lower()
        selected = df[labels != "benign"]

    # Remove any exact rows used during M3 training.
    if training_fingerprints:
        keep = []
        for _, row in selected.iterrows():
            keep.append(
                row_fingerprint(row, features)
                not in training_fingerprints
            )
        selected = selected.loc[np.asarray(keep, dtype=bool)]

    n = min(n, len(selected))
    return selected.sample(
        n=n,
        random_state=seed,
    ).reset_index(drop=True)


def pair_distributions(emb, families, n):
    genuine, impostor = [], []

    for _ in range(n):
        f = families[RNG.integers(len(families))]
        x = emb[f]
        if len(x) < 2:
            continue
        i, j = RNG.choice(len(x), 2, replace=False)
        genuine.append(
            float(sims(x[i:i+1], x[j:j+1])[0, 0])
        )

    for _ in range(n):
        if len(families) < 2:
            break
        f1, f2 = RNG.choice(families, 2, replace=False)
        i = RNG.integers(len(emb[f1]))
        j = RNG.integers(len(emb[f2]))
        impostor.append(
            float(
                sims(
                    emb[f1][i:i+1],
                    emb[f2][j:j+1],
                )[0, 0]
            )
        )

    return np.asarray(genuine), np.asarray(impostor)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("[ADIS] M3 v3 / M4 Recognition Benchmark — Leakage Safe")
    print("=" * 88)

    artifact = joblib.load(MODEL_PATH)

    print(f"[1] Encoder: {artifact.get('encoder_version')}")
    print(f"    Representation: {artifact.get('representation_type')}")
    print(f"    Features: {len(artifact['features'])}")
    print(f"    Dimensions: {artifact['embedding_dim']}")
    print(
        f"    Training families: "
        f"{artifact.get('training_families')}"
    )

    training_fingerprints = set(
        artifact.get("training_fingerprints", [])
    )

    raw = {}
    emb = {}

    print("\n[2] Loading + encoding leakage-safe test data...")

    for name, filename in FILES.items():
        path = RAW_DIR / filename

        n = BENIGN_TEST if name == BENIGN else TEST_PER_FAMILY

        df = load_family(
            name,
            path,
            n,
            SEED + len(raw),
            training_fingerprints,
            artifact["features"],
        )

        raw[name] = df
        emb[name] = encode(df, artifact)

        print(
            f"    {name:<15} raw={len(df):4d} "
            f"embedding={emb[name].shape}"
        )

    known_families = [
        "Bruteforce",
        "DoS",
        "WebAttacks",
        "Botnet",
        "Portscan",
        "DDoS",
    ]

    memory = {
        family: emb[family][:ENROLL_PER_FAMILY]
        for family in known_families
    }

    print("\n[3] Offline M4 memory")
    print(f"    Cells: {sum(len(v) for v in memory.values())}")
    print(f"    Novel: {NOVEL}")

    # Test only the held-out portion after enrollment.
    test_emb = {}
    for family in known_families:
        test_emb[family] = emb[family][ENROLL_PER_FAMILY:]

    test_emb[BENIGN] = emb[BENIGN]
    test_emb[NOVEL] = emb[NOVEL]

    rows = []

    print("\n[4] Recognition benchmark...")

    for family, vectors in test_emb.items():
        scores, predictions = nearest_memory(vectors, memory)

        for score, prediction in zip(scores, predictions):
            rows.append(
                {
                    "true_family": family,
                    "predicted_family": str(prediction),
                    "similarity": float(score),
                }
            )

    df = pd.DataFrame(rows)

    thresholds = [
        0.50,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.92,
        0.95,
    ]

    threshold_results = []

    for threshold in thresholds:
        known_total = 0
        known = 0
        benign_total = 0
        benign_false = 0
        novel_total = 0
        novel_rejected = 0

        for _, row in df.iterrows():
            recognized = row["similarity"] >= threshold
            family = row["true_family"]

            if family == BENIGN:
                benign_total += 1
                if recognized:
                    benign_false += 1

            elif family == NOVEL:
                novel_total += 1
                if not recognized:
                    novel_rejected += 1

            else:
                known_total += 1
                if recognized:
                    known += 1

        threshold_results.append(
            {
                "threshold": threshold,
                "known_recognition_rate": (
                    known / known_total
                    if known_total else 0.0
                ),
                "benign_far": (
                    benign_false / benign_total
                    if benign_total else 0.0
                ),
                "novel_rejection_rate": (
                    novel_rejected / novel_total
                    if novel_total else 0.0
                ),
            }
        )

    genuine, impostor = pair_distributions(
        {k: emb[k] for k in known_families},
        known_families,
        PAIR_SAMPLES,
    )

    eer_thresholds = np.linspace(0.0, 1.0, 1001)
    best = None

    for t in eer_thresholds:
        far = float(np.mean(impostor >= t))
        frr = float(np.mean(genuine < t))
        gap = abs(far - frr)

        if best is None or gap < best["gap"]:
            best = {
                "threshold": float(t),
                "far": far,
                "frr": frr,
                "gap": gap,
            }

    print("\n[5] Results")
    print(
        f"{'Threshold':<10} "
        f"{'Known Rec.':<12} "
        f"{'Benign FAR':<12} "
        f"{'Novel Reject':<12}"
    )

    for r in threshold_results:
        print(
            f"{r['threshold']:<10.2f} "
            f"{r['known_recognition_rate']:<12.4f} "
            f"{r['benign_far']:<12.4f} "
            f"{r['novel_rejection_rate']:<12.4f}"
        )

    print("\n[6] Distribution diagnostics")
    print(
        f"    Genuine median={np.median(genuine):.4f} "
        f"p05={np.percentile(genuine, 5):.4f} "
        f"p95={np.percentile(genuine, 95):.4f}"
    )
    print(
        f"    Impostor median={np.median(impostor):.4f} "
        f"p05={np.percentile(impostor, 5):.4f} "
        f"p95={np.percentile(impostor, 95):.4f}"
    )
    print(
        f"    Approx EER={best['far']:.4f} "
        f"@ threshold={best['threshold']:.3f}"
    )

    report = {
        "benchmark": "M3/M4 v3 leakage-safe",
        "encoder_version": artifact.get("encoder_version"),
        "representation_type": artifact.get("representation_type"),
        "training_families": artifact.get("training_families"),
        "novel_family": NOVEL,
        "training_rows": artifact.get("training_rows"),
        "validation_rows": artifact.get("validation_rows"),
        "test_rows": int(len(df)),
        "threshold_analysis": threshold_results,
        "genuine": {
            "median": float(np.median(genuine)),
            "p05": float(np.percentile(genuine, 5)),
            "p95": float(np.percentile(genuine, 95)),
        },
        "impostor": {
            "median": float(np.median(impostor)),
            "p05": float(np.percentile(impostor, 5)),
            "p95": float(np.percentile(impostor, 95)),
        },
        "eer": best,
        "notes": [
            "Infiltration is excluded from M3 training.",
            "Benign is a supervised training class but is never enrolled in M4 memory.",
            "Exact M3 training rows are excluded from benchmark evaluation.",
            "M4 memory mechanism is unchanged.",
        ],
    }

    path = OUT_DIR / "benchmark_v3.json"
    path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\n[OK] Report: {path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
