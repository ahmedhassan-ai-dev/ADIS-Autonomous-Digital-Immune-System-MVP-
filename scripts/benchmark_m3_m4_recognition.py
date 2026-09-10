# ADIS M3/M4 Recognition Benchmark
# Run from the ADIS project root:
#   python -m scripts.benchmark_m3_m4_recognition

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

SEED = 42
RNG = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "cicids2017"
MODEL_PATH = ROOT / "models" / "behavioral_encoder.joblib"
OUT_DIR = ROOT / "results" / "m3_m4_recognition_benchmark"

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
CANDIDATE_THRESHOLD = 0.75


def load_samples(name, path, n, seed):
    df = pd.read_parquet(path)
    if name == BENIGN:
        selected = df
    else:
        labels = df[LABEL_COL].astype(str).str.strip().str.lower()
        selected = df[labels != "benign"]
    n = min(n, len(selected))
    return selected.sample(n=n, random_state=seed).reset_index(drop=True)


def encode(df, artifact):
    features = artifact["features"]
    x = df[features].copy().replace([np.inf, -np.inf], np.nan)

    for c in features:
        x[c] = pd.to_numeric(x[c], errors="coerce")
        x[c] = x[c].fillna(float(artifact["medians"][c]))

    for c in artifact.get("log_features", []):
        x[c] = np.log1p(np.clip(x[c].to_numpy(dtype=np.float64), 0, None))

    z = artifact["scaler"].transform(x)
    e = artifact["pca"].transform(z)

    if artifact.get("normalization") == "l2":
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)

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


def pair_distributions(emb, families, n):
    genuine, impostor = [], []

    for _ in range(n):
        f = families[RNG.integers(len(families))]
        x = emb[f]
        i, j = RNG.choice(len(x), 2, replace=False)
        genuine.append(float(sims(x[i:i+1], x[j:j+1])[0, 0]))

    for _ in range(n):
        f1, f2 = RNG.choice(families, 2, replace=False)
        i = RNG.integers(len(emb[f1]))
        j = RNG.integers(len(emb[f2]))
        impostor.append(float(sims(
            emb[f1][i:i+1], emb[f2][j:j+1]
        )[0, 0]))

    return np.asarray(genuine), np.asarray(impostor)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("[ADIS] M3/M4 Recognition Benchmark")
    print("=" * 88)

    artifact = joblib.load(MODEL_PATH)
    print(f"[1] Encoder: {artifact.get('encoder_version')}")
    print(f"    Features: {len(artifact['features'])}")
    print(f"    Dimensions: {artifact['embedding_dim']}")
    print(f"    Log features: {len(artifact.get('log_features', []))}")

    raw, emb = {}, {}
    print("\n[2] Loading + encoding...")

    for i, (family, filename) in enumerate(FILES.items()):
        n = BENIGN_TEST if family == BENIGN else ENROLL_PER_FAMILY + TEST_PER_FAMILY
        df = load_samples(family, RAW_DIR / filename, n, SEED + i)
        raw[family] = df
        t0 = time.perf_counter()
        emb[family] = encode(df, artifact)
        print(
            f"    {family:<15} raw={len(df):>5} "
            f"embedding={emb[family].shape} "
            f"time={time.perf_counter()-t0:.3f}s"
        )

    enrolled = [f for f in FILES if f not in (BENIGN, NOVEL)]
    memory = {f: emb[f][:ENROLL_PER_FAMILY] for f in enrolled}

    print("\n[3] Offline M4 memory")
    print(f"    Enrolled families: {', '.join(enrolled)}")
    print(f"    Memory cells: {sum(len(v) for v in memory.values())}")
    print(f"    Held-out novel family: {NOVEL}")

    # Known recognition.
    rows = []
    all_known_scores = []
    all_known_correct = []
    y_true, y_pred = [], []

    print("\n[4] Family-wise KNOWN recognition")
    for family in enrolled:
        q = emb[family][ENROLL_PER_FAMILY:ENROLL_PER_FAMILY+TEST_PER_FAMILY]
        score, pred = nearest_memory(q, memory)
        correct = pred == family
        accepted = score >= CANDIDATE_THRESHOLD

        rows.append({
            "family": family,
            "samples": len(q),
            "median_similarity": float(np.median(score)),
            "p05_similarity": float(np.percentile(score, 5)),
            "p95_similarity": float(np.percentile(score, 95)),
            "nearest_family_accuracy": float(np.mean(correct)),
            "known_accept_rate_t075": float(np.mean(accepted)),
            "correct_and_accepted_t075": float(np.mean(correct & accepted)),
        })

        all_known_scores.extend(score.tolist())
        all_known_correct.extend(correct.tolist())
        y_true.extend([family] * len(q))
        y_pred.extend([
            p if s >= CANDIDATE_THRESHOLD else "UNKNOWN"
            for s, p in zip(score, pred)
        ])

        print(
            f"    {family:<15} acc={np.mean(correct):.4f} "
            f"accept@.75={np.mean(accepted):.4f} "
            f"median={np.median(score):.4f} "
            f"p05={np.percentile(score,5):.4f}"
        )

    pd.DataFrame(rows).to_csv(
        OUT_DIR / "family_wise_known_recognition.csv", index=False
    )

    # Verification threshold sweep.
    print("\n[5] Genuine vs impostor verification")
    pair_emb = {
        f: emb[f][ENROLL_PER_FAMILY:ENROLL_PER_FAMILY+TEST_PER_FAMILY]
        for f in enrolled
    }
    genuine, impostor = pair_distributions(pair_emb, enrolled, PAIR_SAMPLES)

    print(
        f"    Genuine:  median={np.median(genuine):.4f}, "
        f"p05={np.percentile(genuine,5):.4f}, "
        f"p95={np.percentile(genuine,95):.4f}"
    )
    print(
        f"    Impostor: median={np.median(impostor):.4f}, "
        f"p05={np.percentile(impostor,5):.4f}, "
        f"p95={np.percentile(impostor,95):.4f}"
    )

    sweep_rows = []
    for t in np.arange(0.0, 0.991, 0.01):
        tp = np.sum(genuine >= t)
        fn = np.sum(genuine < t)
        fp = np.sum(impostor >= t)
        tn = np.sum(impostor < t)
        tpr = tp / max(tp + fn, 1)
        frr = fn / max(tp + fn, 1)
        far = fp / max(fp + tn, 1)
        sweep_rows.append({
            "threshold": round(float(t), 2),
            "TPR": float(tpr),
            "FAR": float(far),
            "FRR": float(frr),
            "balanced_error": float((far + frr) / 2),
        })

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT_DIR / "threshold_sweep.csv", index=False)
    eer_row = sweep.iloc[(sweep["FAR"] - sweep["FRR"]).abs().argmin()]

    # Operational threshold comparison.
    print("\n[6] Operational threshold comparison")
    known_scores = np.asarray(all_known_scores)
    benign_scores, _ = nearest_memory(emb[BENIGN][:BENIGN_TEST], memory)
    novel_scores, _ = nearest_memory(
        emb[NOVEL][:min(TEST_PER_FAMILY, len(emb[NOVEL]))], memory
    )

    op = []
    for t in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        known_accept = known_scores >= t
        benign_fp = benign_scores >= t
        novel_fp = novel_scores >= t
        correct = np.asarray(all_known_correct, dtype=bool)

        row = {
            "threshold": t,
            "known_accept_rate": float(np.mean(known_accept)),
            "known_correct_and_accepted": float(np.mean(correct & known_accept)),
            "benign_false_match_rate": float(np.mean(benign_fp)),
            "novel_family_false_match_rate": float(np.mean(novel_fp)),
            "novel_family_rejection_rate": float(np.mean(~novel_fp)),
        }
        op.append(row)
        print(
            f"    t={t:.2f} | known={row['known_accept_rate']:.3f} | "
            f"correct={row['known_correct_and_accepted']:.3f} | "
            f"benign_FAR={row['benign_false_match_rate']:.3f} | "
            f"novel_reject={row['novel_family_rejection_rate']:.3f}"
        )

    pd.DataFrame(op).to_csv(
        OUT_DIR / "operational_threshold_comparison.csv", index=False
    )

    # Add benign and novel samples to open-set confusion matrix.
    benign_scores, benign_pred = nearest_memory(emb[BENIGN][:BENIGN_TEST], memory)
    y_true.extend([BENIGN] * len(benign_scores))
    y_pred.extend([
        p if s >= CANDIDATE_THRESHOLD else "UNKNOWN"
        for s, p in zip(benign_scores, benign_pred)
    ])

    novel_q = emb[NOVEL][:min(TEST_PER_FAMILY, len(emb[NOVEL]))]
    novel_scores, novel_pred = nearest_memory(novel_q, memory)
    y_true.extend([NOVEL] * len(novel_scores))
    y_pred.extend([
        p if s >= CANDIDATE_THRESHOLD else "UNKNOWN"
        for s, p in zip(novel_scores, novel_pred)
    ])

    labels = enrolled + [BENIGN, NOVEL, "UNKNOWN"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(OUT_DIR / "confusion_matrix_t075.csv")

    known_true = np.asarray(y_true)[:len(enrolled) * TEST_PER_FAMILY]
    known_pred = np.asarray(y_pred)[:len(enrolled) * TEST_PER_FAMILY]
    closed_acc = float(np.mean(known_true == known_pred))
    closed_f1 = float(f1_score(
        known_true, known_pred,
        labels=enrolled, average="macro", zero_division=0
    ))

    benign_fmr = float(np.mean(np.asarray(y_pred)[np.asarray(y_true) == BENIGN] != "UNKNOWN"))
    novel_fmr = float(np.mean(np.asarray(y_pred)[np.asarray(y_true) == NOVEL] != "UNKNOWN"))

    summary = {
        "encoder_version": artifact.get("encoder_version"),
        "embedding_dim": int(artifact["embedding_dim"]),
        "memory_cells": int(sum(len(v) for v in memory.values())),
        "enrolled_families": enrolled,
        "novel_family": NOVEL,
        "candidate_threshold": CANDIDATE_THRESHOLD,
        "closed_set_accuracy": closed_acc,
        "closed_set_macro_f1": closed_f1,
        "benign_false_match_rate": benign_fmr,
        "novel_family_false_match_rate": novel_fmr,
        "novel_family_rejection_rate": 1.0 - novel_fmr,
        "approx_eer": float(eer_row["balanced_error"]),
        "approx_eer_threshold": float(eer_row["threshold"]),
    }

    (OUT_DIR / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n[7] Confusion matrix @ threshold 0.75")
    print(cm_df.to_string())

    print("\n[8] FINAL SUMMARY")
    print(f"    Closed-set accuracy:     {closed_acc:.4f}")
    print(f"    Closed-set macro F1:     {closed_f1:.4f}")
    print(f"    Benign false-match rate: {benign_fmr:.4f}")
    print(f"    Novel rejection rate:    {1.0-novel_fmr:.4f}")
    print(
        f"    Approx EER:              {eer_row['balanced_error']:.4f} "
        f"@ threshold={eer_row['threshold']:.2f}"
    )
    print(f"\n[ADIS] Results: {OUT_DIR}")
    print("=" * 88)


if __name__ == "__main__":
    main()
