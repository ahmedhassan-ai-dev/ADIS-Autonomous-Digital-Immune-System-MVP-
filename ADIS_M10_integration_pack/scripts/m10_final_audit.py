from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import sklearn
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "behavioral_encoder.joblib"
M10 = ROOT / "results" / "m10_final_demo.json"
M3 = ROOT / "results" / "m3_m4_recognition_benchmark_v3" / "benchmark_v3.json"

print("=" * 88)
print("ADIS M10 — FINAL SYSTEM AUDIT")
print("=" * 88)

failures = []

def require(path, label):
    ok = path.exists()
    print(f"{label:<32}: {'OK' if ok else 'MISSING'}")
    if not ok:
        failures.append(label)
    return ok

require(MODEL, "Production encoder")
require(ROOT / "models" / "merged_binary_detector.joblib", "Binary detector")
require(ROOT / "src" / "pipeline" / "memory_adapter.py", "Memory adapter")
require(ROOT / "src" / "pipeline" / "m4_policy.py", "M4 policy")
require(ROOT / "src" / "sandbox" / "isolation_chamber.py", "Isolation sandbox")
require(M3, "M3/M4 v3 benchmark")
require(M10, "M10 demo result")

print("\n[ENVIRONMENT]")
print("Python :", sys.version.split()[0])
print("NumPy  :", np.__version__)
print("Pandas :", pd.__version__)
print("sklearn:", sklearn.__version__)
print("joblib :", joblib.__version__)

if MODEL.exists():
    artifact = joblib.load(MODEL)
    print("\n[ENCODER]")
    print("version :", artifact.get("encoder_version", artifact.get("version")))
    print("features:", len(artifact.get("features", [])))
    print("dim     :", artifact.get("embedding_dim", getattr(artifact.get("pca"), "n_components_", "?")))
    print("schema  :", artifact.get("feature_schema_hash", artifact.get("schema_hash", "MISSING")))

if M3.exists():
    b = json.loads(M3.read_text(encoding="utf-8"))
    print("\n[M3/M4]")
    print("known recognition :", b.get("thresholds", b).get("0.90", b.get("known_recognition", "see artifact")))
    print("artifact loaded   : OK")

if M10.exists():
    r = json.loads(M10.read_text(encoding="utf-8"))
    print("\n[M10]")
    print("detector    :", r.get("detection", {}).get("detector_decision"))
    print("recognition :", r.get("recognition", {}).get("classification"))
    print("risk        :", r.get("risk", {}).get("risk_score"))
    print("decision    :", r.get("decision", {}).get("action"))
    print("memory      :", r.get("memory_count"))

print("\n" + "=" * 88)
if failures:
    print("[FAIL] Missing required artifacts:", ", ".join(failures))
    raise SystemExit(1)

print("[PASS] M10 FINAL SYSTEM AUDIT")
