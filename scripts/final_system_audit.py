from __future__ import annotations

import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "src/pipeline/contracts.py",
    "src/pipeline/real_adis_pipeline.py",
    "src/pipeline/encoder_adapter.py",
    "src/pipeline/memory_adapter.py",
    "src/detection/lightgbm_detector.py",
    "src/analysis/threat_analyzer.py",
    "src/analysis/risk_engine.py",
    "src/decision/decision_engine.py",
    "src/analysis/threat_analyzer.py",
    "src/investigation/investigator.py",
    "src/sandbox/isolation_chamber.py",
    "scripts/test_m554_admission_gate.py",
    "scripts/test_m554_persistence.py",
    "scripts/test_m61_investigator.py",
    "scripts/test_m64_evidence_quality.py",
    "scripts/test_m7_response_lifecycle.py",
    "scripts/test_m8_attack_replay.py",
    "scripts/benchmark_m9_recognition.py",
]


REQUIRED_ARTIFACTS = [
    "models/behavioral_encoder.joblib",
    "models/merged_binary_detector.joblib",
]


MODULES = [
    "src.pipeline.contracts",
    "src.pipeline.real_adis_pipeline",
    "src.pipeline.encoder_adapter",
    "src.pipeline.memory_adapter",
    "src.detection.lightgbm_detector",
    "src.analysis.threat_analyzer",
    "src.analysis.risk_engine",
    "src.decision.decision_engine",
    "src.investigation.investigator",
    "src.sandbox.isolation_chamber",
]


def check_files():
    print()
    print("[1/4] Required files")

    passed = True

    for relative in REQUIRED_FILES:
        path = ROOT / relative

        if path.exists():
            print(f"[PASS] {relative}")
        else:
            print(f"[FAIL] {relative}")
            passed = False

    return passed


def check_artifacts():
    print()
    print("[2/4] Required model artifacts")

    passed = True

    for relative in REQUIRED_ARTIFACTS:
        path = ROOT / relative

        if path.exists():
            print(f"[PASS] {relative}")
        else:
            print(f"[FAIL] {relative}")
            passed = False

    return passed


def check_imports():
    print()
    print("[3/4] Module imports")

    passed = True

    for module in MODULES:
        try:
            importlib.import_module(module)
            print(f"[PASS] {module}")
        except Exception as exc:
            print(f"[FAIL] {module}")
            print(f"       {exc}")
            passed = False

    return passed


def check_contracts():
    print()
    print("[4/4] ADIS contracts")

    passed = True

    try:
        from src.pipeline.contracts import (
            CONTRACT_VERSION,
            ENCODER_VERSION,
            EMBEDDING_DIM,
            KNOWN_THRESHOLD,
            NEAR_THRESHOLD,
            FEATURE_SCHEMA_HASH,
        )

        checks = [
            (
                "CONTRACT_VERSION",
                CONTRACT_VERSION == "adis-contract-v1",
            ),
            (
                "ENCODER_VERSION",
                ENCODER_VERSION == "adis-behavioral-encoder-v2",
            ),
            (
                "EMBEDDING_DIM",
                EMBEDDING_DIM == 32,
            ),
            (
                "KNOWN_THRESHOLD",
                KNOWN_THRESHOLD == 0.92,
            ),
            (
                "NEAR_THRESHOLD",
                NEAR_THRESHOLD == 0.75,
            ),
            (
                "FEATURE_SCHEMA_HASH",
                bool(FEATURE_SCHEMA_HASH),
            ),
        ]

        for name, result in checks:
            if result:
                print(f"[PASS] {name}")
            else:
                print(f"[FAIL] {name}")
                passed = False

    except Exception as exc:
        print(f"[FAIL] Contract import: {exc}")
        passed = False

    return passed


def main():
    print("=" * 90)
    print("M10 — FINAL ADIS SYSTEM AUDIT")
    print("=" * 90)

    results = [
        check_files(),
        check_artifacts(),
        check_imports(),
        check_contracts(),
    ]

    print()
    print("=" * 90)

    if all(results):
        print("[PASS] M10 FINAL SYSTEM AUDIT")
        print("=" * 90)
        return

    print("[FAIL] M10 FINAL SYSTEM AUDIT")
    print("=" * 90)

    raise SystemExit(1)


if __name__ == "__main__":
    main()