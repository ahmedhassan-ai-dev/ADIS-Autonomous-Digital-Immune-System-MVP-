from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy
import pandas
import sklearn


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "behavioral_encoder.joblib"
OUTPUT = ROOT / "results" / "m3_m4_recognition_benchmark_v3" / "environment_audit.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def main() -> None:
    print("=" * 80)
    print("[ADIS] M3 v3 / M4 Environment & Artifact Audit")
    print("=" * 80)

    if not MODEL.exists():
        raise FileNotFoundError(MODEL)

    artifact = joblib.load(MODEL)

    report = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": numpy.__version__,
        "pandas_version": pandas.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "artifact": {
            "path": str(MODEL),
            "sha256": sha256(MODEL),
            "encoder_version": artifact.get("encoder_version"),
            "representation_type": artifact.get(
                "representation_type"
            ),
            "features": len(
                artifact.get("features", [])
            ),
            "embedding_dim": artifact.get(
                "embedding_dim"
            ),
            "training_rows": artifact.get(
                "training_rows"
            ),
            "validation_rows": artifact.get(
                "validation_rows"
            ),
            "training_families": artifact.get(
                "training_families"
            ),
            "novel_family_excluded": artifact.get(
                "novel_family_excluded"
            ),
            "schema_hash": artifact.get(
                "feature_schema_hash"
            ),
            "random_state": artifact.get(
                "random_state"
            ),
        },
        "m4_policy": {
            "policy_version": "adis-m4-recognition-policy-v1",
            "known_threshold": 0.90,
            "near_threshold": 0.75,
        },
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("[ENVIRONMENT]")
    print("Python :", report["python_version"])
    print("NumPy  :", report["numpy_version"])
    print("Pandas :", report["pandas_version"])
    print("sklearn:", report["scikit_learn_version"])
    print("joblib :", report["joblib_version"])

    print()
    print("[ARTIFACT]")
    for key, value in report["artifact"].items():
        if key != "sha256":
            print(f"{key:<24}: {value}")

    print()
    print("[M4 POLICY]")
    print("Known threshold :", 0.90)
    print("Near threshold  :", 0.75)

    print()
    print("[OK] Environment audit saved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()