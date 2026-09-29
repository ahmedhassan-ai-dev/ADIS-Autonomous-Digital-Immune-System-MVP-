from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline.memory_adapter import ImmuneMemoryAdapter
from src.pipeline.real_adis_pipeline import RealADISPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    PROJECT_ROOT
    / "data"
    / "runtime"
    / "m56_attack_test.parquet"
)

MEMORY = (
    PROJECT_ROOT
    / "models"
    / "immune_memory_m62_test"
)


def main():
    df = pd.read_parquet(DATASET)

    row = df.iloc[0].to_dict()

    memory = ImmuneMemoryAdapter(
        storage_path=str(MEMORY)
    )

    pipeline = RealADISPipeline(
        memory=memory
    )

    result = pipeline.process(row)

    investigation = result.get("investigation")

    assert investigation is not None

    assert investigation["status"] in {
        "NOVEL_THREAT",
        "UNCERTAIN_THREAT",
        "KNOWN_THREAT",
        "BENIGN_ACTIVITY",
        "UNCLASSIFIED",
    }

    assert "evidence" in investigation
    assert "explanation" in investigation
    assert "recommended_action" in investigation

    print("=" * 72)
    print("M6.2 — PIPELINE INVESTIGATION INTEGRATION")
    print("=" * 72)

    print()
    print("Dataset Label :", row.get("Label"))
    print("Status        :", investigation["status"])
    print("Family        :", investigation["behavior_family"])
    print("Threat Type   :", investigation["threat_type"])
    print("Detection     :", investigation["detector_decision"])
    print("Recognition   :", investigation["recognition_classification"])
    print("Risk          :", investigation["risk_score"])
    print("Severity      :", investigation["severity"])
    print("Action        :", investigation["recommended_action"])

    print()
    print("Evidence:")
    for item in investigation["evidence"]:
        print(
            f"  [{item['source']}] "
            f"{item['type']} = {item['value']}"
        )

    print()
    print("Explanation:")
    print(investigation["explanation"])

    print()
    print("[PASS] M6.2 Pipeline Investigation Integration")


if __name__ == "__main__":
    main()