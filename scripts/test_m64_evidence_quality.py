from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.pipeline.real_adis_pipeline import RealADISPipeline
from src.pipeline.memory_adapter import ImmuneMemoryAdapter


DATASET = Path(
    "data/raw/cicids2017/Bruteforce-Tuesday-no-metadata.parquet"
)

MEMORY_PATH = Path(
    "models/m64_test_memory"
)


def main():

    print("=" * 90)
    print("M6.4 — INVESTIGATION EVIDENCE QUALITY TEST")
    print("=" * 90)

    df = pd.read_parquet(DATASET)

    label_col = "Label"

    attack_rows = df[
        df[label_col].astype(str).str.upper() != "BENIGN"
    ]

    if attack_rows.empty:
        raise RuntimeError(
            "No attack rows found."
        )

    row = attack_rows.iloc[0]

    print(
        f"\n[INFO] Dataset label: "
        f"{row[label_col]}"
    )

    flow = row.to_dict()

    memory = ImmuneMemoryAdapter(
        storage_path=str(MEMORY_PATH)
    )

    pipeline = RealADISPipeline(
        memory=memory
    )

    result = pipeline.process(flow)

    investigation = result.get(
        "investigation"
    )

    if not investigation:
        raise RuntimeError(
            "Investigation report missing."
        )

    print(
        f"\nStatus       : "
        f"{investigation.get('status')}"
    )

    print(
        f"Family       : "
        f"{investigation.get('behavior_family')}"
    )

    print(
        f"Threat Type  : "
        f"{investigation.get('threat_type')}"
    )

    print(
        f"Action       : "
        f"{investigation.get('recommended_action')}"
    )

    evidence = investigation.get(
        "evidence",
        []
    )

    print(
        f"\nEvidence count: "
        f"{len(evidence)}"
    )

    for item in evidence:
        print(
            f"  [{item.get('source')}] "
            f"{item.get('type')} = "
            f"{item.get('value')}"
        )

    m3_flow_features = [
        item
        for item in evidence
        if item.get("source")
        == "M3_FLOW_FEATURES"
    ]

    print(
        f"\nM3_FLOW_FEATURES: "
        f"{len(m3_flow_features)}"
    )

    if not m3_flow_features:
        raise RuntimeError(
            "M6.4 FAILED: "
            "M3 behavioral flow evidence "
            "did not reach Investigator."
        )

    print(
        "\n[PASS] M6.4 Evidence Integration "
        "is working."
    )


if __name__ == "__main__":
    main()