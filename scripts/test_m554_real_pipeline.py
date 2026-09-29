from __future__ import annotations

import json

import pandas as pd

from src.pipeline.real_adis_pipeline import RealADISPipeline


DATASET = (
    "data/raw/cicids2017/"
    "Bruteforce-Tuesday-no-metadata.parquet"
)


def main() -> None:
    print("=" * 72)
    print("M5.5.4 — Real ADIS Pipeline Smoke Test")
    print("=" * 72)

    df = pd.read_parquet(DATASET)

    print(f"\nDataset rows    : {len(df):,}")
    print(f"Dataset columns : {len(df.columns)}")

    # One real CIC-IDS2017 flow.
    flow = df.iloc[0].to_dict()

    pipeline = RealADISPipeline()

    print("\n[Pipeline]")
    print(json.dumps(
        pipeline.info(),
        indent=2,
        default=str,
    ))

    print("\n[Processing first flow]")

    result = pipeline.process(flow)

    print("\n[RESULT]")
    print(json.dumps(
        result,
        indent=2,
        default=str,
    ))

    print("\n" + "=" * 72)
    print("REAL PIPELINE SMOKE TEST COMPLETED")
    print("=" * 72)


if __name__ == "__main__":
    main()



    