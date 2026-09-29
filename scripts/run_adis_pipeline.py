from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.pipeline.real_adis_pipeline import RealADISPipeline
from src.pipeline.memory_adapter import ImmuneMemoryAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MEMORY_PATH = PROJECT_ROOT / "models" / "immune_memory"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "runtime"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ADIS production pipeline on CICIDS flows."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to CICIDS Parquet file.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of flows to process.",
    )

    parser.add_argument(
        "--memory-path",
        default=str(DEFAULT_MEMORY_PATH),
        help="Persistent Immune Memory path.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSONL output path.",
    )

    return parser.parse_args()


def json_safe(value: Any):
    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def main():
    args = parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    if args.limit < 1:
        raise ValueError("--limit must be >= 1")

    output_path = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR
        / f"{dataset_path.stem}_adis_runtime.jsonl"
    )

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("[ADIS] Production Pipeline Runner")
    print("=" * 72)

    print("[ADIS] Dataset :", dataset_path)
    print("[ADIS] Limit   :", args.limit)
    print("[ADIS] Memory  :", args.memory_path)
    print("[ADIS] Output  :", output_path)

    print("\n[ADIS] Loading dataset...")

    df = pd.read_parquet(dataset_path)

    if df.empty:
        raise RuntimeError("Dataset is empty.")

    limit = min(args.limit, len(df))

    print("[ADIS] Dataset rows :", len(df))
    print("[ADIS] Processing   :", limit)

    memory = ImmuneMemoryAdapter(
        storage_path=args.memory_path
    )

    pipeline = RealADISPipeline(
        memory=memory
    )

    print(
        "[ADIS] Existing memory cells:",
        memory.count_memories(),
    )

    counters = {
        "BENIGN": 0,
        "ATTACK": 0,
        "NOVEL": 0,
        "UNCERTAIN": 0,
        "KNOWN": 0,
        "ALLOW": 0,
        "INVESTIGATE": 0,
        "ISOLATE": 0,
        "NEW_MEMORY_CREATED": 0,
        "LEARNING_SKIPPED": 0,
    }

    latencies = []

    started = time.perf_counter()

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for position, (_, row) in enumerate(
            df.head(limit).iterrows(),
            start=1,
        ):

            flow = row.to_dict()

            result = pipeline.process(flow)

            context = result["context"]

            detection = context["detection"]
            recognition = context["recognition"]
            decision = result["decision"]
            memory_event = result["memory_event"]

            detector_decision = (
                detection["detector_decision"]
            )

            recognition_class = (
                recognition["classification"]
            )

            decision_action = decision["action"]

            counters[detector_decision] = (
                counters.get(detector_decision, 0) + 1
            )

            counters[recognition_class] = (
                counters.get(recognition_class, 0) + 1
            )

            counters[decision_action] = (
                counters.get(decision_action, 0) + 1
            )

            if memory_event:
                status = memory_event.get("status")

                counters[status] = (
                    counters.get(status, 0) + 1
                )

            latency = float(
                result["latency_ms"]
            )

            latencies.append(latency)

            record = {
                "position": position,
                "dataset_row": int(
                    position - 1
                ),
                "dataset_label": flow.get("Label"),
                "detection": detection,
                "recognition": recognition,
                "behavior": context["behavior"],
                "threat": context["threat"],
                "risk": result["risk"],
                "decision": decision,
                "memory_event": memory_event,
                "latency_ms": latency,
            }

            output_file.write(
                json.dumps(
                    json_safe(record),
                    ensure_ascii=False,
                )
                + "\n"
            )

            if position == 1 or position % 10 == 0:
                print(
                    f"[{position:>5}/{limit}] "
                    f"{detector_decision:<7} | "
                    f"{recognition_class:<9} | "
                    f"{decision_action:<11} | "
                    f"{latency:.2f} ms"
                )

    total_time = (
        time.perf_counter() - started
    )

    print("\n" + "=" * 72)
    print("[ADIS] Runtime Summary")
    print("=" * 72)

    print("Processed flows :", limit)
    print("Runtime seconds :", round(total_time, 3))

    if latencies:
        print(
            "Avg latency     :",
            round(
                sum(latencies) / len(latencies),
                3,
            ),
            "ms",
        )

        print(
            "Min latency     :",
            round(min(latencies), 3),
            "ms",
        )

        print(
            "Max latency     :",
            round(max(latencies), 3),
            "ms",
        )

    print("\nDetector:")
    print("  BENIGN :", counters.get("BENIGN", 0))
    print("  ATTACK :", counters.get("ATTACK", 0))

    print("\nRecognition:")
    print("  NOVEL     :", counters.get("NOVEL", 0))
    print("  UNCERTAIN :", counters.get("UNCERTAIN", 0))
    print("  KNOWN     :", counters.get("KNOWN", 0))

    print("\nDecision:")
    print("  ALLOW       :", counters.get("ALLOW", 0))
    print("  INVESTIGATE :", counters.get("INVESTIGATE", 0))
    print("  ISOLATE     :", counters.get("ISOLATE", 0))

    print("\nMemory:")
    print(
        "  NEW_MEMORY_CREATED :",
        counters.get(
            "NEW_MEMORY_CREATED",
            0,
        ),
    )

    print(
        "  LEARNING_SKIPPED   :",
        counters.get(
            "LEARNING_SKIPPED",
            0,
        ),
    )

    print(
        "  Final memory count  :",
        memory.count_memories(),
    )

    print("\nAudit log:")
    print(" ", output_path)

    print("=" * 72)
    print("[PASS] ADIS production runner completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()