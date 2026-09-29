from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.pipeline.memory_adapter import ImmuneMemoryAdapter
from src.pipeline.real_adis_pipeline import RealADISPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="ADIS end-to-end attack replay demo."
    )

    parser.add_argument(
        "--dataset",
        default="data/runtime/m56_attack_test.parquet",
        help="Attack dataset used for replay.",
    )

    parser.add_argument(
        "--memory-path",
        default="models/immune_memory_m57_demo",
        help="Persistent memory path for the demo.",
    )

    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="Zero-based row to replay.",
    )

    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def process_one(dataset_path: Path, memory_path: Path, row_index: int):
    df = pd.read_parquet(dataset_path)

    if df.empty:
        raise RuntimeError("Dataset is empty.")

    if row_index < 0 or row_index >= len(df):
        raise IndexError(
            f"Row {row_index} is outside dataset range 0..{len(df)-1}"
        )

    row = df.iloc[row_index].to_dict()

    memory = ImmuneMemoryAdapter(
        storage_path=str(memory_path)
    )

    pipeline = RealADISPipeline(memory=memory)

    result = pipeline.process(row)

    return row, result, memory.count_memories()


def print_result(title: str, row: dict, result: dict, memory_count: int):
    context = result["context"]

    detection = context["detection"]
    recognition = context["recognition"]
    threat = context["threat"]
    risk = result["risk"]
    decision = result["decision"]
    memory_event = result["memory_event"]

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    print()
    print("FLOW")
    print("-" * 72)
    print("Dataset Label :", row.get("Label"))
    print("Source        :", row.get("Source", "CICIDS2017"))

    print()
    print("M2 — DETECTION")
    print("-" * 72)
    print("Detector      :", detection.get("detector"))
    print("Decision      :", detection.get("detector_decision"))
    print("Attack Score  :", detection.get("attack_probability"))

    print()
    print("M3 — BEHAVIOR")
    print("-" * 72)
    print("Family        :", threat.get("behavior_family"))
    print("Threat Type   :", threat.get("threat_type"))

    print()
    print("M4 — IMMUNE MEMORY")
    print("-" * 72)
    print("Classification:", recognition.get("classification"))
    print("Similarity    :", recognition.get("similarity"))
    print("Matched ID    :", recognition.get("matched_memory_id"))
    print("Memory Event  :", memory_event.get("status") if memory_event else None)

    print()
    print("RISK")
    print("-" * 72)
    print("Score         :", risk.get("risk_score"))
    print("Severity      :", risk.get("severity"))

    print()
    print("DECISION")
    print("-" * 72)
    print("Action        :", decision.get("action"))
    print("Reason        :", decision.get("reason"))

    print()
    print("Memory Count  :", memory_count)
    print("Latency       :", result.get("latency_ms"), "ms")


def main():
    args = parse_args()

    dataset_path = resolve_path(args.dataset)
    memory_path = resolve_path(args.memory_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    memory_path.mkdir(parents=True, exist_ok=True)

    print()
    print("#" * 72)
    print("# ADIS — M5.7 END-TO-END ATTACK REPLAY")
    print("#" * 72)

    print()
    print("Dataset :", dataset_path)
    print("Memory  :", memory_path)
    print("Row     :", args.row)

    # ------------------------------------------------------------------
    # Exposure 1
    # ------------------------------------------------------------------

    row1, result1, memory_count1 = process_one(
        dataset_path,
        memory_path,
        args.row,
    )

    print_result(
        "EXPOSURE 1 — UNKNOWN ATTACK",
        row1,
        result1,
        memory_count1,
    )

    # ------------------------------------------------------------------
    # Simulated process restart
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("SIMULATED RESTART")
    print("=" * 72)

    print("Creating a fresh Python process...")
    print()

    # We intentionally do not reuse the in-memory pipeline object.
    # A subprocess is used to prove that persistence survives process exit.

    script = f"""
import json
import pandas as pd

from src.pipeline.memory_adapter import ImmuneMemoryAdapter
from src.pipeline.real_adis_pipeline import RealADISPipeline

dataset_path = r"{dataset_path}"
memory_path = r"{memory_path}"
row_index = {args.row}

df = pd.read_parquet(dataset_path)
row = df.iloc[row_index].to_dict()

memory = ImmuneMemoryAdapter(storage_path=memory_path)
pipeline = RealADISPipeline(memory=memory)

result = pipeline.process(row)

output = {{
    "result": result,
    "memory_count": memory.count_memories()
}}

print(json.dumps(output, default=str))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)

        raise RuntimeError(
            "Fresh-process replay failed."
        )

    lines = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]

    json_line = None

    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            json_line = line
            break

    if json_line is None:
        print(completed.stdout)
        print(completed.stderr)

        raise RuntimeError(
            "Could not parse replay subprocess output."
        )

    payload = json.loads(json_line)

    result2 = payload["result"]
    memory_count2 = payload["memory_count"]

    print_result(
        "EXPOSURE 2 — AFTER RESTART",
        row1,
        result2,
        memory_count2,
    )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    classification1 = (
        result1["context"]["recognition"]["classification"]
    )

    classification2 = (
        result2["context"]["recognition"]["classification"]
    )

    memory_event1 = result1["memory_event"]

    memory_status1 = (
        memory_event1.get("status")
        if memory_event1
        else None
    )

    memory_event2 = result2["memory_event"]

    memory_status2 = (
        memory_event2.get("status")
        if memory_event2
        else None
    )

    print()
    print("#" * 72)
    print("# M5.7 VERIFICATION")
    print("#" * 72)

    checks = {
        "Exposure 1 is NOVEL":
            classification1 == "NOVEL",

        "Exposure 1 created memory":
            memory_status1 == "NEW_MEMORY_CREATED",

        "Memory survived restart":
            memory_count2 >= memory_count1,

        "Exposure 2 is not NOVEL":
            classification2 in {"KNOWN", "UNCERTAIN"},

        "Exposure 2 created no new memory":
            memory_status2 != "NEW_MEMORY_CREATED",

        "Persistent memory count stable":
            memory_count2 == memory_count1,
    }

    all_passed = True

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"

        print(
            f"[{status}] {name}"
        )

        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("[PASS] M5.7 END-TO-END ATTACK REPLAY")
    else:
        print("[FAIL] M5.7 END-TO-END ATTACK REPLAY")
        raise SystemExit(1)


if __name__ == "__main__":
    main()