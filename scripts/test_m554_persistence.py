from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_PATH = PROJECT_ROOT / "models" / "immune_memory_m554_persistence_test"
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cicids2017"
    / "Bruteforce-Tuesday-no-metadata.parquet"
)


def run_python(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Child process failed with exit code {result.returncode}"
        )

    return result.stdout


def main():
    print("=" * 70)
    print("[ADIS] M5.5.4.4 — Restart Persistence Test")
    print("=" * 70)

    if MEMORY_PATH.exists():
        print(f"[ADIS] Removing previous test memory: {MEMORY_PATH}")
        shutil.rmtree(MEMORY_PATH)

    print(f"[ADIS] Dataset: {DATASET_PATH.name}")

    # ---------------------------------------------------------
    # Phase 1
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1 — PROCESS A: Learn Attack")
    print("=" * 70)

    phase1_code = f"""
from pathlib import Path
import pandas as pd

from src.pipeline.real_adis_pipeline import RealADISPipeline
from src.pipeline.memory_adapter import ImmuneMemoryAdapter
from src.detection.lightgbm_detector import LightGBMDetector

memory_path = Path(r"{MEMORY_PATH}")
dataset_path = Path(r"{DATASET_PATH}")

df = pd.read_parquet(dataset_path)

detector = LightGBMDetector()

attack_flow = None
attack_index = None

for index, row in df.iterrows():
    flow = row.to_dict()

    result = detector.detect_one(flow)

    if result["decision"].upper() == "ATTACK":
        attack_flow = flow
        attack_index = index
        print("[PHASE 1] Attack found at row:", index)
        print("[PHASE 1] Detector result:", result)
 
        break

if attack_flow is None:
    raise RuntimeError("Could not find a detector-confirmed attack flow.")

memory = ImmuneMemoryAdapter(
    storage_path=str(memory_path)
)

pipeline = RealADISPipeline(
    memory=memory
)

result = pipeline.process(attack_flow)

context = result["context"]
recognition = context["recognition"]
detection = context["detection"]

print("[PHASE 1] Detection :", detection["detector_decision"])
print("[PHASE 1] Recognition:", recognition["classification"])
print("[PHASE 1] Similarity :", recognition["similarity"])
print("[PHASE 1] Decision   :", result["decision"])
print("[PHASE 1] Memory     :", result["memory_event"])
print("[PHASE 1] Memory count:", memory.count_memories())

if recognition["classification"] != "NOVEL":
    raise AssertionError(
        "Phase 1 expected NOVEL recognition for an empty memory."
    )

if result["memory_event"] is None:
    raise AssertionError(
        "Phase 1 expected a memory commit event."
    )

if result["memory_event"].get("status") != "NEW_MEMORY_CREATED":
    raise AssertionError(
        f"Unexpected memory event: {{result['memory_event']}}"
    )

if memory.count_memories() != 1:
    raise AssertionError(
        f"Expected memory count = 1, got {{memory.count_memories()}}"
    )

print("[PHASE 1 PASS] Attack learned and stored.")
"""

    phase1_output = run_python(phase1_code)

    # ---------------------------------------------------------
    # Phase 2
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2 — PROCESS B: Restart + Recognize")
    print("=" * 70)

    # We cannot depend on Python state from Phase 1.
    # The attack flow is reconstructed independently.
    phase2_code = f"""
from pathlib import Path
import pandas as pd

from src.pipeline.real_adis_pipeline import RealADISPipeline
from src.pipeline.memory_adapter import ImmuneMemoryAdapter
from src.detection.lightgbm_detector import LightGBMDetector

memory_path = Path(r"{MEMORY_PATH}")
dataset_path = Path(r"{DATASET_PATH}")

df = pd.read_parquet(dataset_path)

detector = LightGBMDetector()

attack_flow = None

for index, row in df.iterrows():
    flow = row.to_dict()

    result = detector.detect_one(flow)

    if result["decision"].upper() == "ATTACK":
        attack_flow = flow
        print("[PHASE 2] Reconstructed attack at row:", index)
        break

if attack_flow is None:
    raise RuntimeError("Could not reconstruct attack flow.")

memory = ImmuneMemoryAdapter(
    storage_path=str(memory_path)
)

print("[PHASE 2] Memory count after restart:", memory.count_memories())

if memory.count_memories() != 1:
    raise AssertionError(
        "Persistent memory was not restored after process restart."
    )

pipeline = RealADISPipeline(
    memory=memory
)

result = pipeline.process(attack_flow)

context = result["context"]
recognition = context["recognition"]
detection = context["detection"]

print("[PHASE 2] Detection :", detection["detector_decision"])
print("[PHASE 2] Recognition:", recognition["classification"])
print("[PHASE 2] Similarity :", recognition["similarity"])
print("[PHASE 2] Decision   :", result["decision"])
print("[PHASE 2] Memory     :", result["memory_event"])
print("[PHASE 2] Memory count:", memory.count_memories())

if recognition["classification"] != "KNOWN":
    raise AssertionError(
        "Expected KNOWN recognition after restart."
    )

if recognition["similarity"] < 0.92:
    raise AssertionError(
        f"Expected similarity >= 0.92, got {{recognition['similarity']}}"
    )

if result["memory_event"] is not None:
    raise AssertionError(
        "Known attack should not create a duplicate memory."
    )

if memory.count_memories() != 1:
    raise AssertionError(
        f"Expected memory count to remain 1, got {{memory.count_memories()}}"
    )

print("[PHASE 2 PASS] Persistent memory survived restart.")
"""

    run_python(phase2_code)

    print("\n" + "=" * 70)
    print("[PASS] M5.5.4.4 Restart Persistence Test")
    print("=" * 70)
    print("✓ Process A learned the attack")
    print("✓ Memory was written to disk")
    print("✓ Process A terminated")
    print("✓ Process B started independently")
    print("✓ Memory count restored")
    print("✓ Same attack recognized as KNOWN")
    print("✓ Similarity remained >= 0.92")
    print("✓ No duplicate memory created")
    print("=" * 70)


if __name__ == "__main__":
    main()