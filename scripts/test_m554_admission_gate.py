from pathlib import Path
import shutil

import pandas as pd

from src.pipeline.real_adis_pipeline import RealADISPipeline
from src.pipeline.memory_adapter import ImmuneMemoryAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cicids2017"
    / "Bruteforce-Tuesday-no-metadata.parquet"
)

TEST_MEMORY_PATH = (
    PROJECT_ROOT
    / "models"
    / "immune_memory_m554_gate_test"
)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def print_result(title, result):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    context = result["context"]
    detection = context["detection"]
    recognition = context["recognition"]

    print("Detector Decision :", detection["detector_decision"])
    print("Anomaly Score     :", detection["anomaly_score"])
    print("Recognition       :", recognition["classification"])
    print("Similarity        :", recognition["similarity"])
    print("Policy Action     :", recognition["policy_action"])
    print("Risk Score        :", result["risk"]["risk_score"])
    print("Decision          :", result["decision"]["action"])
    print("Memory Event      :", result.get("memory_event"))


def main():
    print("=" * 70)
    print("[ADIS] M5.5.4.3 — Memory Admission Gate Test")
    print("=" * 70)

    if TEST_MEMORY_PATH.exists():
        print(f"[ADIS] Removing previous test memory: {TEST_MEMORY_PATH}")
        shutil.rmtree(TEST_MEMORY_PATH)

    print(f"[ADIS] Loading dataset: {DATA_PATH.name}")

    df = pd.read_parquet(DATA_PATH)

    print(f"[ADIS] Rows loaded: {len(df):,}")

    # ------------------------------------------------------------
    # Create pipeline with isolated test memory
    # ------------------------------------------------------------

    memory = ImmuneMemoryAdapter(
        storage_path=str(TEST_MEMORY_PATH),
        top_k=5,
    )
    pipeline = RealADISPipeline(
        memory=memory
    )

    print("\n[ADIS] Initial memory count:",
          pipeline.memory.count_memories())

    assert_true(
        pipeline.memory.count_memories() == 0,
        "Test memory is not empty."
    )

    # ------------------------------------------------------------
    # Find a benign flow
    # ------------------------------------------------------------

    benign_mask = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("benign")
    )

    benign_rows = df[benign_mask]

    assert_true(
        len(benign_rows) > 0,
        "No benign rows found."
    )

    benign_flow = benign_rows.iloc[0].to_dict()

    # ------------------------------------------------------------
    # TEST A — BENIGN + NOVEL
    # ------------------------------------------------------------

    print("\n[TEST A] Processing unseen BENIGN flow...")

    benign_result_1 = pipeline.process(benign_flow)

    print_result(
        "TEST A — First Benign Exposure",
        benign_result_1,
    )

    memory_count_after_benign = (
        pipeline.memory.count_memories()
    )

    print(
        "\nMemory count after benign:",
        memory_count_after_benign,
    )

    assert_true(
        benign_result_1["context"]["recognition"]["classification"] == "NOVEL",
        "Expected first benign flow to be NOVEL."
    )

    assert_true(
        benign_result_1.get("memory_event", {}).get("status")
        == "LEARNING_SKIPPED",
        "Benign novel flow was not rejected by the learning gate."
    )

    assert_true(
        memory_count_after_benign == 0,
        "CRITICAL: benign flow was inserted into Immune Memory."
    )

    # ------------------------------------------------------------
    # TEST B — SAME BENIGN AGAIN
    # ------------------------------------------------------------

    print("\n[TEST B] Processing same BENIGN flow again...")

    benign_result_2 = pipeline.process(benign_flow)

    print_result(
        "TEST B — Second Benign Exposure",
        benign_result_2,
    )

    memory_count_after_benign_2 = (
        pipeline.memory.count_memories()
    )

    print(
        "\nMemory count after second benign:",
        memory_count_after_benign_2,
    )

    assert_true(
        benign_result_2["context"]["recognition"]["classification"] == "NOVEL",
        "Benign flow became KNOWN after being rejected. Memory gate failed."
    )

    assert_true(
        memory_count_after_benign_2 == 0,
        "Benign flow polluted Immune Memory."
    )

    # ------------------------------------------------------------
    # Find an actual ATTACK flow that the detector recognizes
    # ------------------------------------------------------------

    print("\n[ADIS] Searching for a real ATTACK flow...")

    attack_flow = None
    attack_detector_result = None

    for idx in range(len(df)):
        row = df.iloc[idx]

        label = str(row.get("Label", "")).strip().lower()

        if label == "benign":
            continue

        flow = row.to_dict()

        detection = pipeline.detector.detect_one(flow)

        detector_decision = str(
            detection.get("decision", detection.get("detector_decision", ""))
        ).upper()

        if detector_decision in {
            "ATTACK",
            "ANOMALY",
            "MALICIOUS",
            "SUSPICIOUS",
        }:
            attack_flow = flow
            attack_detector_result = detection

            print(
                f"[ADIS] Attack candidate found at row {idx}"
            )
            print(
                f"[ADIS] Dataset label     : {row['Label']}"
            )
            print(
                f"[ADIS] Detector decision : "
                f"{detector_decision}"
            )
            print(
                f"[ADIS] Anomaly score     : "
                f"{detection.get('anomaly_score')}"
            )
            break

    assert_true(
        attack_flow is not None,
        "Could not find an actual attack flow recognized by LightGBM."
    )

    # ------------------------------------------------------------
    # TEST C — ATTACK + NOVEL
    # ------------------------------------------------------------

    print("\n[TEST C] Processing unseen ATTACK flow...")

    attack_result_1 = pipeline.process(attack_flow)

    print_result(
        "TEST C — First Attack Exposure",
        attack_result_1,
    )

    memory_count_after_attack = (
        pipeline.memory.count_memories()
    )

    print(
        "\nMemory count after attack:",
        memory_count_after_attack,
    )

    assert_true(
        attack_result_1["context"]["detection"]["detector_decision"].upper()
        in {
            "ATTACK",
            "ANOMALY",
            "MALICIOUS",
            "SUSPICIOUS",
        },
        "Pipeline did not preserve the suspicious detector decision."
    )

    assert_true(
        attack_result_1["context"]["recognition"]["classification"] == "NOVEL",
        "Expected first attack exposure to be NOVEL."
    )

    assert_true(
        attack_result_1.get("memory_event") is not None,
        "Novel suspicious attack was not admitted to memory."
    )

    assert_true(
        memory_count_after_attack == 1,
        "Expected exactly one attack memory after admission."
    )

    # ------------------------------------------------------------
    # TEST D — SAME ATTACK AGAIN
    # ------------------------------------------------------------

    print("\n[TEST D] Processing same ATTACK again...")

    attack_result_2 = pipeline.process(attack_flow)

    print_result(
        "TEST D — Second Attack Exposure",
        attack_result_2,
    )

    memory_count_final = (
        pipeline.memory.count_memories()
    )

    print(
        "\nFinal memory count:",
        memory_count_final,
    )

    assert_true(
        attack_result_2["context"]["recognition"]["classification"] == "KNOWN",
        "Stored attack was not recognized on subsequent exposure."
    )

    assert_true(
        attack_result_2["context"]["recognition"]["similarity"] >= 0.92,
        "Known attack similarity is below the production threshold."
    )

    assert_true(
        memory_count_final == 1,
        "Duplicate attack memory was created."
    )

    # ------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("[PASS] M5.5.4.3 Memory Admission Gate")
    print("=" * 70)

    print("✓ BENIGN + NOVEL  → LEARNING_SKIPPED")
    print("✓ BENIGN remains NOVEL")
    print("✓ BENIGN memory count remains 0")
    print("✓ ATTACK + NOVEL  → MEMORY COMMIT")
    print("✓ ATTACK memory count becomes 1")
    print("✓ ATTACK again    → KNOWN")
    print("✓ No duplicate memory created")
    print("=" * 70)


if __name__ == "__main__":
    main()