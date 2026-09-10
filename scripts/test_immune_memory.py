from pathlib import Path
import shutil
import numpy as np

from src.memory.immune_memory import ImmuneMemory


def main():

    print("=" * 75)
    print("[ADIS] M4 — Immune Memory Persistence Test")
    print("=" * 75)

    test_path = Path("models/immune_memory_test")

    # Clean test memory so every test starts fresh.
    if test_path.exists():
        shutil.rmtree(test_path)

    # --------------------------------------------------------------
    # Create memory
    # --------------------------------------------------------------

    memory = ImmuneMemory(
        storage_path=test_path,
        embedding_dim=32,
        known_threshold=0.92,
        near_threshold=0.80,
        encoder_version="behavioral_encoder_v1",
        feature_schema_hash="TEST_SCHEMA_001",
    )

    print("\n[1] Initial memory count:")
    print("    ", memory.count_memories())

    # --------------------------------------------------------------
    # Create deterministic fake antigen
    # --------------------------------------------------------------

    rng = np.random.default_rng(42)

    vector = rng.normal(
        size=32
    ).astype(np.float32)

    vector /= np.linalg.norm(vector)

    class FakeAntigen:
        antigen_id = "test-antigen-001"
        technique_id = "TEST"
        technique_name = "Test Threat"
        tactic = "Testing"
        severity = "HIGH"

        embedding = vector.tolist()

        metadata = {
            "anomaly_score": 0.97,
            "behavior_family": "test_behavior",
            "protocol": 6,
            "flow_duration": 1000,
        }

    antigen = FakeAntigen()

    # --------------------------------------------------------------
    # First exposure
    # --------------------------------------------------------------

    print("\n[2] First exposure:")

    result = memory.recognize_threat(
        vector
    )

    print(result["classification"])
    print("Similarity:", result["similarity"])

    # Should be NOVEL.
    assert result["classification"] == "NOVEL"

    # --------------------------------------------------------------
    # Commit memory
    # --------------------------------------------------------------

    print("\n[3] Committing antigen:")

    commit_result = memory.commit_antigen(
        antigen,
        source_dataset="M4_TEST",
        validated=True,
    )

    print(commit_result)

    assert commit_result["status"] == "NEW_MEMORY_CREATED"

    print(
        "Memory count:",
        memory.count_memories(),
    )

    assert memory.count_memories() == 1

    # --------------------------------------------------------------
    # Second exposure
    # --------------------------------------------------------------

    print("\n[4] Second exposure:")

    result = memory.recognize_threat(
        vector
    )

    print("Classification:", result["classification"])
    print("Similarity:", result["similarity"])

    assert result["classification"] == "KNOWN"

    # --------------------------------------------------------------
    # Re-exposure
    # --------------------------------------------------------------

    print("\n[5] Recording re-exposure:")

    # Attach anomaly_score so ImmuneMemory records the updated score
    result["anomaly_score"] = 0.98

    reexposure = memory.record_reexposure(
        result
    )

    print(reexposure)

    assert reexposure["hit_count"] == 2

    # --------------------------------------------------------------
    # Close and reopen
    # --------------------------------------------------------------

    print("\n[6] Closing memory...")

    memory.close()

    print("[7] Reopening memory from disk...")

    memory2 = ImmuneMemory(
        storage_path=test_path,
        embedding_dim=32,
        known_threshold=0.92,
        near_threshold=0.80,
        encoder_version="behavioral_encoder_v1",
        feature_schema_hash="TEST_SCHEMA_001",
    )

    print(
        "Persistent memory count:",
        memory2.count_memories(),
    )

    assert memory2.count_memories() == 1

    # --------------------------------------------------------------
    # Persistence verification
    # --------------------------------------------------------------

    print("\n[8] Searching after restart:")

    result = memory2.recognize_threat(
        vector
    )

    print("Classification:", result["classification"])
    print("Similarity:", result["similarity"])
    print("Memory ID:", result["memory_id"])

    assert result["classification"] == "KNOWN"

    print("\n" + "=" * 75)
    print("[ADIS] M4 persistence test PASSED")
    print("=" * 75)


if __name__ == "__main__":
    main()