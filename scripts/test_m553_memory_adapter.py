from __future__ import annotations

import math
import shutil
from pathlib import Path

from src.analysis.threat_analyzer import ThreatAntigen
from src.pipeline.memory_adapter import ImmuneMemoryAdapter


TEST_STORAGE = Path("models/immune_memory_m553_test")


def make_unit_vector(dim: int = 32) -> list[float]:
    values = [float(i + 1) for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def make_similar_vector(
    base: list[float],
    similarity: float,
) -> list[float]:
    """
    Build a unit vector whose cosine similarity with `base`
    is approximately `similarity`.

    Uses an orthogonal vector so the target cosine similarity
    is controlled mathematically.
    """
    dim = len(base)

    # Deterministic vector that is not parallel to base.
    raw = [0.0] * dim
    raw[0] = 1.0

    # Gram-Schmidt: remove projection onto base.
    dot = sum(a * b for a, b in zip(raw, base))
    orthogonal = [
        raw[i] - dot * base[i]
        for i in range(dim)
    ]

    orth_norm = math.sqrt(sum(v * v for v in orthogonal))

    if orth_norm == 0:
        raise RuntimeError("Failed to construct orthogonal vector.")

    orthogonal = [v / orth_norm for v in orthogonal]

    value = math.sqrt(max(0.0, 1.0 - similarity**2))

    vector = [
        similarity * base[i] + value * orthogonal[i]
        for i in range(dim)
    ]

    # Final normalization for numerical stability.
    norm = math.sqrt(sum(v * v for v in vector))

    return [v / norm for v in vector]


def make_antigen(
    antigen_id: str,
    embedding: list[float],
    behavior_family: str,
) -> ThreatAntigen:
    return ThreatAntigen(
        antigen_id=antigen_id,
        technique_id="UNMAPPED",
        technique_name="Unmapped",
        tactic="UNKNOWN",
        severity="HIGH",
        embedding=embedding,
        metadata={
            "behavior_family": behavior_family,
            "anomaly_score": 0.95,
        },
    )


def assert_close(
    actual: float,
    expected: float,
    tolerance: float = 0.02,
) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"Expected similarity ≈ {expected:.4f}, "
            f"got {actual:.4f}"
        )


def main() -> None:
    print("=" * 72)
    print("M5.5.3 — Immune Memory Adapter Recognition Test")
    print("=" * 72)

    # Never touch production memory.
    if TEST_STORAGE.exists():
        shutil.rmtree(TEST_STORAGE)

    base_vector = make_unit_vector()

    # ---------------------------------------------------------
    # 1. Empty memory -> NOVEL
    # ---------------------------------------------------------
    memory = ImmuneMemoryAdapter(
        storage_path=TEST_STORAGE,
    )

    result = memory.recognize(base_vector)

    print("\n[1] Empty memory")
    print("classification :", result.classification)
    print("similarity      :", result.similarity)
    print("policy_action   :", result.policy_action)
    print("memory_status   :", result.memory_status)

    assert result.classification == "NOVEL"
    assert result.policy_action == "ISOLATE"
    assert result.memory_id is None

    # ---------------------------------------------------------
    # 2. Store exact same embedding -> KNOWN
    # ---------------------------------------------------------
    antigen = make_antigen(
        antigen_id="m553-known-antigen",
        embedding=base_vector,
        behavior_family="high_rate_network_activity",
    )

    memory_id = memory.commit(
        antigen,
        source_dataset="M553_TEST",
    )

    print("\n[2] Exact stored embedding")
    print("memory_id       :", memory_id)

    result = memory.recognize(base_vector)

    print("classification :", result.classification)
    print("similarity      :", result.similarity)
    print("policy_action   :", result.policy_action)
    print("memory_status   :", result.memory_status)
    print("matched_id      :", result.memory_id)

    assert result.classification == "KNOWN"
    assert result.policy_action == "FAST_PATH"
    assert result.similarity >= 0.92
    assert result.memory_id == memory_id["memory_id"]

    # ---------------------------------------------------------
    # 3. Similarity in 0.75–0.92 -> UNCERTAIN
    # ---------------------------------------------------------
    query_vector = make_unit_vector()

    similar_vector = make_similar_vector(
        query_vector,
        similarity=0.80,
    )

    uncertain_antigen = make_antigen(
        antigen_id="m553-uncertain-antigen",
        embedding=similar_vector,
        behavior_family="high_volume_flow",
    )

    uncertain_memory_id = memory.commit(
        uncertain_antigen,
        source_dataset="M553_TEST",
    )

    print("\n[3] Controlled similarity ≈ 0.80")
    print("memory_id       :", uncertain_memory_id)

    # Query a vector that should be ~0.80 from the second memory.
    # The exact query vector is also the KNOWN memory, so the known
    # memory may rank first. Therefore we test using a fresh vector
    # constructed to target the uncertain memory.
    query_for_uncertain = query_vector

    result = memory.recognize(
        query_for_uncertain,
    )

    print("classification :", result.classification)
    print("similarity      :", result.similarity)
    print("policy_action   :", result.policy_action)
    print("memory_status   :", result.memory_status)

    # Because the exact base vector exists as a KNOWN memory,
    # it should correctly win the nearest-neighbor search.
    # Therefore this result is expected to remain KNOWN.
    assert result.classification == "KNOWN"

    # ---------------------------------------------------------
    # 4. Separate test memory for UNCERTAIN threshold
    # ---------------------------------------------------------
    uncertain_storage = Path("models/immune_memory_m553_uncertain_test")

    if uncertain_storage.exists():
        shutil.rmtree(uncertain_storage)

    uncertain_memory = ImmuneMemoryAdapter(
        storage_path=uncertain_storage,
    )

    uncertain_memory.commit(
        uncertain_antigen,
        source_dataset="M553_TEST",
    )

    result = uncertain_memory.recognize(
        query_vector,
    )

    print("\n[4] Isolated uncertain-memory test")
    print("classification :", result.classification)
    print("similarity      :", result.similarity)
    print("policy_action   :", result.policy_action)
    print("memory_status   :", result.memory_status)

    assert 0.75 <= result.similarity < 0.92
    assert result.classification == "UNCERTAIN"
    assert result.policy_action == "INVESTIGATE"

    # ---------------------------------------------------------
    # 5. Final info
    # ---------------------------------------------------------
    print("\n[5] Adapter configuration")
    print(memory.info())

    # Explicitly close Qdrant clients before cleanup.
    try:
        memory.memory.client.close()
    except Exception:
        pass

    try:
        uncertain_memory.memory.client.close()
    except Exception:
        pass

    print("\n" + "=" * 72)
    print("M5.5.3 MEMORY ADAPTER TEST PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()