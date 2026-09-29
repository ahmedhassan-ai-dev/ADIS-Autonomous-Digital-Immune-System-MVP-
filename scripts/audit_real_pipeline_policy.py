from pathlib import Path
import shutil

import pandas as pd

from src.pipeline.real_adis_pipeline import RealADISPipeline
from src.pipeline.memory_adapter import ImmuneMemoryAdapter
from src.detection.lightgbm_detector import LightGBMDetector
from src.context.security_context import (
    SecurityContext,
    DetectionContext,
    RecognitionContext,
    BehaviorContext,
    ThreatContext,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cicids2017"
    / "Bruteforce-Tuesday-no-metadata.parquet"
)

MEMORY_PATH = (
    PROJECT_ROOT
    / "models"
    / "immune_memory_m555_policy_test"
)


def print_case(title, result):
    context = result["context"]

    detection = context["detection"]
    recognition = context["recognition"]
    threat = context["threat"]
    risk = result["risk"]
    decision = result["decision"]

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print("Detector Decision :", detection["detector_decision"])
    print("Anomaly Score     :", detection["anomaly_score"])

    print("Recognition       :", recognition["classification"])
    print("Similarity        :", recognition["similarity"])
    print("Policy Action     :", recognition["policy_action"])

    print("Behavior Family   :", context["behavior"]["behavior_family"])
    print("Threat Severity   :", threat["severity"])

    print("Risk Score        :", risk["risk_score"])
    print("Risk Severity     :", risk["severity"])

    print("Decision Action   :", decision["action"])
    print("Decision Confidence:", decision["confidence"])

    print("Memory Event      :", result["memory_event"])
    print("Latency (ms)      :", result["latency_ms"])


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(
            f"{message}\nExpected: {expected}\nActual:   {actual}"
        )


def main():
    print("=" * 70)
    print("[ADIS] M5.5.5 — Real Pipeline Policy Audit")
    print("=" * 70)

    # ---------------------------------------------------------
    # Clean test memory
    # ---------------------------------------------------------

    if MEMORY_PATH.exists():
        print(f"[ADIS] Removing previous policy-test memory: {MEMORY_PATH}")
        shutil.rmtree(MEMORY_PATH)

    print("[ADIS] Loading dataset...")
    df = pd.read_parquet(DATASET_PATH)

    print("[ADIS] Rows:", len(df))

    detector = LightGBMDetector()

    memory = ImmuneMemoryAdapter(
        storage_path=str(MEMORY_PATH)
    )

    pipeline = RealADISPipeline(
        memory=memory
    )

    print("[ADIS] Initial memory count:", memory.count_memories())

    # =========================================================
    # CASE 1
    # BENIGN + NOVEL
    # =========================================================

    benign_flow = None

    for _, row in df.iterrows():
        flow = row.to_dict()

        detection = detector.detect_one(flow)

        if detection["decision"].upper() == "BENIGN":
            benign_flow = flow
            break

    if benign_flow is None:
        raise RuntimeError("Could not find BENIGN flow.")

    result_benign_1 = pipeline.process(benign_flow)

    print_case(
        "CASE 1 — BENIGN + NOVEL",
        result_benign_1,
    )

    context = result_benign_1["context"]

    assert_equal(
        context["detection"]["detector_decision"],
        "BENIGN",
        "Case 1 detector decision mismatch.",
    )

    assert_equal(
        context["recognition"]["classification"],
        "NOVEL",
        "Case 1 should start as NOVEL.",
    )

    assert_equal(
        result_benign_1["memory_event"]["status"],
        "LEARNING_SKIPPED",
        "Case 1 must not enter Immune Memory.",
    )

    assert_equal(
        memory.count_memories(),
        0,
        "Benign behavior must not create memory.",
    )

    print("[PASS] BENIGN + NOVEL does not poison memory.")

    # =========================================================
    # CASE 2
    # BENIGN repeated
    # =========================================================

    result_benign_2 = pipeline.process(benign_flow)

    print_case(
        "CASE 2 — BENIGN REPEATED",
        result_benign_2,
    )

    context = result_benign_2["context"]

    assert_equal(
        context["recognition"]["classification"],
        "NOVEL",
        "Benign behavior should remain NOVEL because it was not learned.",
    )

    assert_equal(
        memory.count_memories(),
        0,
        "Repeated benign behavior must not create memory.",
    )

    print("[PASS] BENIGN remains outside Immune Memory.")

    # =========================================================
    # Find ATTACK
    # =========================================================

    attack_flow = None
    attack_dataset_label = None
    attack_index = None

    for index, row in df.iterrows():
        flow = row.to_dict()

        detection = detector.detect_one(flow)

        label = str(flow.get("Label", "")).strip().lower()
        if (
            detection["decision"].upper() == "ATTACK"
            and label != "benign"
        ):
            attack_flow = flow
            attack_index = index
            attack_dataset_label = flow.get("Label")
            break

    if attack_flow is None:
        raise RuntimeError("Could not find detector-confirmed ATTACK flow.")

    print("\n[ADIS] Attack candidate:")
    print("Index :", attack_index)
    print("Label :", attack_dataset_label)

    # =========================================================
    # CASE 3
    # ATTACK + NOVEL
    # =========================================================

    result_attack_1 = pipeline.process(attack_flow)

    print_case(
        "CASE 3 — ATTACK + NOVEL",
        result_attack_1,
    )

    context = result_attack_1["context"]

    assert_equal(
        context["detection"]["detector_decision"],
        "ATTACK",
        "Case 3 detector decision mismatch.",
    )

    assert_equal(
        context["recognition"]["classification"],
        "NOVEL",
        "Case 3 should be NOVEL on first exposure.",
    )

    assert_equal(
        result_attack_1["memory_event"]["status"],
        "NEW_MEMORY_CREATED",
        "Case 3 should create Immune Memory.",
    )

    assert_equal(
        memory.count_memories(),
        1,
        "Case 3 should create exactly one memory.",
    )

    assert_equal(
        result_attack_1["decision"]["action"],
        "ISOLATE",
        "Novel attack must be isolated.",
    )

    print("[PASS] ATTACK + NOVEL → LEARN + ISOLATE.")

    # =========================================================
    # CASE 4
    # ATTACK + KNOWN
    # =========================================================

    result_attack_2 = pipeline.process(attack_flow)

    print_case(
        "CASE 4 — ATTACK + KNOWN",
        result_attack_2,
    )

    context = result_attack_2["context"]

    assert_equal(
        context["detection"]["detector_decision"],
        "ATTACK",
        "Case 4 detector decision mismatch.",
    )

    assert_equal(
        context["recognition"]["classification"],
        "KNOWN",
        "Case 4 should be KNOWN.",
    )

    if context["recognition"]["similarity"] < 0.92:
        raise AssertionError(
            "Case 4 similarity should be >= 0.92."
        )

    if result_attack_2["memory_event"] is not None:
        raise AssertionError(
            "Case 4 must not create duplicate memory."
        )

    assert_equal(
        memory.count_memories(),
        1,
        "Known attack must not create duplicate memory.",
    )

    print("[PASS] ATTACK + KNOWN → no duplicate memory.")

    # =========================================================
    # POLICY AUDIT
    # =========================================================

    known_attack_action = result_attack_2["decision"]["action"]
    known_attack_risk = result_attack_2["risk"]["risk_score"]

    print("\n" + "=" * 70)
    print("POLICY AUDIT — KNOWN ATTACK")
    print("=" * 70)

    print("Detector :", context["detection"]["detector_decision"])
    print("Memory   :", context["recognition"]["classification"])
    print("Risk     :", known_attack_risk)
    print("Action   :", known_attack_action)

    if known_attack_action == "ALLOW":
        print(
            "\n[REVIEW REQUIRED]"
            "\nKNOWN + ATTACK currently results in ALLOW."
            "\nThis is a policy decision that should be explicitly justified."
        )
    else:
        print(
            "\n[INFO]"
            f"\nKNOWN + ATTACK currently results in {known_attack_action}."
        )

    # =========================================================
    # CASE 5
    # ATTACK + KNOWN + HIGH RISK
    #
    # We do not fabricate a result.
    # We test the existing deterministic DecisionEngine directly
    # with the actual context/risk contract.
    # =========================================================

    from src.analysis.risk_engine import RiskAssessment
    from src.decision.decision_engine import DecisionEngine

    decision_engine = DecisionEngine()

    synthetic_high_risk = RiskAssessment(
        risk_score=0.90,
        severity="CRITICAL",
        confidence=1.0,
        risk_factors=[
            "known_attack_high_risk_policy_audit"
        ],
    )

    raw_context = result_attack_2["context"]
    high_risk_context = SecurityContext(
        detection=DetectionContext(**raw_context["detection"]),
        recognition=RecognitionContext(**raw_context["recognition"]),
        behavior=BehaviorContext(**raw_context["behavior"]),
        threat=ThreatContext(**raw_context["threat"]),
        source_dataset=raw_context["source_dataset"],
        flow_metadata=raw_context["flow_metadata"],
    )

    high_risk_decision = decision_engine.decide(
        high_risk_context,
        synthetic_high_risk,
    )

    print("\n" + "=" * 70)
    print("CASE 5 — KNOWN ATTACK + HIGH RISK")
    print("=" * 70)

    print("Recognition :", high_risk_context.recognition.classification)
    print("Risk Score  :", synthetic_high_risk.risk_score)
    print("Severity    :", synthetic_high_risk.severity)
    print("Decision    :", high_risk_decision)

    if high_risk_decision.action != "INVESTIGATE":
        raise AssertionError(
            "Known high-risk behavior should result in INVESTIGATE."
        )

    print("[PASS] KNOWN + HIGH RISK → INVESTIGATE.")

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print("\n" + "=" * 70)
    print("[PASS] M5.5.5 Real Pipeline Policy Audit")
    print("=" * 70)

    print("✓ BENIGN + NOVEL        → no memory learning")
    print("✓ BENIGN repeated       → remains NOVEL")
    print("✓ ATTACK + NOVEL        → ISOLATE + LEARN")
    print("✓ ATTACK + KNOWN        → no duplicate memory")
    print("✓ KNOWN + HIGH RISK     → INVESTIGATE")
    print("✓ Memory count          →", memory.count_memories())

    print("\n[IMPORTANT]")
    print(
        "KNOWN + ATTACK + current risk level produced:",
        known_attack_action,
    )

    print("=" * 70)


if __name__ == "__main__":
    main()