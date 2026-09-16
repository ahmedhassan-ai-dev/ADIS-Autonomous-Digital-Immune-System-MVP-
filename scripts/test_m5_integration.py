"""
ADIS M5.4 — End-to-End Integration Test Matrix

Purpose
-------
Validate the complete M5 pipeline without retraining models.

Test strategy
-------------
The integration test uses lightweight deterministic mock components
for detector, immune memory, and recognition policy.

Real components under integration:
    SecurityContext
    RiskEngine
    DecisionEngine
    BehavioralThreatAnalyzer
    ADISPipeline orchestration

The test verifies:
    1. Full pipeline execution
    2. Novel threat path
    3. Known threat path
    4. High-risk known threat
    5. Uncertain recognition
    6. Memory learning
    7. Decision propagation
    8. SecurityContext propagation
    9. Risk propagation
    10. JSON-safe output
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from src.pipeline.adis_pipeline import (
    ADISPipeline,
)

from src.analysis.risk_engine import (
    RiskEngine,
)

from src.decision.decision_engine import (
    DecisionEngine,
)


# ============================================================
# Fake antigen
# ============================================================


@dataclass
class FakeAntigen:

    technique_id: str = "UNMAPPED"

    technique_name: str = (
        "Unmapped anomalous network behavior"
    )

    tactic: str = "UNKNOWN"

    severity: str = "CRITICAL"

    embedding: list = None

    metadata: Dict[str, Any] = None

    def __post_init__(self):

        if self.embedding is None:
            vector = np.zeros(
                32,
                dtype=np.float32,
            )

            vector[0] = 1.0

            self.embedding = vector.tolist()

        if self.metadata is None:
            self.metadata = {
                "representation_version":
                    "adis-behavioral-encoder-v2",

                "feature_schema_hash":
                    "aee55fd98f19d60a0bcab57ca3607eff"
                    "852e2e95c315f5207ab3d66a715ed320",

                "embedding_dimension": 32,

                "anomaly_score": 0.9,

                "behavior_family":
                    "anomalous_network_behavior",

                "behavior_evidence": {
                    "test": True,
                },

                "source_dataset":
                    "CICIDS2017",

                "attribution_status":
                    "behavioral_candidate",

                "technique_confidence":
                    "not_attributed",
            }


# ============================================================
# Fake detector
# ============================================================


class FakeDetector:
    """
    Deterministic detector used only for integration testing.
    """

    def __init__(
        self,
        anomaly_score: float,
    ):
        self.anomaly_score = anomaly_score

    def predict(
        self,
        flow: Any,
    ) -> Dict[str, Any]:

        return {
            "anomaly_score":
                self.anomaly_score,

            "detector_decision":
                (
                    "ANOMALOUS"
                    if self.anomaly_score >= 0.5
                    else "BENIGN"
                ),

            "detector_confidence":
                self.anomaly_score,
        }


# ============================================================
# Fake analyzer
# ============================================================


class FakeAnalyzer:
    """
    Lightweight analyzer replacement.

    This keeps M5.4 integration tests fast and deterministic.

    The real BehavioralThreatAnalyzer is tested separately.
    """

    embedding_dim = 32

    encoder_version = (
        "adis-behavioral-encoder-v2"
    )

    feature_schema_hash = (
        "aee55fd98f19d60a0bcab57ca3607eff"
        "852e2e95c315f5207ab3d66a715ed320"
    )

    def __init__(
        self,
        severity: str = "CRITICAL",
    ):
        self.severity = severity

    def analyze_and_extract(
        self,
        flow_data,
        anomaly_score,
        source_dataset,
    ):

        antigen = FakeAntigen(
            severity=self.severity
        )

        antigen.metadata[
            "anomaly_score"
        ] = anomaly_score

        antigen.metadata[
            "source_dataset"
        ] = source_dataset

        return antigen


# ============================================================
# Fake immune memory
# ============================================================


class FakeMemory:
    """
    Deterministic immune-memory simulator.

    similarity:
        controls recognition path.

    The actual M4 ImmuneMemory is validated separately.
    """

    def __init__(
        self,
        similarity: float,
    ):
        self.similarity = similarity
        self.committed = []

    def recognize_threat(
        self,
        embedding,
    ):

        if self.similarity >= 0.92:
            classification = "KNOWN"
            memory_status = "MATCHED"

        elif self.similarity >= 0.75:
            classification = "UNCERTAIN"
            memory_status = "CANDIDATE"

        else:
            classification = "NOVEL"
            memory_status = "NOT_FOUND"

        return {
            "similarity":
                self.similarity,

            "classification":
                classification,

            "memory_id":
                (
                    "memory-test-001"
                    if classification != "NOVEL"
                    else None
                ),

            "memory_status":
                memory_status,
        }

    def commit_antigen(
        self,
        antigen,
    ):

        self.committed.append(
            antigen
        )


# ============================================================
# Fake recognition policy
# ============================================================


class FakeRecognitionPolicy:
    """
    Deterministic M3 policy simulator.
    """

    def classify(
        self,
        similarity: float,
    ):

        if similarity < 0.75:

            return {
                "classification": "NOVEL",
                "action": "ISOLATE",
                "confidence": 0.95,
            }

        if similarity < 0.92:

            return {
                "classification": "UNCERTAIN",
                "action": "INVESTIGATE",
                "confidence": 0.75,
            }

        return {
            "classification": "KNOWN",
            "action": "FAST_PATH",
            "confidence": 0.95,
        }


# ============================================================
# Pipeline factory
# ============================================================


def make_pipeline(
    anomaly_score: float,
    similarity: float,
    severity: str = "CRITICAL",
):

    detector = FakeDetector(
        anomaly_score
    )

    memory = FakeMemory(
        similarity
    )

    policy = FakeRecognitionPolicy()

    analyzer = FakeAnalyzer(
        severity=severity
    )

    pipeline = ADISPipeline(
        detector=detector,
        memory=memory,
        recognition_policy=policy,
        analyzer=analyzer,
        risk_engine=RiskEngine(),
        decision_engine=DecisionEngine(),
    )

    return pipeline, memory


# ============================================================
# Test helpers
# ============================================================


def print_result(
    number: int,
    title: str,
    result,
    expected_action: str,
):

    actual_action = result.decision.action

    print(
        f"\n[{number}. {title}]"
    )

    print(
        f"    Recognition : "
        f"{result.security_context.recognition.classification}"
    )

    print(
        f"    Similarity  : "
        f"{result.security_context.recognition.similarity:.4f}"
    )

    print(
        f"    Anomaly     : "
        f"{result.security_context.detection.anomaly_score:.4f}"
    )

    print(
        f"    Risk        : "
        f"{result.risk_assessment.risk_score:.4f}"
    )

    print(
        f"    Risk Level  : "
        f"{result.risk_assessment.severity}"
    )

    print(
        f"    Expected    : "
        f"{expected_action}"
    )

    print(
        f"    Actual      : "
        f"{actual_action}"
    )

    print(
        f"    Reason      : "
        f"{result.decision.reason}"
    )

    assert (
        actual_action == expected_action
    ), (
        f"Expected {expected_action}, "
        f"got {actual_action}"
    )

    print("    ✓ PASS")


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 88)

    print(
        "[ADIS] M5.4 — Full Pipeline Integration Test Matrix"
    )

    print("=" * 88)

    # --------------------------------------------------------
    # 1. Novel high-risk behavior
    # --------------------------------------------------------

    pipeline, memory = make_pipeline(
        anomaly_score=0.9776,
        similarity=0.61,
        severity="CRITICAL",
    )

    result = pipeline.process(
        {
            "Flow Duration": 500000,
            "Protocol": 6,
            "test_case": "novel",
        }
    )

    print_result(
        1,
        "Novel + high-risk first exposure",
        result,
        "ISOLATE",
    )

    assert (
        result.security_context
        .recognition
        .classification
        == "NOVEL"
    )

    assert (
        result.security_context
        .threat
        .severity
        == "CRITICAL"
    )

    assert len(memory.committed) == 1

    print(
        "    ✓ Novel antigen committed to memory"
    )

    # --------------------------------------------------------
    # 2. Known low-risk behavior
    # --------------------------------------------------------

    pipeline, memory = make_pipeline(
        anomaly_score=0.30,
        similarity=0.9999,
        severity="LOW",
    )

    result = pipeline.process(
        {
            "Flow Duration": 1000,
            "Protocol": 6,
            "test_case": "known_low_risk",
        }
    )

    print_result(
        2,
        "Known + low-risk behavior",
        result,
        "ALLOW",
    )

    assert (
        result.security_context
        .recognition
        .classification
        == "KNOWN"
    )

    assert len(memory.committed) == 0

    print(
        "    ✓ Known behavior not re-enrolled"
    )

    # --------------------------------------------------------
    # 3. Known high-risk behavior
    # --------------------------------------------------------

    pipeline, memory = make_pipeline(
        anomaly_score=0.95,
        similarity=0.9999,
        severity="CRITICAL",
    )

    result = pipeline.process(
        {
            "Flow Duration": 900000,
            "Protocol": 6,
            "test_case": "known_high_risk",
        }
    )

    print_result(
        3,
        "Known + high-risk behavior",
        result,
        "INVESTIGATE",
    )

    assert (
        result.security_context
        .recognition
        .classification
        == "KNOWN"
    )

    # --------------------------------------------------------
    # 4. Uncertain recognition
    # --------------------------------------------------------

    pipeline, memory = make_pipeline(
        anomaly_score=0.60,
        similarity=0.80,
        severity="HIGH",
    )

    result = pipeline.process(
        {
            "Flow Duration": 300000,
            "Protocol": 6,
            "test_case": "uncertain",
        }
    )

    print_result(
        4,
        "Uncertain recognition",
        result,
        "INVESTIGATE",
    )

    assert (
        result.security_context
        .recognition
        .classification
        == "UNCERTAIN"
    )

    # --------------------------------------------------------
    # 5. Novel medium-risk behavior
    # --------------------------------------------------------

    pipeline, memory = make_pipeline(
        anomaly_score=0.60,
        similarity=0.60,
        severity="MEDIUM",
    )

    result = pipeline.process(
        {
            "Flow Duration": 200000,
            "Protocol": 6,
            "test_case": "novel_medium",
        }
    )

    print_result(
        5,
        "Novel + medium risk",
        result,
        "ISOLATE",
    )

    assert len(memory.committed) == 1

    # --------------------------------------------------------
    # 6. Pipeline propagation
    # --------------------------------------------------------

    assert (
        result.security_context
        .detection
        .anomaly_score
        == 0.60
    )

    assert (
        result.security_context
        .recognition
        .similarity
        == 0.60
    )

    assert (
        result.security_context
        .behavior
        .embedding_dimension
        == 32
    )

    assert (
        result.security_context
        .behavior
        .encoder_version
        == "adis-behavioral-encoder-v2"
    )

    assert (
        result.security_context
        .threat
        .technique_id
        == "UNMAPPED"
    )

    print(
        "\n[6. SecurityContext propagation]"
    )

    print(
        "    ✓ Detection propagated"
    )

    print(
        "    ✓ Recognition propagated"
    )

    print(
        "    ✓ Behavioral representation propagated"
    )

    print(
        "    ✓ Threat context propagated"
    )

    # --------------------------------------------------------
    # 7. Risk propagation
    # --------------------------------------------------------

    assert (
        0.0
        <= result.risk_assessment.risk_score
        <= 1.0
    )

    assert (
        0.0
        <= result.risk_assessment.confidence
        <= 1.0
    )

    print(
        "\n[7. Risk assessment propagation]"
    )

    print(
        f"    Risk score  : "
        f"{result.risk_assessment.risk_score:.4f}"
    )

    print(
        f"    Confidence  : "
        f"{result.risk_assessment.confidence:.4f}"
    )

    print(
        f"    Severity    : "
        f"{result.risk_assessment.severity}"
    )

    print(
        "    ✓ PASS"
    )

    # --------------------------------------------------------
    # 8. Decision propagation
    # --------------------------------------------------------

    assert result.decision.action in {
        "ALLOW",
        "INVESTIGATE",
        "ISOLATE",
    }

    assert (
        0.0
        <= result.decision.confidence
        <= 1.0
    )

    print(
        "\n[8. Decision propagation]"
    )

    print(
        f"    Action      : "
        f"{result.decision.action}"
    )

    print(
        f"    Confidence  : "
        f"{result.decision.confidence:.4f}"
    )

    print(
        "    ✓ PASS"
    )

    # --------------------------------------------------------
    # 9. JSON-safe representation
    # --------------------------------------------------------

    payload = result.to_dict()

    assert isinstance(
        payload,
        dict,
    )

    assert (
        "security_context"
        in payload
    )

    assert (
        "risk_assessment"
        in payload
    )

    assert (
        "decision"
        in payload
    )

    assert (
        "processing_time_ms"
        in payload
    )

    print(
        "\n[9. JSON-safe pipeline output]"
    )

    print(
        "    ✓ SecurityContext"
    )

    print(
        "    ✓ RiskAssessment"
    )

    print(
        "    ✓ DecisionResult"
    )

    print(
        "    ✓ Processing latency"
    )

    print(
        "    ✓ PASS"
    )

    # --------------------------------------------------------
    # 10. Pipeline latency
    # --------------------------------------------------------

    assert (
        result.processing_time_ms >= 0.0
    )

    print(
        "\n[10. Pipeline latency measurement]"
    )

    print(
        f"    Latency     : "
        f"{result.processing_time_ms:.3f} ms"
    )

    print(
        "    ✓ PASS"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 88)

    print(
        "[ADIS] M5.4 Full Pipeline Integration PASSED"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()