from src.analysis.risk_engine import RiskEngine
from src.context.security_context import (
    SecurityContext,
    DetectionContext,
    RecognitionContext,
    BehaviorContext,
    ThreatContext,
)


def make_context(
    anomaly,
    classification,
    severity,
    confidence=0.0,
):
    return SecurityContext(
        detection=DetectionContext(
            anomaly_score=anomaly,
            detector_decision="ANOMALY",
            detector_confidence=anomaly,
        ),
        recognition=RecognitionContext(
            classification=classification,
            similarity=0.61,
            policy_classification=classification,
            policy_action=(
                "ISOLATE"
                if classification == "NOVEL"
                else "INVESTIGATE"
            ),
            policy_confidence=confidence,
        ),
        behavior=BehaviorContext(
            behavior_family="anomalous_network_behavior",
            evidence={"test": True},
            embedding_dimension=32,
            encoder_version="adis-behavioral-encoder-v2",
            feature_schema_hash=(
                "aee55fd98f19d60a0bcab57ca3607eff852e2e95c315f5207ab3d66a715ed320"
            ),
        ),
        threat=ThreatContext(
            technique_id="UNMAPPED",
            technique_name="Unmapped anomalous network behavior",
            tactic="UNKNOWN",
            severity=severity,
            attribution_status="behavioral_candidate",
            technique_confidence="not_attributed",
        ),
    )


def main():
    print("=" * 80)
    print("[ADIS] M5.2 — Risk Engine Test")
    print("=" * 80)

    engine = RiskEngine()

    # Test 1: Novel + Critical + Very High Anomaly
    context = make_context(
        anomaly=0.9776,
        classification="NOVEL",
        severity="CRITICAL",
        confidence=0.90,
    )

    result = engine.assess(context)

    print("\n[1] Novel high-risk behavior")
    print(f"    {result.summary()}")
    print(f"    Factors: {result.risk_factors}")

    assert 0.0 <= result.risk_score <= 1.0
    assert result.severity == "CRITICAL"
    assert "very_high_anomaly_score=0.9776" in result.risk_factors
    assert "critical_threat_severity" in result.risk_factors
    assert (
        "novel_behavior_not_found_in_immune_memory"
        in result.risk_factors
    )

    print("    ✓ High-risk assessment correct")

    # Test 2: Known + Low + Low anomaly
    context = make_context(
        anomaly=0.20,
        classification="KNOWN",
        severity="LOW",
        confidence=0.95,
    )

    result = engine.assess(context)

    print("\n[2] Known low-risk behavior")
    print(f"    {result.summary()}")

    assert result.risk_score < 0.50
    assert result.severity in ("LOW", "MEDIUM")
    assert (
        "behavior_matches_immune_memory"
        in result.risk_factors
    )

    print("    ✓ Low-risk assessment correct")

    # Test 3: Uncertain behavior
    context = make_context(
        anomaly=0.65,
        classification="UNCERTAIN",
        severity="MEDIUM",
        confidence=0.60,
    )

    result = engine.assess(context)

    print("\n[3] Uncertain behavior")
    print(f"    {result.summary()}")

    assert 0.0 <= result.risk_score <= 1.0
    assert (
        "uncertain_immune_memory_recognition"
        in result.risk_factors
    )

    print("    ✓ Uncertain assessment correct")

    # Test 4: Missing values
    context = SecurityContext()
    result = engine.assess(context)

    print("\n[4] Missing/unknown context")
    print(f"    {result.summary()}")

    assert 0.0 <= result.risk_score <= 1.0
    assert result.severity == "LOW"
    assert result.confidence >= 0.0

    print("    ✓ Missing-value handling correct")

    # Test 5: Weight validation
    try:
        RiskEngine(
            anomaly_weight=0.50,
            severity_weight=0.30,
            novelty_weight=0.10,
            confidence_weight=0.20,
        )

        raise AssertionError("Invalid weights were accepted")

    except ValueError:
        print("\n[5] Weight validation")
        print("    ✓ Invalid weight configuration rejected")

    # Test 6: JSON-safe representation
    data = result.to_dict()

    assert isinstance(data, dict)
    assert "risk_score" in data
    assert "severity" in data
    assert "confidence" in data
    assert "risk_factors" in data

    print("\n[6] JSON-safe representation")
    print("    ✓ to_dict()")

    print("\n" + "=" * 80)
    print("[ADIS] M5.2 Risk Engine PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()