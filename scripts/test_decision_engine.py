from src.analysis.risk_engine import RiskAssessment
from src.context.security_context import (
    SecurityContext,
    DetectionContext,
    RecognitionContext,
    BehaviorContext,
    ThreatContext,
)
from src.decision.decision_engine import DecisionEngine


def make_context(
    recognition,
    policy_action,
    policy_confidence=0.90,
):
    return SecurityContext(
        detection=DetectionContext(
            anomaly_score=0.80,
            detector_decision="ANOMALY",
            detector_confidence=0.90,
        ),
        recognition=RecognitionContext(
            classification=recognition,
            similarity=0.85,
            policy_classification=recognition,
            policy_action=policy_action,
            policy_confidence=policy_confidence,
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
            severity="HIGH",
            attribution_status="behavioral_candidate",
            technique_confidence="not_attributed",
        ),
    )


def make_risk(
    score,
    confidence=0.90,
):
    return RiskAssessment(
        risk_score=score,
        severity="HIGH" if score >= 0.50 else "LOW",
        confidence=confidence,
        risk_factors=[],
    )


def run_case(
    engine,
    name,
    recognition,
    policy_action,
    risk,
    expected_action,
):
    context = make_context(
        recognition=recognition,
        policy_action=policy_action,
    )

    result = engine.decide(
        context,
        risk,
    )

    print(f"\n[{name}]")
    print(f"    Recognition : {recognition}")
    print(f"    Policy      : {policy_action}")
    print(f"    Risk        : {risk.risk_score:.4f}")
    print(f"    Expected    : {expected_action}")
    print(f"    Actual      : {result.action}")
    print(f"    Reason      : {result.reason}")

    assert result.action == expected_action
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.risk_score <= 1.0

    print("    ✓ PASS")


def main():

    print("=" * 80)
    print("[ADIS] M5.3 — Decision Engine Test Matrix")
    print("=" * 80)

    engine = DecisionEngine()

    # =========================================================
    # NOVEL
    # =========================================================

    run_case(
        engine,
        "1. Novel + explicit isolation",
        "NOVEL",
        "ISOLATE",
        make_risk(0.40),
        "ISOLATE",
    )

    run_case(
        engine,
        "2. Novel + high risk",
        "NOVEL",
        "INVESTIGATE",
        make_risk(0.80),
        "ISOLATE",
    )

    run_case(
        engine,
        "3. Novel + medium risk",
        "NOVEL",
        "INVESTIGATE",
        make_risk(0.60),
        "INVESTIGATE",
    )

    # =========================================================
    # UNCERTAIN
    # =========================================================

    run_case(
        engine,
        "4. Uncertain + high risk",
        "UNCERTAIN",
        "INVESTIGATE",
        make_risk(0.80),
        "ISOLATE",
    )

    run_case(
        engine,
        "5. Uncertain + medium risk",
        "UNCERTAIN",
        "INVESTIGATE",
        make_risk(0.60),
        "INVESTIGATE",
    )

    run_case(
        engine,
        "6. Uncertain + low risk",
        "UNCERTAIN",
        "INVESTIGATE",
        make_risk(0.20),
        "INVESTIGATE",
    )

    # =========================================================
    # KNOWN
    # =========================================================

    run_case(
        engine,
        "7. Known + low risk",
        "KNOWN",
        "FAST_PATH",
        make_risk(0.20),
        "ALLOW",
    )

    run_case(
        engine,
        "8. Known + medium risk",
        "KNOWN",
        "FAST_PATH",
        make_risk(0.60),
        "ALLOW",
    )

    run_case(
        engine,
        "9. Known + high risk",
        "KNOWN",
        "FAST_PATH",
        make_risk(0.80),
        "INVESTIGATE",
    )

    # =========================================================
    # UNKNOWN
    # =========================================================

    run_case(
        engine,
        "10. Unknown + low risk",
        "UNKNOWN",
        "",
        make_risk(0.20),
        "ALLOW",
    )

    run_case(
        engine,
        "11. Unknown + high risk",
        "UNKNOWN",
        "",
        make_risk(0.80),
        "INVESTIGATE",
    )

    # =========================================================
    # BOUNDARY
    # =========================================================

    run_case(
        engine,
        "12. Novel exactly at threshold",
        "NOVEL",
        "INVESTIGATE",
        make_risk(0.75),
        "ISOLATE",
    )

    run_case(
        engine,
        "13. Known exactly at threshold",
        "KNOWN",
        "FAST_PATH",
        make_risk(0.75),
        "INVESTIGATE",
    )

    # =========================================================
    # Invalid recognition
    # =========================================================

    run_case(
        engine,
        "14. Invalid recognition + low risk",
        "SOMETHING_UNKNOWN",
        "",
        make_risk(0.20),
        "ALLOW",
    )

    run_case(
        engine,
        "15. Invalid recognition + high risk",
        "SOMETHING_UNKNOWN",
        "",
        make_risk(0.90),
        "INVESTIGATE",
    )

    # =========================================================
    # Missing confidence
    # =========================================================

    context = make_context(
        recognition="KNOWN",
        policy_action="FAST_PATH",
        policy_confidence=None,
    )

    result = engine.decide(
        context,
        make_risk(0.20, confidence=0.70),
    )

    print("\n[16. Missing policy confidence]")
    print(f"    Action      : {result.action}")
    print(f"    Confidence  : {result.confidence:.4f}")

    assert result.action == "ALLOW"
    assert 0.0 <= result.confidence <= 1.0

    print("    ✓ PASS")

    # =========================================================
    # Invalid threshold
    # =========================================================

    print("\n[17. Invalid threshold validation]")

    try:
        DecisionEngine(
            high_risk_threshold=1.5
        )

        raise AssertionError(
            "Invalid threshold was accepted"
        )

    except ValueError:
        print("    ✓ PASS")

    # =========================================================
    # JSON-safe output
    # =========================================================

    data = result.to_dict()

    print("\n[18. JSON-safe representation]")

    assert isinstance(data, dict)
    assert "action" in data
    assert "reason" in data
    assert "confidence" in data
    assert "risk_score" in data
    assert "recognition" in data
    assert "decision_factors" in data

    print("    ✓ PASS")

    print("\n" + "=" * 80)
    print("[ADIS] M5.3 Decision Engine PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()