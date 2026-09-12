from src.context.security_context import (
    DetectionContext,
    RecognitionContext,
    BehaviorContext,
    ThreatContext,
    SecurityContext,
)


def main():
    print("=" * 80)
    print("[ADIS] M5.1 — Security Context Test")
    print("=" * 80)

    context = SecurityContext(
        detection=DetectionContext(
            anomaly_score=0.9776,
            detector_decision="ANOMALY",
            detector_confidence=0.9776,
            latency_ms=12.5,
        ),

        recognition=RecognitionContext(
            classification="NOVEL",
            similarity=0.61,
            policy_classification="NOVEL",
            policy_action="ISOLATE",
            policy_confidence=0.43,
            memory_id=None,
            memory_status="NO_MATCH",
        ),

        behavior=BehaviorContext(
            behavior_family="anomalous_network_behavior",
            evidence={
                "flow_packet_rate": 1234.0,
                "flow_byte_rate": 500000.0,
                "forward_packets": 100.0,
                "backward_packets": 80.0,
            },
            embedding_dimension=32,
            encoder_version="adis-behavioral-encoder-v2",
            feature_schema_hash="aee55fd98f19d60a0bcab57ca3607eff852e2e95c315f5207ab3d66a715ed320",
        ),

        threat=ThreatContext(
            technique_id="UNMAPPED",
            technique_name="Unmapped anomalous network behavior",
            tactic="UNKNOWN",
            severity="CRITICAL",
            attribution_status="behavioral_candidate",
            technique_confidence="not_attributed",
        ),

        source_dataset="CIC-IDS2017",
    )

    print("\n[1] Security context created")

    assert context.detection.anomaly_score == 0.9776
    assert context.recognition.classification == "NOVEL"
    assert context.behavior.embedding_dimension == 32
    assert context.threat.severity == "CRITICAL"

    print("    ✓ Detection context")
    print("    ✓ Recognition context")
    print("    ✓ Behavior context")
    print("    ✓ Threat context")

    print("\n[2] Summary:")
    print(f"    {context.summary()}")

    print("\n[3] JSON-safe representation:")

    data = context.to_dict()

    assert data["detection"]["anomaly_score"] == 0.9776
    assert data["recognition"]["policy_action"] == "ISOLATE"
    assert data["behavior"]["encoder_version"] == "adis-behavioral-encoder-v2"

    print("    ✓ to_dict()")
    print("    ✓ Nested structure")
    print("    ✓ Encoder provenance")
    print("    ✓ Recognition information")

    print("\n" + "=" * 80)
    print("[ADIS] M5.1 Security Context PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()