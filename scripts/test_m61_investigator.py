from __future__ import annotations

from src.investigation.investigator import ThreatInvestigator


def main():
    investigator = ThreatInvestigator()

    # Synthetic pipeline result representing:
    # ATTACK -> NOVEL -> CRITICAL -> ISOLATE
    pipeline_result = {
        "context": {
            "detection": {
                "detector_decision": "ATTACK",
                "attack_probability": 0.9984,
            },
            "behavior": {
                "behavior_family": "FTP-Bruteforce",
                "threat_type": "Credential Attack",
            },
            "threat": {
                "behavior_family": "FTP-Bruteforce",
                "threat_type": "Credential Attack",
            },
            "recognition": {
                "classification": "NOVEL",
                "similarity": 0.0,
                "matched_memory_id": None,
            },
        },
        "risk": {
            "risk_score": 0.9992,
            "severity": "CRITICAL",
        },
        "decision": {
            "action": "ISOLATE",
        },
        "memory_event": {
            "status": "NEW_MEMORY_CREATED",
            "memory_id": "test-memory-001",
        },
    }

    report = investigator.investigate(pipeline_result)

    assert report.status == "NOVEL_THREAT"
    assert report.detector_decision == "ATTACK"
    assert report.recognition_classification == "NOVEL"
    assert report.severity == "CRITICAL"
    assert report.recommended_action == "ISOLATE"
    assert report.behavior_family == "FTP-Bruteforce"
    assert len(report.evidence) >= 5

    print("=" * 72)
    print("M6.1 — INVESTIGATOR TEST")
    print("=" * 72)
    print()

    print("Status          :", report.status)
    print("Behavior Family :", report.behavior_family)
    print("Threat Type     :", report.threat_type)
    print("Detection       :", report.detector_decision)
    print("Recognition     :", report.recognition_classification)
    print("Similarity      :", report.similarity)
    print("Risk Score      :", report.risk_score)
    print("Severity        :", report.severity)
    print("Action          :", report.recommended_action)

    print()
    print("Evidence:")
    for item in report.evidence:
        print(
            f"  [{item['source']}] "
            f"{item['type']} = {item['value']}"
        )

    print()
    print("Explanation:")
    print(report.explanation)

    print()
    print("[PASS] M6.1 Investigator")


if __name__ == "__main__":
    main()