from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class InvestigationReport:
    investigation_id: str
    status: str
    behavior_family: str | None
    threat_type: str | None
    detector_decision: str
    recognition_classification: str
    similarity: float
    risk_score: float
    severity: str
    evidence: list[dict[str, Any]]
    recommended_action: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ThreatInvestigator:
    """
    M6 Investigation Layer.

    This component does not detect attacks and does not make
    independent security decisions.

    It aggregates evidence already produced by:
        M2 Detection
        M3 Behavioral Analysis
        M4 Immune Memory
        Risk Engine

    and creates a structured investigation report.
    """

    VERSION = "adis-investigator-v2"

    def investigate(self, pipeline_result: dict[str, Any]) -> InvestigationReport:
        context = pipeline_result.get("context", {})

        detection = context.get("detection", {})
        behavior = context.get("behavior", {})
        threat = context.get("threat", {})
        recognition = context.get("recognition", {})

        risk = pipeline_result.get("risk", {})
        decision = pipeline_result.get("decision", {})

        detector_decision = str(
            detection.get("detector_decision", "UNKNOWN")
        )

        recognition_classification = str(
            recognition.get("classification", "UNKNOWN")
        )

        similarity = self._safe_float(
            recognition.get("similarity", 0.0)
        )

        risk_score = self._safe_float(
            risk.get("risk_score", 0.0)
        )

        severity = str(
            risk.get("severity", "UNKNOWN")
        )

        behavior_family = (
            threat.get("behavior_family")
            or behavior.get("behavior_family")
        )

        threat_type = (
            threat.get("threat_type")
            or behavior.get("threat_type")
        )

        recommended_action = str(
            decision.get("action", "INVESTIGATE")
        )

        evidence = self._build_evidence(
            detection=detection,
            behavior=behavior,
            threat=threat,
            recognition=recognition,
            risk=risk,
        )

        status = self._determine_status(
            detector_decision=detector_decision,
            recognition_classification=recognition_classification,
        )

        explanation = self._build_explanation(
            detector_decision=detector_decision,
            recognition_classification=recognition_classification,
            similarity=similarity,
            severity=severity,
            recommended_action=recommended_action,
        )

        investigation_id = self._build_id(
            pipeline_result=pipeline_result,
            recognition=recognition,
        )

        return InvestigationReport(
            investigation_id=investigation_id,
            status=status,
            behavior_family=behavior_family,
            threat_type=threat_type,
            detector_decision=detector_decision,
            recognition_classification=recognition_classification,
            similarity=similarity,
            risk_score=risk_score,
            severity=severity,
            evidence=evidence,
            recommended_action=recommended_action,
            explanation=explanation,
        )

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _determine_status(
        detector_decision: str,
        recognition_classification: str,
    ) -> str:
        if detector_decision in {
            "ATTACK",
            "ANOMALY",
            "MALICIOUS",
            "SUSPICIOUS",
        }:
            if recognition_classification == "NOVEL":
                return "NOVEL_THREAT"

            if recognition_classification == "UNCERTAIN":
                return "UNCERTAIN_THREAT"

            if recognition_classification == "KNOWN":
                return "KNOWN_THREAT"

        if detector_decision == "BENIGN":
            return "BENIGN_ACTIVITY"

        return "UNCLASSIFIED"

    @staticmethod
    def _build_evidence(
        detection: dict[str, Any],
        behavior: dict[str, Any],
        threat: dict[str, Any],
        recognition: dict[str, Any],
        risk: dict[str, Any],
    ) -> list[dict[str, Any]]:

        evidence: list[dict[str, Any]] = []

        # 1. Detection Evidence
        detector_decision = detection.get("detector_decision")
        if detector_decision:
            evidence.append(
                {
                    "source": "M2_DETECTION",
                    "type": "detector_decision",
                    "value": detector_decision,
                }
            )

        attack_probability = detection.get("attack_probability")
        if attack_probability is not None:
            evidence.append(
                {
                    "source": "M2_DETECTION",
                    "type": "attack_probability",
                    "value": round(float(attack_probability), 4),
                }
            )

        # 2. Behavior & Threat Evidence
        behavior_family = (
            threat.get("behavior_family")
            or behavior.get("behavior_family")
        )
        if behavior_family:
            evidence.append(
                {
                    "source": "M3_BEHAVIOR",
                    "type": "behavior_family",
                    "value": behavior_family,
                }
            )

        threat_type = (
            threat.get("threat_type")
            or behavior.get("threat_type")
        )
        if threat_type:
            evidence.append(
                {
                    "source": "M3_BEHAVIOR",
                    "type": "threat_type",
                    "value": threat_type,
                }
            )

        # M6.4 - Extract underlying flow features as structured evidence
        raw_evidence = (
            threat.get("behavior_evidence") 
            or behavior.get("behavior_evidence")
            or threat.get("evidence")
            or behavior.get("evidence")
        )
        
        if isinstance(raw_evidence, dict):
            for k, v in raw_evidence.items():
                if isinstance(v, float):
                    v = round(v, 4)
                evidence.append(
                    {
                        "source": "M3_FLOW_FEATURES",
                        "type": k,
                        "value": v,
                    }
                )

        # 3. Memory Evidence
        classification = recognition.get("classification")
        if classification:
            evidence.append(
                {
                    "source": "M4_MEMORY",
                    "type": "recognition",
                    "value": classification,
                }
            )

        similarity = recognition.get("similarity")
        if similarity is not None:
            evidence.append(
                {
                    "source": "M4_MEMORY",
                    "type": "similarity",
                    "value": round(float(similarity), 4),
                }
            )

        # 4. Risk Engine Evidence
        risk_score = risk.get("risk_score")
        if risk_score is not None:
            evidence.append(
                {
                    "source": "RISK_ENGINE",
                    "type": "risk_score",
                    "value": round(float(risk_score), 4),
                }
            )

        severity = risk.get("severity")
        if severity:
            evidence.append(
                {
                    "source": "RISK_ENGINE",
                    "type": "severity",
                    "value": severity,
                }
            )

        return evidence

    @staticmethod
    def _build_explanation(
        detector_decision: str,
        recognition_classification: str,
        similarity: float,
        severity: str,
        recommended_action: str,
    ) -> str:
        return (
            f"Detection={detector_decision}; "
            f"recognition={recognition_classification}; "
            f"similarity={similarity:.4f}; "
            f"severity={severity}; "
            f"recommended_action={recommended_action}."
        )

    @staticmethod
    def _build_id(
        pipeline_result: dict[str, Any],
        recognition: dict[str, Any],
    ) -> str:

        memory_id = recognition.get("matched_memory_id")

        if memory_id:
            return f"investigation-memory-{memory_id}"

        memory_event = pipeline_result.get("memory_event") or {}

        created_id = memory_event.get("memory_id")

        if created_id:
            return f"investigation-memory-{created_id}"

        return "investigation-ephemeral"