from dataclasses import dataclass, field
from typing import List

from src.context.security_context import SecurityContext


@dataclass
class RiskAssessment:
    """
    Deterministic and explainable risk assessment.
    """

    risk_score: float
    severity: str
    confidence: float
    risk_factors: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "risk_score": self.risk_score,
            "severity": self.severity,
            "confidence": self.confidence,
            "risk_factors": self.risk_factors,
        }

    def summary(self) -> str:
        return (
            f"RiskAssessment("
            f"score={self.risk_score:.4f}, "
            f"severity={self.severity}, "
            f"confidence={self.confidence:.4f}, "
            f"factors={len(self.risk_factors)}"
            f")"
        )


class RiskEngine:
    """
    M5.2 — Deterministic Threat Risk Engine.

    Converts SecurityContext into an explainable risk assessment.

    This is an operational heuristic, not a calibrated probability
    of compromise.
    """

    SEVERITY_SCORES = {
        "UNKNOWN": 0.0,
        "LOW": 0.25,
        "MEDIUM": 0.50,
        "HIGH": 0.75,
        "CRITICAL": 1.00,
    }

    def __init__(
        self,
        anomaly_weight: float = 0.50,
        severity_weight: float = 0.30,
        novelty_weight: float = 0.10,
        confidence_weight: float = 0.10,
    ):
        total = (
            anomaly_weight
            + severity_weight
            + novelty_weight
            + confidence_weight
        )

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Risk weights must sum to 1.0, got {total:.6f}"
            )

        self.anomaly_weight = anomaly_weight
        self.severity_weight = severity_weight
        self.novelty_weight = novelty_weight
        self.confidence_weight = confidence_weight

    @staticmethod
    def _clamp(
        value: float,
        low: float = 0.0,
        high: float = 1.0,
    ) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _safe_float(
        value,
        default: float = 0.0,
    ) -> float:
        try:
            if value is None:
                return default

            value = float(value)

            if value != value:
                return default

            return value

        except (TypeError, ValueError):
            return default

    def _novelty_score(
        self,
        classification: str,
    ) -> float:

        classification = (
            classification or "UNKNOWN"
        ).upper()

        mapping = {
            "NOVEL": 1.00,
            "UNCERTAIN": 0.60,
            "KNOWN": 0.20,
            "UNKNOWN": 0.00,
        }

        return mapping.get(
            classification,
            0.00,
        )

    def _severity_from_score(
        self,
        score: float,
    ) -> str:

        if score < 0.25:
            return "LOW"

        if score < 0.50:
            return "MEDIUM"

        if score < 0.75:
            return "HIGH"

        return "CRITICAL"

    def assess(
        self,
        context: SecurityContext,
    ) -> RiskAssessment:

        # ---------------------------------------------------------
        # 1. Anomaly contribution
        # ---------------------------------------------------------

        anomaly = self._clamp(
            self._safe_float(
                context.detection.anomaly_score,
                default=0.0,
            )
        )

        # ---------------------------------------------------------
        # 2. Threat severity contribution
        # ---------------------------------------------------------

        threat_severity = (
            context.threat.severity or "UNKNOWN"
        ).upper()

        severity_score = self.SEVERITY_SCORES.get(
            threat_severity,
            0.0,
        )

        # ---------------------------------------------------------
        # 3. Recognition / novelty contribution
        # ---------------------------------------------------------

        recognition = (
            context.recognition.classification
            or context.recognition.policy_classification
            or "UNKNOWN"
        )

        novelty_score = self._novelty_score(
            recognition
        )

        # ---------------------------------------------------------
        # 4. Recognition confidence
        # ---------------------------------------------------------

        recognition_confidence = self._safe_float(
            context.recognition.policy_confidence,
            default=0.0,
        )

        recognition_confidence = self._clamp(
            recognition_confidence
        )

        # ---------------------------------------------------------
        # 5. Weighted risk score
        # ---------------------------------------------------------

        risk_score = (
            self.anomaly_weight * anomaly
            + self.severity_weight * severity_score
            + self.novelty_weight * novelty_score
            + self.confidence_weight
            * recognition_confidence
        )

        risk_score = self._clamp(
            risk_score
        )

        # ---------------------------------------------------------
        # 6. Explainability
        # ---------------------------------------------------------

        risk_factors = []

        if anomaly >= 0.90:

            risk_factors.append(
                f"very_high_anomaly_score={anomaly:.4f}"
            )

        elif anomaly >= 0.75:

            risk_factors.append(
                f"high_anomaly_score={anomaly:.4f}"
            )

        elif anomaly >= 0.50:

            risk_factors.append(
                f"elevated_anomaly_score={anomaly:.4f}"
            )

        if threat_severity == "CRITICAL":

            risk_factors.append(
                "critical_threat_severity"
            )

        elif threat_severity == "HIGH":

            risk_factors.append(
                "high_threat_severity"
            )

        elif threat_severity == "MEDIUM":

            risk_factors.append(
                "medium_threat_severity"
            )

        if recognition.upper() == "NOVEL":

            risk_factors.append(
                "novel_behavior_not_found_in_immune_memory"
            )

        elif recognition.upper() == "UNCERTAIN":

            risk_factors.append(
                "uncertain_immune_memory_recognition"
            )

        elif recognition.upper() == "KNOWN":

            risk_factors.append(
                "behavior_matches_immune_memory"
            )

        if context.threat.attribution_status != "attributed":

            risk_factors.append(
                "technique_attribution_not_confirmed"
            )

        # ---------------------------------------------------------
        # 7. Final severity
        # ---------------------------------------------------------

        final_severity = self._severity_from_score(
            risk_score
        )

        # ---------------------------------------------------------
        # 8. Confidence
        # ---------------------------------------------------------

        confidence_components = [
            anomaly,
            severity_score,
            recognition_confidence,
        ]

        confidence = (
            sum(confidence_components)
            / len(confidence_components)
        )

        confidence = self._clamp(
            confidence
        )

        return RiskAssessment(
            risk_score=risk_score,
            severity=final_severity,
            confidence=confidence,
            risk_factors=risk_factors,
        )