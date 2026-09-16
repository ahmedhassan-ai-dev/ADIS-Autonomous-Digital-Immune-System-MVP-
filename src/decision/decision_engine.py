from dataclasses import dataclass, field
from typing import List

from src.context.security_context import SecurityContext
from src.analysis.risk_engine import RiskAssessment


@dataclass
class DecisionResult:
    """
    Final deterministic response recommendation produced by M5.3.
    """

    action: str
    reason: str
    confidence: float
    risk_score: float
    recognition: str
    decision_factors: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
            "risk_score": self.risk_score,
            "recognition": self.recognition,
            "decision_factors": self.decision_factors,
        }

    def summary(self) -> str:
        return (
            f"DecisionResult("
            f"action={self.action}, "
            f"risk={self.risk_score:.4f}, "
            f"recognition={self.recognition}, "
            f"confidence={self.confidence:.4f}"
            f")"
        )


class DecisionEngine:
    """
    M5.3 — Deterministic Decision Engine.

    Combines:
        SecurityContext
        +
        RiskAssessment

    and produces one of:

        ALLOW
        INVESTIGATE
        ISOLATE

    This engine does NOT:
        - execute isolation
        - kill processes
        - block network traffic
        - modify endpoints
        - perform recovery

    It only recommends the next security action.
    """

    RISK_HIGH_THRESHOLD = 0.75

    VALID_ACTIONS = {
        "ALLOW",
        "INVESTIGATE",
        "ISOLATE",
    }

    VALID_RECOGNITIONS = {
        "NOVEL",
        "UNCERTAIN",
        "KNOWN",
        "UNKNOWN",
    }

    def __init__(
        self,
        high_risk_threshold: float = RISK_HIGH_THRESHOLD,
    ):
        if not 0.0 <= high_risk_threshold <= 1.0:
            raise ValueError(
                "high_risk_threshold must be between 0.0 and 1.0"
            )

        self.high_risk_threshold = high_risk_threshold

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

    def decide(
        self,
        context: SecurityContext,
        risk_assessment: RiskAssessment,
    ) -> DecisionResult:
        """
        Determine the recommended security action.

        Decision policy:

            NOVEL
                + explicit policy isolation -> ISOLATE
                + high risk                 -> ISOLATE
                + otherwise                 -> INVESTIGATE

            UNCERTAIN
                + high risk                 -> ISOLATE
                + otherwise                 -> INVESTIGATE

            KNOWN
                + high risk                 -> INVESTIGATE
                + otherwise                 -> ALLOW

            UNKNOWN
                + high risk                 -> INVESTIGATE
                + otherwise                 -> ALLOW
        """

        if context is None:
            raise ValueError("context must not be None")

        if risk_assessment is None:
            raise ValueError("risk_assessment must not be None")

        recognition = str(
            context.recognition.classification or "UNKNOWN"
        ).upper()

        if recognition not in self.VALID_RECOGNITIONS:
            recognition = "UNKNOWN"

        policy_action = str(
            context.recognition.policy_action or ""
        ).upper()

        risk_score = self._clamp(
            self._safe_float(risk_assessment.risk_score, default=0.0)
        )

        risk_confidence = self._clamp(
            self._safe_float(risk_assessment.confidence, default=0.0)
        )

        policy_confidence = self._clamp(
            self._safe_float(context.recognition.policy_confidence, default=0.0)
        )

        decision_factors: List[str] = []

        # ============================================================
        # NOVEL
        # ============================================================

        if recognition == "NOVEL":
            decision_factors.append("novel_behavior")

            # Safety override:
            # Recognition policy explicitly requests isolation.
            if policy_action == "ISOLATE":
                action = "ISOLATE"
                reason = (
                    "Novel behavior was not recognized by immune memory "
                    "and the recognition policy recommends isolation."
                )
                decision_factors.append("recognition_policy_isolation")
                confidence = max(policy_confidence, risk_confidence)

            elif risk_score >= self.high_risk_threshold:
                action = "ISOLATE"
                reason = (
                    "Novel behavior combined with high security risk "
                    "requires isolation."
                )
                decision_factors.append("high_security_risk")
                confidence = max(policy_confidence, risk_confidence)

            else:
                action = "INVESTIGATE"
                reason = (
                    "Novel behavior was not recognized by immune memory "
                    "and requires investigation before allowing it."
                )
                decision_factors.append("novel_behavior_requires_investigation")
                confidence = (policy_confidence + risk_confidence) / 2.0

        # ============================================================
        # UNCERTAIN
        # ============================================================

        elif recognition == "UNCERTAIN":
            decision_factors.append("uncertain_memory_recognition")

            if risk_score >= self.high_risk_threshold:
                action = "ISOLATE"
                reason = (
                    "Uncertain immune-memory recognition combined "
                    "with high security risk requires isolation."
                )
                decision_factors.append("high_security_risk")
                confidence = max(policy_confidence, risk_confidence)

            else:
                action = "INVESTIGATE"
                reason = (
                    "Immune-memory recognition is uncertain and "
                    "requires further investigation."
                )
                decision_factors.append("uncertain_recognition_requires_investigation")
                confidence = (policy_confidence + risk_confidence) / 2.0

        # ============================================================
        # KNOWN
        # ============================================================

        elif recognition == "KNOWN":
            decision_factors.append("behavior_matches_immune_memory")

            if risk_score >= self.high_risk_threshold:
                action = "INVESTIGATE"
                reason = (
                    "Known behavior has elevated security risk "
                    "and requires investigation."
                )
                decision_factors.append("high_security_risk")
                confidence = (policy_confidence + risk_confidence) / 2.0

            else:
                action = "ALLOW"
                reason = (
                    "Behavior is recognized by immune memory "
                    "and does not exceed the high-risk threshold."
                )
                decision_factors.append("recognized_low_or_medium_risk")
                confidence = risk_confidence

        # ============================================================
        # UNKNOWN / INVALID
        # ============================================================

        else:
            recognition = "UNKNOWN"
            decision_factors.append("unknown_recognition_state")

            if risk_score >= self.high_risk_threshold:
                action = "INVESTIGATE"
                reason = (
                    "Recognition state is unknown and security risk "
                    "is high, requiring investigation."
                )
                decision_factors.append("high_security_risk")
                confidence = (policy_confidence + risk_confidence) / 2.0

            else:
                action = "ALLOW"
                reason = (
                    "Recognition state is unknown but security risk "
                    "is below the high-risk threshold."
                )
                decision_factors.append("risk_below_high_threshold")
                confidence = risk_confidence

        return DecisionResult(
            action=action,
            reason=reason,
            confidence=self._clamp(confidence),
            risk_score=risk_score,
            recognition=recognition,
            decision_factors=decision_factors,
        )