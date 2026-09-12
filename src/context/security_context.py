from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DetectionContext:
    """
    Output produced by the anomaly detection layer.
    """

    anomaly_score: Optional[float] = None
    detector_decision: Optional[str] = None
    detector_confidence: Optional[float] = None
    latency_ms: Optional[float] = None


@dataclass
class RecognitionContext:
    """
    Output produced by immune-memory recognition and policy.
    """

    classification: str = "UNKNOWN"
    similarity: Optional[float] = None

    policy_classification: Optional[str] = None
    policy_action: Optional[str] = None
    policy_confidence: Optional[float] = None

    memory_id: Optional[str] = None
    memory_status: Optional[str] = None


@dataclass
class BehaviorContext:
    """
    Behavioral interpretation produced by BehavioralThreatAnalyzer.
    """

    behavior_family: str = "unknown"
    evidence: Dict[str, Any] = field(default_factory=dict)

    embedding_dimension: Optional[int] = None
    encoder_version: Optional[str] = None
    feature_schema_hash: Optional[str] = None


@dataclass
class ThreatContext:
    """
    Security-oriented threat interpretation.

    This object deliberately separates:
      - detection
      - recognition
      - behavioral interpretation
      - threat attribution

    from final decision making.
    """

    technique_id: str = "UNMAPPED"
    technique_name: str = "Unmapped anomalous network behavior"
    tactic: str = "UNKNOWN"

    severity: str = "UNKNOWN"

    attribution_status: str = "behavioral_candidate"
    technique_confidence: str = "not_attributed"

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityContext:
    """
    Unified security context consumed by M5 decision intelligence.

    Pipeline:

        Detection
            +
        Recognition
            +
        Behavioral Analysis
            +
        Threat Context
            ↓
        SecurityContext
    """

    detection: DetectionContext = field(default_factory=DetectionContext)

    recognition: RecognitionContext = field(
        default_factory=RecognitionContext
    )

    behavior: BehaviorContext = field(
        default_factory=BehaviorContext
    )

    threat: ThreatContext = field(
        default_factory=ThreatContext
    )

    source_dataset: Optional[str] = None

    flow_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the complete security context to a JSON-safe dictionary.
        """

        return {
            "detection": {
                "anomaly_score": self.detection.anomaly_score,
                "detector_decision": self.detection.detector_decision,
                "detector_confidence": self.detection.detector_confidence,
                "latency_ms": self.detection.latency_ms,
            },

            "recognition": {
                "classification": self.recognition.classification,
                "similarity": self.recognition.similarity,
                "policy_classification": self.recognition.policy_classification,
                "policy_action": self.recognition.policy_action,
                "policy_confidence": self.recognition.policy_confidence,
                "memory_id": self.recognition.memory_id,
                "memory_status": self.recognition.memory_status,
            },

            "behavior": {
                "behavior_family": self.behavior.behavior_family,
                "evidence": self.behavior.evidence,
                "embedding_dimension": self.behavior.embedding_dimension,
                "encoder_version": self.behavior.encoder_version,
                "feature_schema_hash": self.behavior.feature_schema_hash,
            },

            "threat": {
                "technique_id": self.threat.technique_id,
                "technique_name": self.threat.technique_name,
                "tactic": self.threat.tactic,
                "severity": self.threat.severity,
                "attribution_status": self.threat.attribution_status,
                "technique_confidence": self.threat.technique_confidence,
                "metadata": self.threat.metadata,
            },

            "source_dataset": self.source_dataset,
            "flow_metadata": self.flow_metadata,
        }

    def summary(self) -> str:
        """
        Human-readable security summary.
        """

        return (
            f"SecurityContext("
            f"anomaly={self.detection.anomaly_score}, "
            f"recognition={self.recognition.classification}, "
            f"similarity={self.recognition.similarity}, "
            f"behavior={self.behavior.behavior_family}, "
            f"severity={self.threat.severity}, "
            f"technique={self.threat.technique_id}"
            f")"
        )