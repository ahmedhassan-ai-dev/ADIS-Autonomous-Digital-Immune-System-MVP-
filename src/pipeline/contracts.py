from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# ADIS Production Contracts
# M5.5.1
# ============================================================

CONTRACT_VERSION = "adis-contract-v1"

ENCODER_VERSION = "adis-behavioral-encoder-v3-supervised"
SUPPORTED_ENCODER_VERSIONS = {
    "adis-behavioral-encoder-v2",
    ENCODER_VERSION,
}
EMBEDDING_DIM = 32

KNOWN_THRESHOLD = 0.92
NEAR_THRESHOLD = 0.75

FEATURE_SCHEMA_HASH = (
    "aee55fd98f19d60a0bcab57ca3607eff852e2e95c315f5207ab3d66a715ed320"
)


@dataclass(frozen=True)
class DetectionResult:
    """
    Contract between the primary detector and the ADIS pipeline.

    The detector is responsible for supervised attack detection.
    It is NOT responsible for determining whether behavior is
    known or novel.
    """

    anomaly_score: float
    attack_probability: float
    decision: str
    is_anomaly: bool
    threshold: float
    detector_name: str = "LightGBM"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.anomaly_score) <= 1.0:
            raise ValueError("anomaly_score must be in [0, 1].")

        if not 0.0 <= float(self.attack_probability) <= 1.0:
            raise ValueError("attack_probability must be in [0, 1].")

        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("threshold must be in [0, 1].")


@dataclass(frozen=True)
class RecognitionResult:
    """
    Contract between Immune Memory and the recognition policy.
    """

    classification: str
    similarity: float
    policy_classification: str
    policy_action: str
    policy_confidence: float
    memory_id: Optional[str] = None
    memory_status: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not -1.0 <= float(self.similarity) <= 1.0:
            raise ValueError("similarity must be in [-1, 1].")

        if not 0.0 <= float(self.policy_confidence) <= 1.0:
            raise ValueError("policy_confidence must be in [0, 1].")


@dataclass(frozen=True)
class BehaviorResult:
    """
    Contract for the behavioral representation layer.
    """

    behavior_family: str
    evidence: List[str]
    embedding: List[float]
    embedding_dimension: int
    encoder_version: str
    feature_schema_hash: str

    def __post_init__(self) -> None:
        if len(self.embedding) != self.embedding_dimension:
            raise ValueError(
                "Embedding length does not match embedding_dimension."
            )

        if self.embedding_dimension != EMBEDDING_DIM:
            raise ValueError(
                f"ADIS production embedding must be {EMBEDDING_DIM}D."
            )

        if self.encoder_version not in SUPPORTED_ENCODER_VERSIONS:
            raise ValueError(
                f"Unexpected encoder version: {self.encoder_version}"
            )

        if self.feature_schema_hash != FEATURE_SCHEMA_HASH:
            raise ValueError(
                "Behavioral encoder feature schema hash mismatch."
            )


@dataclass(frozen=True)
class ThreatResult:
    """
    Contract for threat interpretation.
    """

    technique_id: str
    technique_name: str
    tactic: str
    severity: str
    attribution_status: str
    technique_confidence: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineInput:
    """
    Canonical input entering the production ADIS pipeline.

    Exactly one network flow is processed at a time in the MVP.
    """

    flow: Dict[str, Any]
    source_dataset: str
    flow_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.flow, dict):
            raise TypeError("PipelineInput.flow must be a dictionary.")

        if not self.source_dataset:
            raise ValueError("source_dataset cannot be empty.")


@dataclass
class PipelineTrace:
    """
    Operational trace of one ADIS decision cycle.

    This object is intentionally independent from the dashboard.
    The dashboard will consume this later.
    """

    contract_version: str = CONTRACT_VERSION

    detection: Optional[DetectionResult] = None
    recognition: Optional[RecognitionResult] = None
    behavior: Optional[BehaviorResult] = None
    threat: Optional[ThreatResult] = None

    risk: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None

    isolation_status: Optional[str] = None
    memory_commit_status: Optional[str] = None

    timings_ms: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "detection": (
                self.detection.__dict__
                if self.detection is not None
                else None
            ),
            "recognition": (
                self.recognition.__dict__
                if self.recognition is not None
                else None
            ),
            "behavior": (
                self.behavior.__dict__
                if self.behavior is not None
                else None
            ),
            "threat": (
                self.threat.__dict__
                if self.threat is not None
                else None
            ),
            "risk": self.risk,
            "decision": self.decision,
            "isolation_status": self.isolation_status,
            "memory_commit_status": self.memory_commit_status,
            "timings_ms": dict(self.timings_ms),
            "metadata": dict(self.metadata),
        }
