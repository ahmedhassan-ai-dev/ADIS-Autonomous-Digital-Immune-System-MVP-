"""
ADIS M5.4 — End-to-End Security Pipeline

Purpose
-------
Integrate the validated ADIS components into one deterministic
security-analysis pipeline.

Pipeline
--------
Raw Flow
    ↓
Anomaly Detection
    ↓
Behavioral Threat Analyzer
    ↓
Immune Memory
    ↓
Recognition Policy
    ↓
Security Context
    ↓
Risk Engine
    ↓
Decision Engine
    ↓
ALLOW / INVESTIGATE / ISOLATE

Important
---------
This module orchestrates existing components.

It does NOT:
    - train models
    - modify model artifacts
    - execute isolation
    - execute blocking
    - execute recovery
    - perform LLM reasoning
    - claim exact MITRE ATT&CK attribution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math
import time

import numpy as np
import pandas as pd

from src.context.security_context import (
    SecurityContext,
    DetectionContext,
    RecognitionContext,
    BehaviorContext,
    ThreatContext,
)

from src.analysis.risk_engine import (
    RiskEngine,
    RiskAssessment,
)

from src.decision.decision_engine import (
    DecisionEngine,
    DecisionResult,
)

from src.analysis.threat_analyzer import (
    BehavioralThreatAnalyzer,
)


@dataclass
class PipelineResult:
    """
    Complete output of the M5.4 ADIS pipeline.
    """

    security_context: SecurityContext
    risk_assessment: RiskAssessment
    decision: DecisionResult

    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-safe representation.
        """

        return {
            "security_context": self.security_context.to_dict(),
            "risk_assessment": self.risk_assessment.to_dict(),
            "decision": self.decision.to_dict(),
            "processing_time_ms": self.processing_time_ms,
        }

    def summary(self) -> str:
        """
        Human-readable pipeline summary.
        """

        return (
            "ADIS Pipeline Result("
            f"anomaly={self.security_context.detection.anomaly_score}, "
            f"recognition="
            f"{self.security_context.recognition.classification}, "
            f"risk={self.risk_assessment.risk_score:.4f}, "
            f"risk_severity={self.risk_assessment.severity}, "
            f"decision={self.decision.action}, "
            f"latency={self.processing_time_ms:.2f}ms"
            f")"
        )


class ADISPipeline:
    """
    End-to-end ADIS M5.4 orchestration layer.

    Existing components are injected where possible so that:

        - unit testing remains easy
        - components can be replaced independently
        - no duplicate decision logic is introduced
    """

    PIPELINE_VERSION = "adis-m5.4-pipeline-v1"

    def __init__(
        self,
        detector: Any,
        memory: Any,
        recognition_policy: Any,
        analyzer: Optional[BehavioralThreatAnalyzer] = None,
        risk_engine: Optional[RiskEngine] = None,
        decision_engine: Optional[DecisionEngine] = None,
    ) -> None:

        if detector is None:
            raise ValueError("detector must not be None")

        if memory is None:
            raise ValueError("memory must not be None")

        if recognition_policy is None:
            raise ValueError(
                "recognition_policy must not be None"
            )

        self.detector = detector
        self.memory = memory
        self.recognition_policy = recognition_policy

        self.analyzer = (
            analyzer
            if analyzer is not None
            else BehavioralThreatAnalyzer()
        )

        self.risk_engine = (
            risk_engine
            if risk_engine is not None
            else RiskEngine()
        )

        self.decision_engine = (
            decision_engine
            if decision_engine is not None
            else DecisionEngine()
        )

    # ============================================================
    # Utilities
    # ============================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Convert a value to finite float safely.
        """

        try:
            value = float(value)
        except Exception:
            return default

        if not math.isfinite(value):
            return default

        return value

    @staticmethod
    def _extract_value(
        obj: Any,
        *names: str,
        default: Any = None,
    ) -> Any:
        """
        Extract a value from either:
            - dict-like objects
            - object attributes
        """

        if obj is None:
            return default

        if isinstance(obj, dict):
            for name in names:
                if name in obj:
                    return obj[name]

        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)

        return default

    # ============================================================
    # Detector
    # ============================================================

    def _run_detector(
        self,
        flow: Any,
    ) -> Dict[str, Any]:
        """
        Run the anomaly detector.

        The detector is expected to expose one of:

            predict(flow)
            detect(flow)
            score(flow)

        Supported outputs include:
            scalar score
            dict
            object with anomaly_score
        """

        detector_output = None

        if hasattr(self.detector, "predict"):
            detector_output = self.detector.predict(flow)

        elif hasattr(self.detector, "detect"):
            detector_output = self.detector.detect(flow)

        elif hasattr(self.detector, "score"):
            detector_output = self.detector.score(flow)

        else:
            raise AttributeError(
                "Detector must provide one of: "
                "predict(), detect(), score()."
            )

        anomaly_score = self._extract_value(
            detector_output,
            "anomaly_score",
            "score",
            "confidence",
            default=None,
        )

        if anomaly_score is None:
            if isinstance(detector_output, (int, float)):
                anomaly_score = detector_output
            elif isinstance(detector_output, np.number):
                anomaly_score = float(detector_output)

        anomaly_score = self._safe_float(
            anomaly_score,
            default=0.0,
        )

        detector_decision = self._extract_value(
            detector_output,
            "decision",
            "detector_decision",
            "label",
            "prediction",
            default=None,
        )

        detector_confidence = self._extract_value(
            detector_output,
            "detector_confidence",
            "confidence",
            default=anomaly_score,
        )

        detector_confidence = self._safe_float(
            detector_confidence,
            default=anomaly_score,
        )

        return {
            "anomaly_score": anomaly_score,
            "detector_decision": (
                str(detector_decision)
                if detector_decision is not None
                else (
                    "ANOMALOUS"
                    if anomaly_score >= 0.5
                    else "BENIGN"
                )
            ),
            "detector_confidence": detector_confidence,
        }

    # ============================================================
    # Recognition
    # ============================================================

    def _run_memory_recognition(
        self,
        embedding: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Query immune memory.

        Expected memory API:

            recognize_threat(embedding)

        A compatibility fallback to search() is also supported.
        """

        if hasattr(
            self.memory,
            "recognize_threat",
        ):
            result = self.memory.recognize_threat(
                embedding
            )

        elif hasattr(
            self.memory,
            "search",
        ):
            result = self.memory.search(
                embedding
            )

        else:
            raise AttributeError(
                "Immune memory must provide "
                "recognize_threat() or search()."
            )

        similarity = self._extract_value(
            result,
            "similarity",
            "best_similarity",
            "score",
            default=0.0,
        )

        similarity = self._safe_float(
            similarity,
            default=0.0,
        )

        classification = self._extract_value(
            result,
            "classification",
            "status",
            "recognition",
            default=None,
        )

        memory_id = self._extract_value(
            result,
            "memory_id",
            "id",
            "point_id",
            default=None,
        )

        memory_status = self._extract_value(
            result,
            "memory_status",
            "status",
            default=None,
        )

        return {
            "raw_result": result,
            "similarity": similarity,
            "classification": (
                str(classification).upper()
                if classification is not None
                else "UNKNOWN"
            ),
            "memory_id": (
                str(memory_id)
                if memory_id is not None
                else None
            ),
            "memory_status": (
                str(memory_status)
                if memory_status is not None
                else None
            ),
        }

    # ============================================================
    # Recognition Policy
    # ============================================================

    def _run_recognition_policy(
        self,
        similarity: float,
    ) -> Dict[str, Any]:
        """
        Run the existing M3 recognition policy.

        Supported policy method names:

            classify(similarity)
            evaluate(similarity)
            recognize(similarity)
        """

        policy_output = None

        if hasattr(
            self.recognition_policy,
            "classify",
        ):
            policy_output = (
                self.recognition_policy.classify(
                    similarity
                )
            )

        elif hasattr(
            self.recognition_policy,
            "evaluate",
        ):
            policy_output = (
                self.recognition_policy.evaluate(
                    similarity
                )
            )

        elif hasattr(
            self.recognition_policy,
            "recognize",
        ):
            policy_output = (
                self.recognition_policy.recognize(
                    similarity
                )
            )

        else:
            raise AttributeError(
                "Recognition policy must provide "
                "classify(), evaluate(), or recognize()."
            )

        classification = self._extract_value(
            policy_output,
            "classification",
            "policy_classification",
            "decision",
            "status",
            default="UNKNOWN",
        )

        action = self._extract_value(
            policy_output,
            "action",
            "policy_action",
            default=None,
        )

        confidence = self._extract_value(
            policy_output,
            "confidence",
            "policy_confidence",
            default=0.0,
        )

        confidence = self._safe_float(
            confidence,
            default=0.0,
        )

        return {
            "raw_result": policy_output,
            "classification": str(
                classification
            ).upper(),
            "action": (
                str(action).upper()
                if action is not None
                else None
            ),
            "confidence": confidence,
        }

    # ============================================================
    # Security Context
    # ============================================================

    def _build_security_context(
        self,
        flow: Any,
        detector_output: Dict[str, Any],
        antigen: Any,
        memory_output: Dict[str, Any],
        policy_output: Dict[str, Any],
    ) -> SecurityContext:
        """
        Fuse outputs into SecurityContext.
        """

        anomaly_score = detector_output[
            "anomaly_score"
        ]

        detection = DetectionContext(
            anomaly_score=anomaly_score,
            detector_decision=detector_output[
                "detector_decision"
            ],
            detector_confidence=detector_output[
                "detector_confidence"
            ],
        )

        recognition = RecognitionContext(
            classification=memory_output[
                "classification"
            ],
            similarity=memory_output[
                "similarity"
            ],
            policy_classification=policy_output[
                "classification"
            ],
            policy_action=policy_output[
                "action"
            ],
            policy_confidence=policy_output[
                "confidence"
            ],
            memory_id=memory_output[
                "memory_id"
            ],
            memory_status=memory_output[
                "memory_status"
            ],
        )

        metadata = getattr(
            antigen,
            "metadata",
            {},
        ) or {}

        behavior = BehaviorContext(
            behavior_family=metadata.get(
                "behavior_family",
                "unknown",
            ),
            evidence=metadata.get(
                "behavior_evidence",
                {},
            ),
            embedding_dimension=metadata.get(
                "embedding_dimension",
                getattr(
                    self.analyzer,
                    "embedding_dim",
                    None,
                ),
            ),
            encoder_version=metadata.get(
                "representation_version",
                getattr(
                    self.analyzer,
                    "encoder_version",
                    None,
                ),
            ),
            feature_schema_hash=metadata.get(
                "feature_schema_hash",
                getattr(
                    self.analyzer,
                    "feature_schema_hash",
                    None,
                ),
            ),
        )

        threat = ThreatContext(
            technique_id=getattr(
                antigen,
                "technique_id",
                "UNMAPPED",
            ),
            technique_name=getattr(
                antigen,
                "technique_name",
                "Unmapped anomalous network behavior",
            ),
            tactic=getattr(
                antigen,
                "tactic",
                "UNKNOWN",
            ),
            severity=getattr(
                antigen,
                "severity",
                "UNKNOWN",
            ),
            attribution_status=metadata.get(
                "attribution_status",
                "behavioral_candidate",
            ),
            technique_confidence=metadata.get(
                "technique_confidence",
                "not_attributed",
            ),
            metadata=metadata,
        )

        flow_metadata: Dict[str, Any] = {}

        if isinstance(flow, pd.Series):
            flow_metadata = flow.to_dict()

        elif isinstance(flow, dict):
            flow_metadata = dict(flow)

        else:
            try:
                flow_metadata = dict(flow)
            except Exception:
                flow_metadata = {}

        return SecurityContext(
            detection=detection,
            recognition=recognition,
            behavior=behavior,
            threat=threat,
            source_dataset=metadata.get(
                "source_dataset"
            ),
            flow_metadata=flow_metadata,
        )

    # ============================================================
    # Antigen extraction
    # ============================================================

    def _analyze_threat(
        self,
        flow: Any,
        anomaly_score: float,
    ) -> Any:
        """
        Convert raw flow into ThreatAntigen.
        """

        return self.analyzer.analyze_and_extract(
            flow_data=flow,
            anomaly_score=anomaly_score,
            source_dataset=(
                "CICIDS2017"
            ),
        )

    # ============================================================
    # Memory learning
    # ============================================================

    def _commit_if_novel(
        self,
        antigen: Any,
        recognition: str,
    ) -> bool:
        """
        Commit novel behavior into immune memory.

        This is deliberately limited to NOVEL recognition.

        The method supports the established commit_antigen()
        interface and a compatibility fallback to commit().
        """

        if recognition != "NOVEL":
            return False

        if hasattr(
            self.memory,
            "commit_antigen",
        ):
            self.memory.commit_antigen(
                antigen
            )
            return True

        if hasattr(
            self.memory,
            "commit",
        ):
            embedding = np.asarray(
                antigen.embedding,
                dtype=np.float32,
            )

            metadata = getattr(
                antigen,
                "metadata",
                {},
            ) or {}

            self.memory.commit(
                embedding=embedding,
                metadata=metadata,
            )

            return True

        return False

    # ============================================================
    # Main pipeline
    # ============================================================

    def process(
        self,
        flow: Any,
        *,
        learn_novel: bool = True,
    ) -> PipelineResult:
        """
        Execute the complete ADIS pipeline.

        Parameters
        ----------
        flow:
            A single network-flow representation accepted by
            BehavioralThreatAnalyzer.

        learn_novel:
            If True, a NOVEL antigen may be committed to immune
            memory after the decision stage.

        Returns
        -------
        PipelineResult
        """

        start = time.perf_counter()

        # --------------------------------------------------------
        # 1. Detection
        # --------------------------------------------------------

        detector_output = self._run_detector(
            flow
        )

        anomaly_score = detector_output[
            "anomaly_score"
        ]

        # --------------------------------------------------------
        # 2. Threat analysis / behavioral encoding
        # --------------------------------------------------------

        antigen = self._analyze_threat(
            flow,
            anomaly_score,
        )

        embedding = np.asarray(
            antigen.embedding,
            dtype=np.float32,
        )

        # --------------------------------------------------------
        # 3. Immune memory
        # --------------------------------------------------------

        memory_output = (
            self._run_memory_recognition(
                embedding
            )
        )

        # --------------------------------------------------------
        # 4. Recognition policy
        # --------------------------------------------------------

        policy_output = (
            self._run_recognition_policy(
                memory_output["similarity"]
            )
        )

        # --------------------------------------------------------
        # 5. Security context
        # --------------------------------------------------------

        context = self._build_security_context(
            flow=flow,
            detector_output=detector_output,
            antigen=antigen,
            memory_output=memory_output,
            policy_output=policy_output,
        )

        # --------------------------------------------------------
        # 6. Risk assessment
        # --------------------------------------------------------

        risk_assessment = (
            self.risk_engine.assess(
                context
            )
        )

        # --------------------------------------------------------
        # 7. Decision
        # --------------------------------------------------------

        decision = (
            self.decision_engine.decide(
                context,
                risk_assessment,
            )
        )

        # --------------------------------------------------------
        # 8. Optional immune learning
        # --------------------------------------------------------

        memory_updated = False

        if learn_novel:
            memory_updated = (
                self._commit_if_novel(
                    antigen,
                    context.recognition.classification,
                )
            )

        # --------------------------------------------------------
        # 9. Final metadata
        # --------------------------------------------------------

        processing_time_ms = (
            time.perf_counter() - start
        ) * 1000.0

        context.flow_metadata[
            "pipeline_version"
        ] = self.PIPELINE_VERSION

        context.flow_metadata[
            "memory_updated"
        ] = memory_updated

        context.flow_metadata[
            "processing_time_ms"
        ] = processing_time_ms

        return PipelineResult(
            security_context=context,
            risk_assessment=risk_assessment,
            decision=decision,
            processing_time_ms=processing_time_ms,
        )