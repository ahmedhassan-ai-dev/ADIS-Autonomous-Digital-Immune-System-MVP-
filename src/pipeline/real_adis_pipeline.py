from __future__ import annotations

import time
from typing import Any, Dict, Optional

import pandas as pd

from src.detection.lightgbm_detector import LightGBMDetector
from src.pipeline.encoder_adapter import BehavioralEncoderAdapter
from src.pipeline.memory_adapter import ImmuneMemoryAdapter
from src.investigation.investigator import ThreatInvestigator
from src.analysis.threat_analyzer import BehavioralThreatAnalyzer
from src.analysis.risk_engine import RiskEngine
from src.decision.decision_engine import DecisionEngine
from src.context.security_context import (
    SecurityContext,
    DetectionContext,
    RecognitionContext,
    BehaviorContext,
    ThreatContext,
)

from src.pipeline.contracts import (
    PipelineTrace,
)


class RealADISPipeline:
    """
    Production-oriented ADIS MVP pipeline.

    Flow:

        raw flow
            ↓
        LightGBM detector
            ↓
        Behavioral Encoder v2
            ↓
        Threat Analyzer
            ↓
        Immune Memory
            ↓
        Security Context
            ↓
        Risk Engine
            ↓
        Decision Engine
    """

    def __init__(
        self,
        detector: Optional[LightGBMDetector] = None,
        encoder: Optional[BehavioralEncoderAdapter] = None,
        memory: Optional[ImmuneMemoryAdapter] = None,
        analyzer: Optional[BehavioralThreatAnalyzer] = None,
        risk_engine: Optional[RiskEngine] = None,
        decision_engine: Optional[DecisionEngine] = None,
    ) -> None:

        self.detector = detector or LightGBMDetector()

        self.encoder = encoder or BehavioralEncoderAdapter()

        self.memory = memory or ImmuneMemoryAdapter()

        self.analyzer = analyzer or BehavioralThreatAnalyzer()

        self.risk_engine = risk_engine or RiskEngine()

        self.decision_engine = decision_engine or DecisionEngine()

        self.investigator = ThreatInvestigator()

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_flow(
        flow: Dict[str, Any] | pd.Series | pd.DataFrame,
    ) -> Dict[str, Any]:

        if isinstance(flow, dict):
            return dict(flow)

        if isinstance(flow, pd.Series):
            return flow.to_dict()

        if isinstance(flow, pd.DataFrame):
            if len(flow) != 1:
                raise ValueError(
                    "RealADISPipeline expects exactly one flow."
                )

            return flow.iloc[0].to_dict()

        raise TypeError(
            "flow must be dict, pandas.Series, or single-row DataFrame."
        )

    # ------------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------------

    def _detect(
        self,
        flow: Dict[str, Any],
    ) -> DetectionContext:

        start = time.perf_counter()

        result = self.detector.detect_one(flow)

        latency_ms = (time.perf_counter() - start) * 1000.0

        anomaly_score = float(
            result.get(
                "anomaly_score",
                result.get("attack_probability", 0.0),
            )
        )

        decision = str(
            result.get(
                "decision",
                "ANOMALY" if anomaly_score >= 0.5 else "BENIGN",
            )
        )

        is_anomaly = bool(
            result.get(
                "is_anomaly",
                anomaly_score >= 0.5,
            )
        )

        return DetectionContext(
            anomaly_score=anomaly_score,
            detector_decision=decision,
            detector_confidence=anomaly_score,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Behavior representation
    # ------------------------------------------------------------------

    def _encode(
        self,
        flow: Dict[str, Any],
    ):
        embedding = self.encoder.encode(flow)

        # Initial conservative behavior family.
        # More specific behavioral classification will be
        # improved later from analyzer evidence.
        behavior_family = "anomalous_network_behavior"

        evidence = [
            "CIC-IDS2017 flow representation",
            "Behavioral Encoder v2",
        ]

        return self.encoder.build_result(
            flow=flow,
            behavior_family=behavior_family,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Threat analysis
    # ------------------------------------------------------------------

    def _analyze(
        self,
        flow: Dict[str, Any],
        detection: DetectionContext,
    ):
        return self.analyzer.analyze_and_extract(
            flow_data=flow,
            anomaly_score=detection.anomaly_score,
            source_dataset="CIC-IDS2017",
        )

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def _recognize(
        self,
        embedding,
    ) -> RecognitionContext:

        result = self.memory.recognize(embedding)

        return RecognitionContext(
            classification=result.classification,
            similarity=result.similarity,
            policy_classification=result.policy_classification,
            policy_action=result.policy_action,
            policy_confidence=result.policy_confidence,
            memory_id=result.memory_id,
            memory_status=result.memory_status,
        )

    # ------------------------------------------------------------------
    # Full context
    # ------------------------------------------------------------------

    def _build_context(
        self,
        flow: Dict[str, Any],
        detection: DetectionContext,
        recognition: RecognitionContext,
        behavior_result,
        antigen,
    ) -> SecurityContext:

        behavior = BehaviorContext(
            behavior_family=behavior_result.behavior_family,
            evidence=behavior_result.evidence,
            embedding_dimension=behavior_result.embedding_dimension,
            encoder_version=behavior_result.encoder_version,
            feature_schema_hash=behavior_result.feature_schema_hash,
        )

        threat = ThreatContext(
            behavior_family=antigen.behavior_family,
            threat_type=antigen.threat_type,
            classification_confidence=(
                antigen.metadata.get(
                    "classification_confidence"
                )
                if isinstance(antigen.metadata, dict)
                else None
            ),
            classification_reason=(
                antigen.metadata.get(
                    "classification_reason"
                )
                if isinstance(antigen.metadata, dict)
                else None
            ),
            behavior_evidence=(
                antigen.metadata.get(
                    "behavior_evidence"
                )
                if isinstance(antigen.metadata, dict)
                else None
            ),
            metadata=(
                antigen.metadata
                if isinstance(antigen.metadata, dict)
                else {}
            ),
            technique_id=antigen.technique_id,
            technique_name=antigen.technique_name,
            tactic=antigen.tactic,
            severity=antigen.severity,
            attribution_status=(
                antigen.metadata.get(
                    "attribution_status",
                    "UNKNOWN",
                )
                if isinstance(antigen.metadata, dict)
                else "UNKNOWN"
            ),
            technique_confidence=(
                antigen.metadata.get(
                    "technique_confidence",
                    0.0,
                )
                if isinstance(antigen.metadata, dict)
                else 0.0
            ),
        )

        return SecurityContext(
            detection=detection,
            recognition=recognition,
            behavior=behavior,
            threat=threat,
            source_dataset="CIC-IDS2017",
            flow_metadata={
                "flow_keys": list(flow.keys()),
            },
        )

    # ------------------------------------------------------------------
    # Novel memory learning
    # ------------------------------------------------------------------

    def _learn_if_novel(
        self,
        recognition: RecognitionContext,
        detection: DetectionContext,
        antigen,
    ) -> Optional[Dict[str, Any]]:
        """
        Learn novel behavior only when:
            1. The behavior is NOVEL.
            2. The detector classified the flow as suspicious.

        Benign novel traffic must not be enrolled into Immune Memory.
        """

        if recognition.classification != "NOVEL":
            return None

        detector_decision = detection.detector_decision.upper()

        suspicious = detector_decision in {
            "ATTACK",
            "ANOMALY",
            "MALICIOUS",
            "SUSPICIOUS",
        }

        if not suspicious:
            return {
                "status": "LEARNING_SKIPPED",
                "reason": "detector_did_not_confirm_suspicious_behavior",
                "detector_decision": detection.detector_decision,
                "anomaly_score": detection.anomaly_score,
            }

        return self.memory.commit(
            antigen,
            source_dataset="CIC-IDS2017",
            validated=True,
        )

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def process(
        self,
        flow: Dict[str, Any] | pd.Series | pd.DataFrame,
    ) -> Dict[str, Any]:

        total_start = time.perf_counter()

        normalized_flow = self._normalize_flow(flow)

        # --------------------------------------------------------------
        # 1. Detection
        # --------------------------------------------------------------

        detection = self._detect(normalized_flow)

        # --------------------------------------------------------------
        # 2. Behavioral representation
        # --------------------------------------------------------------

        behavior_result = self._encode(normalized_flow)

        # --------------------------------------------------------------
        # 3. Threat analysis
        # --------------------------------------------------------------

        antigen = self._analyze(
            normalized_flow,
            detection,
        )

        # --------------------------------------------------------------
        # 4. Immune Memory recognition
        # --------------------------------------------------------------

        recognition = self._recognize(
            behavior_result.embedding,
        )

        # --------------------------------------------------------------
        # 5. Build unified security context
        # --------------------------------------------------------------

        context = self._build_context(
            normalized_flow,
            detection,
            recognition,
            behavior_result,
            antigen,
        )

        # --------------------------------------------------------------
        # 6. Risk assessment
        # --------------------------------------------------------------

        risk = self.risk_engine.assess(context)

        # --------------------------------------------------------------
        # 7. Security decision
        # --------------------------------------------------------------

        decision = self.decision_engine.decide(
            context,
            risk,
        )

        # --------------------------------------------------------------
        # 8. Immune learning
        # --------------------------------------------------------------

        memory_event = self._learn_if_novel(
            recognition,
            detection,
            antigen,
        )

        # --------------------------------------------------------------
        # 9. Total latency
        # --------------------------------------------------------------

        total_latency_ms = (
            time.perf_counter() - total_start
        ) * 1000.0

        result = {
            "context": context.to_dict(),
            "risk": risk.to_dict(),
            "decision": decision.to_dict(),
            "memory_event": memory_event,
            "latency_ms": total_latency_ms,
        }

        investigation = self.investigator.investigate(result)

        result["investigation"] = investigation.to_dict()

        return result

    # ------------------------------------------------------------------
    # Component information
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:

        return {
            "component": "RealADISPipeline",
            "detector": self.detector.info(),
            "encoder": self.encoder.info(),
            "memory": self.memory.info(),
            "risk_engine": {
                "component": "RiskEngine",
            },
            "decision_engine": {
                "component": "DecisionEngine",
            },
        }