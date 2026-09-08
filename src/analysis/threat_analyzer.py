from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import joblib


# ============================================================
# ADIS — M3 Behavioral Threat Representation
# ============================================================

@dataclass
class ThreatAntigen:
    antigen_id: str

    technique_id: str
    technique_name: str
    tactic: str
    severity: str

    embedding: List[float]

    metadata: Dict[str, Any] = field(default_factory=dict)


class BehavioralThreatAnalyzer:
    """
    M3 — Behavioral Threat Representation.

    Converts the raw CIC-IDS2017 feature vector into a stable
    behavioral antigen representation.

    Pipeline:

        Raw Flow Features
              ↓
        Robust Scaling
              ↓
        PCA Projection
              ↓
        L2 Normalization
              ↓
        Behavioral Antigen Vector
    """

    def __init__(
        self,
        encoder_path: Optional[str] = None,
        embedding_dim: int = 32,
        feature_names: Optional[List[str]] = None,
    ):
        self.embedding_dim = embedding_dim
        self.feature_names = feature_names

        self.scaler = None
        self.pca = None

        if encoder_path:
            artifact = joblib.load(encoder_path)

            self.scaler = artifact["scaler"]
            self.pca = artifact["pca"]

            self.feature_names = artifact["features"]
            self.embedding_dim = artifact["embedding_dim"]

    # ========================================================
    # Feature preparation
    # ========================================================

    def _prepare_features(
        self,
        raw_features: np.ndarray,
    ) -> np.ndarray:

        arr = np.asarray(raw_features, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        arr = np.nan_to_num(
            arr,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if self.feature_names is not None:
            expected_dim = len(self.feature_names)

            if arr.shape[1] != expected_dim:
                raise ValueError(
                    f"Feature dimension mismatch. "
                    f"Expected {expected_dim}, got {arr.shape[1]}"
                )

        return arr

    # ========================================================
    # Behavioral embedding
    # ========================================================

    def generate_antigen_vector(
        self,
        raw_features: np.ndarray,
    ) -> List[float]:

        if self.scaler is None or self.pca is None:
            raise RuntimeError(
                "Behavioral encoder is not loaded. "
                "Train and provide an encoder_path."
            )

        X = self._prepare_features(raw_features)

        # Robust scaling
        X_scaled = self.scaler.transform(X)

        # PCA behavioral projection
        embedding = self.pca.transform(X_scaled)

        # L2 normalization for cosine similarity
        norm = np.linalg.norm(
            embedding,
            axis=1,
            keepdims=True,
        )

        norm = np.maximum(norm, 1e-12)

        embedding = embedding / norm

        return embedding[0].astype(np.float32).tolist()

    # ========================================================
    # Behavioral characterization
    # ========================================================

    def _determine_behavior(
        self,
        flow_data: Dict[str, Any],
    ) -> Dict[str, Any]:

        def get(*names, default=0.0):
            for name in names:
                if name in flow_data:
                    value = flow_data[name]

                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return default

            return default

        fwd_packets = get(
            "Total Fwd Packets",
            "Total_Fwd_Packets",
        )

        bwd_packets = get(
            "Total Backward Packets",
            "Total_Backward_Packets",
        )

        flow_packets_s = get(
            "Flow Packets/s",
            "Flow_Packets/s",
        )

        flow_bytes_s = get(
            "Flow Bytes/s",
            "Flow_Bytes/s",
        )

        duration = get(
            "Flow Duration",
            "Flow_Duration",
        )

        syn = get(
            "SYN Flag Count",
            "SYN_Flag_Count",
        )

        rst = get(
            "RST Flag Count",
            "RST_Flag_Count",
        )

        ack = get(
            "ACK Flag Count",
            "ACK_Flag_Count",
        )

        packet_mean = get(
            "Packet Length Mean",
            "Packet_Length_Mean",
        )

        # ----------------------------------------------------
        # Derived behavioral indicators
        # ----------------------------------------------------

        total_packets = fwd_packets + bwd_packets

        asymmetry = (
            fwd_packets / max(bwd_packets, 1.0)
        )

        syn_ratio = (
            syn / max(fwd_packets, 1.0)
        )

        ack_ratio = (
            ack / max(total_packets, 1.0)
        )

        # ----------------------------------------------------
        # Heuristic behavior characterization
        # ----------------------------------------------------

        behavior = "GENERAL_SUSPICIOUS_TRAFFIC"

        confidence = 0.40

        # High-rate / volumetric behavior
        if (
            flow_packets_s > 10000
            or total_packets > 500
            or flow_bytes_s > 1e8
        ):
            behavior = "VOLUMETRIC_NETWORK_ACTIVITY"
            confidence = 0.80

        # SYN-heavy short flows
        elif (
            syn > 0
            and duration < 10000
            and syn_ratio > 0.2
        ):
            behavior = "CONNECTION_PROBING"
            confidence = 0.75

        # Strongly asymmetric traffic
        elif (
            asymmetry > 10
            or asymmetry < 0.1
        ):
            behavior = "TRAFFIC_ASYMMETRY"
            confidence = 0.60

        # Reset-heavy behavior
        elif rst > 0:
            behavior = "CONNECTION_RESET_ACTIVITY"
            confidence = 0.55

        return {
            "behavior_class": behavior,
            "behavior_confidence": confidence,
            "flow_packets_per_second": flow_packets_s,
            "flow_bytes_per_second": flow_bytes_s,
            "packet_count": total_packets,
            "packet_length_mean": packet_mean,
            "flow_duration": duration,
            "syn_count": syn,
            "rst_count": rst,
            "ack_count": ack,
            "traffic_asymmetry": asymmetry,
            "syn_ratio": syn_ratio,
            "ack_ratio": ack_ratio,
        }

    # ========================================================
    # ATT&CK interpretation
    # ========================================================

    def _map_to_attack_technique(
        self,
        behavior: Dict[str, Any],
    ) -> Dict[str, Any]:

        behavior_class = behavior["behavior_class"]

        if behavior_class == "CONNECTION_PROBING":
            return {
                "technique_id": "T1046",
                "technique_name": "Network Service Discovery",
                "tactic": "Discovery",
                "severity": "MEDIUM",
                "mapping_confidence": 0.75,
            }

        if behavior_class == "VOLUMETRIC_NETWORK_ACTIVITY":
            return {
                "technique_id": "T1498",
                "technique_name": "Network Denial of Service",
                "tactic": "Impact",
                "severity": "HIGH",
                "mapping_confidence": 0.80,
            }

        if behavior_class == "TRAFFIC_ASYMMETRY":
            return {
                "technique_id": "T1046",
                "technique_name": "Network Service Discovery",
                "tactic": "Discovery",
                "severity": "MEDIUM",
                "mapping_confidence": 0.45,
            }

        return {
            "technique_id": "UNKNOWN",
            "technique_name": "Unclassified Suspicious Network Behavior",
            "tactic": "Unknown",
            "severity": "MEDIUM",
            "mapping_confidence": 0.25,
        }

    # ========================================================
    # Main analysis
    # ========================================================

    def analyze_and_extract(
        self,
        antigen_id: str,
        flow_dict: Dict[str, Any],
        raw_features: np.ndarray,
        anomaly_score: float,
    ) -> ThreatAntigen:

        behavior = self._determine_behavior(flow_dict)

        technique = self._map_to_attack_technique(
            behavior
        )

        embedding = self.generate_antigen_vector(
            raw_features
        )

        metadata = {
            "anomaly_score": float(anomaly_score),

            "behavior_class":
                behavior["behavior_class"],

            "behavior_confidence":
                behavior["behavior_confidence"],

            "mapping_confidence":
                technique["mapping_confidence"],

            "protocol":
                flow_dict.get(
                    "Protocol",
                    flow_dict.get(
                        "protocol",
                        0,
                    ),
                ),

            **behavior,
        }

        return ThreatAntigen(
            antigen_id=antigen_id,

            technique_id=
                technique["technique_id"],

            technique_name=
                technique["technique_name"],

            tactic=
                technique["tactic"],

            severity=
                technique["severity"],

            embedding=embedding,

            metadata=metadata,
        )