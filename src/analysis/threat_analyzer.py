from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import hashlib
import json
import uuid

import joblib
import numpy as np
import pandas as pd


@dataclass
class ThreatAntigen:
    """
    Represents a behavioral antigen extracted from a network flow.

    The antigen contains:
    - a learned behavioral embedding
    - anomaly information
    - behavior-family information
    - encoder/schema provenance
    """

    antigen_id: str
    technique_id: str
    technique_name: str
    tactic: str
    severity: str
    embedding: List[float]

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def behavior_family(self) -> Optional[str]:
        return (
            self.metadata.get("behavior_family")
            if isinstance(self.metadata, dict)
            else None
        )

    @property
    def threat_type(self) -> Optional[str]:
        return (
            self.metadata.get("threat_type")
            if isinstance(self.metadata, dict)
            else None
        )


class BehavioralThreatAnalyzer:
    """
    Production-oriented analyzer compatible with the versioned behavioral artifact.

    Pipeline:

        Raw Flow
            |
            v
        Feature Validation
            |
            v
        Numeric Conversion
            |
            v
        Median Imputation
            |
            v
        log1p on configured log_features
            |
            v
        RobustScaler
            |
            v
        Whitened PCA
            |
            v
        L2 Normalization
            |
            v
        Behavioral Embedding
    """

    ENCODER_VERSION = "adis-behavioral-encoder-v3-supervised"

    def __init__(
        self,
        encoder_path: Optional[Union[str, Path]] = None,
        input_dim: Optional[int] = None,
        embedding_dim: int = 32,
    ):
        project_root = Path(__file__).resolve().parents[2]

        if encoder_path is None:
            encoder_path = project_root / "models" / "behavioral_encoder.joblib"

        self.encoder_path = Path(encoder_path)

        if not self.encoder_path.exists():
            raise FileNotFoundError(
                f"Behavioral encoder not found:\n{self.encoder_path}"
            )

        print("[ADIS] Loading behavioral encoder...")
        print(f"[ADIS] Encoder: {self.encoder_path}")

        artifact = joblib.load(self.encoder_path)

        if not isinstance(artifact, dict):
            raise TypeError(
                "behavioral_encoder.joblib must contain a dictionary artifact."
            )

        self.artifact = artifact

        # ---------------------------------------------------------
        # Feature schema
        # ---------------------------------------------------------

        self.features = (
            artifact.get("features")
            or artifact.get("feature_names")
            or artifact.get("model_features")
        )

        if self.features is None:
            raise KeyError(
                "Encoder artifact does not contain feature names. "
                "Expected one of: features, feature_names, model_features."
            )

        self.features = list(self.features)

        # ---------------------------------------------------------
        # Preprocessing objects
        # ---------------------------------------------------------

        self.scaler = artifact.get("scaler")

        if self.scaler is None:
            raise KeyError(
                "Encoder artifact does not contain a RobustScaler/scaler object."
            )

        self.pca = artifact.get("pca")

        if self.pca is None:
            raise KeyError(
                "Encoder artifact does not contain a PCA object."
            )

        # Medians handling (supports dict, pd.Series, list, ndarray, or scaler.center_)
        raw_medians = artifact.get("medians")

        if raw_medians is not None:
            if isinstance(raw_medians, dict):
                self.medians = np.asarray(
                    [raw_medians[feature] for feature in self.features],
                    dtype=np.float64,
                )
            elif isinstance(raw_medians, pd.Series):
                self.medians = np.asarray(
                    [raw_medians[feature] for feature in self.features],
                    dtype=np.float64,
                )
            else:
                self.medians = np.asarray(raw_medians, dtype=np.float64)
        else:
            if hasattr(self.scaler, "center_"):
                self.medians = np.asarray(
                    self.scaler.center_,
                    dtype=np.float64,
                )
            else:
                raise KeyError(
                    "Encoder artifact does not contain medians and "
                    "the scaler has no center_ attribute."
                )

        # ---------------------------------------------------------
        # Log-transform configuration
        # ---------------------------------------------------------

        self.log_features = list(artifact.get("log_features", []))

        unknown_log_features = [
            feature for feature in self.log_features if feature not in self.features
        ]

        if unknown_log_features:
            raise ValueError(
                "Encoder artifact contains log features "
                "that are not part of the production schema:\n"
                + "\n".join(f"  - {x}" for x in unknown_log_features)
            )

        print(f"[ADIS] Log-transformed features: {len(self.log_features)}")

        # ---------------------------------------------------------
        # Dimensions
        # ---------------------------------------------------------

        self.input_dim = len(self.features)

        if input_dim is not None and input_dim != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch. "
                f"Encoder expects {self.input_dim}, "
                f"but received {input_dim}."
            )

        if hasattr(self.pca, "n_components_"):
            self.embedding_dim = int(self.pca.n_components_)
        else:
            self.embedding_dim = embedding_dim

        # ---------------------------------------------------------
        # Version / schema
        # ---------------------------------------------------------

        self.encoder_version = artifact.get(
            "encoder_version",
            self.ENCODER_VERSION,
        )

        self.feature_schema_hash = artifact.get(
            "feature_schema_hash",
            self._calculate_schema_hash(self.features),
        )

        print(f"[ADIS] Features: {self.input_dim}")
        print(f"[ADIS] Embedding dimension: {self.embedding_dim}")
        print(f"[ADIS] Encoder version: {self.encoder_version}")
        print(f"[ADIS] Schema hash: {self.feature_schema_hash}")

        # ---------------------------------------------------------
        # Sanity checks
        # ---------------------------------------------------------

        if len(self.medians) != self.input_dim:
            raise ValueError(
                "Median vector length does not match feature count."
            )

        if hasattr(self.scaler, "n_features_in_"):
            if self.scaler.n_features_in_ != self.input_dim:
                raise ValueError(
                    "Scaler feature dimension does not match encoder schema."
                )

        if hasattr(self.pca, "n_features_in_"):
            if self.pca.n_features_in_ != self.input_dim:
                raise ValueError(
                    "PCA input dimension does not match encoder schema."
                )

        # ---------------------------------------------------------
        # Encoder consistency checks
        # ---------------------------------------------------------

        artifact_embedding_dim = artifact.get("embedding_dim")

        if artifact_embedding_dim is not None:
            artifact_embedding_dim = int(artifact_embedding_dim)

            if hasattr(self.pca, "n_components_"):
                if int(self.pca.n_components_) != artifact_embedding_dim:
                    raise ValueError(
                        "Encoder artifact embedding dimension "
                        "does not match PCA."
                    )

        artifact_normalization = artifact.get("normalization")

        if artifact_normalization not in (None, "l2"):
            raise ValueError(
                f"Unsupported encoder normalization: {artifact_normalization}"
            )

        artifact_whiten = artifact.get("whiten")

        if artifact_whiten is not None:
            pca_transform = getattr(self.pca, "pca", self.pca)
            actual_whiten = bool(getattr(pca_transform, "whiten", False))

            if actual_whiten != bool(artifact_whiten):
                raise ValueError("Encoder whitening configuration mismatch.")

    # =============================================================
    # Schema utilities
    # =============================================================

    @staticmethod
    def _calculate_schema_hash(features: List[str]) -> str:
        payload = json.dumps(
            list(features),
            separators=(",", ":"),
            ensure_ascii=True,
        )

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # =============================================================
    # Input normalization
    # =============================================================

    def _prepare_dataframe(
        self,
        flow_data: Union[
            pd.DataFrame,
            pd.Series,
            Dict[str, Any],
            np.ndarray,
            List[float],
        ],
    ) -> pd.DataFrame:

        if isinstance(flow_data, pd.DataFrame):
            if len(flow_data) != 1:
                raise ValueError(
                    "BehavioralThreatAnalyzer expects exactly one flow."
                )

            df = flow_data.copy()

            missing = [
                feature
                for feature in self.features
                if feature not in df.columns
            ]

            if missing:
                raise ValueError(
                    "Input flow is missing required features:\n"
                    + "\n".join(f"  - {x}" for x in missing)
                )

            df = df[self.features]
            return df

        if isinstance(flow_data, pd.Series):
            return self._prepare_dataframe(
                flow_data.to_frame().T
            )

        if isinstance(flow_data, dict):
            return self._prepare_dataframe(
                pd.DataFrame([flow_data])
            )

        if isinstance(flow_data, (np.ndarray, list, tuple)):
            arr = np.asarray(flow_data, dtype=np.float64).reshape(-1)

            if len(arr) != self.input_dim:
                raise ValueError(
                    f"Raw feature vector has {len(arr)} values, "
                    f"but encoder expects {self.input_dim}."
                )

            return pd.DataFrame(
                [arr],
                columns=self.features,
            )

        raise TypeError(
            f"Unsupported flow input type: {type(flow_data)}"
        )

    # =============================================================
    # Behavioral encoding
    # =============================================================

    def encode(
        self,
        flow_data: Union[
            pd.DataFrame,
            pd.Series,
            Dict[str, Any],
            np.ndarray,
            List[float],
        ],
    ) -> np.ndarray:

        df = self._prepare_dataframe(flow_data)

        df = df.apply(
            pd.to_numeric,
            errors="coerce",
        )

        df = df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        for feature, median_value in zip(self.features, self.medians):
            if df[feature].isna().any():
                df[feature] = df[feature].fillna(median_value)

        for feature in self.log_features:
            values = df[feature].to_numpy(dtype=np.float64)
            values = np.maximum(values, 0.0)
            df[feature] = np.log1p(values)

        df = df[self.features]

        X_scaled = self.scaler.transform(df)
        embedding = self.pca.transform(X_scaled)

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        norm = np.linalg.norm(embedding)

        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError(
                "Invalid behavioral embedding produced by encoder."
            )

        embedding = embedding / norm

        return embedding

    # =============================================================
    # Behavior interpretation
    # =============================================================

    def _infer_behavior_family(self, flow: dict) -> dict:
        """
        Infer a conservative behavioral family from flow-level evidence.

        IMPORTANT:
        - Ground-truth labels are NEVER used here.
        - Source/Destination IP/Port/Timestamp are not required.
        - Classification is intentionally conservative.
        - When evidence is insufficient, return a generic anomaly family.
        """
        
        # Clean up CIC-IDS2017 specific spaces from column names to ensure gets work correctly
        clean_flow = {str(k).strip(): v for k, v in flow.items()}

        def num(name: str, default: float = 0.0) -> float:
            value = clean_flow.get(name, default)

            try:
                value = float(value)
            except (TypeError, ValueError):
                return default

            if not np.isfinite(value):
                return default

            return value

        packet_rate = num("Flow Packets/s")
        byte_rate = num("Flow Bytes/s")

        fwd_packets = num("Total Fwd Packets")
        bwd_packets = num("Total Backward Packets")

        fwd_packet_rate = num("Fwd Packets/s")
        bwd_packet_rate = num("Bwd Packets/s")

        flow_duration = num("Flow Duration")

        syn_flags = num("SYN Flag Count")
        rst_flags = num("RST Flag Count")
        ack_flags = num("ACK Flag Count")
        fin_flags = num("FIN Flag Count")
        psh_flags = num("PSH Flag Count")

        total_packets = fwd_packets + bwd_packets

        # -----------------------------------------------------
        # Derived behavioral signals
        # -----------------------------------------------------

        packet_direction_ratio = (
            fwd_packets / max(bwd_packets, 1.0)
        )

        response_ratio = (
            bwd_packets / max(fwd_packets, 1.0)
        )

        flag_total = (
            syn_flags
            + rst_flags
            + ack_flags
            + fin_flags
            + psh_flags
        )

        syn_rst_pattern = syn_flags > 0 and rst_flags > 0

        asymmetric_flow = (
            packet_direction_ratio >= 5.0
            or response_ratio >= 5.0
        )

        # -----------------------------------------------------
        # Evidence object
        # -----------------------------------------------------

        evidence = {
            "flow_packet_rate": packet_rate,
            "flow_byte_rate": byte_rate,
            "forward_packets": fwd_packets,
            "backward_packets": bwd_packets,
            "forward_packet_rate": fwd_packet_rate,
            "backward_packet_rate": bwd_packet_rate,
            "flow_duration": flow_duration,
            "syn_flags": syn_flags,
            "rst_flags": rst_flags,
            "ack_flags": ack_flags,
            "fin_flags": fin_flags,
            "psh_flags": psh_flags,
            "total_packets": total_packets,
            "packet_direction_ratio": packet_direction_ratio,
            "response_ratio": response_ratio,
            "flag_total": flag_total,
        }

        # -----------------------------------------------------
        # 1. High-rate network activity
        # -----------------------------------------------------

        if packet_rate >= 10000:
            return {
                "behavior_family": "high_rate_network_activity",
                "threat_type": "Denial-of-Service Candidate",
                "classification_confidence": "high",
                "classification_reason": (
                    "Extremely high flow packet rate indicates "
                    "high-rate network activity."
                ),
                "evidence": evidence,
            }

        # -----------------------------------------------------
        # 2. High-rate connection attempts
        # -----------------------------------------------------

        if packet_rate >= 5000 and syn_flags > 0:
            return {
                "behavior_family": "high_rate_connection_attempts",
                "threat_type": "Reconnaissance / Connection Flood Candidate",
                "classification_confidence": "high",
                "classification_reason": (
                    "High packet rate combined with SYN activity "
                    "indicates repeated connection attempts."
                ),
                "evidence": evidence,
            }

        # -----------------------------------------------------
        # 3. Connection probing
        # -----------------------------------------------------

        if (
            syn_rst_pattern
            and total_packets <= 20
        ):
            return {
                "behavior_family": "connection_probe_candidate",
                "threat_type": "Reconnaissance Candidate",
                "classification_confidence": "medium",
                "classification_reason": (
                    "SYN/RST activity with a low packet-count flow "
                    "is consistent with connection probing behavior."
                ),
                "evidence": evidence,
            }

        # -----------------------------------------------------
        # 4. Repetitive / asymmetric connection behavior
        # -----------------------------------------------------

        if (
            total_packets >= 20
            and (
                packet_direction_ratio >= 8.0
                or response_ratio >= 8.0
            )
        ):
            return {
                "behavior_family": "repetitive_connection_behavior",
                "threat_type": "Repeated Connection Activity",
                "classification_confidence": "medium",
                "classification_reason": (
                    "Strong directional packet imbalance combined "
                    "with repeated packet activity."
                ),
                "evidence": evidence,
            }

        # -----------------------------------------------------
        # 5. High-volume flow
        # -----------------------------------------------------

        if (
            fwd_packets > 500
            or bwd_packets > 500
            or total_packets > 1000
        ):
            return {
                "behavior_family": "high_volume_flow",
                "threat_type": "High-Volume Network Activity",
                "classification_confidence": "medium",
                "classification_reason": (
                    "Large packet volume indicates unusually "
                    "high-volume network activity."
                ),
                "evidence": evidence,
            }

        # -----------------------------------------------------
        # 6. High-bandwidth flow
        # -----------------------------------------------------

        if byte_rate >= 1_000_000:
            return {
                "behavior_family": "high_bandwidth_flow",
                "threat_type": "High-Bandwidth Network Activity",
                "classification_confidence": "medium",
                "classification_reason": (
                    "High byte rate indicates unusually high "
                    "bandwidth consumption."
                ),
                "evidence": evidence,
            }

        # -----------------------------------------------------
        # 7. Generic anomalous behavior
        # -----------------------------------------------------

        return {
            "behavior_family": "anomalous_network_behavior",
            "threat_type": None,
            "classification_confidence": "low",
            "classification_reason": (
                "Available flow-level evidence is insufficient "
                "for a more specific behavioral classification."
            ),
            "evidence": evidence,
        }

    # =============================================================
    # Severity
    # =============================================================

    @staticmethod
    def _severity_from_anomaly_score(
        anomaly_score: Optional[float],
    ) -> str:

        if anomaly_score is None:
            return "UNKNOWN"

        try:
            score = float(anomaly_score)
        except (TypeError, ValueError):
            return "UNKNOWN"

        if not np.isfinite(score):
            return "UNKNOWN"

        if score >= 0.95:
            return "CRITICAL"

        if score >= 0.80:
            return "HIGH"

        if score >= 0.50:
            return "MEDIUM"

        return "LOW"

    # =============================================================
    # Antigen extraction
    # =============================================================

    def analyze_and_extract(
        self,
        raw_features=None,
        flow_data: Optional[
            Union[
                pd.DataFrame,
                pd.Series,
                Dict[str, Any],
            ]
        ] = None,
        anomaly_score: Optional[float] = None,
        source_dataset: Optional[str] = None,
    ) -> ThreatAntigen:

        # If structured flow data exists, it is preferred.
        if flow_data is not None:

            if isinstance(flow_data, pd.DataFrame):
                if len(flow_data) != 1:
                    raise ValueError(
                        "flow_data DataFrame must contain exactly one row."
                    )

                flow_dict = flow_data.iloc[0].to_dict()

            elif isinstance(flow_data, pd.Series):
                flow_dict = flow_data.to_dict()

            else:
                flow_dict = dict(flow_data)

            embedding = self.encode(flow_data)

        else:

            if raw_features is None:
                raise ValueError(
                    "Either raw_features or flow_data must be provided."
                )

            embedding = self.encode(raw_features)

            flow_dict = {
                feature: float(value)
                for feature, value in zip(
                    self.features,
                    np.asarray(raw_features).reshape(-1),
                )
            }

        # ---------------------------------------------------------
        # Behavioral interpretation
        # ---------------------------------------------------------

        classification = self._infer_behavior_family(flow_dict)

        behavior_family = classification["behavior_family"]
        threat_type = classification["threat_type"]
        classification_confidence = classification["classification_confidence"]
        classification_reason = classification["classification_reason"]
        behavior_evidence = classification["evidence"]

        severity = self._severity_from_anomaly_score(
            anomaly_score
        )

        # ---------------------------------------------------------
        # Behavioral technique representation
        # ---------------------------------------------------------

        technique_map = {
            "high_rate_network_activity": (
                "DOS-HIGH-RATE",
                "High-rate denial-of-service candidate",
                "medium",
            ),
            "high_rate_connection_attempts": (
                "RECON-CONNECTION-FLOOD",
                "High-rate connection attempt candidate",
                "medium",
            ),
            "connection_probe_candidate": (
                "RECON-CONNECTION",
                "Connection probing candidate",
                "medium",
            ),
            "high_volume_flow": (
                "HIGH-VOLUME-NETWORK",
                "High-volume network activity",
                "low",
            ),
            "high_bandwidth_flow": (
                "HIGH-BANDWIDTH-NETWORK",
                "High-bandwidth network activity",
                "low",
            ),
            "repetitive_connection_behavior": (
                "REPEATED-CONNECTION",
                "Repeated connection behavior",
                "low",
            ),
        }

        (
            technique_id,
            technique_name,
            technique_confidence,
        ) = technique_map.get(
            behavior_family,
            (
                "UNMAPPED",
                "Unmapped anomalous network behavior",
                "not_attributed",
            ),
        )
        
        tactic = "UNKNOWN"

        # ---------------------------------------------------------
        # Antigen
        # ---------------------------------------------------------

        antigen_id = str(uuid.uuid4())

        metadata = {
            "representation_version": self.encoder_version,
            "feature_schema_hash": self.feature_schema_hash,
            "embedding_dimension": self.embedding_dim,

            "anomaly_score": anomaly_score,

            "behavior_family": behavior_family,
            "behavior_evidence": behavior_evidence,

            "threat_type": threat_type,
            "classification_confidence": classification_confidence,
            "classification_reason": classification_reason,

            "source_dataset": source_dataset,

            "attribution_status": "behavioral_candidate",
            "technique_confidence": technique_confidence,

            "protocol": flow_dict.get("Protocol"),
            "flow_duration": flow_dict.get("Flow Duration"),

            "encoder_path": str(self.encoder_path),
        }

        return ThreatAntigen(
            antigen_id=antigen_id,
            technique_id=technique_id,
            technique_name=technique_name,
            tactic=tactic,
            severity=severity,
            embedding=embedding.tolist(),
            metadata=metadata,
        )
