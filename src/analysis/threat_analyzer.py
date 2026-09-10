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


class BehavioralThreatAnalyzer:
    """
    Production-oriented behavioral analyzer (v2 compatible).

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

    ENCODER_VERSION = "adis-behavioral-encoder-v2"

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
            actual_whiten = bool(getattr(self.pca, "whiten", False))

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

        # ---------------------------------------------------------
        # DataFrame
        # ---------------------------------------------------------

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

            # Explicitly enforce production feature order.
            df = df[self.features]

            return df

        # ---------------------------------------------------------
        # Series
        # ---------------------------------------------------------

        if isinstance(flow_data, pd.Series):

            return self._prepare_dataframe(
                flow_data.to_frame().T
            )

        # ---------------------------------------------------------
        # Dictionary
        # ---------------------------------------------------------

        if isinstance(flow_data, dict):

            return self._prepare_dataframe(
                pd.DataFrame([flow_data])
            )

        # ---------------------------------------------------------
        # Numpy / List
        # ---------------------------------------------------------

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

        # Convert every feature to numeric while preserving
        # the DataFrame feature names.
        df = df.apply(
            pd.to_numeric,
            errors="coerce",
        )

        # Replace infinities with NaN.
        df = df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # ---------------------------------------------------------
        # Median imputation
        # ---------------------------------------------------------
        for feature, median_value in zip(
            self.features,
            self.medians,
        ):
            if df[feature].isna().any():
                df[feature] = df[feature].fillna(median_value)

        # ---------------------------------------------------------
        # Log transform
        #
        # MUST exactly match the training pipeline.
        # ---------------------------------------------------------
        for feature in self.log_features:
            values = df[feature].to_numpy(dtype=np.float64)
            values = np.maximum(values, 0.0)
            df[feature] = np.log1p(values)

        # Explicitly preserve production feature order.
        df = df[self.features]

        # ---------------------------------------------------------
        # Robust scaling
        # ---------------------------------------------------------
        X_scaled = self.scaler.transform(df)

        # ---------------------------------------------------------
        # PCA
        # ---------------------------------------------------------
        embedding = self.pca.transform(X_scaled)

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        # ---------------------------------------------------------
        # L2 normalization
        # ---------------------------------------------------------
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

    def _infer_behavior_family(
        self,
        flow_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Conservative behavioral interpretation.

        IMPORTANT:
        We intentionally do NOT claim exact MITRE ATT&CK technique
        attribution because the current CIC-IDS2017 no-metadata
        representation does not provide enough context such as
        destination port, process information, authentication logs,
        etc.
        """

        def get(name, default=0.0):
            value = flow_data.get(name, default)

            try:
                value = float(value)

                if not np.isfinite(value):
                    return default

                return value

            except (TypeError, ValueError):
                return default

        packet_rate = get("Flow Packets/s")
        byte_rate = get("Flow Bytes/s")
        fwd_packets = get("Total Fwd Packets")
        bwd_packets = get("Total Backward Packets")
        syn_flags = get("SYN Flag Count")
        rst_flags = get("RST Flag Count")
        duration = get("Flow Duration")

        evidence = {
            "flow_packet_rate": packet_rate,
            "flow_byte_rate": byte_rate,
            "forward_packets": fwd_packets,
            "backward_packets": bwd_packets,
            "syn_flags": syn_flags,
            "rst_flags": rst_flags,
            "flow_duration": duration,
        }

        # Broad behavioral families only.
        if packet_rate > 10000:
            family = "high_rate_network_activity"

        elif packet_rate > 5000 and syn_flags > 0:
            family = "high_rate_connection_attempts"

        elif syn_flags > 0 and rst_flags > 0:
            family = "connection_probe_candidate"

        elif fwd_packets > 500 or bwd_packets > 500:
            family = "high_volume_flow"

        elif byte_rate > 1_000_000:
            family = "high_bandwidth_flow"

        else:
            family = "anomalous_network_behavior"

        return {
            "behavior_family": family,
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

        behavior = self._infer_behavior_family(flow_dict)

        severity = self._severity_from_anomaly_score(
            anomaly_score
        )

        # ---------------------------------------------------------
        # Conservative ATT&CK representation
        # ---------------------------------------------------------

        technique_id = "UNMAPPED"
        technique_name = "Unmapped anomalous network behavior"
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

            "behavior_family": behavior["behavior_family"],
            "behavior_evidence": behavior["evidence"],

            "source_dataset": source_dataset,

            "attribution_status": "behavioral_candidate",
            "technique_confidence": "not_attributed",

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