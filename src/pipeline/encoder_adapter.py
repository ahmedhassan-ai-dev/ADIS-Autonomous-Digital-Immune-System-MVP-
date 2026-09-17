from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import joblib

from src.pipeline.contracts import (
    BehaviorResult,
    EMBEDDING_DIM,
    ENCODER_VERSION,
    FEATURE_SCHEMA_HASH,
)


class BehavioralEncoderAdapter:
    """
    Production adapter for the frozen ADIS Behavioral Encoder v2.

    Artifact structure:
        {
            "encoder_version": ...,
            "features": [...],
            "feature_schema_hash": ...,
            "medians": {...},
            "log_features": [...],
            "scaler": RobustScaler,
            "pca": PCA,
            "embedding_dim": 32,
            "whiten": True,
            "normalization": "l2",
            ...
        }

    Pipeline:
        raw flow
            -> numeric conversion
            -> inf/-inf -> NaN
            -> median imputation
            -> log1p selected features
            -> RobustScaler
            -> PCA / whitening
            -> L2 normalization
            -> 32D embedding
    """

    def __init__(
        self,
        model_path: str | Path = "models/behavioral_encoder.joblib",
    ) -> None:

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Behavioral encoder artifact not found: {self.model_path}"
            )

        self.encoder = joblib.load(self.model_path)

        if not isinstance(self.encoder, dict):
            raise TypeError(
                "Behavioral encoder artifact must be a dictionary."
            )

        self._validate_artifact()

    def _validate_artifact(self) -> None:
        """Validate the frozen encoder metadata and components."""

        required_keys = {
            "encoder_version",
            "features",
            "feature_schema_hash",
            "medians",
            "log_features",
            "scaler",
            "pca",
            "embedding_dim",
            "whiten",
            "normalization",
        }

        missing = required_keys - set(self.encoder.keys())

        if missing:
            raise ValueError(
                "Behavioral encoder artifact is missing keys: "
                + ", ".join(sorted(missing))
            )

        # Version
        artifact_version = self.encoder["encoder_version"]

        if artifact_version != ENCODER_VERSION:
            raise ValueError(
                "Encoder version mismatch: "
                f"expected={ENCODER_VERSION}, "
                f"actual={artifact_version}"
            )

        # Embedding dimension
        embedding_dim = int(self.encoder["embedding_dim"])

        if embedding_dim != EMBEDDING_DIM:
            raise ValueError(
                "Encoder embedding dimension mismatch: "
                f"expected={EMBEDDING_DIM}, "
                f"actual={embedding_dim}"
            )

        # Schema hash
        schema_hash = self.encoder["feature_schema_hash"]

        if schema_hash != FEATURE_SCHEMA_HASH:
            raise ValueError(
                "Encoder feature schema hash mismatch: "
                f"expected={FEATURE_SCHEMA_HASH}, "
                f"actual={schema_hash}"
            )

        # Feature count
        features = self.encoder["features"]

        if not isinstance(features, list):
            raise TypeError(
                "Encoder 'features' must be a list."
            )

        if len(features) != 77:
            raise ValueError(
                "Unexpected encoder feature count: "
                f"expected=77, actual={len(features)}"
            )

        # Log feature validation
        log_features = self.encoder["log_features"]

        if not isinstance(log_features, list):
            raise TypeError(
                "Encoder 'log_features' must be a list."
            )

        for feature in log_features:
            if feature not in features:
                raise ValueError(
                    f"Log feature is not present in feature schema: {feature}"
                )

        # Median validation
        medians = self.encoder["medians"]

        if not isinstance(medians, dict):
            raise TypeError(
                "Encoder 'medians' must be a dictionary."
            )

        missing_medians = [
            feature
            for feature in features
            if feature not in medians
        ]

        if missing_medians:
            raise ValueError(
                "Missing encoder medians for features: "
                + ", ".join(missing_medians)
            )

        # Scaler validation
        scaler = self.encoder["scaler"]

        if not hasattr(scaler, "transform"):
            raise TypeError(
                "Encoder scaler does not expose transform()."
            )

        if getattr(scaler, "center_", None) is None:
            raise ValueError(
                "Encoder scaler is not fitted."
            )

        # PCA validation
        pca = self.encoder["pca"]

        if not hasattr(pca, "transform"):
            raise TypeError(
                "Encoder PCA does not expose transform()."
            )

        if getattr(pca, "components_", None) is None:
            raise ValueError(
                "Encoder PCA is not fitted."
            )

        if pca.components_.shape != (EMBEDDING_DIM, len(features)):
            raise ValueError(
                "Unexpected PCA component shape: "
                f"expected=({EMBEDDING_DIM}, {len(features)}), "
                f"actual={pca.components_.shape}"
            )

        # Whitening
        if bool(self.encoder["whiten"]) != bool(pca.whiten):
            raise ValueError(
                "Encoder whitening metadata does not match PCA."
            )

        # Normalization
        if self.encoder["normalization"] != "l2":
            raise ValueError(
                "Unsupported encoder normalization: "
                f"{self.encoder['normalization']}"
            )

    @property
    def features(self) -> list[str]:
        """Exact frozen feature order."""

        return list(self.encoder["features"])

    @property
    def log_features(self) -> list[str]:
        """Features that use log1p transformation."""

        return list(self.encoder["log_features"])

    @property
    def medians(self) -> dict[str, float]:
        """Frozen training medians."""

        return dict(self.encoder["medians"])

    @property
    def scaler(self):
        return self.encoder["scaler"]

    @property
    def pca(self):
        return self.encoder["pca"]

    def _prepare_flow(
        self,
        flow: Dict[str, Any] | pd.Series | pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare one flow using the frozen encoder schema."""

        if isinstance(flow, pd.DataFrame):

            if len(flow) != 1:
                raise ValueError(
                    "BehavioralEncoderAdapter expects exactly one flow."
                )

            df = flow.copy()

        elif isinstance(flow, pd.Series):

            df = flow.to_frame().T

        elif isinstance(flow, dict):

            df = pd.DataFrame([flow])

        else:
            raise TypeError(
                "flow must be a dict, pandas Series, "
                "or one-row DataFrame."
            )

        missing = [
            feature
            for feature in self.features
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                "Flow is missing encoder features: "
                + ", ".join(missing)
            )

        # Exact production feature order.
        df = df[self.features].copy()

        # Numeric conversion.
        for feature in self.features:
            df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

        # Replace infinities.
        df = df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # Frozen median imputation.
        for feature in self.features:
            median = self.medians[feature]

            df[feature] = df[feature].fillna(
                median
            )

        # Defensive validation.
        if df[self.features].isna().any().any():
            raise ValueError(
                "NaN values remain after median imputation."
            )

        return df

    def encode(
        self,
        flow: Dict[str, Any] | pd.Series | pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate the frozen 32D behavioral representation.
        """

        df = self._prepare_flow(flow)

        # Convert to float64 for sklearn preprocessing using DataFrame format to preserve feature names.
        X = df[self.features].astype(np.float64)

        # ---------------------------------------------------------
        # Step 1 — log1p transformation
        # ---------------------------------------------------------
        if self.log_features:
            log_values = X[self.log_features].to_numpy()

            # log1p requires values >= -1.
            if np.any(log_values < -1.0):
                raise ValueError(
                    "Negative value below -1 detected in "
                    "log-transformed features."
                )

            X[self.log_features] = np.log1p(log_values)

        # ---------------------------------------------------------
        # Step 2 — RobustScaler
        # ---------------------------------------------------------
        X_scaled = self.scaler.transform(X)

        # ---------------------------------------------------------
        # Step 3 — PCA + whitening
        # ---------------------------------------------------------
        X_embedding = self.pca.transform(X_scaled)
        embedding = np.asarray(
            X_embedding[0],
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # Step 4 — L2 normalization
        # ---------------------------------------------------------
        norm = float(
            np.linalg.norm(embedding)
        )

        if not np.isfinite(norm):
            raise ValueError(
                "Behavioral embedding norm is not finite."
            )

        if norm == 0.0:
            raise ValueError(
                "Behavioral embedding has zero magnitude."
            )

        embedding = embedding / norm

        # ---------------------------------------------------------
        # Final validation
        # ---------------------------------------------------------
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                "Invalid behavioral embedding dimension: "
                f"expected={(EMBEDDING_DIM,)}, "
                f"actual={embedding.shape}"
            )

        if not np.all(np.isfinite(embedding)):
            raise ValueError(
                "Behavioral embedding contains NaN or infinity."
            )

        return embedding

    def build_result(
        self,
        flow: Dict[str, Any] | pd.Series | pd.DataFrame,
        behavior_family: str,
        evidence: list[str],
    ) -> BehaviorResult:
        """Build the standardized ADIS BehaviorResult."""

        embedding = self.encode(flow)

        return BehaviorResult(
            behavior_family=behavior_family,
            evidence=list(evidence),
            embedding=embedding.tolist(),
            embedding_dimension=EMBEDDING_DIM,
            encoder_version=ENCODER_VERSION,
            feature_schema_hash=FEATURE_SCHEMA_HASH,
        )

    def info(self) -> Dict[str, Any]:
        return {
            "component": "BehavioralEncoderAdapter",
            "model_path": str(self.model_path),
            "encoder_version": ENCODER_VERSION,
            "embedding_dimension": EMBEDDING_DIM,
            "feature_count": len(self.features),
            "log_feature_count": len(self.log_features),
            "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "whiten": bool(self.encoder["whiten"]),
            "normalization": self.encoder["normalization"],
        }
    