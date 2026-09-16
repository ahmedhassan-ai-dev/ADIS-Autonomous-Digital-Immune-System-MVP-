from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


class M21IsolationForestAdapter:
    """
    Production adapter for the persisted ADIS M2.1 Isolation Forest.

    The persisted detector:
    - was trained on 59 selected CIC-IDS2017 features
    - uses IsolationForest
    - stores its selected feature list
    - stores an optimized threshold based on:
        anomaly_score = -decision_function(X)
    """

    VERSION = "adis-m21-isolation-forest-v1"

    def __init__(self, model_path=None):
        if model_path is None:
            project_root = Path(__file__).resolve().parents[2]
            model_path = (
                project_root
                / "models"
                / "m21_isolation_forest.joblib"
            )

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"M2.1 model not found: {self.model_path}"
            )

        artifact = joblib.load(self.model_path)

        if not isinstance(artifact, dict):
            raise TypeError(
                "M2.1 model artifact must be a dictionary."
            )

        required_keys = {"model", "features", "optimal_threshold"}

        missing = required_keys - set(artifact.keys())

        if missing:
            raise ValueError(
                f"M2.1 artifact is missing keys: {sorted(missing)}"
            )

        self.model = artifact["model"]
        self.features = list(artifact["features"])

        # Stored threshold is based on the negated decision_function.
        self.threshold = float(
            -artifact["optimal_threshold"]
        )

        if len(self.features) != self.model.n_features_in_:
            raise ValueError(
                "M2.1 feature count mismatch: "
                f"artifact={len(self.features)}, "
                f"model={self.model.n_features_in_}"
            )

    def _prepare(self, flow: Any) -> pd.DataFrame:
        """
        Convert one raw flow into the exact 59-feature input
        expected by M2.1.
        """

        if isinstance(flow, pd.DataFrame):
            if len(flow) != 1:
                raise ValueError(
                    "M2.1 adapter expects exactly one flow."
                )

            df = flow.copy()

        elif isinstance(flow, dict):
            df = pd.DataFrame([flow])

        else:
            raise TypeError(
                "flow must be a pandas DataFrame or dict."
            )

        missing = [
            feature
            for feature in self.features
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                "Raw flow is missing M2.1 features: "
                f"{missing}"
            )

        X = df[self.features].copy()

        # Same cleaning logic used during M2.1 training.
        X = X.replace([np.inf, -np.inf], np.nan)

        if X.isna().any().any():
            # Current CIC-IDS2017 production files were verified
            # to contain no NaN/Inf values.
            raise ValueError(
                "M2.1 input contains NaN/Inf values after cleaning. "
                "Training medians are not stored in the persisted "
                "artifact, so missing values cannot be safely imputed."
            )

        return X.astype(np.float32)

    def score(self, flow: Any) -> float:
        """
        Return ADIS anomaly score.

        Higher score = more anomalous.

        anomaly_score = -IsolationForest.decision_function()
        """

        X = self._prepare(flow)

        raw_decision = float(
            self.model.decision_function(X)[0]
        )

        anomaly_score = -raw_decision

        return anomaly_score

    def predict(self, flow: Any) -> int:
        """
        ADIS binary anomaly decision.

        0 = normal
        1 = anomalous
        """

        anomaly_score = self.score(flow)

        return int(
            anomaly_score >= self.threshold
        )

    def detect(self, flow: Any) -> Dict[str, Any]:
        """
        Full detector result for the ADIS pipeline.
        """

        anomaly_score = self.score(flow)

        decision = int(
            anomaly_score >= self.threshold
        )

        return {
            "detector": "M2.1 Isolation Forest",
            "version": self.VERSION,
            "decision": "ANOMALY" if decision else "NORMAL",
            "is_anomaly": bool(decision),
            "anomaly_score": float(anomaly_score),
            "threshold": float(self.threshold),
            "feature_count": len(self.features),
            "model_path": str(self.model_path),
        }