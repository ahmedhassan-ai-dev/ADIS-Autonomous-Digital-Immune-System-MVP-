from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class LightGBMDetector:
    """
    ADIS adapter for the persisted LightGBM binary detector.

    Classification:
        0 -> Benign
        1 -> Attack

    The attack probability is exposed as the detector score.
    """

    MODEL_PATH = Path("models/merged_binary_detector.joblib")
    DEFAULT_THRESHOLD = 0.50

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.model_path = Path(model_path or self.MODEL_PATH)
        self.threshold = float(threshold)

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        self._artifact = self._load_artifact()

        self.model = self._artifact["model"]
        self.features = list(self._artifact["features"])

        if not self.features:
            raise ValueError(
                "Saved detector contains no feature definitions."
            )

        self.n_features = len(self.features)

        # The saved artifact stores the original CICIDS feature names,
        # while the underlying LightGBM Booster stores normalized names.
        self.booster_features = list(
            self.model.booster_.feature_name()
        )

        if len(self.booster_features) != self.n_features:
            raise ValueError(
                "Feature count mismatch between artifact and Booster: "
                f"{self.n_features} vs {len(self.booster_features)}"
            )

        self.feature_mapping = dict(
            zip(
                self.features,
                self.booster_features,
            )
        )

    def _load_artifact(self) -> dict[str, Any]:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"LightGBM model not found: {self.model_path}"
            )

        artifact = joblib.load(self.model_path)

        if not isinstance(artifact, dict):
            raise TypeError(
                "Invalid LightGBM artifact. "
                "Expected a dictionary."
            )

        required_keys = {"model", "features"}
        missing = required_keys - set(artifact.keys())

        if missing:
            raise ValueError(
                "Invalid LightGBM artifact. "
                f"Missing keys: {sorted(missing)}"
            )

        return artifact

    def _prepare_features(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare raw CICIDS features for the saved LightGBM Booster.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input data must be a pandas DataFrame."
            )

        # ---------------------------------------------------------
        # 1. Validate required artifact features
        # ---------------------------------------------------------

        missing_features = [
            feature
            for feature in self.features
            if feature not in data.columns
        ]

        if missing_features:
            raise ValueError(
                "Input is missing required detector features: "
                f"{missing_features}"
            )

        # ---------------------------------------------------------
        # 2. Select exact artifact feature order
        # ---------------------------------------------------------

        X = data.loc[:, self.features].copy()

        # ---------------------------------------------------------
        # 3. Numeric conversion
        # ---------------------------------------------------------

        for column in X.columns:
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce",
            )

        # ---------------------------------------------------------
        # 4. Replace invalid values
        # ---------------------------------------------------------

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # ---------------------------------------------------------
        # 5. Median imputation
        # ---------------------------------------------------------

        medians = X.median(numeric_only=True)

        X = X.fillna(medians)

        # Completely invalid columns -> 0
        X = X.fillna(0.0)

        # ---------------------------------------------------------
        # 6. Rename to exact Booster feature names
        # ---------------------------------------------------------

        X = X.rename(
            columns=self.feature_mapping
        )

        # ---------------------------------------------------------
        # 7. Final exact order validation
        # ---------------------------------------------------------

        X = X.loc[:, self.booster_features]

        return X

    def predict_proba(
        self,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return attack probability for each input row.
        """

        X = self._prepare_features(data)

        probabilities = self.model.predict_proba(
            X,
            validate_features=True,
        )

        probabilities = np.asarray(
            probabilities
        )

        if (
            probabilities.ndim != 2
            or probabilities.shape[1] < 2
        ):
            raise RuntimeError(
                "Unexpected LightGBM probability output shape: "
                f"{probabilities.shape}"
            )

        # Class 1 = Attack
        return probabilities[:, 1]

    def predict(
        self,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Binary prediction.

        0 -> Benign
        1 -> Attack
        """

        attack_probability = self.predict_proba(
            data
        )

        return (
            attack_probability >= self.threshold
        ).astype(int)

    def score(
        self,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Attack probability.

        Higher score = stronger evidence of attack.
        """

        return self.predict_proba(data)

    def detect(
        self,
        data: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Return structured detection results.
        """

        probabilities = self.predict_proba(
            data
        )

        decisions = (
            probabilities >= self.threshold
        ).astype(int)

        results: list[dict[str, Any]] = []

        for probability, decision in zip(
            probabilities,
            decisions,
        ):
            score = float(probability)

            results.append(
                {
                    "anomaly_score": score,
                    "attack_probability": score,
                    "decision": (
                        "ATTACK"
                        if int(decision) == 1
                        else "BENIGN"
                    ),
                    "is_anomaly": bool(decision),
                    "threshold": self.threshold,
                }
            )

        return results

    def detect_one(
        self,
        flow: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Detect a single network flow.
        """

        data = pd.DataFrame([flow])

        return self.detect(data)[0]

    def info(self) -> dict[str, Any]:
        """
        Return detector metadata.
        """

        return {
            "detector": "LightGBM",
            "model_path": str(
                self.model_path
            ),
            "feature_count": self.n_features,
            "threshold": self.threshold,
            "task": "binary_attack_detection",
            "schema_mapping": True,
        }