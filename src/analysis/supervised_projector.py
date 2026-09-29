from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


@dataclass
class SupervisedBehavioralProjector:
    """
    32D supervised behavioral projector.

    Interface-compatible with the existing ADIS encoder:
      - n_features_in_
      - n_components_
      - transform(X)

    Representation:
      - RobustScaler is applied before this projector.
      - LDA branch captures family-discriminative directions.
      - PCA branch preserves residual behavioral variation.
      - The two branches are concatenated to exactly 32 dimensions.
    """

    pca_components: int = 26
    lda_components: int = 6
    lda_weight: float = 2.0
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.pca_components < 1:
            raise ValueError("pca_components must be >= 1")
        if self.lda_components < 1:
            raise ValueError("lda_components must be >= 1")
        if self.lda_weight <= 0:
            raise ValueError("lda_weight must be > 0")

        self.pca = PCA(
            n_components=self.pca_components,
            whiten=True,
            random_state=self.random_state,
        )
        self.lda = LinearDiscriminantAnalysis(
            n_components=self.lda_components,
            solver="svd",
        )
        self.n_features_in_: int | None = None
        self.n_components_: int = self.pca_components + self.lda_components
        self.classes_: Any = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupervisedBehavioralProjector":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

        n_classes = len(np.unique(y))
        max_lda = n_classes - 1
        if self.lda_components > max_lda:
            self.lda_components = max_lda
            self.n_components_ = self.pca_components + self.lda_components
            self.lda = LinearDiscriminantAnalysis(
                n_components=self.lda_components,
                solver="svd",
            )

        self.n_features_in_ = int(X.shape[1])

        # Fit both views on exactly the training split.
        self.pca.fit(X)
        self.lda.fit(X, y)
        self.classes_ = self.lda.classes_
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Projector is not fitted.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        p = self.pca.transform(X)
        l = self.lda.transform(X) * float(self.lda_weight)

        e = np.concatenate([l, p], axis=1)

        if e.shape[1] != self.n_components_:
            raise RuntimeError(
                f"Unexpected embedding dimension: {e.shape[1]} "
                f"!= {self.n_components_}"
            )

        return e

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
