from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from types import SimpleNamespace

import numpy as np

from src.memory.immune_memory import ImmuneMemory
from src.pipeline.contracts import (
    EMBEDDING_DIM,
    ENCODER_VERSION,
    FEATURE_SCHEMA_HASH,
    RecognitionResult,
)
from src.pipeline.m4_policy import M4RecognitionPolicy


class ImmuneMemoryAdapter:
    """
    Production adapter for the ADIS Immune Memory.

    Responsibilities:
    - Load the persistent Qdrant-backed ImmuneMemory.
    - Enforce the frozen ADIS encoder/schema contract.
    - Normalize and validate embeddings.
    - Perform recognition.
    - Re-map raw memory results to the frozen ADIS policy (M4 Policy v1):
          < 0.75       -> NOVEL
          0.75 - 0.90  -> UNCERTAIN
          >= 0.90      -> KNOWN
    - Commit validated novel threats.
    - Record re-exposures.
    """

    def __init__(
        self,
        storage_path: str | Path = "models/immune_memory",
        top_k: int = 5,
    ) -> None:

        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        self.storage_path = Path(storage_path)
        self.top_k = int(top_k)
        self.policy = M4RecognitionPolicy()

        self.memory = ImmuneMemory(
            storage_path=self.storage_path,
            embedding_dim=EMBEDDING_DIM,
            known_threshold=self.policy.known_threshold,
            near_threshold=self.policy.near_threshold,
            encoder_version=ENCODER_VERSION,
            feature_schema_hash=FEATURE_SCHEMA_HASH,
        )

        self._validate_memory_configuration()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_memory_configuration(self) -> None:
        """Validate the underlying ImmuneMemory configuration."""

        if int(self.memory.embedding_dim) != EMBEDDING_DIM:
            raise ValueError(
                "Immune Memory embedding dimension mismatch: "
                f"expected={EMBEDDING_DIM}, "
                f"actual={self.memory.embedding_dim}"
            )

        if float(self.memory.known_threshold) != self.policy.known_threshold:
            raise ValueError(
                "Immune Memory known threshold mismatch: "
                f"expected={self.policy.known_threshold}, "
                f"actual={self.memory.known_threshold}"
            )

        if float(self.memory.near_threshold) != self.policy.near_threshold:
            raise ValueError(
                "Immune Memory near threshold mismatch: "
                f"expected={self.policy.near_threshold}, "
                f"actual={self.memory.near_threshold}"
            )

        if self.memory.encoder_version != ENCODER_VERSION:
            raise ValueError(
                "Immune Memory encoder version mismatch: "
                f"expected={ENCODER_VERSION}, "
                f"actual={self.memory.encoder_version}"
            )

        if self.memory.feature_schema_hash != FEATURE_SCHEMA_HASH:
            raise ValueError(
                "Immune Memory feature schema hash mismatch."
            )

    # ------------------------------------------------------------------
    # Embedding validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_embedding(
        embedding: list[float] | np.ndarray,
    ) -> np.ndarray:
        """Validate one ADIS 32D behavioral embedding."""

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        if vector.shape != (EMBEDDING_DIM,):
            raise ValueError(
                "Invalid immune-memory embedding dimension: "
                f"expected={(EMBEDDING_DIM,)}, "
                f"actual={vector.shape}"
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "Immune-memory embedding contains NaN or infinity."
            )

        norm = float(np.linalg.norm(vector))

        if norm == 0.0:
            raise ValueError(
                "Immune-memory embedding has zero magnitude."
            )

        # Defensive normalization.
        vector = vector / norm

        return vector

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize(
        self,
        embedding: list[float] | np.ndarray,
    ) -> RecognitionResult:
        """
        Recognize a behavioral embedding against Immune Memory.

        Production classification (M4 Policy v1):
            similarity < 0.75
                -> NOVEL

            0.75 <= similarity < 0.90
                -> UNCERTAIN

            similarity >= 0.90
                -> KNOWN
        """

        vector = self._validate_embedding(embedding)

        raw_result = self.memory.recognize_threat(
            query_vector=vector,
            top_k=self.top_k,
        )

        similarity = float(
            raw_result.get("similarity", 0.0)
        )
        similarity = max(-1.0, min(1.0, similarity))

        raw_memory_id = raw_result.get("memory_id")
        if isinstance(raw_memory_id, dict):
            memory_id = raw_memory_id.get("memory_id")
        else:
            memory_id = raw_memory_id

        # --------------------------------------------------------------
        # Frozen ADIS recognition policy
        # --------------------------------------------------------------

        classification, policy_action = self.policy.classify(
            similarity
        )

        if classification == "KNOWN":
            policy_confidence = similarity
        else:
            policy_confidence = max(
                0.0,
                1.0 - similarity,
            )

        return RecognitionResult(
            classification=classification,
            similarity=similarity,
            policy_classification=classification,
            policy_action=policy_action,
            policy_confidence=float(
                min(1.0, policy_confidence)
            ),
            memory_id=memory_id,
            memory_status=raw_result.get("status"),
            metadata={
                "raw_memory_classification": raw_result.get(
                    "classification"
                ),
                "raw_memory_status": raw_result.get(
                    "status"
                ),
                "candidate_count": len(
                    raw_result.get("candidates", [])
                ),
                "known_threshold": self.policy.known_threshold,
                "near_threshold": self.policy.near_threshold,
                "policy_version": self.policy.policy_version,
                "encoder_version": ENCODER_VERSION,
                "feature_schema_hash": FEATURE_SCHEMA_HASH,
            },
        )

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit(
        self,
        antigen: Any,
        source_dataset: Optional[str] = None,
        validated: bool = True,
    ) -> Dict[str, Any]:
        """
        Commit a validated threat antigen to Immune Memory.
        """

        if antigen is None:
            raise ValueError(
                "Cannot commit a null antigen."
            )

        embedding = getattr(
            antigen,
            "embedding",
            None,
        )

        if embedding is None:
            raise ValueError(
                "Antigen does not contain an embedding."
            )

        self._validate_embedding(embedding)

        return self.memory.commit_antigen(
            antigen=antigen,
            source_dataset=source_dataset,
            validated=validated,
        )

    # ------------------------------------------------------------------
    # Benchmark Enrollment
    # ------------------------------------------------------------------

    def enroll_embedding(
        self,
        embedding: list[float] | np.ndarray,
        behavior_family: str,
        source_dataset: str = "M9_BENCHMARK_ENROLLMENT",
    ) -> Dict[str, Any]:
        """
        Enroll a benchmark prototype into persistent Immune Memory.

        This is intentionally a thin adapter-level helper for benchmark
        and calibration workflows. It does not bypass ImmuneMemory.

        The embedding is wrapped as a minimal antigen-compatible object
        and committed through the normal persistent memory path.
        """

        vector = self._validate_embedding(embedding)

        if not behavior_family:
            raise ValueError(
                "behavior_family must not be empty."
            )

        antigen = SimpleNamespace(
            antigen_id=(
                f"m9-prototype-{behavior_family.lower()}"
            ),
            technique_id="M9_PROTOTYPE",
            technique_name="M9 Benchmark Prototype",
            tactic="BENCHMARK",
            severity="BENCHMARK",
            embedding=vector.tolist(),
            metadata={
                "behavior_family": behavior_family,
                "representation_version": ENCODER_VERSION,
                "source": source_dataset,
                "anomaly_score": 1.0,
            },
        )

        return self.memory.commit_antigen(
            antigen=antigen,
            source_dataset=source_dataset,
            validated=True,
        )

    # ------------------------------------------------------------------
    # Benchmark Search
    # ------------------------------------------------------------------

    def search_candidates(
        self,
        embedding: list[float] | np.ndarray,
    ) -> Dict[str, Any]:
        """
        Return raw compatible memory candidates for benchmark analysis.

        Production recognition should continue using recognize().
        This method exists so evaluation code can inspect the matched
        family and similarity without reaching directly into Qdrant.
        """

        vector = self._validate_embedding(embedding)

        return self.memory.recognize_threat(
            query_vector=vector,
            top_k=self.top_k,
        )

    # ------------------------------------------------------------------
    # Re-exposure
    # ------------------------------------------------------------------

    def record_reexposure(
        self,
        recognition_result: RecognitionResult | Dict[str, Any],
        anomaly_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Record a recognized re-exposure.

        The underlying ImmuneMemory expects a dictionary-like
        recognition result, so the adapter converts the production
        contract into the required structure.
        """

        if isinstance(
            recognition_result,
            RecognitionResult,
        ):

            payload = {
                "memory_id": recognition_result.memory_id,
                "similarity": recognition_result.similarity,
                "anomaly_score": anomaly_score,
            }

        elif isinstance(
            recognition_result,
            dict,
        ):

            payload = dict(
                recognition_result
            )

            if anomaly_score is not None:
                payload["anomaly_score"] = anomaly_score

        else:
            raise TypeError(
                "recognition_result must be "
                "RecognitionResult or dict."
            )

        return self.memory.record_reexposure(
            payload
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def count_memories(self) -> int:
        """Return the number of stored immune memories."""

        return int(
            self.memory.count_memories()
        )

    def close(self) -> None:
        self.memory.close()

    # ------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        return {
            "component": "ImmuneMemoryAdapter",
            "storage_path": str(self.storage_path),
            "backend": "Qdrant",
            "mode": "local_persistent",
            "embedding_dimension": EMBEDDING_DIM,
            "known_threshold": self.policy.known_threshold,
            "near_threshold": self.policy.near_threshold,
            "policy_version": self.policy.policy_version,
            "encoder_version": ENCODER_VERSION,
            "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "top_k": self.top_k,
            "memory_count": self.count_memories(),
        }