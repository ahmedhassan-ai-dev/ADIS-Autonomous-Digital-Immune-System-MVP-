from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


class ImmuneMemory:
    """
    Persistent vector-based immune memory for ADIS.

    Responsibilities:
    - Persist antigen embeddings using Qdrant.
    - Recognize previously observed / similar threats.
    - Maintain memory lifecycle metadata.
    - Prevent duplicate memories using deterministic signatures.
    - Track repeated exposures.
    - Protect against incompatible encoder/schema versions.
    """

    COLLECTION_NAME = "adis_immune_memory"

    MEMORY_VERSION = "1.0"

    def __init__(
        self,
        storage_path: str | Path = "models/immune_memory",
        embedding_dim: int = 32,
        known_threshold: float = 0.92,
        near_threshold: float = 0.80,
        encoder_version: str = "behavioral_encoder_v1",
        feature_schema_hash: Optional[str] = None,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.embedding_dim = embedding_dim

        self.known_threshold = float(known_threshold)
        self.near_threshold = float(near_threshold)

        self.encoder_version = encoder_version
        self.feature_schema_hash = feature_schema_hash

        # Local persistent Qdrant storage.
        self.client = QdrantClient(path=str(self.storage_path))

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """
        Create the collection if it does not exist.
        """

        collections = self.client.get_collections().collections
        existing_names = {collection.name for collection in collections}

        if self.COLLECTION_NAME not in existing_names:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalize_vector(vector: List[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)

        if arr.size == 0:
            raise ValueError("Embedding vector is empty.")

        if not np.all(np.isfinite(arr)):
            raise ValueError("Embedding vector contains NaN or infinity.")

        norm = np.linalg.norm(arr)

        if norm == 0:
            raise ValueError("Embedding vector has zero magnitude.")

        return arr / norm

    @staticmethod
    def _stable_json(data: Dict[str, Any]) -> str:
        return json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _build_signature(
        self,
        embedding: np.ndarray,
        behavior_family: str,
        technique_id: Optional[str],
    ) -> str:
        """
        Create a deterministic signature.

        We intentionally do NOT use a random UUID as the only identity.
        Similar re-exposures can therefore be recognized as the same
        biological-style memory rather than creating endless duplicates.
        """

        # Quantize slightly to avoid tiny floating point differences
        # producing different identities.
        quantized = np.round(embedding, decimals=4).tolist()

        payload = {
            "memory_version": self.MEMORY_VERSION,
            "encoder_version": self.encoder_version,
            "feature_schema_hash": self.feature_schema_hash,
            "behavior_family": behavior_family,
            "technique_id": technique_id,
            "embedding": quantized,
        }

        raw = self._stable_json(payload).encode("utf-8")

        return hashlib.sha256(raw).hexdigest()

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit_antigen(
        self,
        antigen,
        source_dataset: Optional[str] = None,
        validated: bool = True,
    ) -> Dict[str, Any]:
        """
        Commit a threat antigen into persistent immune memory.

        If an equivalent memory already exists:
            - do NOT create another point
            - update hit_count
            - update last_seen
            - update anomaly/severity metadata

        Returns a structured memory record.
        """

        embedding = self._normalize_vector(antigen.embedding)

        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {self.embedding_dim}, "
                f"got {embedding.shape[0]}"
            )

        metadata = dict(getattr(antigen, "metadata", {}) or {})

        behavior_family = metadata.get(
            "behavior_family",
            "unknown_anomalous_behavior",
        )

        technique_id = getattr(antigen, "technique_id", None)

        signature = self._build_signature(
            embedding=embedding,
            behavior_family=behavior_family,
            technique_id=technique_id,
        )

        # Convert deterministic signature to a valid UUID for Qdrant compatibility
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, signature))

        now = self._utc_now()

        # --------------------------------------------------------------
        # Check if this antigen already exists
        # --------------------------------------------------------------

        existing = self.client.retrieve(
            collection_name=self.COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )

        if existing:
            point = existing[0]

            payload = dict(point.payload or {})

            hit_count = int(payload.get("hit_count", 1)) + 1

            payload.update(
                {
                    "last_seen": now,
                    "hit_count": hit_count,
                    "anomaly_score": float(
                        metadata.get(
                            "anomaly_score",
                            payload.get("anomaly_score", 0.0),
                        )
                    ),
                    "severity": getattr(
                        antigen,
                        "severity",
                        payload.get("severity", "UNKNOWN"),
                    ),
                    "validated": bool(validated),
                }
            )

            self.client.set_payload(
                collection_name=self.COLLECTION_NAME,
                payload=payload,
                points=[point_id],
            )

            return {
                "memory_id": point_id,
                "signature": signature,
                "status": "UPDATED_EXISTING_MEMORY",
                "hit_count": hit_count,
                "first_seen": payload.get("first_seen"),
                "last_seen": now,
            }

        # --------------------------------------------------------------
        # New memory
        # --------------------------------------------------------------

        payload = {
            "memory_version": self.MEMORY_VERSION,
            "memory_id": point_id,
            "signature": signature,
            "antigen_id": getattr(antigen, "antigen_id", None),

            "technique_id": technique_id,
            "technique_name": getattr(
                antigen,
                "technique_name",
                None,
            ),
            "tactic": getattr(
                antigen,
                "tactic",
                None,
            ),

            "behavior_family": behavior_family,

            "severity": getattr(
                antigen,
                "severity",
                "UNKNOWN",
            ),

            "anomaly_score": float(
                metadata.get("anomaly_score", 0.0)
            ),

            "first_seen": now,
            "last_seen": now,
            "hit_count": 1,

            "validated": bool(validated),

            "encoder_version": self.encoder_version,
            "feature_schema_hash": self.feature_schema_hash,

            "source_dataset": source_dataset,

            "created_at": now,
            "updated_at": now,
        }

        # Keep useful antigen metadata without making payload huge.
        safe_metadata = {
            "protocol": metadata.get("protocol"),
            "flow_duration": metadata.get("flow_duration"),
            "target_port": metadata.get("target_port"),
            "representation_version": metadata.get(
                "representation_version"
            ),
        }

        payload["metadata"] = safe_metadata

        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload,
        )

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point],
        )

        return {
            "memory_id": point_id,
            "signature": signature,
            "status": "NEW_MEMORY_CREATED",
            "hit_count": 1,
            "first_seen": now,
            "last_seen": now,
        }

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize_threat(
        self,
        query_vector: List[float] | np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Search immune memory.

        Classification:
            score >= known_threshold
                -> KNOWN

            near_threshold <= score < known_threshold
                -> NEAR_MATCH

            score < near_threshold
                -> NOVEL

        Only memories generated by the same encoder/schema are eligible.
        """

        vector = self._normalize_vector(query_vector)

        if vector.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query dimension mismatch: "
                f"expected {self.embedding_dim}, "
                f"got {vector.shape[0]}"
            )

        # Current Qdrant API.
        result = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=vector.tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        points = result.points

        if not points:
            return {
                "status": "NOVEL",
                "matched": False,
                "classification": "NOVEL",
                "similarity": 0.0,
                "memory_id": None,
                "payload": None,
                "candidates": [],
            }

        # --------------------------------------------------------------
        # Defensive compatibility check.
        # --------------------------------------------------------------

        compatible_points = []

        for point in points:
            payload = point.payload or {}

            stored_encoder = payload.get("encoder_version")
            stored_schema = payload.get("feature_schema_hash")

            if stored_encoder != self.encoder_version:
                continue

            if stored_schema != self.feature_schema_hash:
                continue

            compatible_points.append(point)

        if not compatible_points:
            return {
                "status": "NOVEL",
                "matched": False,
                "classification": "NOVEL",
                "similarity": 0.0,
                "memory_id": None,
                "payload": None,
                "candidates": [],
                "reason": "No compatible memories found.",
            }

        top_match = compatible_points[0]

        similarity = float(top_match.score)

        if similarity >= self.known_threshold:
            classification = "KNOWN"
            matched = True
        elif similarity >= self.near_threshold:
            classification = "NEAR_MATCH"
            matched = False
        else:
            classification = "NOVEL"
            matched = False

        candidates = []

        for point in compatible_points:
            candidates.append(
                {
                    "memory_id": str(point.id),
                    "similarity": float(point.score),
                    "behavior_family": (
                        point.payload or {}
                    ).get("behavior_family"),
                    "technique_id": (
                        point.payload or {}
                    ).get("technique_id"),
                }
            )

        return {
            "status": classification,
            "matched": matched,
            "classification": classification,
            "similarity": similarity,
            "memory_id": str(top_match.id),
            "payload": dict(top_match.payload or {}),
            "candidates": candidates,
        }

    # ------------------------------------------------------------------
    # Re-exposure
    # ------------------------------------------------------------------

    def record_reexposure(self, recognition_result):
        """
        Record a re-exposure of a previously recognized antigen.

        Accepts the full result returned by recognize_threat().
        """

        from datetime import datetime, timezone

        # ---------------------------------------------------------
        # Extract memory ID safely
        # ---------------------------------------------------------

        if isinstance(recognition_result, dict):

            memory_id = (
                recognition_result.get("memory_id")
                or recognition_result.get("id")
            )

            similarity = recognition_result.get(
                "similarity",
                0.0,
            )

            anomaly_score = recognition_result.get(
                "anomaly_score"
            )

        else:

            memory_id = recognition_result
            similarity = 0.0
            anomaly_score = None

        if memory_id is None:
            return {
                "status": "REEXPOSURE_FAILED",
                "reason": "No memory_id found in recognition result.",
            }

        # ---------------------------------------------------------
        # Retrieve existing memory
        # ---------------------------------------------------------

        existing = self.client.retrieve(
            collection_name=self.COLLECTION_NAME,
            ids=[memory_id],
            with_payload=True,
            with_vectors=False,
        )

        if not existing:
            return {
                "status": "REEXPOSURE_FAILED",
                "reason": f"Memory ID not found: {memory_id}",
            }

        point = existing[0]

        payload = dict(
            point.payload or {}
        )

        # ---------------------------------------------------------
        # Update counters
        # ---------------------------------------------------------

        current_hit_count = int(
            payload.get("hit_count", 1)
        )

        new_hit_count = current_hit_count + 1

        now = datetime.now(
            timezone.utc
        ).isoformat()

        payload["hit_count"] = new_hit_count
        payload["last_seen"] = now

        if anomaly_score is not None:
            payload["last_anomaly_score"] = float(
                anomaly_score
            )

        if similarity is not None:
            payload["last_similarity"] = float(
                similarity
            )

        # ---------------------------------------------------------
        # Update Qdrant payload
        # ---------------------------------------------------------

        self.client.set_payload(
            collection_name=self.COLLECTION_NAME,
            payload=payload,
            points=[memory_id],
        )

        return {
            "status": "REEXPOSURE_RECORDED",
            "memory_id": memory_id,
            "hit_count": new_hit_count,
            "last_seen": now,
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def count_memories(self) -> int:
        """
        Return number of stored immune memories.
        """

        try:
            result = self.client.count(
                collection_name=self.COLLECTION_NAME,
                exact=True,
            )

            return int(result.count)

        except Exception:
            try:
                info = self.client.get_collection(
                    self.COLLECTION_NAME
                )

                points_count = getattr(
                    info,
                    "points_count",
                    None,
                )

                return int(points_count or 0)

            except Exception:
                return 0

    def get_collection_info(self) -> Dict[str, Any]:
        return {
            "collection": self.COLLECTION_NAME,
            "embedding_dim": self.embedding_dim,
            "distance": "COSINE",
            "memory_count": self.count_memories(),
            "encoder_version": self.encoder_version,
            "feature_schema_hash": self.feature_schema_hash,
            "known_threshold": self.known_threshold,
            "near_threshold": self.near_threshold,
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Close Qdrant local storage if supported by installed version.
        """

        try:
            self.client.close()
        except Exception:
            pass