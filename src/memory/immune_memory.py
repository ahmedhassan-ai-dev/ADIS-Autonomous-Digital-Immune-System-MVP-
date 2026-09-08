from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


# ============================================================
# ADIS — M4 Persistent Immune Memory
# ============================================================

class ImmuneMemory:
    """
    M4 — Persistent Immune Memory.

    Stores validated threat antigens and recognizes
    previously observed behavioral patterns.
    """

    COLLECTION_NAME = "adis_immune_memory"

    def __init__(
        self,
        storage_path: Optional[str] = None,
        embedding_dim: int = 32,
    ):

        self.embedding_dim = embedding_dim

        if storage_path:

            Path(
                storage_path
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            self.client = QdrantClient(
                path=storage_path
            )

        else:

            self.client = QdrantClient(
                location=":memory:"
            )

        self._ensure_collection()

    # ========================================================
    # Collection
    # ========================================================

    def _ensure_collection(self):

        collections = (
            self.client
            .get_collections()
            .collections
        )

        exists = any(
            c.name == self.COLLECTION_NAME
            for c in collections
        )

        if not exists:

            self.client.create_collection(
                collection_name=
                    self.COLLECTION_NAME,

                vectors_config=
                    VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                    ),
            )

    # ========================================================
    # Commit antigen
    # ========================================================

    def commit_antigen(
        self,
        antigen,
    ) -> str:

        point_id = str(
            uuid.uuid4()
        )

        now = datetime.now(
            timezone.utc
        ).isoformat()

        payload = {

            "antigen_id":
                antigen.antigen_id,

            "technique_id":
                antigen.technique_id,

            "technique_name":
                antigen.technique_name,

            "tactic":
                antigen.tactic,

            "severity":
                antigen.severity,

            "first_seen":
                now,

            "last_seen":
                now,

            "exposure_count":
                1,

            **antigen.metadata,
        }

        self.client.upsert(

            collection_name=
                self.COLLECTION_NAME,

            points=[
                PointStruct(

                    id=point_id,

                    vector=
                        antigen.embedding,

                    payload=payload,
                )
            ],
        )

        return point_id

    # ========================================================
    # Recognition
    # ========================================================

    def recognize_threat(
        self,
        query_vector: List[float],
        similarity_threshold: float = 0.90,
    ) -> Optional[Dict[str, Any]]:

        results = (
            self.client
            .query_points(
                collection_name=
                    self.COLLECTION_NAME,

                query=query_vector,

                limit=1,

                with_payload=True,
            )
            .points
        )

        if not results:

            return {
                "matched": False,
                "similarity_score": 0.0,
                "payload": None,
            }

        top_match = results[0]

        similarity = float(
            top_match.score
        )

        matched = (
            similarity
            >= similarity_threshold
        )

        return {

            "matched":
                matched,

            "similarity_score":
                round(
                    similarity,
                    6,
                ),

            "point_id":
                top_match.id,

            "payload":
                top_match.payload,
        }

    # ========================================================
    # Reinforcement
    # ========================================================

    def reinforce_memory(
        self,
        point_id: str,
    ):

        result = self.client.retrieve(
            collection_name=
                self.COLLECTION_NAME,

            ids=[point_id],

            with_payload=True,

            with_vectors=False,
        )

        if not result:
            return False

        point = result[0]

        payload = (
            point.payload
            or {}
        )

        exposure_count = int(
            payload.get(
                "exposure_count",
                1,
            )
        )

        payload[
            "exposure_count"
        ] = exposure_count + 1

        payload[
            "last_seen"
        ] = datetime.now(
            timezone.utc
        ).isoformat()

        self.client.set_payload(
            collection_name=
                self.COLLECTION_NAME,

            payload=payload,

            points=[point_id],
        )

        return True

    # ========================================================
    # Memory statistics
    # ========================================================

    def count_memories(self) -> int:

        info = self.client.get_collection(
            collection_name=
                self.COLLECTION_NAME
        )

        return info.points_count or 0