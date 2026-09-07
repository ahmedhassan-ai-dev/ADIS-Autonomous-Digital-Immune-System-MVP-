from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from src.analysis.threat_analyzer import ThreatAntigen


class ImmuneMemory:
    """
    Qdrant-backed Vector Store representing the Persistent Immune Memory of ADIS.
    Stores validated threat signatures and provides high-speed Cosine Similarity retrieval.
    """

    COLLECTION_NAME = "adis_immune_memory"

    def __init__(self, storage_path: Optional[str] = None, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim

        # In-memory storage by default, or local file storage
        if storage_path:
            Path(storage_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=storage_path)
        else:
            self.client = QdrantClient(location=":memory:")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        exists = any(c.name == self.COLLECTION_NAME for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def commit_antigen(self, antigen: ThreatAntigen) -> str:
        """
        Encodes a validated threat into persistent Immune Memory (Exposure 1).
        """
        point_id = str(uuid.uuid4())
        payload = {
            "antigen_id": antigen.antigen_id,
            "technique_id": antigen.technique_id,
            "technique_name": antigen.technique_name,
            "tactic": antigen.tactic,
            "severity": antigen.severity,
            **antigen.metadata,
        }

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=antigen.embedding,
                    payload=payload,
                )
            ],
        )
        return point_id

    def recognize_threat(
        self,
        query_vector: List[float],
        similarity_threshold: float = 0.90,
    ) -> Optional[Dict[str, Any]]:
        """
        Queries the Immune Memory to detect previously encountered threat patterns (Exposure 2).
        """
        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=1,
        ).points

        if not results:
            return None

        top_match = results[0]
        if top_match.score >= similarity_threshold:
            return {
                "matched": True,
                "similarity_score": round(float(top_match.score), 4),
                "point_id": top_match.id,
                "payload": top_match.payload,
            }

        return {
            "matched": False,
            "similarity_score": round(float(top_match.score), 4),
            "payload": top_match.payload,
        }

    def count_memories(self) -> int:
        info = self.client.get_collection(collection_name=self.COLLECTION_NAME)
        return info.points_count


    