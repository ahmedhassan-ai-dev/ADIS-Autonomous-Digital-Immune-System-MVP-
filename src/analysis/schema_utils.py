import hashlib
import json
from typing import List


def feature_schema_hash(features: List[str]) -> str:
    """
    Deterministic hash for model feature ordering/schema.
    """

    payload = json.dumps(
        features,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()