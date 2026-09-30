from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class M4RecognitionPolicy:
    """
    Frozen ADIS M4 recognition policy.

    Thresholds:
        similarity >= 0.92 -> KNOWN / FAST_PATH
        similarity >= 0.75 -> UNCERTAIN / INVESTIGATE
        similarity <  0.75 -> NOVEL / ISOLATE
    """

    known_threshold: float = 0.92
    near_threshold: float = 0.75
    policy_version: str = "adis-m4-recognition-policy-v1"

    def __post_init__(self) -> None:
        if not 0.0 <= self.near_threshold <= 1.0:
            raise ValueError("near_threshold must be in [0, 1].")

        if not 0.0 <= self.known_threshold <= 1.0:
            raise ValueError("known_threshold must be in [0, 1].")

        if self.known_threshold <= self.near_threshold:
            raise ValueError(
                "known_threshold must be greater than near_threshold."
            )

    def classify(self, similarity: float) -> tuple[str, str]:
        score = float(similarity)

        if score >= self.known_threshold:
            return "KNOWN", "FAST_PATH"

        if score >= self.near_threshold:
            return "UNCERTAIN", "INVESTIGATE"

        return "NOVEL", "ISOLATE"

    def to_dict(self) -> dict:
        return {
            "policy_version": self.policy_version,
            "known_threshold": self.known_threshold,
            "near_threshold": self.near_threshold,
        }
