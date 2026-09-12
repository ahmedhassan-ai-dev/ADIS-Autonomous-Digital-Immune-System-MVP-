"""
ADIS M4 — Recognition Policy v1

Converts immune-memory similarity into an operational decision.

Policy:
    similarity >= KNOWN_THRESHOLD
        -> KNOWN

    NOVEL_THRESHOLD <= similarity < KNOWN_THRESHOLD
        -> UNCERTAIN

    similarity < NOVEL_THRESHOLD
        -> NOVEL
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RecognitionDecision:
    classification: str
    similarity: float
    threshold: float
    confidence: float
    action: str


class RecognitionPolicy:
    """
    Operational policy for M4 immune-memory recognition.

    The policy intentionally separates:
        KNOWN
        UNCERTAIN
        NOVEL

    This prevents weak memory matches from immediately entering
    the trusted fast path.
    """

    VERSION = "adis-recognition-policy-v1"

    # Calibrated from the current M3/M4 benchmark.
    NOVEL_THRESHOLD = 0.75
    KNOWN_THRESHOLD = 0.92

    def __init__(
        self,
        novel_threshold: float = NOVEL_THRESHOLD,
        known_threshold: float = KNOWN_THRESHOLD,
    ):
        if not 0.0 <= novel_threshold <= 1.0:
            raise ValueError("novel_threshold must be between 0 and 1.")

        if not 0.0 <= known_threshold <= 1.0:
            raise ValueError("known_threshold must be between 0 and 1.")

        if novel_threshold >= known_threshold:
            raise ValueError(
                "novel_threshold must be lower than known_threshold."
            )

        self.novel_threshold = float(novel_threshold)
        self.known_threshold = float(known_threshold)

    def classify(self, similarity: float) -> RecognitionDecision:
        similarity = float(similarity)

        # Numerical safety.
        similarity = max(-1.0, min(1.0, similarity))

        if similarity >= self.known_threshold:
            classification = "KNOWN"
            action = "FAST_PATH"
            threshold = self.known_threshold

            # Confidence rises from 0 at threshold to 1 at similarity 1.
            confidence = (
                similarity - self.known_threshold
            ) / (1.0 - self.known_threshold)

        elif similarity >= self.novel_threshold:
            classification = "UNCERTAIN"
            action = "INVESTIGATE"
            threshold = self.novel_threshold

            # 0 at novel threshold, 1 at known threshold.
            confidence = (
                similarity - self.novel_threshold
            ) / (
                self.known_threshold - self.novel_threshold
            )

        else:
            classification = "NOVEL"
            action = "ISOLATE"
            threshold = self.novel_threshold

            # Higher confidence as similarity approaches -1.
            confidence = (
                self.novel_threshold - similarity
            ) / (self.novel_threshold + 1.0)

        confidence = max(0.0, min(1.0, confidence))

        return RecognitionDecision(
            classification=classification,
            similarity=similarity,
            threshold=threshold,
            confidence=confidence,
            action=action,
        )