import time
from typing import Dict, Any
import numpy as np

from src.analysis.threat_analyzer import BehavioralThreatAnalyzer, ThreatAntigen


class IsolationSandbox:
    """
    Simulates a controlled deception / execution environment (Honeypot/Sandbox).
    Validates suspicious traffic without contaminating the operational network.
    """

    def __init__(self, analyzer: BehavioralThreatAnalyzer):
        self.analyzer = analyzer

    def investigate(
        self,
        event_id: str,
        flow_dict: Dict[str, float],
        raw_features: np.ndarray,
        anomaly_score: float,
    ) -> ThreatAntigen:
        """
        Executes an isolated dynamic investigation on suspicious flow traffic.
        """
        start_time = time.perf_counter()

        # Simulated deep packet inspection & behavioral observation delay
        time.sleep(0.01)

        antigen = self.analyzer.analyze_and_extract(
            antigen_id=event_id,
            flow_dict=flow_dict,
            raw_features=raw_features,
            anomaly_score=anomaly_score,
        )

        antigen.metadata["investigation_duration_ms"] = round(
            (time.perf_counter() - start_time) * 1000, 3
        )
        antigen.metadata["quarantine_status"] = "CONFIRMED_THREAT"

        return antigen