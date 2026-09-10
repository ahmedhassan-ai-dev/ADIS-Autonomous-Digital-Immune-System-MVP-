import time
from typing import Dict, Any, Optional

from src.analysis.threat_analyzer import (
    BehavioralThreatAnalyzer,
    ThreatAntigen,
)


class IsolationSandbox:
    """
    MVP investigation chamber.

    IMPORTANT:
    This is a simulated flow-level investigation.
    It does NOT execute malware, packets, or binaries.
    """

    def __init__(
        self,
        analyzer: BehavioralThreatAnalyzer,
        investigation_delay_ms: float = 10.0,
    ):
        self.analyzer = analyzer
        self.investigation_delay_ms = investigation_delay_ms

    def investigate(
        self,
        raw_features=None,
        flow_data=None,
        anomaly_score: Optional[float] = None,
        source_dataset: Optional[str] = None,
    ) -> ThreatAntigen:

        start = time.perf_counter()

        # ---------------------------------------------------------
        # Simulated isolation/investigation
        # ---------------------------------------------------------

        time.sleep(
            self.investigation_delay_ms / 1000.0
        )

        antigen = self.analyzer.analyze_and_extract(
            raw_features=raw_features,
            flow_data=flow_data,
            anomaly_score=anomaly_score,
            source_dataset=source_dataset,
        )

        investigation_duration_ms = (
            time.perf_counter() - start
        ) * 1000.0

        # ---------------------------------------------------------
        # Investigation metadata
        # ---------------------------------------------------------

        antigen.metadata.update(
            {
                "investigation_duration_ms":
                    round(investigation_duration_ms, 3),

                "investigation_mode":
                    "flow_feature_simulation",

                "isolation_status":
                    "SIMULATED_ISOLATION",

                "validation_status":
                    "SIMULATED",

                "validated":
                    False,
            }
        )

        return antigen