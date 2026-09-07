from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np


@dataclass
class ThreatAntigen:
    antigen_id: str
    technique_id: str
    technique_name: str
    tactic: str
    severity: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BehavioralThreatAnalyzer:
    """
    Analyzes suspicious network telemetry, maps attack behaviors to
    MITRE ATT&CK techniques, and generates stable normalized Antigen Vectors.
    """

    def __init__(self, input_dim: int = 77, embedding_dim: int = 32):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Deterministic, persistent projection matrix across the analyzer lifecycle
        rng = np.random.default_rng(seed=42)
        self.projection_matrix = rng.standard_normal((input_dim, embedding_dim)).astype(np.float32)

    def _determine_technique(self, flow_data: Dict[str, Any]) -> Dict[str, str]:
        # Behavioral heuristic mapping based on network characteristics
        fwd_pkts_s = flow_data.get("Flow_Packets/s", flow_data.get("Flow Packets/s", 0.0))
        syn_flags = flow_data.get("SYN_Flag_Count", flow_data.get("SYN Flag Count", 0.0))
        dst_port = flow_data.get("Destination_Port", flow_data.get("Destination Port", 0.0))
        flow_duration = flow_data.get("Flow_Duration", flow_data.get("Flow Duration", 0.0))

        # Port Scanning behavior: Brief flows, repeated SYN probes across ports
        if syn_flags > 0 and (flow_duration < 10000 or fwd_pkts_s > 5000):
            return {
                "technique_id": "T1046",
                "technique_name": "Network Service Discovery",
                "tactic": "Discovery",
                "severity": "MEDIUM",
            }
        
        # Volumetric DoS/DDoS behavior: Floods of packets or sustained saturation
        if fwd_pkts_s > 10000 or flow_data.get("Total_Fwd_Packets", flow_data.get("Total Fwd Packets", 0)) > 500:
            return {
                "technique_id": "T1498",
                "technique_name": "Network Denial of Service",
                "tactic": "Impact",
                "severity": "HIGH",
            }

        # Authentication Brute Force: Targeted service ports (SSH: 22, FTP: 21, Telnet: 23)
        if dst_port in [21, 22, 23, 3389]:
            return {
                "technique_id": "T1110",
                "technique_name": "Brute Force",
                "tactic": "Credential Access",
                "severity": "HIGH",
            }

        # Default fallback categorization
        return {
            "technique_id": "T1190",
            "technique_name": "Exploit Public-Facing Application",
            "tactic": "Initial Access",
            "severity": "CRITICAL",
        }

    def generate_antigen_vector(self, raw_features: np.ndarray) -> List[float]:
        """
        Projects raw tabular flow features into a dense, normalized L2 Antigen Vector.
        """
        arr = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0).flatten()

        # Consistent matrix projection
        if len(arr) == self.projection_matrix.shape[0]:
            projected = np.dot(arr, self.projection_matrix)
        else:
            # Dynamic fallback if feature count differs
            rng = np.random.default_rng(seed=42)
            fallback_proj = rng.standard_normal((len(arr), self.embedding_dim)).astype(np.float32)
            projected = np.dot(arr, fallback_proj)

        # L2 Normalization for Cosine Distance in Qdrant
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm

        return projected.tolist()

    def analyze_and_extract(
        self,
        antigen_id: str,
        flow_dict: Dict[str, Any],
        raw_features: np.ndarray,
        anomaly_score: float,
    ) -> ThreatAntigen:
        tech_meta = self._determine_technique(flow_dict)
        embedding = self.generate_antigen_vector(raw_features)

        metadata = {
            "anomaly_score": float(anomaly_score),
            "target_port": flow_dict.get("Destination_Port", flow_dict.get("Destination Port", 0)),
            "protocol": flow_dict.get("Protocol", 6),
            "flow_duration": flow_dict.get("Flow_Duration", flow_dict.get("Flow Duration", 0)),
        }

        return ThreatAntigen(
            antigen_id=antigen_id,
            technique_id=tech_meta["technique_id"],
            technique_name=tech_meta["technique_name"],
            tactic=tech_meta["tactic"],
            severity=tech_meta["severity"],
            embedding=embedding,
            metadata=metadata,
        )




    