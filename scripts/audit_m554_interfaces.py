from __future__ import annotations

import inspect

from src.detection.lightgbm_detector import LightGBMDetector
from src.pipeline.encoder_adapter import BehavioralEncoderAdapter
from src.pipeline.memory_adapter import ImmuneMemoryAdapter
from src.analysis.threat_analyzer import BehavioralThreatAnalyzer
from src.analysis.risk_engine import RiskEngine
from src.decision.decision_engine import DecisionEngine


def audit_component(name: str, cls: type, methods: list[str]) -> None:
    print(f"\n{'=' * 72}")
    print(name)
    print(f"{'=' * 72}")

    print("Constructor:")
    print(" ", inspect.signature(cls))

    instance = cls()

    for method_name in methods:
        print(f"\n{method_name}:")

        if not hasattr(instance, method_name):
            print("  MISSING")
            continue

        method = getattr(instance, method_name)

        try:
            print(" ", inspect.signature(method))
        except (TypeError, ValueError):
            print("  <signature unavailable>")


def main() -> None:
    print("M5.5.4.1 — Real Component Interface Audit")

    audit_component(
        "LightGBMDetector",
        LightGBMDetector,
        [
            "predict_proba",
            "predict",
            "score",
            "detect",
            "detect_one",
            "info",
        ],
    )

    audit_component(
        "BehavioralEncoderAdapter",
        BehavioralEncoderAdapter,
        [
            "encode",
            "build_result",
            "info",
        ],
    )

    audit_component(
        "ImmuneMemoryAdapter",
        ImmuneMemoryAdapter,
        [
            "recognize",
            "commit",
            "record_reexposure",
            "count_memories",
            "info",
        ],
    )

    audit_component(
        "BehavioralThreatAnalyzer",
        BehavioralThreatAnalyzer,
        [
            "analyze_and_extract",
            "analyze",
            "extract",
        ],
    )

    audit_component(
        "RiskEngine",
        RiskEngine,
        [
            "assess",
            "evaluate",
            "calculate",
        ],
    )

    audit_component(
        "DecisionEngine",
        DecisionEngine,
        [
            "decide",
            "evaluate",
            "make_decision",
        ],
    )

    print(f"\n{'=' * 72}")
    print("INTERFACE AUDIT COMPLETE")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
    