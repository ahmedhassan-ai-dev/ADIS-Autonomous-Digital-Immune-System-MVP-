from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.analysis.threat_analyzer import BehavioralThreatAnalyzer
from src.pipeline.memory_adapter import ImmuneMemoryAdapter
from src.sandbox.isolation_chamber import IsolationSandbox
from src.pipeline.m4_policy import M4RecognitionPolicy

try:
    from src.analysis.risk_engine import RiskEngine
except Exception:
    RiskEngine = None

try:
    from src.decision.decision_engine import DecisionEngine
except Exception:
    DecisionEngine = None

try:
    from src.analysis.threat_investigator import ThreatInvestigator
except Exception:
    ThreatInvestigator = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "merged_binary_detector.joblib"
ENCODER_PATH = PROJECT_ROOT / "models" / "behavioral_encoder.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cicids2017" / "Portscan-Friday-no-metadata.parquet"
MEMORY_PATH = PROJECT_ROOT / "models" / "immune_memory_m10_demo"
RESULT_PATH = PROJECT_ROOT / "results" / "m10_final_demo.json"


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    return value


def _clean_detector_row(row: pd.DataFrame, features):
    x = row[features].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    medians = x.median(numeric_only=True)
    return x.fillna(medians)


def _build_risk(risk_engine, context):
    if risk_engine is None:
        score = float(context["anomaly_score"])
        return {
            "risk_score": score,
            "severity": "CRITICAL" if score >= 0.85 else "HIGH" if score >= 0.65 else "MEDIUM",
            "factors": {"anomaly_score": score},
        }

    candidates = [
        lambda: risk_engine.assess(context),
        lambda: risk_engine.evaluate(context),
        lambda: risk_engine.calculate(context),
        lambda: risk_engine.compute(context),
    ]
    for fn in candidates:
        try:
            out = fn()
            return _jsonable(out)
        except (TypeError, AttributeError, KeyError):
            continue

    score = float(context["anomaly_score"])
    return {
        "risk_score": score,
        "severity": "CRITICAL" if score >= 0.85 else "HIGH" if score >= 0.65 else "MEDIUM",
        "factors": {"anomaly_score": score},
        "fallback": True,
    }


def _build_decision(decision_engine, context):
    if decision_engine is not None:
        for fn in (
            lambda: decision_engine.decide(context),
            lambda: decision_engine.evaluate(context),
            lambda: decision_engine.select_action(context),
        ):
            try:
                out = fn()
                if isinstance(out, str):
                    return {"action": out}
                return _jsonable(out)
            except (TypeError, AttributeError, KeyError):
                pass

    risk = float(context.get("risk", {}).get("risk_score", 0.0))
    recognition = context.get("recognition", {}).get("classification", "NOVEL")
    if recognition == "KNOWN" and risk < 0.85:
        action = "ALLOW"
    elif recognition == "UNCERTAIN":
        action = "INVESTIGATE"
    else:
        action = "ISOLATE"
    return {"action": action, "fallback": True}


def main():
    print("=" * 90)
    print("ADIS M10 — FINAL END-TO-END IMMUNE LOOP")
    print("=" * 90)

    for p in (MODEL_PATH, ENCODER_PATH, DATA_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    detector_bundle = joblib.load(MODEL_PATH)
    detector = detector_bundle["model"]
    features = list(detector_bundle["features"])

    analyzer = BehavioralThreatAnalyzer(
        encoder_path=ENCODER_PATH,
        input_dim=len(features),
    )

    memory = ImmuneMemoryAdapter(
        storage_path=MEMORY_PATH,
        top_k=5,
    )

    policy = M4RecognitionPolicy()
    sandbox = IsolationSandbox(analyzer=analyzer, investigation_delay=0.01)

    risk_engine = None
    if RiskEngine is not None:
        try:
            risk_engine = RiskEngine()
        except Exception:
            pass

    decision_engine = None
    if DecisionEngine is not None:
        try:
            decision_engine = DecisionEngine()
        except Exception:
            pass

    investigator = None
    if ThreatInvestigator is not None:
        try:
            investigator = ThreatInvestigator()
        except Exception:
            pass

    df = pd.read_parquet(DATA_PATH)
    label_col = "Label" if "Label" in df.columns else None
    if label_col:
        mask = ~df[label_col].astype(str).str.strip().str.lower().eq("benign")
        attacks = df.loc[mask].reset_index(drop=True)
    else:
        attacks = df.reset_index(drop=True)

    if attacks.empty:
        raise RuntimeError("No attack flow available for M10 demo.")

    row = attacks.iloc[0]
    detector_row = _clean_detector_row(attacks.iloc[[0]], features)

    t0 = time.perf_counter()
    anomaly_score = float(detector.predict_proba(detector_row)[:, 1][0])
    detector_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    antigen = analyzer.analyze_and_extract(
        flow_data=row,
        anomaly_score=anomaly_score,
        source_dataset=DATA_PATH.name,
    )
    analysis_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    recognition_obj = memory.recognize(antigen.embedding)
    memory_ms = (time.perf_counter() - t0) * 1000

    recognition = _jsonable(recognition_obj)
    if not isinstance(recognition, dict):
        recognition = {
            "classification": recognition_obj.classification,
            "similarity": recognition_obj.similarity,
            "policy_action": recognition_obj.policy_action,
            "memory_id": recognition_obj.memory_id,
        }

    similarity = float(recognition.get("similarity", 0.0))
    classification, policy_action = policy.classify(similarity)

    context = {
        "detection": {
            "detector_decision": "ATTACK" if anomaly_score >= 0.5 else "BENIGN",
            "anomaly_score": anomaly_score,
        },
        "behavior": {
            "behavior_family": antigen.metadata.get("behavior_family"),
            "embedding_dimension": len(antigen.embedding),
        },
        "threat": {
            "behavior_family": antigen.metadata.get("behavior_family"),
            "threat_type": antigen.technique_name,
            "technique_id": antigen.technique_id,
            "tactic": antigen.tactic,
            "severity": antigen.severity,
        },
        "recognition": {
            **recognition,
            "classification": classification,
            "policy_action": policy_action,
        },
        "anomaly_score": anomaly_score,
    }

    risk = _build_risk(risk_engine, context)
    context["risk"] = risk
    decision = _build_decision(decision_engine, context)

    action = str(decision.get("action", policy_action))

    response = {
        "status": "OBSERVED",
        "action": action,
        "isolation": None,
    }

    if action == "ISOLATE":
        try:
            # Different project revisions exposed slightly different signatures.
            for fn in (
                lambda: sandbox.execute_response(row, action="ISOLATE"),
                lambda: sandbox.execute_response(row, "ISOLATE"),
                lambda: sandbox.execute_response(flow=row, action="ISOLATE"),
            ):
                try:
                    response["isolation"] = _jsonable(fn())
                    break
                except (TypeError, AttributeError):
                    continue
        except Exception as exc:
            response["isolation"] = {"error": str(exc)}

    context["decision"] = decision
    investigation = None
    if investigator is not None:
        try:
            investigation = _jsonable(
                investigator.investigate({
                    "context": context,
                    "risk": risk,
                    "decision": decision,
                })
            )
        except Exception as exc:
            investigation = {"error": str(exc)}

    result = {
        "run_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "mode": "m10_final_end_to_end",
        "components": {
            "detector_model": str(MODEL_PATH),
            "behavioral_encoder": str(ENCODER_PATH),
            "memory_path": str(MEMORY_PATH),
            "encoder_version": getattr(analyzer, "encoder_version", None),
            "embedding_dim": getattr(analyzer, "embedding_dim", None),
        },
        "detection": context["detection"],
        "behavior": context["behavior"],
        "threat": context["threat"],
        "recognition": context["recognition"],
        "risk": risk,
        "decision": decision,
        "response": response,
        "investigation": investigation,
        "latency_ms": {
            "detector": detector_ms,
            "behavioral_analysis": analysis_ms,
            "memory": memory_ms,
            "total": detector_ms + analysis_ms + memory_ms,
        },
        "memory_count": memory.count_memories(),
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(_jsonable(result), indent=2),
        encoding="utf-8",
    )

    memory.close()

    print("\n[M10 SUMMARY]")
    print(f"Detector        : {result['detection']['detector_decision']} ({anomaly_score:.4f})")
    print(f"Behavior family : {result['behavior']['behavior_family']}")
    print(f"Recognition     : {classification} ({similarity:.4f})")
    print(f"Policy action   : {policy_action}")
    print(f"Risk            : {risk.get('risk_score', 0.0):.4f} / {risk.get('severity', 'UNKNOWN')}")
    print(f"Decision        : {action}")
    print(f"Memory cells    : {result['memory_count']}")
    print(f"Result          : {RESULT_PATH}")
    print("\n[PASS] M10 END-TO-END LOOP EXECUTED")


if __name__ == "__main__":
    main()
