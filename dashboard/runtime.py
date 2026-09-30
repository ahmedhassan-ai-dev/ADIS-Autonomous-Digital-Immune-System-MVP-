"""Session-isolated access to the documented ADIS production pipeline."""

from __future__ import annotations

import tempfile
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

PROJECT_CHECKS = [
    ("M4 · Recognition policy", "scripts/test_recognition_policy.py"),
    ("M5 · Decision engine matrix", "scripts/test_decision_engine.py"),
    ("M5 · Security context", "scripts/test_security_context.py"),
    ("M7 · Response lifecycle", "scripts/test_m7_response_lifecycle.py"),
]


@st.cache_resource(show_spinner=False)
def get_pipeline(session_id: str):
    """Cache model loading while keeping each browser session's memory separate."""
    from src.pipeline.real_adis_pipeline import RealADISPipeline
    from src.pipeline.memory_adapter import ImmuneMemoryAdapter

    memory_path = Path(tempfile.gettempdir()) / "adis-soc-demo" / session_id
    memory = ImmuneMemoryAdapter(storage_path=memory_path, top_k=5)
    return RealADISPipeline(memory=memory)


def make_flow(features, packets: float, bytes_total: float, duration: float, syn: float, medians=None):
    """Build one clearly labeled synthetic CIC-style flow for interactive inference."""
    medians = medians or {}
    flow = {name: float(medians.get(name, 0.0)) for name in features}
    for name in features:
        key = name.lower().replace(" ", "_")
        if "duration" in key:
            flow[name] = duration
        elif "packet" in key and ("/s" in key or "_s" in key):
            flow[name] = packets / max(duration / 1_000_000, 0.001)
        elif "bytes" in key and ("/s" in key or "_s" in key):
            flow[name] = bytes_total / max(duration / 1_000_000, 0.001)
        elif ("total_length" in key or "length_total" in key) and "fwd" in key:
            flow[name] = bytes_total * 0.8
        elif ("total_length" in key or "length_total" in key) and "bwd" in key:
            flow[name] = bytes_total * 0.2
        elif "totlen" in key or "total_length" in key or "length_total" in key:
            flow[name] = bytes_total
        elif "subflow_fwd_bytes" in key:
            flow[name] = bytes_total * 0.8
        elif "subflow_bwd_bytes" in key:
            flow[name] = bytes_total * 0.2
        elif "packet" in key and ("count" in key or "total" in key or "fwd" in key or "bwd" in key):
            flow[name] = packets if "bwd" not in key else max(1.0, packets * 0.2)
        elif "syn" in key:
            flow[name] = syn
    return flow


@st.cache_data(show_spinner=False, ttl=300)
def load_runtime_flows():
    """Load the repository's controlled runtime replay flows."""
    import pandas as pd

    path = ROOT / "data" / "runtime" / "m56_attack_test.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Runtime flow dataset not found: {path}")
    return pd.read_parquet(path)


def select_runtime_flow(frame, label: str, row_index: int, features: list[str]):
    """Return only model input columns; never pass the ground-truth label downstream."""
    label_column = "Label" if "Label" in frame.columns else None
    if label_column:
        selected = frame[frame[label_column].astype(str).str.strip() == label]
    else:
        selected = frame
    if selected.empty:
        raise ValueError(f"No runtime flows found for {label}.")
    if row_index < 0 or row_index >= len(selected):
        raise IndexError("Selected runtime flow row is outside the available range.")
    missing = [feature for feature in features if feature not in selected.columns]
    if missing:
        raise ValueError(f"Runtime flow is missing {len(missing)} required model features.")
    return selected.iloc[row_index][features].to_dict()


def analyze_flow(pipeline, flow: dict) -> dict:
    """Run the production analysis pipeline and the documented M7 simulator."""
    from src.sandbox.isolation_chamber import IsolationSandbox

    result = pipeline.process(flow)
    decision = result.get("decision", {})
    action = decision.get("action", "INVESTIGATE")
    response = IsolationSandbox().execute_response(
        action=action,
        flow=flow,
        reason=decision.get("reason", "ADIS policy-selected response."),
    )
    result["response_lifecycle"] = response.to_dict()
    return result


def run_project_check(script: str, timeout_seconds: int = 120) -> dict:
    """Run one allowlisted repository check and return its visible outcome."""
    allowed = {path for _, path in PROJECT_CHECKS}
    if script not in allowed:
        raise ValueError("This project check is not on the dashboard allowlist.")

    module = Path(script).with_suffix("").as_posix().replace("/", ".")
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", module],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-4000:],
    }
