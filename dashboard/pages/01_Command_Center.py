from __future__ import annotations

import uuid
import subprocess
from datetime import datetime
from pathlib import Path

import streamlit as st
from dashboard.runtime import (
    PROJECT_CHECKS,
    analyze_flow,
    get_pipeline,
    load_runtime_flows,
    make_flow,
    run_project_check,
    select_runtime_flow,
)

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "ADIS_COMPREHENSIVE_DOCUMENTATION.md"

if "adis_session_id" not in st.session_state:
    st.session_state.adis_session_id = uuid.uuid4().hex
if "adis_events" not in st.session_state:
    st.session_state.adis_events = []

st.title("ADIS SOC command center", icon=":material/security:")
st.caption("Run a project CIC-IDS2017 flow or a synthetic scenario through the integrated detection, memory, risk, and response pipeline.")

try:
    pipeline = get_pipeline(st.session_state.adis_session_id)
    detector_info = pipeline.detector.info()
    features = pipeline.detector.features
    memory_count = pipeline.memory.memory.count_memories()
    known_threshold = pipeline.memory.policy.known_threshold
    near_threshold = pipeline.memory.policy.near_threshold
    system_ready = True
except Exception as exc:
    pipeline = None
    features = []
    memory_count = 0
    known_threshold, near_threshold = 0.92, 0.75
    system_ready = False
    init_error = str(exc)

kpi = st.columns(4)
kpi_slots = [column.empty() for column in kpi]
if system_ready and known_threshold != 0.92:
    st.info(f"The documentation specifies KNOWN at 0.92; the live M4 policy currently reports {known_threshold:.2f}. Results below use the running policy.")

with st.container(border=True):
    st.subheader("Run a live flow")
    st.caption("Choose a real CIC-IDS2017 runtime flow or create a synthetic profile. Analyze it, then replay the same flow to compare memory recognition.")
    st.caption("Steps: choose a flow → analyze exposure 1 → replay the exact same flow to see whether memory changes the classification and response.")
    if st.button("Start clean demo session", help="Starts with a fresh, session-isolated immune memory and clears this session's flow history."):
        st.session_state.adis_session_id = uuid.uuid4().hex
        st.session_state.adis_events = []
        for key in ("adis_last_flow", "adis_last_source", "adis_exposure", "adis_first_exposure_result"):
            st.session_state.pop(key, None)
        st.rerun()
    source = st.selectbox("Flow source", ["Project runtime flow", "Synthetic profile"])
    runtime_frame = None
    runtime_labels = []
    if source == "Project runtime flow":
        try:
            runtime_frame = load_runtime_flows()
            if "Label" in runtime_frame.columns:
                runtime_labels = sorted(runtime_frame["Label"].dropna().astype(str).unique().tolist())
            else:
                runtime_labels = ["Runtime flow"]
        except Exception as exc:
            st.warning(f"Runtime flow data is unavailable ({exc}). Switch to Synthetic profile to continue.")

    with st.form("adis_flow_form"):
        if source == "Project runtime flow" and runtime_frame is not None and runtime_labels:
            runtime_label = st.selectbox("Scenario label", runtime_labels)
            label_rows = runtime_frame[
                runtime_frame["Label"].astype(str).str.strip() == runtime_label
            ] if "Label" in runtime_frame.columns else runtime_frame
            row_index = st.number_input(
                "Flow number in this scenario",
                min_value=0,
                max_value=max(0, len(label_rows) - 1),
                value=0,
                step=1,
            )
            source_description = f"CIC-IDS2017 · {runtime_label} · row {int(row_index)}"
        elif source == "Synthetic profile":
            preset = st.selectbox("Starting profile", ["Baseline", "Elevated traffic", "High-volume traffic"])
            defaults = {
                "Baseline": (8.0, 6000.0, 2_000_000.0, 0.0),
                "Elevated traffic": (80.0, 65000.0, 1_000_000.0, 8.0),
                "High-volume traffic": (900.0, 2_000_000.0, 500_000.0, 180.0),
            }[preset]
            left, right = st.columns(2)
            with left:
                packets = st.number_input("Packet count", min_value=1.0, max_value=1_000_000.0, value=defaults[0], step=1.0)
                bytes_total = st.number_input("Total bytes", min_value=0.0, max_value=1_000_000_000.0, value=defaults[1], step=100.0)
            with right:
                duration = st.number_input("Flow duration (µs)", min_value=1.0, max_value=1_000_000_000.0, value=defaults[2], step=1000.0)
                syn = st.number_input("SYN flag count", min_value=0.0, max_value=1_000_000.0, value=defaults[3], step=1.0)
            source_description = f"Synthetic · {preset}"
        else:
            source_description = "Unavailable"
        submitted = st.form_submit_button(
            "Analyze as exposure 1",
            type="primary",
            disabled=not system_ready or (source == "Project runtime flow" and runtime_frame is None),
        )

    replay_clicked = st.button(
        "Replay last flow as next exposure",
        disabled=not system_ready or not st.session_state.get("adis_last_flow"),
        help="Sends the exact same feature values through ADIS again so you can compare recognition and response.",
    )
    if submitted and pipeline:
        try:
            if source == "Project runtime flow":
                flow = select_runtime_flow(runtime_frame, runtime_label, int(row_index), features)
            else:
                flow = make_flow(
                    features,
                    packets,
                    bytes_total,
                    duration,
                    syn,
                    medians=pipeline.encoder.medians,
                )
            with st.spinner("ADIS is processing the flow…"):
                result = analyze_flow(pipeline, flow)
            st.session_state.adis_last_flow = flow
            st.session_state.adis_last_source = source_description
            st.session_state.adis_exposure = 1
            st.session_state.adis_first_exposure_result = result
            memory_count = pipeline.memory.memory.count_memories()
            st.session_state.adis_events.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "result": result,
                "profile": source_description,
                "exposure": 1,
            })
            del st.session_state.adis_events[50:]
            st.success("Flow analyzed. Results below are from this run.")
        except Exception as exc:
            st.error(f"The pipeline could not analyze this flow: {exc}")
    elif replay_clicked and pipeline:
        try:
            with st.spinner("Replaying the same flow through ADIS…"):
                result = analyze_flow(pipeline, st.session_state.adis_last_flow)
            st.session_state.adis_exposure += 1
            memory_count = pipeline.memory.memory.count_memories()
            st.session_state.adis_events.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "result": result,
                "profile": st.session_state.get("adis_last_source", "Replay"),
                "exposure": st.session_state.adis_exposure,
            })
            del st.session_state.adis_events[50:]
            st.success(f"Exposure {st.session_state.adis_exposure} completed on the same flow.")
        except Exception as exc:
            st.error(f"The flow replay failed: {exc}")

for slot, label, value in zip(
    kpi_slots,
    ("Pipeline", "Immune memory", "Flows analyzed", "Recognition bounds"),
    ("Ready" if system_ready else "Unavailable", f"{memory_count} patterns", len(st.session_state.adis_events), f"{near_threshold:.2f} / {known_threshold:.2f}"),
):
    slot.metric(label, value)

if not system_ready:
    st.error(f"ADIS components could not be loaded: {init_error}")

st.subheader("Project validation suite")
st.caption("Run the repository's existing M4, M5, and M7 checks and see each result here.")
if st.button("Run all 4 project checks", type="secondary"):
    results = []
    with st.status("Starting ADIS validation checks…", expanded=True) as status:
        for index, (label, script) in enumerate(PROJECT_CHECKS, start=1):
            status.update(label=f"Running {index}/{len(PROJECT_CHECKS)} · {label}", state="running")
            try:
                outcome = run_project_check(script)
            except subprocess.TimeoutExpired:
                outcome = {"passed": False, "returncode": None, "stdout": "", "stderr": "Timed out after 120 seconds."}
            except Exception as exc:
                outcome = {"passed": False, "returncode": None, "stdout": "", "stderr": str(exc)}
            results.append({"label": label, "script": script, **outcome})
            status.write(("✅ PASS" if outcome["passed"] else "❌ FAIL") + f" · {label}")
        passed = sum(item["passed"] for item in results)
        status.update(
            label=f"Validation complete · {passed}/{len(results)} passed",
            state="complete" if passed == len(results) else "error",
            expanded=True,
        )
    st.session_state.adis_check_results = results

if st.session_state.get("adis_check_results"):
    check_results = st.session_state.adis_check_results
    passed = sum(item["passed"] for item in check_results)
    st.metric("Latest check run", f"{passed}/{len(check_results)} passed")
    for item in check_results:
        title = ("✅ " if item["passed"] else "❌ ") + item["label"]
        with st.expander(title):
            st.caption(item["script"])
            if item["stdout"]:
                st.code(item["stdout"], language="text")
            if item["stderr"]:
                st.error(item["stderr"])

if st.session_state.adis_events:
    latest = st.session_state.adis_events[0]
    result = latest["result"]
    context = result.get("context", {})
    detection = context.get("detection", {})
    recognition = context.get("recognition", {})
    behavior = context.get("behavior", {})
    threat = context.get("threat", {})
    risk = result.get("risk", {})
    decision = result.get("decision", {})
    score = float(detection.get("anomaly_score", 0.0) or 0.0)
    similarity = float(recognition.get("similarity", 0.0) or 0.0)
    risk_score = float(risk.get("risk_score", 0.0) or 0.0)

    st.subheader("Latest decision")
    st.caption(f"Exposure {latest.get('exposure', 1)} · {latest['profile']}")
    metrics = st.columns(4)
    metrics[0].metric("Detection", str(detection.get("detector_decision", "—")), f"score {score:.3f}")
    metrics[1].metric("Memory match", str(recognition.get("classification", "—")), f"similarity {similarity:.3f}")
    metrics[2].metric("Risk", str(risk.get("severity", "—")), f"score {risk_score:.3f}")
    response = result.get("response_lifecycle", {})
    metrics[3].metric("Simulated response", str(decision.get("action", "—")), str(response.get("lifecycle_status", "—")))

    if latest.get("exposure", 1) > 1:
        first = st.session_state.get("adis_first_exposure_result", {})
        first_context = first.get("context", {})
        first_detection = first_context.get("detection", {})
        first_recognition = first_context.get("recognition", {})
        first_decision = first.get("decision", {})
        st.markdown("**First exposure → replay**")
        st.write(
            f"Detection: {first_detection.get('detector_decision', '—')} → {detection.get('detector_decision', '—')} · "
            f"Memory: {first_recognition.get('classification', '—')} → {recognition.get('classification', '—')} · "
            f"Response: {first_decision.get('action', '—')} → {decision.get('action', '—')}"
        )
        first_risk = first.get("risk", {}).get("risk_score", 0.0)
        first_latency = float(first.get("latency_ms", 0.0) or 0.0)
        replay_latency = float(result.get("latency_ms", 0.0) or 0.0)
        st.caption(f"Risk score: {float(first_risk or 0.0):.3f} → {risk_score:.3f} · Pipeline latency: {first_latency:.1f} ms → {replay_latency:.1f} ms")

    with st.container(border=True):
        st.markdown(f"**Behavior:** {behavior.get('behavior_family', '—')} · **Threat severity:** {threat.get('severity', '—')}")
        st.write(decision.get("reason", "Decision rationale is not available."))
        factors = risk.get("risk_factors", [])
        if factors:
            st.caption("Risk evidence: " + " · ".join(map(str, factors)))
        st.caption(f"Live recognition policy: KNOWN ≥ {known_threshold:.2f} · UNCERTAIN {near_threshold:.2f}–{known_threshold:.2f} · NOVEL < {near_threshold:.2f} · Test profile: {latest['profile']}")
        memory_event = result.get("memory_event")
        if memory_event:
            st.caption(f"Immune memory event: {memory_event.get('status', 'recorded')}")

    with st.expander("Pipeline stages and investigation evidence"):
        stages = [
            ("M2 · Innate detection", detection.get("detector_decision", "—"), score),
            ("M3 · Behavioral representation", behavior.get("behavior_family", "—"), 1.0),
            ("M4 · Immune memory", recognition.get("classification", "—"), max(0.0, min(1.0, similarity))),
            ("M5 · Risk assessment", risk.get("severity", "—"), risk_score),
            ("M5 · Response decision", decision.get("action", "—"), 1.0),
        ]
        for label, value, progress in stages:
            st.progress(max(0.0, min(1.0, float(progress or 0.0))), text=f"{label} · {value}")
        investigation = result.get("investigation", {})
        if investigation:
            st.write("Investigation summary")
            st.json(investigation)
        if response:
            st.write(f"M7 simulated lifecycle · {response.get('lifecycle_status', '—')} · validation: {response.get('validation_status', '—')}")
            transitions = response.get("transition_history", [])
            if transitions:
                st.dataframe(transitions, hide_index=True, width="stretch")
            st.caption("The sandbox models response transitions; it does not isolate a real host or network.")

    st.subheader("Recent flow decisions")
    chart_events = list(reversed(st.session_state.adis_events[:20]))
    st.caption("Latest 20 runs · detection, memory match, and risk")
    st.line_chart({
        "Detection score": [float(event["result"].get("context", {}).get("detection", {}).get("anomaly_score", 0.0) or 0.0) for event in chart_events],
        "Memory similarity": [float(event["result"].get("context", {}).get("recognition", {}).get("similarity", 0.0) or 0.0) for event in chart_events],
        "Risk score": [float(event["result"].get("risk", {}).get("risk_score", 0.0) or 0.0) for event in chart_events],
    })
    for event in st.session_state.adis_events[:8]:
        item = event["result"]
        ctx = item.get("context", {})
        d = ctx.get("detection", {})
        r = ctx.get("recognition", {})
        dec = item.get("decision", {})
        st.write(f"`{event['time']}` · Exposure {event.get('exposure', 1)} · {event['profile']} · {d.get('detector_decision', '—')} · {r.get('classification', '—')} · **{dec.get('action', '—')}**")
else:
    st.info("Run a flow to populate this SOC view with live detection, recognition, risk, and response results.")

with st.expander("ADIS operating model"):
    st.write("ADIS follows the documented closed loop: incoming network flow → innate detection → behavioral representation → immune memory recognition → risk assessment → deterministic response. A suspicious novel behavior can enter investigation and validated learning; a recognized behavior can follow a memory-assisted response.")
    if DOC.exists():
        st.caption(f"Project source of truth: {DOC.relative_to(ROOT).as_posix()} · Documentation v2.0 (M1–M10)")
