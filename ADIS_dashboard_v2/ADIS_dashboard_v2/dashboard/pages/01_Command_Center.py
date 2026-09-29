from pathlib import Path
import json
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results" / "m10_final_demo.json"
AUDIT = ROOT / "results" / "m10_final_audit.json"

def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def pct(x):
    try: return f"{float(x)*100:.1f}%"
    except Exception: return "—"

data = load_json(RESULT)
audit = load_json(AUDIT)

st.markdown('<div class="adis-title">🛡️ ADIS Command Center</div>', unsafe_allow_html=True)
st.markdown('<div class="adis-subtitle">Live view of the final autonomous cyber-immune loop.</div>', unsafe_allow_html=True)
st.write("")

# Top status
det = data.get("detector", {})
risk = data.get("risk", {})
recognition = data.get("recognition", {})
decision = data.get("decision", {})

det_score = det.get("score", data.get("detector_score", 0.0))
det_label = det.get("label", data.get("detector_label", "UNKNOWN"))
family = data.get("behavior_family", "—")
rec_label = recognition.get("classification", data.get("recognition", "—"))
sim = recognition.get("similarity", data.get("similarity", 0.0))
risk_score = risk.get("score", data.get("risk_score", 0.0))
severity = risk.get("severity", data.get("severity", "—"))
action = decision.get("action", data.get("decision", data.get("policy_action", "—")))
memory_cells = data.get("memory_cells", 0)

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Detection", det_label, f"{float(det_score):.4f}" if isinstance(det_score,(int,float)) else None)
c2.metric("Behavior", family)
c3.metric("Recognition", rec_label, f"{float(sim):.4f}" if isinstance(sim,(int,float)) else None)
c4.metric("Risk", severity, f"{float(risk_score):.4f}" if isinstance(risk_score,(int,float)) else None)
c5.metric("Response", action, f"{memory_cells} memory cells")

st.write("")
st.markdown("""
<div class="pipeline">
  <span class="node">Telemetry</span><span class="arrow">→</span>
  <span class="node">M2 Detection</span><span class="arrow">→</span>
  <span class="node">M3 Behavior</span><span class="arrow">→</span>
  <span class="node">M4 Memory</span><span class="arrow">→</span>
  <span class="node">Risk</span><span class="arrow">→</span>
  <span class="node">Policy</span><span class="arrow">→</span>
  <span class="node">Response</span>
</div>
""", unsafe_allow_html=True)

st.write("")
left,right = st.columns([1.15,.85])

with left:
    st.subheader("Current Threat")
    with st.container(border=True):
        st.write(f"**Detector:** `{det_label}`")
        st.write(f"**Behavior family:** `{family}`")
        st.write(f"**Memory recognition:** `{rec_label}`")
        st.write(f"**Risk:** `{severity}` — `{float(risk_score):.4f}`")
        st.write(f"**Policy / response:** `{action}`")
        if rec_label == "NOVEL":
            st.warning("Novel behavior: investigation / isolation path is active.")
        elif rec_label == "KNOWN":
            st.success("Known behavior: memory-assisted fast path is available.")
        else:
            st.info("Uncertain behavior: investigation path is recommended.")

with right:
    st.subheader("System Health")
    with st.container(border=True):
        checks = [
            ("Production encoder", audit.get("production_encoder", True)),
            ("Binary detector", audit.get("binary_detector", True)),
            ("Memory adapter", audit.get("memory_adapter", True)),
            ("M4 policy", audit.get("m4_policy", True)),
            ("Isolation sandbox", audit.get("isolation_sandbox", True)),
            ("M3/M4 benchmark", audit.get("m3_m4_benchmark", True)),
            ("M10 demo", audit.get("m10_demo_result", RESULT.exists())),
        ]
        for label, ok in checks:
            st.write(("🟢" if ok else "🔴") + f" {label}")

st.write("")
st.subheader("Evidence Chain")
evidence = data.get("evidence", [])
if isinstance(evidence, list) and evidence:
    for item in evidence:
        if isinstance(item, dict):
            stage = item.get("stage", item.get("source", "SYSTEM"))
            key = item.get("key", item.get("name", "evidence"))
            value = item.get("value", item.get("detail", ""))
            st.markdown(f"**{stage}** · `{key}` → `{value}`")
else:
    st.info("The latest M10 artifact does not expose a structured evidence list. See the Lab and raw artifact tabs for deeper inspection.")

with st.expander("Raw M10 result"):
    st.json(data)
