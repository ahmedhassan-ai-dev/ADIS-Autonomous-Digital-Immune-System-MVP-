from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

st.title("🧬 Project Evolution — M1 → M10")
st.caption("A visual record of how the prototype evolved into the final immune loop.")

rows = [
("M1","Telemetry / Dataset Foundation","Data contract, schema validation, synthetic + CIC-style flow preparation","FOUNDATION"),
("M2","Innate Detection","Fast anomaly / attack detection layer","VALIDATED"),
("M3","Behavioral Representation","v2 investigation → v3 supervised representation; 77 features → 32D","VALIDATED"),
("M4","Immune Memory & Policy","Persistent Qdrant memory + similarity recognition + policy thresholds","VALIDATED"),
("M5","Investigation Integration","Threat analysis and contextual evidence","VALIDATED"),
("M6","Evidence Quality","Evidence integration / risk context validation","PASS"),
("M7","Adaptive Response","ALLOW / INVESTIGATE / ISOLATE / explicit RECOVERY lifecycle","PASS"),
("M8","Attack Replay","Persistence, restart recognition, duplicate-memory prevention","PASS"),
("M9","Large-Scale Benchmark","Thousands of flows, threshold analysis and novel-family testing","PASS"),
("M10","Final End-to-End Loop","Detector → behavior → memory → risk → policy → isolation + result artifact","PASS"),
]
for code,title,desc,status in rows:
    with st.container(border=True):
        a,b,c = st.columns([.12,.28,.50])
        a.markdown(f"### {code}")
        b.markdown(f"**{title}**  \n`{status}`")
        c.write(desc)

st.write("")
st.subheader("The central project claim")
st.success("ADIS is not only a detector. The prototype demonstrates a loop in which suspicious behavior can be represented, remembered persistently, recognized again, and routed into an adaptive response lifecycle.")
