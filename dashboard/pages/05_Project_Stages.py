import streamlit as st
st.title("🧬 Project Evolution — M1 → M10")
st.caption("The engineering path from data foundation to an end-to-end adaptive immune loop.")

rows=[
("M1","Telemetry / Dataset Foundation","Data contract, schema validation, synthetic + CIC-style preparation","FOUNDATION"),
("M2","Innate Detection","Fast anomaly / attack detection","VALIDATED"),
("M3","Behavioral Representation","v2 PCA baseline → v3 supervised LDA+PCA","VALIDATED"),
("M4","Immune Memory & Policy","Persistent memory + recognition policy","VALIDATED"),
("M5","Investigation Integration","Threat analysis and contextual evidence","VALIDATED"),
("M6","Evidence Quality","Evidence-chain / risk-context validation","PASS"),
("M7","Adaptive Response","ALLOW / INVESTIGATE / ISOLATE / RECOVERY","PASS"),
("M8","Attack Replay","Persistence, restart recognition, duplicate prevention","PASS"),
("M9","Recognition Benchmark","Thresholds, FAR/FRR-style diagnostics, novel-family testing","PASS"),
("M10","Final End-to-End Loop","Detector → behavior → memory → risk → policy → response","PASS"),
]
for code,title,desc,status in rows:
    with st.container(border=True):
        a,b,c=st.columns([.12,.32,.56])
        a.markdown(f"### {code}")
        b.markdown(f"**{title}**  \n`{status}`")
        c.write(desc)
st.divider()
st.success("Central claim: ADIS is designed to detect suspicious behavior, represent it, remember validated knowledge persistently, recognize related re-exposures, and route them through an adaptive response lifecycle.")
