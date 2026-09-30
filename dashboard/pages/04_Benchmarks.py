from pathlib import Path
import json, pandas as pd, streamlit as st

ROOT=Path(__file__).resolve().parents[2]
BENCH=ROOT/"results"/"m3_m4_recognition_benchmark_v3"/"benchmark_v3.json"
ENV=ROOT/"results"/"m3_m4_recognition_benchmark_v3"/"environment_audit.json"

def load(p):
    try:return json.loads(p.read_text(encoding="utf-8"))
    except:return {}

b,e=load(BENCH),load(ENV)
st.title("📊 Benchmarks & Evidence")
st.caption("Measured evidence from generated project artifacts.")

if b:
    st.subheader("M3 v3 / M4 threshold operating points")
    rows=b.get("threshold_results",b.get("thresholds",[]))
    if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    dist=b.get("distribution_diagnostics",{})
    if dist:
        c=st.columns(4)
        c[0].metric("Genuine median",f"{dist.get('genuine_median','—')}")
        c[1].metric("Impostor median",f"{dist.get('impostor_median','—')}")
        c[2].metric("Approx EER",f"{dist.get('eer','—')}")
        c[3].metric("EER threshold",f"{dist.get('eer_threshold','—')}")
else: st.warning(f"Missing `{BENCH.relative_to(ROOT)}`")

st.divider()
st.subheader("Environment / artifact audit")
if e: st.json(e)
else: st.info("Environment audit artifact not found.")

st.divider()
st.subheader("What these experiments establish")
for x in [
"Known recognition measures whether enrolled threat families are recovered from unseen test flows.",
"Benign FAR measures unwanted matches between normal traffic and threat memory.",
"Novel rejection tests a held-out family excluded from M3 training and M4 enrollment.",
"Threshold selection should be driven by measured validation/operating characteristics.",
"Leakage safety keeps training, validation, enrollment and test evidence separated.",
]: st.write("→ "+x)
