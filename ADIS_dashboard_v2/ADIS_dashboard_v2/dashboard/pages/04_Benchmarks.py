from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "results" / "m3_m4_recognition_benchmark_v3" / "benchmark_v3.json"
ENV = ROOT / "results" / "m3_m4_recognition_benchmark_v3" / "environment_audit.json"

st.title("📊 Benchmarks & Evidence")
st.caption("Results are read from the project's generated artifacts whenever available.")

def load(path):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return {}

b = load(BENCH)
e = load(ENV)

if b:
    st.subheader("M3 v3 / M4 Recognition")
    thresholds = b.get("threshold_results", b.get("thresholds", []))
    if thresholds:
        df = pd.DataFrame(thresholds)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Benchmark JSON exists but does not expose threshold rows in the expected structure.")

    dist = b.get("distribution_diagnostics", {})
    if dist:
        c1,c2,c3 = st.columns(3)
        c1.metric("Genuine median", f"{dist.get('genuine_median','—')}")
        c2.metric("Impostor median", f"{dist.get('impostor_median','—')}")
        c3.metric("Approx EER", f"{dist.get('eer','—')}")
else:
    st.warning(f"Benchmark artifact not found: `{BENCH.relative_to(ROOT)}`")

st.write("")
st.subheader("Environment / Artifact Audit")
if e:
    st.json(e)
else:
    st.info("Environment audit artifact not found.")

st.write("")
st.subheader("What the benchmark is proving")
st.markdown("""
- **Known recognition:** learned families should be recognized after enrollment.
- **Benign FAR:** normal behavior should rarely match a threat memory.
- **Novel rejection:** a held-out family should remain outside the enrolled memory.
- **Threshold analysis:** the policy should be chosen from measured operating characteristics, not guesswork.
- **Leakage safety:** held-out families and validation/test flows must remain separated from enrollment/training.
""")
