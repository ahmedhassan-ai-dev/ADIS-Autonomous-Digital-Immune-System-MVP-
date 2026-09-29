from pathlib import Path
import json
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
mem_root = ROOT / "models"
results_root = ROOT / "results"

st.title("🧠 Immune Memory")
st.caption("Persistent threat knowledge — not a static signature list.")

encoder = ROOT / "models" / "behavioral_encoder.joblib"
stages = [
    ("Embedding", "32D behavioral representation"),
    ("Distance", "Cosine similarity"),
    ("Persistence", "Qdrant local persistent storage"),
    ("Policy", "KNOWN ≥ 0.90 · UNCERTAIN ≥ 0.75 · NOVEL < 0.75"),
]

c1,c2,c3,c4 = st.columns(4)
for col,(label,value) in zip((c1,c2,c3,c4),stages):
    col.metric(label,value)

st.write("")
st.subheader("Current artifacts")
for p in sorted(mem_root.glob("immune_memory*")):
    if p.is_dir():
        st.write(f"📦 `{p.relative_to(ROOT)}`")

st.write("")
st.subheader("Why persistence matters")
st.write("""
The memory is intended to survive process restarts so that knowledge learned from a validated exposure
can be reused later. M8 explicitly tested persistence/restart recognition and duplicate-memory prevention.
""")

with st.expander("Encoder contract"):
    st.write("Production artifact:", encoder.relative_to(ROOT) if encoder.exists() else "missing")
    st.write("Version: `adis-behavioral-encoder-v3-supervised`")
    st.write("Features: `77`")
    st.write("Embedding dimension: `32`")
    st.write("Schema hash: `aee55fd98f19d60a0bcab57ca3607eff852e2e95c315f5207ab3d66a715ed320`")
