from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

st.title("Immune memory", icon=":material/memory:")
st.caption("How ADIS turns validated threat behavior into persistent recognition.")

st.markdown("""
The M4 memory compares each 32-dimensional behavioral representation with enrolled threat patterns. It distinguishes a strong match from an ambiguous near-match and behavior that is novel to memory.
""")

cols = st.columns(3)
cols[0].metric("Behavior representation", "32 dimensions")
cols[1].metric("Known match", "Similarity ≥ 0.92")
cols[2].metric("Novel behavior", "Similarity < 0.75")

st.subheader("Documented recognition routes")
st.dataframe([
    {"Similarity": "S ≥ 0.92", "Recognition": "KNOWN", "SOC route": "Memory-assisted fast path"},
    {"Similarity": "0.75 ≤ S < 0.92", "Recognition": "UNCERTAIN", "SOC route": "Investigate"},
    {"Similarity": "S < 0.75", "Recognition": "NOVEL", "SOC route": "Investigate / isolate"},
], hide_index=True, width="stretch")

st.subheader("Why persistent memory matters")
st.write("A first exposure can require investigation and validation. When the system sees related behavior again, memory can provide a similarity match and inform the response. The documented M8 replay stage checks persistence across restarts and duplicate prevention.")

memory_dirs = [path for path in (ROOT / "models").glob("immune_memory*") if path.is_dir()]
st.metric("Memory stores present in this workspace", len(memory_dirs))
st.caption("Store contents and serialized model artifacts are kept out of the SOC view.")
