import streamlit as st

st.title("ℹ️ About ADIS")
st.markdown("## Autonomous Digital Immune System")
st.write("""
ADIS is a proof-of-concept adaptive cybersecurity framework inspired by the human immune system.

The design goal is not simply to stop an attack. The system should treat a suspicious exposure as a
learning opportunity: detect it, contain it safely, investigate its behavior, convert validated evidence
into a reusable representation, remember it, and use that memory during future exposures.
""")

st.subheader("The core hypothesis")
st.info("A threat should not be detected and forgotten. After a validated first exposure, the system should behave differently when the same family or a related variation appears again.")

st.subheader("Architecture")
st.markdown("""
**Telemetry**
→ **M2 Innate Detection**
→ **M3 Behavioral Representation**
→ **M4 Persistent Immune Memory**
→ **Risk & Policy**
→ **Isolation / Investigation**
→ **Validated Learning**
→ **Future Recognition**
""")

st.subheader("What is implemented in the MVP")
items = [
    ("Detection", "A binary first-line detector identifies suspicious activity."),
    ("Behavioral representation", "Production v3 encoder: 77 input features → 32D supervised representation."),
    ("Memory", "Persistent Qdrant-backed vector memory with duplicate prevention and lifecycle metadata."),
    ("Recognition policy", "KNOWN / UNCERTAIN / NOVEL mapped to FAST_PATH / INVESTIGATE / ISOLATE."),
    ("Investigation", "Evidence and risk context are assembled into a structured result."),
    ("Response lifecycle", "ALLOW, INVESTIGATE, ISOLATE and explicit RECOVERY states."),
    ("Replay", "Persistence and restart recognition were validated."),
    ("Benchmarking", "Leakage-safe recognition benchmarking and threshold diagnostics."),
    ("Integration", "M10 executes the end-to-end path and emits a machine-readable result."),
]
for title,desc in items:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(desc)

st.subheader("Important scope")
st.write("This is a controlled proof-of-concept / hackathon MVP, not a replacement for production enterprise security infrastructure.")
