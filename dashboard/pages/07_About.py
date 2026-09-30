import streamlit as st
st.title("ℹ️ About ADIS")
st.caption("Autonomous Digital Immune System — Hackathon MVP / Proof of Concept")

st.markdown("""
## The idea
ADIS is an adaptive cybersecurity framework inspired by biological immune systems.

The key idea is **not** to detect an attack and forget it. A suspicious exposure can become a validated
behavioral memory that changes how the system handles related future exposures.
""")

st.info("Core hypothesis: first exposure → investigate and learn; re-exposure → recognize faster and respond with memory-assisted confidence.")

st.subheader("Architecture")
st.code("""Telemetry
   ↓
M2 Innate Detection
   ↓
M3 Behavioral Representation
   ↓
M4 Persistent Immune Memory
   ↓
Risk + Policy + Safety
   ↓
Isolation / Investigation
   ↓
Validated Learning
   ↓
Future Recognition""")

st.subheader("Implemented MVP capabilities")
items=[
("M1 Data foundation","Controlled telemetry and CIC-style network-flow preparation."),
("M2 Detection","Binary first-line suspicious/attack detection."),
("M3 Representation","77 behavioral features mapped to a 32D supervised representation."),
("M4 Memory","Persistent vector memory and similarity-based recognition."),
("Policy","KNOWN / UNCERTAIN / NOVEL routed to FAST_PATH / INVESTIGATE / ISOLATE."),
("Investigation","Structured evidence and risk context."),
("Response","Controlled response lifecycle including isolation and recovery."),
("Replay & benchmark","Persistence/restart testing and leakage-safe recognition benchmarking."),
("M10 integration","Single end-to-end executable loop with machine-readable output."),
]
for title,desc in items:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(desc)

st.subheader("Scope")
st.warning("ADIS is a controlled proof-of-concept / hackathon MVP. It is not a replacement for enterprise security infrastructure.")
