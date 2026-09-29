import streamlit as st

st.title("🔄 The ADIS Immune Loop")
st.caption("The project's core mechanism — from first exposure to adaptive response.")

st.markdown("""
<div class="pipeline">
<span class="node">1 · Detect</span><span class="arrow">→</span>
<span class="node">2 · Isolate</span><span class="arrow">→</span>
<span class="node">3 · Investigate</span><span class="arrow">→</span>
<span class="node">4 · Represent</span><span class="arrow">→</span>
<span class="node">5 · Remember</span><span class="arrow">→</span>
<span class="node">6 · Recognize</span><span class="arrow">→</span>
<span class="node">7 · Respond</span><span class="arrow">→</span>
<span class="node">8 · Validate / Recover</span>
</div>
""", unsafe_allow_html=True)

st.write("")
st.subheader("How each stage appears in ADIS")

stages = [
    ("M1", "Telemetry & Dataset", "Build a controlled behavioral event/flow representation and validate the data contract."),
    ("M2", "Innate Detection", "Fast first-line anomaly/attack detection. The detector decides whether activity deserves deeper immune processing."),
    ("M3", "Behavioral Representation", "The v3 supervised encoder transforms 77 behavioral features into a 32D embedding."),
    ("M4", "Immune Memory + Policy", "Persistent vector memory compares the embedding with learned threat memories and maps similarity to KNOWN / UNCERTAIN / NOVEL."),
    ("M5", "Investigation Context", "Turn suspicious behavior into evidence and a structured threat profile."),
    ("M6", "Evidence Quality", "Validate that detection, behavior, memory and risk outputs form a coherent evidence chain."),
    ("M7", "Adaptive Response", "Execute controlled ALLOW / INVESTIGATE / ISOLATE / explicit RECOVERY lifecycle transitions."),
    ("M8", "Attack Replay", "Demonstrate persistence, restart recognition and duplicate-memory prevention."),
    ("M9", "Recognition Benchmark", "Stress the memory across thousands of flows and analyze thresholds, benign FAR and novel rejection."),
    ("M10", "End-to-End System", "Run the full loop as one demonstrable system and emit a final result artifact."),
]
for code, title, desc in stages:
    with st.container(border=True):
        st.markdown(f"### {code} · {title}")
        st.write(desc)

st.info("Demo principle: a first exposure should be slower and investigation-heavy; a subsequent related exposure should benefit from persistent immune knowledge.")
