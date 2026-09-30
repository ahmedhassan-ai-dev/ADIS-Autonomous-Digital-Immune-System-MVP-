import streamlit as st
st.title("🔄 ADIS Immune Loop")
st.caption("The core adaptive defense mechanism.")

st.markdown("""<div class="pipeline">
<span class="node">1 Detect</span><span class="arrow">→</span><span class="node">2 Isolate</span><span class="arrow">→</span>
<span class="node">3 Investigate</span><span class="arrow">→</span><span class="node">4 Represent</span><span class="arrow">→</span>
<span class="node">5 Remember</span><span class="arrow">→</span><span class="node">6 Recognize</span><span class="arrow">→</span>
<span class="node">7 Respond</span><span class="arrow">→</span><span class="node">8 Validate / Recover</span>
</div>""",unsafe_allow_html=True)

st.write("")
st.subheader("First exposure vs. re-exposure")
a,b=st.columns(2)
with a:
    st.markdown("### 🧬 Primary exposure")
    st.write("Unknown behavior is detected, isolated/investigated, represented and—after validation—stored as reusable immune knowledge.")
with b:
    st.markdown("### ⚡ Secondary exposure")
    st.write("A related behavior is compared against persistent memory. A high-confidence match can enter the memory-assisted fast path.")

st.divider()
st.subheader("M1 → M10 implementation map")
stages=[
("M1","Telemetry & Dataset","Data contract, validation and CIC-style flow preparation."),
("M2","Innate Detection","Fast first-line suspicious/attack detection."),
("M3","Behavioral Representation","77 features → 32D supervised v3 representation."),
("M4","Immune Memory + Policy","Persistent vector memory + similarity + KNOWN/UNCERTAIN/NOVEL policy."),
("M5","Investigation Context","Threat evidence and structured behavior profile."),
("M6","Evidence Quality","Consistency and evidence-chain validation."),
("M7","Adaptive Response","Controlled ALLOW / INVESTIGATE / ISOLATE / RECOVERY lifecycle."),
("M8","Attack Replay","Persistence, restart recognition and duplicate-memory prevention."),
("M9","Recognition Benchmark","Large-scale threshold and novel-family evaluation."),
("M10","End-to-End System","Full integrated loop with machine-readable final artifact."),
]
for code,title,desc in stages:
    with st.container(border=True):
        x,y=st.columns([.16,.84]); x.markdown(f"### {code}"); y.markdown(f"**{title}**  \n{desc}")
