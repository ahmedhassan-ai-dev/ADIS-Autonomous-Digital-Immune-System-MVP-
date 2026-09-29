from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="ADIS — Digital Immune SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Shared frame
st.markdown("""
<style>
:root {
  --adis-bg: #07111f;
  --adis-panel: #0d1b2a;
  --adis-panel2: #102338;
  --adis-text: #e7eef7;
  --adis-muted: #91a4b8;
  --adis-line: rgba(255,255,255,.08);
}
.stApp { background: linear-gradient(180deg,#06101d 0%,#091525 100%); }
[data-testid="stSidebar"] { background: #07111f; border-right: 1px solid var(--adis-line); }
.block-container { max-width: 1500px; padding-top: 1.5rem; }
.adis-title { font-size: 2.25rem; font-weight: 800; letter-spacing: -.03em; }
.adis-subtitle { color: var(--adis-muted); margin-top: -.5rem; }
.adis-chip {
    display:inline-block; padding:.28rem .65rem; border-radius:999px;
    border:1px solid var(--adis-line); background:rgba(255,255,255,.035);
    color:var(--adis-muted); font-size:.8rem; margin-right:.35rem;
}
.adis-card {
    padding: 1rem 1.1rem; border:1px solid var(--adis-line);
    border-radius:14px; background:rgba(13,27,42,.82);
}
.pipeline {
    display:flex; gap:.35rem; align-items:center; flex-wrap:wrap;
    padding:1rem; border:1px solid var(--adis-line); border-radius:14px;
    background:rgba(255,255,255,.02);
}
.node {
    padding:.55rem .75rem; border-radius:10px;
    background:#102338; border:1px solid rgba(255,255,255,.08);
    font-size:.82rem; font-weight:650;
}
.arrow { color:#6f849a; }
.small { color:var(--adis-muted); font-size:.82rem; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🛡️ ADIS")
st.sidebar.caption("Autonomous Digital Immune System")
st.sidebar.divider()
st.sidebar.markdown("**System**")
st.sidebar.markdown("🟢 Production demo ready")
st.sidebar.markdown("M10 end-to-end validated")
st.sidebar.divider()
st.sidebar.caption("Use the navigation above to explore the SOC, project evidence, Lab, Vision, and About.")

pg = st.navigation(
    {
        "OPERATIONS": [
            st.Page("pages/01_Command_Center.py", title="Command Center", icon=":material/security:", default=True),
            st.Page("pages/02_Immune_Loop.py", title="Immune Loop", icon=":material/account_tree:"),
            st.Page("pages/03_Immune_Memory.py", title="Immune Memory", icon=":material/memory:"),
        ],
        "EVIDENCE": [
            st.Page("pages/04_Benchmarks.py", title="Benchmarks", icon=":material/monitoring:"),
            st.Page("pages/05_Project_Stages.py", title="Project Stages", icon=":material/timeline:"),
            st.Page("pages/06_Lab.py", title="ADIS Lab", icon=":material/science:"),
        ],
        "PROJECT": [
            st.Page("pages/07_About.py", title="About ADIS", icon=":material/info:"),
            st.Page("pages/08_Vision.py", title="Vision & Roadmap", icon=":material/auto_awesome:"),
        ],
    }
)
pg.run()
