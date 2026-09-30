import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

st.set_page_config(
    page_title="ADIS — Digital Immune SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
  --bg:#06101d; --panel:#0c1b2b; --panel2:#10263c;
  --text:#e8f0f8; --muted:#8fa4b8; --line:rgba(255,255,255,.09);
  --good:#35d07f; --warn:#ffb84d; --bad:#ff6675; --cyan:#55c7ff;
}
.stApp {background:linear-gradient(180deg,#06101d 0%,#091626 100%); color:var(--text);}
[data-testid="stSidebar"]{background:#07111f;border-right:1px solid var(--line);}
.block-container{max-width:1550px;padding-top:1.4rem;}
.adis-hero{padding:1.25rem 1.35rem;border:1px solid var(--line);border-radius:18px;
background:linear-gradient(135deg,rgba(16,38,60,.95),rgba(9,23,38,.9));margin-bottom:1rem;}
.adis-title{font-size:2.35rem;font-weight:850;letter-spacing:-.035em;}
.adis-subtitle{color:var(--muted);font-size:1rem;}
.adis-chip{display:inline-block;padding:.28rem .65rem;border-radius:999px;border:1px solid var(--line);
background:rgba(255,255,255,.035);color:var(--muted);font-size:.78rem;margin:.2rem .25rem 0 0;}
.adis-card{padding:1rem 1.1rem;border:1px solid var(--line);border-radius:14px;background:rgba(12,27,43,.82);}
.pipeline{display:flex;gap:.4rem;align-items:center;flex-wrap:wrap;padding:1rem;
border:1px solid var(--line);border-radius:14px;background:rgba(255,255,255,.02);}
.node{padding:.55rem .75rem;border-radius:10px;background:#10263c;border:1px solid var(--line);
font-size:.82rem;font-weight:650;}
.arrow{color:#71879d}.small{color:var(--muted);font-size:.82rem;}
[data-testid="stMetric"]{background:rgba(12,27,43,.7);border:1px solid var(--line);
padding:.75rem;border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## ADIS")
st.sidebar.caption("Autonomous Digital Immune System")
st.sidebar.caption("SOC workspace · M1–M10")
st.sidebar.divider()
st.sidebar.caption("Project narrative and architecture follow `docs/ADIS_COMPREHENSIVE_DOCUMENTATION.md`.")

pg = st.navigation({
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
})
pg.run()
