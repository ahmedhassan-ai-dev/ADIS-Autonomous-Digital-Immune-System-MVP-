from pathlib import Path
import subprocess, sys, json, time
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

# Only whitelisted, repository-owned experiments are exposed.
EXPERIMENTS = {
    "M10 — Final Immune Loop": {
        "module": "scripts.final_immune_loop",
        "result": ROOT / "results" / "m10_final_demo.json",
        "description": "Run the controlled final end-to-end immune loop."
    },
    "M10 — Final System Audit": {
        "module": "scripts.m10_final_audit",
        "result": ROOT / "results" / "m10_final_audit.json",
        "description": "Validate production artifacts and M10 integration health."
    },
    "M3/M4 v3 — Recognition Benchmark": {
        "module": "scripts.benchmark_m3_m4_recognition_v3",
        "result": ROOT / "results" / "m3_m4_recognition_benchmark_v3" / "benchmark_v3.json",
        "description": "Run the leakage-safe recognition benchmark."
    },
}

st.title("🧪 ADIS Lab")
st.caption("A controlled experiment surface for the same reproducible scripts used during development.")

name = st.selectbox("Experiment", list(EXPERIMENTS))
cfg = EXPERIMENTS[name]
st.write(cfg["description"])
st.code(f"python -m {cfg['module']}", language="powershell")

if st.button("▶ Run experiment", type="primary"):
    with st.status(f"Running {name}…", expanded=True) as status:
        start = time.perf_counter()
        try:
            p = subprocess.run(
                [PYTHON, "-m", cfg["module"]],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=300,
            )
            elapsed = time.perf_counter() - start
            st.write(f"Exit code: `{p.returncode}` · {elapsed:.1f}s")
            if p.stdout:
                st.code(p.stdout[-12000:], language="text")
            if p.stderr:
                st.warning(p.stderr[-8000:])
            if p.returncode == 0:
                status.update(label="Experiment completed", state="complete")
            else:
                status.update(label="Experiment failed", state="error")
        except subprocess.TimeoutExpired:
            status.update(label="Experiment timed out", state="error")
            st.error("The experiment exceeded the 5-minute Lab limit.")

st.divider()
st.subheader("Recent artifact")
if cfg["result"].exists():
    try:
        payload = json.loads(cfg["result"].read_text(encoding="utf-8"))
        st.json(payload)
    except Exception:
        st.info(f"Artifact exists: `{cfg['result'].relative_to(ROOT)}`")
else:
    st.info("No result artifact yet. Run the experiment above.")
