from pathlib import Path
import subprocess, sys, json, time
import streamlit as st

ROOT=Path(__file__).resolve().parents[2]
PYTHON=sys.executable
EXPERIMENTS={
"M10 — Final Immune Loop":("scripts.final_immune_loop",ROOT/"results"/"m10_final_demo.json","Run the final end-to-end loop."),
"M10 — Final System Audit":("scripts.m10_final_audit",ROOT/"results"/"m10_final_audit.json","Audit production integration."),
"M3/M4 v3 — Recognition Benchmark":("scripts.benchmark_m3_m4_recognition_v3",ROOT/"results"/"m3_m4_recognition_benchmark_v3"/"benchmark_v3.json","Run the leakage-safe recognition benchmark."),
"M3 v3 — Environment Audit":("scripts.audit_m3_v3_environment",ROOT/"results"/"m3_m4_recognition_benchmark_v3"/"environment_audit.json","Audit Python and production artifact compatibility."),
}

st.title("🧪 ADIS Lab")
st.caption("Run repository-owned, reproducible experiments without leaving the SOC interface.")

name=st.selectbox("Experiment",list(EXPERIMENTS))
module,result,desc=EXPERIMENTS[name]
st.write(desc)
st.code(f"python -m {module}",language="powershell")

if st.button("▶ Run experiment",type="primary"):
    start=time.perf_counter()
    with st.status(f"Running {name}…",expanded=True) as status:
        try:
            p=subprocess.run([PYTHON,"-m",module],cwd=str(ROOT),capture_output=True,text=True,timeout=300)
            st.write(f"Exit code: `{p.returncode}` · {time.perf_counter()-start:.1f}s")
            if p.stdout: st.code(p.stdout[-14000:],language="text")
            if p.stderr: st.warning(p.stderr[-9000:])
            status.update(label="Completed" if p.returncode==0 else "Failed",state="complete" if p.returncode==0 else "error")
        except subprocess.TimeoutExpired:
            status.update(label="Timed out",state="error")
            st.error("Experiment exceeded 5 minutes.")

st.divider()
st.subheader("Latest artifact")
if result.exists():
    try: st.json(json.loads(result.read_text(encoding="utf-8")))
    except: st.info(f"Artifact: `{result.relative_to(ROOT)}`")
else: st.info("No artifact yet.")
