# ADIS Dashboard v2

This package replaces the presentation layer with a multipage Streamlit SOC/Research interface.

## Pages

1. Command Center — live M10 result
2. Immune Loop — M1→M10 explanation
3. Immune Memory — persistence / encoder contract
4. Benchmarks — M3 v3/M4 evidence
5. Project Stages — complete project evolution
6. ADIS Lab — controlled, whitelisted experiment runner
7. About ADIS — project details and implemented capabilities
8. Vision & Roadmap — future direction

## Install

Copy the `dashboard/` directory from this package into the project root and replace the existing
dashboard entrypoint.

Run:

```powershell
python -m py_compile dashboard\app.py
python -m py_compile dashboard\pages\01_Command_Center.py
python -m py_compile dashboard\pages\02_Immune_Loop.py
python -m py_compile dashboard\pages\03_Immune_Memory.py
python -m py_compile dashboard\pages\04_Benchmarks.py
python -m py_compile dashboard\pages\05_Project_Stages.py
python -m py_compile dashboard\pages\06_Lab.py
python -m py_compile dashboard\pages\07_About.py
python -m py_compile dashboard\pages\08_Vision.py
```

Then:

```powershell
python -m streamlit run dashboard\app.py
```

The Lab intentionally exposes only repository-owned ADIS scripts. It does not accept arbitrary shell
commands from the browser.

## Expected project artifacts

The UI dynamically reads the current repository artifacts when available, including:

- results/m10_final_demo.json
- results/m10_final_audit.json
- results/m3_m4_recognition_benchmark_v3/benchmark_v3.json
- results/m3_m4_recognition_benchmark_v3/environment_audit.json
- models/behavioral_encoder.joblib
- models/immune_memory*

No model retraining is performed by the dashboard itself.
