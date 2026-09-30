# ADIS Dashboard

Ready-to-run Streamlit dashboard for the ADIS Autonomous Digital Immune System MVP.

The dashboard narrative, architecture, milestones, and documented operating
policy are grounded in `docs/ADIS_COMPREHENSIVE_DOCUMENTATION.md`. The Command
Center also includes a live flow simulator that runs each submitted synthetic
or repository CIC-IDS2017 runtime flow through `RealADISPipeline` and the M7
simulated response lifecycle. Replaying an exact flow compares first and later
memory recognition. Simulator memory is isolated per browser session under the
system temporary directory.

The Command Center validation button invokes the repository's existing M4
recognition, M5 decision/context, and M7 response lifecycle scripts and reports
each script's pass/fail status and output.

## Run

From the ADIS repository root:

```powershell
python -m streamlit run dashboard/app.py
```

The dashboard expects the repository's existing `models/`, `results/`, `src/`, and `scripts/` directories.

## Pages

- Command Center
- Immune Loop
- Immune Memory
- Benchmarks
- Project Stages
- ADIS Lab
- About ADIS
- Vision & Roadmap

Model artifacts are loaded by the pipeline and are not presented as raw files
in the SOC interface.

The Lab only exposes repository-owned scripts through an explicit allowlist.
