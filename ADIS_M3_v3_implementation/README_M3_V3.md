# ADIS M3 v3 package

## What this changes

M3 v3 replaces the unsupervised PCA-only representation with a supervised
LDA + PCA representation while preserving the existing `behavioral_encoder.joblib`
contract used by M4/ThreatAnalyzer.

Training classes:
- Benign
- Bruteforce
- DoS
- WebAttacks
- Botnet
- Portscan
- DDoS

Novel and completely excluded from M3 training:
- Infiltration

The projector is stored in the artifact's existing `pca` field and exposes:
- `n_features_in_`
- `n_components_`
- `transform(X)`

Therefore existing code that does:
    artifact["pca"].transform(z)
continues to work.

## IMPORTANT

The builder backs up the current v2 artifact to:
    models/behavioral_encoder_v2_backup.joblib

Run from the project root with the normal `.venv` first:

    python -m py_compile src/analysis/supervised_projector.py
    python -m py_compile scripts/build_behavioral_encoder_v3.py
    python -m py_compile scripts/benchmark_m3_m4_recognition_v3.py

Then:

    python -m scripts.build_behavioral_encoder_v3

Then:

    python -m scripts.benchmark_m3_m4_recognition_v3

If v3 is better, keep:
    models/behavioral_encoder.joblib

If v3 is worse, restore:
    Copy-Item models/behavioral_encoder_v2_backup.joblib models/behavioral_encoder.joblib -Force

Do not change M4 thresholds yet. The benchmark is for representation quality first.
