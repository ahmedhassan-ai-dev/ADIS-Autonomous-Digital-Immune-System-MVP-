# ADIS — Autonomous Digital Immune System

> **An adaptive cybersecurity framework inspired by the human immune system.**

ADIS is a proof-of-concept autonomous cyber defense platform designed to detect suspicious network behavior, investigate novel threats, learn reusable behavioral representations, preserve validated knowledge in persistent **Immune Memory**, and respond through a controlled adaptive response lifecycle.

The system is built around a continuous cyber-defense loop:

```text
Telemetry
    ↓
M2 — Innate Detection
    ↓
M3 — Behavioral Representation
    ↓
M4 — Immune Memory Recognition
    ↓
Known / Uncertain / Novel
    ↓
M6 — Investigation & Evidence
    ↓
M7 — Adaptive Response
    ↓
Isolation / Validation / Recovery
    ↓
Persistent Learning
```

---

## 🎯 Why ADIS?

Traditional detection systems often treat each suspicious event independently.

ADIS explores a different approach:

> **A threat that has already been investigated and validated should become reusable security knowledge.**

This creates an adaptive defense cycle:

```text
Unknown Threat
      ↓
Detect
      ↓
Investigate
      ↓
Learn
      ↓
Remember
      ↓
Recognize on Re-exposure
      ↓
Respond Faster
```

The project therefore combines:

* Behavioral anomaly detection
* Supervised behavioral representation learning
* Persistent vector-based threat memory
* Similarity-based recognition
* Evidence-driven investigation
* Policy-controlled response
* Isolation and recovery lifecycle
* End-to-end attack replay
* Large-scale recognition benchmarking
* SOC-style visualization

---

# 🧠 System Architecture

```text
                         ┌─────────────────────┐
                         │   Network Telemetry │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ M2 — Innate Layer   │
                         │ Anomaly Detection   │
                         └──────────┬──────────┘
                                    │
                           Suspicious Flow
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ M3 — Behavioral     │
                         │ Representation      │
                         │ Encoder v3          │
                         └──────────┬──────────┘
                                    │
                              32D Embedding
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ M4 — Immune Memory  │
                         │ Persistent Qdrant   │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
                    KNOWN                     NOVEL
                       │                         │
                       ▼                         ▼
                 Fast Path                  Investigation
                                                 │
                                                 ▼
                                      Evidence & Risk Analysis
                                                 │
                                                 ▼
                                      M7 — Response Lifecycle
                                                 │
                                    ┌────────────┴────────────┐
                                    ▼                         ▼
                                 ISOLATE                   RECOVERY
```

---

# 🔬 Project Milestones

| Milestone          | Component                            | Status |
| ------------------ | ------------------------------------ | -----: |
| M1                 | Dataset & Telemetry Pipeline         |      ✅ |
| M2                 | Innate Anomaly Detection             |      ✅ |
| M3                 | Behavioral Representation            |      ✅ |
| M3 v3              | Supervised Behavioral Encoder        |      ✅ |
| M4                 | Persistent Immune Memory             |      ✅ |
| M4 Policy          | Recognition Policy                   |      ✅ |
| M6                 | Investigation & Evidence             |      ✅ |
| M7                 | Adaptive Response Lifecycle          |      ✅ |
| M8                 | End-to-End Attack Replay             |      ✅ |
| M9                 | Large-Scale Recognition Benchmark    |      ✅ |
| M10                | Final End-to-End Immune Loop         |      ✅ |
| Dashboard          | SOC / Immune Visualization           |      ✅ |
| Full Documentation | Architecture + Experiments + Results |     🔄 |

---

# 🧬 M3 — Behavioral Representation

The final behavioral representation uses the production artifact:

```text
Encoder:
adis-behavioral-encoder-v3-supervised

Representation:
supervised_lda_plus_pca

Input features:
77

Embedding:
32 dimensions

Training rows:
35,748

Validation rows:
8,938
```

The encoder was trained using a leakage-safe split and explicitly excludes:

```text
Infiltration
```

from training so that it can be used as a held-out novel family during recognition evaluation.

The production encoder is stored as:

```text
models/behavioral_encoder.joblib
```

---

# 🛡️ M4 — Immune Memory

ADIS uses persistent vector-based Immune Memory to store validated threat representations.

The memory layer provides:

* Persistent storage
* Similarity search
* Threat recognition
* Re-exposure tracking
* Duplicate-memory prevention
* Restart persistence
* Encoder/schema compatibility validation

The production embedding contract is:

```text
Embedding dimension : 32
Encoder version     : adis-behavioral-encoder-v3-supervised
Schema hash          : aee55fd98f19d60a0bcab57ca3607eff852e2e95c315f5207ab3d66a715ed320
```

The memory backend uses local persistent Qdrant storage.

---

# ⚖️ M4 Recognition Policy

The current recognition policy is:

| Similarity | Classification | Action      |
| ---------: | -------------- | ----------- |
|     ≥ 0.90 | KNOWN          | FAST_PATH   |
|  0.75–0.90 | UNCERTAIN      | INVESTIGATE |
|     < 0.75 | NOVEL          | ISOLATE     |

The policy is intentionally separated from the raw vector database.

This allows the memory layer to retrieve evidence while the policy layer determines the operational decision.

---

# 🔎 M6 — Investigation

When ADIS encounters behavior that cannot be confidently recognized, it moves toward investigation rather than blindly treating the event as known.

The investigation layer combines evidence from multiple system components:

```text
M2 Detection
      +
M3 Behavioral Analysis
      +
Flow Features
      +
M4 Recognition
      +
Risk Engine
      ↓
Investigation Evidence
```

The M6.4 evidence-quality test validated integration of:

* Detection decision
* Behavioral family
* Flow-level features
* Memory recognition
* Similarity
* Risk score
* Severity

---

# 🔄 M7 — Adaptive Response Lifecycle

ADIS implements an explicit response lifecycle rather than a single irreversible action.

Supported actions include:

```text
ALLOW
INVESTIGATE
ISOLATE
RECOVER
```

Example:

```text
ISOLATE
   ↓
ISOLATED
   ↓
INVESTIGATING
   ↓
VALIDATED
   ↓
RECOVERED
```

Recovery is deliberately explicit and does not automatically happen immediately after isolation.

This allows the response policy to require successful validation before recovery.

---

# 🧪 M8 — Attack Replay

M8 validates the complete lifecycle using a controlled attack replay.

The replay validates:

1. Threat detection
2. Behavioral encoding
3. Memory recognition
4. Isolation
5. Investigation
6. Persistence
7. Restart recognition
8. Duplicate-memory prevention

The test also verifies that Immune Memory survives process restart.

---

# 📊 M9 — Recognition Benchmark

ADIS was evaluated across multiple CICIDS2017 behavioral families.

The benchmark includes:

```text
Benign
Botnet
Bruteforce
DDoS
DoS
Infiltration
Portscan
WebAttacks
```

The large-scale benchmark evaluated:

```text
8,000 flows
```

using 1,000 flows per dataset.

The benchmark uses leakage-safe enrollment and keeps the novel family outside the enrolled memory.

---

# 🧪 M3/M4 v3 Recognition Results

The final M3 v3 representation significantly improved the separation between genuine and impostor behavior compared with the previous representation.

Observed distributions:

```text
Genuine similarity
median ≈ 0.9148

Impostor similarity
median ≈ -0.0859
```

Approximate equal-error-rate analysis:

```text
EER ≈ 0.0676
threshold ≈ 0.387
```

Operational results:

| Threshold | Known Recognition | Benign FAR | Novel Rejection |
| --------: | ----------------: | ---------: | --------------: |
|      0.75 |            97.33% |     19.60% |          77.78% |
|      0.80 |            96.42% |     14.30% |          88.89% |
|      0.85 |            96.33% |     11.60% |          97.22% |
|  **0.90** |        **95.58%** |  **8.60%** |        **100%** |
|      0.92 |            95.25% |      7.70% |            100% |
|      0.95 |            93.33% |      6.20% |            100% |

The production M4 policy currently uses:

```text
Known threshold = 0.90
Near threshold  = 0.75
```

The important result is not simply a single threshold value, but the improved separation produced by M3 v3.

---

# 🧠 Why M3 v3 Matters

The earlier representation produced substantial overlap between genuine and impostor similarities.

M3 v3 changes the representation objective toward:

```text
Same behavioral family
        ↓
Similar embeddings

Different behavioral families
        ↓
Separated embeddings
```

This makes Immune Memory more useful as an actual recognition mechanism rather than simply a nearest-neighbor lookup layer.

---

# 🚀 M10 — Final Immune Loop

The final M10 demonstration connects the major ADIS components into one end-to-end pipeline.

```text
Network Flow
     ↓
M2 Detection
     ↓
M3 Behavioral Encoding
     ↓
M4 Immune Memory
     ↓
Recognition Policy
     ↓
Risk Engine
     ↓
Response Policy
     ↓
Isolation / Investigation
     ↓
Recovery
```

The final M10 demonstration successfully executed the complete loop.

Example final decision:

```text
Detector:
ATTACK

Confidence:
0.9776

Risk:
0.9776

Severity:
CRITICAL

Recognition:
NOVEL

Decision:
ISOLATE
```

The result is written to:

```text
results/m10_final_demo.json
```

---

# 📈 Current Project Status

### Core Immune Loop

```text
M1  ██████████ 100%
M2  ██████████ 100%
M3  ██████████ 100%
M4  ██████████ 100%
M6  ██████████ 100%
M7  ██████████ 100%
M8  ██████████ 100%
M9  ██████████ 100%
M10 ██████████ 100%
```

These percentages represent engineering milestone completion, not model accuracy.

---

# 🖥️ Dashboard

ADIS includes a SOC-style dashboard designed to visualize the immune-defense process.

The dashboard presents the system as an operational security workflow rather than only exposing machine-learning metrics.

Planned/available views include:

* System overview
* Detection events
* Behavioral analysis
* Immune Memory
* Recognition decisions
* Investigation evidence
* Risk assessment
* Response lifecycle
* Attack replay
* Experimental laboratory

---

# 🧪 ADIS Laboratory

The project also provides a laboratory-style interface for running controlled experiments.

The Lab is intended to expose the same concepts validated through the project's experimental scripts:

```text
Dataset
   ↓
Detection
   ↓
Embedding
   ↓
Memory Search
   ↓
Recognition
   ↓
Risk
   ↓
Policy
   ↓
Response
```

This makes the research and engineering results interactively demonstrable.

---

# 🗂️ Project Structure

```text
ADIS/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── runtime/
│   └── results/
│
├── models/
│   ├── behavioral_encoder.joblib
│   └── ...
│
├── src/
│   ├── analysis/
│   ├── detection/
│   ├── memory/
│   ├── pipeline/
│   ├── response/
│   └── sandbox/
│
├── dashboard/
│
├── scripts/
│   ├── build_behavioral_encoder_v3.py
│   ├── benchmark_m3_m4_recognition_v3.py
│   ├── test_m64_evidence_quality.py
│   ├── test_m7_response_lifecycle.py
│   ├── test_m8_attack_replay.py
│   ├── benchmark_m9_recognition.py
│   ├── final_immune_loop.py
│   └── m10_final_audit.py
│
├── results/
│
├── docs/
│
├── tests/
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. Clone

```bash
git clone <REPOSITORY_URL>
cd ADIS-Autonomous-Digital-Immune-System-MVP-
```

## 2. Create virtual environment

```powershell
python -m venv .venv
```

## 3. Activate

```powershell
.\.venv\Scripts\Activate.ps1
```

## 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

---

# ▶️ Run the Final Demo

Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m scripts.final_immune_loop
```

Expected high-level flow:

```text
ATTACK
  ↓
Behavior Analysis
  ↓
Memory Recognition
  ↓
Risk
  ↓
Policy
  ↓
Response
```

---

# 🧪 Run Validation Experiments

### M6.4 — Evidence Quality

```powershell
python -m scripts.test_m64_evidence_quality
```

### M7 — Response Lifecycle

```powershell
python -m scripts.test_m7_response_lifecycle
```

### M8 — Attack Replay

```powershell
python -m scripts.test_m8_attack_replay
```

### M9 — Recognition Benchmark

```powershell
python -m scripts.benchmark_m9_recognition --max-per-file 1000 --memory-path models/immune_memory_m9_benchmark
```

### M3/M4 v3 Benchmark

```powershell
python -m scripts.benchmark_m3_m4_recognition_v3
```

### M10 Final Demo

```powershell
python -m scripts.final_immune_loop
```

### M10 Final Audit

```powershell
python -m scripts.m10_final_audit
```

---

# 🔐 Security & Safety Model

ADIS is a controlled proof-of-concept.

The system does not treat machine-learning predictions as unrestricted authority.

Instead:

```text
Detection
   ↓
Behavior
   ↓
Memory
   ↓
Risk
   ↓
Policy & Safety Engine
   ↓
Response
```

This separation is important because a false-positive detection should not automatically imply an irreversible response.

---

# ⚠️ Limitations

ADIS is currently a research and hackathon-oriented proof-of-concept.

It is **not** intended to replace production enterprise security infrastructure.

Important limitations include:

* Evaluation is based primarily on controlled datasets.
* The behavioral representation is trained on CICIDS2017-derived traffic families.
* Novel-family evaluation uses Infiltration as a held-out family.
* The response environment is controlled/simulated.
* Production-scale endpoint deployment has not been demonstrated.
* The current system does not represent a complete enterprise SOC.
* The current detector and encoder artifacts were produced under different scikit-learn versions, and loading the detector may emit an `InconsistentVersionWarning`.

The last point should be treated as a reproducibility/maintenance issue rather than hidden.

---

# 🔮 Future Work

Potential extensions include:

* Continual learning
* Online threat-memory evolution
* Multi-host correlation
* Attack graph construction
* Threat-intelligence enrichment
* Endpoint telemetry
* Identity-aware detection
* Distributed Immune Memory
* Multi-agent cyber defense
* Advanced adversarial evaluation
* Automated recovery
* Enterprise SIEM integration
* Cloud deployment
* LLM-assisted security reasoning

The LLM should remain behind the policy and safety layer rather than becoming unrestricted authority over security actions.

---

# 📚 Documentation

The full technical documentation is maintained separately from this README.

Recommended documentation structure:

```text
docs/
│
├── README.md
├── architecture.md
├── immune-loop.md
├── dataset.md
├── m1-data-pipeline.md
├── m2-detection.md
├── m3-behavioral-representation.md
├── m3-v3-supervised-encoder.md
├── m4-immune-memory.md
├── m4-recognition-policy.md
├── m6-investigation.md
├── m7-response-lifecycle.md
├── m8-attack-replay.md
├── m9-recognition-benchmark.md
├── m10-final-loop.md
├── experiments/
│   ├── recognition-benchmark.md
│   ├── threshold-analysis.md
│   └── evidence-quality.md
└── final-report.md
```

---

# 👥 Project

**ADIS — Autonomous Digital Immune System**

**Track:** AI / Cybersecurity

**Stage:** Hackathon MVP / Research Proof of Concept

---

# 📄 License

To be defined.

---

## ⭐ Project Principle

> **A cyber defense system should not only detect threats. It should learn from validated encounters and become better prepared for the next exposure.**
