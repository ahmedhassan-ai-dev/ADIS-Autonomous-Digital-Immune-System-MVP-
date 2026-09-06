# ADIS Technical Decisions

## Decision 001 — Synthetic Telemetry

### Decision

Use synthetic cybersecurity telemetry for the hackathon MVP.

### Reason

The objective is to demonstrate the adaptive immune loop rather than train a production intrusion detection model.

Synthetic data allows controlled reproduction of:

* Normal behavior
* Novel behavior
* Threat variants
* First exposure
* Subsequent exposure

---

## Decision 002 — Isolation Forest

### Decision

Use Isolation Forest as the initial anomaly detection algorithm.

### Reason

It is:

* Unsupervised
* Fast
* Lightweight
* Easy to train
* Suitable for the MVP
* Available through scikit-learn

Autoencoders remain an optional future enhancement.

---

## Decision 003 — Docker-Based Isolation

### Decision

Use Docker as the controlled investigation environment.

### Reason

Docker provides lightweight isolation and reproducibility while avoiding the complexity of building a full malware sandbox.

---

## Decision 004 — Qdrant

### Decision

Use Qdrant for Immune Memory vector search.

### Reason

The MVP requires fast similarity-based retrieval of threat representations.

Qdrant provides a suitable vector database for this purpose.

---

## Decision 005 — SQLite

### Decision

Use SQLite for lightweight metadata storage.

### Reason

Threat metadata does not require a separate production database during the hackathon.

---

## Decision 006 — Streamlit

### Decision

Use Streamlit for the MVP dashboard.

### Reason

It enables rapid development of an interactive demonstration interface in Python.

---

## Decision 007 — No RL in MVP

### Decision

Reinforcement Learning is excluded from the initial MVP.

### Reason

The hackathon objective is to validate the core immune loop. RL introduces additional training, safety, and integration complexity without being necessary to demonstrate the central concept.

RL may be evaluated in future development.

---

## Decision 008 — No Real Malware

### Decision

The MVP will use synthetic and controlled behaviors.

### Reason

This provides a safer and more reproducible demonstration while still validating the detection, investigation, memory, and response workflow.
