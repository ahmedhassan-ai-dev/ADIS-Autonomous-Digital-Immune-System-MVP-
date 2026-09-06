# ADIS MVP Scope

## Objective

The ADIS hackathon MVP demonstrates a complete digital immune loop in a controlled environment.

The MVP is not intended to provide production-grade enterprise cybersecurity protection. Its purpose is to validate the core adaptive defense concept.

## Core Immune Loop

```text
Threat Exposure
      ↓
Anomaly Detection
      ↓
Isolation
      ↓
Behavioral Investigation
      ↓
Threat Representation
      ↓
Immune Memory
      ↓
Subsequent Recognition
      ↓
Automated Containment
```

## MVP Components

### 1. Synthetic Telemetry

Generate controlled host and network telemetry representing:

* Normal activity
* Suspicious process behavior
* Authentication anomalies
* Suspicious network activity
* Attack variants

No real malware is required.

### 2. Anomaly Detection

Use an unsupervised anomaly detection model.

Initial implementation:

* Isolation Forest

Optional future enhancement:

* Autoencoder

### 3. Isolation / Deception

Suspicious activity is routed to a controlled Docker-based investigation environment.

The environment records simulated:

* Process events
* File activity
* Network activity
* Command execution

### 4. Behavioral Analysis

The system extracts meaningful behavioral characteristics from collected telemetry.

Examples:

* Failed authentication count
* Process creation frequency
* Network connection frequency
* Destination port patterns
* File operation frequency
* Process relationships

### 5. Immune Memory

Validated threat representations are stored in a vector database.

Initial implementation:

* Qdrant

Metadata:

* SQLite

### 6. Subsequent Recognition

A repeated or behaviorally related threat is compared against Immune Memory.

The system calculates a similarity score and determines whether previously learned knowledge can be reused.

### 7. Automated Containment

High-confidence matches trigger a simulated containment action.

Examples:

* Block
* Isolate
* Reject connection
* Terminate simulated process

### 8. Dashboard

Streamlit provides a visual representation of the immune loop.

The dashboard should show:

* Current telemetry
* Anomaly score
* Detection event
* Isolation event
* Behavioral analysis
* Memory encoding
* Memory match
* Similarity score
* Containment action
* First-exposure vs subsequent-exposure latency

## Success Criteria

The MVP is considered successful when it can demonstrate:

1. A suspicious behavior is detected.
2. The behavior is isolated.
3. Its characteristics are analyzed.
4. Validated knowledge is stored in Immune Memory.
5. A related behavior is introduced again.
6. The system retrieves the relevant memory.
7. The related threat is recognized.
8. The system performs faster automated containment.

## Out of Scope

The following are not required for the hackathon MVP:

* Real malware execution
* Production enterprise deployment
* Reinforcement Learning
* Large Language Models
* Multi-agent systems
* Kubernetes
* Large-scale distributed infrastructure
* Advanced adversarial attack generation
* Full endpoint detection and response platform

## Development Priority

The priority order is:

**Working end-to-end immune loop > advanced AI features > UI polish**
