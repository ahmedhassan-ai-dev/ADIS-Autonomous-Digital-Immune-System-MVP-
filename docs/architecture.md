# ADIS Architecture

## High-Level Architecture

```text
                  ┌──────────────────┐
                  │ Synthetic        │
                  │ Telemetry        │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Anomaly          │
                  │ Detection        │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Isolation /      │
                  │ Deception        │
                  │ Environment      │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Behavioral       │
                  │ Analysis         │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Threat           │
                  │ Representation   │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Immune Memory    │
                  │ Qdrant           │
                  └────────┬─────────┘
                           │
                    Future Exposure
                           │
                           ▼
                  ┌──────────────────┐
                  │ Similarity       │
                  │ Recognition      │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Response /       │
                  │ Containment      │
                  └──────────────────┘
```

## Component Responsibilities

### Telemetry

Responsible for generating or receiving normalized security events.

Input:

* Synthetic event configuration

Output:

* Structured telemetry records

### Detection

Responsible for identifying deviations from learned normal behavior.

Initial model:

* Isolation Forest

Output:

* Anomaly score
* Suspicious / normal decision

### Sandbox

Responsible for safely simulating and observing suspicious behavior.

Output:

* Behavioral events

### Analysis

Converts raw events into structured behavioral characteristics.

Output:

* Threat feature representation
* MITRE ATT&CK mapping
* Confidence information

### Memory

Stores validated threat representations and enables similarity retrieval.

Technology:

* Qdrant

### Response

Executes safe containment actions based on detection and memory confidence.

### Pipeline

Coordinates the complete immune loop.

### Dashboard

Provides observability into the system state and demonstration scenario.

## Design Principle

Every component should have:

* A defined input
* A defined output
* A clear responsibility
* Independent testability

Components should communicate through simple structured data rather than tightly coupled implementation details.
