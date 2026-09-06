# ADIS Immune Loop

## Biological Analogy

ADIS is inspired by several high-level mechanisms of the biological immune system:

| Biological Immune System | ADIS                            |
| ------------------------ | ------------------------------- |
| Self                     | Normal system behavior          |
| Non-Self                 | Suspicious / anomalous behavior |
| Antigen                  | Threat representation           |
| Immune Response          | Automated containment           |
| Immune Memory            | Persistent threat knowledge     |
| Re-exposure              | Subsequent related attack       |

## First Exposure

During the first exposure to a previously unseen behavior:

```text
Unknown Behavior
      ↓
Anomaly Detection
      ↓
Suspicious
      ↓
Isolation
      ↓
Behavior Observation
      ↓
Feature Extraction
      ↓
Validation
      ↓
Immune Memory
```

The system does not immediately treat every anomaly as a confirmed threat. Suspicious behavior should be investigated and validated before becoming persistent immune knowledge.

## Subsequent Exposure

When similar behavior appears again:

```text
New Event
      ↓
Anomaly Detection
      ↓
Threat Representation
      ↓
Immune Memory Search
      ↓
Similarity Score
      ↓
Confidence Evaluation
      ↓
Automated Containment
```

## Core Concept

The primary MVP hypothesis is:

> A validated representation of a previously observed threat can be reused to accelerate recognition and response to the same or behaviorally related threat.

## Important Constraint

Immune Memory should contain validated threat knowledge rather than blindly storing every anomaly.

This reduces the risk of contaminating the memory with false positives or benign behavior.

## MVP Demonstration

The demonstration should contain two exposures:

### Exposure 1

An unknown suspicious behavior is detected, investigated, and stored.

### Exposure 2

A related behavior is introduced.

The system retrieves the previously stored representation and performs faster recognition and containment.

## Metrics

The demonstration should compare:

* First-exposure detection time
* Investigation time
* Memory retrieval time
* Second-exposure recognition time
* Containment time
* Similarity score
