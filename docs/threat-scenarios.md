# ADIS Threat Scenarios

The hackathon MVP uses controlled synthetic scenarios.

No real malware execution is required.

## Scenario 1 — SSH Brute Force

### Normal

Typical authentication activity.

### Suspicious

A large number of failed authentication attempts within a short period.

Example features:

```text
failed_logins
authentication_rate
unique_source_count
connection_frequency
destination_port
```

### First Exposure

The behavior is unknown to Immune Memory.

Expected:

```text
Detect → Isolate → Analyze → Store
```

### Second Exposure

A behaviorally similar brute-force pattern appears.

Expected:

```text
Detect → Memory Match → Fast Containment
```

---

## Scenario 2 — Suspicious Process Execution

### Normal

Typical process creation patterns.

### Suspicious

An unusual process execution chain or abnormal process spawning frequency.

Example features:

```text
process_spawn_count
parent_child_relationship
execution_frequency
cpu_usage
network_activity
```

### First Exposure

The behavior is investigated and stored.

### Second Exposure

A related process execution pattern appears.

Expected:

```text
Detect → Memory Match → Containment
```

## Scenario Selection

The MVP should prioritize one primary scenario for the live demonstration.

The recommended primary scenario is:

**SSH Brute Force**

A second scenario can be added if implementation time allows.

## Safety

All scenarios are simulated or executed in controlled environments.

The MVP should not require real-world exploitation or malicious payload execution.
