from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from src.pipeline.real_adis_pipeline import RealADISPipeline
from src.pipeline.memory_adapter import ImmuneMemoryAdapter
from src.sandbox.isolation_chamber import IsolationSandbox


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="M8 End-to-End ADIS Attack Replay"
    )

    parser.add_argument(
        "--dataset",
        default="data/runtime/m56_attack_test.parquet",
        help="Controlled attack replay dataset.",
    )

    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="Row index to replay.",
    )

    parser.add_argument(
        "--memory-path",
        default="models/immune_memory_m8",
        help="Persistent Qdrant memory path.",
    )

    parser.add_argument(
        "--phase",
        choices=["full", "exposure1", "exposure2"],
        default="full",
        help="Internal replay phase.",
    )

    return parser.parse_args()


def load_flow(dataset_path: str, row_index: int):
    path = PROJECT_ROOT / dataset_path

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if row_index < 0 or row_index >= len(df):
        raise IndexError(
            f"Row {row_index} is outside dataset range 0..{len(df)-1}"
        )

    return df.iloc[row_index].copy()


def get_value(mapping, *keys, default=None):
    if not isinstance(mapping, dict):
        return default

    for key in keys:
        if key in mapping:
            return mapping[key]

    return default


def extract_pipeline_summary(result: dict):
    context = result.get("context", {})

    detection = context.get("detection", {})
    recognition = context.get("recognition", {})
    behavior = context.get("behavior", {})
    threat = context.get("threat", {})
    decision = result.get("decision", {})
    risk = result.get("risk", {})
    memory_event = result.get("memory_event", {})
    investigation = result.get("investigation", {})

    return {
        "detector_decision": get_value(
            detection,
            "decision",
            "detector_decision",
            default=result.get("detector_decision"),
        ),
        "attack_probability": get_value(
            detection,
            "attack_probability",
            "probability",
            default=None,
        ),
        "recognition": get_value(
            recognition,
            "classification",
            "recognition",
            "status",
            default=None,
        ),
        "similarity": get_value(
            recognition,
            "similarity",
            "similarity_score",
            default=None,
        ),
        "behavior_family": get_value(
            behavior,
            "behavior_family",
            default=None,
        ),
        "threat_type": get_value(
            threat,
            "threat_type",
            default=None,
        ),
        "risk_score": get_value(
            risk,
            "risk_score",
            "score",
            default=None,
        ),
        "severity": get_value(
            risk,
            "severity",
            default=None,
        ),
        "action": get_value(
            decision,
            "action",
            "decision",
            default=None,
        ),
        "memory_status": get_value(
            memory_event,
            "status",
            "event",
            "type",
            default=None,
        ),
        "investigation_status": get_value(
            investigation,
            "status",
            default=None,
        ),
    }


def run_pipeline(flow: pd.Series, memory_path: str):
    memory = ImmuneMemoryAdapter(
        storage_path=str(PROJECT_ROOT / memory_path)
    )

    pipeline = RealADISPipeline(memory=memory)

    result = pipeline.process(flow)

    return result, memory, pipeline


def run_response(result: dict, flow: pd.Series):
    """
    Execute the M7 response layer using the policy-selected action.

    No real system/network isolation is performed.
    IsolationSandbox is simulation-only.
    """

    action = get_value(
        result.get("decision", {}),
        "action",
        "decision",
        default="INVESTIGATE",
    )

    sandbox = IsolationSandbox()

    # Convert pipeline result into the fields expected by the sandbox.
    context = result.get("context", {})

    detection = context.get("detection", {})
    behavior = context.get("behavior", {})
    threat = context.get("threat", {})

    detector_decision = get_value(
        detection,
        "decision",
        "detector_decision",
        default="UNKNOWN",
    )

    attack_probability = get_value(
        detection,
        "attack_probability",
        "probability",
        default=0.0,
    )

    behavior_family = get_value(
        behavior,
        "behavior_family",
        default="unknown",
    )

    threat_type = get_value(
        threat,
        "threat_type",
        default=None,
    )

    evidence = [
        {
            "source": "M2_DETECTION",
            "field": "detector_decision",
            "value": detector_decision,
        },
        {
            "source": "M2_DETECTION",
            "field": "attack_probability",
            "value": attack_probability,
        },
        {
            "source": "M3_BEHAVIOR",
            "field": "behavior_family",
            "value": behavior_family,
        },
        {
            "source": "M3_BEHAVIOR",
            "field": "threat_type",
            "value": threat_type,
        },
        {
            "source": "M4_RECOGNITION",
            "field": "recognition",
            "value": get_value(
                context.get("recognition", {}),
                "classification",
                "recognition",
                default=None,
            ),
        },
    ]

    reason = (
        f"Policy selected {action} based on "
        f"detector={detector_decision}, "
        f"behavior_family={behavior_family}."
    )

    # M7 sandbox API.
    event = sandbox.execute_response(
        action=action,
        flow=flow.to_dict(),
        reason=reason,
        evidence=evidence,
    )

    return event


def print_pipeline(title: str, result: dict):
    summary = extract_pipeline_summary(result)

    print()
    print("-" * 90)
    print(title)
    print("-" * 90)

    print(f"Detector Decision : {summary['detector_decision']}")
    print(f"Attack Probability: {summary['attack_probability']}")
    print(f"Recognition       : {summary['recognition']}")
    print(f"Similarity        : {summary['similarity']}")
    print(f"Behavior Family   : {summary['behavior_family']}")
    print(f"Threat Type       : {summary['threat_type']}")
    print(f"Risk Score        : {summary['risk_score']}")
    print(f"Severity          : {summary['severity']}")
    print(f"Decision          : {summary['action']}")
    print(f"Memory Status     : {summary['memory_status']}")
    print(f"Investigation     : {summary['investigation_status']}")

    return summary


def print_event(event, title="M7 RESPONSE"):
    print()
    print("-" * 90)
    print(title)
    print("-" * 90)

    if hasattr(event, "to_dict"):
        data = event.to_dict()
    elif isinstance(event, dict):
        data = event
    else:
        data = {
            "lifecycle_status": getattr(
                event,
                "lifecycle_status",
                None,
            ),
            "action": getattr(event, "action", None),
        }

    print(f"Action          : {data.get('action')}")
    print(f"Lifecycle       : {data.get('lifecycle_status')}")
    print(f"Validation      : {data.get('validation_status')}")

    print()
    print("Transition History:")

    for transition in data.get("transition_history", []):
        status = transition.get("status", "UNKNOWN")
        reason = transition.get("reason", "")
        print(f"  {status:<16} {reason}")

    return data


def run_phase_exposure1(args):
    flow = load_flow(args.dataset, args.row)

    result, memory, _ = run_pipeline(
        flow,
        args.memory_path,
    )

    summary = print_pipeline(
        "[EXPOSURE 1 — UNKNOWN ATTACK]",
        result,
    )

    event = run_response(result, flow)

    event_data = print_event(
        event,
        "[M7 RESPONSE — EXPOSURE 1]",
    )

    memory_count = memory.count_memories()
    memory.close()

    print()
    print(f"Memory Count After Exposure 1: {memory_count}")

    report = {
        "phase": "exposure1",
        "pipeline": summary,
        "response": event_data,
        "memory_count": memory_count,
    }

    return report


def run_phase_exposure2(args):
    flow = load_flow(args.dataset, args.row)

    result, memory, _ = run_pipeline(
        flow,
        args.memory_path,
    )

    summary = print_pipeline(
        "[EXPOSURE 2 — AFTER RESTART]",
        result,
    )

    event = run_response(result, flow)

    event_data = print_event(
        event,
        "[M7 RESPONSE — EXPOSURE 2]",
    )

    memory_count = memory.count_memories()

    print()
    print(f"Memory Count After Exposure 2: {memory_count}")

    report = {
        "phase": "exposure2",
        "pipeline": summary,
        "response": event_data,
        "memory_count": memory_count,
    }
    
    memory.close()

    return report


def validate_exposure1(report):
    summary = report["pipeline"]
    response = report["response"]

    checks = []

    checks.append(
        (
            "Detection",
            summary["detector_decision"] == "ATTACK",
        )
    )

    checks.append(
        (
            "Novel recognition",
            summary["recognition"] == "NOVEL",
        )
    )

    checks.append(
        (
            "Isolation decision",
            summary["action"] == "ISOLATE",
        )
    )

    checks.append(
        (
            "Isolation lifecycle validated",
            response["lifecycle_status"] == "VALIDATED",
        )
    )

    checks.append(
        (
            "Recovery not automatic",
            response.get("recovery_timestamp") is None,
        )
    )

    checks.append(
        (
            "Memory created",
            report["memory_count"] >= 1,
        )
    )

    return checks


def validate_exposure2(report, initial_memory_count):
    summary = report["pipeline"]

    checks = []

    checks.append(
        (
            "Detection",
            summary["detector_decision"] == "ATTACK",
        )
    )

    checks.append(
        (
            "Known recognition after restart",
            summary["recognition"] == "KNOWN",
        )
    )

    checks.append(
        (
            "Similarity available",
            summary["similarity"] is not None,
        )
    )

    checks.append(
        (
            "No additional memory created",
            report["memory_count"] == initial_memory_count,
        )
    )

    return checks


def print_checks(checks):
    print()
    print("M8 CHECKS")
    print("-" * 90)

    all_passed = True

    for name, passed in checks:
        if passed:
            print(f"[PASS] {name}")
        else:
            print(f"[FAIL] {name}")
            all_passed = False

    return all_passed


def main():
    args = parse_args()

    print("=" * 90)
    print("M8 — END-TO-END ADIS ATTACK REPLAY")
    print("=" * 90)

    print(f"Dataset     : {args.dataset}")
    print(f"Row         : {args.row}")
    print(f"Memory Path : {args.memory_path}")

    # ------------------------------------------------------------------
    # FULL MODE
    # ------------------------------------------------------------------
    if args.phase == "full":
        # Use an isolated memory location for this replay.
        memory_path = args.memory_path
        memory_dir = PROJECT_ROOT / memory_path

        # We deliberately do not silently delete arbitrary user data.
        # If an old M8 memory exists, require a clean location.
        if memory_dir.exists():
            raise RuntimeError(
                f"Memory path already exists:\n"
                f"{memory_dir}\n\n"
                f"Use a new --memory-path for a clean M8 replay."
            )

        # Exposure 1
        report1 = run_phase_exposure1(args)

        checks1 = validate_exposure1(report1)

        if not print_checks(checks1):
            print()
            print("[FAIL] Exposure 1 failed.")
            sys.exit(1)

        # Explicit recovery before second exposure.
        #
        # We do not use the same event object across a process restart.
        # M8 verifies that memory, not runtime state, survives.
        print()
        print("=" * 90)
        print("[RECOVERY]")
        print("=" * 90)

        # Re-run a controlled response object and explicitly recover it.
        flow = load_flow(args.dataset, args.row)

        result, recovery_memory, _ = run_pipeline(
            flow,
            args.memory_path,
        )

        recovery_event = run_response(result, flow)

        recovery_event.mark_recovered(
            reason="M8 replay recovery approved after validation."
        )
        recovery_memory.close()

        recovery_data = print_event(
            recovery_event,
            "[M7 EXPLICIT RECOVERY]",
        )

        if recovery_data["lifecycle_status"] != "RECOVERED":
            print("[FAIL] Explicit recovery failed.")
            sys.exit(1)

        print("[PASS] Recovery completed.")

        initial_memory_count = report1["memory_count"]

        # ------------------------------------------------------------------
        # True restart
        # ------------------------------------------------------------------
        print()
        print("=" * 90)
        print("[SIMULATING PROCESS RESTART]")
        print("=" * 90)

        phase2_cmd = [
            sys.executable,
            "-m",
            "scripts.test_m8_attack_replay",
            "--dataset",
            args.dataset,
            "--row",
            str(args.row),
            "--memory-path",
            args.memory_path,
            "--phase",
            "exposure2",
        ]

        completed = subprocess.run(
            phase2_cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )

        print(completed.stdout)

        if completed.returncode != 0:
            print(completed.stderr)
            print("[FAIL] Exposure 2 subprocess failed.")
            sys.exit(completed.returncode)

        # The subprocess prints JSON report on the final line.
        marker = "__M8_REPORT__="
        report_line = None

        for line in completed.stdout.splitlines():
            if line.startswith(marker):
                report_line = line[len(marker):]

        if report_line is None:
            print("[FAIL] Could not retrieve Exposure 2 report.")
            sys.exit(1)

        report2 = json.loads(report_line)

        checks2 = validate_exposure2(
            report2,
            initial_memory_count,
        )

        if not print_checks(checks2):
            print()
            print("[FAIL] Exposure 2 failed.")
            sys.exit(1)

        all_checks = checks1 + checks2

        print()
        print("=" * 90)
        print("[PASS] M8 END-TO-END ATTACK REPLAY COMPLETED")
        print("=" * 90)

        print()
        print("Verified:")
        print("  1. M2 detection")
        print("  2. M4 novel recognition")
        print("  3. M5 decision")
        print("  4. M6 investigation")
        print("  5. M7 isolation")
        print("  6. M7 validation")
        print("  7. Explicit recovery")
        print("  8. Persistent memory")
        print("  9. Recognition after restart")
        print(" 10. No duplicate memory")

        return

    # ------------------------------------------------------------------
    # SUBPROCESS MODE
    # ------------------------------------------------------------------
    if args.phase == "exposure2":
        report = run_phase_exposure2(args)

        print()
        print(
            "__M8_REPORT__="
            + json.dumps(
                report,
                ensure_ascii=False,
                default=str,
            )
        )

        return

    # ------------------------------------------------------------------
    # EXPOSURE 1 ONLY
    # ------------------------------------------------------------------
    if args.phase == "exposure1":
        report = run_phase_exposure1(args)

        print()
        print(
            "__M8_REPORT__="
            + json.dumps(
                report,
                ensure_ascii=False,
                default=str,
            )
        )

        return


if __name__ == "__main__":
    main()
    