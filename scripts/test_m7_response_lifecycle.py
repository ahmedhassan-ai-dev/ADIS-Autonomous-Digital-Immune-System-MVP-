from __future__ import annotations

from src.sandbox.isolation_chamber import (
    IsolationSandbox,
)


def print_history(event):

    print("\nTransition History:")

    for transition in event.transition_history:

        print(
            f"  "
            f"{transition['status']:<15} "
            f"{transition['reason']}"
        )


def main():

    print("=" * 90)
    print("M7 — ADAPTIVE RESPONSE LIFECYCLE TEST")
    print("=" * 90)

    sandbox = IsolationSandbox(
        investigation_delay=0.01
    )

    dummy_flow = {
        "Flow Duration": 1000,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 2,
        "Flow Packets/s": 100.0,
        "Flow Bytes/s": 5000.0,
    }

    # ================================================================
    # 1. ALLOW
    # ================================================================

    print("\n[1/3] Testing ALLOW...")

    allow_event = sandbox.execute_response(
        action="ALLOW",
        flow=dummy_flow,
        reason="Low-risk benign activity.",
    )

    print(
        f"Action  : {allow_event.action}"
    )

    print(
        f"Status  : {allow_event.lifecycle_status}"
    )

    assert (
        allow_event.lifecycle_status
        == "ALLOWED"
    )

    print_history(
        allow_event
    )

    print(
        "[PASS] ALLOW lifecycle."
    )

    # ================================================================
    # 2. INVESTIGATE
    # ================================================================

    print("\n[2/3] Testing INVESTIGATE...")

    investigate_event = sandbox.execute_response(
        action="INVESTIGATE",
        flow=dummy_flow,
        reason="Uncertain behavior requires investigation.",
    )

    print(
        f"Action  : {investigate_event.action}"
    )

    print(
        f"Status  : {investigate_event.lifecycle_status}"
    )

    assert (
        investigate_event.lifecycle_status
        == "VALIDATED"
    )

    assert (
        investigate_event.validation_status
        == "VALIDATED"
    )

    print_history(
        investigate_event
    )

    print(
        "[PASS] INVESTIGATE lifecycle."
    )

    # ================================================================
    # 3. ISOLATE
    # ================================================================

    print("\n[3/3] Testing ISOLATE...")

    isolate_event = sandbox.execute_response(
        action="ISOLATE",
        flow=dummy_flow,
        reason="Novel high-risk behavior.",
    )

    print(
        f"Action  : {isolate_event.action}"
    )

    print(
        f"Status  : {isolate_event.lifecycle_status}"
    )

    assert (
        isolate_event.lifecycle_status
        == "VALIDATED"
    )

    assert (
        isolate_event.validation_status
        == "VALIDATED"
    )

    assert (
        isolate_event.recovery_timestamp
        is None
    )

    print_history(
        isolate_event
    )

    print(
        "\n[INFO] Recovery has NOT happened automatically."
    )

    print(
        "[PASS] ISOLATE lifecycle."
    )

    # ================================================================
    # 4. Explicit recovery
    # ================================================================

    print(
        "\n[4/4] Testing explicit RECOVERY..."
    )

    recovered_event = sandbox.recover(
        isolate_event,
        reason="Validation successful; recovery policy approved.",
    )

    print(
        f"Action  : {recovered_event.action}"
    )

    print(
        f"Status  : {recovered_event.lifecycle_status}"
    )

    print(
        f"Recovery timestamp: "
        f"{recovered_event.recovery_timestamp}"
    )

    assert (
        recovered_event.lifecycle_status
        == "RECOVERED"
    )

    assert (
        recovered_event.recovery_timestamp
        is not None
    )

    print_history(
        recovered_event
    )

    print(
        "[PASS] RECOVERY lifecycle."
    )

    # ================================================================
    # FINAL
    # ================================================================

    print("\n" + "=" * 90)
    print("[PASS] M7 RESPONSE LIFECYCLE COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    main()