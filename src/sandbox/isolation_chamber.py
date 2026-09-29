from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class IsolationEvent:
    """
    Immutable-style audit event representing one response lifecycle.

    This MVP performs SIMULATED isolation only.
    No real host/network containment is executed.
    """

    event_id: str
    timestamp: float

    action: str
    lifecycle_status: str

    reason: str
    evidence: list[dict[str, Any]] = field(default_factory=list)

    recovery_timestamp: Optional[float] = None

    antigen_id: Optional[str] = None

    validation_status: Optional[str] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    transition_history: list[dict[str, Any]] = field(
        default_factory=list
    )

    def add_transition(
        self,
        status: str,
        reason: str,
    ) -> None:
        self.lifecycle_status = status

        self.transition_history.append(
            {
                "timestamp": time.time(),
                "status": status,
                "reason": reason,
            }
        )

    def mark_recovered(
        self,
        reason: str = "Recovery policy completed.",
    ) -> None:
        """
        Explicit recovery transition.

        Recovery is intentionally NOT automatic after validation.
        """

        if self.lifecycle_status != "VALIDATED":
            raise RuntimeError(
                "Recovery is only allowed after VALIDATED state."
            )

        self.recovery_timestamp = time.time()

        self.add_transition(
            "RECOVERED",
            reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "lifecycle_status": self.lifecycle_status,
            "reason": self.reason,
            "evidence": self.evidence,
            "recovery_timestamp": self.recovery_timestamp,
            "antigen_id": self.antigen_id,
            "validation_status": self.validation_status,
            "metadata": self.metadata,
            "transition_history": self.transition_history,
        }


class IsolationSandbox:
    """
    Controlled simulated isolation/investigation environment.

    IMPORTANT:
        This class does not execute malware or perform real network
        containment. It only simulates the response lifecycle.
    """

    VERSION = "adis-isolation-sandbox-v2"

    def __init__(
        self,
        analyzer=None,
        investigation_delay: float = 0.01,
    ):
        self.analyzer = analyzer
        self.investigation_delay = investigation_delay

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _new_event(
        action: str,
        reason: str,
        evidence: Optional[list[dict[str, Any]]] = None,
        antigen_id: Optional[str] = None,
    ) -> IsolationEvent:

        return IsolationEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            action=action,
            lifecycle_status="DETECTED",
            reason=reason,
            evidence=evidence or [],
            antigen_id=antigen_id,
            metadata={
                "sandbox_version": IsolationSandbox.VERSION,
                "isolation_mode": "SIMULATED",
            },
        )

    # ------------------------------------------------------------------
    # ALLOW
    # ------------------------------------------------------------------

    def allow(
        self,
        reason: str,
        evidence: Optional[list[dict[str, Any]]] = None,
        antigen_id: Optional[str] = None,
    ) -> IsolationEvent:

        event = self._new_event(
            action="ALLOW",
            reason=reason,
            evidence=evidence,
            antigen_id=antigen_id,
        )

        event.add_transition(
            "ALLOWED",
            "Policy selected ALLOW.",
        )

        return event

    # ------------------------------------------------------------------
    # INVESTIGATE
    # ------------------------------------------------------------------

    def investigate(
        self,
        flow: dict[str, Any],
        reason: str,
        evidence: Optional[list[dict[str, Any]]] = None,
        antigen_id: Optional[str] = None,
    ) -> IsolationEvent:

        event = self._new_event(
            action="INVESTIGATE",
            reason=reason,
            evidence=evidence,
            antigen_id=antigen_id,
        )

        event.add_transition(
            "INVESTIGATING",
            "Investigation started.",
        )

        time.sleep(
            self.investigation_delay
        )

        analysis_result = None

        if self.analyzer is not None:

            try:
                analysis_result = (
                    self.analyzer.analyze_and_extract(
                        flow
                    )
                )

            except Exception as exc:

                event.metadata[
                    "analysis_error"
                ] = str(exc)

        if analysis_result is not None:

            event.metadata[
                "analysis_completed"
            ] = True

            if hasattr(
                analysis_result,
                "metadata",
            ):
                metadata = (
                    analysis_result.metadata
                )

                if isinstance(
                    metadata,
                    dict,
                ):
                    event.metadata.update(
                        metadata
                    )

        event.validation_status = "VALIDATED"

        event.add_transition(
            "VALIDATED",
            "Investigation completed and evidence validated.",
        )

        return event

    # ------------------------------------------------------------------
    # ISOLATE
    # ------------------------------------------------------------------

    def isolate(
        self,
        flow: dict[str, Any],
        reason: str,
        evidence: Optional[list[dict[str, Any]]] = None,
        antigen_id: Optional[str] = None,
    ) -> IsolationEvent:

        event = self._new_event(
            action="ISOLATE",
            reason=reason,
            evidence=evidence,
            antigen_id=antigen_id,
        )

        event.add_transition(
            "ISOLATED",
            "Policy selected simulated isolation.",
        )

        event.metadata[
            "isolation_status"
        ] = "SIMULATED_ISOLATION"

        event.add_transition(
            "INVESTIGATING",
            "Investigation started inside isolated environment.",
        )

        time.sleep(
            self.investigation_delay
        )

        analysis_result = None

        if self.analyzer is not None:

            try:
                analysis_result = (
                    self.analyzer.analyze_and_extract(
                        flow
                    )
                )

            except Exception as exc:

                event.metadata[
                    "analysis_error"
                ] = str(exc)

        if analysis_result is not None:

            event.metadata[
                "analysis_completed"
            ] = True

            if hasattr(
                analysis_result,
                "metadata",
            ):

                metadata = (
                    analysis_result.metadata
                )

                if isinstance(
                    metadata,
                    dict,
                ):

                    event.metadata.update(
                        metadata
                    )

        event.validation_status = "VALIDATED"

        event.add_transition(
            "VALIDATED",
            "Isolated investigation completed and evidence validated.",
        )

        # IMPORTANT:
        # Do NOT recover automatically.
        #
        # Recovery must be a separate policy-controlled transition.

        return event

    # ------------------------------------------------------------------
    # RECOVERY
    # ------------------------------------------------------------------

    def recover(
        self,
        event: IsolationEvent,
        reason: str = "Recovery policy approved.",
    ) -> IsolationEvent:

        event.mark_recovered(
            reason=reason
        )

        event.metadata[
            "isolation_status"
        ] = "RECOVERED"

        return event

    # ------------------------------------------------------------------
    # Generic response dispatcher
    # ------------------------------------------------------------------

    def execute_response(
        self,
        action: str,
        flow: dict[str, Any],
        reason: str,
        evidence: Optional[list[dict[str, Any]]] = None,
        antigen_id: Optional[str] = None,
    ) -> IsolationEvent:

        normalized_action = str(
            action
        ).upper().strip()

        if normalized_action == "ALLOW":

            return self.allow(
                reason=reason,
                evidence=evidence,
                antigen_id=antigen_id,
            )

        if normalized_action == "INVESTIGATE":

            return self.investigate(
                flow=flow,
                reason=reason,
                evidence=evidence,
                antigen_id=antigen_id,
            )

        if normalized_action == "ISOLATE":

            return self.isolate(
                flow=flow,
                reason=reason,
                evidence=evidence,
                antigen_id=antigen_id,
            )

        raise ValueError(
            f"Unsupported response action: {action}"
        )

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    def investigate_legacy(
        self,
        flow: dict[str, Any],
        antigen=None,
    ):
        """
        Backward-compatible helper for old code paths.

        Returns the antigen instead of IsolationEvent.
        """

        event = self.isolate(
            flow=flow,
            reason="Legacy investigation path.",
            antigen_id=(
                getattr(
                    antigen,
                    "antigen_id",
                    None,
                )
                if antigen is not None
                else None
            ),
        )

        if antigen is not None:

            try:

                if hasattr(
                    antigen,
                    "metadata",
                ):

                    antigen.metadata.update(
                        event.metadata
                    )

                    antigen.metadata[
                        "validation_status"
                    ] = "VALIDATED"

                    antigen.metadata[
                        "validated"
                    ] = True

            except Exception:
                pass

        return antigen