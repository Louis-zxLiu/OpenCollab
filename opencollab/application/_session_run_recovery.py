"""Completion recovery capability for the session run loop."""

from __future__ import annotations

from typing import Any

from opencollab.application._session_run_shared import (
    _COMPLETION_RECOVERY_NUDGE,
    _EMPTY_STOP_PLACEHOLDER,
    PendingStep,
)
from opencollab.application.completion import _resolve_completion_disposition
from opencollab.domain.completion import CompletionDisposition
from opencollab.domain.session import SessionPhase


async def handle_completion_disposition(
    owner: Any,
    pending: PendingStep,
) -> bool:
    """Handle provider truncation and return whether the response was consumed."""
    response = pending.response
    disposition = _resolve_completion_disposition(response)
    if disposition not in {
        CompletionDisposition.OUTPUT_TRUNCATED,
        CompletionDisposition.CONTEXT_OVERFLOW,
    }:
        return False

    if (
        disposition is CompletionDisposition.OUTPUT_TRUNCATED
        and owner._recovery_attempts < owner._max_recovery_attempts
    ):
        owner._recovery_attempts += 1
        if owner.tracer:
            owner.tracer.log_step(
                step_type="completion_recovery",
                payload={
                    "disposition": disposition.value,
                    "attempt": owner._recovery_attempts,
                },
                latency=pending.latency,
            )
        # Keep role alternation valid when the incomplete assistant response
        # was omitted from the durable transcript.
        if owner.state.messages and owner.state.messages[-1]["role"] in ("user", "tool"):
            owner.state.append_message({"role": "assistant", "content": _EMPTY_STOP_PLACEHOLDER})
        owner.state.append_message({"role": "user", "content": _COMPLETION_RECOVERY_NUDGE})
        owner.state.transition_to(SessionPhase.AUTOSAVING)
        return True

    reason = "output truncated: provider reached its generation limit"
    owner.state.append_message(
        {
            "role": "system",
            "content": "[Output truncated by the provider. Partial response preserved; session stopped.]",
        }
    )
    await owner.event_publisher.emit(owner.event_factory.error(reason))
    await owner.finish_step(pending.latency)
    owner.clear_pending_step()
    owner.state.transition_to(SessionPhase.STOPPED, reason=reason)
    return True


__all__ = ["handle_completion_disposition"]
