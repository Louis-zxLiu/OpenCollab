"""Application-level completion semantics."""

from __future__ import annotations

from opencollab.application.ports import CompletionResponse
from opencollab.domain.completion import CompletionDisposition


def _resolve_completion_disposition(
    response: CompletionResponse,
) -> CompletionDisposition:
    """Use adapter-normalized semantics, with a narrow legacy fallback."""
    disposition = getattr(response, "disposition", None)
    if isinstance(disposition, CompletionDisposition):
        return disposition
    if isinstance(disposition, str):
        try:
            return CompletionDisposition(disposition)
        except ValueError:
            pass
    reason = getattr(response, "finish_reason", None)
    if reason in {"length", "max_tokens", "max_output_tokens"}:
        return CompletionDisposition.OUTPUT_TRUNCATED
    if reason in {"model_context_window_exceeded", "context_length_exceeded"}:
        return CompletionDisposition.CONTEXT_OVERFLOW
    if getattr(response, "tool_calls", None):
        return CompletionDisposition.TOOL_CALLS
    return CompletionDisposition.COMPLETED


def _is_discardable_completion(response: CompletionResponse) -> bool:
    """Return whether a response cannot be committed as a normal turn."""
    return _resolve_completion_disposition(response) in {
        CompletionDisposition.OUTPUT_TRUNCATED,
        CompletionDisposition.CONTEXT_OVERFLOW,
    }


__all__ = ["_resolve_completion_disposition"]
