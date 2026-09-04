"""Provider-independent completion dispositions."""

from __future__ import annotations

from enum import Enum


class CompletionDisposition(Enum):
    """Semantic outcome of one provider generation."""

    COMPLETED = "completed"
    TOOL_CALLS = "tool_calls"
    OUTPUT_TRUNCATED = "output_truncated"
    CONTEXT_OVERFLOW = "context_overflow"


__all__ = ["CompletionDisposition"]
