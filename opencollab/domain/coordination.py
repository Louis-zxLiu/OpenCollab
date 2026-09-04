"""Pure coordination payload policy shared by orchestration adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CoordinationPolicy:
    """Bounds for short delegation metadata, independent of any tool name."""

    assignment_bytes: int = 1024
    context_bytes: int = 2048
    total_bytes: int = 3072

    def __post_init__(self) -> None:
        values = (self.assignment_bytes, self.context_bytes, self.total_bytes)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("coordination limits must be positive integers")

    def validate_text(self, value: Any, field: str) -> str | None:
        """Return a user-facing validation error, or None for valid text."""
        limit = {
            "assignment": self.assignment_bytes,
            "context": self.context_bytes,
        }.get(field)
        if limit is None:
            raise ValueError(f"unknown coordination field: {field}")
        if not isinstance(value, str):
            return f"{field} must be a string"
        if "\x00" in value:
            return f"{field} must not contain NUL bytes"
        if len(value.encode("utf-8")) > limit:
            return f"{field} exceeds the {limit}-byte limit"
        return None

    def validate(self, assignment: Any, context: Any = "") -> str | None:
        """Validate one coordination payload using the same policy everywhere."""
        error = self.validate_text(assignment, "assignment")
        if error:
            return error
        error = self.validate_text(context, "context")
        if error:
            return error
        if len(assignment.encode("utf-8")) + len(context.encode("utf-8")) > self.total_bytes:
            return f"assignment and context exceed the {self.total_bytes}-byte limit"
        return None


DEFAULT_COORDINATION_POLICY = CoordinationPolicy()


__all__ = ["CoordinationPolicy", "DEFAULT_COORDINATION_POLICY"]
