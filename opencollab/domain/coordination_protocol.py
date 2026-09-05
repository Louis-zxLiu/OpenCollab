"""Pure coordination protocol values and state transitions.

The scheduler owns I/O and task lifecycle.  This module only describes the
state needed to make coordination decisions deterministic and serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping


class WaitKind(str, Enum):
    ANY = "any"
    CHILD = "child"
    MESSAGE = "message"
    EFFECT = "effect"


class AgentLifecycleStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETIRED = "retired"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class CoordinationWait:
    """A non-terminal suspension condition for one Agent."""

    wait_for: WaitKind = WaitKind.ANY
    effect_ids: frozenset[str] = field(default_factory=frozenset)
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.wait_for, WaitKind):
            object.__setattr__(self, "wait_for", WaitKind(self.wait_for))
        if any(not isinstance(value, str) or not value for value in self.effect_ids):
            raise ValueError("wait effect IDs must be non-empty strings")
        if "\x00" in self.reason:
            raise ValueError("wait reason must not contain NUL bytes")
        if len(self.reason.encode("utf-8")) > 1024:
            raise ValueError("wait reason exceeds 1024 bytes")


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    """Positive executable evidence tied to an adopted effect revision."""

    evidence_id: str
    effect_id: str
    tool_call_id: str
    tool_name: str
    command: str
    exit_code: int
    verified: bool
    workspace_revision: str | None = None
    result_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("evidence_id", "effect_id", "tool_call_id", "tool_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.command, str) or not self.command.strip():
            raise ValueError("verification command must be non-empty")
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise ValueError("verification exit_code must be an integer")


def register_visible_effect(required: Iterable[str], effect_id: str) -> frozenset[str]:
    """Register an effect that is visible but not yet adopted."""
    if not isinstance(effect_id, str) or not effect_id:
        raise ValueError("effect_id must be a non-empty string")
    return frozenset(set(required) | {effect_id})


def adopt_visible_effect(required: Iterable[str], effect_id: str) -> frozenset[str]:
    """Remove one explicitly adopted effect from the pending set."""
    if not isinstance(effect_id, str) or not effect_id:
        raise ValueError("effect_id must be a non-empty string")
    return frozenset(value for value in required if value != effect_id)


def can_execute_normal_work(required: Iterable[str]) -> bool:
    return not any(required)


def wait_matches(wait: CoordinationWait | None, event_kind: WaitKind, effect_id: str | None = None) -> bool:
    """Return whether an incoming scheduler event releases a wait."""
    if wait is None:
        return False
    if wait.wait_for not in (WaitKind.ANY, event_kind):
        return False
    if wait.wait_for is WaitKind.EFFECT and wait.effect_ids:
        return effect_id in wait.effect_ids
    return True


def mark_superseded(status: AgentLifecycleStatus) -> AgentLifecycleStatus:
    if status is AgentLifecycleStatus.TERMINAL:
        return status
    return AgentLifecycleStatus.SUPERSEDED


def is_addressable(status: AgentLifecycleStatus) -> bool:
    return status is AgentLifecycleStatus.ACTIVE


def verification_passes(
    evidence: Mapping[str, VerificationEvidence],
    effect_id: str,
    evidence_ids: Iterable[str],
) -> bool:
    """Require positive executable probes for the exact adopted effect."""
    selected = [evidence.get(value) for value in evidence_ids]
    return bool(selected) and all(
        item is not None
        and item.effect_id == effect_id
        and item.verified
        and item.exit_code == 0
        and bool(item.command.strip())
        for item in selected
    )


__all__ = [
    "AgentLifecycleStatus",
    "CoordinationWait",
    "VerificationEvidence",
    "WaitKind",
    "adopt_visible_effect",
    "can_execute_normal_work",
    "is_addressable",
    "mark_superseded",
    "register_visible_effect",
    "verification_passes",
    "wait_matches",
]
