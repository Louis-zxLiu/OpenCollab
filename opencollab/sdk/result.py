"""The single public result and infrastructure error for Python runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

T = TypeVar("T")


class RunError(RuntimeError):
    """Infrastructure failed, or a caller explicitly rejected a non-OK result."""

    def __init__(self, message: str, *, result: RunResult[Any] | None = None) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True, slots=True, kw_only=True)
class RunResult(Generic[T]):
    """Common outcome returned by agent, team, and workflow runs."""

    output: T | None
    status: Literal["completed", "stopped", "failed"]
    reason: str | None = None
    tokens: int | None = None
    artifacts: Path | None = None
    error: BaseException | None = field(default=None, repr=False, compare=False)
    metrics: dict[str, Any] = field(default_factory=dict)
    agent_failures: tuple[dict[str, Any], ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the run completed normally."""
        return self.status == "completed"

    def raise_for_status(self) -> RunResult[T]:
        """Return self when completed; otherwise raise one evidence-bearing error."""
        if not self.ok:
            detail = self.reason or self.status
            raise RunError(f"run {self.status}: {detail}", result=self)
        return self


@dataclass(frozen=True, slots=True, kw_only=True)
class RollbackPlan:
    """Secret-free rollback preview returned by the public SDK."""

    target_effect_ids: frozenset[str]
    invalidated_effect_ids: frozenset[str]
    affected_agent_ids: frozenset[int]
    checkpoint_by_agent: dict[int, str | None]
    digest: str
    checkpoint_filesystem_digests: dict[int, str | None] = field(default_factory=dict)
    checkpoint_environment_digests: dict[int, str | None] = field(default_factory=dict)
    checkpoint_identity_digests: dict[int, str | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class RestoreResult:
    """Secret-free Scope restore outcome."""

    agent_id: int
    checkpoint_id: str | None
    status: Literal["restored", "failed", "skipped"]
    filesystem_digest: str | None = None
    environment_digest: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CheckpointResult:
    """Secret-free checkpoint metadata returned by the public SDK."""

    checkpoint_id: str
    filesystem_digest: str
    environment_digest: str
    workspace_identity_digest: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RollbackResult:
    """Secret-free result of an explicit rollback operation."""

    plan: RollbackPlan
    restores: tuple[RestoreResult, ...]
    invalidated: bool

    @property
    def affected_agent_ids(self) -> frozenset[int]:
        return self.plan.affected_agent_ids

    @property
    def invalidated_effect_ids(self) -> frozenset[str]:
        return self.plan.invalidated_effect_ids


__all__ = [
    "CheckpointResult",
    "RestoreResult",
    "RollbackPlan",
    "RollbackResult",
    "RunError",
    "RunResult",
]
