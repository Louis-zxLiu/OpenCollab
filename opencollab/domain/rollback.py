"""Pure values and graph operations for explicit effect rollback."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal, Mapping

EffectKind = Literal["child_result", "message", "tool_result"]
EffectStatus = Literal["active", "invalidated"]
CheckpointBoundary = Literal["initial", "effect", "message", "tool"]
RestoreStatus = Literal["restored", "failed", "skipped"]


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """An exact, immutable Scope environment mapping."""

    entries: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        names = [name for name, _ in self.entries]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("environment entries must be sorted and unique")
        if any(not isinstance(name, str) or not isinstance(value, str) for name, value in self.entries):
            raise TypeError("environment entries must contain strings")

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> "EnvironmentSnapshot":
        return cls(tuple(sorted(values.items())))

    def as_dict(self) -> dict[str, str]:
        return dict(self.entries)

    def digest(self) -> str:
        payload = "\0".join(f"{name}\0{value}" for name, value in self.entries)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EffectRef:
    """One node in the causal effect graph."""

    effect_id: str
    producer_aid: int
    kind: EffectKind
    epoch: int
    attempt: int
    parent_effect_ids: tuple[str, ...] = ()
    status: EffectStatus = "active"
    content_digest: str = ""

    def __post_init__(self) -> None:
        if not self.effect_id:
            raise ValueError("effect_id is required")
        if self.producer_aid < 0 or self.epoch < 0 or self.attempt < 0:
            raise ValueError("effect identifiers and counters must be non-negative")
        if self.effect_id in self.parent_effect_ids:
            raise ValueError("an effect cannot be its own parent")
        if len(set(self.parent_effect_ids)) != len(self.parent_effect_ids):
            raise ValueError("parent effect IDs must be unique")


@dataclass(frozen=True, slots=True)
class ScopeCheckpoint:
    """Filesystem, environment, and causal state for one Agent Scope."""

    checkpoint_id: str
    owner_aid: int
    sequence: int
    filesystem_revision: str
    environment: EnvironmentSnapshot
    causal_frontier: frozenset[str] = field(default_factory=frozenset)
    boundary: CheckpointBoundary = "initial"
    boundary_effect_id: str | None = None
    workspace_identity: str = ""
    filesystem_digest: str = ""

    def __post_init__(self) -> None:
        if not self.checkpoint_id or not self.filesystem_revision:
            raise ValueError("checkpoint identity and filesystem revision are required")
        if self.owner_aid < 0 or self.sequence < 0:
            raise ValueError("checkpoint owner and sequence must be non-negative")
        if not self.workspace_identity:
            raise ValueError("workspace identity is required")


@dataclass(frozen=True, slots=True)
class RollbackPlan:
    """A reviewable, side-effect-free rollback decision."""

    target_effect_ids: frozenset[str]
    invalidated_effect_ids: frozenset[str]
    affected_agent_ids: frozenset[int]
    checkpoint_by_agent: Mapping[int, ScopeCheckpoint | None]
    coordinator_agent_ids: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class RestoreResult:
    agent_id: int
    checkpoint_id: str | None
    status: RestoreStatus
    filesystem_digest: str | None = None
    environment_digest: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RollbackResult:
    plan: RollbackPlan
    restores: tuple[RestoreResult, ...]
    invalidated: bool


def compute_descendants(effects: Mapping[str, EffectRef], roots: set[str]) -> frozenset[str]:
    """Return roots and all known transitive descendants."""
    children: dict[str, set[str]] = {}
    for effect in effects.values():
        for parent in effect.parent_effect_ids:
            children.setdefault(parent, set()).add(effect.effect_id)
    result: set[str] = set()
    stack = list(roots)
    while stack:
        effect_id = stack.pop()
        if effect_id in result or effect_id not in effects:
            continue
        result.add(effect_id)
        stack.extend(children.get(effect_id, ()))
    return frozenset(result)


def compute_affected_agents(
    effects: Mapping[str, EffectRef],
    consumers: Mapping[str, set[int]],
    invalidated_effect_ids: set[str],
) -> frozenset[int]:
    """Return producers and consumers that depend on invalidated effects."""
    affected: set[int] = set()
    for effect_id in invalidated_effect_ids:
        effect = effects.get(effect_id)
        if effect is not None:
            affected.add(effect.producer_aid)
        affected.update(consumers.get(effect_id, ()))
    return frozenset(affected)


def select_checkpoint(
    checkpoints: Mapping[int, tuple[ScopeCheckpoint, ...]],
    aid: int,
    invalidated_effect_ids: set[str],
) -> ScopeCheckpoint | None:
    """Select the newest checkpoint not causally contaminated by rollback."""
    candidates = [
        checkpoint
        for checkpoint in checkpoints.get(aid, ())
        if not checkpoint.causal_frontier.intersection(invalidated_effect_ids)
    ]
    return max(candidates, key=lambda checkpoint: checkpoint.sequence, default=None)


def digest_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


__all__ = [
    "CheckpointBoundary",
    "EffectKind",
    "EffectRef",
    "EnvironmentSnapshot",
    "RestoreResult",
    "RollbackPlan",
    "RollbackResult",
    "ScopeCheckpoint",
    "compute_affected_agents",
    "compute_descendants",
    "digest_content",
    "select_checkpoint",
]
