"""Pure causal rollback values and graph operations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping

_EffectKind = Literal["child_result", "teammate_message", "tool_result"]
_EffectStatus = Literal["untrusted", "verified", "quarantined"]
_BoundaryKind = Literal[
    "scope_initialization",
    "child_result",
    "teammate_message",
    "tool_call",
]
_RestoreStatus = Literal["restored", "skipped", "pending", "failed"]
_AdoptionStatus = Literal["adopted", "skipped", "conflict", "failed"]
_WorkspaceEntryKind = Literal["file", "symlink"]
_WorkspaceEffectState = Literal["unseen", "collected", "delivered", "acknowledged"]


@dataclass(frozen=True, slots=True)
class BaselineEntry:
    """One immutable file identity in a workspace baseline or Effect."""

    path: str
    kind: _WorkspaceEntryKind
    mode: int
    size: int
    content_hash: str
    control_plane: bool = False

    def __post_init__(self) -> None:
        normalized = self.path.replace("\\", "/")
        parts = normalized.split("/")
        if (
            not normalized
            or normalized.startswith("/")
            or "\0" in normalized
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError("workspace entry path must be a safe relative path")
        if self.kind not in {"file", "symlink"}:
            raise ValueError("workspace entry kind is invalid")
        if self.mode < 0 or self.size < 0:
            raise ValueError("workspace entry mode and size must be non-negative")
        if not self.content_hash:
            raise ValueError("workspace entry content hash is required")


@dataclass(frozen=True, slots=True)
class WorkspaceBaseline:
    """Ignored workspace inputs captured before an Agent run starts."""

    entries: tuple[BaselineEntry, ...] = ()

    def __post_init__(self) -> None:
        paths = [entry.path for entry in self.entries]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("workspace baseline entries must have unique sorted paths")

    def by_path(self) -> dict[str, BaselineEntry]:
        return {entry.path: entry for entry in self.entries}

    def digest(self) -> str:
        payload = "\0".join(
            f"{entry.path}\0{entry.kind}\0{entry.mode}\0{entry.size}\0{entry.content_hash}"
            for entry in self.entries
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class WorkspaceRevision:
    """Immutable Git snapshot exported by one Agent Scope."""

    revision: str
    base_revision: str
    changed: bool = True
    files: tuple[BaselineEntry, ...] = ()

    def __post_init__(self) -> None:
        if not self.revision or not self.base_revision:
            raise ValueError("workspace revision and base revision are required")


@dataclass(frozen=True, slots=True)
class AdoptionResult:
    """Outcome of applying an immutable workspace revision to one Scope."""

    status: _AdoptionStatus
    revision: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class EffectRef:
    """One immutable node in the causal effect graph."""

    effect_id: str
    producer_aid: int
    kind: _EffectKind
    epoch: int
    attempt: int
    parent_effect_ids: tuple[str, ...] = ()
    content_hash: str = ""
    status: _EffectStatus = "untrusted"
    workspace_revision: str | None = None
    base_workspace_revision: str | None = None
    workspace_files: tuple[BaselineEntry, ...] = ()

    def __post_init__(self) -> None:
        if not self.effect_id:
            raise ValueError("effect_id cannot be empty")
        if self.epoch < 0 or self.attempt < 0:
            raise ValueError("effect epoch and attempt must be non-negative")
        if len(set(self.parent_effect_ids)) != len(self.parent_effect_ids):
            raise ValueError("parent effect IDs must be unique")
        if self.effect_id in self.parent_effect_ids:
            raise ValueError("an effect cannot be its own parent")
        if (self.workspace_revision is None) != (self.base_workspace_revision is None):
            raise ValueError("workspace revision and base revision must be supplied together")


@dataclass(frozen=True, slots=True)
class LineageEnvelope:
    """Typed sidecar attached to a delivered result."""

    effect: EffectRef
    consumer_aid: int | None = None
    source_message_id: str | None = None


@dataclass(frozen=True, slots=True)
class RollbackState:
    """Per-session causal state restored at checkpoint boundaries."""

    branch: str = "main"
    epoch: int = 0
    attempt: int = 0
    causal_frontier: frozenset[str] = field(default_factory=frozenset)
    quarantined_effects: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.branch:
            raise ValueError("rollback branch cannot be empty")
        if self.epoch < 0 or self.attempt < 0:
            raise ValueError("rollback epoch and attempt must be non-negative")


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """Immutable, exact environment variable mapping."""

    entries: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        names = [name for name, _ in self.entries]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("environment entries must have unique sorted names")
        if any(not isinstance(name, str) or not isinstance(value, str) for name, value in self.entries):
            raise TypeError("environment names and values must be strings")

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> EnvironmentSnapshot:
        return cls(tuple(sorted(values.items())))

    def as_dict(self) -> dict[str, str]:
        return dict(self.entries)

    def digest(self) -> str:
        payload = "\0".join(f"{name}\0{value}" for name, value in self.entries)
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointBoundary:
    """The consumption or execution boundary protected by a checkpoint."""

    kind: _BoundaryKind
    effect_id: str | None = None


@dataclass(frozen=True, slots=True)
class ScopeCheckpoint:
    """Exact filesystem, environment, and causal state for one Agent Scope."""

    checkpoint_id: str
    owner_aid: int
    sequence: int
    filesystem_revision: str
    environment: EnvironmentSnapshot
    causal_frontier: frozenset[str]
    boundary_kind: _BoundaryKind
    boundary_effect_id: str | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint_id or not self.filesystem_revision:
            raise ValueError("checkpoint identity and filesystem revision are required")
        if self.owner_aid < 0 or self.sequence < 0:
            raise ValueError("checkpoint owner and sequence must be non-negative")


@dataclass(frozen=True, slots=True)
class RestoreResult:
    """Bounded outcome of restoring one Agent Scope."""

    agent_id: int
    checkpoint_id: str | None
    status: _RestoreStatus
    files_changed: int = 0
    reason: str | None = None


def compute_ancestors(effects: Mapping[str, EffectRef], effect_ids: set[str]) -> set[str]:
    """Return known transitive ancestors, excluding the supplied nodes."""
    ancestors: set[str] = set()
    stack = list(effect_ids)
    while stack:
        current = effects.get(stack.pop())
        if current is None:
            continue
        for parent_id in current.parent_effect_ids:
            if parent_id not in effect_ids and parent_id not in ancestors:
                ancestors.add(parent_id)
                stack.append(parent_id)
    return ancestors


def compute_descendants(effects: Mapping[str, EffectRef], root_ids: set[str]) -> set[str]:
    """Return known transitive descendants, including known roots."""
    children: dict[str, set[str]] = {}
    for effect_id, effect in effects.items():
        for parent_id in effect.parent_effect_ids:
            children.setdefault(parent_id, set()).add(effect_id)
    visited: set[str] = set()
    stack = list(root_ids)
    while stack:
        effect_id = stack.pop()
        if effect_id in visited or effect_id not in effects:
            continue
        visited.add(effect_id)
        stack.extend(children.get(effect_id, ()))
    return visited


def reduce_causal_frontier(
    effects: Mapping[str, EffectRef],
    frontier: set[str] | frozenset[str],
    consumed_effect_id: str,
) -> frozenset[str]:
    """Add an effect while removing frontier nodes represented by its ancestry."""
    if consumed_effect_id not in effects:
        raise ValueError(f"unknown effect_id: {consumed_effect_id}")
    represented = compute_ancestors(effects, {consumed_effect_id})
    return frozenset((set(frontier) - represented) | {consumed_effect_id})


def quarantine_effect(effect: EffectRef) -> EffectRef:
    """Return a quarantined copy without mutating graph values."""
    return replace(effect, status="quarantined")


def compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def lineage_envelope_to_dict(envelope: LineageEnvelope) -> dict[str, Any]:
    effect = envelope.effect
    return {
        "effect": {
            "effect_id": effect.effect_id,
            "producer_aid": effect.producer_aid,
            "kind": effect.kind,
            "epoch": effect.epoch,
            "attempt": effect.attempt,
            "parent_effect_ids": list(effect.parent_effect_ids),
            "content_hash": effect.content_hash,
            "status": effect.status,
            "workspace_revision": effect.workspace_revision,
            "base_workspace_revision": effect.base_workspace_revision,
            "workspace_files": [
                {
                    "path": entry.path,
                    "kind": entry.kind,
                    "mode": entry.mode,
                    "size": entry.size,
                    "content_hash": entry.content_hash,
                    "control_plane": entry.control_plane,
                }
                for entry in effect.workspace_files
            ],
        },
        "consumer_aid": envelope.consumer_aid,
        "source_message_id": envelope.source_message_id,
    }


def lineage_envelope_from_dict(value: object) -> LineageEnvelope | None:
    if not isinstance(value, dict) or not isinstance(value.get("effect"), dict):
        return None
    effect = value["effect"]
    try:
        return LineageEnvelope(
            effect=EffectRef(
                effect_id=str(effect["effect_id"]),
                producer_aid=int(effect["producer_aid"]),
                kind=effect["kind"],
                epoch=int(effect["epoch"]),
                attempt=int(effect["attempt"]),
                parent_effect_ids=tuple(str(item) for item in effect.get("parent_effect_ids", ())),
                content_hash=str(effect.get("content_hash", "")),
                status=effect.get("status", "untrusted"),
                workspace_revision=(
                    str(effect["workspace_revision"]) if effect.get("workspace_revision") is not None else None
                ),
                base_workspace_revision=(
                    str(effect["base_workspace_revision"])
                    if effect.get("base_workspace_revision") is not None
                    else None
                ),
                workspace_files=tuple(
                    BaselineEntry(
                        path=str(item["path"]),
                        kind=item["kind"],
                        mode=int(item["mode"]),
                        size=int(item["size"]),
                        content_hash=str(item["content_hash"]),
                        control_plane=bool(item.get("control_plane", False)),
                    )
                    for item in effect.get("workspace_files", ())
                    if isinstance(item, dict)
                ),
            ),
            consumer_aid=(int(value["consumer_aid"]) if value.get("consumer_aid") is not None else None),
            source_message_id=(str(value["source_message_id"]) if value.get("source_message_id") is not None else None),
        )
    except (KeyError, TypeError, ValueError):
        return None
