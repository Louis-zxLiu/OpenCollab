"""Immutable identifiers and deterministic lineage algorithms for Team history."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class TurnRecord:
    team_id: str
    turn_id: str
    agent_id: int
    epoch: int
    attempt: int
    status: str
    timestamp: str
    replay_id: str | None = None
    parent_turn_id: str | None = None

    def __post_init__(self):
        if not self.team_id or not self.turn_id or min(self.agent_id, self.epoch, self.attempt) < 0:
            raise ValueError("invalid turn identity")
        if self.parent_turn_id == self.turn_id:
            raise ValueError("self-parent turn")

    def digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    entry_id: str
    team_id: str
    agent_id: int
    epoch: int
    boundary: str
    status: str
    timestamp: str
    sequence: int = 0
    attempt: int = 0
    turn_id: str | None = None
    checkpoint_id: str | None = None
    effect_id: str | None = None
    replay_id: str | None = None
    parent_entry_ids: tuple[str, ...] = ()
    filesystem_digest: str | None = None
    environment_digest: str | None = None
    workspace_identity_digest: str | None = None
    failure_reason: str | None = None

    def __post_init__(self):
        if not self.entry_id or not self.team_id or min(self.agent_id, self.epoch, self.sequence, self.attempt) < 0:
            raise ValueError("invalid history identity")
        if self.entry_id in self.parent_entry_ids:
            raise ValueError("self-parent history")
        if len(set(self.parent_entry_ids)) != len(self.parent_entry_ids):
            raise ValueError("duplicate parent history")

    def digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    source_entry_id: str
    source_checkpoint_id: str
    checkpoint_by_agent: tuple[tuple[int, str], ...]
    epoch_by_agent: tuple[tuple[int, int], ...]
    effect_fingerprints: tuple[tuple[str, str], ...]
    frontier_by_agent: tuple[tuple[int, tuple[str, ...]], ...]
    history_digest: str

    def __post_init__(self):
        for pairs in (self.checkpoint_by_agent, self.epoch_by_agent, self.effect_fingerprints, self.frontier_by_agent):
            keys = [key for key, _ in pairs]
            if len(set(keys)) != len(keys) or keys != sorted(keys):
                raise ValueError("plan bindings must be sorted and unique")

    @property
    def affected_agent_ids(self) -> frozenset[int]:
        return frozenset(aid for aid, _ in self.checkpoint_by_agent)

    def digest(self) -> str:
        return _digest(asdict(self))


def validate_lineage(entries: Mapping[str, HistoryEntry]) -> None:
    """Validate without recursion, including malformed externally supplied graphs."""
    visited, active = set(), set()
    for root in entries:
        stack = [(root, False)]
        while stack:
            key, exiting = stack.pop()
            if exiting:
                active.remove(key)
                visited.add(key)
                continue
            if key in active:
                raise ValueError("history cycle")
            if key in visited:
                continue
            if key not in entries:
                raise ValueError("unknown parent history")
            entry = entries[key]
            if key != entry.entry_id:
                raise ValueError("history identity mismatch")
            active.add(key)
            stack.append((key, True))
            for parent in reversed(entry.parent_entry_ids):
                if parent not in entries:
                    raise ValueError("unknown parent history")
                if entries[parent].team_id != entry.team_id:
                    raise ValueError("foreign Team lineage")
                if entries[parent].status == "invalidated":
                    raise ValueError("invalidated parent history")
                stack.append((parent, False))


__all__ = ["HistoryEntry", "ReplayPlan", "TurnRecord", "validate_lineage"]
