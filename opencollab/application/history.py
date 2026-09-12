"""Append-only application history and its redacted read contract."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from typing import Protocol
from uuid import uuid4

from opencollab.domain.history import HistoryEntry


@dataclass(frozen=True, slots=True)
class HistoryView:
    entry_id: str
    checkpoint_id: str | None
    effect_id: str | None
    turn_id: str | None
    agent_id: int
    epoch: int
    boundary: str
    status: str
    timestamp: str
    filesystem_digest: str | None
    environment_digest: str | None
    workspace_identity_digest: str | None
    replayable: bool
    failure_reason: str | None


class HistoryPort(Protocol):
    def history_list(self) -> tuple[HistoryView, ...]: ...
    def history_get(self, entry_id: str) -> HistoryView: ...
    def history_replayable(self, checkpoint_id: str) -> bool: ...
    async def replay_from(self, checkpoint_id: str, message: str, budget_tokens: int): ...


class HistoryIndex:
    """Store only immutable metadata; snapshot values stay in RollbackService."""

    def __init__(self):
        self.team_id = "team-" + uuid4().hex
        self._entries: dict[str, HistoryEntry] = {}
        self._checkpoints: dict[str, str] = {}
        self._by_effect: dict[str, str] = {}
        self._epochs: dict[int, int] = {}
        self._heads: dict[int, str] = {}
        self._replays: dict[int, str] = {}
        self._closed = False

    @contextmanager
    def transaction(self):
        previous = (self._entries.copy(), self._checkpoints.copy(), self._heads.copy(), self._by_effect.copy())
        try:
            yield
        except BaseException:
            self._entries, self._checkpoints, self._heads, self._by_effect = previous
            raise

    def append(self, entry: HistoryEntry) -> HistoryEntry:
        if self._closed:
            raise RuntimeError("history_closed")
        if entry.team_id != self.team_id:
            raise ValueError("foreign Team history")
        if entry.entry_id in self._entries:
            raise ValueError("duplicate history entry")
        if entry.epoch != self._epochs.get(entry.agent_id, 0):
            raise ValueError("old epoch history")
        stored = replace(entry, sequence=len(self._entries) + 1)
        # Existing frozen nodes were already validated. A new node can only
        # reference existing parents, so it cannot introduce a cycle.
        for parent_id in stored.parent_entry_ids:
            parent = self._entries.get(parent_id)
            if parent is None:
                raise ValueError("unknown parent history")
            if parent.team_id != entry.team_id or parent.status == "invalidated":
                raise ValueError("invalid parent history")
        if stored.checkpoint_id and stored.checkpoint_id in self._checkpoints:
            raise ValueError("duplicate checkpoint history")
        self._entries[stored.entry_id] = stored
        if stored.effect_id and not stored.checkpoint_id:
            self._by_effect[stored.effect_id] = stored.entry_id
        if stored.checkpoint_id:
            self._checkpoints[stored.checkpoint_id] = stored.entry_id
        self._heads[stored.agent_id] = stored.entry_id
        return stored

    def record(self, aid: int, epoch: int, boundary: str, *, parents=None, **metadata) -> HistoryEntry:
        if parents is None:
            parents = (self._heads[aid],) if aid in self._heads else ()
        return self.append(HistoryEntry(
            entry_id="history-" + uuid4().hex, team_id=self.team_id, agent_id=aid,
            epoch=epoch, boundary=boundary, status=metadata.pop("status", "stable"),
            timestamp=datetime.now(timezone.utc).isoformat(), parent_entry_ids=tuple(parents),
            replay_id=self._replays.get(aid), **metadata,
        ))

    def branch(self, aid: int, epoch: int, replay_id: str, source_entry_id: str):
        if epoch != self._epochs.get(aid, 0) + 1:
            raise ValueError("replay epoch must increment once")
        self._epochs[aid] = epoch
        self._replays[aid] = replay_id
        self._heads[aid] = source_entry_id

    def entries(self) -> tuple[HistoryEntry, ...]:
        return tuple(self._entries.values())

    def get(self, entry_id: str) -> HistoryEntry:
        try:
            return self._entries[entry_id]
        except KeyError:
            raise ValueError("unknown_history_entry") from None

    def checkpoint(self, checkpoint_id: str) -> HistoryEntry:
        try:
            return self._entries[self._checkpoints[checkpoint_id]]
        except KeyError:
            raise ValueError("invalid_checkpoint") from None

    def view(self, entry: HistoryEntry, unavailable: str | None = None) -> HistoryView:
        reason = "handle_closed" if self._closed else unavailable or entry.failure_reason
        ready = bool(entry.checkpoint_id and entry.status == "stable" and reason is None)
        return HistoryView(
            entry.entry_id, entry.checkpoint_id, entry.effect_id, entry.turn_id,
            entry.agent_id, entry.epoch, entry.boundary, entry.status, entry.timestamp,
            entry.filesystem_digest, entry.environment_digest, entry.workspace_identity_digest,
            ready, reason,
        )

    def digest(self) -> str:
        return sha256("".join(e.digest() for e in self.entries()).encode()).hexdigest()

    def close(self):
        self._closed = True


__all__ = ["HistoryIndex", "HistoryPort", "HistoryView"]
