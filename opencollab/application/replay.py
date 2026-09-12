"""Historical replay orchestration over the existing Scheduler resource lifecycle."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from uuid import uuid4

from opencollab.application.history import HistoryIndex
from opencollab.domain.history import ReplayPlan, TurnRecord
from opencollab.domain.rollback import compute_affected_agents, compute_descendants


@dataclass(slots=True)
class _ReplayTiming:
    started_at: float
    turn_started_at: float | None = None
    finished_at: float | None = None


@dataclass(frozen=True, slots=True)
class ReplayRun:
    replay_id: str | None
    status: str
    affected_agent_ids: tuple[int, ...] = ()
    old_epochs: tuple[tuple[int, int], ...] = ()
    new_epochs: tuple[tuple[int, int], ...] = ()
    task: asyncio.Task | None = None
    phase_latency_ms: tuple[tuple[str, float], ...] = ()
    failure_reason: str | None = None
    timing: _ReplayTiming | None = None


class SchedulerHistoryMixin:
    """Own stable history, fencing and replay; adapters only persist Scope state."""

    def _init_history(self):
        self._history = HistoryIndex()
        self._history_turns = {}
        self._history_attempts = {}
        self._history_bindings = {}
        self._history_anchors = {}
        self._history_closed = False
        self._history_enabled = False

    def _history_scope(self, aid):
        environment = getattr(self._sessions.get(aid), "env", None)
        enabled = self._history_enabled or bool(self._rollback_checkpointed_scopes)
        return enabled and self._has_effect_scopes(aid) and callable(getattr(environment, "checkpoint_scope", None))

    def _history_turn(self, aid):
        return self._history_turns.get(aid)

    def _begin_history_turn(self, aid):
        if not self._history_scope(aid):
            return
        previous = self._history_turns.get(aid)
        attempt = self._history_attempts.get(aid, 0) + 1
        self._history_attempts[aid] = attempt
        entry = self._history.record(aid, self.current_effect_epoch(aid), "turn_started", status="running")
        self._history_turns[aid] = TurnRecord(
            entry.team_id, "turn-" + uuid4().hex, aid, entry.epoch, attempt,
            "running", entry.timestamp, entry.replay_id, previous.turn_id if previous else None,
        )

    def _history_record(self, aid, boundary, **metadata):
        turn = self._history_turn(aid)
        return self._history.record(
            aid, self.current_effect_epoch(aid), boundary,
            turn_id=turn.turn_id if turn else None,
            attempt=turn.attempt if turn else 0, **metadata,
        )

    def _effect_history(self, effect):
        parents = {
            self._history._by_effect[key] for key in effect.parent_effect_ids if key in self._history._by_effect
        }
        head = self._history._heads.get(effect.producer_aid)
        if head:
            parents.add(head)
        self._history_record(
            effect.producer_aid, effect.kind, effect_id=effect.effect_id, parents=tuple(sorted(parents)),
        )

    def _history_failure(self, aid, boundary, reason):
        # Reasons are enumerated by the caller, never taken from exception text.
        self._history_record(aid, boundary, status="failed", failure_reason=reason)
        self._rollback_service._log("history_failure", {"aid": aid, "boundary": boundary, "reason": reason})

    @contextmanager
    def _tool_history_transaction(self, aid, messages):
        if not self._history_scope(aid):
            yield
            return
        try:
            with self._rollback_service.effect_transaction(), self._history.transaction():
                for message in messages:
                    self._record_lifecycle_effect(
                        producer_aid=aid, kind="tool_result", content=str(message.get("content", "")),
                    )
                yield
        except BaseException:
            self._history_failure(aid, "tool_result", "effect_registration_failed")
            raise

    async def _complete_history_turn(self, aid, content):
        if not self._history_scope(aid):
            return True
        try:
            with self._rollback_service.effect_transaction(), self._history.transaction():
                effect_id = self._record_lifecycle_effect(
                    producer_aid=aid, kind="child_result" if aid else "turn_result", content=content,
                )
                if aid:
                    self._rollback_child_effects[aid] = effect_id
            await self._autosave_history(aid, "turn_completed")
            return True
        except Exception:
            self._history_failure(aid, "turn_completed", "effect_registration_failed")
            return False

    def _checkpoint_binding(self, checkpoint):
        effects = self._rollback_service.effects
        fingerprints = tuple(
            (key, self._rollback_service._effect_fingerprint(effects[key]))
            for key in sorted(checkpoint.causal_frontier)
        )
        digest = sha256(json.dumps((
            checkpoint.filesystem_revision, checkpoint.filesystem_digest,
            checkpoint.environment.digest(), checkpoint.workspace_identity, fingerprints,
        ), separators=(",", ":")).encode()).hexdigest()
        return digest

    async def _checkpoint_history(self, aid, boundary, frontier=None):
        service = self._rollback_service
        if frontier is None:
            frontier = service.causal_frontier(aid)
        for key in frontier:
            if key not in service.effects or service.effects[key].status != "active":
                raise ValueError("invalid_checkpoint_frontier")
        checkpoint = await service.create_checkpoint(aid, boundary, frontier)
        await service.validate_checkpoint(aid, checkpoint.checkpoint_id)
        entry = self._history_record(
            aid, boundary, checkpoint_id=checkpoint.checkpoint_id,
            effect_id=max(frontier, default=None),
            filesystem_digest=checkpoint.filesystem_digest,
            environment_digest=checkpoint.environment.digest(),
            workspace_identity_digest=sha256(checkpoint.workspace_identity.encode()).hexdigest(),
        )
        self._history_bindings[checkpoint.checkpoint_id] = (self._checkpoint_binding(checkpoint), entry.entry_id)
        self._rollback_checkpointed_scopes.add(aid)
        return checkpoint

    async def _autosave_history(self, aid, boundary):
        if not self._history_scope(aid) or self._history_closed or aid in self._rollback_fenced:
            return
        try:
            await self.create_checkpoint(aid, boundary)
        except Exception:
            self._history_failure(aid, boundary, "persistence_failure")

    def _history_unavailable(self, entry):
        if self._history_closed:
            return "handle_closed"
        if not entry.checkpoint_id:
            return None
        service = self._rollback_service
        if entry.agent_id not in service._environments:
            return "scope_unavailable"
        try:
            checkpoint = service.get_checkpoint(entry.agent_id, entry.checkpoint_id)
            if self._checkpoint_binding(checkpoint) != self._history_bindings[entry.checkpoint_id][0]:
                return "invalid_checkpoint"
        except (KeyError, ValueError):
            return "invalid_checkpoint"
        if any(service.effects[key].status != "active" for key in checkpoint.causal_frontier):
            return "invalidated_source"
        return None

    def history_list(self):
        return tuple(self._history.view(e, self._history_unavailable(e)) for e in self._history.entries())

    def history_get(self, entry_id):
        entry = self._history.get(entry_id)
        return self._history.view(entry, self._history_unavailable(entry))

    def history_replayable(self, checkpoint_id):
        try:
            entry = self._history.checkpoint(checkpoint_id)
        except ValueError:
            return False
        return self._history.view(entry, self._history_unavailable(entry)).replayable

    def close_history(self):
        self._history_closed = True
        self._history.close()

    def _replay_plan(self, checkpoint_id):
        source = self._history.checkpoint(checkpoint_id)
        if self._history_unavailable(source) or source.status != "stable":
            raise ValueError("invalid_checkpoint")
        service = self._rollback_service
        checkpoint = service.get_checkpoint(source.agent_id, checkpoint_id)
        roots = {
            key for key, effect in service.effects.items()
            if effect.producer_aid == source.agent_id and key not in checkpoint.causal_frontier
            and effect.status == "active"
        }
        descendants = compute_descendants(service.effects, roots)
        affected = set(compute_affected_agents(service.effects, service._consumers, set(descendants)))
        affected.update((source.agent_id, 0))
        affected.update(self._pending_rollback_ancestors(affected))
        selections = {source.agent_id: checkpoint_id}
        for aid in sorted(affected - {source.agent_id}):
            candidates = [
                e for e in self._history.entries()
                if e.agent_id == aid and e.checkpoint_id and e.status == "stable"
                and not self._history_unavailable(e)
                and not service.get_checkpoint(aid, e.checkpoint_id).causal_frontier.intersection(descendants)
            ]
            earlier = [e for e in candidates if e.sequence <= source.sequence]
            selected = max(earlier, key=lambda e: e.sequence, default=None)
            if selected is None:
                selected = next((e for e in candidates if e.boundary == "initial"), None)
            if selected is None:
                raise ValueError("invalid_checkpoint")
            selections[aid] = selected.checkpoint_id
        frontiers = tuple((aid, tuple(sorted(service.causal_frontier(aid)))) for aid in sorted(affected))
        fingerprints = tuple(
            (key, service._effect_fingerprint(value)) for key, value in sorted(service.effects.items())
        )
        return ReplayPlan(
            source.entry_id, checkpoint_id, tuple(sorted(selections.items())),
            tuple((aid, self.current_effect_epoch(aid)) for aid in sorted(affected)),
            fingerprints, frontiers, self._history.digest(),
        )

    async def _validate_replay_plan(self, plan):
        for aid, checkpoint_id in plan.checkpoint_by_agent:
            cp = await self._rollback_service.validate_checkpoint(aid, checkpoint_id)
            if self._checkpoint_binding(cp) != self._history_bindings[checkpoint_id][0]:
                raise ValueError("invalid_checkpoint")

    async def replay_from(self, checkpoint_id, message, budget_tokens):
        if self._history_closed or self._shutting_down:
            raise RuntimeError("handle_closed")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be non-empty")
        if isinstance(budget_tokens, bool) or not isinstance(budget_tokens, int) or budget_tokens < 1:
            return ReplayRun(None, "budget_exhausted", failure_reason="budget_exhausted")
        # Budget exhaustion is independent of restore success and checked before fencing.
        cap = self._per_agent_cap()
        available = self._max_budget_tokens - self.used_tokens
        if budget_tokens > available or (cap is not None and budget_tokens > cap - self._session_used_tokens(0)):
            return ReplayRun(None, "budget_exhausted", failure_reason="budget_exhausted")
        phase_start = time.perf_counter()
        timing = _ReplayTiming(phase_start)
        timings = []
        try:
            plan = self._replay_plan(checkpoint_id)
            await self._validate_replay_plan(plan)
        except Exception:
            return ReplayRun(None, "invalid_checkpoint", failure_reason="invalid_checkpoint")
        timings.append(("preflight", (time.perf_counter() - phase_start) * 1000))
        async with self._rollback_operation_lock:
            if self._history_closed:
                raise RuntimeError("handle_closed")
            try:
                current = self._replay_plan(checkpoint_id)
                if current.digest() != plan.digest():
                    return ReplayRun(None, "stale_plan", failure_reason="stale_plan")
                await self._validate_replay_plan(current)
                if self._replay_plan(checkpoint_id).digest() != plan.digest():
                    return ReplayRun(None, "stale_plan", failure_reason="stale_plan")
            except Exception:
                return ReplayRun(None, "invalid_checkpoint", failure_reason="invalid_checkpoint")
            checkpoints = {
                aid: self._rollback_service.get_checkpoint(aid, cid) for aid, cid in plan.checkpoint_by_agent
            }
            affected = set(plan.affected_agent_ids)
            phase = "fence_failed"
            operation = None
            try:
                operation = self._begin_rollback(("replay", checkpoint_id), checkpoints)
                self._fence_agents(affected)
                phase_start = time.perf_counter()
                phase = "settlement_failed"
                await self._cancel_fenced_tasks(affected)
                await self._reject_rollback_messages(affected)
                await self._settle_rollback_turns(affected)
                timings.append(("settlement", (time.perf_counter() - phase_start) * 1000))
                phase_start = time.perf_counter()
                phase = "quiesce_failed"
                await self._quiesce_agents(affected)
                self._reset_fenced_sessions(affected)
                timings.append(("quiesce", (time.perf_counter() - phase_start) * 1000))
                phase_start = time.perf_counter()
                # Capture a stable undo boundary only after old owners are quiescent.
                phase = "invalid_checkpoint"
                for aid in sorted(affected):
                    await self._checkpoint_history(aid, "before_replay")
                timings.append(("before_replay_checkpoint", (time.perf_counter() - phase_start) * 1000))
                phase_start = time.perf_counter()
                phase = "restore_failed"
                for aid, cp in sorted(checkpoints.items()):
                    restored = await self._rollback_service.rollback_to_checkpoint(aid, cp.checkpoint_id)
                    if restored.status != "restored":
                        raise RuntimeError("restore_failed")
                    if (restored.filesystem_digest != cp.filesystem_digest
                            or restored.environment_digest != cp.environment.digest()):
                        phase = "verification_failed"
                        raise RuntimeError("verification_failed")
                timings.append(("restore_and_verification", (time.perf_counter() - phase_start) * 1000))
                phase_start = time.perf_counter()
                phase = "budget_exhausted"
                self.reserve_continuation_budget(0, budget_tokens)
                replay_id = "replay-" + uuid4().hex
                phase = "replay_start_failed"
                self._finish_rollback(operation, True)
                for aid in sorted(affected):
                    epoch = self.current_effect_epoch(aid) + 1
                    self._rollback_epochs[aid] = epoch
                    source_entry = self._history.checkpoint(checkpoints[aid].checkpoint_id)
                    self._history.branch(aid, epoch, replay_id, source_entry.entry_id)
                    self._history_anchors[aid] = checkpoints[aid].causal_frontier
                    self._rollback_child_effects.pop(aid, None)
                    self._rollback_operations.pop(aid, None)
                    self._history_turns.pop(aid, None)
                    # Keep the immutable role/tool instructions required to run
                    # the Agent, but discard prior user, assistant, and tool turns.
                    messages = self._sessions[aid].state.messages
                    messages[:] = [message for message in messages if message.get("role") == "system"]
                self._rollback_fenced.difference_update(affected)
                self._rollback_failed.difference_update(affected)
                self._rollback_affected.difference_update(affected)
                self._rollback_interrupted.difference_update(affected)
                summary = json.dumps({
                    "checkpoint_id": checkpoint_id,
                    "boundary": self._history.get(plan.source_entry_id).boundary,
                    "affected_agent_ids": sorted(affected),
                }, sort_keys=True)
                task = asyncio.create_task(self._run_replay_turn(
                    replay_id, message + "\nStable history summary: " + summary, affected, timing,
                ))
                timings.append(("new_turn", (time.perf_counter() - phase_start) * 1000))
                return ReplayRun(
                    replay_id, "started", tuple(sorted(affected)), plan.epoch_by_agent,
                    tuple((aid, self.current_effect_epoch(aid)) for aid in sorted(affected)), task, tuple(timings),
                    timing=timing,
                )
            except BaseException as exc:
                if operation is not None:
                    self._finish_rollback(operation, False)
                    self._fence_agents(affected)
                self._history_failure(0, "replay_failed", phase)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                return ReplayRun(
                    None,
                    phase,
                    tuple(sorted(affected)),
                    plan.epoch_by_agent,
                    failure_reason=phase,
                )

    async def _run_replay_turn(self, replay_id, message, affected, timing):
        timing.turn_started_at = time.perf_counter()
        token = self._continuation_aid.set(0)
        try:
            self._history_record(0, "replay_started", status="running")
            value = await self.run_turn(0, message)
            self._history_record(0, "replay_completed")
            return value
        except BaseException:
            self._fence_agents(affected)
            self._rollback_failed.update(affected)
            self._history_failure(0, "replay_failed", "replay_start_failed")
            raise
        finally:
            self._release_turn_lease(0)
            self._continuation_aid.reset(token)
            timing.finished_at = time.perf_counter()


__all__ = ["ReplayRun", "SchedulerHistoryMixin"]
