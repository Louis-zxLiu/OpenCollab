"""Scheduler integration for explicit, dependency-scoped rollback."""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass
from typing import Any

from opencollab.application.rollback import RollbackService
from opencollab.domain.rollback import EffectKind, RollbackPlan, RollbackResult, ScopeCheckpoint
from opencollab.domain.session import SessionPhase


@dataclass
class _RollbackOperation:
    key: tuple
    affected: frozenset[int]
    epochs: dict[int, int]
    checkpoints: dict[int, ScopeCheckpoint]
    phase: str = "fencing"
    failure_phase: str | None = None


class SchedulerRollbackMixin:
    """Fence affected agents, restore their Scopes, and remain idle."""

    def _init_rollback(self) -> None:
        self._rollback_service = RollbackService(self._tracer)
        self._init_history()
        # Serialize the scheduler-side fence/cancel/restore sequence as one
        # operation.  The service lock protects its graph, but cannot protect
        # the scheduler's task and lease maps.
        self._rollback_operation_lock = asyncio.Lock()
        self._rollback_turn_epoch: contextvars.ContextVar[tuple[int, int] | None] = (
            contextvars.ContextVar("rollback_turn_epoch", default=None)
        )
        self._rollback_fenced: set[int] = set()
        self._rollback_affected: set[int] = set()
        self._rollback_failed: set[int] = set()
        self._rollback_epochs: dict[int, int] = {}
        self._rollback_interrupted: set[int] = set()
        self._rollback_child_effects: dict[int, str] = {}
        self._rollback_checkpointed_scopes: set[int] = set()
        self._rollback_operations: dict[int, _RollbackOperation] = {}
        self._rollback_continuations: set[int] = set()
        self._continuation_aid: contextvars.ContextVar[int | None] = contextvars.ContextVar(
            "continuation_aid", default=None
        )

    def register_effect_environment(self, aid: int, environment: Any) -> None:
        self._rollback_service.register_environment(aid, environment)
        self._rollback_epochs.setdefault(aid, 0)

    def current_effect_epoch(self, aid: int) -> int:
        return self._rollback_epochs.get(aid, 0)

    def _has_effect_scopes(self, *aids: int) -> bool:
        """Environment-free sessions retain messaging, without rollback effects."""
        return all(aid in self._rollback_service._environments for aid in aids)

    def _record_lifecycle_effect(
        self,
        *,
        producer_aid: int,
        kind: EffectKind,
        content: str,
        consumer_aid: int | None = None,
    ) -> str:
        try:
            self._assert_effect_agent(producer_aid)
            if consumer_aid is not None:
                self._assert_effect_agent(consumer_aid)
            with self._rollback_service.effect_transaction(), self._history.transaction():
                effect = self.create_effect_ref(
                    producer_aid=producer_aid,
                    kind=kind,
                    epoch=self.current_effect_epoch(producer_aid),
                    attempt=0,
                    content=content,
                )
                if consumer_aid is not None:
                    self.consume_effect_ref(effect.effect_id, consumer_aid)
            return effect.effect_id
        except Exception:
            for aid in (producer_aid, consumer_aid):
                if aid is not None and self.table.get(aid) is not None:
                    self.table.get(aid).state.fail("effect_registration_failed")
            self._rollback_service._log(
                "effect_registration_failed", {"producer_aid": producer_aid, "kind": kind},
            )
            raise

    def _record_child_effect(self, child_aid: int, parent_aid: int, content: str) -> str:
        self._assert_effect_agent(child_aid)
        self._assert_effect_agent(parent_aid)
        existing = self._rollback_child_effects.get(child_aid)
        if existing is not None:
            self.consume_effect_ref(existing, parent_aid)
            return existing
        effect_id = self._record_lifecycle_effect(
            producer_aid=child_aid,
            kind="child_result",
            content=content,
            consumer_aid=parent_aid,
        )
        self._rollback_child_effects[child_aid] = effect_id
        return effect_id

    def _assert_effect_agent(self, aid: int, *, epoch: int | None = None) -> None:
        self._assert_agent_active(aid, epoch=epoch)
        if aid not in self._sessions or self.table.get(aid) is None:
            raise RuntimeError(f"Agent {aid} is not registered")
        self._rollback_service._require_environment(aid)

    def _assert_agent_active(self, aid: int, *, epoch: int | None = None) -> None:
        if aid in self._rollback_fenced:
            raise RuntimeError(f"Agent {aid} is fenced by an explicit rollback")
        turn = self._rollback_turn_epoch.get()
        if turn is not None and turn[0] == aid and turn[1] != self.current_effect_epoch(aid):
            raise RuntimeError(
                f"Agent {aid} epoch {turn[1]} is stale; current epoch is "
                f"{self.current_effect_epoch(aid)}"
            )
        if epoch is not None and epoch != self.current_effect_epoch(aid):
            raise RuntimeError(
                f"Agent {aid} epoch {epoch} is stale; current epoch is "
                f"{self.current_effect_epoch(aid)}"
            )

    def create_effect_ref(
        self,
        *,
        producer_aid: int,
        kind: EffectKind,
        epoch: int,
        attempt: int,
        parent_effect_ids: tuple[str, ...] = (),
        content: str = "",
    ):
        self._assert_effect_agent(producer_aid, epoch=epoch)
        parents = parent_effect_ids or self._rollback_service.causal_frontier(producer_aid)
        for parent in parents:
            if parent not in self._history_anchors.get(producer_aid, ()):
                self._assert_effect_available(parent)
        with self._rollback_service.effect_transaction(), self._history.transaction():
            effect = self._rollback_service.create_effect(
                producer_aid=producer_aid, kind=kind, epoch=epoch, attempt=attempt,
                parent_effect_ids=parent_effect_ids, content=content,
            )
            self._effect_history(effect)
            return effect

    def consume_effect_ref(self, effect_id: str, consumer_aid: int) -> None:
        self._assert_effect_agent(consumer_aid)
        self._assert_effect_available(effect_id)
        with self._rollback_service.effect_transaction(), self._history.transaction():
            self._rollback_service.register_consumer(effect_id, consumer_aid)

    def _assert_effect_available(self, effect_id: str) -> None:
        effect = self._rollback_service.effects.get(effect_id)
        if effect is None:
            raise ValueError(f"unknown effect: {effect_id}")
        if effect.status != "active":
            raise ValueError(f"cannot use invalidated effect: {effect_id}")
        if effect.epoch != self.current_effect_epoch(effect.producer_aid):
            raise ValueError("old epoch effect")
        self._assert_agent_active(effect.producer_aid)

    async def create_checkpoint(self, aid: int, boundary="initial", causal_frontier=None):
        self._assert_agent_active(aid)
        session = self._sessions.get(aid)
        environment = getattr(session, "env", None) if session is not None else None
        if environment is None:
            raise RuntimeError(f"Agent {aid} has no environment")
        self.register_effect_environment(aid, environment)
        if causal_frontier is None:
            causal_frontier = self._rollback_service.causal_frontier(aid)
        return await self._checkpoint_history(aid, boundary, causal_frontier)

    def preview_rollback(self, effect_ids: set[str]) -> RollbackPlan:
        base_plan = self._rollback_service.preview_rollback(effect_ids)
        coordinators = self._pending_rollback_ancestors(
            set(base_plan.affected_agent_ids)
        )
        for operation in self._rollback_operations.values():
            if operation.phase == "failed" and operation.key == ("effects", frozenset(effect_ids)):
                coordinators.update(operation.affected - base_plan.affected_agent_ids)
        if not coordinators:
            return base_plan
        return self._rollback_service.preview_rollback(
            effect_ids,
            coordinator_agent_ids=frozenset(coordinators),
        )

    async def rollback_to_checkpoint(self, aid: int, checkpoint_id: str):
        async with self._rollback_operation_lock:
            checkpoint = await self._rollback_service.validate_checkpoint(aid, checkpoint_id)
            operation = self._begin_rollback(("checkpoint", aid, checkpoint_id), {aid: checkpoint})
            try:
                await self._prepare_rollback(operation)
                result = await self._rollback_service.rollback_to_checkpoint(aid, checkpoint_id)
                self._finish_rollback(operation, result.status == "restored")
                return result
            except BaseException:
                self._finish_rollback(operation, False)
                raise

    async def rollback_effect(
        self,
        effect_ids: set[str],
        *,
        expected_plan_digest: str,
    ) -> RollbackResult:
        async with self._rollback_operation_lock:
            plan = self.preview_rollback(effect_ids)
            if expected_plan_digest != plan.digest():
                raise ValueError("rollback plan digest does not match the current plan")
            await self._rollback_service.validate_plan(plan)
            if self.preview_rollback(effect_ids).digest() != expected_plan_digest:
                raise ValueError("rollback plan changed during checkpoint preflight")
            operation = self._begin_rollback(
                ("effects", frozenset(effect_ids)),
                {aid: checkpoint for aid, checkpoint in plan.checkpoint_by_agent.items() if checkpoint is not None},
            )
            try:
                await self._prepare_rollback(operation)
                result = await self._rollback_service.rollback_effect(
                    effect_ids,
                    expected_plan_digest=expected_plan_digest,
                    coordinator_agent_ids=plan.coordinator_agent_ids,
                )
                self._finish_rollback(operation, result.invalidated)
                return result
            except BaseException:
                self._finish_rollback(operation, False)
                raise

    def _begin_rollback(self, key: tuple, checkpoints: dict[int, ScopeCheckpoint]) -> _RollbackOperation:
        affected = frozenset(checkpoints)
        for aid in affected:
            previous = self._rollback_operations.get(aid)
            if previous is not None and (
                previous.phase != "failed" or previous.key != key
                or previous.affected != affected or previous.checkpoints != checkpoints
            ):
                raise RuntimeError("Agent has an unresolved rollback; resume or retry that operation first")
        operation = _RollbackOperation(
            key, affected, {aid: self.current_effect_epoch(aid) for aid in affected}, checkpoints
        )
        self._rollback_affected.update(affected)
        self._rollback_failed.update(affected)
        for aid in affected:
            self._rollback_operations[aid] = operation
        return operation

    async def _prepare_rollback(self, operation: _RollbackOperation) -> None:
        affected = set(operation.affected)
        self._fence_agents(affected)
        operation.phase = "cancelling"
        # Cancel delivery owners before taking recipient locks they may hold.
        await self._cancel_fenced_tasks(affected)
        await self._reject_rollback_messages(affected)
        operation.phase = "settling"
        await self._settle_rollback_turns(affected)
        operation.phase = "quiescing"
        await self._quiesce_agents(affected)
        self._reset_fenced_sessions(affected)
        operation.phase = "restoring"

    def _finish_rollback(self, operation: _RollbackOperation, succeeded: bool) -> None:
        if not succeeded:
            operation.failure_phase = operation.phase
        operation.phase = "completed" if succeeded else "failed"
        if succeeded:
            self._rollback_failed.difference_update(operation.affected)
            self._rollback_continuations.update(operation.affected)
        else:
            self._rollback_failed.update(operation.affected)
        self._rollback_service._log("rollback_operation_settled", {
            "phase": operation.phase, "failure_phase": operation.failure_phase,
            "affected_agent_ids": sorted(operation.affected), "epochs": operation.epochs,
        })

    def _pending_rollback_ancestors(self, affected: set[int]) -> set[int]:
        """Include coordinators still awaiting an affected child's result."""
        coordinators: set[int] = set()
        frontier = list(affected)
        while frontier:
            child_aid = frontier.pop()
            origin = self._spawn_origin.get(child_aid) or self._startup_origin.get(
                child_aid
            )
            if origin is None:
                continue
            parent_aid, _tool_call_id = origin
            if parent_aid in affected or parent_aid in coordinators:
                continue
            coordinators.add(parent_aid)
            frontier.append(parent_aid)
        return coordinators

    def resume_after_rollback(self, aids: set[int]) -> None:
        """Explicitly release rollback fences before a caller retries agents."""
        if self._rollback_operation_lock.locked():
            raise RuntimeError("rollback operation is still active")
        unknown = set(aids).difference(self._rollback_affected)
        if unknown:
            raise ValueError(f"agents were not affected by the latest rollback: {sorted(unknown)}")
        failed = set(aids).intersection(self._rollback_failed)
        if failed:
            raise RuntimeError(
                "cannot resume agents after failed restore: "
                f"{sorted(failed)}"
            )
        self._rollback_fenced.difference_update(aids)
        self._rollback_affected.difference_update(aids)
        self._rollback_failed.difference_update(aids)
        active_turn_aids = set(self._active_run_tasks.values())
        self._rollback_interrupted.difference_update(aids - active_turn_aids)
        for aid in aids:
            self._rollback_epochs[aid] = self.current_effect_epoch(aid) + 1
            self._history._epochs[aid] = self._rollback_epochs[aid]
            self._history_anchors[aid] = self._rollback_service.causal_frontier(aid)
            self._rollback_operations.pop(aid, None)
            self._rollback_child_effects.pop(aid, None)

    async def continue_team_turn(
        self,
        aid: int,
        message: str,
        *,
        budget_tokens: int | None = None,
    ) -> str:
        """Start a fresh explicit turn after the interrupted Team turn settled."""
        if self._rollback_fenced:
            raise RuntimeError("resume_after_rollback is required before continuing the Team")
        if self._active_run_tasks:
            raise RuntimeError("the previous Team turn is still active")
        if aid not in self._rollback_continuations:
            raise RuntimeError("no settled rollback is available for continuation")
        self._assert_agent_active(aid)
        if budget_tokens is None:
            raise RuntimeError("budget_exhausted: an explicit continuation budget is required")
        self.reserve_continuation_budget(aid, budget_tokens)
        self._rollback_interrupted.discard(aid)
        token = self._continuation_aid.set(aid)
        try:
            result = await self.run_turn(aid, message)
        except BaseException:
            # The explicit budget is a lease for this attempt. A provider or
            # turn-start failure must release it, while keeping the settled
            # rollback eligible for another explicit attempt.
            self._release_turn_lease(aid)
            raise
        else:
            self._rollback_continuations.discard(aid)
            return result
        finally:
            self._continuation_aid.reset(token)

    def _fence_agents(self, aids: set[int]) -> None:
        self._rollback_fenced.update(aids)
        self._rollback_interrupted.update(aids)
        for aid in aids:
            cancel_event = self._turn_cancel_events.get(aid)
            if cancel_event is not None:
                cancel_event.set()

    async def _reject_rollback_messages(self, aids: set[int]) -> None:
        """Reject queued messages whose sender or recipient is being restored."""
        events = []
        targets = sorted(
            target_aid
            for target_aid, inbox in self._message_inbox.items()
            if any(
                message.from_aid in aids or message.to_aid in aids
                for message in inbox
            )
        )
        for target_aid in targets:
            lock = self._locks.setdefault(target_aid, asyncio.Lock())
            async with lock:
                inbox = self._message_inbox.get(target_aid, [])
                session = self._sessions.get(target_aid)
                retained = []
                for message in inbox:
                    if message.from_aid not in aids and message.to_aid not in aids:
                        retained.append(message)
                        continue
                    detail = "message epoch was invalidated by explicit rollback"
                    if session is not None:
                        self._mark_message_rejected(session.state, message, detail)
                    self._trace_message_decision(
                        "message_refused",
                        reason="rollback_epoch_invalidated",
                        from_aid=message.from_aid,
                        to_aid=message.to_aid,
                        summary=message.summary,
                        content=message.content,
                        message_id=message.message_id or None,
                        restored=True,
                    )
                    events.append(
                        self._events.agent_message_rejected_on_restore(
                            message.from_aid,
                            message.to_aid,
                            detail,
                        )
                    )
                if retained:
                    self._message_inbox[target_aid] = retained
                else:
                    self._message_inbox.pop(target_aid, None)
                if session is not None:
                    self._autosave_session(target_aid)
        for event in events:
            await self._safe_emit_scheduler_event(event)

    async def _quiesce_agents(self, aids: set[int]) -> None:
        for aid in sorted(aids):
            session = self._sessions.get(aid)
            pending = [task for task in getattr(session, "pending_cleanup_tasks", ()) if not task.done()]
            if pending:
                done, outstanding = await asyncio.wait(pending, timeout=10.0)
                if outstanding:
                    raise RuntimeError(f"Agent {aid} session cleanup did not quiesce")
                for task in done:
                    if not task.cancelled() and task.exception() is not None:
                        raise RuntimeError(f"Agent {aid} session cleanup failed") from task.exception()
            environment = getattr(session, "env", None) if session is not None else None
            quiesce = getattr(environment, "quiesce", None)
            if callable(quiesce):
                try:
                    await quiesce()
                except Exception as exc:
                    raise RuntimeError(
                        f"cannot continue rollback for agent {aid}: "
                        "execution environment did not quiesce"
                    ) from exc

    async def _cancel_fenced_tasks(self, aids: set[int]) -> None:
        current_task = asyncio.current_task()
        tasks = {
            task
            for aid in aids
            for task in (self._tasks.get(aid), self._startup_tasks.get(aid), self._message_delivery_tasks.get(aid))
            if task is not None and task is not current_task and not task.done()
        }
        tasks.update(
            task for task, aid in self._active_run_tasks.items()
            if aid in aids and task is not current_task and not task.done()
        )
        for task in tasks:
            task.cancel()
        if tasks:
            _done, pending = await asyncio.wait(tasks, timeout=10.0)
            if pending:
                raise RuntimeError("rollback task cancellation did not quiesce")
            for task in tasks:
                if not task.cancelled():
                    task.exception()

    def _reset_fenced_sessions(self, aids: set[int]) -> None:
        for aid in aids:
            session = self._sessions.get(aid)
            if session is None:
                continue
            if not session.state.pending_events.is_empty():
                raise RuntimeError(
                    f"Agent {aid} still has pending events after rollback settlement"
                )
            session.state.clear_active_turn()
            if session.state.phase.is_terminal():
                session.state.resume_to_idle()
            elif session.state.phase is not SessionPhase.IDLE:
                session.state.set_phase(SessionPhase.IDLE)
            self._turn_cancel_events.pop(aid, None)
            self._release_leases(aid)
            self._delivery_committed.discard(aid)

    async def _settle_rollback_turns(self, aids: set[int]) -> None:
        """Close interrupted turns through the normal pending-row path."""
        for aid in sorted(aids):
            session = self._sessions.get(aid)
            if session is None:
                continue
            state = session.state
            if (
                state.phase is SessionPhase.AWAITING_EVENTS
                or not state.pending_events.is_empty()
            ):
                await self._settle_cancelled_suspended_turn(
                    aid,
                    reason="interrupted by explicit rollback",
                    affected_aids=aids,
                )
            elif aid in self._rollback_interrupted and state.phase.is_terminal():
                # A driver may have caught CancelledError and already moved to
                # STOPPED before the external rollback coroutine resumed. Keep
                # the public run result distinguishable from an ordinary stop.
                state.terminal_reason = "interrupted by explicit rollback"
                state.clear_active_turn()
                session.result = "Error: interrupted by explicit rollback"


__all__ = ["SchedulerRollbackMixin"]
