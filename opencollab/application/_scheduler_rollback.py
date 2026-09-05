"""Private Scheduler capability for causal rollback and Scope restore."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any

from opencollab.domain.coordination_protocol import WaitKind
from opencollab.domain.pending import PendingRowError, RowStatus
from opencollab.domain.rollback import CheckpointBoundary
from opencollab.domain.session import SessionPhase

logger = logging.getLogger(__name__)


class SchedulerRollbackMixin:
    async def _capture_effect_workspace(self, effect: Any, producer_aid: int) -> Any:
        """Attach the producer Scope's immutable file revision to an Effect."""
        session = self._sessions.get(producer_aid)
        environment = getattr(session, "env", None) if session is not None else None
        capture = getattr(environment, "collect_workspace_effect", None)
        if not callable(capture):
            capture = getattr(environment, "capture_workspace_revision", None)
        if not callable(capture):
            raise RuntimeError(f"Agent {producer_aid} has no revision-capable isolated Scope")
        try:
            revision = await capture(effect.effect_id, owner_aid=producer_aid)
        except Exception as exc:
            self._trace_rollback(
                "workspace_effect_capture_failed",
                {
                    "effect_id": effect.effect_id,
                    "producer_aid": producer_aid,
                    "workspace": getattr(environment, "workspace", None),
                    "error": str(exc),
                },
            )
            raise
        if not revision.changed:
            return effect
        attached = self._lineage.attach_workspace_revision(effect.effect_id, revision)
        self._trace_rollback(
            "workspace_effect_captured",
            {
                "effect_id": effect.effect_id,
                "producer_aid": producer_aid,
                "revision": revision.revision,
                "base_revision": revision.base_revision,
            },
        )
        return attached

    async def _create_child_result_effect(self, child_aid: int, parent_aid: int, result: str) -> Any:
        if self._lineage is None or not self._rollback_enabled:
            return None
        child_session = self._sessions.get(child_aid)
        if child_session is None:
            return None
        state = child_session.state
        effect = self._lineage.create_effect(
            producer_aid=child_aid,
            attempt=state.rollback.attempt,
            branch_id=state.rollback.branch,
            epoch=state.rollback.epoch,
            kind="child_result",
            parent_effect_ids=tuple(sorted(state.rollback.causal_frontier)),
            content=result or "",
        )
        effect = await self._capture_effect_workspace(effect, child_aid)
        self._lineage.register_consumer(effect.effect_id, parent_aid)
        return effect

    async def _deliver_to_parent(
        self,
        child_aid: int,
        result: str,
        status: RowStatus,
        *,
        error: str | None = None,
    ) -> None:
        """Deliver a child Effect, failing closed without suspending its parent."""
        origin = self._spawn_origin.get(child_aid)
        if origin is None:
            child_scb = self.table.get(child_aid)
            if child_scb is not None and child_scb.parent_aid is not None:
                await self._wake_coordination_wait(child_scb.parent_aid, WaitKind.CHILD)
            return
        parent_aid, tool_call_id = origin
        lineage_effect = None
        if self._lineage is not None and self._rollback_enabled:
            try:
                parent_session = self._sessions.get(parent_aid)
                parent_env = getattr(parent_session, "env", None) if parent_session is not None else None
                if parent_env is None or not callable(getattr(parent_env, "checkpoint_scope", None)):
                    raise RuntimeError(f"Agent {parent_aid} has no checkpointable Scope before child delivery")
                self._lineage.register_environment(parent_aid, parent_env)
                await self._lineage.checkpoint(
                    parent_aid,
                    CheckpointBoundary("child_result"),
                    parent_session.state.rollback.causal_frontier,
                )
                lineage_effect = await self._create_child_result_effect(
                    child_aid,
                    parent_aid,
                    result,
                )
            except Exception as exc:
                detail = str(exc).strip() or type(exc).__name__
                result = f"Error: workspace Effect delivery failed: {detail[:500]}"
                status = RowStatus.FAILED
                error = result
                self._trace_rollback(
                    "workspace_effect_delivery_failed",
                    {
                        "child_aid": child_aid,
                        "parent_aid": parent_aid,
                        "error": detail[:500],
                    },
                )
        try:
            await self._wake(
                parent_aid,
                tool_call_id,
                result,
                status,
                child_aid=child_aid,
                error=error,
                lineage_effect=lineage_effect,
            )
            if lineage_effect is not None:
                await self._mark_effect_delivered(child_aid, lineage_effect.effect_id)
        except PendingRowError as exc:
            retry_tool_call_id = await self._recover_delivery_route(
                child_aid,
                parent_aid,
                exc,
            )
            if retry_tool_call_id is not None:
                try:
                    await self._wake(
                        parent_aid,
                        retry_tool_call_id,
                        result,
                        status,
                        child_aid=child_aid,
                        error=error,
                        lineage_effect=lineage_effect,
                    )
                    if lineage_effect is not None:
                        await self._mark_effect_delivered(child_aid, lineage_effect.effect_id)
                    return
                except PendingRowError as retry_exc:
                    exc = retry_exc
                    await self._recover_delivery_route(child_aid, parent_aid, retry_exc)
            logger.error("misrouted completion from child %s: %s", child_aid, exc)
            await self._safe_emit_scheduler_event(
                self._events.agent_failed(parent_aid, self._role_of(parent_aid), str(exc))
            )

    async def _recover_delivery_route(
        self,
        child_aid: int,
        parent_aid: int,
        cause: PendingRowError,
    ) -> str | None:
        """Route a unique child row or fail every producer-less pending row."""
        parent_scb = self.table.get(parent_aid)
        parent_session = self._sessions.get(parent_aid)
        if parent_scb is None or parent_session is None:
            self._spawn_origin.pop(child_aid, None)
            return None

        lock = self._locks.setdefault(parent_aid, asyncio.Lock())
        should_resume = False
        async with lock:
            table = parent_scb.state.pending_events
            child_rows = [
                tool_call_id
                for tool_call_id, row in table.rows.items()
                if row.status is RowStatus.PENDING and row.ref == child_aid
            ]
            if len(child_rows) == 1:
                return child_rows[0]

            reason = f"Error: child completion routing failed: {cause}"
            for tool_call_id, row in tuple(table.rows.items()):
                if row.status is RowStatus.PENDING:
                    table.fill(
                        tool_call_id,
                        result=reason,
                        status=RowStatus.FAILED,
                        error=reason,
                    )
            self._spawn_origin.pop(child_aid, None)
            in_flight = self._tasks.get(parent_aid)
            should_resume = (
                not self._shutting_down
                and parent_scb.state.phase is SessionPhase.AWAITING_EVENTS
                and table.is_complete()
                and (in_flight is None or in_flight.done())
            )
            if should_resume:
                self._reserve_turn_lease(parent_aid)
                self._start_agent_task(parent_aid, parent_session)
            elif (
                not self._shutting_down
                and parent_scb.state.phase is SessionPhase.AWAITING_EVENTS
                and table.is_complete()
                and in_flight is not None
                and not in_flight.done()
            ):
                in_flight.add_done_callback(
                    lambda finished: asyncio.create_task(
                        self._resume_after_parent_task(
                            parent_aid,
                            parent_session,
                            finished,
                        )
                    )
                )

        if should_resume:
            await self._safe_emit_scheduler_event(self._events.agent_resumed(parent_aid, self._role_of(parent_aid)))
        return None

    async def adopt_effect(self, consumer_aid: int, effect_id: str) -> str:
        """Adopt one visible Effect revision into the caller's isolated Scope."""
        protocol = getattr(self, "_coordination_protocol", None)
        if protocol is not None:
            try:
                protocol.require_addressable(consumer_aid)
            except ValueError as exc:
                return f"Error: {exc}"
        if self._lineage is None or not self._rollback_enabled:
            return "Error: workspace Effect adoption requires rollback-enabled Team mode."
        effect = self._lineage.get_effect(effect_id)
        if effect is None:
            return "Error: unknown effect_id."
        if effect.status == "quarantined":
            return "Error: quarantined Effects cannot be adopted."
        if not self._lineage.consumer_can_access(effect_id, consumer_aid):
            return "Error: this Agent has not received the requested Effect."
        revision = self._lineage.workspace_revision(effect_id)
        if revision is None:
            producer = self._sessions.get(effect.producer_aid)
            producer_env = getattr(producer, "env", None) if producer is not None else None
            acknowledge = getattr(producer_env, "acknowledge_workspace_effect", None)
            if callable(acknowledge):
                try:
                    await acknowledge(effect_id)
                except Exception as exc:
                    self._trace_rollback(
                        "workspace_effect_acknowledge_failed",
                        {
                            "effect_id": effect_id,
                            "producer_aid": effect.producer_aid,
                            "workspace": getattr(producer_env, "workspace", None),
                            "error": str(exc),
                        },
                    )
                    return f"Error: workspace Effect acknowledgement failed: {exc}"
            self._adopt_causal_effect(consumer_aid, effect_id)
            return f"Effect {effect_id} has no workspace changes to adopt."
        session = self._sessions.get(consumer_aid)
        environment = getattr(session, "env", None) if session is not None else None
        adopt = getattr(environment, "adopt_workspace_revision", None)
        if not callable(adopt):
            return "Error: this Agent Scope cannot adopt workspace revisions."
        outcome = await adopt(revision)
        if outcome.status != "adopted":
            reason = outcome.reason or outcome.status
            self._trace_rollback(
                "workspace_effect_adoption_failed",
                {
                    "effect_id": effect_id,
                    "producer_aid": effect.producer_aid,
                    "consumer_aid": consumer_aid,
                    "workspace": getattr(environment, "workspace", None),
                    "reason": reason,
                },
            )
            return f"Error: Effect adoption {outcome.status}: {reason}"
        producer = self._sessions.get(effect.producer_aid)
        producer_env = getattr(producer, "env", None) if producer is not None else None
        acknowledge = getattr(producer_env, "acknowledge_workspace_effect", None)
        if callable(acknowledge):
            try:
                await acknowledge(effect_id)
            except Exception as exc:
                self._trace_rollback(
                    "workspace_effect_acknowledge_failed",
                    {
                        "effect_id": effect_id,
                        "producer_aid": effect.producer_aid,
                        "workspace": getattr(producer_env, "workspace", None),
                        "error": str(exc),
                    },
                )
                return f"Error: workspace Effect acknowledgement failed: {exc}"
        self._adopt_causal_effect(consumer_aid, effect_id)
        self._trace_rollback(
            "workspace_effect_adopted",
            {
                "effect_id": effect_id,
                "producer_aid": effect.producer_aid,
                "consumer_aid": consumer_aid,
                "revision": revision.revision,
            },
        )
        return f"Adopted Effect {effect_id} into this Agent Scope."

    def _adopt_causal_effect(self, consumer_aid: int, effect_id: str) -> None:
        session = self._sessions.get(consumer_aid)
        state = getattr(session, "state", None) if session is not None else None
        if state is not None:
            state.adopt_visible_effect(effect_id)
        if self._lineage is not None:
            frontier = self._lineage.consume(consumer_aid, effect_id)
            if state is not None:
                state.rollback = dataclasses.replace(
                    state.rollback,
                    causal_frontier=frontier,
                )

    async def publish_effect(self, consumer_aid: int, effect_id: str) -> str:
        """Publish one visible Effect revision to the source workspace."""
        if self._lineage is None or not self._rollback_enabled:
            return "Error: workspace Effect publishing requires rollback-enabled Team mode."
        effect = self._lineage.get_effect(effect_id)
        if effect is None:
            return "Error: unknown effect_id."
        if effect.status == "quarantined":
            return "Error: quarantined Effects cannot be published."
        if not self._lineage.consumer_can_access(effect_id, consumer_aid):
            return "Error: this Agent has not received the requested Effect."
        revision = self._lineage.workspace_revision(effect_id)
        if revision is None:
            return f"Effect {effect_id} has no workspace changes to publish."
        lead = self._sessions.get(0)
        environment = getattr(lead, "env", None) if lead is not None else None
        publish = getattr(environment, "publish_workspace_revision", None)
        if not callable(publish):
            return "Error: source workspace cannot publish workspace revisions."
        outcome = await publish(revision)
        if outcome.status not in {"adopted", "skipped"}:
            reason = outcome.reason or outcome.status
            self._trace_rollback(
                "workspace_effect_publish_failed",
                {
                    "effect_id": effect_id,
                    "producer_aid": effect.producer_aid,
                    "consumer_aid": consumer_aid,
                    "workspace": getattr(environment, "source_workspace", None)
                    or getattr(environment, "workspace", None),
                    "reason": reason,
                },
            )
            return f"Error: Effect publish {outcome.status}: {reason}"
        acknowledged = await self._acknowledge_workspace_effect(effect)
        if acknowledged is not None:
            return acknowledged
        self._trace_rollback(
            "workspace_effect_published",
            {
                "effect_id": effect_id,
                "producer_aid": effect.producer_aid,
                "consumer_aid": consumer_aid,
                "revision": revision.revision,
            },
        )
        return f"Published Effect {effect_id} to the source workspace."

    async def _acknowledge_workspace_effect(self, effect: Any) -> str | None:
        producer = self._sessions.get(effect.producer_aid)
        producer_env = getattr(producer, "env", None) if producer is not None else None
        acknowledge = getattr(producer_env, "acknowledge_workspace_effect", None)
        if not callable(acknowledge):
            return None
        try:
            await acknowledge(effect.effect_id)
        except Exception as exc:
            self._trace_rollback(
                "workspace_effect_acknowledge_failed",
                {
                    "effect_id": effect.effect_id,
                    "producer_aid": effect.producer_aid,
                    "workspace": getattr(producer_env, "workspace", None),
                    "error": str(exc),
                },
            )
            return f"Error: workspace Effect acknowledgement failed: {exc}"
        return None

    async def _mark_effect_delivered(self, producer_aid: int, effect_id: str) -> None:
        session = self._sessions.get(producer_aid)
        environment = getattr(session, "env", None) if session is not None else None
        marker = getattr(environment, "mark_workspace_effect_delivered", None)
        if callable(marker):
            await marker(effect_id)

    async def _checkpoint_after_spawn(self, aid: int, task: str) -> None:
        if self._lineage is None or not self._rollback_enabled:
            return
        env = self._startup_envs.get(aid)
        required = (
            "checkpoint_scope",
            "restore_scope",
            "capture_workspace_revision",
            "adopt_workspace_revision",
        )
        if env is None or any(not callable(getattr(env, name, None)) for name in required):
            raise RuntimeError(f"Agent {aid} has no checkpointable revision-capable Scope; startup aborted")
        try:
            self._lineage.register_environment(aid, env)
            checkpoint = await self._lineage.checkpoint(
                aid,
                CheckpointBoundary("scope_initialization"),
                self._sessions[aid].state.rollback.causal_frontier,
            )
            self._trace_rollback(
                "checkpoint_created",
                {"aid": aid, "checkpoint_id": checkpoint.checkpoint_id, "label": task[:20]},
            )
        except Exception as exc:
            self._trace_rollback("checkpoint_failed", {"aid": aid, "error": str(exc)})
            raise RuntimeError(f"checkpoint failed for Agent {aid}; execution was not started: {exc}") from exc

    async def _handle_invalidation(
        self, effect_id: str, reporter_aid: int, reason: str, evidence: str = ""
    ) -> set[str]:
        # Causal invalidation is a rollback-enabled capability.  In ordinary
        # teams the tool must be a no-op so legacy sessions keep their normal
        # execution semantics and are never quarantined without restore.
        if self._lineage is None or not self._rollback_enabled:
            return set()
        affected = self._lineage.quarantine(effect_id, reason, evidence or f"reported by agent {reporter_aid}")
        self._trace_rollback(
            "effects_quarantined",
            {"effect_id": effect_id, "reporter_aid": reporter_aid, "count": len(affected)},
        )
        affected_consumers = self._lineage.compute_affected_agents(affected)
        producer_agents = {
            effect.producer_aid
            for current in affected
            if (effect := self._lineage.get_effect(current)) is not None
        }
        affected_agents = affected_consumers | producer_agents
        target_aid = self._coordination_protocol.nearest_active_coordinator(
            affected_consumers
        )
        if target_aid is None and self._coordination_protocol.is_addressable(reporter_aid):
            target_aid = reporter_aid
        for aid in sorted(affected_agents):
            if aid != target_aid:
                await self._supersede_agent(aid, effect_id)
        for aid in affected_agents:
            if aid != target_aid:
                continue
            self._rollback_pending.add(aid)
            self._rollback_barriers.setdefault(aid, asyncio.Event()).clear()
        for aid in affected_agents:
            # Affected agents may already be executing a tool batch when the
            # invalidation arrives. Let that batch append its complete tool
            # result block before restoring its epoch; otherwise the provider
            # would see an assistant tool call without a matching tool output.
            if self._agent_is_executing_tools(aid):
                self._deferred_tool_rollbacks.setdefault(aid, set()).update(affected)
                continue
            await self._rollback_agent(aid, affected)
        return affected

    async def _supersede_agent(self, aid: int, root_effect_id: str) -> None:
        """Fence an invalidated producer so its old aid cannot compete."""
        session = self._sessions.get(aid)
        if session is None:
            return
        self._coordination_protocol.supersede(aid)
        self._rollback_pending.discard(aid)
        session.state.wait_condition = None
        session.state.pending_events.clear()
        session.state.clear_active_turn()
        if not session.state.phase.is_terminal():
            session.state.cancel(
                f"agent aid {aid} superseded by rollback of effect {root_effect_id}"
            )
        self._message_inbox.pop(aid, None)
        self._release_leases(aid)
        task = self._tasks.get(aid)
        current = asyncio.current_task()
        if task is not None and not task.done() and task is not current:
            task.cancel()
        startup = self._startup_tasks.get(aid)
        if startup is not None and not startup.done() and startup is not current:
            startup.cancel()
        origin = self._spawn_origin.pop(aid, None) or self._startup_origin.pop(aid, None)
        if origin is not None:
            await self._fail_cancelled_origin(
                origin[0],
                origin[1],
                f"agent aid {aid} was superseded by rollback of effect {root_effect_id}",
            )
        self._trace_rollback(
            "agent_superseded",
            {"aid": aid, "root_effect_id": root_effect_id},
        )

    def _agent_is_executing_tools(self, aid: int) -> bool:
        session = self._sessions.get(aid)
        state = getattr(session, "state", None) if session is not None else None
        return getattr(state, "phase", None) is SessionPhase.EXECUTING_TOOLS

    async def _flush_deferred_tool_rollback(self, aid: int) -> None:
        invalidated = self._deferred_tool_rollbacks.pop(aid, None)
        if not invalidated:
            return
        await self._rollback_agent(aid, invalidated)
        # The rollback changes epoch/frontier after the normal step-end save
        # was queued. Persist the restored control-plane state as well.
        autosave = getattr(self, "_autosave_session", None)
        if callable(autosave):
            owner = autosave(aid)
            if owner is not None:
                await owner

    async def _rollback_agent(self, aid: int, quarantined_effects: set[str]) -> None:
        if self._lineage is None or not self._rollback_enabled or self._sessions.get(aid) is None:
            self._trace_rollback("rollback_skipped", {"aid": aid, "reason": "agent unavailable"})
            return
        # Environment-native restore acquires the Scope command lock.  This is
        # the quiescence boundary: an in-flight process is allowed to finish,
        # and restore cannot begin until its command owner releases the lock.
        target = self._lineage.select_checkpoint(aid, quarantined_effects)
        if target is None:
            self._trace_rollback("rollback_skipped", {"aid": aid, "reason": "no relevant checkpoint"})
            self._rollback_pending.discard(aid)
            self._rollback_barriers.setdefault(aid, asyncio.Event()).set()
            return
        result = await self._lineage.restore_agent(aid, quarantined_effects)
        if result is None or result.status != "restored":
            self._trace_rollback(
                "rollback_failed",
                {
                    "aid": aid,
                    "checkpoint_id": target.checkpoint_id,
                    "error": getattr(result, "reason", None) or "unknown",
                },
            )
            self._rollback_pending.discard(aid)
            self._rollback_barriers.setdefault(aid, asyncio.Event()).set()
            return
        state = self._sessions[aid].state
        had_active_turn = state.active_turn_start_message_index is not None
        runner = getattr(self._sessions[aid], "runner", None)
        reset_runtime = getattr(runner, "reset_for_restore", None)
        if callable(reset_runtime):
            reset_runtime()
        if not state.pending_events.is_empty():
            state.pending_events.clear()
        state.clear_active_turn()
        state.rollback = dataclasses.replace(
            state.rollback,
            epoch=state.rollback.epoch + 1,
            attempt=state.rollback.attempt + 1,
            causal_frontier=target.causal_frontier,
            quarantined_effects=state.rollback.quarantined_effects | frozenset(quarantined_effects),
        )
        self._trace_rollback(
            "agent_rolled_back",
            {
                "aid": aid,
                "checkpoint_id": target.checkpoint_id,
                "new_epoch": state.rollback.epoch,
                "files_changed": result.files_changed,
            },
        )
        self._rollback_pending.discard(aid)
        self._rollback_barriers.setdefault(aid, asyncio.Event()).set()
        # The same coordinating Session remains runnable.  A restored Agent is
        # resumed only when it had a live/deferred turn; otherwise its next
        # normal scheduler wake starts the corrected attempt.
        if aid in self._sessions and state.phase is not SessionPhase.DONE:
            state.set_phase(SessionPhase.IDLE)
            current = self._tasks.get(aid)
            has_turn = had_active_turn or state.pending_external_user_turn is not None
            if has_turn and (current is None or current.done()) and not self._shutting_down:
                self._reserve_turn_lease(aid)
                self._start_agent_task(aid, self._sessions[aid])

    def _trace_rollback(self, step_type: str, payload: dict[str, Any]) -> None:
        if self._tracer is not None:
            self._tracer.log_step(step_type=step_type, payload=payload)
