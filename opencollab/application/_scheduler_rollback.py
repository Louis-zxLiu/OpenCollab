"""Private Scheduler capability for causal rollback and Scope restore."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

from opencollab.domain.rollback import CheckpointBoundary
from opencollab.domain.session import SessionPhase


class SchedulerRollbackMixin:
    def _create_child_result_effect(self, child_aid: int, parent_aid: int, result: str) -> Any:
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
        self._lineage.register_consumer(effect.effect_id, parent_aid)
        return effect

    async def _checkpoint_after_spawn(self, aid: int, task: str) -> None:
        if self._lineage is None or not self._rollback_enabled:
            return
        env = self._startup_envs.get(aid)
        if env is None or not callable(getattr(env, "checkpoint_scope", None)):
            raise RuntimeError(
                f"Agent {aid} has no checkpointable isolated Scope; startup aborted"
            )
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
            raise RuntimeError(
                f"checkpoint failed for Agent {aid}; execution was not started: {exc}"
            ) from exc

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
        affected_agents = self._lineage.compute_affected_agents(affected)
        for aid in affected_agents:
            self._rollback_pending.add(aid)
            self._rollback_barriers.setdefault(aid, asyncio.Event()).clear()
        for aid in affected_agents:
            await self._rollback_agent(aid, affected)
        return affected

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
            has_turn = (
                had_active_turn
                or state.pending_external_user_turn is not None
            )
            if has_turn and (current is None or current.done()) and not self._shutting_down:
                self._reserve_turn_lease(aid)
                self._start_agent_task(aid, self._sessions[aid])

    def _trace_rollback(self, step_type: str, payload: dict[str, Any]) -> None:
        if self._tracer is not None:
            self._tracer.log_step(step_type=step_type, payload=payload)
