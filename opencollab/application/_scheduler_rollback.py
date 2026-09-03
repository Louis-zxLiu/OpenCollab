"""Private Scheduler capability for causal rollback and Scope restore."""

from __future__ import annotations

import dataclasses
from typing import Any

from opencollab.domain.rollback import CheckpointBoundary


class SchedulerRollbackMixin:
    def _create_child_result_effect(self, child_aid: int, parent_aid: int, result: str) -> Any:
        if self._lineage is None:
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
        if self._lineage is None:
            return
        env = self._startup_envs.get(aid)
        if env is None or not hasattr(env, "checkpoint_scope"):
            return
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

    async def _handle_invalidation(
        self, effect_id: str, reporter_aid: int, reason: str, evidence: str = ""
    ) -> set[str]:
        if self._lineage is None:
            return set()
        affected = self._lineage.quarantine(effect_id, reason, evidence or f"reported by agent {reporter_aid}")
        self._trace_rollback(
            "effects_quarantined",
            {"effect_id": effect_id, "reporter_aid": reporter_aid, "count": len(affected)},
        )
        for aid in self._lineage.compute_affected_agents(affected):
            await self._rollback_agent(aid, affected)
        return affected

    async def _rollback_agent(self, aid: int, quarantined_effects: set[str]) -> None:
        if self._lineage is None or self._sessions.get(aid) is None:
            self._trace_rollback("rollback_skipped", {"aid": aid, "reason": "agent unavailable"})
            return
        target = self._lineage.select_checkpoint(aid, quarantined_effects)
        if target is None:
            self._trace_rollback("rollback_skipped", {"aid": aid, "reason": "no relevant checkpoint"})
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
            return
        state = self._sessions[aid].state
        state.rollback = dataclasses.replace(
            state.rollback,
            epoch=state.rollback.epoch + 1,
            attempt=state.rollback.attempt + 1,
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

    def _trace_rollback(self, step_type: str, payload: dict[str, Any]) -> None:
        if self._tracer is not None:
            self._tracer.log_step(step_type=step_type, payload=payload)
