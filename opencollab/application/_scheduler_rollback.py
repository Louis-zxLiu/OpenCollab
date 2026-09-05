"""Scheduler integration for explicit, dependency-scoped rollback."""

from __future__ import annotations

import asyncio
from typing import Any

from opencollab.application.rollback import RollbackService
from opencollab.domain.rollback import EffectKind, RollbackPlan, RollbackResult
from opencollab.domain.session import SessionPhase


class SchedulerRollbackMixin:
    """Fence affected agents, restore their Scopes, and remain idle."""

    def _init_rollback(self) -> None:
        self._rollback_service = RollbackService(self._tracer)
        self._rollback_fenced: set[int] = set()

    def register_effect_environment(self, aid: int, environment: Any) -> None:
        self._rollback_service.register_environment(aid, environment)

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
        if producer_aid in self._rollback_fenced:
            raise RuntimeError(f"Agent {producer_aid} is fenced from creating effects")
        return self._rollback_service.create_effect(
            producer_aid=producer_aid,
            kind=kind,
            epoch=epoch,
            attempt=attempt,
            parent_effect_ids=parent_effect_ids,
            content=content,
        )

    def consume_effect_ref(self, effect_id: str, consumer_aid: int) -> None:
        if consumer_aid in self._rollback_fenced:
            raise RuntimeError(f"Agent {consumer_aid} is fenced from consuming effects")
        self._rollback_service.register_consumer(effect_id, consumer_aid)

    async def create_checkpoint(self, aid: int, boundary="initial", causal_frontier=frozenset()):
        session = self._sessions.get(aid)
        environment = getattr(session, "env", None) if session is not None else None
        if environment is None:
            raise RuntimeError(f"Agent {aid} has no environment")
        self.register_effect_environment(aid, environment)
        return await self._rollback_service.create_checkpoint(aid, boundary, causal_frontier)

    def preview_rollback(self, effect_ids: set[str]) -> RollbackPlan:
        return self._rollback_service.preview_rollback(effect_ids)

    async def rollback_to_checkpoint(self, aid: int, checkpoint_id: str):
        self._fence_agents({aid})
        await self._cancel_fenced_tasks({aid})
        result = await self._rollback_service.rollback_to_checkpoint(aid, checkpoint_id)
        self._reset_fenced_sessions({aid})
        return result

    async def rollback_effect(self, effect_ids: set[str]) -> RollbackResult:
        plan = self.preview_rollback(effect_ids)
        self._fence_agents(set(plan.affected_agent_ids))
        await self._cancel_fenced_tasks(set(plan.affected_agent_ids))
        result = await self._rollback_service.rollback_effect(effect_ids)
        self._reset_fenced_sessions(set(plan.affected_agent_ids))
        return result

    def resume_after_rollback(self, aids: set[int]) -> None:
        """Explicitly release rollback fences before a caller retries agents."""
        self._rollback_fenced.difference_update(aids)

    def _fence_agents(self, aids: set[int]) -> None:
        self._rollback_fenced.update(aids)
        for aid in aids:
            cancel_event = self._turn_cancel_events.get(aid)
            if cancel_event is not None:
                cancel_event.set()
            session = self._sessions.get(aid)
            if session is not None:
                environment = getattr(session, "env", None)
                if environment is not None:
                    revoke = getattr(environment, "revoke_for_rollback", None)
                    if callable(revoke):
                        revoke()

    async def _cancel_fenced_tasks(self, aids: set[int]) -> None:
        tasks = {
            task
            for aid in aids
            for task in (self._tasks.get(aid), self._startup_tasks.get(aid))
            if task is not None and not task.done()
        }
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _reset_fenced_sessions(self, aids: set[int]) -> None:
        for aid in aids:
            session = self._sessions.get(aid)
            if session is None:
                continue
            session.state.pending_events.clear()
            session.state.clear_active_turn()
            if session.state.phase.is_terminal():
                session.state.resume_to_idle()
            elif session.state.phase is not SessionPhase.IDLE:
                session.state.set_phase(SessionPhase.IDLE)
            self._turn_cancel_events.pop(aid, None)


__all__ = ["SchedulerRollbackMixin"]
