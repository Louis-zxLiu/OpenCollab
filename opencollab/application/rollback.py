"""Explicit rollback use case over causal effects and Scope checkpoints."""

from __future__ import annotations

import secrets
from dataclasses import replace
from typing import Any, Mapping

from opencollab.application.ports import CheckpointableEnvironmentPort, TracePort
from opencollab.domain.rollback import (
    CheckpointBoundary,
    EffectKind,
    EffectRef,
    RollbackPlan,
    RollbackResult,
    ScopeCheckpoint,
    compute_affected_agents,
    compute_descendants,
    digest_content,
    select_checkpoint,
)


class RollbackService:
    """Build plans and execute explicit Scope restores without auto-restart."""

    def __init__(self, trace: TracePort | None = None) -> None:
        self._trace = trace
        self._effects: dict[str, EffectRef] = {}
        self._consumers: dict[str, set[int]] = {}
        self._checkpoints: dict[int, list[ScopeCheckpoint]] = {}
        self._environments: dict[int, CheckpointableEnvironmentPort] = {}

    @property
    def effects(self) -> Mapping[str, EffectRef]:
        return self._effects

    def register_environment(self, aid: int, environment: CheckpointableEnvironmentPort) -> None:
        self._environments[aid] = environment

    def create_effect(
        self,
        *,
        effect_id: str | None = None,
        producer_aid: int,
        kind: EffectKind,
        epoch: int,
        attempt: int,
        parent_effect_ids: tuple[str, ...] = (),
        content: str = "",
    ) -> EffectRef:
        effect = EffectRef(
            effect_id=effect_id or f"effect-{secrets.token_hex(12)}",
            producer_aid=producer_aid,
            kind=kind,
            epoch=epoch,
            attempt=attempt,
            parent_effect_ids=parent_effect_ids,
            content_digest=digest_content(content),
        )
        unknown = set(parent_effect_ids).difference(self._effects)
        if unknown:
            raise ValueError(f"unknown parent effects: {sorted(unknown)}")
        if effect.effect_id in self._effects:
            raise ValueError(f"effect already exists: {effect.effect_id}")
        self._effects[effect.effect_id] = effect
        self._log("effect_created", {"effect_id": effect.effect_id, "producer_aid": producer_aid})
        return effect

    def register_consumer(self, effect_id: str, consumer_aid: int) -> None:
        if effect_id not in self._effects:
            raise ValueError(f"unknown effect: {effect_id}")
        self._consumers.setdefault(effect_id, set()).add(consumer_aid)

    async def create_checkpoint(
        self,
        aid: int,
        boundary: CheckpointBoundary = "initial",
        causal_frontier: frozenset[str] = frozenset(),
    ) -> ScopeCheckpoint:
        environment = self._require_environment(aid)
        checkpoint = await environment.checkpoint_scope(
            boundary,
            owner_aid=aid,
            causal_frontier=causal_frontier,
        )
        self._checkpoints.setdefault(aid, []).append(checkpoint)
        self._log("checkpoint_created", {"aid": aid, "checkpoint_id": checkpoint.checkpoint_id})
        return checkpoint

    def preview_rollback(self, effect_ids: set[str]) -> RollbackPlan:
        targets = frozenset(effect_ids)
        if not targets:
            raise ValueError("at least one target effect is required")
        unknown = targets.difference(self._effects)
        if unknown:
            raise ValueError(f"unknown effects: {sorted(unknown)}")
        invalidated = compute_descendants(self._effects, set(targets))
        affected = compute_affected_agents(self._effects, self._consumers, set(invalidated))
        checkpoint_map = {
            aid: select_checkpoint(
                {key: tuple(value) for key, value in self._checkpoints.items()},
                aid,
                set(invalidated),
            )
            for aid in affected
        }
        return RollbackPlan(
            target_effect_ids=targets,
            invalidated_effect_ids=invalidated,
            affected_agent_ids=affected,
            checkpoint_by_agent=checkpoint_map,
        )

    async def rollback_to_checkpoint(self, aid: int, checkpoint_id: str) -> Any:
        checkpoint = next(
            (item for item in self._checkpoints.get(aid, ()) if item.checkpoint_id == checkpoint_id),
            None,
        )
        if checkpoint is None:
            raise ValueError(f"unknown checkpoint {checkpoint_id!r} for agent {aid}")
        result = await self._require_environment(aid).restore_scope(checkpoint)
        self._log(
            "checkpoint_restored",
            {"aid": aid, "checkpoint_id": checkpoint_id, "status": result.status},
        )
        return result

    async def rollback_effect(self, effect_ids: set[str]) -> RollbackResult:
        plan = self.preview_rollback(effect_ids)
        restores = []
        for aid in sorted(plan.affected_agent_ids):
            checkpoint = plan.checkpoint_by_agent[aid]
            if checkpoint is None:
                restores.append(
                    self._skipped_restore(aid, "no uncontaminated checkpoint is available")
                )
                continue
            restores.append(await self._require_environment(aid).restore_scope(checkpoint))
        for effect_id in plan.invalidated_effect_ids:
            self._effects[effect_id] = replace(self._effects[effect_id], status="invalidated")
        self._log(
            "rollback_completed",
            {
                "target_count": len(plan.target_effect_ids),
                "invalidated_count": len(plan.invalidated_effect_ids),
                "affected_count": len(plan.affected_agent_ids),
            },
        )
        return RollbackResult(plan, tuple(restores), invalidated=True)

    def _require_environment(self, aid: int) -> CheckpointableEnvironmentPort:
        try:
            return self._environments[aid]
        except KeyError as exc:
            raise RuntimeError(f"Agent {aid} has no checkpointable Scope") from exc

    @staticmethod
    def _skipped_restore(aid: int, reason: str):
        from opencollab.domain.rollback import RestoreResult

        return RestoreResult(aid, None, "skipped", reason=reason)

    def _log(self, step_type: str, payload: Mapping[str, Any]) -> None:
        if self._trace is not None:
            self._trace.log_step(step_type=step_type, payload=dict(payload))


__all__ = ["RollbackService"]
