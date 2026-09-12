"""Explicit rollback use case over causal effects and Scope checkpoints."""

from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import contextmanager
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Mapping

from opencollab.application.ports import CheckpointableEnvironmentPort, TracePort
from opencollab.domain.rollback import (
    CheckpointBoundary,
    EffectKind,
    EffectRef,
    RestoreResult,
    RollbackPlan,
    RollbackResult,
    ScopeCheckpoint,
    compute_affected_agents,
    compute_descendants,
    digest_content,
    select_checkpoint,
)

logger = logging.getLogger(__name__)


class RollbackService:
    """Build plans and execute explicit Scope restores without auto-restart."""

    def __init__(self, trace: TracePort | None = None) -> None:
        self._trace = trace
        self._effects: dict[str, EffectRef] = {}
        self._consumers: dict[str, set[int]] = {}
        self._checkpoints: dict[int, list[ScopeCheckpoint]] = {}
        self._environments: dict[int, CheckpointableEnvironmentPort] = {}
        self._causal_frontiers: dict[int, set[str]] = {}
        self._rollback_lock = asyncio.Lock()

    @property
    def effects(self) -> Mapping[str, EffectRef]:
        return MappingProxyType(self._effects)

    def register_environment(self, aid: int, environment: CheckpointableEnvironmentPort) -> None:
        self._environments[aid] = environment

    @contextmanager
    def effect_transaction(self):
        """Commit synchronous graph mutations together; never hold across an await."""
        effects = self._effects.copy()
        consumers = {key: value.copy() for key, value in self._consumers.items()}
        frontiers = {key: value.copy() for key, value in self._causal_frontiers.items()}
        try:
            yield
        except BaseException:
            self._effects = effects
            self._consumers = consumers
            self._causal_frontiers = frontiers
            raise

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
        if not parent_effect_ids:
            parent_effect_ids = tuple(sorted(self._causal_frontiers.get(producer_aid, set())))
        if len(set(parent_effect_ids)) != len(parent_effect_ids):
            raise ValueError("parent effect IDs must be unique")
        parent_effect_ids = tuple(sorted(parent_effect_ids))
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
        invalidated_parents = {
            parent
            for parent in parent_effect_ids
            if self._effects[parent].status == "invalidated"
        }
        if invalidated_parents:
            raise ValueError(
                "an effect cannot depend on invalidated parents: "
                f"{sorted(invalidated_parents)}"
            )
        if effect.effect_id in self._effects:
            raise ValueError(f"effect already exists: {effect.effect_id}")
        self._assert_acyclic(effect)
        self._effects[effect.effect_id] = effect
        self._causal_frontiers.setdefault(producer_aid, set()).add(effect.effect_id)
        self._log("effect_created", {"effect_id": effect.effect_id, "producer_aid": producer_aid})
        return effect

    def register_consumer(self, effect_id: str, consumer_aid: int) -> None:
        if effect_id not in self._effects:
            raise ValueError(f"unknown effect: {effect_id}")
        if self._effects[effect_id].status != "active":
            raise ValueError(f"cannot consume invalidated effect: {effect_id}")
        self._consumers.setdefault(effect_id, set()).add(consumer_aid)
        self._causal_frontiers.setdefault(consumer_aid, set()).add(effect_id)

    def causal_frontier(self, aid: int) -> frozenset[str]:
        """Return the effects produced or explicitly consumed by one Agent."""
        return frozenset(self._causal_frontiers.get(aid, set()))

    def get_checkpoint(self, aid: int, checkpoint_id: str) -> ScopeCheckpoint:
        checkpoint = next(
            (item for item in self._checkpoints.get(aid, ()) if item.checkpoint_id == checkpoint_id),
            None,
        )
        if checkpoint is None:
            raise ValueError(f"unknown checkpoint {checkpoint_id!r} for agent {aid}")
        return checkpoint

    async def validate_checkpoint(self, aid: int, checkpoint_id: str) -> ScopeCheckpoint:
        checkpoint = self.get_checkpoint(aid, checkpoint_id)
        if any(
            effect_id in self._effects and self._effects[effect_id].status == "invalidated"
            for effect_id in checkpoint.causal_frontier
        ):
            raise ValueError("checkpoint contains invalidated effects")
        environment = self._require_environment(aid)
        if environment.workspace != checkpoint.workspace_identity:
            raise ValueError(f"workspace identity changed for agent {aid}")
        await environment.validate_checkpoint_scope(checkpoint)
        return checkpoint

    async def validate_plan(self, plan: RollbackPlan) -> None:
        """Run all checks that must succeed before Scheduler fencing."""
        await self._preflight_plan(plan)

    async def create_checkpoint(
        self,
        aid: int,
        boundary: CheckpointBoundary = "initial",
        causal_frontier: frozenset[str] | None = None,
    ) -> ScopeCheckpoint:
        environment = self._require_environment(aid)
        if causal_frontier is None:
            causal_frontier = self.causal_frontier(aid)
        checkpoint = await environment.checkpoint_scope(
            boundary,
            owner_aid=aid,
            causal_frontier=causal_frontier,
        )
        self._checkpoints.setdefault(aid, []).append(checkpoint)
        self._log("checkpoint_created", {"aid": aid, "checkpoint_id": checkpoint.checkpoint_id})
        return checkpoint

    def preview_rollback(
        self,
        effect_ids: set[str],
        *,
        coordinator_agent_ids: frozenset[int] = frozenset(),
    ) -> RollbackPlan:
        targets = frozenset(effect_ids)
        if not targets:
            raise ValueError("at least one target effect is required")
        unknown = targets.difference(self._effects)
        if unknown:
            raise ValueError(f"unknown effects: {sorted(unknown)}")
        invalidated_targets = {
            effect_id
            for effect_id in targets
            if self._effects[effect_id].status == "invalidated"
        }
        if invalidated_targets:
            raise ValueError(
                "cannot roll back an already invalidated effect: "
                f"{sorted(invalidated_targets)}"
            )
        invalidated = compute_descendants(self._effects, set(targets))
        historical_invalidated = {
            effect_id
            for effect_id, effect in self._effects.items()
            if effect.status == "invalidated"
        }
        affected = compute_affected_agents(self._effects, self._consumers, set(invalidated))
        affected = frozenset((*affected, *coordinator_agent_ids))
        checkpoint_map = {
            aid: select_checkpoint(
                self._checkpoints,
                aid,
                set(invalidated),
                historical_invalidated,
            )
            for aid in affected
        }
        return RollbackPlan(
            target_effect_ids=targets,
            invalidated_effect_ids=invalidated,
            affected_agent_ids=affected,
            checkpoint_by_agent=checkpoint_map,
            coordinator_agent_ids=coordinator_agent_ids,
            effect_fingerprints={
                effect_id: self._effect_fingerprint(self._effects[effect_id])
                for effect_id in sorted(invalidated)
            },
        )

    async def rollback_to_checkpoint(self, aid: int, checkpoint_id: str) -> RestoreResult:
        checkpoint = await self.validate_checkpoint(aid, checkpoint_id)
        try:
            result = await self._require_environment(aid).restore_scope(checkpoint)
        except (OSError, RuntimeError, ValueError) as exc:
            result = self._failed_restore(aid, checkpoint_id, exc)
        if result.status == "restored":
            self._causal_frontiers[aid] = set(checkpoint.causal_frontier)
        self._log(
            "checkpoint_restored",
            {"aid": aid, "checkpoint_id": checkpoint_id, "status": result.status},
        )
        return result

    async def rollback_effect(
        self,
        effect_ids: set[str],
        *,
        expected_plan_digest: str,
        coordinator_agent_ids: frozenset[int] = frozenset(),
    ) -> RollbackResult:
        async with self._rollback_lock:
            plan = self.preview_rollback(
                effect_ids,
                coordinator_agent_ids=coordinator_agent_ids,
            )
            if expected_plan_digest != plan.digest():
                raise ValueError("rollback plan digest does not match the current plan")
            await self._preflight_plan(plan)
            restores = []
            for aid in sorted(plan.affected_agent_ids):
                checkpoint = plan.checkpoint_by_agent[aid]
                assert checkpoint is not None
                try:
                    restore = await self._require_environment(aid).restore_scope(
                        checkpoint
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    restore = self._failed_restore(aid, checkpoint.checkpoint_id, exc)
                restores.append(restore)
            if any(restore.status != "restored" for restore in restores):
                self._log(
                    "rollback_failed",
                    {
                        "target_count": len(plan.target_effect_ids),
                        "affected_count": len(plan.affected_agent_ids),
                    },
                )
                return RollbackResult(plan, tuple(restores), invalidated=False)
            with self.effect_transaction():
                for aid, checkpoint in plan.checkpoint_by_agent.items():
                    assert checkpoint is not None
                    self._causal_frontiers[aid] = set(checkpoint.causal_frontier)
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

    def _assert_acyclic(self, candidate: EffectRef) -> None:
        """Reject a malformed graph even when a caller supplies an explicit ID."""
        visiting: set[str] = set()
        visited: set[str] = set()
        stack = [(candidate.effect_id, False)]
        while stack:
            effect_id, exiting = stack.pop()
            if exiting:
                visiting.remove(effect_id)
                visited.add(effect_id)
                continue
            if effect_id in visiting:
                raise ValueError("effect graph contains a cycle")
            if effect_id in visited:
                continue
            effect = candidate if effect_id == candidate.effect_id else self._effects.get(effect_id)
            if effect is None:
                continue
            visiting.add(effect_id)
            stack.append((effect_id, True))
            stack.extend((parent, False) for parent in reversed(effect.parent_effect_ids))

    async def _preflight_plan(self, plan: RollbackPlan) -> None:
        missing = [aid for aid, checkpoint in plan.checkpoint_by_agent.items() if checkpoint is None]
        if missing:
            raise ValueError(
                "rollback has no uncontaminated checkpoint for agents: "
                f"{sorted(missing)}"
            )
        for aid, checkpoint in plan.checkpoint_by_agent.items():
            assert checkpoint is not None
            await self.validate_checkpoint(aid, checkpoint.checkpoint_id)

    @staticmethod
    def _effect_fingerprint(effect: EffectRef) -> str:
        return ":".join(
            (
                effect.effect_id,
                str(effect.producer_aid),
                effect.kind,
                str(effect.epoch),
                str(effect.attempt),
                ",".join(effect.parent_effect_ids),
                effect.content_digest,
                effect.status,
            )
        )

    def _require_environment(self, aid: int) -> CheckpointableEnvironmentPort:
        try:
            return self._environments[aid]
        except KeyError as exc:
            raise RuntimeError(f"Agent {aid} has no checkpointable Scope") from exc

    @staticmethod
    def _failed_restore(
        aid: int,
        checkpoint_id: str,
        error: BaseException,
    ) -> RestoreResult:
        reason = str(error).strip() or type(error).__name__
        return RestoreResult(
            aid,
            checkpoint_id,
            "failed",
            reason=reason[:500],
        )

    def _log(self, step_type: str, payload: Mapping[str, Any]) -> None:
        if self._trace is not None:
            try:
                self._trace.log_step(step_type=step_type, payload=dict(payload))
            except Exception as exc:
                logger.error("rollback %s trace failed: %s", step_type, exc)


__all__ = ["RollbackService"]
