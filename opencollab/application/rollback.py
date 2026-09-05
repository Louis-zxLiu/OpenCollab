"""Application service coordinating causal invalidation and Scope restore."""

from __future__ import annotations

import secrets
import time
from dataclasses import replace
from typing import Any, Mapping

from opencollab.application.ports import CheckpointableEnvironmentPort, TracePort
from opencollab.domain.rollback import (
    CheckpointBoundary,
    EffectRef,
    RestoreResult,
    ScopeCheckpoint,
    WorkspaceRevision,
    compute_content_hash,
    compute_descendants,
    reduce_causal_frontier,
)


class RollbackService:
    """Own the causal graph, consumer index, checkpoints, and restore outcomes."""

    def __init__(self, trace_port: TracePort | None = None) -> None:
        self._trace = trace_port
        self._effects: dict[str, EffectRef] = {}
        self._consumers: dict[str, set[int]] = {}
        self._checkpoints: dict[int, list[ScopeCheckpoint]] = {}
        self._environments: dict[int, CheckpointableEnvironmentPort] = {}
        self._frontiers: dict[int, frozenset[str]] = {}

    def register_environment(self, aid: int, environment: CheckpointableEnvironmentPort) -> None:
        self._environments[aid] = environment

    def has_checkpoint(self, aid: int) -> bool:
        return bool(self._checkpoints.get(aid))

    def create_effect(
        self,
        producer_aid: int,
        attempt: int,
        branch_id: str,
        epoch: int,
        kind: str,
        parent_effect_ids: tuple[str, ...],
        content: str,
    ) -> EffectRef:
        effect = EffectRef(
            effect_id=f"e_{time.time_ns():x}_{secrets.token_hex(4)}",
            producer_aid=producer_aid,
            attempt=attempt,
            kind=kind,
            epoch=epoch,
            parent_effect_ids=parent_effect_ids,
            content_hash=compute_content_hash(content),
        )
        self._effects[effect.effect_id] = effect
        self._log("effect_created", {"effect_id": effect.effect_id, "kind": kind})
        return effect

    def register_consumer(self, effect_id: str, consumer_aid: int) -> None:
        if effect_id not in self._effects:
            raise ValueError(f"unknown effect_id: {effect_id}")
        self._consumers.setdefault(effect_id, set()).add(consumer_aid)

    def consumer_can_access(self, effect_id: str, consumer_aid: int) -> bool:
        """Whether delivery made this Effect visible to the Agent."""
        return consumer_aid in self._consumers.get(effect_id, ())

    def attach_workspace_revision(
        self,
        effect_id: str,
        revision: WorkspaceRevision,
    ) -> EffectRef:
        effect = self._effects.get(effect_id)
        if effect is None:
            raise ValueError(f"unknown effect_id: {effect_id}")
        updated = replace(
            effect,
            workspace_revision=revision.revision,
            base_workspace_revision=revision.base_revision,
            workspace_files=revision.files,
        )
        self._effects[effect_id] = updated
        return updated

    def workspace_revision(self, effect_id: str) -> WorkspaceRevision | None:
        effect = self._effects.get(effect_id)
        if effect is None or effect.workspace_revision is None or effect.base_workspace_revision is None:
            return None
        return WorkspaceRevision(
            revision=effect.workspace_revision,
            base_revision=effect.base_workspace_revision,
            files=effect.workspace_files,
        )

    def consume(self, consumer_aid: int, effect_id: str) -> frozenset[str]:
        frontier = reduce_causal_frontier(
            self._effects,
            self._frontiers.get(consumer_aid, frozenset()),
            effect_id,
        )
        self._frontiers[consumer_aid] = frontier
        self.register_consumer(effect_id, consumer_aid)
        return frontier

    def is_quarantined(self, effect_id: str) -> bool:
        effect = self._effects.get(effect_id)
        return effect is not None and effect.status == "quarantined"

    def get_effect(self, effect_id: str) -> EffectRef | None:
        return self._effects.get(effect_id)

    def quarantine(self, effect_id: str, reason: str, evidence: str = "") -> set[str]:
        if effect_id not in self._effects:
            raise ValueError(f"unknown effect_id: {effect_id}")
        affected = compute_descendants(self._effects, {effect_id})
        for current in affected:
            self._effects[current] = replace(self._effects[current], status="quarantined")
        self._log(
            "effect_quarantined",
            {
                "effect_id": effect_id,
                "count": len(affected),
                "reason": reason[:200],
                "evidence": evidence[:500],
            },
        )
        return affected

    def compute_affected_agents(self, effect_ids: set[str]) -> set[int]:
        return {aid for effect_id in effect_ids for aid in self._consumers.get(effect_id, ())}

    async def checkpoint(
        self,
        aid: int,
        boundary: CheckpointBoundary,
        causal_frontier: frozenset[str],
    ) -> ScopeCheckpoint:
        environment = self._environments.get(aid)
        if environment is None:
            raise RuntimeError(f"Agent {aid} has no checkpointable Scope")
        checkpoint = await environment.checkpoint_scope(
            boundary,
            owner_aid=aid,
            causal_frontier=causal_frontier,
        )
        self._checkpoints.setdefault(aid, []).append(checkpoint)
        return checkpoint

    def select_checkpoint(self, aid: int, invalidated: set[str]) -> ScopeCheckpoint | None:
        checkpoints = self._checkpoints.get(aid, ())
        relevant = [checkpoint for checkpoint in checkpoints if not (set(checkpoint.causal_frontier) & invalidated)]
        return max(relevant, key=lambda checkpoint: checkpoint.sequence, default=None)

    async def restore_affected(self, invalidated: set[str]) -> dict[int, RestoreResult | None]:
        outcomes: dict[int, RestoreResult | None] = {}
        for aid in self.compute_affected_agents(invalidated):
            checkpoint = self.select_checkpoint(aid, invalidated)
            if checkpoint is None:
                outcomes[aid] = None
                continue
            outcomes[aid] = await self._environments[aid].restore_scope(checkpoint)
        return outcomes

    async def restore_agent(self, aid: int, invalidated: set[str]) -> RestoreResult | None:
        checkpoint = self.select_checkpoint(aid, invalidated)
        if checkpoint is None:
            return None
        environment = self._environments.get(aid)
        if environment is None:
            return None
        return await environment.restore_scope(checkpoint)

    def rebuild_from_messages(self, messages: list[dict[str, Any]]) -> None:
        from opencollab.domain.rollback import lineage_envelope_from_dict

        for message in messages:
            envelope = lineage_envelope_from_dict(message.get("_lineage"))
            if envelope is None:
                continue
            self._effects[envelope.effect.effect_id] = envelope.effect
            if envelope.consumer_aid is not None:
                self._consumers.setdefault(envelope.effect.effect_id, set()).add(envelope.consumer_aid)

    def rebuild_from_snapshot(self, quarantined_effect_ids: set[str]) -> None:
        for effect_id in quarantined_effect_ids:
            effect = self._effects.get(effect_id)
            if effect is not None:
                self._effects[effect_id] = replace(effect, status="quarantined")

    def _log(self, step_type: str, payload: Mapping[str, Any]) -> None:
        if self._trace is not None:
            self._trace.log_step(step_type=step_type, payload=dict(payload))
