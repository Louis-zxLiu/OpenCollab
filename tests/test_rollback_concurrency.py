"""Scheduler races at the explicit rollback boundary."""

from __future__ import annotations

import asyncio

import pytest
from scheduler_awaiting_test_support import ScriptedSession, build_scheduler

from opencollab.application.scheduler_types import QueuedTeammateMessage
from opencollab.domain.pending import PendingRow, RowKind, RowStatus
from opencollab.domain.rollback import EnvironmentSnapshot, RestoreResult, ScopeCheckpoint
from opencollab.domain.scheduler import SessionControlBlock
from opencollab.domain.session import SessionPhase


class CheckpointEnvironment:
    def __init__(self, aid: int) -> None:
        self.aid = aid
        self.workspace = f"/scope/{aid}"
        self._environment = EnvironmentSnapshot.from_mapping({"SCOPE": str(aid)})
        self._sequence = 0

    def snapshot_environment(self) -> EnvironmentSnapshot:
        return self._environment

    def replace_environment(self, snapshot: EnvironmentSnapshot) -> None:
        self._environment = snapshot

    async def validate_checkpoint_scope(self, checkpoint):
        assert checkpoint.workspace_identity == self.workspace

    async def checkpoint_scope(self, boundary, *, owner_aid, causal_frontier):
        self._sequence += 1
        return ScopeCheckpoint(
            checkpoint_id=f"cp-{self.aid}-{self._sequence}",
            owner_aid=owner_aid,
            sequence=self._sequence,
            filesystem_revision=f"revision-{self.aid}-{self._sequence}",
            environment=self._environment,
            causal_frontier=causal_frontier,
            boundary=boundary,
            workspace_identity=self.workspace,
            filesystem_digest=f"digest-{self.aid}-{self._sequence}",
        )

    async def restore_scope(self, checkpoint):
        self._environment = checkpoint.environment
        return RestoreResult(
            self.aid,
            checkpoint.checkpoint_id,
            "restored",
            filesystem_digest=checkpoint.filesystem_digest,
            environment_digest=checkpoint.environment.digest(),
        )


class BlockingRestoreEnvironment(CheckpointEnvironment):
    def __init__(self, aid: int) -> None:
        super().__init__(aid)
        self.restore_started = asyncio.Event()
        self.allow_restore = asyncio.Event()

    async def restore_scope(self, checkpoint):
        self.restore_started.set()
        await self.allow_restore.wait()
        return await super().restore_scope(checkpoint)


def _register_child(scheduler, session, environment, *, aid: int = 1) -> None:
    session.state.aid = aid
    session.env = environment
    scheduler.table.add(
        SessionControlBlock(
            aid=aid,
            parent_aid=0,
            agent=session.agent,
            state=session.state,
        )
    )
    scheduler._sessions[aid] = session
    scheduler.register_effect_environment(aid, environment)


async def _checkpoint_agent(scheduler, aid: int, environment) -> None:
    session = scheduler._sessions[aid]
    session.env = environment
    scheduler.register_effect_environment(aid, environment)
    await scheduler.create_checkpoint(aid)


async def test_message_blocked_on_recipient_lock_is_refused_after_fence():
    scheduler, _events = build_scheduler(ScriptedSession("lead", []), [])
    child = ScriptedSession("coder", [])
    environment = CheckpointEnvironment(1)
    _register_child(scheduler, child, environment)
    await scheduler.create_checkpoint(1)
    target = scheduler.create_effect_ref(
        producer_aid=1,
        kind="tool_result",
        epoch=0,
        attempt=0,
    )
    effect_ids_before = set(scheduler._rollback_service.effects)
    recipient_lock = scheduler._locks.setdefault(1, asyncio.Lock())
    await recipient_lock.acquire()
    send = asyncio.create_task(
        scheduler.send_message(0, 1, "late", "must not cross rollback")
    )
    await asyncio.sleep(0)

    plan = scheduler.preview_rollback({target.effect_id})
    result = await scheduler.rollback_effect(
        {target.effect_id},
        expected_plan_digest=plan.digest(),
    )
    recipient_lock.release()
    acknowledgement = await send

    assert result.invalidated is True
    assert "message refused" in acknowledgement
    assert scheduler._message_inbox.get(1, []) == []
    assert child.state.pending_user_messages == []
    assert set(scheduler._rollback_service.effects) == effect_ids_before
    assert scheduler._rollback_service.causal_frontier(1) == frozenset()


async def test_rollback_rejects_already_queued_messages_before_team_wait_can_stall():
    scheduler, _events = build_scheduler(ScriptedSession("lead", []), [])
    child = ScriptedSession("coder", [])
    _register_child(scheduler, child, CheckpointEnvironment(1))
    await _checkpoint_agent(scheduler, 0, CheckpointEnvironment(0))
    await scheduler.create_checkpoint(1)
    target = scheduler.create_effect_ref(
        producer_aid=1,
        kind="tool_result",
        epoch=0,
        attempt=0,
    )
    scheduler.consume_effect_ref(target.effect_id, 0)
    message = QueuedTeammateMessage(
        from_aid=1,
        to_aid=0,
        summary="old result",
        content="must be rejected",
        xml="<teammate-message>must be rejected</teammate-message>",
        message_id="queued-old",
        from_epoch=0,
        to_epoch=0,
    )
    scheduler._message_inbox[0] = [message]
    scheduler._sessions[0].state.queue_pending_user_message(
        {"content": message.xml, "message_id": message.message_id}
    )

    plan = scheduler.preview_rollback({target.effect_id})
    result = await scheduler.rollback_effect(
        {target.effect_id}, expected_plan_digest=plan.digest()
    )

    assert result.invalidated is True
    assert scheduler._message_inbox.get(0, []) == []
    assert scheduler._sessions[0].state.pending_user_messages[0]["delivery_status"] == "rejected"
    assert scheduler._quiescent() is True


async def test_late_child_result_cannot_pollute_graph_during_rollback():
    lead = ScriptedSession("lead", [])
    scheduler, _events = build_scheduler(lead, [])
    child = ScriptedSession("coder", [])
    child_environment = CheckpointEnvironment(1)
    _register_child(scheduler, child, child_environment)
    await _checkpoint_agent(scheduler, 0, CheckpointEnvironment(0))
    await scheduler.create_checkpoint(1)
    lead.state.set_phase(SessionPhase.AWAITING_EVENTS)
    lead.state.pending_events.add(
        PendingRow(
            tool_call_id="child-1",
            kind=RowKind.CHILD_AGENT,
            order=0,
            ref=1,
            status=RowStatus.PENDING,
        )
    )
    scheduler._spawn_origin[1] = (0, "child-1")
    target = scheduler.create_effect_ref(
        producer_aid=1,
        kind="tool_result",
        epoch=0,
        attempt=0,
    )
    parent_lock = scheduler._locks.setdefault(0, asyncio.Lock())
    await parent_lock.acquire()
    delivery = asyncio.create_task(
        scheduler._deliver_to_parent(1, "late result", RowStatus.DONE)
    )
    await asyncio.sleep(0)

    plan = scheduler.preview_rollback({target.effect_id})
    assert plan.coordinator_agent_ids == frozenset({0})
    assert plan.affected_agent_ids == frozenset({0, 1})
    result = await scheduler.rollback_effect(
        {target.effect_id},
        expected_plan_digest=plan.digest(),
    )
    parent_lock.release()
    await delivery

    assert result.invalidated is True
    assert set(scheduler._rollback_service.effects) == {target.effect_id}
    assert lead.state.pending_events.is_empty()
    assert scheduler._rollback_service.causal_frontier(0) == frozenset()


async def test_effect_creation_is_rejected_while_restore_is_in_progress():
    scheduler, _events = build_scheduler(ScriptedSession("lead", []), [])
    environment = BlockingRestoreEnvironment(0)
    await _checkpoint_agent(scheduler, 0, environment)
    target = scheduler.create_effect_ref(
        producer_aid=0,
        kind="tool_result",
        epoch=0,
        attempt=0,
    )
    plan = scheduler.preview_rollback({target.effect_id})
    rollback = asyncio.create_task(
        scheduler.rollback_effect(
            {target.effect_id},
            expected_plan_digest=plan.digest(),
        )
    )
    await environment.restore_started.wait()

    with pytest.raises(RuntimeError, match="fenced"):
        scheduler.create_effect_ref(
            producer_aid=0,
            kind="tool_result",
            epoch=0,
            attempt=1,
        )
    environment.allow_restore.set()
    result = await rollback

    assert result.invalidated is True
    assert set(scheduler._rollback_service.effects) == {target.effect_id}
