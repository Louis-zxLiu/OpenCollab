"""Fault injection at the thin rollback transaction and lifecycle boundaries."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from scheduler_awaiting_test_support import ScriptedSession, build_scheduler
from test_rollback_concurrency import CheckpointEnvironment, _checkpoint_agent, _register_child

from opencollab.application.rollback import RollbackService
from opencollab.domain.pending import PendingRow, RowKind, RowStatus
from opencollab.domain.rollback import EnvironmentSnapshot
from opencollab.domain.session import SessionPhase


async def _ready_scheduler():
    scheduler, _ = build_scheduler(ScriptedSession("lead", []), [])
    await _checkpoint_agent(scheduler, 0, CheckpointEnvironment(0))
    child = ScriptedSession("coder", [])
    _register_child(scheduler, child, CheckpointEnvironment(1))
    await scheduler.create_checkpoint(1)
    return scheduler, child


def _effect(scheduler, aid=0):
    return scheduler.create_effect_ref(producer_aid=aid, kind="tool_result", epoch=0, attempt=0)


async def test_direct_restore_restores_frontier_without_invalidating_effects():
    scheduler, _ = await _ready_scheduler()
    effect = _effect(scheduler, 1)
    scheduler.consume_effect_ref(effect.effect_id, 0)
    checkpoint = scheduler._rollback_service._checkpoints[0][0]
    result = await scheduler.rollback_to_checkpoint(0, checkpoint.checkpoint_id)
    assert result.status == "restored"
    assert scheduler._rollback_service.causal_frontier(0) == checkpoint.causal_frontier
    assert scheduler._rollback_service.effects[effect.effect_id].status == "active"
    scheduler.resume_after_rollback({0})
    created = scheduler.create_effect_ref(producer_aid=0, kind="tool_result", epoch=1, attempt=1)
    assert created.parent_effect_ids == ()


async def test_direct_restore_rejects_checkpoint_from_invalidated_branch_before_fence():
    scheduler, _ = await _ready_scheduler()
    effect = _effect(scheduler)
    contaminated = await scheduler.create_checkpoint(0, "tool")
    plan = scheduler.preview_rollback({effect.effect_id})
    outcome = await scheduler.rollback_effect({effect.effect_id}, expected_plan_digest=plan.digest())
    assert outcome.invalidated
    scheduler.resume_after_rollback({0})
    frontier = scheduler._rollback_service.causal_frontier(0)
    with pytest.raises(ValueError, match="invalidated effects"):
        await scheduler.rollback_to_checkpoint(0, contaminated.checkpoint_id)
    assert scheduler._rollback_service.causal_frontier(0) == frontier
    assert scheduler.current_effect_epoch(0) == 1
    assert not scheduler._rollback_fenced
    assert not scheduler._rollback_operations


@pytest.mark.parametrize("phase", ["_cancel_fenced_tasks", "_settle_rollback_turns", "_quiesce_agents"])
@pytest.mark.parametrize("direct", [False, True])
async def test_exception_after_fence_retains_retryable_operation(monkeypatch, phase, direct):
    scheduler, _ = await _ready_scheduler()
    effect = _effect(scheduler)
    checkpoint = scheduler._rollback_service._checkpoints[0][0]
    original = getattr(scheduler, phase)

    async def fail(_aids):
        raise RuntimeError("injected operation failure")

    async def execute():
        if direct:
            return await scheduler.rollback_to_checkpoint(0, checkpoint.checkpoint_id)
        plan = scheduler.preview_rollback({effect.effect_id})
        return await scheduler.rollback_effect({effect.effect_id}, expected_plan_digest=plan.digest())

    monkeypatch.setattr(scheduler, phase, fail)
    with pytest.raises(RuntimeError, match="injected"):
        await execute()
    assert scheduler._rollback_affected == {0}
    assert scheduler._rollback_fenced == {0}
    assert scheduler._rollback_operations[0].phase == "failed"
    assert scheduler._rollback_service.effects[effect.effect_id].status == "active"
    with pytest.raises(RuntimeError, match="failed restore"):
        scheduler.resume_after_rollback({0})
    monkeypatch.setattr(scheduler, phase, original)
    await execute()
    scheduler.resume_after_rollback({0})
    assert scheduler.current_effect_epoch(0) == 1
    with pytest.raises(ValueError, match="not affected"):
        scheduler.resume_after_rollback({0})


async def test_disjoint_rollback_does_not_lose_previous_fence_and_overlap_rejected():
    scheduler, _ = await _ready_scheduler()
    first = scheduler._rollback_service._checkpoints[0][0]
    second = scheduler._rollback_service._checkpoints[1][0]
    await scheduler.rollback_to_checkpoint(0, first.checkpoint_id)
    with pytest.raises(RuntimeError, match="unresolved rollback"):
        await scheduler.rollback_to_checkpoint(0, first.checkpoint_id)
    await scheduler.rollback_to_checkpoint(1, second.checkpoint_id)
    assert scheduler._rollback_affected == {0, 1}
    scheduler.resume_after_rollback({0})
    assert scheduler._rollback_fenced == {1}
    scheduler.resume_after_rollback({1})


@pytest.mark.parametrize("failure_point", ["create_effect", "register_consumer", "queue_pending_user_message"])
async def test_message_enqueue_failure_rolls_back_graph_inbox_and_pending(monkeypatch, failure_point):
    scheduler, child = await _ready_scheduler()
    service = scheduler._rollback_service
    owner = child.state if failure_point == "queue_pending_user_message" else service
    original = getattr(owner, failure_point)

    def mutate_then_fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected enqueue failure")

    monkeypatch.setattr(owner, failure_point, mutate_then_fail)
    with pytest.raises(RuntimeError, match="injected"):
        await scheduler.send_message(0, 1, "handoff", "private payload")
    assert child.state.pending_user_messages == []
    assert scheduler._message_inbox == {}
    assert dict(service.effects) == {}
    assert service.causal_frontier(1) == frozenset()
    assert service._consumers == {}


@pytest.mark.parametrize("guard", ["unregistered", "no_environment", "fenced", "old_epoch"])
async def test_automatic_lifecycle_effect_obeys_registration_fence_and_epoch(guard):
    scheduler, _ = await _ready_scheduler()
    aid = 0
    token = None
    if guard == "unregistered":
        aid = 99
    elif guard == "no_environment":
        scheduler._rollback_service._environments.pop(0)
    elif guard == "fenced":
        scheduler._rollback_fenced.add(0)
    else:
        token = scheduler._rollback_turn_epoch.set((0, 0))
        scheduler._rollback_epochs[0] = 1
    try:
        with pytest.raises(RuntimeError):
            scheduler._record_lifecycle_effect(producer_aid=aid, kind="message", content="x", consumer_aid=1)
        assert dict(scheduler._rollback_service.effects) == {}
    finally:
        if token is not None:
            scheduler._rollback_turn_epoch.reset(token)


@pytest.mark.parametrize("field,value", [
    ("filesystem_revision", "different-revision"),
    ("filesystem_digest", "different-tree"),
    ("workspace_identity", "/different-workspace"),
    ("causal_frontier", frozenset({"different-effect"})),
    ("environment", EnvironmentSnapshot.from_mapping({"PRIVATE": "different-value"})),
])
async def test_plan_digest_binds_checkpoint_content(field, value):
    scheduler, _ = await _ready_scheduler()
    effect = _effect(scheduler)
    plan = scheduler.preview_rollback({effect.effect_id})
    checkpoint = plan.checkpoint_by_agent[0]
    changed = replace(plan, checkpoint_by_agent={0: replace(checkpoint, **{field: value})})
    assert changed.digest() != plan.digest()


async def test_rollback_cancels_delivery_owner_before_acquiring_recipient_lock():
    scheduler, child = await _ready_scheduler()
    started = asyncio.Event()

    async def blocking_add(message):
        child.state.append_message({"role": "user", "content": message})
        started.set()
        await asyncio.Event().wait()

    child.add_user_message = blocking_add
    child.state.set_phase(SessionPhase.DONE)
    sending = asyncio.create_task(scheduler.send_message(0, 1, "queued", "content"))
    await started.wait()
    message_effect = next(iter(scheduler._rollback_service.effects))
    plan = scheduler.preview_rollback({message_effect})
    result = await asyncio.wait_for(
        scheduler.rollback_effect({message_effect}, expected_plan_digest=plan.digest()), 2
    )
    assert result.invalidated
    assert sending.cancelled()
    assert not scheduler._message_delivery_tasks
    assert not scheduler._message_inbox
    assert child.state.messages == []


def test_effect_transaction_restores_partially_updated_consumer_graph():
    service = RollbackService()
    with pytest.raises(RuntimeError):
        with service.effect_transaction():
            effect = service.create_effect(producer_aid=0, kind="message", epoch=0, attempt=0)
            service.register_consumer(effect.effect_id, 1)
            raise RuntimeError("injected")
    assert dict(service.effects) == {}
    assert service._consumers == {}
    assert service.causal_frontier(1) == frozenset()


async def test_message_waiting_for_lock_rejects_completed_resume_epoch():
    scheduler, _ = await _ready_scheduler()
    lock = scheduler._locks.setdefault(1, asyncio.Lock())
    await lock.acquire()
    sending = asyncio.create_task(scheduler.send_message(0, 1, "late", "old epoch"))
    await asyncio.sleep(0)
    scheduler._rollback_epochs[1] += 1
    lock.release()
    assert "stale" in await sending
    assert not scheduler._message_inbox
    assert not scheduler._rollback_service.effects


async def test_continuation_requires_budget_before_creating_a_turn():
    scheduler, _ = await _ready_scheduler()
    checkpoint = scheduler._rollback_service._checkpoints[0][0]
    await scheduler.rollback_to_checkpoint(0, checkpoint.checkpoint_id)
    scheduler.resume_after_rollback({0})
    with pytest.raises(RuntimeError, match="budget_exhausted"):
        await scheduler.continue_team_turn(0, "retry")
    assert not scheduler._tasks
    assert not scheduler._rollback_service.effects


async def test_continuation_keeps_explicit_initial_budget():
    scheduler, _ = await _ready_scheduler()
    checkpoint = scheduler._rollback_service._checkpoints[0][0]
    await scheduler.rollback_to_checkpoint(0, checkpoint.checkpoint_id)
    scheduler.resume_after_rollback({0})
    observed = []

    async def finish(session):
        observed.append(scheduler._turn_lease[0])
        session.state.set_phase(SessionPhase.DONE)
        session.state.append_message({"role": "assistant", "content": "done"})
        return "done"

    scheduler._sessions[0]._steps = [finish]
    assert await scheduler.continue_team_turn(0, "retry", budget_tokens=1234) == "done"
    assert observed == [1234]


async def test_failed_continuation_keeps_explicit_retry_available(monkeypatch):
    scheduler, _ = await _ready_scheduler()
    checkpoint = scheduler._rollback_service._checkpoints[0][0]
    await scheduler.rollback_to_checkpoint(0, checkpoint.checkpoint_id)
    scheduler.resume_after_rollback({0})

    attempts = 0

    async def fail_once(_aid, _message, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("provider failed before the continuation turn started")

    monkeypatch.setattr(scheduler, "run_turn", fail_once)
    with pytest.raises(RuntimeError, match="provider failed"):
        await scheduler.continue_team_turn(0, "retry", budget_tokens=1234)

    assert attempts == 1
    assert 0 in scheduler._rollback_continuations
    assert 0 not in scheduler._turn_lease

    async def succeed(_aid, _message, **_kwargs):
        return "continued"

    monkeypatch.setattr(scheduler, "run_turn", succeed)
    assert await scheduler.continue_team_turn(0, "retry again", budget_tokens=1234) == "continued"
    assert 0 not in scheduler._rollback_continuations


async def test_unaffected_agent_cannot_consume_a_fenced_producers_effect():
    scheduler, _ = await _ready_scheduler()
    effect = _effect(scheduler)
    scheduler._rollback_fenced.add(0)
    with pytest.raises(RuntimeError, match="fenced"):
        scheduler.consume_effect_ref(effect.effect_id, 1)
    with pytest.raises(RuntimeError, match="fenced"):
        scheduler.create_effect_ref(
            producer_aid=1, kind="tool_result", epoch=0, attempt=0,
            parent_effect_ids=(effect.effect_id,),
        )
    assert scheduler._rollback_service.causal_frontier(1) == frozenset()
    assert len(scheduler._rollback_service.effects) == 1


async def test_settling_coordinator_does_not_cancel_an_independent_live_sibling():
    scheduler, _ = await _ready_scheduler()
    sibling = ScriptedSession("independent", [])
    _register_child(scheduler, sibling, CheckpointEnvironment(2), aid=2)
    lead = scheduler._sessions[0]
    lead.state.set_phase(SessionPhase.AWAITING_EVENTS)
    for aid in (1, 2):
        row_id = f"child-{aid}"
        lead.state.pending_events.add(PendingRow(row_id, RowKind.CHILD_AGENT, aid, ref=aid))
        scheduler._spawn_origin[aid] = (0, row_id)
    sibling_task = asyncio.create_task(asyncio.Event().wait())
    scheduler._tasks[2] = sibling_task
    scheduler._turn_lease[2] = 100
    target = _effect(scheduler, 1)
    try:
        plan = scheduler.preview_rollback({target.effect_id})
        assert plan.affected_agent_ids == frozenset({0, 1})
        result = await scheduler.rollback_effect({target.effect_id}, expected_plan_digest=plan.digest())
        assert result.invalidated
        assert not sibling_task.done()
        assert scheduler._turn_lease[2] == 100
        assert scheduler.current_effect_epoch(2) == 0
        assert sibling.state.phase is SessionPhase.IDLE
        assert lead.state.pending_events.is_empty()
    finally:
        sibling_task.cancel()
        await asyncio.gather(sibling_task, return_exceptions=True)


async def test_child_effect_failure_restores_pending_row_and_frontier(monkeypatch):
    scheduler, _ = await _ready_scheduler()
    lead = scheduler._sessions[0]
    row = PendingRow("child", RowKind.CHILD_AGENT, 0, ref=1)
    lead.state.pending_events.add(row)
    scheduler._spawn_origin[1] = (0, "child")
    original = scheduler._rollback_service.register_consumer

    def fail_after_consume(*args):
        original(*args)
        raise RuntimeError("consumer fault")

    monkeypatch.setattr(scheduler._rollback_service, "register_consumer", fail_after_consume)
    with pytest.raises(RuntimeError, match="consumer fault"):
        await scheduler._deliver_to_parent(1, "child result", RowStatus.DONE)
    assert lead.state.pending_events.rows["child"] == row
    assert scheduler._spawn_origin[1] == (0, "child")
    assert not scheduler._rollback_service.effects
    assert not scheduler._rollback_child_effects
    assert scheduler._rollback_service.causal_frontier(0) == frozenset()
