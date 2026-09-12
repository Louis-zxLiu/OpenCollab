"""Replay serializes plan validation and excludes responses from earlier epochs."""

import asyncio

import pytest
from scheduler_awaiting_test_support import ScriptedSession, terminal
from test_history_application import live_scheduler
from test_rollback_concurrency import CheckpointEnvironment, _register_child


async def test_each_affected_agent_branches_from_its_own_restored_checkpoint():
    scheduler, _ = live_scheduler(terminal("lead first"), terminal("lead replay"))
    child = ScriptedSession("coder", [terminal("child first")])
    _register_child(scheduler, child, CheckpointEnvironment(1))
    await scheduler.run("lead")
    await scheduler.run_turn(1, "child")
    source = next(e for e in scheduler.history_list() if e.agent_id == 1 and e.boundary == "initial")
    child_effect = next(e for e in scheduler._rollback_service.effects.values() if e.producer_aid == 1)
    scheduler.consume_effect_ref(child_effect.effect_id, 0)
    plan = scheduler._replay_plan(source.checkpoint_id)
    expected = {aid: scheduler._history.checkpoint(cid).entry_id for aid, cid in plan.checkpoint_by_agent}
    replay = await scheduler.replay_from(source.checkpoint_id, "retry child and lead", 1024)
    assert replay.affected_agent_ids == (0, 1)
    await replay.task
    lead_start = next(e for e in scheduler._history.entries()
                      if e.replay_id == replay.replay_id and e.boundary == "replay_started")
    assert lead_start.parent_entry_ids == (expected[0],)
    assert scheduler._history._heads[1] == expected[1]


async def test_stale_plan_rejected_after_waiting_for_lock():
    scheduler, lead = live_scheduler()
    await scheduler.run("initial")
    cp = next(e for e in scheduler.history_list() if e.checkpoint_id)
    await scheduler._rollback_operation_lock.acquire()
    task = asyncio.create_task(scheduler.replay_from(cp.checkpoint_id, "retry", 100))
    await asyncio.sleep(0)
    scheduler._history_record(0, "audit_boundary")
    scheduler._rollback_operation_lock.release()
    assert (await task).status == "stale_plan"
    assert not scheduler._rollback_fenced


async def test_replay_isolates_sibling_and_rejects_old_epoch_effect():
    scheduler, lead = live_scheduler(terminal("first"), terminal("second"))
    sibling = ScriptedSession("observer", [terminal("observe")])
    _register_child(scheduler, sibling, CheckpointEnvironment(2), aid=2)
    await scheduler.run_turn(2, "observe")
    await scheduler.run("first")
    cp = next(e for e in scheduler.history_list() if e.agent_id == 0 and e.boundary == "initial")
    sibling_messages = list(sibling.state.messages)
    result = await scheduler.replay_from(cp.checkpoint_id, "retry", 100)
    assert result.affected_agent_ids == (0,)
    await result.task
    assert scheduler.current_effect_epoch(2) == 0
    assert sibling.state.messages == sibling_messages
    with pytest.raises(RuntimeError, match="stale"):
        scheduler.create_effect_ref(producer_aid=0, kind="tool_result", epoch=0, attempt=1)
    old = next(e for e in scheduler._rollback_service.effects.values() if e.producer_aid == 0 and e.epoch == 0)
    with pytest.raises(ValueError, match="old epoch"):
        scheduler.consume_effect_ref(old.effect_id, 2)
