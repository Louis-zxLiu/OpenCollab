"""Historical replay must restore Scope state before admitting a fresh Lead turn."""

from dataclasses import replace

from scheduler_awaiting_test_support import terminal
from test_history_application import live_scheduler


async def test_same_checkpoint_creates_independent_replays_and_new_history():
    scheduler, lead = live_scheduler(terminal("first"), terminal("second"), terminal("third"))
    lead.state.messages.append({"role": "system", "content": "stable role instructions"})
    await scheduler.run("initial")
    original = scheduler.history_list()
    cp = next(e for e in original if e.boundary == "initial")
    first = await scheduler.replay_from(cp.checkpoint_id, "retry one", 1024)
    assert first.status == "started"
    assert await first.task == "second"
    second = await scheduler.replay_from(cp.checkpoint_id, "retry two", 1024)
    assert second.status == "started"
    assert await second.task == "third"
    assert first.replay_id != second.replay_id
    assert first.old_epochs == ((0, 0),) and first.new_epochs == ((0, 1),)
    assert second.old_epochs == ((0, 1),) and second.new_epochs == ((0, 2),)
    assert scheduler.history_list()[:len(original)] == original
    assert lead.state.messages[0] == {"role": "system", "content": "stable role instructions"}
    assert not any(message.get("content") == "first" for message in lead.state.messages)
    entries = scheduler._history.entries()
    assert {e.replay_id for e in entries if e.boundary == "turn_completed"} >= {first.replay_id, second.replay_id}
    assert scheduler.history_replayable(cp.checkpoint_id)


async def test_invalid_checkpoint_and_budget_fail_before_fence():
    scheduler, lead = live_scheduler()
    await scheduler.run("initial")
    invalid = await scheduler.replay_from("missing", "retry", 100)
    assert invalid.status == invalid.failure_reason == "invalid_checkpoint"
    cp = next(e for e in scheduler.history_list() if e.checkpoint_id)
    exhausted = await scheduler.replay_from(cp.checkpoint_id, "retry", 2_000_000)
    assert exhausted.status == exhausted.failure_reason == "budget_exhausted"
    assert not scheduler._rollback_fenced
    assert scheduler.current_effect_epoch(0) == 0


async def test_restore_failure_keeps_source_effect_and_agent_fenced(monkeypatch):
    scheduler, lead = live_scheduler()
    await scheduler.run("initial")
    cp = next(e for e in scheduler.history_list() if e.boundary == "turn_completed")
    before = dict(scheduler._rollback_service.effects)
    restore = lead.env.restore_scope

    async def fail(checkpoint):
        from opencollab.domain.rollback import RestoreResult
        return RestoreResult(0, checkpoint.checkpoint_id, "failed", reason="private provider path")

    monkeypatch.setattr(lead.env, "restore_scope", fail)
    outcome = await scheduler.replay_from(cp.checkpoint_id, "retry", 1024)
    assert outcome.status == "restore_failed"
    assert dict(scheduler._rollback_service.effects) == before
    assert scheduler._rollback_fenced == {0}
    assert not any(e.boundary == "replay_completed" for e in scheduler.history_list())
    assert "private provider path" not in repr(scheduler.history_list())
    monkeypatch.setattr(lead.env, "restore_scope", restore)
    lead._steps.append(terminal("recovered"))
    retry = await scheduler.replay_from(cp.checkpoint_id, "retry after transient failure", 1024)
    assert retry.status == "started"
    assert await retry.task == "recovered"
    assert not scheduler._rollback_fenced


async def test_invalidated_source_is_rejected_before_fencing():
    scheduler, lead = live_scheduler()
    await scheduler.run("initial")
    cp = next(e for e in scheduler.history_list() if e.boundary == "turn_completed")
    effect = next(iter(scheduler._rollback_service.effects.values()))
    scheduler._rollback_service._effects[effect.effect_id] = replace(effect, status="invalidated")
    assert not scheduler.history_replayable(cp.checkpoint_id)
    assert (await scheduler.replay_from(cp.checkpoint_id, "retry", 100)).status == "invalid_checkpoint"
    assert not scheduler._rollback_fenced
