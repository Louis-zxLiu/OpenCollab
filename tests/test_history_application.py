"""Automatic history must describe committed lifecycle events, never plausible placeholders."""

from dataclasses import replace

import pytest
from scheduler_awaiting_test_support import ScriptedSession, build_scheduler, terminal
from test_rollback_concurrency import CheckpointEnvironment, _register_child

from opencollab.application.history import HistoryIndex
from opencollab.domain.pending import PendingRow, RowKind
from opencollab.domain.session import SessionPhase


def live_scheduler(*steps, environment=None):
    lead = ScriptedSession("lead", list(steps) or [terminal("done")])
    lead.env = environment or CheckpointEnvironment(0)
    scheduler, _ = build_scheduler(lead, [])
    scheduler._history_enabled = True
    return scheduler, lead


async def test_normal_turn_records_effect_and_scope_and_turn_checkpoints():
    scheduler, lead = live_scheduler()
    assert await scheduler.run("task") == "done"
    entries = scheduler.history_list()
    assert {"initial", "turn_completed", "terminal", "turn_result"} <= {e.boundary for e in entries}
    assert any(e.effect_id and not e.checkpoint_id for e in entries)
    assert all(e.turn_id for e in entries if e.boundary == "turn_completed")
    assert all(e.filesystem_digest and e.environment_digest and e.workspace_identity_digest
               for e in entries if e.checkpoint_id)
    assert tuple(e.timestamp for e in entries) == tuple(sorted(e.timestamp for e in entries))


async def test_child_acceptance_commits_effect_and_checkpoint():
    scheduler, lead = live_scheduler()
    child = ScriptedSession("coder", [terminal("child result")])
    _register_child(scheduler, child, CheckpointEnvironment(1))
    # The child and parent each pass a normal driver boundary; no manual effect/checkpoint.
    await scheduler.run("lead")
    lead.state.pending_events.add(PendingRow("child", RowKind.CHILD_AGENT, 0, ref=1))
    lead.state.set_phase(SessionPhase.AWAITING_EVENTS)
    scheduler._spawn_origin[1] = (0, "child")
    lead._steps.append(terminal("accepted"))
    await scheduler._drive_agent(1, child)
    if scheduler._tasks.get(0):
        await scheduler._tasks[0]
    entries = scheduler.history_list()
    assert any(e.boundary == "child_result" and e.agent_id == 1 for e in entries)
    assert any(e.boundary == "child_result_accepted" and e.checkpoint_id for e in entries)
    assert scheduler._rollback_service.causal_frontier(0)


async def test_accepted_message_records_effect_and_checkpoint():
    scheduler, lead = live_scheduler(terminal("first"), terminal("accepted"))
    child = ScriptedSession("coder", [])
    _register_child(scheduler, child, CheckpointEnvironment(1))
    await scheduler.run("task")
    await scheduler.send_message(1, 0, "summary", "private message contents")
    if scheduler._tasks.get(0):
        await scheduler._tasks[0]
    entries = scheduler.history_list()
    assert any(e.boundary == "message" and e.effect_id for e in entries)
    assert any(e.boundary == "message_accepted" and e.checkpoint_id for e in entries)
    assert "private message contents" not in repr(entries)


async def test_enqueue_failure_rolls_back_history_frontiers_and_inbox(monkeypatch):
    scheduler, lead = live_scheduler()
    child = ScriptedSession("coder", [])
    _register_child(scheduler, child, CheckpointEnvironment(1))
    await scheduler.run("task")
    previous = scheduler.history_list()
    effects = dict(scheduler._rollback_service.effects)
    frontier = scheduler._rollback_service.causal_frontier(1)

    def refuse(*args):
        raise ValueError("injected pending failure")

    monkeypatch.setattr(lead.state, "queue_pending_user_message", refuse)
    with pytest.raises(ValueError):
        await scheduler.send_message(1, 0, "safe", "secret")
    assert scheduler.history_list() == previous
    assert dict(scheduler._rollback_service.effects) == effects
    assert scheduler._rollback_service.causal_frontier(1) == frontier
    assert not scheduler._message_inbox.get(0)
    assert not lead.state.pending_user_messages


def test_index_preserves_old_branch_and_rejects_epoch_and_duplicate():
    index = HistoryIndex()
    original = index.record(0, 0, "initial", checkpoint_id="cp")
    index.branch(0, 1, "replay-a", original.entry_id)
    branch = index.record(0, 1, "replay_started")
    assert index.get(original.entry_id) == original
    assert branch.parent_entry_ids == (original.entry_id,)
    with pytest.raises(ValueError, match="old epoch"):
        index.record(0, 0, "turn_completed")
    with pytest.raises(ValueError, match="duplicate"):
        index.append(replace(branch))
