"""The public facade exposes metadata, preserves one-shot team compatibility, and closes replay."""

import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from scheduler_awaiting_test_support import terminal
from test_history_application import live_scheduler

from opencollab.application.replay import ReplayRun
from opencollab.application.scheduler_types import SchedulerTurnError
from opencollab.domain.session import SessionPhase
from opencollab.sdk import TeamHandle
from opencollab.sdk.history import ReplayHandle


async def test_replay_timing_is_fixed_at_execution_even_when_wait_is_late_or_repeated(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("opencollab.application.replay.time", SimpleNamespace(perf_counter=lambda: clock[0]))

    async def retry(session):
        clock[0] += 7
        return await terminal("replayed")(session)

    scheduler, _ = live_scheduler(terminal("initial"), retry)
    await scheduler.run("initial")
    checkpoint = next(e for e in scheduler.history_list() if e.boundary == "initial")
    execution = await scheduler.replay_from(checkpoint.checkpoint_id, "retry", 1024)
    await execution.task
    handle = ReplayHandle(execution)
    clock[0] = 500
    first = await handle.wait()
    clock[0] = 1000
    assert await handle.wait() == first
    assert dict(first.phase_latency_ms)["replay_turn"] == 7000
    assert dict(first.phase_latency_ms)["complete_replay_task"] == 7000


@pytest.mark.parametrize("structured", [True, False])
async def test_replay_failure_redacts_identifier_shaped_private_values(structured):
    private_value = "private_workspace_secret"

    async def fail():
        if structured:
            raise SchedulerTurnError(0, SessionPhase.ERROR, private_value, None)
        raise type(private_value, (Exception,), {})(private_value)

    execution = ReplayRun("replay-private", "started", task=asyncio.create_task(fail()))
    result = await ReplayHandle(execution).wait()
    assert private_value not in repr(result)
    assert result.failure_reason == ("error" if structured else "replay_failed")


async def test_history_facade_is_redacted_and_replay_handle_survives_team_completion(monkeypatch):
    scheduler, lead = live_scheduler(terminal("private model output"), terminal("private replay output"))
    task = asyncio.create_task(scheduler.run("private user text"))
    handle = TeamHandle(scheduler, object(), task, cleanup_timeout=1, artifacts=None)
    await handle.wait()
    entries = handle.history.list()
    cp = next(e for e in entries if e.checkpoint_id)
    assert handle.history.get(cp.entry_id) == cp
    assert handle.history.replayable(cp.checkpoint_id)
    expected = {"entry_id", "checkpoint_id", "effect_id", "turn_id", "agent_id", "epoch", "boundary",
                "status", "timestamp", "filesystem_digest", "environment_digest",
                "workspace_identity_digest", "replayable", "failure_reason"}
    assert set(asdict(cp)) == expected
    assert not any(text in repr(entries) for text in ("private model", "/scope/", "SCOPE"))
    replay = await handle.replay_from(cp.checkpoint_id, message="Retry", budget_tokens=1024)
    replay_result = await replay.wait()
    assert replay_result.status == "completed"
    phases = dict(replay_result.phase_latency_ms)
    assert phases["replay_turn"] >= 0
    assert phases["complete_replay_task"] >= phases["replay_turn"]

    async def close(*args):
        return None
    monkeypatch.setattr("opencollab.sdk.team.close_team_runtime", close)
    await handle.close()
    assert not handle.history.replayable(cp.checkpoint_id)
    assert handle.history.get(cp.entry_id).failure_reason == "handle_closed"
    with pytest.raises(RuntimeError, match="closed"):
        await handle.replay_from(cp.checkpoint_id, message="Retry")


async def test_replay_result_preserves_safe_failure_reason_without_exception_payload():
    async def fail():
        raise SchedulerTurnError(0, SessionPhase.ERROR, "provider_unavailable", None)

    execution = ReplayRun("replay-1", "started", task=asyncio.create_task(fail()))
    result = await ReplayHandle(execution).wait()
    assert result.status == "replay_start_failed"
    assert result.failure_reason == "provider_unavailable"

    async def fail_with_private_payload():
        raise SchedulerTurnError(0, SessionPhase.ERROR, "failure at /private/workspace", None)

    execution = ReplayRun("replay-2", "started", task=asyncio.create_task(fail_with_private_payload()))
    result = await ReplayHandle(execution).wait()
    assert result.failure_reason == "error"
    assert "/private/workspace" not in repr(result)

    async def fail_with_provider_payload():
        raise SchedulerTurnError(0, SessionPhase.ERROR, "RateLimitError: upstream returned 429", None)

    execution = ReplayRun("replay-3", "started", task=asyncio.create_task(fail_with_provider_payload()))
    result = await ReplayHandle(execution).wait()
    assert result.failure_reason == "provider_rate_limited"
    assert "upstream" not in repr(result)
