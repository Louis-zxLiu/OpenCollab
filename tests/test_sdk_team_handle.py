"""Public live-Team and rollback facade tests."""

from __future__ import annotations

import asyncio

import pytest

from opencollab.application.scheduler_types import SchedulerTurnError
from opencollab.domain.rollback import (
    RestoreResult as DomainRestoreResult,
)
from opencollab.domain.rollback import (
    RollbackPlan as DomainRollbackPlan,
)
from opencollab.domain.rollback import (
    RollbackResult as DomainRollbackResult,
)
from opencollab.domain.session import SessionPhase
from opencollab.sdk import OpenCollab, TeamHandle
from opencollab.sdk import team as team_module


class FakeScheduler:
    used_tokens = 3

    def __init__(self) -> None:
        self.resumed: set[int] = set()
        self.executed = False
        self.closed = False
        self.continued = []
        self.plan = DomainRollbackPlan(
            target_effect_ids=frozenset({"effect-1"}),
            invalidated_effect_ids=frozenset({"effect-1"}),
            affected_agent_ids=frozenset({1}),
            checkpoint_by_agent={1: None},
        )

    async def run(self, message: str) -> str:
        await asyncio.sleep(0)
        return message

    def preview_rollback(self, effect_ids):
        assert effect_ids == {"effect-1"}
        return self.plan

    async def rollback_effect(self, effect_ids, *, expected_plan_digest):
        assert expected_plan_digest == self.plan.digest()
        self.executed = True
        return DomainRollbackResult(
            self.plan,
            (
                DomainRestoreResult(
                    1,
                    None,
                    "restored",
                    filesystem_digest="fs-digest",
                    environment_digest="env-digest",
                ),
            ),
            True,
        )

    def resume_after_rollback(self, agent_ids):
        self.resumed = set(agent_ids)

    async def rollback_to_checkpoint(self, aid, checkpoint_id):
        return DomainRestoreResult(aid, checkpoint_id, "restored")

    async def continue_team_turn(self, aid, message, *, budget_tokens=None):
        self.continued.append((aid, message, budget_tokens))
        return "continued"

    async def cleanup(self, cleanup_timeout):
        self.closed = True


class PreparedScheduler:
    def __init__(self) -> None:
        self.table = type("_Table", (), {"entries": {0: object(), 2: object()}})()
        self.prebuilt = False
        self.checkpoints: list[tuple[int, str]] = []
        self.cleaned = False

    async def ensure_team_prebuilt(self):
        self.prebuilt = True

    async def create_checkpoint(self, aid, boundary):
        assert self.prebuilt is True
        self.checkpoints.append((aid, boundary))

    async def cleanup(self):
        self.cleaned = True


async def test_live_team_handle_exposes_safe_rollback_views(monkeypatch, tmp_path):
    scheduler = FakeScheduler()
    context = object()

    async def prepare(**kwargs):
        return scheduler, context

    async def close_runtime(actual_scheduler, actual_context, cleanup_timeout):
        assert actual_scheduler is scheduler
        assert actual_context is context
        scheduler.closed = True

    monkeypatch.setattr("opencollab.sdk.client.prepare_team", prepare)
    monkeypatch.setattr(team_module, "close_team_runtime", close_runtime)

    client = OpenCollab(tmp_path, config={"model": "test", "provider": "openai"})
    handle = await client.start_team("initial", artifacts=tmp_path / "artifacts")

    assert isinstance(handle, TeamHandle)
    plan = handle.rollback.preview({"effect-1"})
    assert plan.digest == scheduler.plan.digest()
    assert plan.checkpoint_by_agent == {1: None}

    result = await handle.rollback.execute(
        {"effect-1"},
        expected_plan_digest=plan.digest,
    )
    assert result.invalidated is True
    assert result.restores[0].environment_digest == "env-digest"
    handle.rollback.resume(result.affected_agent_ids)
    assert scheduler.resumed == {1}

    assert (await handle.wait()).output == "initial"
    assert await handle.rollback.continue_turn(0, "continue") == "continued"
    assert scheduler.continued == [(0, "continue", None)]
    assert (await handle.wait()).output == "continued"
    await handle.close()
    assert scheduler.closed is True


async def test_team_handle_passes_explicit_continuation_budget(monkeypatch, tmp_path):
    scheduler = FakeScheduler()
    context = type("Context", (), {"tracer": None})()

    async def prepare(**kwargs):
        return scheduler, context

    monkeypatch.setattr("opencollab.sdk.client.prepare_team", prepare)
    client = OpenCollab(tmp_path, config={"model": "test", "provider": "openai"})
    handle = await client.start_team("initial")

    await handle.wait()
    await handle.rollback.continue_turn(0, "continue", budget_tokens=4096)

    assert scheduler.continued == [(0, "continue", 4096)]
    await handle.close()


async def test_prepare_team_checkpoints_prebuilt_scopes_before_return(monkeypatch, tmp_path):
    from opencollab.bootstrap import programmatic as programmatic_module
    from opencollab.bootstrap.programmatic_team import prepare_team

    scheduler = PreparedScheduler()
    context = type("_Context", (), {"tracer": None})()
    monkeypatch.setattr(
        "opencollab.bootstrap.programmatic_team.build_runtime_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        "opencollab.bootstrap.programmatic_team.load_team_config",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(programmatic_module, "_claim_artifacts", lambda _path: None)
    monkeypatch.setattr(
        programmatic_module,
        "build_scheduler",
        lambda *_args, **_kwargs: scheduler,
    )

    actual, actual_context = await prepare_team(
        prompt="task",
        config={"budget": 10},
        workspace=str(tmp_path),
        team_config_path=None,
        max_tokens=10,
        artifacts=None,
        trace=False,
        use_worktrees=False,
    )

    assert actual is scheduler
    assert actual_context is context
    assert scheduler.checkpoints == [(0, "initial"), (2, "initial")]


async def test_wait_reports_rollback_cancellation_without_swallowing_wait_cancellation():
    scheduler = FakeScheduler()

    async def interrupted_run():
        raise SchedulerTurnError(0, SessionPhase.STOPPED, "rollback_interrupted", None)

    run_task = asyncio.create_task(interrupted_run())
    handle = TeamHandle(
        scheduler,
        object(),
        run_task,
        cleanup_timeout=1,
        artifacts=None,
    )

    result = await handle.wait()

    assert result.status == "stopped"
    assert result.reason == "rollback_interrupted"

    waiting_task = asyncio.create_task(asyncio.sleep(60))
    waiting_handle = TeamHandle(
        scheduler,
        object(),
        waiting_task,
        cleanup_timeout=1,
        artifacts=None,
    )
    waiter = asyncio.create_task(waiting_handle.wait())
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    waiting_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting_task


async def test_wait_does_not_mislabel_plain_cancellation_as_rollback():
    task = asyncio.create_task(asyncio.Event().wait())
    handle = TeamHandle(FakeScheduler(), object(), task, cleanup_timeout=1, artifacts=None)
    task.cancel()
    result = await handle.wait()
    assert result.status == "stopped"
    assert result.reason == "cancelled"


async def test_close_failure_can_be_retried_but_rollbacks_remain_closed(monkeypatch):
    scheduler = FakeScheduler()
    task = asyncio.create_task(scheduler.run("done"))
    handle = TeamHandle(scheduler, object(), task, cleanup_timeout=1, artifacts=None)
    await handle.wait()
    calls = 0

    async def cleanup(*_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(team_module, "close_team_runtime", cleanup)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await handle.close()
    assert not handle._closed
    with pytest.raises(RuntimeError, match="closed"):
        handle.rollback.preview({"effect-1"})
    await handle.close()
    await handle.close()
    assert calls == 2
    assert handle._closed
    with pytest.raises(RuntimeError, match="closed"):
        handle.rollback.resume({1})
    operations = [
        handle.rollback.create_checkpoint(0),
        handle.rollback.execute({"effect-1"}, expected_plan_digest="digest"),
        handle.rollback.restore_checkpoint(0, "checkpoint"),
        handle.rollback.continue_turn(0, "continue"),
        handle.continue_turn(0, "continue"),
    ]
    for operation in operations:
        with pytest.raises(RuntimeError, match="closed"):
            await operation


def test_public_restore_error_does_not_echo_adapter_secrets():
    restored = DomainRestoreResult(1, "checkpoint", "failed", reason="secret=value /private/path")
    result = team_module._restore_view(restored)
    assert result.reason == "scope_restore_failed"
    assert "private" not in repr(result)
    assert "secret" not in repr(result)


async def test_cancelling_active_wait_does_not_cancel_the_owned_team():
    task = asyncio.create_task(asyncio.Event().wait())
    handle = TeamHandle(FakeScheduler(), object(), task, cleanup_timeout=1, artifacts=None)
    waiter = asyncio.create_task(handle.wait())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not task.done()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
