"""Checkpointed live Teams must cover children created after start_team."""

import asyncio
import subprocess

import pytest
from scheduler_awaiting_test_support import ScriptedSession, build_scheduler, terminal

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.worktree_pool import WorktreePool


async def _scheduler(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "--allow-empty", "-qm", "initial"], cwd=tmp_path, check=True,
    )
    lead = ScriptedSession("lead", [])
    lead.env = LocalEnvironment(str(tmp_path))
    await lead.env.setup()
    child = ScriptedSession("coder", [terminal("done")])
    scheduler, _ = build_scheduler(lead, [child])
    scheduler._worktree_pool = WorktreePool(str(tmp_path), use_worktrees=True)
    build = scheduler._session_factory.build_spawn_session

    def build_with_environment(**kwargs):
        session = build(**kwargs)
        session.env = kwargs["env"]
        return session

    scheduler._session_factory.build_spawn_session = build_with_environment
    return scheduler, child


async def test_dynamic_child_is_checkpointed_before_its_first_turn(tmp_path):
    scheduler, child = await _scheduler(tmp_path)
    await scheduler.create_checkpoint(0)
    observed = []

    async def first_turn(session):
        effect = scheduler.create_effect_ref(
            producer_aid=session.state.aid, kind="tool_result", epoch=0, attempt=0,
        )
        plan = scheduler.preview_rollback({effect.effect_id})
        checkpoint = plan.checkpoint_by_agent[session.state.aid]
        assert checkpoint is not None
        assert checkpoint.workspace_identity == session.env.workspace
        assert checkpoint.environment.digest() == session.env.snapshot_environment().digest()
        assert checkpoint.environment.as_dict()["PWD"] == session.env.workspace
        observed.append(checkpoint)
        return await terminal("done")(session)

    child._steps = [first_turn]
    try:
        aid = await scheduler.spawn(0, "coder", "produce a result")
        await scheduler._tasks[aid]
        assert len(observed) == 1
    finally:
        await scheduler.cleanup()


async def test_child_checkpoint_failure_prevents_driver_and_releases_leases(tmp_path, monkeypatch):
    scheduler, child = await _scheduler(tmp_path)
    await scheduler.create_checkpoint(0)
    before = scheduler._budget_committed()

    async def fail_checkpoint(*args, **kwargs):
        raise RuntimeError("checkpoint storage unavailable")

    monkeypatch.setattr(scheduler._rollback_service, "create_checkpoint", fail_checkpoint)
    try:
        with pytest.raises(RuntimeError, match="checkpoint storage unavailable"):
            await scheduler.spawn(0, "coder", "must not run")
        await asyncio.sleep(0)
        assert child._steps
        assert set(scheduler._sessions) == {0}
        assert scheduler._budget_committed() == before
        assert not scheduler._tasks
        assert not scheduler._startup_tasks
    finally:
        await scheduler.cleanup()


async def test_non_checkpointed_scheduler_keeps_ordinary_spawn_behavior(tmp_path, monkeypatch):
    scheduler, _ = await _scheduler(tmp_path)

    async def forbidden_checkpoint(*args, **kwargs):
        pytest.fail("ordinary scheduler must not opt in to checkpoints")

    monkeypatch.setattr(scheduler, "create_checkpoint", forbidden_checkpoint)
    try:
        aid = await scheduler.spawn(0, "coder", "ordinary task")
        await scheduler._tasks[aid]
    finally:
        await scheduler.cleanup()
