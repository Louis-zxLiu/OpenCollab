"""Correctness tests for explicit effect rollback v1."""

from __future__ import annotations

import asyncio
import os
import subprocess

import pytest

from opencollab.adapters._env_container_worktree import ContainerWorktreeEnvironment
from opencollab.adapters._env_docker import DockerEnvironment
from opencollab.adapters._env_worktree import WorktreeEnvironment
from opencollab.adapters.env import LocalEnvironment
from opencollab.application.rollback import RollbackService
from opencollab.domain.rollback import (
    EnvironmentSnapshot,
    RestoreResult,
    RollbackResult,
    ScopeCheckpoint,
)


class FakeEnvironment:
    def __init__(self, aid: int, workspace: str = "/tmp/scope") -> None:
        self.aid = aid
        self.workspace = workspace
        self.environment = EnvironmentSnapshot.from_mapping({"BASE": "1"})
        self.checkpoints: list[ScopeCheckpoint] = []
        self.restored: list[str] = []

    def snapshot_environment(self):
        return self.environment

    def replace_environment(self, snapshot):
        self.environment = snapshot

    async def validate_checkpoint_scope(self, checkpoint):
        assert checkpoint.workspace_identity == self.workspace

    async def checkpoint_scope(self, boundary, *, owner_aid, causal_frontier):
        checkpoint = ScopeCheckpoint(
            checkpoint_id=f"cp-{self.aid}-{len(self.checkpoints)}",
            owner_aid=owner_aid,
            sequence=len(self.checkpoints),
            filesystem_revision=f"revision-{self.aid}-{len(self.checkpoints)}",
            environment=self.environment,
            causal_frontier=causal_frontier,
            boundary=boundary,
            workspace_identity=self.workspace,
            filesystem_digest=f"digest-{self.aid}-{len(self.checkpoints)}",
        )
        self.checkpoints.append(checkpoint)
        return checkpoint

    async def restore_scope(self, checkpoint):
        self.restored.append(checkpoint.checkpoint_id)
        self.environment = checkpoint.environment
        return RestoreResult(
            self.aid,
            checkpoint.checkpoint_id,
            "restored",
            filesystem_digest=checkpoint.filesystem_digest,
            environment_digest=checkpoint.environment.digest(),
        )


class FailingEnvironment(FakeEnvironment):
    async def restore_scope(self, checkpoint):
        self.restored.append(checkpoint.checkpoint_id)
        return RestoreResult(
            self.aid,
            checkpoint.checkpoint_id,
            "failed",
            reason="test restore failure",
        )


class RecoverableEnvironment(FakeEnvironment):
    def __init__(self, aid: int) -> None:
        super().__init__(aid)
        self.fail_restore = True

    async def restore_scope(self, checkpoint):
        if self.fail_restore:
            return RestoreResult(
                self.aid,
                checkpoint.checkpoint_id,
                "failed",
                reason="temporary restore failure",
            )
        return await super().restore_scope(checkpoint)


class RaisingEnvironment(FakeEnvironment):
    async def restore_scope(self, checkpoint):
        raise OSError("restore transport failed")


async def _service_with_graph():
    service = RollbackService()
    environments = {aid: FakeEnvironment(aid) for aid in range(1, 7)}
    for aid, environment in environments.items():
        service.register_environment(aid, environment)
        await service.create_checkpoint(aid, causal_frontier=frozenset())
    e1 = service.create_effect(producer_aid=1, kind="tool_result", epoch=0, attempt=0)
    e2 = service.create_effect(
        producer_aid=2,
        kind="child_result",
        epoch=0,
        attempt=0,
        parent_effect_ids=(e1.effect_id,),
    )
    e3 = service.create_effect(
        producer_aid=3,
        kind="message",
        epoch=0,
        attempt=0,
        parent_effect_ids=(e2.effect_id,),
    )
    e4 = service.create_effect(
        producer_aid=4,
        kind="message",
        epoch=0,
        attempt=0,
        parent_effect_ids=(e2.effect_id,),
    )
    e5 = service.create_effect(
        producer_aid=5,
        kind="message",
        epoch=0,
        attempt=0,
        parent_effect_ids=(e1.effect_id,),
    )
    service.register_consumer(e2.effect_id, 3)
    service.register_consumer(e3.effect_id, 4)
    service.register_consumer(e2.effect_id, 6)
    service.register_consumer(e5.effect_id, 5)
    return service, environments, e1, e2, e3, e4, e5


async def test_preview_rollback_selects_only_target_branch():
    service, _environments, e1, e2, e3, e4, e5 = await _service_with_graph()

    plan = service.preview_rollback({e2.effect_id})

    assert plan.invalidated_effect_ids == frozenset({e2.effect_id, e3.effect_id, e4.effect_id})
    assert plan.affected_agent_ids == frozenset({2, 3, 4, 6})
    assert 1 not in plan.affected_agent_ids
    assert 5 not in plan.affected_agent_ids
    assert e1.effect_id not in plan.invalidated_effect_ids
    assert e5.effect_id not in plan.invalidated_effect_ids


async def test_rollback_restores_selected_agents_and_invalidates_graph():
    service, environments, _e1, e2, _e3, _e4, _e5 = await _service_with_graph()

    plan = service.preview_rollback({e2.effect_id})
    result = await service.rollback_effect(
        {e2.effect_id}, expected_plan_digest=plan.digest()
    )

    assert result.invalidated is True
    assert result.plan.affected_agent_ids == frozenset({2, 3, 4, 6})
    assert {aid for aid, env in environments.items() if env.restored} == {2, 3, 4, 6}
    assert service.effects[e2.effect_id].status == "invalidated"


async def test_failed_restore_keeps_effects_active():
    service = RollbackService()
    good = FakeEnvironment(1)
    bad = FailingEnvironment(2)
    for environment in (good, bad):
        service.register_environment(environment.aid, environment)
        await service.create_checkpoint(environment.aid)
    effect = service.create_effect(producer_aid=2, kind="tool_result", epoch=0, attempt=0)

    plan = service.preview_rollback({effect.effect_id})
    result = await service.rollback_effect(
        {effect.effect_id}, expected_plan_digest=plan.digest()
    )

    assert result.invalidated is False
    assert result.restores[0].status == "failed"
    assert service.effects[effect.effect_id].status == "active"


async def test_restore_exception_is_reported_without_invalidating_effect():
    service = RollbackService()
    environment = RaisingEnvironment(2)
    service.register_environment(2, environment)
    checkpoint = await service.create_checkpoint(2)
    effect = service.create_effect(
        producer_aid=2,
        kind="tool_result",
        epoch=0,
        attempt=0,
    )
    plan = service.preview_rollback({effect.effect_id})

    result = await service.rollback_effect(
        {effect.effect_id}, expected_plan_digest=plan.digest()
    )
    direct = await service.rollback_to_checkpoint(2, checkpoint.checkpoint_id)

    assert result.invalidated is False
    assert result.restores[0].status == "failed"
    assert result.restores[0].reason == "restore transport failed"
    assert direct.status == "failed"
    assert service.effects[effect.effect_id].status == "active"


async def test_plan_digest_is_deterministic_and_detects_changes():
    service, _environments, _e1, e2, _e3, _e4, _e5 = await _service_with_graph()
    first = service.preview_rollback({e2.effect_id})
    second = service.preview_rollback({e2.effect_id})

    assert first.digest() == second.digest()
    with pytest.raises(ValueError, match="plan digest"):
        await service.rollback_effect({e2.effect_id}, expected_plan_digest="stale")


async def test_checkpoint_selection_rejects_contaminated_frontier():
    service = RollbackService()
    environment = FakeEnvironment(7)
    service.register_environment(7, environment)
    clean = await service.create_checkpoint(7, causal_frontier=frozenset())
    contaminated = await service.create_checkpoint(7, causal_frontier=frozenset({"bad"}))

    selected = service._checkpoints[7][0]
    assert selected == clean
    assert contaminated.sequence > clean.sequence


async def test_resume_frontier_is_restored_before_new_effect_creation():
    service = RollbackService()
    environment = FakeEnvironment(9)
    service.register_environment(9, environment)
    checkpoint = await service.create_checkpoint(9, causal_frontier=frozenset())
    effect = service.create_effect(producer_aid=9, kind="tool_result", epoch=0, attempt=0)
    service.register_consumer(effect.effect_id, 9)
    plan = service.preview_rollback({effect.effect_id})

    result = await service.rollback_effect(
        {effect.effect_id}, expected_plan_digest=plan.digest()
    )

    assert result.invalidated is True
    assert service.causal_frontier(9) == checkpoint.causal_frontier
    next_effect = service.create_effect(producer_aid=9, kind="tool_result", epoch=1, attempt=0)
    assert next_effect.parent_effect_ids == ()


async def test_invalidated_effect_cannot_be_consumed_again():
    service = RollbackService()
    environment = FakeEnvironment(11)
    service.register_environment(11, environment)
    await service.create_checkpoint(11)
    effect = service.create_effect(producer_aid=11, kind="tool_result", epoch=0, attempt=0)
    plan = service.preview_rollback({effect.effect_id})
    result = await service.rollback_effect(
        {effect.effect_id}, expected_plan_digest=plan.digest()
    )

    assert result.invalidated is True
    with pytest.raises(ValueError, match="invalidated"):
        service.register_consumer(effect.effect_id, 12)


async def test_workspace_identity_mismatch_is_rejected_before_restore():
    service = RollbackService()
    environment = FakeEnvironment(10)
    service.register_environment(10, environment)
    await service.create_checkpoint(10)
    environment.workspace = "/different-workspace"
    effect = service.create_effect(producer_aid=10, kind="tool_result", epoch=0, attempt=0)
    plan = service.preview_rollback({effect.effect_id})

    with pytest.raises(ValueError, match="workspace identity"):
        await service.rollback_effect(
            {effect.effect_id}, expected_plan_digest=plan.digest()
        )
    assert environment.restored == []


async def test_environment_snapshot_restores_add_modify_delete_without_host_mutation():
    before = os.environ.get("OPENCOLLAB_ROLLBACK_TEST")
    environment = FakeEnvironment(8)
    service = RollbackService()
    service.register_environment(8, environment)
    checkpoint = await service.create_checkpoint(8)
    original = checkpoint.environment
    environment.replace_environment(
        EnvironmentSnapshot.from_mapping({"BASE": "2", "ADDED": "yes"})
    )

    result = await service.rollback_to_checkpoint(8, checkpoint.checkpoint_id)

    assert result.status == "restored", result.reason
    assert environment.snapshot_environment() == original
    assert os.environ.get("OPENCOLLAB_ROLLBACK_TEST") == before


def _git(workspace, *args: str) -> None:
    subprocess.run(("git", *args), cwd=workspace, check=True, capture_output=True)


@pytest.mark.skipif(os.name == "nt", reason="LocalEnvironment requires POSIX dirfd support")
async def test_git_checkpoint_restores_tracked_and_ignored_files(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "OpenCollab Test")
    _git(tmp_path, "config", "user.email", "test@opencollab.invalid")
    (tmp_path / ".gitignore").write_text("*.generated\n", encoding="utf-8")
    (tmp_path / "tracked.txt").write_text("before\n", encoding="utf-8")
    (tmp_path / "baseline.generated").write_text("baseline\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "initial")

    environment = LocalEnvironment(str(tmp_path))
    checkpoint = await environment.checkpoint_scope(
        "initial", owner_aid=1, causal_frontier=frozenset()
    )
    (tmp_path / "tracked.txt").write_text("changed\n", encoding="utf-8")
    (tmp_path / "baseline.generated").write_text("changed baseline\n", encoding="utf-8")
    (tmp_path / "new.generated").write_text("new output\n", encoding="utf-8")
    (tmp_path / "new.txt").write_text("new untracked\n", encoding="utf-8")

    result = await environment.restore_scope(checkpoint)

    assert result.status == "restored", result.reason
    assert (tmp_path / "tracked.txt").read_text(encoding="utf-8") == "before\n"
    assert (tmp_path / "baseline.generated").read_text(encoding="utf-8") == "baseline\n"
    assert not (tmp_path / "new.generated").exists()
    assert not (tmp_path / "new.txt").exists()
    await environment.cleanup()


@pytest.mark.skipif(os.name == "nt", reason="WorktreeEnvironment requires POSIX dirfd support")
async def test_worktree_setup_inherits_parent_scope_and_rebinds_child_pwd(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "OpenCollab Test")
    _git(tmp_path, "config", "user.email", "test@opencollab.invalid")
    (tmp_path / "tracked.txt").write_text("before\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "initial")

    parent = EnvironmentSnapshot.from_mapping(
        {"BASE": "parent", "ADDED": "yes", "PWD": str(tmp_path)}
    )
    environment = WorktreeEnvironment(str(tmp_path), branch_name="inheritance-test")
    workspace = await environment.setup(parent_environment=parent)

    snapshot = environment.snapshot_environment().as_dict()
    assert snapshot["BASE"] == "parent"
    assert snapshot["ADDED"] == "yes"
    assert snapshot["PWD"] == workspace
    assert environment.workspace == workspace
    checkpoint = await environment.checkpoint_scope(
        "initial", owner_aid=1, causal_frontier=frozenset()
    )
    assert checkpoint.workspace_identity == environment.workspace
    await environment.cleanup()


@pytest.mark.skipif(os.name == "nt", reason="LocalEnvironment requires POSIX dirfd support")
async def test_git_checkpoint_restore_works_with_a_new_adapter_instance(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "OpenCollab Test")
    _git(tmp_path, "config", "user.email", "test@opencollab.invalid")
    (tmp_path / "tracked.txt").write_text("before\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "initial")

    environment = LocalEnvironment(str(tmp_path))
    checkpoint = await environment.checkpoint_scope(
        "initial", owner_aid=1, causal_frontier=frozenset()
    )
    (tmp_path / "tracked.txt").write_text("changed\n", encoding="utf-8")

    from opencollab.adapters.git_checkpoints import GitCheckpointAdapter

    result = await GitCheckpointAdapter(environment).restore_scope(checkpoint)

    assert result.status == "restored", result.reason
    assert (tmp_path / "tracked.txt").read_text(encoding="utf-8") == "before\n"
    await environment.cleanup()


@pytest.mark.skipif(os.name == "nt", reason="LocalEnvironment requires POSIX dirfd support")
async def test_git_checkpoint_preserves_opencollab_control_plane(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "OpenCollab Test")
    _git(tmp_path, "config", "user.email", "test@opencollab.invalid")
    (tmp_path / "tracked.txt").write_text("before\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "initial")
    control = tmp_path / ".opencollab"
    control.mkdir()
    (control / "trace.jsonl").write_text("keep\n", encoding="utf-8")

    environment = LocalEnvironment(str(tmp_path))
    checkpoint = await environment.checkpoint_scope(
        "initial", owner_aid=1, causal_frontier=frozenset()
    )
    (control / "trace.jsonl").write_text("updated\n", encoding="utf-8")
    (tmp_path / "tracked.txt").write_text("changed\n", encoding="utf-8")

    result = await environment.restore_scope(checkpoint)

    assert result.status == "restored"
    assert (control / "trace.jsonl").read_text(encoding="utf-8") == "updated\n"
    await environment.cleanup()


@pytest.mark.skipif(os.name == "nt", reason="LocalEnvironment requires POSIX dirfd support")
async def test_git_restore_treats_glob_like_filenames_as_literal_paths(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "OpenCollab Test")
    _git(tmp_path, "config", "user.email", "test@opencollab.invalid")
    (tmp_path / "tracked.txt").write_text("before\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "initial")
    control = tmp_path / ".opencollab"
    control.mkdir()
    (control / "evidence").write_text("keep\n", encoding="utf-8")

    environment = LocalEnvironment(str(tmp_path))
    checkpoint = await environment.checkpoint_scope(
        "initial", owner_aid=1, causal_frontier=frozenset()
    )
    names = ("*", "?", "[]", ":(glob)*", ":(exclude)tracked.txt")
    for name in names:
        (tmp_path / name).write_text("remove\n", encoding="utf-8")

    result = await environment.restore_scope(checkpoint)

    assert result.status == "restored", result.reason
    assert (control / "evidence").read_text(encoding="utf-8") == "keep\n"
    assert all(not (tmp_path / name).exists() for name in names)
    await environment.cleanup()


@pytest.mark.skipif(os.name == "nt", reason="LocalEnvironment requires POSIX dirfd support")
async def test_local_quiesce_waits_for_cancelled_file_owner(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "OpenCollab Test")
    _git(tmp_path, "config", "user.email", "test@opencollab.invalid")
    (tmp_path / "tracked.txt").write_text("before\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "initial")
    environment = LocalEnvironment(str(tmp_path))
    checkpoint = await environment.checkpoint_scope(
        "initial", owner_aid=1, causal_frontier=frozenset()
    )
    started = asyncio.Event()
    release = asyncio.Event()
    loop = asyncio.get_running_loop()
    original = __import__(
        "opencollab.adapters._env_local", fromlist=["write_regular_bytes_atomic"]
    ).write_regular_bytes_atomic

    def blocked(*args, **kwargs):
        loop.call_soon_threadsafe(started.set)
        while not release.is_set():
            import time

            time.sleep(0.01)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "opencollab.adapters._env_local.write_regular_bytes_atomic", blocked
    )
    writer = asyncio.create_task(environment.write_file("tracked.txt", "late\n"))
    await started.wait()
    writer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writer

    waiting = asyncio.create_task(environment.quiesce())
    await asyncio.sleep(0)
    assert not waiting.done()
    release.set()
    await waiting
    assert (tmp_path / "tracked.txt").read_text(encoding="utf-8") == "late\n"
    await environment.write_file("tracked.txt", "after\n")
    result = await environment.restore_scope(checkpoint)
    assert result.status == "restored"
    assert (tmp_path / "tracked.txt").read_text(encoding="utf-8") == "before\n"
    await environment.cleanup()


async def test_producer_effect_enters_default_checkpoint_frontier():
    service = RollbackService()
    environment = FakeEnvironment(12)
    service.register_environment(12, environment)
    initial = await service.create_checkpoint(12)
    effect = service.create_effect(
        producer_aid=12, kind="tool_result", epoch=0, attempt=0
    )
    later = await service.create_checkpoint(12, boundary="effect")

    assert initial.causal_frontier == frozenset()
    assert later.causal_frontier == frozenset({effect.effect_id})
    plan = service.preview_rollback({effect.effect_id})
    assert plan.checkpoint_by_agent[12] == initial


async def test_later_rollback_does_not_select_checkpoint_from_invalidated_branch():
    service = RollbackService()
    environment = FakeEnvironment(13)
    service.register_environment(13, environment)
    initial = await service.create_checkpoint(13)
    first = service.create_effect(
        producer_aid=13, kind="tool_result", epoch=0, attempt=0
    )
    branch_checkpoint = await service.create_checkpoint(13, boundary="effect")
    first_plan = service.preview_rollback({first.effect_id})
    first_result = await service.rollback_effect(
        {first.effect_id}, expected_plan_digest=first_plan.digest()
    )
    assert first_result.invalidated

    second = service.create_effect(
        producer_aid=13, kind="tool_result", epoch=1, attempt=0
    )
    second_plan = service.preview_rollback({second.effect_id})

    assert second_plan.checkpoint_by_agent[13] == initial
    assert second_plan.checkpoint_by_agent[13] != branch_checkpoint


async def test_scheduler_rollback_is_explicit_and_resume_releases_fence():
    from scheduler_awaiting_test_support import ScriptedSession, build_scheduler

    scheduler, _events = build_scheduler(ScriptedSession("lead", []), [])
    environment = FakeEnvironment(0)
    scheduler.register_effect_environment(0, environment)
    await scheduler._rollback_service.create_checkpoint(0)
    effect = scheduler.create_effect_ref(
        producer_aid=0,
        kind="tool_result",
        epoch=0,
        attempt=0,
    )

    plan = scheduler.preview_rollback({effect.effect_id})
    result = await scheduler.rollback_effect(
        {effect.effect_id}, expected_plan_digest=plan.digest()
    )

    assert result.plan.affected_agent_ids == frozenset({0})
    assert result.restores[0].status == "restored"
    with pytest.raises(RuntimeError, match="fenced"):
        scheduler.create_effect_ref(
            producer_aid=0,
            kind="tool_result",
            epoch=1,
            attempt=1,
        )
    scheduler.resume_after_rollback({0})
    scheduler.create_effect_ref(
        producer_aid=0,
        kind="tool_result",
        epoch=1,
        attempt=1,
    )


async def test_scheduler_rejects_missing_checkpoint_before_fencing():
    from scheduler_awaiting_test_support import ScriptedSession, build_scheduler

    scheduler, _events = build_scheduler(ScriptedSession("lead", []), [])
    scheduler.register_effect_environment(0, FakeEnvironment(0))
    effect = scheduler.create_effect_ref(
        producer_aid=0,
        kind="tool_result",
        epoch=0,
        attempt=0,
    )
    plan = scheduler.preview_rollback({effect.effect_id})
    with pytest.raises(ValueError, match="checkpoint"):
        await scheduler.rollback_effect(
            {effect.effect_id}, expected_plan_digest=plan.digest()
        )
    scheduler.create_effect_ref(
        producer_aid=0,
        kind="tool_result",
        epoch=0,
        attempt=1,
    )


async def test_scheduler_rejects_resume_for_unaffected_agent():
    from scheduler_awaiting_test_support import ScriptedSession, build_scheduler

    scheduler, _events = build_scheduler(ScriptedSession("lead", []), [])
    with pytest.raises(ValueError, match="not affected"):
        scheduler.resume_after_rollback({99})


async def test_scheduler_settles_pending_rows_after_tool_phase_cancellation():
    from scheduler_awaiting_test_support import ScriptedSession, build_scheduler

    from opencollab.domain.pending import PendingRow, RowKind, RowStatus
    from opencollab.domain.session import SessionPhase

    scheduler, _events = build_scheduler(ScriptedSession("lead", []), [])
    session = scheduler._sessions[0]
    session.state.set_phase(SessionPhase.EXECUTING_TOOLS)
    session.state.pending_events.add(
        PendingRow(
            tool_call_id="tool-1",
            kind=RowKind.CHILD_AGENT,
            order=0,
            status=RowStatus.PENDING,
        )
    )
    environment = FakeEnvironment(0)
    scheduler.register_effect_environment(0, environment)
    await scheduler._rollback_service.create_checkpoint(0)
    effect = scheduler.create_effect_ref(
        producer_aid=0,
        kind="tool_result",
        epoch=0,
        attempt=0,
    )
    plan = scheduler.preview_rollback({effect.effect_id})

    result = await scheduler.rollback_effect(
        {effect.effect_id}, expected_plan_digest=plan.digest()
    )

    assert result.invalidated is True
    assert session.state.pending_events.is_empty()
    assert session.state.phase is SessionPhase.IDLE
    with pytest.raises(RuntimeError, match="fenced"):
        scheduler.create_effect_ref(
            producer_aid=0,
            kind="tool_result",
            epoch=0,
            attempt=1,
        )


async def test_scheduler_failed_restore_keeps_fence_until_explicit_retry():
    from scheduler_awaiting_test_support import ScriptedSession, build_scheduler

    scheduler, _events = build_scheduler(ScriptedSession("lead", []), [])
    environment = FailingEnvironment(0)
    scheduler.register_effect_environment(0, environment)
    await scheduler._rollback_service.create_checkpoint(0)
    effect = scheduler.create_effect_ref(
        producer_aid=0,
        kind="tool_result",
        epoch=0,
        attempt=0,
    )
    plan = scheduler.preview_rollback({effect.effect_id})

    result = await scheduler.rollback_effect(
        {effect.effect_id}, expected_plan_digest=plan.digest()
    )

    assert result.invalidated is False
    with pytest.raises(RuntimeError, match="failed restore"):
        scheduler.resume_after_rollback({0})
    with pytest.raises(RuntimeError, match="fenced"):
        scheduler.create_effect_ref(
            producer_aid=0,
            kind="tool_result",
            epoch=0,
            attempt=1,
        )


async def test_scheduler_failed_restore_can_be_retried_then_resumed_explicitly():
    from scheduler_awaiting_test_support import ScriptedSession, build_scheduler

    scheduler, _events = build_scheduler(ScriptedSession("lead", []), [])
    environment = RecoverableEnvironment(0)
    scheduler.register_effect_environment(0, environment)
    await scheduler._rollback_service.create_checkpoint(0)
    effect = scheduler.create_effect_ref(
        producer_aid=0,
        kind="tool_result",
        epoch=0,
        attempt=0,
    )

    first_plan = scheduler.preview_rollback({effect.effect_id})
    first = await scheduler.rollback_effect(
        {effect.effect_id}, expected_plan_digest=first_plan.digest()
    )
    assert first.invalidated is False
    assert scheduler.current_effect_epoch(0) == 0

    environment.fail_restore = False
    retry_plan = scheduler.preview_rollback({effect.effect_id})
    retry = await scheduler.rollback_effect(
        {effect.effect_id}, expected_plan_digest=retry_plan.digest()
    )
    assert retry.invalidated is True
    scheduler.resume_after_rollback({0})
    assert scheduler.current_effect_epoch(0) == 1
    with pytest.raises(RuntimeError, match="stale"):
        scheduler.create_effect_ref(
            producer_aid=0,
            kind="tool_result",
            epoch=0,
            attempt=1,
        )
    scheduler.create_effect_ref(
        producer_aid=0,
        kind="tool_result",
        epoch=1,
        attempt=1,
    )


async def test_scheduler_serializes_rollback_operations():
    from scheduler_awaiting_test_support import ScriptedSession, build_scheduler

    scheduler, _events = build_scheduler(ScriptedSession("lead", []), [])
    environment = FakeEnvironment(0)
    scheduler.register_effect_environment(0, environment)
    await scheduler._rollback_service.create_checkpoint(0)
    effect = scheduler.create_effect_ref(
        producer_aid=0,
        kind="tool_result",
        epoch=0,
        attempt=0,
    )
    plan = scheduler.preview_rollback({effect.effect_id})

    first, second = await asyncio.gather(
        scheduler.rollback_effect({effect.effect_id}, expected_plan_digest=plan.digest()),
        scheduler.rollback_effect({effect.effect_id}, expected_plan_digest=plan.digest()),
        return_exceptions=True,
    )

    outcomes = (first, second)
    assert sum(isinstance(item, RollbackResult) for item in outcomes) == 1
    errors = [item for item in outcomes if isinstance(item, Exception)]
    assert len(errors) == 1
    assert "invalidated" in str(errors[0])


def test_scheduler_rejects_stale_turn_context_for_message_operations():
    from scheduler_awaiting_test_support import ScriptedSession, build_scheduler

    scheduler, _events = build_scheduler(ScriptedSession("lead", []), [])
    scheduler.register_effect_environment(0, FakeEnvironment(0))
    token = scheduler._rollback_turn_epoch.set((0, 0))
    scheduler._rollback_epochs[0] = 1
    try:
        with pytest.raises(RuntimeError, match="stale"):
            scheduler._assert_agent_active(0)
    finally:
        scheduler._rollback_turn_epoch.reset(token)


def test_docker_exec_uses_scope_environment_without_exposing_values_to_trace():
    environment = DockerEnvironment(
        image="python:3.11-slim",
        container_id="a" * 64,
        exec_workdir="/workspace",
    )
    environment.set_environment_variable("ROLLBACK_PROBE", "scope-value")

    argv = environment._exec_argv("printf ok", "token")

    assert "-e" in argv
    assert "ROLLBACK_PROBE=scope-value" in argv
    assert environment.snapshot_environment().as_dict()["ROLLBACK_PROBE"] == "scope-value"


def test_container_worktree_exposes_checkpoint_contract():
    from opencollab.application.ports import CheckpointableEnvironmentPort

    environment = ContainerWorktreeEnvironment(
        container_id="a" * 64,
        repository_root="/repo",
        worktree_root="/worktrees",
    )

    assert isinstance(environment, CheckpointableEnvironmentPort)
