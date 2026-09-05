"""Correctness tests for explicit effect rollback v1."""

from __future__ import annotations

import os
import subprocess

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.application.rollback import RollbackService
from opencollab.domain.rollback import (
    EnvironmentSnapshot,
    RestoreResult,
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

    result = await service.rollback_effect({e2.effect_id})

    assert result.invalidated is True
    assert result.plan.affected_agent_ids == frozenset({2, 3, 4, 6})
    assert {aid for aid, env in environments.items() if env.restored} == {2, 3, 4, 6}
    assert service.effects[e2.effect_id].status == "invalidated"


async def test_checkpoint_selection_rejects_contaminated_frontier():
    service = RollbackService()
    environment = FakeEnvironment(7)
    service.register_environment(7, environment)
    clean = await service.create_checkpoint(7, causal_frontier=frozenset())
    contaminated = await service.create_checkpoint(7, causal_frontier=frozenset({"bad"}))

    selected = service._checkpoints[7][0]
    assert selected == clean
    assert contaminated.sequence > clean.sequence


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


async def test_scheduler_rollback_is_explicit_and_resume_releases_fence():
    from scheduler_awaiting_test_support import ScriptedSession, build_scheduler

    scheduler, _events = build_scheduler(ScriptedSession("lead", []), [])
    effect = scheduler.create_effect_ref(
        producer_aid=0,
        kind="tool_result",
        epoch=0,
        attempt=0,
    )

    result = await scheduler.rollback_effect({effect.effect_id})

    assert result.plan.affected_agent_ids == frozenset({0})
    assert result.restores[0].status == "skipped"
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
