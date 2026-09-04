from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from opencollab.adapters.env import WorktreeEnvironment
from opencollab.adapters.worktree_pool import WorktreePool
from opencollab.application._scheduler_rollback import SchedulerRollbackMixin
from opencollab.application.rollback import RollbackService
from opencollab.application.scheduler_lifecycle import LifecycleMixin
from opencollab.bootstrap.tool_registry import KNOWN_TOOL_NAMES
from opencollab.domain.pending import RowStatus
from opencollab.domain.rollback import (
    AdoptionResult,
    CheckpointBoundary,
    EffectRef,
    LineageEnvelope,
    RollbackState,
    WorkspaceRevision,
    lineage_envelope_from_dict,
    lineage_envelope_to_dict,
)


def _git(cwd, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(cwd), *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(path):
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.name", "OpenCollab Tests")
    _git(path, "config", "user.email", "tests@opencollab.invalid")
    (path / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-qm", "baseline")
    return path


def test_workspace_revision_round_trips_in_lineage_envelope() -> None:
    effect = EffectRef(
        effect_id="effect",
        producer_aid=9,
        kind="teammate_message",
        epoch=2,
        attempt=3,
        workspace_revision="revision",
        base_workspace_revision="base",
    )
    envelope = LineageEnvelope(effect, consumer_aid=17)

    assert lineage_envelope_from_dict(lineage_envelope_to_dict(envelope)) == envelope


async def test_workspace_effect_freezes_complete_scope_without_moving_head_or_index(
    tmp_path,
) -> None:
    source = _repo(tmp_path / "repo")
    parent = WorktreeEnvironment(str(source), branch_name="effect-parent", require_git=True)
    sibling = WorktreeEnvironment(str(source), branch_name="effect-sibling", require_git=True)
    child = None
    try:
        await parent.setup()
        await sibling.setup()
        spawn_revision = await parent.capture_workspace_revision("spawn", owner_aid=0)
        child = WorktreeEnvironment(
            str(source),
            branch_name="effect-child",
            require_git=True,
            base_revision=spawn_revision.revision,
        )
        await child.setup()

        await child.write_file("committed.cpp", "int committed;\n")
        commit = await child.exec_cmd("git add committed.cpp && git commit -qm child")
        assert commit.returncode == 0
        await child.write_file("tracked.txt", "staged\n")
        stage = await child.exec_cmd("git add tracked.txt")
        assert stage.returncode == 0
        await child.write_file("untracked.txt", "untracked\n")

        head_before = _git(child.workspace, "rev-parse", "HEAD")
        index_before = _git(child.workspace, "diff", "--cached")
        revision = await child.capture_workspace_revision("child-result", owner_aid=2)

        assert revision.changed is True
        assert _git(child.workspace, "rev-parse", "HEAD") == head_before
        assert _git(child.workspace, "diff", "--cached") == index_before

        outcome = await parent.adopt_workspace_revision(revision)

        assert outcome.status == "adopted"
        assert await parent.read_file("committed.cpp") == "int committed;\n"
        assert await parent.read_file("tracked.txt") == "staged\n"
        assert await parent.read_file("untracked.txt") == "untracked\n"
        assert await sibling.read_file("tracked.txt") == "base\n"
        assert not (tmp_path / "repo" / "committed.cpp").exists()
    finally:
        if child is not None:
            await child.cleanup()
        await sibling.cleanup()
        await parent.cleanup()


async def test_no_change_effect_and_conflict_preserve_consumer_scope(tmp_path) -> None:
    source = _repo(tmp_path / "repo")
    producer = WorktreeEnvironment(str(source), branch_name="no-change-producer", require_git=True)
    consumer = WorktreeEnvironment(str(source), branch_name="conflict-consumer", require_git=True)
    try:
        await producer.setup()
        await consumer.setup()
        clean = await producer.capture_workspace_revision("clean", owner_aid=7)
        assert clean.changed is False

        await producer.write_file("effect.txt", "producer\n")
        revision = await producer.capture_workspace_revision("changed", owner_aid=7)
        await consumer.write_file("local.txt", "consumer\n")

        outcome = await consumer.adopt_workspace_revision(revision)

        assert outcome.status == "conflict"
        assert await consumer.read_file("local.txt") == "consumer\n"
        assert not (Path(consumer.workspace) / "effect.txt").exists()
    finally:
        await consumer.cleanup()
        await producer.cleanup()


async def test_child_scope_starts_from_parent_files_and_environment(tmp_path) -> None:
    source = _repo(tmp_path / "repo")
    parent = WorktreeEnvironment(str(source), branch_name="scope-parent", require_git=True)
    pool = WorktreePool(str(source), use_worktrees=True, rollback_enabled=True)
    try:
        await parent.setup()
        parent.set_environment_variable("ROUND_NAME", "final")
        await parent.write_file("adopted.cpp", "int main() {}\n")
        revision = await parent.capture_workspace_revision("spawn", owner_aid=11)

        child = await pool.acquire(
            "arbitrary-role",
            parent_environment=parent,
            parent_workspace_revision=revision,
        )

        assert await child.read_file("adopted.cpp") == "int main() {}\n"
        assert child.environment_view()["ROUND_NAME"] == "final"
        assert child.environment_view()["PWD"] == child.workspace
        child.set_environment_variable("ROUND_NAME", "child")
        assert parent.environment_view()["ROUND_NAME"] == "final"
    finally:
        await pool.release()
        await parent.cleanup()


class _AdoptionEnvironment:
    def __init__(self) -> None:
        self.adopted: list[WorkspaceRevision] = []

    async def adopt_workspace_revision(self, revision: WorkspaceRevision) -> AdoptionResult:
        self.adopted.append(revision)
        return AdoptionResult("adopted", revision=revision.revision)


class _AdoptionHarness(SchedulerRollbackMixin):
    def __init__(self) -> None:
        self._lineage = RollbackService()
        self._rollback_enabled = True
        self.environment = _AdoptionEnvironment()
        self._sessions = {42: SimpleNamespace(env=self.environment)}
        self.traces = []

    def _trace_rollback(self, step_type, payload) -> None:
        self.traces.append((step_type, payload))


async def test_only_effect_consumer_can_adopt_and_quarantine_blocks_adoption() -> None:
    scheduler = _AdoptionHarness()
    effect = scheduler._lineage.create_effect(8, 0, "branch", 0, "child_result", (), "done")
    revision = WorkspaceRevision("revision", "base")
    scheduler._lineage.attach_workspace_revision(effect.effect_id, revision)

    denied = await scheduler.adopt_effect(42, effect.effect_id)
    assert denied == "Error: this Agent has not received the requested Effect."

    scheduler._lineage.register_consumer(effect.effect_id, 42)
    accepted = await scheduler.adopt_effect(42, effect.effect_id)
    assert accepted.startswith("Adopted Effect")
    assert scheduler.environment.adopted == [revision]

    scheduler._lineage.quarantine(effect.effect_id, "invalid")
    quarantined = await scheduler.adopt_effect(42, effect.effect_id)
    assert quarantined == "Error: quarantined Effects cannot be adopted."


class _DeliveryLineage:
    def register_environment(self, aid, environment) -> None:
        pass

    async def checkpoint(self, aid, boundary: CheckpointBoundary, frontier):
        return None


class _DeliveryHarness(SchedulerRollbackMixin, LifecycleMixin):
    def __init__(self) -> None:
        state = SimpleNamespace(rollback=RollbackState())
        env = SimpleNamespace(checkpoint_scope=lambda: None)
        self._spawn_origin = {3: (1, "tool-call")}
        self._sessions = {1: SimpleNamespace(state=state, env=env)}
        self._lineage = _DeliveryLineage()
        self._rollback_enabled = True
        self.wake = None

    async def _create_child_result_effect(self, child_aid, parent_aid, result):
        raise RuntimeError("capture unavailable")

    def _trace_rollback(self, step_type, payload) -> None:
        pass

    async def _wake(self, parent_aid, tool_call_id, result, status, **kwargs) -> None:
        self.wake = (parent_aid, tool_call_id, result, status, kwargs)


async def test_effect_capture_failure_wakes_parent_with_bounded_failure() -> None:
    scheduler = _DeliveryHarness()

    await scheduler._deliver_to_parent(3, "child output", RowStatus.DONE)

    assert scheduler.wake is not None
    assert scheduler.wake[3] is RowStatus.FAILED
    assert scheduler.wake[2] == "Error: workspace Effect delivery failed: capture unavailable"
    assert scheduler.wake[4]["lineage_effect"] is None


def test_transitional_child_adoption_tool_is_removed() -> None:
    assert "adopt_effect" in KNOWN_TOOL_NAMES
    assert "adopt_child_changes" not in KNOWN_TOOL_NAMES
