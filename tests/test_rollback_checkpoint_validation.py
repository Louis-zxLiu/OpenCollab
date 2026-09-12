"""Real Git checkpoint integrity is preflighted before Scheduler mutation."""

from __future__ import annotations

import subprocess
from dataclasses import replace

import pytest
from scheduler_awaiting_test_support import ScriptedSession, build_scheduler

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.git_checkpoints import GitCheckpointAdapter


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.mark.parametrize("tamper", ["ref", "digest", "pwd", "identity"])
async def test_invalid_checkpoint_fails_before_fence_or_filesystem_mutation(tmp_path, tamper):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Audit")
    _git(tmp_path, "config", "user.email", "audit@example.invalid")
    (tmp_path / "tracked").write_text("original", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "initial")
    environment = LocalEnvironment(str(tmp_path))
    lead = ScriptedSession("lead", [])
    lead.env = environment
    scheduler, _ = build_scheduler(lead, [])
    checkpoint = await scheduler.create_checkpoint(0)
    effect = scheduler.create_effect_ref(producer_aid=0, kind="tool_result", epoch=0, attempt=0)
    (tmp_path / "tracked").write_text("keep current contents", encoding="utf-8")
    if tamper == "ref":
        _git(tmp_path, "update-ref", f"refs/opencollab/checkpoints/{checkpoint.checkpoint_id}", "HEAD")
    elif tamper == "digest":
        checkpoint = replace(checkpoint, filesystem_digest="wrong digest")
    elif tamper == "identity":
        checkpoint = replace(checkpoint, workspace_identity=str(tmp_path / "other"))
    else:
        values = checkpoint.environment.as_dict()
        values["PWD"] = str(tmp_path / "other")
        checkpoint = replace(checkpoint, environment=type(checkpoint.environment).from_mapping(values))
    scheduler._rollback_service._checkpoints[0][0] = checkpoint
    plan = scheduler.preview_rollback({effect.effect_id})
    try:
        with pytest.raises(ValueError):
            await scheduler.rollback_effect({effect.effect_id}, expected_plan_digest=plan.digest())
        with pytest.raises(ValueError):
            await scheduler.rollback_to_checkpoint(0, checkpoint.checkpoint_id)
        assert not scheduler._rollback_fenced
        assert not scheduler._rollback_operations
        assert scheduler.current_effect_epoch(0) == 0
        assert (tmp_path / "tracked").read_text(encoding="utf-8") == "keep current contents"
    finally:
        await environment.cleanup()


async def test_checkpoint_restores_names_with_whitespace_and_new_staged_files(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Audit")
    _git(tmp_path, "config", "user.email", "audit@example.invalid")
    original = tmp_path / " leading\nnewline.txt"
    original.write_text("original", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "initial")
    environment = LocalEnvironment(str(tmp_path))
    try:
        checkpoint = await environment.checkpoint_scope("initial", owner_aid=0, causal_frontier=frozenset())
        original.write_text("changed", encoding="utf-8")
        added = tmp_path / "new staged"
        added.write_text("new", encoding="utf-8")
        _git(tmp_path, "add", ".")
        index_before = _git(tmp_path, "write-tree")
        result = await environment.restore_scope(checkpoint)
        assert result.status == "restored", result.reason
        assert original.read_text(encoding="utf-8") == "original"
        assert not added.exists()
        assert _git(tmp_path, "write-tree") == index_before
    finally:
        await environment.cleanup()


async def test_failed_checkpoint_creation_cleans_only_its_owned_ref(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Audit")
    _git(tmp_path, "config", "user.email", "audit@example.invalid")
    _git(tmp_path, "commit", "--allow-empty", "-qm", "initial")
    environment = LocalEnvironment(str(tmp_path))
    adapter = GitCheckpointAdapter(environment)
    first = await adapter.checkpoint_scope("initial", owner_aid=0, causal_frontier=frozenset())

    def fail_snapshot():
        raise RuntimeError("snapshot failure after Git ref creation")

    monkeypatch.setattr(environment, "snapshot_environment", fail_snapshot)
    try:
        with pytest.raises(RuntimeError, match="snapshot failure"):
            await adapter.checkpoint_scope("initial", owner_aid=0, causal_frontier=frozenset())
        assert _git(tmp_path, "for-each-ref", "--format=%(refname)", "refs/opencollab/checkpoints") == (
            f"refs/opencollab/checkpoints/{first.checkpoint_id}"
        )
        assert set(adapter._refs) == {first.checkpoint_id}
    finally:
        await adapter.discard()
        await environment.cleanup()
