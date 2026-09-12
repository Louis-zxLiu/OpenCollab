"""Resource ownership regressions using real local I/O and deterministic barriers."""

import asyncio
import threading

import pytest
from test_container_worktree_env import _env, _git, _repo, local_docker  # noqa: F401

from opencollab.adapters._env_local import LocalEnvironment


async def test_local_cleanup_discards_only_owned_checkpoint_refs(tmp_path):
    repo = _repo(tmp_path / "repo")
    environment = LocalEnvironment(str(repo))
    cp = await environment.checkpoint_scope("initial", owner_aid=0, causal_frontier=frozenset())
    ref = f"refs/opencollab/checkpoints/{cp.checkpoint_id}"
    foreign = "refs/opencollab/checkpoints/foreign"
    _git(repo, "update-ref", foreign, cp.filesystem_revision)
    before = _git(repo, "count-objects", "-v")
    assert _git(repo, "for-each-ref", "--format=%(refname)", ref) == ref
    await environment.cleanup()
    await environment.cleanup()
    assert _git(repo, "for-each-ref", "--format=%(refname)", ref) == ""
    assert _git(repo, "rev-parse", foreign) == cp.filesystem_revision
    assert _git(repo, "count-objects", "-v") == before  # No GC or object reclamation claim.


async def test_quiescence_waiter_cancellation_keeps_file_owner_tracked(tmp_path, monkeypatch):
    from opencollab.adapters import _env_local as module

    environment = LocalEnvironment(str(tmp_path))
    started, release = threading.Event(), threading.Event()
    original = module.write_regular_bytes_atomic

    def blocked(*args, **kwargs):
        started.set()
        if not release.wait(5):
            raise TimeoutError("test barrier was not released")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "write_regular_bytes_atomic", blocked)
    writer = asyncio.create_task(environment.write_file("value", "late"))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        writer.cancel()
        await asyncio.gather(writer, return_exceptions=True)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(environment.quiesce(), 0.01)
        assert environment._file_operations
        assert all(not task.done() for task in environment._file_operations)
        release.set()
        await environment.quiesce()
        assert (tmp_path / "value").read_text() == "late"
    finally:
        release.set()
        await environment.cleanup()


async def test_local_quiesce_cancels_active_command_and_remains_usable(tmp_path):
    environment = LocalEnvironment(str(tmp_path))
    command = asyncio.create_task(environment.exec_cmd("printf ready > ready; exec sleep 30"))
    try:
        for _ in range(200):
            if (tmp_path / "ready").exists():
                break
            await asyncio.sleep(0.01)
        assert (tmp_path / "ready").exists()
        await environment.quiesce()
        assert command.cancelled()
        assert not environment._command_operations
        assert (await environment.exec_cmd("printf usable")).stdout == "usable"
    finally:
        await environment.cleanup()
        await asyncio.gather(command, return_exceptions=True)


@pytest.mark.usefixtures("local_docker")
async def test_container_cleanup_after_abort_is_retryable(tmp_path, monkeypatch):
    from opencollab.adapters._env_base import ExecResult

    repo = _repo(tmp_path / "repo")
    environment = _env(repo, tmp_path, "cleanup-retry")
    await environment.setup()
    cp = await environment.checkpoint_scope("initial", owner_aid=0, causal_frontier=frozenset())
    original = environment._git

    async def fail_remove(workdir, *args, **kwargs):
        if args[:2] == ("worktree", "remove"):
            return ExecResult(1, "", "injected remove failure")
        return await original(workdir, *args, **kwargs)

    await environment.abort()
    monkeypatch.setattr(environment, "_git", fail_remove)
    with pytest.raises(RuntimeError, match="could not be removed"):
        await environment.cleanup()
    assert environment.revoked and environment._worktree_registered
    with pytest.raises(RuntimeError, match="aborted"):
        await environment.exec_cmd("true")
    monkeypatch.setattr(environment, "_git", original)
    await environment.cleanup()
    await environment.cleanup()
    assert "cleanup-retry" not in _git(repo, "worktree", "list")
    assert not _git(repo, "for-each-ref", "--format=%(refname)", f"refs/opencollab/checkpoints/{cp.checkpoint_id}")


async def test_ignored_magic_paths_do_not_capture_control_files(tmp_path):
    repo = _repo(tmp_path / "repo")
    (repo / ".gitignore").write_text("*\n")
    (repo / "*").write_text("baseline")
    (repo / ".opencollab").mkdir()
    evidence = repo / ".opencollab" / "evidence"
    evidence.write_text("before")
    environment = LocalEnvironment(str(repo))
    try:
        cp = await environment.checkpoint_scope("initial", owner_aid=0, causal_frontier=frozenset())
        evidence.write_text("preserve this update")
        (repo / "*").write_text("changed")
        result = await environment.restore_scope(cp)
        assert result.status == "restored", result.reason
        assert (repo / "*").read_text() == "baseline"
        assert evidence.read_text() == "preserve this update"
    finally:
        await environment.cleanup()
