"""Workspace containment and command ownership at the container adapter boundary."""

from __future__ import annotations

import asyncio

import pytest
from test_container_worktree_env import _env, _repo, local_docker  # noqa: F401

from opencollab.adapters._env_docker import DockerEnvironment


@pytest.mark.parametrize("path", ["../secret", "sub/../../secret", "/outside/secret", "/workspace2/secret"])
async def test_docker_file_apis_reject_lexical_escape_before_transport(path, monkeypatch):
    environment = DockerEnvironment(container_id="c" * 64, workspace="/workspace")
    environment._attached_bound = True

    async def forbidden(*args, **kwargs):
        pytest.fail("invalid path reached Docker transport")

    monkeypatch.setattr(environment, "_docker", forbidden)
    with pytest.raises(ValueError, match="escapes"):
        await environment.read_file(path)
    with pytest.raises(ValueError, match="escapes"):
        await environment.read_text_range(path, offset=1, limit=2, max_chars=20)
    with pytest.raises(ValueError, match="escapes"):
        await environment.write_file(path, "must not escape")


@pytest.mark.usefixtures("local_docker")
async def test_container_file_apis_reject_symlink_escape(tmp_path):
    repo = _repo(tmp_path / "repo")
    environment = _env(repo, tmp_path, "path-containment")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("unchanged", encoding="utf-8")
    try:
        await environment.setup()
        from pathlib import Path

        Path(environment.workspace, "escape").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="escapes"):
            await environment.read_file("escape/secret")
        with pytest.raises(ValueError, match="escapes"):
            await environment.read_text_range("escape/secret", offset=1, limit=2, max_chars=20)
        with pytest.raises(ValueError, match="escapes"):
            await environment.write_file("escape/new", "outside write")
        assert not (outside / "new").exists()
        assert (outside / "secret").read_text(encoding="utf-8") == "unchanged"
        await environment.write_file("nested/inside", "valid")
        assert await environment.read_file("nested/inside") == "valid"
    finally:
        await environment.cleanup()


@pytest.mark.usefixtures("local_docker")
async def test_container_internal_git_uses_active_exec_tracking(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    environment = _env(repo, tmp_path, "git-quiescence")
    await environment.setup()
    original = environment._docker
    started = asyncio.Event()

    async def blocking_git(*args, **kwargs):
        if args[-1].startswith("git status"):
            started.set()
            await asyncio.Event().wait()
        return await original(*args, **kwargs)

    async def confirmed_cancel(_token):
        return True

    monkeypatch.setattr(environment, "_docker", blocking_git)
    monkeypatch.setattr(environment, "_cancel_inner", confirmed_cancel)
    task = asyncio.create_task(environment._git(environment.workspace, "status"))
    await started.wait()
    assert task in environment._active_execs.values()
    await environment.quiesce()
    assert task.cancelled()
    assert not environment.revoked
    assert not environment._active_execs
    monkeypatch.setattr(environment, "_docker", original)
    await environment.cleanup()
