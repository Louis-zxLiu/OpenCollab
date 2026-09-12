"""Real Docker checkpoint/restore coverage for ContainerWorktreeEnvironment."""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess

import pytest

from opencollab.adapters._env_container_worktree import ContainerWorktreeEnvironment
from opencollab.adapters._env_docker import DockerEnvironment
from opencollab.adapters.git_checkpoints import GitCheckpointAdapter

pytestmark = pytest.mark.docker


def _git(path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=path, check=True, capture_output=True)


async def test_real_container_worktree_checkpoint_and_restore(tmp_path):
    image = os.environ.get("OPENCOLLAB_ROLLBACK_BENCHMARK_IMAGE")
    if not image:
        pytest.fail("BLOCKED: OPENCOLLAB_ROLLBACK_BENCHMARK_IMAGE is not configured")

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "OpenCollab Test")
    _git(repo, "config", "user.email", "test@opencollab.invalid")
    (repo / ".gitignore").write_text("*.ignored\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("before\n", encoding="utf-8")
    (repo / "deleted.txt").write_text("restore me\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "initial")

    base = DockerEnvironment(image=image, workspace="/repo")
    worktree = None
    try:
        await base.setup(mount_dir=str(repo))
        container_id = base._container_id
        if not container_id:
            pytest.fail("Docker setup did not return a container ID")
        global_config_before = await base.exec_cmd(
            "if test -f /root/.gitconfig; then sha256sum /root/.gitconfig; else printf absent; fi"
        )
        worktree = ContainerWorktreeEnvironment(
            container_id=container_id,
            repository_root="/repo",
            worktree_root="/tmp/opencollab-worktrees",
            branch_name="rollback-real-test",
        )
        await worktree.setup()
        assert (
            await worktree.exec_cmd(
                f"HOME={worktree._git_home} git config --global --get-all safe.directory"
            )
        ).stdout.splitlines() == ["/repo", worktree.workspace]
        assert (
            await worktree.exec_cmd(
                "if test -f /root/.gitconfig; then sha256sum /root/.gitconfig; else printf absent; fi"
            )
        ).stdout == global_config_before.stdout
        worktree.set_environment_variable("ROLLBACK_SCOPE", "before")
        await worktree.write_file("baseline.ignored", "ignored baseline\n")
        checkpoint = await worktree.checkpoint_scope(
            "initial", owner_aid=1, causal_frontier=frozenset()
        )

        await worktree.write_file("tracked.txt", "changed\n")
        await worktree.exec_cmd("rm -- deleted.txt")
        await worktree.write_file("baseline.ignored", "changed ignored baseline\n")
        await worktree.write_file("new.ignored", "ignored\n")
        await worktree.write_file("new.txt", "untracked\n")
        await worktree.write_file("new-staged.txt", "staged after checkpoint\n")
        assert (await worktree.exec_cmd("git add -- new-staged.txt")).returncode == 0
        await worktree.exec_cmd("mkdir -p .opencollab && printf keep > .opencollab/evidence")
        worktree.set_environment_variable("ROLLBACK_SCOPE", "after")

        result = await GitCheckpointAdapter(worktree).restore_scope(checkpoint)

        assert result.status == "restored", result.reason
        assert result.filesystem_digest == checkpoint.filesystem_digest
        assert result.environment_digest == checkpoint.environment.digest()
        assert (await worktree.read_file("tracked.txt")) == "before\n"
        assert (await worktree.read_file("deleted.txt")) == "restore me\n"
        assert (await worktree.read_file("baseline.ignored")) == "ignored baseline\n"
        assert (await worktree.exec_cmd("test ! -e new.ignored")).returncode == 0
        assert (await worktree.exec_cmd("test ! -e new.txt")).returncode == 0
        assert (await worktree.exec_cmd("test ! -e new-staged.txt")).returncode == 0
        assert (await worktree.read_file(".opencollab/evidence")) == "keep"
        assert worktree.snapshot_environment().digest() == checkpoint.environment.digest()
        assert (await worktree.exec_cmd("ln -s /tmp escape")).returncode == 0
        with pytest.raises(ValueError, match="escapes"):
            await worktree.write_file("escape/forbidden.txt", "must not escape")
        with pytest.raises(ValueError, match="escapes"):
            await worktree.read_file("escape/forbidden.txt")
        await worktree.abort()
        assert worktree.revoked
        await worktree.cleanup()
        await worktree.cleanup()
        assert (await base.exec_cmd(f"test ! -e {shlex.quote(worktree.workspace)}")).returncode == 0
        assert (await base.exec_cmd(f"test ! -e {shlex.quote(worktree._git_home)}")).returncode == 0
        refs = await base.exec_cmd("git for-each-ref --format='%(refname)' refs/opencollab/checkpoints")
        assert refs.returncode == 0 and not refs.stdout.strip()
    finally:
        if worktree is not None:
            await worktree.cleanup()
        await base.cleanup()


@pytest.mark.parametrize("command_text", ["sleep 30", "sleep 30; printf late > late"])
async def test_real_cancelled_exec_leaves_docker_scope_usable(tmp_path, command_text):
    image = os.environ.get("OPENCOLLAB_ROLLBACK_BENCHMARK_IMAGE")
    if not image:
        pytest.fail("BLOCKED: OPENCOLLAB_ROLLBACK_BENCHMARK_IMAGE is not configured")

    base = DockerEnvironment(image=image, workspace="/repo")
    try:
        await base.setup()
        user_failure = await base.exec_cmd("printf 'opencollab:quiesce-failed' >&2; exit 125")
        assert user_failure.returncode == 125 and not base.revoked
        command = asyncio.create_task(base.exec_cmd("printf ready > ready; " + command_text))
        for _ in range(100):
            if (await base.exec_cmd("test -f ready")).returncode == 0:
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("active command did not become ready")
        command.cancel()
        with pytest.raises(asyncio.CancelledError):
            await command

        result = await base.exec_cmd("printf usable")
        assert result.returncode == 0
        assert result.stdout == "usable"
        assert not base.revoked
        assert (await base.exec_cmd("test ! -e late")).returncode == 0
        processes = await base.exec_cmd(
            'for s in /proc/[0-9]*/stat; do read -r row < "$s" || continue; '
            '[[ "$row" = *") Z "* ]] && exit 1; done; exit 0'
        )
        assert processes.returncode == 0
    finally:
        await base.cleanup()


async def test_real_container_scheduler_quiesces_then_restores(tmp_path):
    from scheduler_awaiting_test_support import ScriptedSession, build_scheduler

    image = os.environ.get("OPENCOLLAB_ROLLBACK_BENCHMARK_IMAGE")
    if not image:
        pytest.fail("BLOCKED: OPENCOLLAB_ROLLBACK_BENCHMARK_IMAGE is not configured")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "OpenCollab Test")
    _git(repo, "config", "user.email", "test@opencollab.invalid")
    (repo / "tracked").write_text("original", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "initial")
    base = DockerEnvironment(image=image, workspace="/repo")
    worktree = None
    command = None
    try:
        await base.setup(mount_dir=str(repo))
        worktree = ContainerWorktreeEnvironment(
            container_id=base._container_id, repository_root="/repo",
            worktree_root="/tmp/opencollab-worktrees", branch_name="scheduler-rollback",
        )
        await worktree.setup()
        lead = ScriptedSession("lead", [])
        lead.env = worktree
        scheduler, _ = build_scheduler(lead, [])
        await scheduler.create_checkpoint(0)
        effect = scheduler.create_effect_ref(producer_aid=0, kind="tool_result", epoch=0, attempt=0)
        await worktree.write_file("tracked", "changed")
        command = asyncio.create_task(worktree.exec_cmd("sleep 30"))
        await asyncio.sleep(0.2)
        plan = scheduler.preview_rollback({effect.effect_id})
        result = await scheduler.rollback_effect({effect.effect_id}, expected_plan_digest=plan.digest())
        assert result.invalidated
        assert command.cancelled()
        assert not worktree.revoked
        assert scheduler._rollback_fenced == {0}
        assert await worktree.read_file("tracked") == "original"
        scheduler.resume_after_rollback({0})
        with pytest.raises(RuntimeError, match="stale"):
            scheduler.create_effect_ref(producer_aid=0, kind="tool_result", epoch=0, attempt=1)
        assert (await worktree.exec_cmd("printf usable")).stdout == "usable"
    finally:
        if command is not None and not command.done():
            command.cancel()
            await asyncio.gather(command, return_exceptions=True)
        if worktree is not None:
            await worktree.cleanup()
        await base.cleanup()
