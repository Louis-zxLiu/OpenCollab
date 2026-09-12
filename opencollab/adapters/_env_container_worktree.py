"""A Git worktree carved out inside an already-running task container."""

from __future__ import annotations

import asyncio
import posixpath
import shlex
import uuid
from collections.abc import Callable

from opencollab.adapters._env_base import ExecResult
from opencollab.adapters._env_docker import DockerEnvironment
from opencollab.adapters.git_patch import guarded_staged_diff_command
from opencollab.adapters.git_worktree_evidence import (
    ABSENT_REF_OLD_VALUE,
    parse_own_commits,
    select_diff_base,
    validate_worktree_branch,
)

CONTAINER_GIT_TIMEOUT_SECONDS = 60.0


class ContainerWorktreeEnvironment(DockerEnvironment):
    """One agent's own checkout of the task repository, inside the container.

    The host-side twin, ``WorktreeEnvironment``, makes its worktree on the host
    file system. That is unusable where the repository under test exists only
    inside a container: reaching it from the host would take a bind mount, and a
    mount is exactly what a harness that extracts its patch from a bounded
    container archive cannot allow. This puts the worktree on the far side of
    the same boundary instead -- every Git command runs through ``docker exec``
    -- so the repository never has to be exposed to the host at all.

    Worktrees are placed outside the repository root on purpose. An archive of
    the repository root is what the surrounding harness reads the final patch
    out of, and a worktree nested inside it would arrive there as a directory
    full of files nobody wrote.

    Unlike the host twin this does not refuse a repository root with
    uncommitted changes. There the root is a workspace handed in from outside
    and dirt signals a setup mistake; here it is also an agent's own workspace,
    so refusing would fail a teammate's seat for no reason other than that a
    sibling had started working. The worktree is a clean checkout of the root's
    HEAD either way, which is the isolation this class exists to provide: work
    an agent has not committed is not in anyone else's tree.

    The evidence it reports is the same as the host twin's -- ``get_diff`` plus
    ``diff_base``, ``head_commit``, ``own_commits`` and ``own_commit_count`` --
    because that is the whole surface the scheduler's ``worktree_changes``
    record reads, and a handoff must join across the two the same way in either
    place.
    """

    def __init__(
        self,
        *,
        container_id: str,
        repository_root: str,
        worktree_root: str,
        branch_name: str | None = None,
        command_prefix: Callable[[str], str] | str | None = None,
        timeout_returncode: int = -1,
    ) -> None:
        branch = validate_worktree_branch(
            branch_name or f"opencollab-wt-{uuid.uuid4().hex[:12]}"
        )
        repository_root = _absolute_container_path(repository_root, "repository root")
        worktree_root = _absolute_container_path(worktree_root, "worktree root")
        worktree_dir = posixpath.join(worktree_root, branch)
        super().__init__(
            workspace=worktree_dir,
            container_id=container_id,
            exec_workdir=worktree_dir,
            command_prefix=command_prefix,
            timeout_returncode=timeout_returncode,
        )
        self.source_workspace = repository_root
        self._repository_root = repository_root
        self._worktree_root = worktree_root
        self._worktree_dir = worktree_dir
        self._branch = branch
        self._branch_owned = False
        self._worktree_registered = False
        self._base_commit: str | None = None
        self._diff_base: str | None = None
        self._head_commit: str | None = None
        self._own_commits: tuple[str, ...] = ()
        self._own_commit_count: int | None = None
        self._checkpoint_adapter = None
        self._git_home = f"/tmp/.opencollab-git-{uuid.uuid4().hex}"
        self._git_home_created = False
        self._cleanup_task: asyncio.Task | None = None

    async def checkpoint_scope(self, boundary, *, owner_aid: int, causal_frontier):
        if self._checkpoint_adapter is None:
            from opencollab.adapters.git_checkpoints import GitCheckpointAdapter

            self._checkpoint_adapter = GitCheckpointAdapter(self)
        return await self._checkpoint_adapter.checkpoint_scope(
            boundary,
            owner_aid=owner_aid,
            causal_frontier=causal_frontier,
        )

    async def restore_scope(self, checkpoint):
        if self._checkpoint_adapter is None:
            from opencollab.adapters.git_checkpoints import GitCheckpointAdapter

            self._checkpoint_adapter = GitCheckpointAdapter(self)
        return await self._checkpoint_adapter.restore_scope(checkpoint)

    async def validate_checkpoint_scope(self, checkpoint) -> None:
        from opencollab.adapters.git_checkpoints import GitCheckpointAdapter

        await GitCheckpointAdapter(self).validate_checkpoint_scope(checkpoint)

    @property
    def diff_base(self) -> str | None:
        """The revision the last ``get_diff`` measured against, if any."""
        return self._diff_base

    @property
    def head_commit(self) -> str | None:
        """Where HEAD stood at that same reading, if it could be read."""
        return self._head_commit

    @property
    def own_commits(self) -> tuple[str, ...]:
        """The commits this worktree made since ``diff_base``, newest first."""
        return self._own_commits

    @property
    def own_commit_count(self) -> int | None:
        """How many commits ``own_commits`` was cut from, or ``None``."""
        return self._own_commit_count

    async def _git(
        self,
        workdir: str,
        *args: str,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run one Git command in the container, outside the agent's shell.

        Not ``exec_cmd``: that runs what the agent's session runs, through the
        image's login shell and whatever command prefix the harness set. The
        worktree's own bookkeeping has to be argv-exact and independent of that.
        """
        git_environment = dict(env or {})
        git_environment["HOME"] = self._git_home
        return await self._exec(
            shlex.join(("git", *args)),
            timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
            workdir=workdir,
            environment=git_environment,
            raw=True,
            allow_revoked=asyncio.current_task() is self._cleanup_task,
        )

    async def _checkpoint_git(self, *args: str, env: dict[str, str] | None = None) -> str:
        result = await self._git(self._worktree_dir, *args, env=env)
        if result.returncode != 0 or result.stdout_truncated or result.stderr_truncated:
            detail = result.stderr.strip() or f"git exited with status {result.returncode}"
            raise RuntimeError(detail)
        return result.stdout.rstrip("\r\n")

    async def checkpoint_workspace_identity(self) -> str:
        result = await self._docker(
            "exec",
            "-w",
            self._worktree_dir,
            "--",
            self._container_id or "",
            "pwd",
            "-P",
            timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise RuntimeError("cannot verify container workspace identity")
        identity = result.stdout.decode("utf-8", errors="strict").strip()
        if identity != self._worktree_dir:
            raise RuntimeError("container workspace identity changed")
        return identity

    async def _checkpoint_temp_path(self, prefix: str) -> str:
        await self._bind_attached()
        result = await self._docker(
            "exec",
            "--",
            self._container_id or "",
            "mktemp",
            f"/tmp/{prefix}XXXXXX",
            timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise RuntimeError("cannot create container checkpoint index")
        path = result.stdout.decode("utf-8", errors="strict").strip()
        await self._remove_checkpoint_temp_path(path)
        return path

    async def _remove_checkpoint_temp_path(self, path: str) -> None:
        if self._container_id is None:
            return
        await self._docker(
            "exec",
            "--",
            self._container_id,
            "rm",
            "-f",
            "--",
            path,
            timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
        )

    async def setup(self, mount_dir: str | None = None, *, parent_environment=None) -> str:
        await super().setup(mount_dir)
        self._ensure_active()
        await self._configure_git_safety()
        await self._create_worktree()
        if parent_environment is not None:
            self.replace_environment(parent_environment)
        self.bind_workspace(self._worktree_dir)
        return self._worktree_dir

    async def _configure_git_safety(self) -> None:
        """Trust only this adapter's repository paths in an isolated Git home."""
        await self._bind_attached()
        container_id = self._container_id
        if container_id is None:
            raise RuntimeError("container worktree is not attached to a container")
        made = await self._docker(
            "exec",
            "--",
            container_id,
            "mkdir",
            "-m",
            "700",
            "--",
            self._git_home,
            timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
        )
        if made.returncode != 0:
            raise RuntimeError("cannot create isolated container Git configuration")
        self._git_home_created = True
        for path in (self._repository_root, self._worktree_dir):
            configured = await self._docker(
                "exec",
                "-e",
                f"HOME={self._git_home}",
                "--",
                container_id,
                "git",
                "config",
                "--global",
                "--add",
                "safe.directory",
                path,
                timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
            )
            if configured.returncode != 0:
                raise RuntimeError("cannot configure container Git repository ownership")

    async def _create_worktree(self) -> None:
        made = await self._docker(
            "exec",
            "--",
            self._container_id or "",
            "mkdir",
            "-p",
            "--",
            self._worktree_root,
            timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
        )
        if made.returncode != 0:
            raise RuntimeError("cannot create the container worktree root")
        base = await self._git(self._repository_root, "rev-parse", "--verify", "HEAD^{commit}")
        if base.returncode != 0 or base.stdout_truncated:
            raise RuntimeError("cannot resolve container worktree base commit")
        self._base_commit = base.stdout.strip()
        # The claimed ref is an ownership lease, not the worktree's HEAD: a
        # detached worktree keeps its own commits from moving that lease, so a
        # later external advance is observable and cannot be deleted by us.
        claimed = await self._git(
            self._repository_root,
            "update-ref",
            f"refs/heads/{self._branch}",
            self._base_commit,
            ABSENT_REF_OLD_VALUE,
        )
        if claimed.returncode != 0:
            raise RuntimeError(
                f"cannot atomically claim container worktree branch: {claimed.stderr.strip()}"
            )
        self._branch_owned = True
        added = await self._git(
            self._repository_root,
            "worktree",
            "add",
            "--detach",
            self._worktree_dir,
            self._base_commit,
        )
        if added.returncode != 0:
            raise RuntimeError(f"container git worktree add failed: {added.stderr.strip()}")
        self._worktree_registered = True

    async def get_diff(self) -> str:
        """This worktree's changes since the point its current work started from."""
        self._ensure_active()
        if self._base_commit is None:
            raise RuntimeError("container worktree base commit is unavailable")
        self._diff_base = await self._resolve_diff_base()
        await self._resolve_own_commits(self._diff_base)
        result = await self.exec_cmd(
            guarded_staged_diff_command(base_revision=self._diff_base)
        )
        if result.stdout_truncated or result.stderr_truncated:
            raise RuntimeError(
                f"container worktree diff exceeded capture limit at {self._worktree_dir}"
            )
        if result.returncode != 0:
            detail = result.stderr.strip() or f"git exited with status {result.returncode}"
            raise RuntimeError(
                f"container worktree diff extraction failed: {detail}; "
                f"worktree retained at {self._worktree_dir}"
            )
        return result.stdout

    async def _resolve_diff_base(self) -> str:
        assert self._base_commit is not None
        reflog = await self._git(
            self._worktree_dir, "log", "-g", "--format=%H%x09%gs", "HEAD"
        )
        if reflog.returncode != 0 or reflog.stdout_truncated:
            return self._base_commit
        return select_diff_base(reflog.stdout, fallback=self._base_commit)

    async def _resolve_own_commits(self, base_revision: str) -> None:
        self._head_commit = None
        self._own_commits = ()
        self._own_commit_count = None
        head = await self._git(self._worktree_dir, "rev-parse", "HEAD")
        if head.returncode != 0 or head.stdout_truncated:
            return
        self._head_commit = head.stdout.strip() or None
        listed = await self._git(
            self._worktree_dir, "rev-list", f"{base_revision}..HEAD"
        )
        if listed.returncode != 0 or listed.stdout_truncated:
            return
        self._own_commits, self._own_commit_count = parse_own_commits(listed.stdout)

    async def cleanup(self) -> None:
        """Release owned resources after quiescence; failed steps stay retryable."""
        async with self._lifecycle_lock:
            await self._abort_resources_locked()
            self._cleanup_task = asyncio.current_task()
            try:
                if self._checkpoint_adapter is not None:
                    await self._checkpoint_adapter.discard()
                    self._checkpoint_adapter = None
                if self._worktree_registered:
                    removed = await self._git(
                        self._repository_root, "worktree", "remove", "--force", self._worktree_dir
                    )
                    if removed.returncode != 0:
                        raise RuntimeError("owned container worktree could not be removed")
                    self._worktree_registered = False
                if self._branch_owned and self._base_commit is not None:
                    ref = f"refs/heads/{self._branch}"
                    released = await self._git(
                        self._repository_root, "update-ref", "-d", ref, self._base_commit
                    )
                    if released.returncode != 0:
                        current = await self._git(self._repository_root, "for-each-ref", "--format=%(objectname)", ref)
                        if current.returncode != 0 or current.stdout.strip() == self._base_commit:
                            raise RuntimeError("owned container branch lease could not be released")
                        # An externally advanced ref is no longer ours to delete.
                    self._branch_owned = False
                if self._git_home_created and self._container_id is not None:
                    removed_config = await self._docker(
                        "exec", "--", self._container_id, "rm", "-rf", "--", self._git_home,
                        timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
                    )
                    if removed_config.returncode != 0:
                        raise RuntimeError("owned container Git configuration could not be removed")
                    self._git_home_created = False
                await self._cleanup_attached_resources()
            finally:
                self._cleanup_task = None


def _absolute_container_path(path: str, label: str) -> str:
    if not isinstance(path, str) or not path.startswith("/") or "\0" in path:
        raise ValueError(f"container {label} must be an absolute path without NUL bytes")
    normalized = posixpath.normpath(path)
    if normalized == "/":
        raise ValueError(f"container {label} must not be the container root")
    return normalized


__all__ = ["CONTAINER_GIT_TIMEOUT_SECONDS", "ContainerWorktreeEnvironment"]
