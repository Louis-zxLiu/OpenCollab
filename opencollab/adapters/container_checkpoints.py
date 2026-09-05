"""Git-backed Scope checkpoints executed entirely inside a task container."""

from __future__ import annotations

import shlex
import uuid

from opencollab.domain.rollback import (
    CheckpointBoundary,
    RestoreResult,
    ScopeCheckpoint,
)

_CONTROL_PLANE = ".opencollab"


class ContainerGitCheckpointAdapter:
    """Mirror the host checkpoint contract through ContainerWorktree Git calls."""

    def __init__(self, environment) -> None:
        self._environment = environment
        self._sequence = 0
        self._checkpoints: dict[str, ScopeCheckpoint] = {}
        self._lock = None

    async def checkpoint_scope(
        self,
        boundary: CheckpointBoundary,
        *,
        owner_aid: int,
        causal_frontier: frozenset[str],
    ) -> ScopeCheckpoint:
        import asyncio

        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            checkpoint_id = f"cp-{uuid.uuid4().hex}"
            index_path = await self._make_temp_index()
            try:
                head = await self._indexed(index_path, "rev-parse", "HEAD")
                await self._indexed(index_path, "read-tree", "HEAD")
                await self._indexed(index_path, "add", "-A")
                await self._indexed(index_path, "reset", "-q", "HEAD", "--", _CONTROL_PLANE)
                await self._stage_ignored(index_path)
                tree = await self._indexed(index_path, "write-tree")
                commit = await self._indexed(
                    index_path,
                    "commit-tree",
                    tree,
                    "-p",
                    head,
                    "-m",
                    f"OpenCollab checkpoint {checkpoint_id}",
                )
                await self._git("update-ref", f"refs/opencollab/checkpoints/{checkpoint_id}", commit)
                digest = await self._indexed(index_path, "rev-parse", f"{commit}^{{tree}}")
            finally:
                await self._environment.exec_cmd(f"rm -f -- {shlex.quote(index_path)}")
            self._sequence += 1
            checkpoint = ScopeCheckpoint(
                checkpoint_id=checkpoint_id,
                owner_aid=owner_aid,
                sequence=self._sequence,
                filesystem_revision=commit,
                environment=self._environment.snapshot_environment(),
                causal_frontier=causal_frontier,
                boundary=boundary,
                workspace_identity=self._environment.workspace,
                filesystem_digest=digest,
            )
            self._checkpoints[checkpoint_id] = checkpoint
            return checkpoint

    async def restore_scope(self, checkpoint: ScopeCheckpoint) -> RestoreResult:
        import asyncio

        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._checkpoints.get(checkpoint.checkpoint_id) != checkpoint:
                return RestoreResult(
                    checkpoint.owner_aid,
                    checkpoint.checkpoint_id,
                    "failed",
                    reason="checkpoint ownership mismatch",
                )
            if self._environment.workspace != checkpoint.workspace_identity:
                return RestoreResult(
                    checkpoint.owner_aid,
                    checkpoint.checkpoint_id,
                    "failed",
                    reason="workspace identity changed",
                )
            index_path = await self._make_temp_index()
            try:
                await self._indexed(
                    index_path,
                    "read-tree",
                    "--reset",
                    "-u",
                    checkpoint.filesystem_revision,
                )
                protected = set(
                    line
                    for line in (
                        await self._indexed(
                            index_path,
                            "ls-tree",
                            "-r",
                            "--name-only",
                            checkpoint.filesystem_revision,
                        )
                    ).splitlines()
                    if line
                )
                await self._clean_untracked(protected)
                await self._indexed(index_path, "diff", "--quiet", checkpoint.filesystem_revision, "--")
                self._environment.replace_environment(checkpoint.environment)
            except (OSError, RuntimeError) as exc:
                return RestoreResult(
                    checkpoint.owner_aid,
                    checkpoint.checkpoint_id,
                    "failed",
                    reason=str(exc)[:500],
                )
            finally:
                await self._environment.exec_cmd(f"rm -f -- {shlex.quote(index_path)}")
            return RestoreResult(
                checkpoint.owner_aid,
                checkpoint.checkpoint_id,
                "restored",
                filesystem_digest=checkpoint.filesystem_digest,
                environment_digest=checkpoint.environment.digest(),
            )

    async def _make_temp_index(self) -> str:
        result = await self._environment.exec_cmd("mktemp /tmp/opencollab-checkpoint-XXXXXX")
        if result.returncode != 0 or result.stdout_truncated:
            raise RuntimeError("cannot create container checkpoint index")
        return result.stdout.strip()

    async def _stage_ignored(self, index_path: str) -> None:
        output = await self._indexed(
            index_path,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
        )
        paths = [path for path in output.split("\0") if path and not self._is_control_path(path)]
        if paths:
            await self._indexed(index_path, "add", "-f", "--", *paths)

    async def _clean_untracked(self, protected: set[str]) -> None:
        regular = await self._git("ls-files", "--others", "--exclude-standard", "-z")
        ignored = await self._git("ls-files", "--others", "--ignored", "--exclude-standard", "-z")
        paths = [
            path
            for path in {*regular.split("\0"), *ignored.split("\0")}
            if path and path not in protected and not self._is_control_path(path)
        ]
        if paths:
            await self._git("clean", "-fdx", "--", *paths)

    @staticmethod
    def _is_control_path(path: str) -> bool:
        return path == _CONTROL_PLANE or path.startswith(f"{_CONTROL_PLANE}/")

    async def _indexed(self, index_path: str, *args: str) -> str:
        command = self._git_command(index_path, args)
        result = await self._environment.exec_cmd(command)
        if result.returncode != 0 or result.stdout_truncated or result.stderr_truncated:
            detail = result.stderr.strip()
            raise RuntimeError(detail or f"git exited with status {result.returncode}")
        return result.stdout.strip()

    async def _git(self, *args: str) -> str:
        result = await self._environment._git(self._environment.workspace, *args)
        if result.returncode != 0 or result.stdout_truncated or result.stderr_truncated:
            detail = result.stderr.strip()
            raise RuntimeError(detail or f"git exited with status {result.returncode}")
        return result.stdout.strip()

    def _git_command(self, index_path: str, args: tuple[str, ...]) -> str:
        workspace = shlex.quote(self._environment.workspace)
        index = shlex.quote(index_path)
        quoted_args = " ".join(shlex.quote(arg) for arg in args)
        return f"cd -- {workspace} && GIT_INDEX_FILE={index} git -c safe.directory={workspace} {quoted_args}"


__all__ = ["ContainerGitCheckpointAdapter"]
