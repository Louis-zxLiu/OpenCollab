"""Git-backed Scope checkpoints for local and host worktree environments."""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid

from opencollab.adapters._env_process import PROCESS_OUTPUT_CAPTURE_BYTES, run_process
from opencollab.application.ports import CheckpointableEnvironmentPort
from opencollab.domain.rollback import (
    CheckpointBoundary,
    RestoreResult,
    ScopeCheckpoint,
)

_GIT_TIMEOUT = 60.0
_CONTROL_PLANE = ".opencollab"


class GitCheckpointAdapter:
    """Create exact temporary-index commits without changing HEAD or index."""

    def __init__(self, environment: CheckpointableEnvironmentPort) -> None:
        self._environment = environment
        self._sequence = 0
        self._checkpoints: dict[str, ScopeCheckpoint] = {}
        self._refs: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def checkpoint_scope(
        self,
        boundary: CheckpointBoundary,
        *,
        owner_aid: int,
        causal_frontier: frozenset[str],
    ) -> ScopeCheckpoint:
        async with self._lock:
            checkpoint_id = f"cp-{uuid.uuid4().hex}"
            commit, ref = await self._write_snapshot(checkpoint_id, "checkpoint")
            self._sequence += 1
            digest = await self._git("rev-parse", f"{commit}^{{tree}}")
            checkpoint = ScopeCheckpoint(
                checkpoint_id=checkpoint_id,
                owner_aid=owner_aid,
                sequence=self._sequence,
                filesystem_revision=commit,
                environment=self._environment.snapshot_environment(),
                causal_frontier=causal_frontier,
                boundary=boundary,
                workspace_identity=os.path.realpath(self._environment.workspace),
                filesystem_digest=digest,
            )
            self._checkpoints[checkpoint_id] = checkpoint
            self._refs[checkpoint_id] = ref
            return checkpoint

    async def restore_scope(self, checkpoint: ScopeCheckpoint) -> RestoreResult:
        async with self._lock:
            owned = self._checkpoints.get(checkpoint.checkpoint_id)
            if owned != checkpoint:
                return RestoreResult(
                    checkpoint.owner_aid,
                    checkpoint.checkpoint_id,
                    "failed",
                    reason="checkpoint ownership mismatch",
                )
            current_identity = os.path.realpath(self._environment.workspace)
            if current_identity != checkpoint.workspace_identity:
                return RestoreResult(
                    checkpoint.owner_aid,
                    checkpoint.checkpoint_id,
                    "failed",
                    reason="workspace identity changed",
                )
            fd, index_path = tempfile.mkstemp(prefix="opencollab-restore-")
            os.close(fd)
            os.unlink(index_path)
            env = os.environ.copy()
            env["GIT_INDEX_FILE"] = index_path
            try:
                await self._git(
                    "read-tree",
                    "--reset",
                    "-u",
                    checkpoint.filesystem_revision,
                    env=env,
                )
                protected = set(
                    line
                    for line in (
                        await self._git(
                            "ls-tree",
                            "-r",
                            "--name-only",
                            checkpoint.filesystem_revision,
                            env=env,
                        )
                    ).splitlines()
                    if line
                )
                await self._clean_untracked(protected)
                verified = await self._git(
                    "diff",
                    "--quiet",
                    checkpoint.filesystem_revision,
                    "--",
                    env=env,
                )
                if verified != "":
                    return RestoreResult(
                        checkpoint.owner_aid,
                        checkpoint.checkpoint_id,
                        "failed",
                        filesystem_digest=verified,
                        reason="filesystem contents differ after restore",
                    )
                self._environment.replace_environment(checkpoint.environment)
            except (OSError, RuntimeError) as exc:
                return RestoreResult(
                    checkpoint.owner_aid,
                    checkpoint.checkpoint_id,
                    "failed",
                    reason=str(exc)[:500],
                )
            finally:
                try:
                    os.unlink(index_path)
                except FileNotFoundError:
                    pass
            return RestoreResult(
                checkpoint.owner_aid,
                checkpoint.checkpoint_id,
                "restored",
                filesystem_digest=checkpoint.filesystem_digest,
                environment_digest=checkpoint.environment.digest(),
            )

    async def _write_snapshot(self, reference: str, kind: str) -> tuple[str, str]:
        fd, index_path = tempfile.mkstemp(prefix="opencollab-checkpoint-")
        os.close(fd)
        os.unlink(index_path)
        env = os.environ.copy()
        env.update(
            GIT_INDEX_FILE=index_path,
            GIT_AUTHOR_NAME="OpenCollab Checkpoint",
            GIT_AUTHOR_EMAIL="checkpoint@opencollab.invalid",
            GIT_COMMITTER_NAME="OpenCollab Checkpoint",
            GIT_COMMITTER_EMAIL="checkpoint@opencollab.invalid",
        )
        try:
            head = await self._git("rev-parse", "HEAD", env=env)
            await self._git("read-tree", "HEAD", env=env)
            await self._git("add", "-A", env=env)
            await self._git("reset", "-q", "HEAD", "--", _CONTROL_PLANE, env=env)
            await self._stage_ignored(env)
            tree = await self._git("write-tree", env=env)
            commit = await self._git(
                "commit-tree",
                tree,
                "-p",
                head,
                "-m",
                f"OpenCollab {kind} {reference}",
                env=env,
            )
            ref = f"refs/opencollab/checkpoints/{reference}"
            await self._git("update-ref", ref, commit, env=env)
            return commit, ref
        finally:
            try:
                os.unlink(index_path)
            except FileNotFoundError:
                pass

    async def _stage_ignored(self, env: dict[str, str]) -> None:
        output = await self._git(
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            env=env,
        )
        paths = [path for path in output.split("\0") if path and not self._is_control_path(path)]
        if paths:
            await self._git("add", "-f", "--", *paths, env=env)

    async def _clean_untracked(self, protected: set[str]) -> None:
        regular = await self._git(
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        )
        ignored = await self._git(
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
        )
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

    async def _git(self, *args: str, env: dict[str, str] | None = None) -> str:
        result = await run_process(
            ("git", *args),
            shell=False,
            cwd=self._environment.workspace,
            timeout=_GIT_TIMEOUT,
            output_limit=PROCESS_OUTPUT_CAPTURE_BYTES,
            env=env,
        )
        if result.returncode != 0 or result.stdout_dropped_bytes or result.stderr_dropped_bytes:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(detail or f"git exited with status {result.returncode}")
        return result.stdout.decode("utf-8", errors="replace").strip()

    async def discard(self) -> None:
        async with self._lock:
            for checkpoint_id, ref in tuple(self._refs.items()):
                await self._git("update-ref", "-d", ref)
                self._refs.pop(checkpoint_id, None)
                self._checkpoints.pop(checkpoint_id, None)


__all__ = ["GitCheckpointAdapter"]
