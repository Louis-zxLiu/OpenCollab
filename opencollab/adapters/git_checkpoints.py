"""Git-backed Scope checkpoints for local and host worktree environments."""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid

from opencollab.adapters._env_process import PROCESS_OUTPUT_CAPTURE_BYTES, run_process
from opencollab.application.async_timeout import await_owned_operation
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
        self._refs: dict[str, tuple[str, str]] = {}
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
            try:
                commit, _ref = await self._write_snapshot(checkpoint_id, "checkpoint")
                digest = await self._git("rev-parse", f"{commit}^{{tree}}")
                checkpoint = ScopeCheckpoint(
                    checkpoint_id=checkpoint_id,
                    owner_aid=owner_aid,
                    sequence=self._sequence + 1,
                    filesystem_revision=commit,
                    environment=self._environment.snapshot_environment(),
                    causal_frontier=causal_frontier,
                    boundary=boundary,
                    workspace_identity=await self._workspace_identity(),
                    filesystem_digest=digest,
                )
            except BaseException:
                await await_owned_operation(self._discard_reference(checkpoint_id), propagate_cancellation=False)
                raise
            self._sequence += 1
            self._checkpoints[checkpoint_id] = checkpoint
            return checkpoint

    async def restore_scope(self, checkpoint: ScopeCheckpoint) -> RestoreResult:
        async with self._lock:
            try:
                await self._validate_checkpoint(checkpoint)
            except (RuntimeError, ValueError) as exc:
                return RestoreResult(
                    checkpoint.owner_aid,
                    checkpoint.checkpoint_id,
                    "failed",
                    reason=str(exc),
                )
            index_path = await self._create_index_path("opencollab-restore-")
            env = self._process_environment()
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
                            "-z",
                            "--name-only",
                            checkpoint.filesystem_revision,
                            env=env,
                        )
                    ).split("\0")
                    if line
                )
                await self._clean_untracked(protected, env)
                await self._git("add", "-A", env=env)
                tracked_control = await self._git("ls-files", "--", _CONTROL_PLANE, env=env)
                if tracked_control:
                    await self._git(
                        "rm",
                        "-r",
                        "--cached",
                        "--ignore-unmatch",
                        "--",
                        _CONTROL_PLANE,
                        env=env,
                    )
                await self._stage_ignored(env)
                verified = await self._git("write-tree", env=env)
                if verified != checkpoint.filesystem_digest:
                    return RestoreResult(
                        checkpoint.owner_aid,
                        checkpoint.checkpoint_id,
                        "failed",
                        filesystem_digest=verified,
                        reason="filesystem contents differ after restore",
                    )
                self._environment.replace_environment(checkpoint.environment)
                restored_environment = self._environment.snapshot_environment()
                if restored_environment.digest() != checkpoint.environment.digest():
                    return RestoreResult(
                        checkpoint.owner_aid,
                        checkpoint.checkpoint_id,
                        "failed",
                        reason="environment differs after restore",
                    )
                if await self._workspace_identity() != checkpoint.workspace_identity:
                    return RestoreResult(
                        checkpoint.owner_aid,
                        checkpoint.checkpoint_id,
                        "failed",
                        reason="workspace identity changed during restore",
                    )
            except (OSError, RuntimeError) as exc:
                return RestoreResult(
                    checkpoint.owner_aid,
                    checkpoint.checkpoint_id,
                    "failed",
                    reason=str(exc)[:500],
                )
            finally:
                await self._remove_index_path(index_path)
            return RestoreResult(
                checkpoint.owner_aid,
                checkpoint.checkpoint_id,
                "restored",
                filesystem_digest=checkpoint.filesystem_digest,
                environment_digest=checkpoint.environment.digest(),
            )

    async def validate_checkpoint_scope(self, checkpoint: ScopeCheckpoint) -> None:
        async with self._lock:
            await self._validate_checkpoint(checkpoint)

    async def _validate_checkpoint(self, checkpoint: ScopeCheckpoint) -> None:
        if await self._workspace_identity() != checkpoint.workspace_identity:
            raise ValueError("workspace identity changed")
        pwd = checkpoint.environment.as_dict().get("PWD")
        if pwd is not None and pwd != checkpoint.workspace_identity:
            raise ValueError("checkpoint PWD does not match its workspace identity")
        try:
            revision = await self._git("rev-parse", "--verify", self._ref_for(checkpoint.checkpoint_id))
            tree = await self._git("rev-parse", "--verify", f"{revision}^{{tree}}")
        except RuntimeError as exc:
            raise ValueError("checkpoint reference is unavailable") from exc
        if revision != checkpoint.filesystem_revision:
            raise ValueError("checkpoint revision does not match its reference")
        if tree != checkpoint.filesystem_digest:
            raise ValueError("checkpoint filesystem digest does not match its revision")

    async def _write_snapshot(self, reference: str, kind: str) -> tuple[str, str]:
        index_path = await self._create_index_path("opencollab-checkpoint-")
        env = self._process_environment()
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
            tracked_control = await self._git("ls-files", "--", _CONTROL_PLANE, env=env)
            if tracked_control:
                await self._git(
                    "rm",
                    "-r",
                    "--cached",
                    "--ignore-unmatch",
                    "--",
                    _CONTROL_PLANE,
                    env=env,
                )
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
            # Own the exact ref/object before awaiting its creation acknowledgement.
            self._refs[reference] = (ref, commit)
            await self._git("update-ref", ref, commit, env=env)
            return commit, ref
        finally:
            await self._remove_index_path(index_path)

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
            await self._git("add", "-f", "--", *(f":(literal){path}" for path in paths), env=env)

    async def _clean_untracked(self, protected: set[str], env: dict[str, str]) -> None:
        regular = await self._git(
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            env=env,
        )
        ignored = await self._git(
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            env=env,
        )
        paths = [
            path
            for path in {*regular.split("\0"), *ignored.split("\0")}
            if path and path not in protected and not self._is_control_path(path)
        ]
        if paths:
            literal_paths = tuple(f":(literal){path}" for path in paths)
            await self._git("clean", "-fdx", "--", *literal_paths, env=env)

    @staticmethod
    def _is_control_path(path: str) -> bool:
        return path == _CONTROL_PLANE or path.startswith(f"{_CONTROL_PLANE}/")

    @staticmethod
    def _ref_for(checkpoint_id: str) -> str:
        return f"refs/opencollab/checkpoints/{checkpoint_id}"

    async def _git(self, *args: str, env: dict[str, str] | None = None) -> str:
        checkpoint_git = getattr(self._environment, "_checkpoint_git", None)
        if callable(checkpoint_git):
            return await checkpoint_git(*args, env=env)
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
        return result.stdout.decode("utf-8", errors="replace").rstrip("\r\n")

    async def _create_index_path(self, prefix: str) -> str:
        create = getattr(self._environment, "_checkpoint_temp_path", None)
        if callable(create):
            return await create(prefix)
        fd, path = tempfile.mkstemp(prefix=prefix)
        os.close(fd)
        os.unlink(path)
        return path

    async def _remove_index_path(self, path: str) -> None:
        remove = getattr(self._environment, "_remove_checkpoint_temp_path", None)
        if callable(remove):
            await remove(path)
            return
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def _process_environment(self) -> dict[str, str]:
        process_environment = getattr(self._environment, "process_environment", None)
        if callable(process_environment):
            return dict(process_environment())
        return os.environ.copy()

    async def _workspace_identity(self) -> str:
        probe = getattr(self._environment, "checkpoint_workspace_identity", None)
        if callable(probe):
            return await probe()
        return os.path.realpath(self._environment.workspace)

    async def discard(self) -> None:
        async with self._lock:
            for checkpoint_id in tuple(self._refs):
                await self._discard_reference(checkpoint_id)

    async def _discard_reference(self, checkpoint_id: str) -> None:
        owned = self._refs.get(checkpoint_id)
        if owned is None:
            return
        ref, revision = owned
        try:
            await self._git("update-ref", "-d", ref, revision)
        except RuntimeError:
            current = await self._git("for-each-ref", "--format=%(objectname)", ref)
            if current == revision:
                raise
            # The ref was deleted or advanced by another owner.  Do not
            # remove its new value, but stop claiming ownership locally.
        self._refs.pop(checkpoint_id, None)
        self._checkpoints.pop(checkpoint_id, None)


__all__ = ["GitCheckpointAdapter"]
