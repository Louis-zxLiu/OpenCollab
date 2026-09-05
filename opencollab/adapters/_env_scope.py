"""Private Agent Scope state and host-worktree checkpoint capability."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import uuid
from collections.abc import Mapping
from types import MappingProxyType

from opencollab.adapters._env_process import PROCESS_OUTPUT_CAPTURE_BYTES, run_process
from opencollab.adapters._workspace_baseline import changed_entries, validate_baseline
from opencollab.domain.rollback import (
    AdoptionResult,
    CheckpointBoundary,
    EnvironmentSnapshot,
    RestoreResult,
    ScopeCheckpoint,
    WorkspaceBaseline,
    WorkspaceRevision,
)

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONTROL_ENV_NAMES = frozenset({"OPENCOLLAB_ROLLBACK_KEY"})
_MAX_ENV_NAME_BYTES = 256
_MAX_ENV_VALUE_BYTES = 128 * 1024
_GIT_TIMEOUT = 60.0


def _validate_environment_entry(name: str, value: str | None = None) -> None:
    if not isinstance(name, str) or not _ENV_NAME_RE.fullmatch(name):
        raise ValueError("environment variable name is invalid")
    if name in _CONTROL_ENV_NAMES:
        raise ValueError("control-plane environment variables cannot enter an Agent Scope")
    if len(name.encode()) > _MAX_ENV_NAME_BYTES:
        raise ValueError("environment variable name exceeds the size limit")
    if value is not None:
        if not isinstance(value, str) or "\0" in value:
            raise ValueError("environment variable value must be text without NUL bytes")
        if len(value.encode()) > _MAX_ENV_VALUE_BYTES:
            raise ValueError("environment variable value exceeds the size limit")


class _ScopeState:
    """Mutable private owner of one Agent's exact environment mapping."""

    def __init__(self, initial: Mapping[str, str]) -> None:
        self.command_lock = asyncio.Lock()
        self._values = {name: value for name, value in initial.items() if name not in _CONTROL_ENV_NAMES}
        self._native_names = frozenset(self._values)

    def snapshot(self) -> EnvironmentSnapshot:
        return EnvironmentSnapshot.from_mapping(self._values)

    def replace(self, snapshot: EnvironmentSnapshot) -> None:
        self._values = snapshot.as_dict()

    def set(self, name: str, value: str) -> None:
        _validate_environment_entry(name, value)
        self._values[name] = value

    def unset(self, name: str) -> None:
        _validate_environment_entry(name)
        self._values.pop(name, None)

    def view(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._values))

    def process_environment(self) -> dict[str, str]:
        return dict(self._values)

    def tombstones(self) -> tuple[str, ...]:
        return tuple(sorted(self._native_names - self._values.keys()))

    def replace_native(self, initial: Mapping[str, str]) -> None:
        values = {name: value for name, value in initial.items() if name not in _CONTROL_ENV_NAMES}
        self._values = values
        self._native_names = frozenset(values)


class _HostGitCheckpoints:
    """Create protected Git objects without moving an Agent's HEAD or index."""

    def __init__(self, scope: _ScopeState, workspace: str, baseline: WorkspaceBaseline | None = None) -> None:
        self._scope = scope
        self._workspace = workspace
        self._baseline = baseline or WorkspaceBaseline()
        self._sequence = 0
        self._checkpoints: dict[str, ScopeCheckpoint] = {}
        self._refs: dict[str, str] = {}

    async def _git(self, *args: str, env: dict[str, str] | None = None):
        result = await run_process(
            ("git", *args),
            shell=False,
            cwd=self._workspace,
            timeout=_GIT_TIMEOUT,
            output_limit=PROCESS_OUTPUT_CAPTURE_BYTES,
            env=env or self._scope.process_environment(),
        )
        if result.returncode != 0 or result.stdout_dropped_bytes or result.stderr_dropped_bytes:
            detail = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(detail or f"git exited with status {result.returncode}")
        return result.stdout.decode(errors="replace").strip()

    async def create(
        self,
        boundary: CheckpointBoundary,
        *,
        owner_aid: int,
        causal_frontier: frozenset[str],
    ) -> ScopeCheckpoint:
        async with self._scope.command_lock:
            validate_baseline(self._workspace, self._baseline)
            checkpoint_id = f"cp_{uuid.uuid4().hex}"
            fd, index_path = tempfile.mkstemp(prefix="opencollab-index-")
            os.close(fd)
            os.unlink(index_path)
            env = self._scope.process_environment()
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
                await self._stage_ignored_outputs(env)
                tree = await self._git("write-tree", env=env)
                commit = await self._git(
                    "commit-tree", tree, "-p", head, "-m", f"OpenCollab checkpoint {checkpoint_id}", env=env
                )
                ref = f"refs/opencollab/checkpoints/{owner_aid}/{checkpoint_id}"
                await self._git("update-ref", ref, commit, env=env)
            finally:
                try:
                    os.unlink(index_path)
                except FileNotFoundError:
                    pass
            self._sequence += 1
            checkpoint = ScopeCheckpoint(
                checkpoint_id=checkpoint_id,
                owner_aid=owner_aid,
                sequence=self._sequence,
                filesystem_revision=commit,
                environment=self._scope.snapshot(),
                causal_frontier=causal_frontier,
                boundary_kind=boundary.kind,
                boundary_effect_id=boundary.effect_id,
            )
            self._checkpoints[checkpoint_id] = checkpoint
            self._refs[checkpoint_id] = ref
            return checkpoint

    async def restore(self, checkpoint: ScopeCheckpoint) -> RestoreResult:
        owned = self._checkpoints.get(checkpoint.checkpoint_id)
        if owned != checkpoint:
            return RestoreResult(
                checkpoint.owner_aid,
                checkpoint.checkpoint_id,
                "failed",
                reason="checkpoint ownership mismatch",
            )
        async with self._scope.command_lock:
            try:
                validate_baseline(self._workspace, self._baseline)
                changed = await self._git("status", "--porcelain", "-z")
                await self._git("read-tree", "--reset", "-u", checkpoint.filesystem_revision)
                await self._clean_non_baseline_ignored_outputs()
            except (OSError, RuntimeError) as exc:
                return RestoreResult(
                    checkpoint.owner_aid,
                    checkpoint.checkpoint_id,
                    "failed",
                    reason=str(exc)[:500],
                )
            self._scope.replace(checkpoint.environment)
            return RestoreResult(
                checkpoint.owner_aid,
                checkpoint.checkpoint_id,
                "restored",
                files_changed=len([item for item in changed.split("\0") if item]),
            )

    async def capture_revision(
        self,
        reference_id: str,
        *,
        owner_aid: int,
        base_revision: str,
    ) -> WorkspaceRevision:
        """Freeze tracked and untracked Scope files without changing its index."""
        async with self._scope.command_lock:
            validate_baseline(self._workspace, self._baseline)
            fd, index_path = tempfile.mkstemp(prefix="opencollab-effect-index-")
            os.close(fd)
            os.unlink(index_path)
            env = self._scope.process_environment()
            env.update(
                GIT_INDEX_FILE=index_path,
                GIT_AUTHOR_NAME="OpenCollab Effect",
                GIT_AUTHOR_EMAIL="effect@opencollab.invalid",
                GIT_COMMITTER_NAME="OpenCollab Effect",
                GIT_COMMITTER_EMAIL="effect@opencollab.invalid",
            )
            try:
                await self._git("read-tree", "HEAD", env=env)
                await self._git("add", "-A", env=env)
                await self._stage_ignored_outputs(env)
                tree = await self._git("write-tree", env=env)
                base_tree = await self._git("rev-parse", f"{base_revision}^{{tree}}", env=env)
                commit = await self._git(
                    "commit-tree",
                    tree,
                    "-p",
                    base_revision,
                    "-m",
                    f"OpenCollab effect {reference_id}",
                    env=env,
                )
                ref = f"refs/opencollab/effects/{owner_aid}/{reference_id}"
                await self._git("update-ref", ref, commit, env=env)
            finally:
                try:
                    os.unlink(index_path)
                except FileNotFoundError:
                    pass
            self._refs[f"effect_{owner_aid}_{reference_id}"] = ref
            names = await self._git("diff-tree", "--no-commit-id", "--name-only", "-r", commit, env=env)
            files = changed_entries(self._workspace, names.splitlines(), self._baseline)
            return WorkspaceRevision(commit, base_revision, tree != base_tree, files=files)

    async def _stage_ignored_outputs(self, env: dict[str, str]) -> None:
        ignored = await self._git("ls-files", "--others", "--ignored", "--exclude-standard", "-z", env=env)
        baseline_paths = self._baseline.by_path()
        paths = [
            path
            for path in ignored.split("\0")
            if path and path not in baseline_paths and not path == ".opencollab" and not path.startswith(".opencollab/")
        ]
        if paths:
            await self._git("add", "-f", "--", *paths, env=env)

    async def _clean_non_baseline_ignored_outputs(self) -> None:
        ignored = await self._git("ls-files", "--others", "--ignored", "--exclude-standard", "-z")
        baseline_paths = self._baseline.by_path()
        paths = [
            path
            for path in ignored.split("\0")
            if path and path not in baseline_paths and path != ".opencollab" and not path.startswith(".opencollab/")
        ]
        if paths:
            await self._git("clean", "-fd", "--", *paths)

    async def adopt_revision(self, revision: WorkspaceRevision) -> AdoptionResult:
        """Apply an immutable revision only to a clean coordinating Scope."""
        async with self._scope.command_lock:
            if not revision.changed:
                return AdoptionResult("skipped", revision=revision.revision)
            env = self._scope.process_environment()
            env.update(
                GIT_COMMITTER_NAME="OpenCollab Effect",
                GIT_COMMITTER_EMAIL="effect@opencollab.invalid",
            )
            try:
                await self._git("cat-file", "-e", f"{revision.revision}^{{commit}}")
                parent = await self._git("rev-parse", f"{revision.revision}^")
                if parent != revision.base_revision:
                    return AdoptionResult("failed", reason="workspace revision base mismatch")
                status = await self._git("status", "--porcelain", "--untracked-files=all")
                if status:
                    return AdoptionResult("conflict", reason="Scope is not clean")
                await self._git("cherry-pick", "-x", revision.revision, env=env)
                return AdoptionResult("adopted", revision=revision.revision)
            except RuntimeError as exc:
                try:
                    await self._git("cherry-pick", "--abort")
                except RuntimeError:
                    pass
                return AdoptionResult("failed", reason=str(exc)[:500])

    async def discard(self) -> None:
        async with self._scope.command_lock:
            failures: list[Exception] = []
            for checkpoint_id, ref in tuple(self._refs.items()):
                try:
                    await self._git("update-ref", "-d", ref)
                except (OSError, RuntimeError) as exc:
                    failures.append(exc)
                else:
                    self._refs.pop(checkpoint_id, None)
                    self._checkpoints.pop(checkpoint_id, None)
            if failures:
                raise RuntimeError("failed to discard one or more Scope checkpoints") from failures[0]
