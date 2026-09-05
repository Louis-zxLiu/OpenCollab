"""Lifecycle for per-spawn WorktreeEnvironments.

A spawned agent running in parallel needs its own physical workspace so it
cannot corrupt a sibling's edits. WorktreePool encapsulates: create a worktree
env for a given role, remember it for cleanup, tear them all down at the end.
"""

from __future__ import annotations

import logging
import uuid

from opencollab.adapters.env import (
    ContainerWorktreeEnvironment,
    DockerEnvironment,
    Environment,
    LocalEnvironment,
    WorktreeEnvironment,
)
from opencollab.application.async_timeout import await_owned_operation
from opencollab.application.exception_notes import add_exception_note
from opencollab.domain.identity import role_storage_slug, validate_role_identity
from opencollab.domain.rollback import WorkspaceRevision

logger = logging.getLogger(__name__)

# Where per-agent checkouts go inside a container. Deliberately not under the
# repository root: an archive of that root is what a harness reads the run's
# result out of, and a worktree nested in it would arrive there as a directory
# full of files no agent wrote.
CONTAINER_WORKTREE_ROOT = "/opencollab-worktrees"


async def _finish_cleanup(operation):
    return await await_owned_operation(operation, propagate_cancellation=True)


class WorktreePool:
    """Lends out worktree-isolated environments and tracks them for cleanup.

    When use_worktrees is False, hands out LocalEnvironment(workspace) instead
    — caller code does not need to branch on the mode.

    ``base_environment`` says where the repository being worked on actually
    lives. Left unset, it is the host file system and a worktree is made there.
    Set to a container this run is attached to, every worktree is carved out
    inside that container instead, because a repository that exists only there
    cannot be reached from the host without a mount — and a harness that reads
    its result out of a bounded container archive is precisely one that cannot
    allow a mount. Either way the agents get the same isolation and report the
    same evidence; only the side of the wall changes.
    """

    def __init__(
        self,
        workspace: str,
        *,
        use_worktrees: bool,
        base_environment: Environment | None = None,
        rollback_enabled: bool = False,
    ):
        self._workspace = workspace
        self._use_worktrees = use_worktrees
        self._base_environment = base_environment
        self._rollback_enabled = bool(rollback_enabled)
        self._envs: list[Environment] = []

    async def acquire(
        self,
        role: str,
        *,
        parent_environment: Environment | None = None,
        parent_workspace_revision: WorkspaceRevision | None = None,
    ) -> Environment:
        """Create (and remember) an isolated env for a spawned agent of this role."""
        role = validate_role_identity(role)
        base = self._base_environment
        if self._rollback_enabled and not self._use_worktrees:
            raise RuntimeError("rollback-enabled Team mode requires isolated Git worktrees")
        baseline = None
        seed = None
        if parent_environment is not None:
            capture_baseline = getattr(parent_environment, "capture_workspace_baseline", None)
            if callable(capture_baseline):
                baseline = await capture_baseline()
            seed = parent_environment.snapshot_environment()
        if not self._use_worktrees:
            # Without isolation every agent works where the run works, which is
            # the supplied environment when there is one and the host workspace
            # otherwise. The shared environment is not tracked for cleanup here:
            # it belongs to whoever handed it in.
            if base is not None:
                return base
            try:
                env = LocalEnvironment(self._workspace)
                if seed is not None:
                    env.replace_environment(seed)
            except TypeError as exc:
                if "_scope" not in str(exc):
                    raise
                env = LocalEnvironment(self._workspace)
            self._envs.append(env)
            return env

        branch = f"opencollab-{role_storage_slug(role)}-{uuid.uuid4().hex[:8]}"
        env = self._build_worktree(
            branch,
            base,
            baseline=baseline,
            base_revision=(parent_workspace_revision.revision if parent_workspace_revision is not None else None),
        )
        try:
            await env.setup()
            seed_baseline = getattr(env, "seed_workspace_baseline", None)
            if baseline is not None and callable(seed_baseline) and not getattr(env, "_baseline_supplied", False):
                await seed_baseline(baseline)
            seed = parent_environment.snapshot_environment() if parent_environment is not None else None
            if seed is not None:
                # Container setup captures its native environment. A child
                # Scope is nevertheless forked from its parent, so restore the
                # captured seed after setup while retaining the child PWD.
                previous_pwd = env.environment_view().get("PWD")
                env.replace_environment(seed)
                if previous_pwd is not None:
                    env.bind_workspace(previous_pwd)
        except BaseException as original:
            try:
                await _finish_cleanup(env.cleanup())
            except BaseException as cleanup_exc:
                self._envs.append(env)
                logger.warning("partial worktree cleanup failed", exc_info=True)
                add_exception_note(
                    original,
                    f"partial worktree retained for cleanup retry: {type(cleanup_exc).__name__}: {cleanup_exc}",
                )
            raise original
        self._envs.append(env)
        return env

    def _build_worktree(
        self,
        branch: str,
        base: Environment | None,
        *,
        baseline=None,
        base_revision: str | None = None,
    ) -> Environment:
        """An isolated view of wherever the run's repository actually is."""
        if base is None:
            return WorktreeEnvironment(
                self._workspace,
                branch_name=branch,
                require_git=self._rollback_enabled,
                base_revision=base_revision,
                baseline=baseline,
            )
        if isinstance(base, DockerEnvironment) and base.container_reference is not None:
            return ContainerWorktreeEnvironment(
                container_id=base.container_reference,
                repository_root=base.workspace,
                worktree_root=CONTAINER_WORKTREE_ROOT,
                branch_name=branch,
                command_prefix=base.command_prefix,
                base_revision=base_revision,
                baseline=baseline,
            )
        if getattr(base, "local_filesystem", False):
            return WorktreeEnvironment(
                base.workspace,
                branch_name=branch,
                require_git=self._rollback_enabled,
                base_revision=base_revision,
                baseline=baseline,
            )
        raise TypeError(f"worktree isolation is not available for this environment: {type(base).__name__}")

    def track(self, env: Environment) -> None:
        """Track a lazily initialized environment (used for the Lead Scope)."""
        if env not in self._envs:
            self._envs.append(env)

    async def release(self) -> None:
        """Tear down every environment this pool has handed out.

        One failing teardown must not abort the others, so each is isolated;
        the failure is logged rather than swallowed so it is diagnosable.
        """
        await _finish_cleanup(self._release_owned())

    async def _release_owned(self) -> None:
        failures: list[str] = []
        for env in tuple(self._envs):
            try:
                await _finish_cleanup(env.cleanup())
            except BaseException as exc:
                failures.append(f"{env.workspace}: {type(exc).__name__}: {exc}")
                logger.warning(
                    "environment cleanup failed for %s",
                    env.workspace,
                    exc_info=True,
                )
            else:
                self._envs.remove(env)
        if failures:
            raise OSError("environment pool cleanup failed; retry state retained: " + "; ".join(failures))

    async def release_env(self, env: Environment) -> None:
        """Release one failed spawn's environment without touching siblings."""
        if env not in self._envs:
            return
        try:
            await _finish_cleanup(env.cleanup())
        except BaseException:
            logger.warning("environment cleanup failed for %s", env.workspace, exc_info=True)
            raise
        self._envs.remove(env)

    async def cleanup(self) -> None:
        """Compatibility alias for older callers."""
        await self.release()
