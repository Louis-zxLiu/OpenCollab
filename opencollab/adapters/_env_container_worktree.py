"""A Git worktree carved out inside an already-running task container."""

from __future__ import annotations

import logging
import posixpath
import shlex
import uuid
from collections.abc import Callable

from opencollab.adapters._env_base import ExecResult
from opencollab.adapters._env_docker import DockerEnvironment
from opencollab.adapters._env_scope import _ScopeState
from opencollab.adapters._workspace_baseline import is_control_plane
from opencollab.adapters.git_patch import guarded_staged_diff_command
from opencollab.adapters.git_worktree_evidence import (
    ABSENT_REF_OLD_VALUE,
    parse_own_commits,
    select_diff_base,
    validate_worktree_branch,
)
from opencollab.domain.rollback import (
    AdoptionResult,
    BaselineEntry,
    CheckpointBoundary,
    RestoreResult,
    ScopeCheckpoint,
    WorkspaceBaseline,
    WorkspaceRevision,
)

logger = logging.getLogger(__name__)

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
        _scope: _ScopeState | None = None,
        base_revision: str | None = None,
        baseline: WorkspaceBaseline | None = None,
    ) -> None:
        branch = validate_worktree_branch(branch_name or f"opencollab-wt-{uuid.uuid4().hex[:12]}")
        repository_root = _absolute_container_path(repository_root, "repository root")
        worktree_root = _absolute_container_path(worktree_root, "worktree root")
        worktree_dir = posixpath.join(worktree_root, branch)
        super().__init__(
            workspace=worktree_dir,
            container_id=container_id,
            exec_workdir=worktree_dir,
            command_prefix=command_prefix,
            timeout_returncode=timeout_returncode,
            _scope=_scope,
        )
        self.source_workspace = repository_root
        self._repository_root = repository_root
        self._worktree_root = worktree_root
        self._worktree_dir = worktree_dir
        self._branch = branch
        self._branch_owned = False
        self._worktree_registered = False
        self._base_commit: str | None = None
        self._requested_base_revision = base_revision
        self._baseline = baseline or WorkspaceBaseline()
        self._diff_base: str | None = None
        self._head_commit: str | None = None
        self._own_commits: tuple[str, ...] = ()
        self._own_commit_count: int | None = None
        self._scope_checkpoints: dict[str, ScopeCheckpoint] = {}
        self._effect_refs: set[str] = set()
        self._checkpoint_sequence = 0
        self._workspace_effects: dict[str, str] = {}
        self.bind_workspace(worktree_dir)

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

    async def checkpoint_scope(
        self,
        boundary: CheckpointBoundary,
        *,
        owner_aid: int,
        causal_frontier: frozenset[str],
    ) -> ScopeCheckpoint:
        await self.setup()
        await self._validate_baseline()
        checkpoint_id = f"cp_{uuid.uuid4().hex}"
        script = (
            "set -eu; idx=$(mktemp); trap 'rm -f \"$idx\"' EXIT; "
            "GIT_INDEX_FILE=$idx git read-tree HEAD; "
            "GIT_INDEX_FILE=$idx git add -A; "
            "GIT_INDEX_FILE=$idx git reset -q HEAD -- .opencollab 2>/dev/null || true; "
            "GIT_INDEX_FILE=$idx git ls-files --others --ignored --exclude-standard -z | "
            "while IFS= read -r -d '' p; do case \"$p\" in .opencollab|.opencollab/*) ;; "
            "*) GIT_INDEX_FILE=$idx git add -f -- \"$p\" ;; esac; done; "
            "tree=$(GIT_INDEX_FILE=$idx git write-tree); "
            "commit=$(printf '%s\\n' 'OpenCollab checkpoint' | "
            "GIT_AUTHOR_NAME='OpenCollab Checkpoint' GIT_AUTHOR_EMAIL='checkpoint@opencollab.invalid' "
            "GIT_COMMITTER_NAME='OpenCollab Checkpoint' GIT_COMMITTER_EMAIL='checkpoint@opencollab.invalid' "
            'git commit-tree "$tree" -p HEAD); '
            f"git update-ref refs/opencollab/checkpoints/{owner_aid}/{checkpoint_id} "
            '"$commit"; printf \'%s\' "$commit"'
        )
        result = await self._exec(script, timeout=CONTAINER_GIT_TIMEOUT_SECONDS)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "container Git checkpoint failed")
        self._checkpoint_sequence += 1
        checkpoint = ScopeCheckpoint(
            checkpoint_id=checkpoint_id,
            owner_aid=owner_aid,
            sequence=self._checkpoint_sequence,
            filesystem_revision=result.stdout.strip(),
            environment=self.snapshot_environment(),
            causal_frontier=causal_frontier,
            boundary_kind=boundary.kind,
            boundary_effect_id=boundary.effect_id,
        )
        self._scope_checkpoints[checkpoint_id] = checkpoint
        return checkpoint

    async def capture_workspace_revision(self, reference_id: str, *, owner_aid: int) -> WorkspaceRevision:
        """Freeze this container worktree as an immutable Effect revision."""
        self._ensure_active()
        if not self._worktree_registered:
            await self.setup()
        if self._base_commit is None:
            raise RuntimeError("workspace Effect base revision is unavailable")
        await self._validate_baseline()
        ref_name = f"refs/opencollab/effects/{owner_aid}/{reference_id}"
        baseline_paths = [shlex.quote(path) for path in self._baseline.by_path()]
        ignored_skip = ".opencollab|.opencollab/*"
        if baseline_paths:
            ignored_skip = "|".join((ignored_skip, *baseline_paths))
        script = (
            "set -eu; idx=$(mktemp); trap 'rm -f \"$idx\"' EXIT; "
            "GIT_INDEX_FILE=$idx git read-tree HEAD; "
            "GIT_INDEX_FILE=$idx git add -A; "
            "GIT_INDEX_FILE=$idx git reset -q HEAD -- .opencollab 2>/dev/null || true; "
            "GIT_INDEX_FILE=$idx git ls-files --others --ignored --exclude-standard -z | "
            f"while IFS= read -r -d '' p; do case \"$p\" in {ignored_skip}) ;; "
            "*) GIT_INDEX_FILE=$idx git add -f -- \"$p\" ;; esac; done; "
            "tree=$(GIT_INDEX_FILE=$idx git write-tree); "
            "commit=$(printf '%s\\n' "
            f"{shlex.quote(f'OpenCollab effect {reference_id}')}"
            " | GIT_AUTHOR_NAME='OpenCollab Effect' "
            "GIT_AUTHOR_EMAIL='effect@opencollab.invalid' "
            "GIT_COMMITTER_NAME='OpenCollab Effect' "
            "GIT_COMMITTER_EMAIL='effect@opencollab.invalid' "
            f'git commit-tree "$tree" -p {shlex.quote(self._base_commit)}); '
            f'git update-ref {shlex.quote(ref_name)} "$commit"; printf \'%s\' "$commit"'
        )
        result = await self._exec(script, timeout=CONTAINER_GIT_TIMEOUT_SECONDS)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "container Effect capture failed")
        self._effect_refs.add(ref_name)
        effect_tree = await self._git(self._worktree_dir, "rev-parse", f"{result.stdout.strip()}^{{tree}}")
        base_tree = await self._git(self._worktree_dir, "rev-parse", f"{self._base_commit}^{{tree}}")
        if effect_tree.returncode != 0 or base_tree.returncode != 0:
            raise RuntimeError("cannot compare container workspace revision")
        return WorkspaceRevision(
            result.stdout.strip(),
            self._base_commit,
            effect_tree.stdout.strip() != base_tree.stdout.strip(),
            files=await self._effect_files(result.stdout.strip()),
        )

    async def _effect_files(self, revision: str) -> tuple[BaselineEntry, ...]:
        result = await self._git(self._worktree_dir, "diff-tree", "--no-commit-id", "--name-only", "-r", "-z", revision)
        if result.returncode != 0:
            raise RuntimeError("cannot enumerate container workspace Effect")
        entries: list[BaselineEntry] = []
        for path in (item for item in result.stdout.split("\0") if item and not is_control_plane(item)):
            stat = await self._exec(f"stat -c '%a %s' -- {shlex.quote(path)}", timeout=CONTAINER_GIT_TIMEOUT_SECONDS)
            digest = await self._exec(f"sha256sum -- {shlex.quote(path)}", timeout=CONTAINER_GIT_TIMEOUT_SECONDS)
            if stat.returncode != 0 or digest.returncode != 0:
                continue
            mode, size = stat.stdout.strip().split()
            entries.append(BaselineEntry(path, "file", int(mode, 8), int(size), digest.stdout.split()[0]))
        return tuple(sorted(entries, key=lambda item: item.path))

    async def capture_workspace_baseline(self) -> WorkspaceBaseline:
        """Capture ignored inputs from this container worktree."""
        await self.setup()
        result = await self._exec(
            "git ls-files --others --ignored --exclude-standard -z",
            timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "cannot capture container workspace baseline")
        entries: list[BaselineEntry] = []
        for path in (item for item in result.stdout.split("\0") if item and not is_control_plane(item)):
            stat = await self._exec(
                f"stat -c '%a %s' -- {shlex.quote(path)}",
                timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
            )
            digest = await self._exec(
                f"sha256sum -- {shlex.quote(path)}",
                timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
            )
            if stat.returncode != 0 or digest.returncode != 0:
                continue
            mode, size = stat.stdout.strip().split()
            entries.append(BaselineEntry(path, "file", int(mode, 8), int(size), digest.stdout.split()[0]))
        self._baseline = WorkspaceBaseline(tuple(sorted(entries, key=lambda item: item.path)))
        return self._baseline

    async def _validate_baseline(self) -> None:
        for entry in self._baseline.entries:
            stat = await self._exec(
                f"stat -c '%a %s' -- {shlex.quote(entry.path)}",
                timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
            )
            digest = await self._exec(f"sha256sum -- {shlex.quote(entry.path)}", timeout=CONTAINER_GIT_TIMEOUT_SECONDS)
            if stat.returncode != 0 or digest.returncode != 0:
                raise RuntimeError(f"baseline mutation: missing {entry.path}")
            mode, size = stat.stdout.strip().split()
            if int(mode, 8) != entry.mode or int(size) != entry.size or digest.stdout.split()[0] != entry.content_hash:
                raise RuntimeError(f"baseline mutation: {entry.path}")

    async def seed_workspace_baseline(self, baseline: WorkspaceBaseline) -> None:
        await self.setup()
        for entry in baseline.entries:
            source = posixpath.join(self._repository_root, entry.path)
            target = posixpath.join(self._worktree_dir, entry.path)
            result = await self._exec(
                f"mkdir -p -- {shlex.quote(posixpath.dirname(target))} && "
                f"cp -a -- {shlex.quote(source)} {shlex.quote(target)}",
                timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr or f"cannot seed baseline path {entry.path}")
        self._baseline = baseline

    async def collect_workspace_effect(self, effect_id: str, *, owner_aid: int) -> WorkspaceRevision:
        self._workspace_effects[effect_id] = "collected"
        revision = await self.capture_workspace_revision(effect_id, owner_aid=owner_aid)
        if not revision.changed:
            self._workspace_effects[effect_id] = "acknowledged"
        return revision

    async def mark_workspace_effect_delivered(self, effect_id: str) -> None:
        state = self._workspace_effects.get(effect_id)
        if state is None:
            raise RuntimeError(f"unknown workspace Effect {effect_id}")
        if state == "collected":
            self._workspace_effects[effect_id] = "delivered"

    async def acknowledge_workspace_effect(self, effect_id: str) -> None:
        if effect_id not in self._workspace_effects:
            raise RuntimeError(f"unknown workspace Effect {effect_id}")
        self._workspace_effects[effect_id] = "acknowledged"

    async def adopt_workspace_revision(self, revision: WorkspaceRevision) -> AdoptionResult:
        """Cherry-pick a frozen revision into this clean container worktree."""
        self._ensure_active()
        if not self._worktree_registered:
            await self.setup()
        if not revision.changed:
            return AdoptionResult("skipped", revision=revision.revision)
        valid = await self._git(self._worktree_dir, "cat-file", "-e", f"{revision.revision}^{{commit}}")
        if valid.returncode != 0:
            return AdoptionResult("failed", reason="workspace revision is unavailable")
        parent = await self._git(self._worktree_dir, "rev-parse", f"{revision.revision}^")
        if parent.returncode != 0 or parent.stdout.strip() != revision.base_revision:
            return AdoptionResult("failed", reason="workspace revision base mismatch")
        status = await self._git(self._worktree_dir, "status", "--porcelain", "--untracked-files=all")
        if status.returncode != 0:
            return AdoptionResult("failed", reason="cannot inspect container Scope")
        if status.stdout:
            return AdoptionResult("conflict", reason="Scope is not clean")
        picked = await self._git(
            self._worktree_dir,
            "-c",
            "user.name=OpenCollab Effect",
            "-c",
            "user.email=effect@opencollab.invalid",
            "cherry-pick",
            "-x",
            revision.revision,
        )
        if picked.returncode != 0:
            await self._git(self._worktree_dir, "cherry-pick", "--abort")
            return AdoptionResult("failed", reason=(picked.stderr or "cherry-pick failed")[:500])
        return AdoptionResult("adopted", revision=revision.revision)

    async def publish_workspace_revision(self, revision: WorkspaceRevision) -> AdoptionResult:
        """Apply one immutable Effect revision to the container repository root."""
        self._ensure_active()
        if not self._worktree_registered:
            await self.setup()
        if not revision.changed:
            return AdoptionResult("skipped", revision=revision.revision)
        baseline_error = await self._validate_repository_baseline()
        if baseline_error is not None:
            return AdoptionResult("failed", reason=baseline_error)
        valid = await self._git(self._repository_root, "cat-file", "-e", f"{revision.revision}^{{commit}}")
        if valid.returncode != 0:
            return AdoptionResult("failed", reason="workspace revision is unavailable")
        parent = await self._git(self._repository_root, "rev-parse", f"{revision.revision}^")
        if parent.returncode != 0 or parent.stdout.strip() != revision.base_revision:
            return AdoptionResult("failed", reason="workspace revision base mismatch")
        head_tree = await self._git(self._repository_root, "rev-parse", "HEAD^{tree}")
        base_tree = await self._git(self._repository_root, "rev-parse", f"{revision.base_revision}^{{tree}}")
        if (
            head_tree.returncode != 0
            or base_tree.returncode != 0
            or head_tree.stdout.strip() != base_tree.stdout.strip()
        ):
            return AdoptionResult("conflict", reason="source workspace is not at the revision base")
        changed = await self._git(
            self._repository_root,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-z",
            revision.revision,
        )
        if changed.returncode != 0:
            return AdoptionResult("failed", reason=changed.stderr.strip() or "cannot inspect workspace revision")
        baseline_paths = self._baseline.by_path()
        for path in (item for item in changed.stdout.split("\0") if item):
            if is_control_plane(path) or path in baseline_paths:
                return AdoptionResult("failed", reason=f"workspace revision touches protected path {path}")
        status = await self._git(self._repository_root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
        if status.returncode != 0:
            return AdoptionResult("failed", reason=status.stderr.strip() or "cannot inspect source workspace")
        dirty = [path for path in _status_paths(status.stdout) if not is_control_plane(path)]
        if dirty:
            return AdoptionResult("conflict", reason=f"source workspace is not clean: {dirty[0]}")
        added = await self._git(
            self._repository_root,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-z",
            "--diff-filter=A",
            revision.revision,
        )
        for path in (item for item in added.stdout.split("\0") if item):
            tracked = await self._git(self._repository_root, "ls-files", "--error-unmatch", "--", path)
            exists = await self._exec(
                f"test -e {shlex.quote(posixpath.join(self._repository_root, path))} "
                f"-o -L {shlex.quote(posixpath.join(self._repository_root, path))}",
                timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
            )
            if exists.returncode == 0 and tracked.returncode != 0:
                return AdoptionResult("conflict", reason=f"source workspace already has untracked {path}")
        picked = await self._git(self._repository_root, "cherry-pick", "--no-commit", revision.revision)
        if picked.returncode != 0:
            await self._git(self._repository_root, "cherry-pick", "--abort")
            return AdoptionResult("failed", reason=(picked.stderr or "cherry-pick failed")[:500])
        return AdoptionResult("adopted", revision=revision.revision)

    async def _validate_repository_baseline(self) -> str | None:
        for entry in self._baseline.entries:
            path = posixpath.join(self._repository_root, entry.path)
            stat = await self._exec(
                f"stat -c '%a %s' -- {shlex.quote(path)}",
                timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
            )
            digest = await self._exec(f"sha256sum -- {shlex.quote(path)}", timeout=CONTAINER_GIT_TIMEOUT_SECONDS)
            if stat.returncode != 0 or digest.returncode != 0:
                return f"baseline mutation: missing {entry.path}"
            mode, size = stat.stdout.strip().split()
            if int(mode, 8) != entry.mode or int(size) != entry.size or digest.stdout.split()[0] != entry.content_hash:
                return f"baseline mutation: {entry.path}"
        return None

    async def restore_scope(self, checkpoint: ScopeCheckpoint) -> RestoreResult:
        if self._scope_checkpoints.get(checkpoint.checkpoint_id) != checkpoint:
            return RestoreResult(
                checkpoint.owner_aid,
                checkpoint.checkpoint_id,
                "failed",
                reason="checkpoint ownership mismatch",
            )
        await self._validate_baseline()
        reset = await self._git(self._worktree_dir, "read-tree", "--reset", "-u", checkpoint.filesystem_revision)
        if reset.returncode != 0:
            return RestoreResult(checkpoint.owner_aid, checkpoint.checkpoint_id, "failed", reason=reset.stderr[:500])
        ignored = await self._git(self._worktree_dir, "ls-files", "--others", "--ignored", "--exclude-standard", "-z")
        if ignored.returncode != 0:
            return RestoreResult(checkpoint.owner_aid, checkpoint.checkpoint_id, "failed", reason=ignored.stderr[:500])
        baseline_paths = self._baseline.by_path()
        paths = [
            path
            for path in ignored.stdout.split("\0")
            if path and path not in baseline_paths and path != ".opencollab" and not path.startswith(".opencollab/")
        ]
        if paths:
            cleaned = await self._git(self._worktree_dir, "clean", "-fd", "--", *paths)
            if cleaned.returncode != 0:
                return RestoreResult(
                    checkpoint.owner_aid,
                    checkpoint.checkpoint_id,
                    "failed",
                    reason=cleaned.stderr[:500],
                )
        self._scope.replace(checkpoint.environment)
        return RestoreResult(checkpoint.owner_aid, checkpoint.checkpoint_id, "restored")

    async def discard_scope_checkpoints(self) -> None:
        failures: list[str] = []
        for checkpoint_id, checkpoint in tuple(self._scope_checkpoints.items()):
            ref = f"refs/opencollab/checkpoints/{checkpoint.owner_aid}/{checkpoint_id}"
            result = await self._git(self._repository_root, "update-ref", "-d", ref)
            if result.returncode == 0:
                self._scope_checkpoints.pop(checkpoint_id, None)
            else:
                failures.append(result.stderr.strip() or ref)
        for ref in tuple(self._effect_refs):
            result = await self._git(self._repository_root, "update-ref", "-d", ref)
            if result.returncode == 0:
                self._effect_refs.discard(ref)
            else:
                failures.append(result.stderr.strip() or ref)
        if failures:
            raise RuntimeError("failed to discard container Scope refs: " + "; ".join(failures))

    async def _git(self, workdir: str, *args: str) -> ExecResult:
        """Run one Git command in the container, outside the agent's shell.

        Not ``exec_cmd``: that runs what the agent's session runs, through the
        image's login shell and whatever command prefix the harness set. The
        worktree's own bookkeeping has to be argv-exact and independent of that.
        """
        await self._bind_attached()
        container_id = self._container_id
        if container_id is None:
            raise RuntimeError("container worktree is not attached to a container")
        result = await self._docker(
            "exec",
            "-w",
            workdir,
            "--",
            container_id,
            "git",
            "-c",
            f"safe.directory={workdir}",
            *args,
            timeout=CONTAINER_GIT_TIMEOUT_SECONDS,
        )
        return result.to_exec_result()

    async def setup(self, mount_dir: str | None = None) -> str:
        await super().setup(mount_dir)
        self._ensure_active()
        if not self._worktree_registered:
            await self._create_worktree()
        return self._worktree_dir

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
        self._base_commit = self._requested_base_revision or base.stdout.strip()
        if self._requested_base_revision:
            available = await self._git(
                self._repository_root,
                "cat-file",
                "-e",
                f"{self._requested_base_revision}^{{commit}}",
            )
            if available.returncode != 0:
                raise RuntimeError("requested parent workspace revision is unavailable")
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
            raise RuntimeError(f"cannot atomically claim container worktree branch: {claimed.stderr.strip()}")
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
        await self.seed_workspace_baseline(self._baseline)

    async def get_diff(self) -> str:
        """This worktree's changes since the point its current work started from."""
        self._ensure_active()
        if self._base_commit is None:
            raise RuntimeError("container worktree base commit is unavailable")
        await self._validate_baseline()
        self._diff_base = await self._resolve_diff_base()
        await self._resolve_own_commits(self._diff_base)
        result = await self.exec_cmd(
            guarded_staged_diff_command(
                base_revision=self._diff_base,
                baseline_paths=tuple(self._baseline.by_path()),
            )
        )
        if result.stdout_truncated or result.stderr_truncated:
            raise RuntimeError(f"container worktree diff exceeded capture limit at {self._worktree_dir}")
        if result.returncode != 0:
            detail = result.stderr.strip() or f"git exited with status {result.returncode}"
            raise RuntimeError(
                f"container worktree diff extraction failed: {detail}; worktree retained at {self._worktree_dir}"
            )
        return result.stdout

    async def _resolve_diff_base(self) -> str:
        assert self._base_commit is not None
        reflog = await self._git(self._worktree_dir, "log", "-g", "--format=%H%x09%gs", "HEAD")
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
        listed = await self._git(self._worktree_dir, "rev-list", f"{base_revision}..HEAD")
        if listed.returncode != 0 or listed.stdout_truncated:
            return
        self._own_commits, self._own_commit_count = parse_own_commits(listed.stdout)

    async def cleanup(self) -> None:
        """Remove the worktree and release its branch lease, then detach.

        Every step is best effort and reported rather than raised: the container
        is the harness's to tear down, and a worktree left behind inside one
        that is about to be removed costs nothing, while a raising cleanup would
        turn a finished run into a failed one.
        """
        if any(state in {"collected", "delivered"} for state in self._workspace_effects.values()):
            raise RuntimeError(
                "refusing to clean undelivered container workspace Effects; "
                f"worktree retained at {self._worktree_dir}"
            )
        if self._worktree_registered:
            try:
                await self.discard_scope_checkpoints()
            except Exception as exc:
                logger.warning("container Scope refs were retained: %s", exc)
            removed = await self._git(self._repository_root, "worktree", "remove", "--force", self._worktree_dir)
            if removed.returncode != 0:
                logger.warning(
                    "container worktree not removed at %s: %s",
                    self._worktree_dir,
                    removed.stderr.strip(),
                )
            else:
                self._worktree_registered = False
        if self._branch_owned and self._base_commit is not None:
            # Delete only a lease that still stands where it was claimed, so a
            # branch something else advanced is left for its owner to explain.
            released = await self._git(
                self._repository_root,
                "update-ref",
                "-d",
                f"refs/heads/{self._branch}",
                self._base_commit,
            )
            if released.returncode == 0:
                self._branch_owned = False
            else:
                logger.warning("container worktree branch lease retained: %s", released.stderr.strip())
        await super().cleanup()


def _absolute_container_path(path: str, label: str) -> str:
    if not isinstance(path, str) or not path.startswith("/") or "\0" in path:
        raise ValueError(f"container {label} must be an absolute path without NUL bytes")
    normalized = posixpath.normpath(path)
    if normalized == "/":
        raise ValueError(f"container {label} must not be the container root")
    return normalized


def _status_paths(output: str) -> list[str]:
    paths: list[str] = []
    records = [record for record in output.split("\0") if record]
    index = 0
    while index < len(records):
        record = records[index]
        paths.append(record[3:] if len(record) >= 4 and record[2] == " " else record)
        if len(record) >= 2 and (record[0] in {"R", "C"} or record[1] in {"R", "C"}):
            index += 1
            if index < len(records):
                paths.append(records[index])
        index += 1
    return paths


__all__ = ["CONTAINER_GIT_TIMEOUT_SECONDS", "ContainerWorktreeEnvironment"]
