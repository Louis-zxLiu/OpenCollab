"""Publish immutable workspace revisions back to their source workspace."""

from __future__ import annotations

import os
from collections.abc import Mapping

from opencollab.adapters._env_process import PROCESS_OUTPUT_CAPTURE_BYTES, run_process
from opencollab.adapters._workspace_baseline import is_control_plane, validate_baseline
from opencollab.domain.rollback import AdoptionResult, WorkspaceBaseline, WorkspaceRevision

_GIT_TIMEOUT = 60.0


async def _git(workspace: str, *args: str, env: Mapping[str, str] | None = None):
    result = await run_process(
        ("git", *args),
        shell=False,
        cwd=workspace,
        timeout=_GIT_TIMEOUT,
        output_limit=PROCESS_OUTPUT_CAPTURE_BYTES,
        env=dict(env) if env is not None else None,
    )
    return result.to_exec_result()


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


def _non_control_status(output: str) -> list[str]:
    return [path for path in _status_paths(output) if not is_control_plane(path)]


async def _revision_paths(workspace: str, revision: WorkspaceRevision, *, diff_filter: str | None = None) -> list[str]:
    args = ["diff-tree", "--no-commit-id", "--name-only", "-r", "-z"]
    if diff_filter is not None:
        args.append(f"--diff-filter={diff_filter}")
    args.append(revision.revision)
    result = await _git(workspace, *args)
    if result.returncode != 0 or result.stdout_truncated or result.stderr_truncated:
        raise RuntimeError(result.stderr.strip() or "cannot inspect workspace revision")
    return [path for path in result.stdout.split("\0") if path]


async def _validate_revision_for_publish(
    workspace: str,
    revision: WorkspaceRevision,
    baseline: WorkspaceBaseline,
) -> AdoptionResult | None:
    valid = await _git(workspace, "cat-file", "-e", f"{revision.revision}^{{commit}}")
    if valid.returncode != 0:
        return AdoptionResult("failed", reason="workspace revision is unavailable")
    parent = await _git(workspace, "rev-parse", f"{revision.revision}^")
    if parent.returncode != 0 or parent.stdout.strip() != revision.base_revision:
        return AdoptionResult("failed", reason="workspace revision base mismatch")
    head_tree = await _git(workspace, "rev-parse", "HEAD^{tree}")
    base_tree = await _git(workspace, "rev-parse", f"{revision.base_revision}^{{tree}}")
    if head_tree.returncode != 0 or base_tree.returncode != 0 or head_tree.stdout.strip() != base_tree.stdout.strip():
        return AdoptionResult("conflict", reason="source workspace is not at the revision base")
    changed_paths = await _revision_paths(workspace, revision)
    baseline_paths = baseline.by_path()
    protected = [path for path in changed_paths if is_control_plane(path) or path in baseline_paths]
    if protected:
        return AdoptionResult("failed", reason=f"workspace revision touches protected path {protected[0]}")
    return None


async def publish_host_workspace_revision(
    source_workspace: str,
    revision: WorkspaceRevision,
    baseline: WorkspaceBaseline,
    *,
    env: Mapping[str, str] | None = None,
) -> AdoptionResult:
    """Apply one immutable Effect revision to a clean host source workspace."""
    if not revision.changed:
        return AdoptionResult("skipped", revision=revision.revision)
    source = os.path.realpath(os.path.abspath(source_workspace))
    try:
        validate_baseline(source, baseline)
    except RuntimeError as exc:
        return AdoptionResult("failed", reason=str(exc)[:500])
    preflight = await _validate_revision_for_publish(source, revision, baseline)
    if preflight is not None:
        return preflight
    status = await _git(source, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if status.returncode != 0 or status.stdout_truncated or status.stderr_truncated:
        return AdoptionResult("failed", reason=status.stderr.strip() or "cannot inspect source workspace")
    dirty_paths = _non_control_status(status.stdout)
    if dirty_paths:
        return AdoptionResult("conflict", reason=f"source workspace is not clean: {dirty_paths[0]}")
    for path in await _revision_paths(source, revision, diff_filter="A"):
        target = os.path.join(source, *path.split("/"))
        tracked = await _git(source, "ls-files", "--error-unmatch", "--", path)
        if os.path.lexists(target) and tracked.returncode != 0:
            return AdoptionResult("conflict", reason=f"source workspace already has untracked {path}")
    picked = await _git(source, "cherry-pick", "--no-commit", revision.revision, env=env)
    if picked.returncode != 0:
        await _git(source, "cherry-pick", "--abort")
        return AdoptionResult("failed", reason=(picked.stderr.strip() or "cherry-pick failed")[:500])
    return AdoptionResult("adopted", revision=revision.revision)


__all__ = ["publish_host_workspace_revision"]
