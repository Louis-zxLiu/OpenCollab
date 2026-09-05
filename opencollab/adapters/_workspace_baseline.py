"""Filesystem helpers for baseline manifests and workspace Effects."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from opencollab.domain.rollback import BaselineEntry, WorkspaceBaseline


def _normalize_relative_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def is_control_plane(path: str) -> bool:
    normalized = _normalize_relative_path(path)
    return normalized == ".opencollab" or normalized.startswith(".opencollab/")


def entry_for_path(root: str, relative: str) -> BaselineEntry:
    relative = _normalize_relative_path(relative)
    parts = relative.split("/")
    if not relative or relative == "." or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"workspace baseline contains unsupported path type: {relative or '.'}")
    target = Path(root, *parts)
    stat = target.lstat()
    if target.is_symlink():
        kind = "symlink"
        payload = os.readlink(target).encode()
        size = len(payload)
    elif target.is_file():
        kind = "file"
        payload = target.read_bytes()
        size = stat.st_size
    else:
        raise ValueError(f"workspace baseline contains unsupported path type: {relative}")
    return BaselineEntry(
        path=relative,
        kind=kind,
        mode=stat.st_mode & 0o7777,
        size=size,
        content_hash=hashlib.sha256(payload).hexdigest(),
        control_plane=is_control_plane(relative),
    )


def baseline_from_paths(root: str, paths: list[str]) -> WorkspaceBaseline:
    entries = tuple(
        sorted(
            (entry_for_path(root, path) for path in paths if not is_control_plane(path)),
            key=lambda entry: entry.path,
        )
    )
    return WorkspaceBaseline(entries)


def validate_baseline(root: str, baseline: WorkspaceBaseline) -> None:
    for expected in baseline.entries:
        try:
            actual = entry_for_path(root, expected.path)
        except (FileNotFoundError, NotADirectoryError):
            raise RuntimeError(f"baseline mutation: missing {expected.path}") from None
        if actual != expected:
            raise RuntimeError(f"baseline mutation: {expected.path}")


def seed_baseline(source_root: str, target_root: str, baseline: WorkspaceBaseline) -> None:
    source_real = os.path.realpath(source_root)
    target_real = os.path.realpath(target_root)
    for entry in baseline.entries:
        source = os.path.join(source_real, *entry.path.split("/"))
        target = os.path.join(target_real, *entry.path.split("/"))
        parent = os.path.realpath(os.path.dirname(target))
        if os.path.commonpath((target_real, parent)) != target_real:
            raise RuntimeError("baseline target escapes worktree")
        os.makedirs(parent, exist_ok=True)
        if os.path.lexists(target):
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
            else:
                os.unlink(target)
        if entry.kind == "symlink":
            os.symlink(os.readlink(source), target)
        else:
            shutil.copy2(source, target, follow_symlinks=False)


def changed_entries(root: str, paths: list[str], baseline: WorkspaceBaseline) -> tuple[BaselineEntry, ...]:
    baseline_paths = baseline.by_path()
    entries = []
    for path in paths:
        normalized = _normalize_relative_path(path)
        if is_control_plane(normalized) or normalized in baseline_paths:
            continue
        try:
            entries.append(entry_for_path(root, normalized))
        except (FileNotFoundError, NotADirectoryError):
            continue
    return tuple(sorted(entries, key=lambda entry: entry.path))


__all__ = [
    "baseline_from_paths",
    "changed_entries",
    "entry_for_path",
    "is_control_plane",
    "seed_baseline",
    "validate_baseline",
]
