"""Private helpers for host worktree environments."""

from __future__ import annotations

_PORCELAIN_STATUS_CHARS = frozenset(" MADRCU?!")


def dirty_path_preview(output: str, *, limit: int = 12) -> str:
    paths: list[str] = []
    for record in filter(None, output.split("\0")):
        if (
            len(record) >= 4
            and record[0] in _PORCELAIN_STATUS_CHARS
            and record[1] in _PORCELAIN_STATUS_CHARS
            and record[2] == " "
        ):
            paths.append(record[3:])
        else:
            paths.append(record)
    preview = ", ".join(repr(path) for path in paths[:limit])
    if len(paths) > limit:
        preview += f", ... ({len(paths) - limit} more)"
    return preview


__all__ = ["dirty_path_preview"]
