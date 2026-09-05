"""Public environment contract and command result value."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from opencollab.domain.rollback import EnvironmentSnapshot

ENV_FILE_WRITE_LIMIT_BYTES = 4 * 1024 * 1024


@dataclass(slots=True)
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    stdout_dropped_bytes: int = 0
    stderr_dropped_bytes: int = 0


@dataclass(slots=True)
class TextFileRange:
    lines: list[str]
    start_line: int
    total_lines: int | None
    has_more: bool
    chars_truncated: bool = False


class Environment:
    """Abstract execution environment used by tools and workflows."""

    workspace: str = "."
    host_workspace: str | None = None
    source_workspace: str | None = None
    local_filesystem: bool = False
    process_isolated: bool = False

    def __init__(self) -> None:
        self._aborted = False
        self._scope_values = os.environ.copy()

    def bind_workspace(self, workspace: str) -> None:
        """Bind the Scope PWD to the workspace used by the environment."""
        self._scope_values["PWD"] = os.fspath(workspace)

    def snapshot_environment(self) -> EnvironmentSnapshot:
        return EnvironmentSnapshot.from_mapping(self._scope_values)

    def replace_environment(self, snapshot: EnvironmentSnapshot) -> None:
        """Replace Scope state without modifying the host process environment."""
        self._scope_values = snapshot.as_dict()

    def set_environment_variable(self, name: str, value: str) -> None:
        if not isinstance(name, str) or not name or "\0" in name:
            raise ValueError("environment variable name is invalid")
        if not isinstance(value, str) or "\0" in value:
            raise ValueError("environment variable value is invalid")
        self._scope_values[name] = value

    def unset_environment_variable(self, name: str) -> None:
        if not isinstance(name, str) or not name or "\0" in name:
            raise ValueError("environment variable name is invalid")
        self._scope_values.pop(name, None)

    def environment_view(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._scope_values))

    def process_environment(self) -> dict[str, str]:
        """Return the environment inherited by newly started commands."""
        return dict(self._scope_values)

    @property
    def revoked(self) -> bool:
        return self._aborted

    def revoke(self) -> None:
        self._aborted = True

    def _ensure_active(self) -> None:
        if self._aborted:
            raise RuntimeError("Execution environment has been aborted.")

    async def setup(self, mount_dir: str | None = None) -> str:
        """Prepare the environment and return its ready workspace identity."""
        self._ensure_active()
        if mount_dir is not None:
            raise ValueError("mount_dir is supported only by container environments")
        return self.workspace

    async def exec_cmd(self, cmd: str, timeout: float = 120.0) -> ExecResult:
        raise NotImplementedError

    async def read_file(self, path: str) -> str:
        raise NotImplementedError

    async def read_text_range(
        self,
        path: str,
        *,
        offset: int,
        limit: int,
        max_chars: int,
    ) -> TextFileRange:
        content = await self.read_file(path)
        lines = content.splitlines()
        start = max(0, offset - 1)
        end = min(len(lines), start + limit)
        selected = lines[start:end]
        joined = "\n".join(selected)
        chars_truncated = len(joined) > max_chars
        if chars_truncated:
            selected = joined[:max_chars].split("\n")
        return TextFileRange(
            lines=selected,
            start_line=start + 1,
            total_lines=len(lines),
            has_more=end < len(lines) or chars_truncated,
            chars_truncated=chars_truncated,
        )

    async def write_file(self, path: str, content: str) -> None:
        raise NotImplementedError

    async def write_temp_file(
        self,
        content: str,
        *,
        prefix: str,
        suffix: str = ".tmp",
    ) -> str:
        raise NotImplementedError

    async def remove_file(self, path: str) -> None:
        raise NotImplementedError

    async def cleanup(self) -> None:
        pass

    async def abort(self) -> None:
        self.revoke()
        await self.cleanup()


__all__ = ["ENV_FILE_WRITE_LIMIT_BYTES", "Environment", "ExecResult", "TextFileRange"]
