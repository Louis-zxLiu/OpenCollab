"""Host process sandbox selection for local worktree command execution."""

from __future__ import annotations

import os
import shutil

HOST_PROCESS_SANDBOX_ENV = "OPENCOLLAB_HOST_PROCESS_SANDBOX"

_DISABLED_VALUES = frozenset({"", "0", "false", "no", "none", "off"})


def host_process_sandbox_prefix() -> tuple[str, ...] | None:
    """Return an argv prefix that executes a shell command inside a host sandbox."""
    requested = os.environ.get(HOST_PROCESS_SANDBOX_ENV, "").strip().lower()
    if requested in _DISABLED_VALUES:
        return None
    if requested != "firejail":
        raise ValueError(
            f"unsupported {HOST_PROCESS_SANDBOX_ENV} value {requested!r}; "
            "supported values are 'firejail' or 'none'"
        )
    if os.name != "posix":
        raise RuntimeError(f"{HOST_PROCESS_SANDBOX_ENV}=firejail requires a POSIX host")
    firejail = shutil.which("firejail")
    if firejail is None:
        raise RuntimeError(f"{HOST_PROCESS_SANDBOX_ENV}=firejail but firejail is not available on PATH")
    return (firejail, "--quiet", "--noprofile", "--", "/bin/sh", "-s")


__all__ = ["HOST_PROCESS_SANDBOX_ENV", "host_process_sandbox_prefix"]
