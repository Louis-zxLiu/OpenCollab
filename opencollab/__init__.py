"""OpenCollab's compact, lazily loaded public Python API."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opencollab.sdk import (
        CheckpointResult,
        OpenCollab,
        RestoreResult,
        RollbackClient,
        RollbackPlan,
        RollbackResult,
        RunError,
        RunResult,
        TeamHandle,
        workflow,
    )

__version__ = "0.6.0"

__all__ = [
    "CheckpointResult",
    "OpenCollab",
    "RestoreResult",
    "RollbackClient",
    "RollbackPlan",
    "RollbackResult",
    "RunError",
    "RunResult",
    "TeamHandle",
    "workflow",
]
_PUBLIC_MODULES = {
    "CheckpointResult": "opencollab.sdk.result",
    "OpenCollab": "opencollab.sdk.client",
    "RestoreResult": "opencollab.sdk.result",
    "RollbackClient": "opencollab.sdk.team",
    "RollbackPlan": "opencollab.sdk.result",
    "RollbackResult": "opencollab.sdk.result",
    "RunError": "opencollab.sdk.result",
    "RunResult": "opencollab.sdk.result",
    "TeamHandle": "opencollab.sdk.team",
    "workflow": "opencollab.workflows",
}


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_PUBLIC_MODULES[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
