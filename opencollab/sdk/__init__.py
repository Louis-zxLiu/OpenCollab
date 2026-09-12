"""Compact public Python API; package SemVer is its compatibility contract."""

from __future__ import annotations

from opencollab.workflows import workflow

from .client import OpenCollab
from .result import (
    CheckpointResult,
    RestoreResult,
    RollbackPlan,
    RollbackResult,
    RunError,
    RunResult,
)
from .team import RollbackClient, TeamHandle

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
