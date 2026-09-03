"""Runtime context + workspace safety policy — the pre-session wiring inputs.

``RuntimeContext`` bundles what the CLI and SDK entry points resolve before any
session exists (workspace, config overrides, tracer, sinks); the safety-policy
factory turns an environment into its sandbox interceptor. Re-exported from
``bootstrap.container`` so existing import paths keep resolving.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from opencollab.adapters.env import Environment
from opencollab.adapters.safety import SandboxInterceptor
from opencollab.adapters.trace import Tracer
from opencollab.application.ports import (
    AskUserPort,
    EventPublisherPort,
    PermissionPort,
    SafetyPolicyPort,
)


@dataclass
class RuntimeContext:
    workspace: str
    config: dict
    tracer: Tracer | None
    event_sink: EventPublisherPort | None
    permission_policy: PermissionPort | None
    ask_policy: AskUserPort | None = None


def build_runtime_context(
    workspace: str,
    cli_overrides: dict,
    *,
    trace: bool,
    event_sink: EventPublisherPort | None = None,
    permission_policy: PermissionPort | None = None,
    ask_policy: AskUserPort | None = None,
    run_id_prefix: str = "",
) -> RuntimeContext:
    """Resolve the workspace path and optional tracer into a ``RuntimeContext``."""
    abs_workspace = os.path.abspath(workspace)
    # Runtime observability belongs to the workspace control plane. Keeping it
    # under ``.opencollab`` prevents a traced start-up from creating a
    # top-level, untracked ``trajectories/`` directory before Git worktree
    # validation runs. The whole control-plane directory is ignored by the
    # workspace baseline and is never part of an Agent patch.
    trace_dir = Path(abs_workspace) / ".opencollab" / "trajectories"
    tracer = (
        Tracer(
            run_id=f"{run_id_prefix}{uuid.uuid4().hex[:8]}",
            output_dir=str(trace_dir),
        )
        if trace
        else None
    )

    return RuntimeContext(
        workspace=abs_workspace,
        config=dict(cli_overrides),
        tracer=tracer,
        event_sink=event_sink,
        permission_policy=permission_policy,
        ask_policy=ask_policy,
    )


def build_workspace_safety_policy(env: Environment) -> SafetyPolicyPort | None:
    """A sandbox interceptor scoped to ``env``'s workspace, or None without one."""
    if env is None or not getattr(env, "workspace", None):
        return None
    return SandboxInterceptor(env.workspace)
