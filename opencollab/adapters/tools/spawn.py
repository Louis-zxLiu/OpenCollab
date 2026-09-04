"""Spawn agent tools bound to a scheduler port."""

from __future__ import annotations

from typing import Any

from opencollab.adapters.tools.base import Tool
from opencollab.application.ports import SchedulerPort
from opencollab.application.scheduler_types import (
    DuplicateSpawnError,
    TeamPrebuiltError,
)
from opencollab.application.self_collaboration import validate_review_iterations
from opencollab.application.tool_execution import DeferredCall, ToolRuntime
from opencollab.domain.coordination import (
    DEFAULT_COORDINATION_POLICY,
    CoordinationPolicy,
)
from opencollab.domain.identity import validate_role_identity


def _coordination_parameters(policy: CoordinationPolicy) -> dict[str, Any]:
    """Build the shared short-payload schema for every coordination tool."""
    return {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Short assignment only; use the task context for the original task.",
                "maxLength": policy.assignment_bytes,
            },
            "context": {
                "type": "string",
                "description": "Concise facts or constraints only; omit full history and reasoning.",
                "maxLength": policy.context_bytes,
            },
        },
    }


def _spawn_parameters(policy: CoordinationPolicy) -> dict[str, Any]:
    parameters = _coordination_parameters(policy)
    parameters["properties"]["role"] = {
        "type": "string",
        "description": "The specialist role named by the team topology.",
    }
    parameters["required"] = ["role", "task"]
    return parameters


def _review_parameters(policy: CoordinationPolicy) -> dict[str, Any]:
    parameters = _coordination_parameters(policy)
    parameters["properties"]["max_iterations"] = {
        "type": "integer",
        "description": "Max review iterations (default 3).",
    }
    parameters["required"] = ["task"]
    return parameters


class SpawnAgentTool(Tool):
    """Tool that an agent uses to spawn a child agent asynchronously.

    Returns immediately with a ``DeferredCall`` referencing the child agent's
    aid. The child runs in parallel and its result is injected into the
    parent's message history when complete.
    """

    name = "spawn_agent"
    description = (
        "Spawn a specialist agent to work on a short assignment. Keep task and "
        "context concise; do not copy the full problem, history, or reasoning. "
        "You will pause until the "
        "agent finishes, then its result is delivered straight back to you as "
        "this tool call's result — so you can act on it in the same turn. Spawn "
        "several at once to run them in parallel; you resume when all are done. "
        "The roles you may spawn are listed in your team context; a team with an "
        "open topology also accepts a custom role name."
    )
    # Keep the default schema available to callers that inspect the class
    # directly; instances replace it when a composition root supplies policy.
    parameters = _spawn_parameters(DEFAULT_COORDINATION_POLICY)

    def __init__(
        self,
        scheduler: SchedulerPort,
        coordination_policy: CoordinationPolicy | None = None,
    ):
        self._scheduler = scheduler
        self._coordination_policy = coordination_policy or DEFAULT_COORDINATION_POLICY
        self.parameters = _spawn_parameters(self._coordination_policy)

    async def execute_with_runtime(
        self,
        params: dict[str, Any],
        runtime: ToolRuntime,
    ) -> DeferredCall | str:
        try:
            role = validate_role_identity(params["role"])
        except ValueError as exc:
            return f"Not spawned: invalid role identity ({exc})."
        task = params.get("task")
        context = params.get("context", "")
        error = self._coordination_policy.validate(task, context)
        if error:
            return f"Not spawned: {error}."
        parent_aid = runtime.aid
        # Scheduler.spawn is the authoritative single-flight boundary. Convert
        # its domain-specific conflict into a synchronous tool result so no
        # pending row is registered for the rejected duplicate.
        try:
            aid = await self._scheduler.spawn(
                parent_aid, role, task, context, tool_call_id=runtime.tool_call_id
            )
        except DuplicateSpawnError as exc:
            return (
                f"Not spawned: this task is already being handled by agent "
                f"aid={exc.existing_aid}. Do not spawn another agent for the same task — "
                f"its result will be delivered to you as a tool result, and you "
                f"can act on it then."
            )
        except TeamPrebuiltError as exc:
            # A fixed roster is a fact about the team, not a fault in the call.
            # The scheduler already wrote the attempt down and phrased the reply
            # for the model; pass it through unchanged rather than wrapping it in
            # an error shape that reads as a malfunction.
            return str(exc)
        # Defer with the child aid so the deferral path can register a pending
        # row keyed by this tool call; the child's result fills it on completion.
        return DeferredCall(ref=aid)


class SpawnWithReviewTool(Tool):
    """Tool for coding tasks requiring mandatory code review."""

    name = "spawn_with_review"
    # One invocation may legitimately contain six sequential model turns
    # (coder + reviewer across three iterations). Each child session already
    # enforces its own provider/session deadline, so the ordinary single-tool
    # wall would cancel a healthy review loop midway through a later iteration.
    disable_outer_timeout = True
    description = (
        "Spawn a coding task with mandatory code review. A Coder implements the task, "
        "then a Reviewer checks the work. If the review fails, the Coder retries with "
        "feedback. Max 3 iterations. Blocks until complete."
    )
    parameters = _review_parameters(DEFAULT_COORDINATION_POLICY)

    def __init__(
        self,
        scheduler: SchedulerPort,
        coordination_policy: CoordinationPolicy | None = None,
    ):
        self._scheduler = scheduler
        self._coordination_policy = coordination_policy or DEFAULT_COORDINATION_POLICY
        self.parameters = _review_parameters(self._coordination_policy)

    async def execute_with_runtime(
        self,
        params: dict[str, Any],
        runtime: ToolRuntime,
    ) -> str:
        task = params.get("task")
        context = params.get("context", "")
        error = self._coordination_policy.validate(task, context)
        if error:
            return f"Not started: {error}."
        max_iter = params.get("max_iterations", 3)
        try:
            max_iter = validate_review_iterations(max_iter)
        except ValueError as exc:
            return f"Not started: {exc}."
        parent_aid = runtime.aid
        return await self._scheduler.spawn_with_review(
            parent_aid, task, context, max_iter
        )


__all__ = ["SpawnAgentTool", "SpawnWithReviewTool"]
