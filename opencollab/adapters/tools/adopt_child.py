"""Explicitly adopt a completed child revision into the coordinating Scope."""

from __future__ import annotations

from typing import Any

from opencollab.adapters.tools.base import Tool

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "adopt_child_changes",
        "description": "Adopt a completed child Git revision into this Agent worktree.",
        "parameters": {
            "type": "object",
            "properties": {
                "child_aid": {"type": "integer", "minimum": 0},
                "revision": {"type": "string"},
            },
            "required": ["child_aid", "revision"],
        },
    },
}


class AdoptChildChangesTool(Tool):
    name = "adopt_child_changes"
    description = TOOL_DEFINITION["function"]["description"]
    parameters = TOOL_DEFINITION["function"]["parameters"]

    def __init__(self, scheduler: Any):
        self._scheduler = scheduler

    async def execute_with_runtime(self, params: dict[str, Any], runtime: Any) -> str:
        try:
            return await self._scheduler.adopt_child_changes(
                runtime.aid, int(params["child_aid"]), str(params["revision"])
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return f"Error: {exc}"

    @staticmethod
    def get_definition() -> dict[str, Any]:
        return TOOL_DEFINITION

