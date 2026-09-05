"""Suspend an Agent until the Scheduler observes a coordination event."""

from __future__ import annotations

from typing import Any

from opencollab.adapters.tools.base import Tool

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "await_coordination",
        "description": (
            "Wait without polling until a child, teammate message, or requested "
            "effect becomes available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "wait_for": {
                    "type": "string",
                    "enum": ["any", "child", "message", "effect"],
                },
                "effect_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "maxItems": 64,
                },
                "reason": {"type": "string", "maxLength": 1024},
            },
            "required": ["wait_for"],
            "additionalProperties": False,
        },
    },
}


class AwaitCoordinationTool(Tool):
    name = "await_coordination"
    description = TOOL_DEFINITION["function"]["description"]
    parameters = TOOL_DEFINITION["function"]["parameters"]

    def __init__(self, scheduler: Any):
        self._scheduler = scheduler

    async def execute_with_runtime(self, params: dict[str, Any], runtime: Any) -> str:
        try:
            wait_for = params.get("wait_for", "any")
            effect_ids = params.get("effect_ids", ())
            reason = params.get("reason", "")
            if not isinstance(wait_for, str) or wait_for not in {"any", "child", "message", "effect"}:
                raise ValueError("wait_for must be any, child, message, or effect")
            if not isinstance(effect_ids, list) or any(not isinstance(value, str) for value in effect_ids):
                raise ValueError("effect_ids must be a list of strings")
            if wait_for == "effect" and not effect_ids:
                raise ValueError("effect_ids is required when wait_for is effect")
            if not isinstance(reason, str):
                raise ValueError("reason must be a string")
            return await self._scheduler.await_coordination(
                runtime.aid, wait_for, effect_ids, reason
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            return f"Error: {exc}"

    @staticmethod
    def get_definition() -> dict[str, Any]:
        return TOOL_DEFINITION


__all__ = ["AwaitCoordinationTool"]
