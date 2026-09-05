"""Publish an immutable workspace Effect to the source workspace."""

from __future__ import annotations

from typing import Any

from opencollab.adapters.tools.base import Tool

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "publish_effect",
        "description": "Publish a visible immutable workspace Effect to the source workspace.",
        "parameters": {
            "type": "object",
            "properties": {"effect_id": {"type": "string"}},
            "required": ["effect_id"],
        },
    },
}


class PublishEffectTool(Tool):
    name = "publish_effect"
    description = TOOL_DEFINITION["function"]["description"]
    parameters = TOOL_DEFINITION["function"]["parameters"]

    def __init__(self, scheduler: Any):
        self._scheduler = scheduler

    async def execute_with_runtime(self, params: dict[str, Any], runtime: Any) -> str:
        try:
            effect_id = str(params["effect_id"]).strip()
            if not effect_id:
                raise ValueError("effect_id is required")
            return await self._scheduler.publish_effect(runtime.aid, effect_id)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return f"Error: {exc}"

    @staticmethod
    def get_definition() -> dict[str, Any]:
        return TOOL_DEFINITION


__all__ = ["PublishEffectTool"]
