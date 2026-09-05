"""Report a typed verification verdict for an adopted effect."""

from __future__ import annotations

from typing import Any

from opencollab.adapters.tools.base import Tool

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "report_verification",
        "description": "Report PASS or FAIL for an adopted effect using executable evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "effect_id": {"type": "string", "minLength": 1},
                "evidence_ids": {"type": "array", "items": {"type": "string", "minLength": 1}},
                "status": {"type": "string", "enum": ["pass", "fail"]},
                "summary": {"type": "string", "minLength": 1, "maxLength": 4000},
            },
            "required": ["effect_id", "evidence_ids", "status", "summary"],
            "additionalProperties": False,
        },
    },
}


class ReportVerificationTool(Tool):
    name = "report_verification"
    description = TOOL_DEFINITION["function"]["description"]
    parameters = TOOL_DEFINITION["function"]["parameters"]

    def __init__(self, scheduler: Any):
        self._scheduler = scheduler

    async def execute_with_runtime(self, params: dict[str, Any], runtime: Any) -> str:
        try:
            return await self._scheduler.report_verification(
                runtime.aid,
                params["effect_id"],
                params["evidence_ids"],
                params["status"],
                params["summary"],
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return f"Error: {exc}"

    @staticmethod
    def get_definition() -> dict[str, Any]:
        return TOOL_DEFINITION


__all__ = ["ReportVerificationTool"]
