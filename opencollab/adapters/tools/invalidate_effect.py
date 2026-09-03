"""Tool used by an agent to invalidate an upstream causal effect."""

from __future__ import annotations

from typing import Any

from opencollab.adapters.tools.base import Tool

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "invalidate_effect",
        "description": (
            "Report that an upstream result is invalid. The runtime quarantines "
            "the result and its descendants and identifies affected agents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "effect_id": {
                    "type": "string",
                    "description": "Effect ID of the result to invalidate",
                },
                "reason": {
                    "type": "string",
                    "description": "Why the result is invalid",
                },
                "evidence": {
                    "type": "string",
                    "description": "Optional supporting test output or error text",
                },
            },
            "required": ["effect_id", "reason"],
        },
    },
}


class InvalidateEffectTool(Tool):
    """Quarantine an effect and ask the scheduler to restore affected agents."""

    name = "invalidate_effect"
    description = TOOL_DEFINITION["function"]["description"]
    parameters = TOOL_DEFINITION["function"]["parameters"]

    def __init__(self, scheduler: Any):
        self._scheduler = scheduler

    async def execute_with_runtime(
        self,
        params: dict[str, Any],
        runtime: Any,
    ) -> str:
        effect_id = params["effect_id"]
        reason = params["reason"][:200]
        evidence = params.get("evidence", "")[:500]
        lineage = self._scheduler._lineage
        if lineage is None:
            return "Error: lineage tracking is not enabled."

        try:
            handler = getattr(self._scheduler, "_handle_invalidation", None)
            if callable(handler):
                affected_effects = await handler(
                    effect_id,
                    runtime.aid,
                    reason,
                    evidence,
                )
            else:
                affected_effects = lineage.quarantine(
                    effect_id=effect_id,
                    reason=reason,
                    evidence=evidence,
                )
        except ValueError as exc:
            return f"Error: {exc}"

        affected_agents = lineage.compute_affected_agents(affected_effects)
        tracer = getattr(self._scheduler, "_tracer", None)
        if tracer is not None:
            tracer.log_step(
                step_type="effect_invalidated",
                payload={
                    "effect_id": effect_id,
                    "reporter_aid": runtime.aid,
                    "reason": reason,
                    "evidence": evidence,
                    "affected_effects_count": len(affected_effects),
                    "affected_agents": sorted(affected_agents),
                },
            )

        affected_summary = sorted(affected_effects)[:5]
        if len(affected_effects) > 5:
            affected_summary.append("...")
        return (
            f"Invalidated effect {effect_id} and {len(affected_effects) - 1} descendants.\n"
            f"Affected effects: {affected_summary}\n"
            f"Affected agents: {sorted(affected_agents)}"
        )

    @staticmethod
    def get_definition() -> dict[str, Any]:
        return TOOL_DEFINITION
