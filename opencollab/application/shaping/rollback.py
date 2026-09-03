"""Read-time filtering for quarantined causal effects."""

from __future__ import annotations

from typing import Any

from opencollab.application.rollback import RollbackService


class RollbackQuarantineShaper:
    """Replace invalid evidence with an auditable marker before an LLM call."""

    def __init__(self, rollback_service: RollbackService):
        self._rollback = rollback_service

    def shape(self, messages: list[dict[str, Any]], **_kwargs: Any) -> list[dict[str, Any]]:
        shaped: list[dict[str, Any]] = []
        for message in messages:
            lineage = message.get("_lineage")
            effect = lineage.get("effect") if isinstance(lineage, dict) else None
            effect_id = effect.get("effect_id") if isinstance(effect, dict) else None
            if not effect_id or not self._rollback.is_quarantined(effect_id):
                shaped.append(message)
                continue
            shaped.append({**message, "content": f"[quarantined evidence: {effect_id}]"})
        return shaped
