from __future__ import annotations

from typing import Any

from opencollab.core.llm import estimate_messages_tokens
from opencollab.core.session.events import EventBus, SessionEvent
from opencollab.core.session.state import SessionState

# Compaction thresholds (ref: opencode PRUNE_MINIMUM / PRUNE_PROTECT)
DEFAULT_COMPACTION_THRESHOLD = 64_000  # tokens — trigger compaction
COMPACTION_KEEP_RECENT = 8  # keep last N messages un-summarized


class ContextCompactor:
    def __init__(
        self,
        *,
        state: SessionState,
        llm: Any,
        event_bus: EventBus,
        tracer: Any = None,
        compaction_threshold: int = DEFAULT_COMPACTION_THRESHOLD,
        auto_save=None,
    ):
        self.state = state
        self.llm = llm
        self.event_bus = event_bus
        self.tracer = tracer
        self.compaction_threshold = compaction_threshold
        self.auto_save = auto_save

    def should_compact(self) -> bool:
        # Auto-compact if context is too large (ref: opencode isOverflow)
        estimated = estimate_messages_tokens(self.state.messages)
        return estimated > self.compaction_threshold

    async def compact(self) -> None:
        """Summarize older messages to reduce context size."""
        await self.event_bus.emit(SessionEvent(type="compaction", data={"reason": "context_overflow"}))

        if len(self.state.messages) <= COMPACTION_KEEP_RECENT + 2:
            return  # Not enough messages to compact

        system_msg, older, recent = self._split_messages_for_compaction()
        summary_request, older_text = self._build_compaction_prompt(older)
        summary_text = await self._call_compaction_llm(summary_request, older_text)

        self._rebuild_compacted_messages(system_msg, older, recent, summary_text)

        if self.tracer:
            self.tracer.log_step(
                step_type="compaction",
                payload={"messages_compacted": len(older), "summary_len": len(summary_text)},
                tokens=0,
                latency=0,
            )
        if self.auto_save:
            self.auto_save()

    def _split_messages_for_compaction(self) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        # Split: system prompt | older messages | recent messages
        system_msg = self.state.messages[0]
        older = self.state.messages[1 : -COMPACTION_KEEP_RECENT]
        recent = self.state.messages[-COMPACTION_KEEP_RECENT:]
        return system_msg, older, recent

    def _build_compaction_prompt(self, older: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[str]]:
        older_text = []
        for m in older:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, str) and content:
                older_text.append(f"[{role}]: {content[:2000]}")
            elif m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    older_text.append(f"[tool_call]: {tc['function']['name']}(...)")

        summary_request = [
            {
                "role": "system",
                "content": (
                    "You are a context compaction assistant. Summarize the following conversation history into a "
                    "concise but complete summary. Preserve: all file paths mentioned, key decisions made, current "
                    "task status, and any errors encountered. Be factual and brief."
                ),
            },
            {"role": "user", "content": "\n".join(older_text)},
        ]
        return summary_request, older_text

    async def _call_compaction_llm(self, summary_request: list[dict[str, str]], older_text: list[str]) -> str:
        try:
            summary_resp = await self.llm.complete(summary_request, temperature=0.0)
            summary_text = summary_resp.content or "[compaction failed]"
            self.state.used_tokens += summary_resp.usage.total_tokens
            return summary_text
        except Exception:
            return "\n".join(older_text[:5000])  # Fallback: keep raw truncated

    def _rebuild_compacted_messages(
        self,
        system_msg: dict[str, Any],
        older: list[dict[str, Any]],
        recent: list[dict[str, Any]],
        summary_text: str,
    ) -> None:
        # Rebuild messages: system + compaction summary + recent
        self.state.messages = [
            system_msg,
            {
                "role": "system",
                "content": f"[Context compacted — summary of {len(older)} earlier messages]:\n{summary_text}",
            },
            *recent,
        ]
