from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol


@dataclass
class SessionEvent:
    """Lightweight event emitted by Session-compatible loops."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)


EventCallback = Callable[[SessionEvent], Awaitable[None] | None]


class EventSink(Protocol):
    async def emit(self, event: SessionEvent) -> None:
        ...


class EventBus:
    def __init__(self, on_event: EventCallback | None = None):
        self.on_event = on_event

    async def emit(self, event: SessionEvent) -> None:
        if not self.on_event:
            return

        try:
            result = self.on_event(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            # Event consumers must not interrupt the agent loop.
            return
