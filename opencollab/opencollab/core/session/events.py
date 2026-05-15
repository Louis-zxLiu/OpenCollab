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
    def __init__(self, target: EventSink | EventCallback | None = None):
        self._sink: EventSink | None = None
        self._callback: EventCallback | None = None
        self.set_target(target)

    @property
    def sink(self) -> EventSink | None:
        return self._sink

    @property
    def on_event(self) -> EventCallback | None:
        return self._callback

    @on_event.setter
    def on_event(self, callback: EventCallback | None) -> None:
        self._sink = None
        self._callback = callback

    def set_target(self, target: EventSink | EventCallback | None) -> None:
        if target is not None and hasattr(target, "emit"):
            self._sink = target  # type: ignore[assignment]
            self._callback = None
            return
        self._sink = None
        self._callback = target  # type: ignore[assignment]

    async def emit(self, event: SessionEvent) -> None:
        try:
            if self._sink:
                result = self._sink.emit(event)
                if asyncio.iscoroutine(result):
                    await result
                return

            if self._callback:
                result = self._callback(event)
                if asyncio.iscoroutine(result):
                    await result
        except Exception:
            # Event consumers must not interrupt the agent loop.
            return

