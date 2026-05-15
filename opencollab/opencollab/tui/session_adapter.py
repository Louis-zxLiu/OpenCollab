from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from opencollab.core.session import EventSink, PermissionPolicy, SessionEvent


class TuiEventSink(EventSink):
    def __init__(self, tui):
        self.tui = tui

    async def emit(self, event: SessionEvent) -> None:
        result = self.tui.event_handler(event)
        if asyncio.iscoroutine(result):
            await result


class TuiPermissionPolicy(PermissionPolicy):
    def __init__(self, confirm_fn: Callable[[str], Awaitable[bool]]):
        self._confirm_fn = confirm_fn

    async def confirm(self, prompt: str) -> bool:
        return await self._confirm_fn(prompt)

