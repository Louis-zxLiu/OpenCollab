from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SessionPhase(Enum):
    IDLE = "idle"
    PRECHECK = "precheck"
    COMPACTING = "compacting"
    CALLING_LLM = "calling_llm"
    HANDLING_RESPONSE = "handling_response"
    EXECUTING_TOOLS = "executing_tools"
    AUTOSAVING = "autosaving"
    DONE = "done"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"
    ERROR = "error"


@dataclass
class SessionState:
    messages: list[dict[str, Any]]
    used_tokens: int = 0
    step_count: int = 0
    is_done: bool = False
    recent_call_hashes: list[str] = field(default_factory=list)
    phase: SessionPhase = SessionPhase.IDLE

