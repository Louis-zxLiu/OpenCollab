"""Session — the sole stateful container.

First Principle: Session = Agent + Message List + Environment.
Session drives the agent loop: call LLM → execute tools → append results → repeat.

Addresses three critical concerns:
1. Context compaction — auto-summarize when tokens exceed threshold
2. Budget enforcement — hard stop when token budget exhausted
3. Loop breaking — detect repeated identical tool calls and force interrupt
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from opencollab.core.agent import Agent
from opencollab.core.env import Environment, LocalEnvironment
from opencollab.core.events import EventBus, EventCallback, SessionEvent
from opencollab.core.llm import LLMClient, LLMResponse, estimate_messages_tokens
from opencollab.core.tracer import Tracer


class BudgetExceededError(Exception):
    pass


class LoopDetectedError(Exception):
    pass


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


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

# Compaction thresholds (ref: opencode PRUNE_MINIMUM / PRUNE_PROTECT)
DEFAULT_COMPACTION_THRESHOLD = 64_000  # tokens — trigger compaction
COMPACTION_KEEP_RECENT = 8  # keep last N messages un-summarized

# Loop detection (ref: opencode doom_loop detection — 3 identical calls)
MAX_SIMILAR_CALLS = 3
MAX_CALL_HASH_WINDOW = 200

# Output truncation for tool results (ref: openclaw truncateOversizedToolResults)
MAX_TOOL_OUTPUT_CHARS = 16_000


class Session:
    """Stateful conversation container. Drives the agent loop.

    Design refs:
    - kimi-cli: KimiSoul._agent_loop / _step — async step loop with max_steps
    - opencode: prompt.ts loop — step → tool exec → compaction check → repeat
    - openclaw: runEmbeddedAttempt — retry + compaction + budget
    """

    def __init__(
        self,
        agent: Agent,
        env: Environment | None = None,
        tracer: Tracer | None = None,
        max_budget_tokens: int = 200_000,
        max_steps: int = 100,
        compaction_threshold: int = DEFAULT_COMPACTION_THRESHOLD,
        on_event: EventCallback | None = None,
        confirm_fn: Callable[[str], Awaitable[bool]] | None = None,
        repo_map: str | None = None,
        auto_save_path: str | None = None,
    ):
        self.agent = agent
        self.env = env or LocalEnvironment()
        self.tracer = tracer
        self.max_budget_tokens = max_budget_tokens
        self.max_steps = max_steps
        self.compaction_threshold = compaction_threshold
        self.event_bus = EventBus(on_event)
        self.confirm_fn = confirm_fn  # Human-in-the-loop callback
        self._auto_save_path = auto_save_path

        # State — inject repo map into system prompt if provided
        # (ref: 机制一 — 知识就是 Prompt, 直接作为 System 注入)
        system_content = agent.system_prompt
        if repo_map:
            system_content += f"\n\nProject Structure:\n{repo_map}"
        self.state = SessionState(messages=[{"role": "system", "content": system_content}])
        self._pending_response: LLMResponse | None = None
        self._pending_latency: float = 0.0

        # LLM client (lazily matches agent config)
        self._llm = LLMClient(
            model=agent.model,
            api_key=agent.api_key,
            base_url=agent.base_url,
            provider=agent.provider,
        )

    # ---- Public API ----

    @property
    def on_event(self) -> EventCallback | None:
        return self.event_bus.on_event

    @on_event.setter
    def on_event(self, value: EventCallback | None) -> None:
        self.event_bus.on_event = value

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.state.messages

    @messages.setter
    def messages(self, value: list[dict[str, Any]]) -> None:
        self.state.messages = value

    @property
    def used_tokens(self) -> int:
        return self.state.used_tokens

    @used_tokens.setter
    def used_tokens(self, value: int) -> None:
        self.state.used_tokens = value

    @property
    def step_count(self) -> int:
        return self.state.step_count

    @step_count.setter
    def step_count(self, value: int) -> None:
        self.state.step_count = value

    @property
    def is_done(self) -> bool:
        return self.state.is_done

    @is_done.setter
    def is_done(self, value: bool) -> None:
        self.state.is_done = value

    @property
    def _recent_call_hashes(self) -> list[str]:
        return self.state.recent_call_hashes

    @_recent_call_hashes.setter
    def _recent_call_hashes(self, value: list[str]) -> None:
        self.state.recent_call_hashes = value

    @property
    def phase(self) -> SessionPhase:
        return self.state.phase

    @phase.setter
    def phase(self, value: SessionPhase) -> None:
        self.state.phase = value

    async def run_loop(self, cancel_event: asyncio.Event | None = None) -> str:
        """Run the agent loop until completion, cancellation, or budget exhaustion.

        Returns the final assistant text response.
        """
        try:
            self.phase = SessionPhase.IDLE
            while not self.is_done and not self._is_terminal_phase() and self.step_count < self.max_steps:
                await self._advance(cancel_event)

        except asyncio.CancelledError:
            if self.tracer:
                self.tracer.flush()
            raise
        except Exception:
            self.phase = SessionPhase.ERROR
            raise

        # Extract final assistant response
        for msg in reversed(self.messages):
            if msg["role"] == "assistant" and msg.get("content"):
                return msg["content"]
        return ""

    async def add_user_message(self, content: str) -> None:
        """Append a user message to context."""
        self.messages.append({"role": "user", "content": content})
        self.is_done = False
        self._recent_call_hashes.clear()
        self._auto_save()

    def snapshot(self) -> Session:
        """Deep copy for branching / time-travel (ref: design doc snapshot)."""
        new = Session(
            agent=self.agent,
            env=self.env,
            tracer=self.tracer,
            max_budget_tokens=self.max_budget_tokens,
            max_steps=self.max_steps,
            compaction_threshold=self.compaction_threshold,
            on_event=self.on_event,
            confirm_fn=self.confirm_fn,
        )
        new.messages = copy.deepcopy(self.messages)
        new.used_tokens = self.used_tokens
        new.step_count = self.step_count
        return new

    def save(self, path: str) -> None:
        """Persist session as JSONL (ref: design doc — serialize to JSONL)."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            for msg in self.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    def _auto_save(self) -> None:
        """Persist to the configured auto-save path when enabled."""
        if self._auto_save_path:
            self.save(self._auto_save_path)

    @classmethod
    def load(cls, path: str, agent: Agent, **kwargs) -> Session:
        """Restore session from JSONL."""
        session = cls(agent=agent, **kwargs)
        session.messages = []
        allowed_roles = {"system", "user", "assistant", "tool"}
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    msg = json.loads(line)
                    if not isinstance(msg, dict):
                        raise ValueError(f"Invalid message at line {lineno}: expected object")
                    role = msg.get("role")
                    if role not in allowed_roles:
                        raise ValueError(f"Invalid message role at line {lineno}: {role}")
                    session.messages.append(msg)

        if not session.messages:
            session.messages = [{"role": "system", "content": agent.system_prompt}]
        return session

    # ---- Internal: state machine ----

    def _is_terminal_phase(self) -> bool:
        return self.phase in {
            SessionPhase.DONE,
            SessionPhase.CANCELLED,
            SessionPhase.BUDGET_EXCEEDED,
            SessionPhase.ERROR,
        }

    async def _advance(self, cancel_event: asyncio.Event | None = None) -> None:
        match self.state.phase:
            case SessionPhase.IDLE:
                self.state.phase = SessionPhase.PRECHECK
            case SessionPhase.PRECHECK:
                await self._precheck(cancel_event)
            case SessionPhase.COMPACTING:
                await self._run_compaction()
            case SessionPhase.CALLING_LLM:
                await self._run_llm_call()
            case SessionPhase.HANDLING_RESPONSE:
                await self._handle_pending_response()
            case SessionPhase.EXECUTING_TOOLS:
                await self._execute_pending_tools()
            case SessionPhase.AUTOSAVING:
                await self._autosave_pending_step()
            case _:
                self.state.phase = SessionPhase.ERROR
                raise RuntimeError(f"Cannot advance terminal phase: {self.state.phase.value}")

    async def _precheck(self, cancel_event: asyncio.Event | None) -> None:
        if cancel_event and cancel_event.is_set():
            self.messages.append({"role": "system", "content": "[Session interrupted by user]"})
            await self.event_bus.emit(SessionEvent(type="error", data={"reason": "cancelled"}))
            self.phase = SessionPhase.CANCELLED
            return

        if self.used_tokens >= self.max_budget_tokens:
            self.messages.append({
                "role": "system",
                "content": f"[Budget exceeded: {self.used_tokens} tokens used. Session stopped.]",
            })
            await self.event_bus.emit(SessionEvent(type="error", data={"reason": "budget_exceeded"}))
            self.phase = SessionPhase.BUDGET_EXCEEDED
            return

        if self._should_compact():
            self.phase = SessionPhase.COMPACTING
            return

        self.phase = SessionPhase.CALLING_LLM

    async def _run_compaction(self) -> None:
        await self._compact()
        self.phase = SessionPhase.CALLING_LLM

    async def _run_llm_call(self) -> None:
        self.step_count += 1
        await self.event_bus.emit(SessionEvent(type="step_start", data={"step": self.step_count}))
        start = time.monotonic()

        tools = self._build_tool_schemas()
        response = await self._call_llm(tools)
        latency = time.monotonic() - start
        self.used_tokens += response.usage.total_tokens

        self._record_llm_trace(response, latency)
        self._append_assistant_message(response)
        self._pending_response = response
        self._pending_latency = latency
        self.phase = SessionPhase.HANDLING_RESPONSE

    async def _handle_pending_response(self) -> None:
        response = self._pending_response
        if response is None:
            self.phase = SessionPhase.ERROR
            raise RuntimeError("Cannot handle assistant response before calling LLM")

        if response.content:
            await self.event_bus.emit(SessionEvent(type="text_delta", data={"content": response.content}))

        if response.tool_calls:
            self.phase = SessionPhase.EXECUTING_TOOLS
            return

        self.is_done = True
        await self._finish_step(self._pending_latency)
        self._clear_pending_step()
        self.phase = SessionPhase.DONE

    async def _execute_pending_tools(self) -> None:
        response = self._pending_response
        if response is None:
            self.phase = SessionPhase.ERROR
            raise RuntimeError("Cannot execute tools before calling LLM")

        await self._process_tool_calls(response.tool_calls)
        self.phase = SessionPhase.AUTOSAVING

    async def _autosave_pending_step(self) -> None:
        await self._finish_step(self._pending_latency)
        self._clear_pending_step()
        self.phase = SessionPhase.DONE if self.is_done else SessionPhase.PRECHECK

    def _clear_pending_step(self) -> None:
        self._pending_response = None
        self._pending_latency = 0.0

    # ---- Internal: single step ----

    async def _step(self) -> None:
        """Execute one LLM call → process tool calls → append results.

        Ref: kimi-cli KimiSoul._step, opencode SessionProcessor.
        """
        self.step_count += 1
        await self.event_bus.emit(SessionEvent(type="step_start", data={"step": self.step_count}))
        start = time.monotonic()

        tools = self._build_tool_schemas()
        response = await self._call_llm(tools)
        latency = time.monotonic() - start
        self.used_tokens += response.usage.total_tokens

        self._record_llm_trace(response, latency)
        self._append_assistant_message(response)
        await self._handle_assistant_response(response)
        await self._finish_step(latency)

    def _build_tool_schemas(self) -> list[dict] | None:
        return self.agent.tool_schemas() or None

    async def _call_llm(self, tools: list[dict] | None) -> LLMResponse:
        return await self._llm.complete(
            messages=self.messages,
            tools=tools,
            temperature=self.agent.temperature,
        )

    def _record_llm_trace(self, response: LLMResponse, latency: float) -> None:
        if self.tracer:
            tool_calls_log = None
            if response.tool_calls:
                tool_calls_log = [
                    {"id": tc.get("id"), "name": tc.get("function", {}).get("name"),
                     "arguments": tc.get("function", {}).get("arguments", "")}
                    for tc in response.tool_calls
                ]
            self.tracer.log_step(
                step_type="llm_call",
                payload={
                    "model": self.agent.model,
                    "finish_reason": response.finish_reason,
                    "content": response.content,
                    "tool_calls": tool_calls_log,
                },
                tokens=response.usage.total_tokens,
                latency=latency,
            )

    def _append_assistant_message(self, response: LLMResponse) -> None:
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if response.content:
            assistant_msg["content"] = response.content
        if response.tool_calls:
            assistant_msg["tool_calls"] = response.tool_calls
        self.messages.append(assistant_msg)

    async def _handle_assistant_response(self, response: LLMResponse) -> None:
        if response.content:
            await self.event_bus.emit(SessionEvent(type="text_delta", data={"content": response.content}))

        if response.tool_calls:
            await self._process_tool_calls(response.tool_calls)
        else:
            # No tool calls → agent is done
            self.is_done = True

    async def _finish_step(self, latency: float) -> None:
        await self.event_bus.emit(SessionEvent(type="step_end", data={"step": self.step_count, "latency": latency}))
        self._auto_save()

    async def _process_tool_calls(self, tool_calls: list[dict]) -> None:
        """Execute tool calls and append results.

        Includes:
        - Loop detection (ref: opencode doom_loop — 3 identical calls)
        - Output truncation (ref: openclaw truncateOversizedToolResults)
        - Human-in-the-loop confirmation
        """
        for tc in tool_calls:
            func = tc["function"]
            tool_name = func["name"]
            tool_id = tc["id"]

            try:
                args = self._parse_tool_args(func)
            except json.JSONDecodeError:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": f"Error: invalid JSON arguments: {func['arguments'][:200]}",
                })
                continue

            recent_same = self._detect_repeated_tool_call(tool_name, args)
            if recent_same >= MAX_SIMILAR_CALLS:
                warning = (
                    f"[Loop detected: tool '{tool_name}' called {recent_same} times with identical arguments. "
                    f"You are stuck in a loop. Try a completely different approach or ask for help.]"
                )
                self.messages.append({"role": "tool", "tool_call_id": tool_id, "content": warning})
                await self.event_bus.emit(
                    SessionEvent(type="loop_detected", data={"tool": tool_name, "count": recent_same})
                )
                continue

            tool = self._find_tool(tool_name)
            if not tool:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": f"Error: unknown tool '{tool_name}'. Available: {[t.name for t in self.agent.tools]}",
                })
                continue

            await self.event_bus.emit(SessionEvent(type="tool_start", data={"tool": tool_name, "args": args}))

            result, tool_latency = await self._execute_tool(tool, args)
            result = self._truncate_tool_result(result)

            if self.tracer:
                # Cap result in trace to 4k to keep trajectory files manageable
                trace_result = result if len(result) <= 4096 else result[:2048] + "\n...[truncated]...\n" + result[-2048:]
                self.tracer.log_step(
                    step_type="tool_exec",
                    payload={"tool": tool_name, "args": args, "result_len": len(result), "result": trace_result},
                    tokens=0,
                    latency=tool_latency,
                )

            self._append_tool_result(tool_id, result)
            await self.event_bus.emit(SessionEvent(type="tool_end", data={"tool": tool_name, "latency": tool_latency}))

    def _parse_tool_args(self, func: dict) -> dict:
        args_str = func["arguments"]
        return json.loads(args_str) if args_str else {}

    def _detect_repeated_tool_call(self, tool_name: str, args: dict) -> int:
        # Loop detection — hash the (tool_name, args) tuple
        call_hash = hashlib.md5(json.dumps({"name": tool_name, "args": args}, sort_keys=True).encode()).hexdigest()
        self._recent_call_hashes.append(call_hash)
        if len(self._recent_call_hashes) > MAX_CALL_HASH_WINDOW:
            self._recent_call_hashes = self._recent_call_hashes[-MAX_CALL_HASH_WINDOW:]

        # Check for repeated identical calls (ref: opencode doom_loop)
        return sum(1 for h in self._recent_call_hashes[-MAX_SIMILAR_CALLS * 2 :] if h == call_hash)

    def _find_tool(self, tool_name: str):
        return self.agent.find_tool(tool_name)

    async def _execute_tool(self, tool, args: dict) -> tuple[str, float]:
        start = time.monotonic()
        try:
            result = await tool.execute(args, env=self.env, confirm_fn=self.confirm_fn)
        except PermissionError as e:
            result = f"Permission denied: {e}"
        except Exception as e:
            result = f"Tool execution error: {type(e).__name__}: {e}"

        return result, time.monotonic() - start

    def _truncate_tool_result(self, result: str) -> str:
        if len(result) > MAX_TOOL_OUTPUT_CHARS:
            return result[:MAX_TOOL_OUTPUT_CHARS // 2] + \
                f"\n\n... [{len(result) - MAX_TOOL_OUTPUT_CHARS} chars truncated] ...\n\n" + \
                result[-MAX_TOOL_OUTPUT_CHARS // 2:]
        return result

    def _append_tool_result(self, tool_id: str, result: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": tool_id, "content": result})

    # ---- Context compaction ----

    def _should_compact(self) -> bool:
        # Auto-compact if context is too large (ref: opencode isOverflow)
        estimated = estimate_messages_tokens(self.messages)
        return estimated > self.compaction_threshold

    async def _compact(self) -> None:
        """Summarize older messages to reduce context size.

        Strategy (ref: opencode compaction.ts):
        1. Keep system prompt + last N messages intact
        2. Summarize everything in between via LLM
        3. Replace summarized messages with a single system message
        """
        await self.event_bus.emit(SessionEvent(type="compaction", data={"reason": "context_overflow"}))

        if len(self.messages) <= COMPACTION_KEEP_RECENT + 2:
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
        self._auto_save()

    def _split_messages_for_compaction(self) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        # Split: system prompt | older messages | recent messages
        system_msg = self.messages[0]
        older = self.messages[1 : -COMPACTION_KEEP_RECENT]
        recent = self.messages[-COMPACTION_KEEP_RECENT:]
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
            {"role": "system", "content": "You are a context compaction assistant. Summarize the following conversation history into a concise but complete summary. Preserve: all file paths mentioned, key decisions made, current task status, and any errors encountered. Be factual and brief."},
            {"role": "user", "content": "\n".join(older_text)},
        ]
        return summary_request, older_text

    async def _call_compaction_llm(self, summary_request: list[dict[str, str]], older_text: list[str]) -> str:
        try:
            summary_resp = await self._llm.complete(summary_request, temperature=0.0)
            summary_text = summary_resp.content or "[compaction failed]"
            self.used_tokens += summary_resp.usage.total_tokens
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
        self.messages = [
            system_msg,
            {"role": "system", "content": f"[Context compacted — summary of {len(older)} earlier messages]:\n{summary_text}"},
            *recent,
        ]
