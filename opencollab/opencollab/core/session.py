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
from typing import Any, Callable, Awaitable

from opencollab.core.agent import Agent
from opencollab.core.env import Environment, LocalEnvironment
from opencollab.core.llm import LLMClient, LLMResponse, estimate_messages_tokens
from opencollab.core.tracer import Tracer


class BudgetExceededError(Exception):
    pass


class LoopDetectedError(Exception):
    pass


# ---------------------------------------------------------------------------
# Event callbacks for UI integration (ref: kimi-cli Wire pattern)
# ---------------------------------------------------------------------------

@dataclass
class SessionEvent:
    """Lightweight event for TUI rendering."""

    type: str  # "text_delta", "tool_start", "tool_end", "step_start", "step_end",
    #   "compaction", "budget_warning", "loop_detected", "error"
    data: dict[str, Any] = field(default_factory=dict)


EventCallback = Callable[[SessionEvent], Awaitable[None] | None]


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
        self.on_event = on_event
        self.confirm_fn = confirm_fn  # Human-in-the-loop callback
        self._auto_save_path = auto_save_path

        # State — inject repo map into system prompt if provided
        # (ref: 机制一 — 知识就是 Prompt, 直接作为 System 注入)
        system_content = agent.system_prompt
        if repo_map:
            system_content += f"\n\nProject Structure:\n{repo_map}"
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
        self.used_tokens: int = 0
        self.step_count: int = 0
        self.is_done: bool = False

        # Loop detection state
        self._recent_call_hashes: list[str] = []

        # LLM client (lazily matches agent config)
        self._llm = LLMClient(
            model=agent.model,
            api_key=agent.api_key,
            base_url=agent.base_url,
            provider=agent.provider,
        )

    # ---- Public API ----

    async def run_loop(self, cancel_event: asyncio.Event | None = None) -> str:
        """Run the agent loop until completion, cancellation, or budget exhaustion.

        Returns the final assistant text response.
        """
        try:
            while not self.is_done and self.step_count < self.max_steps:
                # Check cancellation
                if cancel_event and cancel_event.is_set():
                    self.messages.append({"role": "system", "content": "[Session interrupted by user]"})
                    await self._emit(SessionEvent(type="error", data={"reason": "cancelled"}))
                    break

                # Check budget
                if self.used_tokens >= self.max_budget_tokens:
                    raise BudgetExceededError(
                        f"Token budget exhausted: {self.used_tokens}/{self.max_budget_tokens}"
                    )

                # Auto-compact if context is too large (ref: opencode isOverflow)
                estimated = estimate_messages_tokens(self.messages)
                if estimated > self.compaction_threshold:
                    await self._compact()

                # Execute one step
                await self._step()

        except asyncio.CancelledError:
            if self.tracer:
                self.tracer.flush()
            raise
        except BudgetExceededError:
            self.messages.append({
                "role": "system",
                "content": f"[Budget exceeded: {self.used_tokens} tokens used. Session stopped.]",
            })
            await self._emit(SessionEvent(type="error", data={"reason": "budget_exceeded"}))

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

    # ---- Internal: single step ----

    async def _step(self) -> None:
        """Execute one LLM call → process tool calls → append results.

        Ref: kimi-cli KimiSoul._step, opencode SessionProcessor.
        """
        self.step_count += 1
        await self._emit(SessionEvent(type="step_start", data={"step": self.step_count}))
        start = time.monotonic()

        # Build tool schemas
        tools = self.agent.tool_schemas() or None

        # Call LLM
        response = await self._llm.complete(
            messages=self.messages,
            tools=tools,
            temperature=self.agent.temperature,
        )

        latency = time.monotonic() - start
        self.used_tokens += response.usage.total_tokens

        # Log to tracer — full content for debugging
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

        # Append assistant message
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if response.content:
            assistant_msg["content"] = response.content
        if response.tool_calls:
            assistant_msg["tool_calls"] = response.tool_calls
        self.messages.append(assistant_msg)

        # Emit text delta for TUI
        if response.content:
            await self._emit(SessionEvent(type="text_delta", data={"content": response.content}))

        # Process tool calls
        if response.tool_calls:
            await self._process_tool_calls(response.tool_calls)
        else:
            # No tool calls → agent is done
            self.is_done = True

        await self._emit(SessionEvent(type="step_end", data={"step": self.step_count, "latency": latency}))
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
                args_str = func["arguments"]
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": f"Error: invalid JSON arguments: {func['arguments'][:200]}",
                })
                continue

            # Loop detection — hash the (tool_name, args) tuple
            call_hash = hashlib.md5(json.dumps({"name": tool_name, "args": args}, sort_keys=True).encode()).hexdigest()
            self._recent_call_hashes.append(call_hash)
            if len(self._recent_call_hashes) > MAX_CALL_HASH_WINDOW:
                self._recent_call_hashes = self._recent_call_hashes[-MAX_CALL_HASH_WINDOW:]

            # Check for repeated identical calls (ref: opencode doom_loop)
            recent_same = sum(1 for h in self._recent_call_hashes[-MAX_SIMILAR_CALLS * 2 :] if h == call_hash)
            if recent_same >= MAX_SIMILAR_CALLS:
                warning = (
                    f"[Loop detected: tool '{tool_name}' called {recent_same} times with identical arguments. "
                    f"You are stuck in a loop. Try a completely different approach or ask for help.]"
                )
                self.messages.append({"role": "tool", "tool_call_id": tool_id, "content": warning})
                await self._emit(SessionEvent(type="loop_detected", data={"tool": tool_name, "count": recent_same}))
                continue

            # Find and execute tool
            tool = self.agent.find_tool(tool_name)
            if not tool:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": f"Error: unknown tool '{tool_name}'. Available: {[t.name for t in self.agent.tools]}",
                })
                continue

            await self._emit(SessionEvent(type="tool_start", data={"tool": tool_name, "args": args}))

            # Execute with timing
            start = time.monotonic()
            try:
                result = await tool.execute(args, env=self.env, confirm_fn=self.confirm_fn)
            except PermissionError as e:
                result = f"Permission denied: {e}"
            except Exception as e:
                result = f"Tool execution error: {type(e).__name__}: {e}"

            tool_latency = time.monotonic() - start

            # Truncate oversized output (ref: openclaw tool result guard)
            if len(result) > MAX_TOOL_OUTPUT_CHARS:
                truncated = result[:MAX_TOOL_OUTPUT_CHARS // 2] + \
                    f"\n\n... [{len(result) - MAX_TOOL_OUTPUT_CHARS} chars truncated] ...\n\n" + \
                    result[-MAX_TOOL_OUTPUT_CHARS // 2:]
                result = truncated

            if self.tracer:
                # Cap result in trace to 4k to keep trajectory files manageable
                trace_result = result if len(result) <= 4096 else result[:2048] + "\n...[truncated]...\n" + result[-2048:]
                self.tracer.log_step(
                    step_type="tool_exec",
                    payload={"tool": tool_name, "args": args, "result_len": len(result), "result": trace_result},
                    tokens=0,
                    latency=tool_latency,
                )

            self.messages.append({"role": "tool", "tool_call_id": tool_id, "content": result})
            await self._emit(SessionEvent(type="tool_end", data={"tool": tool_name, "latency": tool_latency}))

    # ---- Context compaction ----

    async def _compact(self) -> None:
        """Summarize older messages to reduce context size.

        Strategy (ref: opencode compaction.ts):
        1. Keep system prompt + last N messages intact
        2. Summarize everything in between via LLM
        3. Replace summarized messages with a single system message
        """
        await self._emit(SessionEvent(type="compaction", data={"reason": "context_overflow"}))

        if len(self.messages) <= COMPACTION_KEEP_RECENT + 2:
            return  # Not enough messages to compact

        # Split: system prompt | older messages | recent messages
        system_msg = self.messages[0]
        older = self.messages[1 : -COMPACTION_KEEP_RECENT]
        recent = self.messages[-COMPACTION_KEEP_RECENT:]

        # Build summary prompt
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

        try:
            summary_resp = await self._llm.complete(summary_request, temperature=0.0)
            summary_text = summary_resp.content or "[compaction failed]"
            self.used_tokens += summary_resp.usage.total_tokens
        except Exception:
            summary_text = "\n".join(older_text[:5000])  # Fallback: keep raw truncated

        # Rebuild messages: system + compaction summary + recent
        self.messages = [
            system_msg,
            {"role": "system", "content": f"[Context compacted — summary of {len(older)} earlier messages]:\n{summary_text}"},
            *recent,
        ]

        if self.tracer:
            self.tracer.log_step(
                step_type="compaction",
                payload={"messages_compacted": len(older), "summary_len": len(summary_text)},
                tokens=0,
                latency=0,
            )
        self._auto_save()

    # ---- Event emission ----

    async def _emit(self, event: SessionEvent) -> None:
        if self.on_event:
            try:
                result = self.on_event(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                # UI callback errors should not crash the agent loop.
                return
