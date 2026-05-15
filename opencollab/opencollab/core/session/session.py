from __future__ import annotations

import asyncio
import copy
from importlib import import_module
from typing import Awaitable, Callable

from opencollab.core.agent import Agent
from opencollab.core.env import Environment, LocalEnvironment
from opencollab.core.llm import LLMResponse
from opencollab.core.session.compactor import DEFAULT_COMPACTION_THRESHOLD, ContextCompactor
from opencollab.core.session.events import EventBus, EventCallback, EventSink, SessionEvent
from opencollab.core.session.runner import SessionRunner
from opencollab.core.session.state import SessionPhase, SessionState
from opencollab.core.session.storage import SessionStore
from opencollab.core.session.tools import CallbackPermissionPolicy, PermissionPolicy, ToolCallProcessor
from opencollab.core.tracer import Tracer


class BudgetExceededError(Exception):
    pass


class LoopDetectedError(Exception):
    pass


class Session:
    """Public facade for a stateful agent session."""

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
        event_sink: EventSink | None = None,
        event_bus: EventBus | None = None,
        permission_policy: PermissionPolicy | None = None,
        llm=None,
        llm_client=None,
        store=None,
    ):
        self.agent = agent
        self.env = env or LocalEnvironment()
        self.tracer = tracer
        self.max_budget_tokens = max_budget_tokens
        self.max_steps = max_steps
        self.compaction_threshold = compaction_threshold
        self.event_bus = event_bus if event_bus is not None else EventBus(event_sink if event_sink is not None else on_event)
        if event_bus is not None and (event_sink is not None or on_event is not None):
            self.event_bus.set_target(event_sink if event_sink is not None else on_event)
        self._confirm_fn = confirm_fn
        self._permission_policy = permission_policy or (
            CallbackPermissionPolicy(confirm_fn) if confirm_fn is not None else None
        )
        self._auto_save_path = auto_save_path
        self.store = store if store is not None else SessionStore()

        system_content = agent.system_prompt
        if repo_map:
            system_content += f"\n\nProject Structure:\n{repo_map}"
        self.state = SessionState(messages=[{"role": "system", "content": system_content}])

        injected_llm = llm if llm is not None else llm_client
        if injected_llm is not None:
            self._llm = injected_llm
        else:
            llm_cls = getattr(import_module("opencollab.core.session"), "LLMClient")
            self._llm = llm_cls(
                model=agent.model,
                api_key=agent.api_key,
                base_url=agent.base_url,
                provider=agent.provider,
            )
        self._build_runtime()

    def _build_runtime(self) -> None:
        self.tool_processor = ToolCallProcessor(
            agent=self.agent,
            env=self.env,
            state=self.state,
            event_bus=self.event_bus,
            tracer=self.tracer,
            permission_policy=self.permission_policy,
        )
        self.compactor = ContextCompactor(
            state=self.state,
            llm=self._llm,
            event_bus=self.event_bus,
            tracer=self.tracer,
            compaction_threshold=self.compaction_threshold,
            auto_save=self._auto_save,
        )
        self.runner = SessionRunner(
            agent=self.agent,
            state=self.state,
            llm=self._llm,
            event_bus=self.event_bus,
            tool_processor=self.tool_processor,
            compactor=self.compactor,
            tracer=self.tracer,
            max_budget_tokens=self.max_budget_tokens,
            max_steps=self.max_steps,
            auto_save=self._auto_save,
        )

    @property
    def on_event(self) -> EventCallback | None:
        return self.event_bus.on_event

    @on_event.setter
    def on_event(self, value: EventCallback | None) -> None:
        self.event_bus.on_event = value

    @property
    def confirm_fn(self) -> Callable[[str], Awaitable[bool]] | None:
        return self._confirm_fn

    @confirm_fn.setter
    def confirm_fn(self, value: Callable[[str], Awaitable[bool]] | None) -> None:
        self._confirm_fn = value
        self.permission_policy = CallbackPermissionPolicy(value) if value is not None else None

    @property
    def permission_policy(self) -> PermissionPolicy | None:
        return self._permission_policy

    @permission_policy.setter
    def permission_policy(self, value: PermissionPolicy | None) -> None:
        self._permission_policy = value
        if hasattr(self, "tool_processor"):
            self.tool_processor.permission_policy = value

    @property
    def messages(self) -> list[dict]:
        return self.state.messages

    @messages.setter
    def messages(self, value: list[dict]) -> None:
        self.state.replace_messages(value)

    @property
    def used_tokens(self) -> int:
        return self.state.used_tokens

    @used_tokens.setter
    def used_tokens(self, value: int) -> None:
        self.state.set_used_tokens(value)

    @property
    def step_count(self) -> int:
        return self.state.step_count

    @step_count.setter
    def step_count(self, value: int) -> None:
        self.state.set_step_count(value)

    @property
    def is_done(self) -> bool:
        return self.state.is_done

    @is_done.setter
    def is_done(self, value: bool) -> None:
        self.state.mark_done(value)

    @property
    def _recent_call_hashes(self) -> list[str]:
        return self.state.recent_call_hashes

    @_recent_call_hashes.setter
    def _recent_call_hashes(self, value: list[str]) -> None:
        self.state.replace_recent_tool_hashes(value)

    @property
    def phase(self) -> SessionPhase:
        return self.state.phase

    @phase.setter
    def phase(self, value: SessionPhase) -> None:
        self.state.set_phase(value)

    async def run_loop(self, cancel_event: asyncio.Event | None = None) -> str:
        return await self.runner.run_loop(cancel_event)

    async def add_user_message(self, content: str) -> None:
        self.state.append_message({"role": "user", "content": content})
        self.state.reset_for_user_turn()
        self._auto_save()

    def snapshot(self) -> Session:
        new = Session(
            agent=self.agent,
            env=self.env,
            tracer=self.tracer,
            max_budget_tokens=self.max_budget_tokens,
            max_steps=self.max_steps,
            compaction_threshold=self.compaction_threshold,
            on_event=self.on_event,
            confirm_fn=self.confirm_fn,
            event_sink=self.event_bus.sink,
            permission_policy=self.permission_policy,
        )
        new.messages = copy.deepcopy(self.messages)
        new.used_tokens = self.used_tokens
        new.step_count = self.step_count
        return new

    def save(self, path: str) -> None:
        self.store.save(path, self.messages)

    def _auto_save(self) -> None:
        if self._auto_save_path:
            self.save(self._auto_save_path)

    @classmethod
    def load(cls, path: str, agent: Agent, **kwargs) -> Session:
        session = cls(agent=agent, **kwargs)
        session.messages = session.store.load_messages(path, agent.system_prompt)
        return session

    # Compatibility helpers for tests and older internal callers.
    async def _advance(self, cancel_event: asyncio.Event | None = None) -> None:
        await self.runner._advance(cancel_event)

    async def _step(self) -> None:
        self.state.set_phase(SessionPhase.CALLING_LLM)
        await self.runner._run_llm_call()
        await self.runner._handle_pending_response()
        if self.phase == SessionPhase.EXECUTING_TOOLS:
            await self.runner._execute_pending_tools()
        if self.phase == SessionPhase.AUTOSAVING:
            await self.runner._autosave_pending_step()

    def _should_compact(self) -> bool:
        return self.compactor.should_compact()

    async def _compact(self) -> None:
        await self.compactor.compact()

    async def _process_tool_calls(self, tool_calls: list[dict]) -> None:
        await self.tool_processor.process(tool_calls)

    def _build_tool_schemas(self) -> list[dict] | None:
        return self.runner._build_tool_schemas()

    async def _call_llm(self, tools: list[dict] | None) -> LLMResponse:
        return await self.runner._call_llm(tools)

    def _record_llm_trace(self, response: LLMResponse, latency: float) -> None:
        self.runner._record_llm_trace(response, latency)

    def _append_assistant_message(self, response: LLMResponse) -> None:
        self.runner._append_assistant_message(response)

    async def _handle_assistant_response(self, response: LLMResponse) -> None:
        if response.content:
            await self.event_bus.emit(SessionEvent(type="text_delta", data={"content": response.content}))
        if response.tool_calls:
            await self.tool_processor.process(response.tool_calls)
        else:
            self.state.mark_done()

    async def _finish_step(self, latency: float) -> None:
        await self.runner._finish_step(latency)
