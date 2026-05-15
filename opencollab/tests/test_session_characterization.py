import asyncio
import copy

import pytest

from opencollab.core import session as session_mod
from opencollab.core.events import EventBus as CompatEventBus
from opencollab.core.events import SessionEvent as CompatSessionEvent
from opencollab.core.llm import LLMResponse, Usage
from opencollab.core.session import EventBus, Session, SessionEvent, SessionStore


def run(coro):
    return asyncio.run(coro)


def llm_response(content=None, tool_calls=None, input_tokens=1, output_tokens=1, finish_reason="stop"):
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        finish_reason=finish_reason,
    )


def tool_call(call_id="call-1", name="fake_tool", arguments='{"value": 1}'):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class FakeLLMClient:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []

    async def complete(self, messages, tools=None, temperature=0.0):
        self.calls.append({
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
            "temperature": temperature,
        })
        if not self.responses:
            raise AssertionError("FakeLLMClient received an unexpected complete() call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeAgent:
    def __init__(self, tools=None):
        self.name = "fake-agent"
        self.system_prompt = "You are a fake agent."
        self.tools = tools or []
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.api_key = "fake-key"
        self.base_url = "https://fake.invalid"
        self.temperature = 0.25

    def tool_schemas(self):
        return [tool.to_openai_schema() for tool in self.tools]

    def find_tool(self, name):
        for tool in self.tools:
            if tool.name.lower() == name.lower():
                return tool
        return None


class FakeTool:
    def __init__(self, name="fake_tool", result="tool result", exc=None):
        self.name = name
        self.result = result
        self.exc = exc
        self.calls = []

    def to_openai_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "fake tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    async def execute(self, args, env=None, confirm_fn=None):
        self.calls.append({"args": copy.deepcopy(args), "env": env, "confirm_fn": confirm_fn})
        if self.exc:
            raise self.exc
        return self.result(args) if callable(self.result) else self.result


class FakeTracer:
    def __init__(self):
        self.steps = []
        self.flush_count = 0

    def log_step(self, step_type, payload, tokens, latency):
        self.steps.append({
            "step_type": step_type,
            "payload": copy.deepcopy(payload),
            "tokens": tokens,
            "latency": latency,
        })

    def flush(self):
        self.flush_count += 1


@pytest.fixture
def install_fake_llm(monkeypatch):
    def _install(fake_llm):
        monkeypatch.setattr(session_mod, "LLMClient", lambda **kwargs: fake_llm)
        return fake_llm

    return _install


def event_collector():
    events = []

    def on_event(event):
        events.append(event)

    return events, on_event


def test_session_package_and_compat_event_imports_are_preserved():
    assert Session is session_mod.Session
    assert SessionEvent is session_mod.SessionEvent
    assert CompatEventBus is EventBus
    assert CompatSessionEvent is SessionEvent


def test_event_bus_accepts_sink_and_swallows_sink_exception():
    class BadSink:
        async def emit(self, _event):
            raise RuntimeError("sink failed")

    run(EventBus(BadSink()).emit(SessionEvent(type="error", data={"reason": "boom"})))


def test_run_loop_cancellation_appends_interruption_and_emits_error(install_fake_llm):
    fake_llm = install_fake_llm(FakeLLMClient())
    events, on_event = event_collector()
    session = session_mod.Session(agent=FakeAgent(), on_event=on_event)
    cancel_event = asyncio.Event()
    cancel_event.set()

    result = run(session.run_loop(cancel_event=cancel_event))

    assert result == ""
    assert fake_llm.calls == []
    assert session.messages[-1] == {"role": "system", "content": "[Session interrupted by user]"}
    assert session.is_done is False
    assert [(event.type, event.data) for event in events] == [("error", {"reason": "cancelled"})]


def test_budget_exceeded_stops_before_llm_call_and_emits_error(install_fake_llm):
    fake_llm = install_fake_llm(FakeLLMClient())
    events, on_event = event_collector()
    session = session_mod.Session(agent=FakeAgent(), max_budget_tokens=10, on_event=on_event)
    session.used_tokens = 10

    result = run(session.run_loop())

    assert result == ""
    assert fake_llm.calls == []
    assert session.is_done is False
    assert session.messages[-1] == {
        "role": "system",
        "content": "[Budget exceeded: 10 tokens used. Session stopped.]",
    }
    assert [(event.type, event.data) for event in events] == [("error", {"reason": "budget_exceeded"})]


def test_compaction_trigger_summarizes_older_messages_then_runs_step(install_fake_llm):
    fake_llm = install_fake_llm(FakeLLMClient([
        llm_response(content="compact summary", input_tokens=3, output_tokens=4),
        llm_response(content="final after compaction", input_tokens=5, output_tokens=6),
    ]))
    tracer = FakeTracer()
    events, on_event = event_collector()
    session = session_mod.Session(
        agent=FakeAgent(),
        tracer=tracer,
        compaction_threshold=0,
        on_event=on_event,
    )
    session.messages.extend({"role": "user", "content": f"message {idx}"} for idx in range(10))
    original_recent = copy.deepcopy(session.messages[-session_mod.COMPACTION_KEEP_RECENT:])

    result = run(session.run_loop())

    assert result == "final after compaction"
    assert fake_llm.calls[0]["temperature"] == 0.0
    assert fake_llm.calls[0]["messages"][0]["content"].startswith("You are a context compaction assistant.")
    assert fake_llm.calls[1]["messages"][1]["content"] == (
        "[Context compacted — summary of 2 earlier messages]:\ncompact summary"
    )
    assert session.messages[0] == {"role": "system", "content": "You are a fake agent."}
    assert session.messages[2:10] == original_recent
    assert session.used_tokens == 18
    assert "compaction" in [event.type for event in events]
    assert tracer.steps[0]["step_type"] == "compaction"


def test_no_tool_calls_marks_done_and_emits_text_delta(install_fake_llm):
    fake_llm = install_fake_llm(FakeLLMClient([
        llm_response(content="plain answer", input_tokens=2, output_tokens=3),
    ]))
    tracer = FakeTracer()
    events, on_event = event_collector()
    session = session_mod.Session(agent=FakeAgent(), tracer=tracer, on_event=on_event)

    result = run(session.run_loop())

    assert result == "plain answer"
    assert session.is_done is True
    assert session.step_count == 1
    assert session.used_tokens == 5
    assert session.messages[-1] == {"role": "assistant", "content": "plain answer"}
    assert [event.type for event in events] == ["step_start", "text_delta", "step_end"]
    assert fake_llm.calls[0]["tools"] is None
    assert tracer.steps[0]["step_type"] == "llm_call"
    assert tracer.steps[0]["payload"]["content"] == "plain answer"


def test_session_accepts_explicit_llm_client():
    fake_llm = FakeLLMClient([
        llm_response(content="explicit llm answer", input_tokens=2, output_tokens=3),
    ])
    session = session_mod.Session(agent=FakeAgent(), llm=fake_llm)

    assert session._llm is fake_llm
    assert session.runner.llm is fake_llm
    assert session.compactor.llm is fake_llm

    result = run(session.run_loop())

    assert result == "explicit llm answer"
    assert fake_llm.calls


def test_run_loop_when_already_done_returns_latest_assistant_without_llm_call(install_fake_llm):
    fake_llm = install_fake_llm(FakeLLMClient())
    session = session_mod.Session(agent=FakeAgent())
    session.messages.append({"role": "assistant", "content": "already done"})
    session.is_done = True

    result = run(session.run_loop())

    assert result == "already done"
    assert fake_llm.calls == []


def test_tool_calls_execute_append_tool_result_and_continue(install_fake_llm):
    tool = FakeTool(result=lambda args: f"echo {args['value']}")
    fake_llm = install_fake_llm(FakeLLMClient([
        llm_response(tool_calls=[tool_call(arguments='{"value": 7}')], finish_reason="tool_calls"),
        llm_response(content="done"),
    ]))
    tracer = FakeTracer()
    events, on_event = event_collector()

    async def confirm_fn(_prompt):
        return True

    session = session_mod.Session(
        agent=FakeAgent(tools=[tool]),
        tracer=tracer,
        on_event=on_event,
        confirm_fn=confirm_fn,
    )

    result = run(session.run_loop())

    assert result == "done"
    assert session.step_count == 2
    assert session.confirm_fn is confirm_fn
    assert tool.calls[0]["args"] == {"value": 7}
    assert tool.calls[0]["env"] is session.env
    assert tool.calls[0]["confirm_fn"] is not None
    assert run(tool.calls[0]["confirm_fn"]("confirm?")) is True
    assert session.messages[1]["role"] == "assistant"
    assert session.messages[1]["tool_calls"][0]["function"]["name"] == "fake_tool"
    assert session.messages[2] == {"role": "tool", "tool_call_id": "call-1", "content": "echo 7"}
    assert session.messages[3] == {"role": "assistant", "content": "done"}
    assert [event.type for event in events] == [
        "step_start",
        "tool_start",
        "tool_end",
        "step_end",
        "step_start",
        "text_delta",
        "step_end",
    ]
    assert [step["step_type"] for step in tracer.steps] == ["llm_call", "tool_exec", "llm_call"]
    assert fake_llm.calls[0]["tools"][0]["function"]["name"] == "fake_tool"


def test_invalid_json_tool_arguments_append_error_tool_message_without_execution(install_fake_llm):
    tool = FakeTool()
    install_fake_llm(FakeLLMClient([
        llm_response(tool_calls=[tool_call(arguments="{not json")], finish_reason="tool_calls"),
        llm_response(content="recovered"),
    ]))
    events, on_event = event_collector()
    session = session_mod.Session(agent=FakeAgent(tools=[tool]), on_event=on_event)

    result = run(session.run_loop())

    assert result == "recovered"
    assert tool.calls == []
    assert session.messages[2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "Error: invalid JSON arguments: {not json",
    }
    assert "tool_start" not in [event.type for event in events]


def test_unknown_tool_appends_available_tools_error(install_fake_llm):
    known_tool = FakeTool(name="known_tool")
    install_fake_llm(FakeLLMClient([
        llm_response(tool_calls=[tool_call(name="missing_tool", arguments='{"value": 1}')], finish_reason="tool_calls"),
        llm_response(content="after unknown"),
    ]))
    events, on_event = event_collector()
    session = session_mod.Session(agent=FakeAgent(tools=[known_tool]), on_event=on_event)

    result = run(session.run_loop())

    assert result == "after unknown"
    assert known_tool.calls == []
    assert session.messages[2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "Error: unknown tool 'missing_tool'. Available: ['known_tool']",
    }
    assert "tool_start" not in [event.type for event in events]


def test_loop_detection_skips_third_identical_tool_call(install_fake_llm):
    tool = FakeTool(result="same result")
    install_fake_llm(FakeLLMClient([
        llm_response(tool_calls=[tool_call(call_id="call-1", arguments='{"value": 1}')], finish_reason="tool_calls"),
        llm_response(tool_calls=[tool_call(call_id="call-2", arguments='{"value": 1}')], finish_reason="tool_calls"),
        llm_response(tool_calls=[tool_call(call_id="call-3", arguments='{"value": 1}')], finish_reason="tool_calls"),
        llm_response(content="escaped loop"),
    ]))
    events, on_event = event_collector()
    session = session_mod.Session(agent=FakeAgent(tools=[tool]), on_event=on_event)

    result = run(session.run_loop())

    assert result == "escaped loop"
    assert len(tool.calls) == 2
    loop_messages = [msg for msg in session.messages if msg.get("content", "").startswith("[Loop detected:")]
    assert loop_messages == [{
        "role": "tool",
        "tool_call_id": "call-3",
        "content": (
            "[Loop detected: tool 'fake_tool' called 3 times with identical arguments. "
            "You are stuck in a loop. Try a completely different approach or ask for help.]"
        ),
    }]
    assert [(event.type, event.data) for event in events if event.type == "loop_detected"] == [
        ("loop_detected", {"tool": "fake_tool", "count": 3})
    ]


def test_tool_output_is_truncated_before_appending_to_messages(install_fake_llm):
    long_result = (
        "a" * (session_mod.MAX_TOOL_OUTPUT_CHARS // 2)
        + "b" * 123
        + "c" * (session_mod.MAX_TOOL_OUTPUT_CHARS // 2)
    )
    tool = FakeTool(result=long_result)
    install_fake_llm(FakeLLMClient([
        llm_response(tool_calls=[tool_call(arguments='{"value": 1}')], finish_reason="tool_calls"),
        llm_response(content="after truncation"),
    ]))
    session = session_mod.Session(agent=FakeAgent(tools=[tool]))

    result = run(session.run_loop())

    assert result == "after truncation"
    tool_output = session.messages[2]["content"]
    assert tool_output != long_result
    assert tool_output.startswith("a" * (session_mod.MAX_TOOL_OUTPUT_CHARS // 2))
    assert "\n\n... [123 chars truncated] ...\n\n" in tool_output
    assert tool_output.endswith("c" * (session_mod.MAX_TOOL_OUTPUT_CHARS // 2))


def test_event_callback_exception_is_swallowed(install_fake_llm):
    install_fake_llm(FakeLLMClient([
        llm_response(content="answer despite bad callback"),
    ]))

    def bad_on_event(_event):
        raise RuntimeError("callback failed")

    session = session_mod.Session(agent=FakeAgent(), on_event=bad_on_event)

    result = run(session.run_loop())

    assert result == "answer despite bad callback"
    assert session.is_done is True
    assert session.messages[-1] == {"role": "assistant", "content": "answer despite bad callback"}


# Characterizes historical/current behavior: mutating runtime config after
# Session construction desyncs facade fields from already-built runtime objects.
# This is not recommended; new code should inject env/max_steps via constructors.
def test_session_runtime_config_desync_after_mutating_env_and_max_steps(install_fake_llm):
    install_fake_llm(FakeLLMClient())
    old_env = object()
    new_env = object()
    old_max_steps = 7
    new_max_steps = 3
    session = session_mod.Session(
        agent=FakeAgent(),
        env=old_env,
        max_steps=old_max_steps,
    )

    session.env = new_env
    session.max_steps = new_max_steps

    assert session.env is new_env
    assert session.tool_processor.env is old_env
    assert session.runner.max_steps == old_max_steps


def test_team_lead_session_runtime_uses_constructor_env_and_max_steps(install_fake_llm):
    install_fake_llm(FakeLLMClient())
    from opencollab.team.orchestrator import Team

    lead_env = object()
    lead_max_steps = 7

    team = Team(
        workspace=".",
        model="fake-model",
        provider="fake-provider",
        api_key="fake-key",
        lead_env=lead_env,
        lead_max_steps=lead_max_steps,
        use_worktrees=False,
    )

    assert team.lead_session.env is lead_env
    assert team.lead_session.tool_processor.env is lead_env
    assert team.lead_session.max_steps == lead_max_steps
    assert team.lead_session.runner.max_steps == lead_max_steps


def test_save_and_load_round_trip_only_messages(tmp_path, install_fake_llm):
    install_fake_llm(FakeLLMClient())
    agent = FakeAgent()
    session = session_mod.Session(agent=agent)
    session.messages.extend([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ])
    session.used_tokens = 123
    session.step_count = 4
    session.is_done = True
    path = tmp_path / "session.jsonl"

    session.save(str(path))
    loaded = session_mod.Session.load(str(path), agent=agent)

    assert loaded.messages == session.messages
    assert loaded.used_tokens == 0
    assert loaded.step_count == 0
    assert loaded.is_done is False


def test_session_accepts_explicit_store(install_fake_llm):
    install_fake_llm(FakeLLMClient())

    class FakeStore:
        def __init__(self):
            self.save_calls = []
            self.load_calls = []
            self.loaded_messages = [{"role": "system", "content": "loaded from fake store"}]

        def save(self, path, messages):
            self.save_calls.append((path, copy.deepcopy(messages)))

        def load_messages(self, path, system_prompt):
            self.load_calls.append((path, system_prompt))
            return copy.deepcopy(self.loaded_messages)

    fake_store = FakeStore()
    agent = FakeAgent()
    session = session_mod.Session(agent=agent, store=fake_store)
    session.messages.append({"role": "user", "content": "hello"})

    session.save("fake-session.jsonl")
    loaded = session_mod.Session.load("fake-session.jsonl", agent=agent, store=fake_store)

    assert session.store is fake_store
    assert fake_store.save_calls == [("fake-session.jsonl", session.messages)]
    assert loaded.store is fake_store
    assert fake_store.load_calls == [("fake-session.jsonl", agent.system_prompt)]
    assert loaded.messages == fake_store.loaded_messages


def test_session_store_preserves_messages_only_jsonl_semantics(tmp_path):
    store = SessionStore()
    path = tmp_path / "stored.jsonl"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]

    store.save(str(path), messages)

    assert store.load_messages(str(path), "fallback") == messages
