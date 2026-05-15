# Session Refactor Notes

Source reviewed: `opencollab/opencollab/core/session.py`

This note records the current behavior of `Session` before refactoring. Treat the items in "must keep unchanged" and "easy to break" as regression checklist items.

## 1. Session 当前有哪些状态字段？

Constructor/config fields:

- `agent`: current `Agent`; provides system prompt, tool schemas, model config, temperature, and tool lookup.
- `env`: execution `Environment`; defaults to `LocalEnvironment()`.
- `tracer`: optional `Tracer` for LLM/tool/compaction trajectory logging.
- `max_budget_tokens`: hard token budget checked by `run_loop`.
- `max_steps`: maximum loop iterations checked by `run_loop`.
- `compaction_threshold`: estimated context token threshold that triggers `_compact`.
- `on_event`: optional UI/callback event sink.
- `confirm_fn`: optional human approval callback passed into tools.
- `_auto_save_path`: optional JSONL path for persistence after state changes.

Conversation/progress fields:

- `messages`: OpenAI-style message list. Initialized with one system message; `repo_map` is appended to that first system message when provided.
- `used_tokens`: accumulated LLM tokens from normal completions and successful compaction summaries.
- `step_count`: number of `_step` calls already started. It is monotonic across user turns in the same session.
- `is_done`: marks whether the current user turn has reached an assistant response without tool calls.

Loop/tool state:

- `_recent_call_hashes`: bounded history of hashed `(tool_name, args)` tuples for loop detection.
- `_llm`: `LLMClient` created from the agent model/provider/api config.

Persistence-derived state:

- `save()` persists only `messages`.
- `load()` restores only `messages`; `used_tokens`, `step_count`, `is_done`, `_recent_call_hashes`, and `_auto_save_path` are initialized from the new `Session(...)` call rather than from JSONL.
- `snapshot()` copies `messages`, `used_tokens`, and `step_count`, but not `is_done`, `_recent_call_hashes`, `_auto_save_path`, or injected `repo_map` state.

Primary references: `session.py:78-120`, `session.py:168-190`, `session.py:192-226`.

## 2. run_loop 的隐式状态转移是什么？

`run_loop()` is the outer state machine, but it encodes the state transitions through fields and appended messages rather than explicit enum states.

Current implicit transitions:

1. `ready/running`: loop continues while `not self.is_done and self.step_count < self.max_steps`.
2. `cancelled`: if `cancel_event.is_set()`, append system message `[Session interrupted by user]`, emit `error(cancelled)`, and break. `is_done` is not set to `True`.
3. `budget_exceeded`: if `used_tokens >= max_budget_tokens`, raise `BudgetExceededError`; handler appends a budget-exceeded system message and emits `error(budget_exceeded)`. `is_done` is not set to `True`.
4. `needs_compaction`: if estimated message tokens exceed `compaction_threshold`, run `_compact()` before the next LLM step.
5. `stepping`: call `_step()`, which may append an assistant message, tool messages, and may set `is_done`.
6. `tool_continuation`: when `_step()` receives tool calls, it leaves `is_done == False`, so the outer loop runs another step with the tool results in context.
7. `completed`: when `_step()` receives no tool calls, it sets `is_done = True`, causing `run_loop()` to exit.
8. `max_steps_exhausted`: if `step_count >= max_steps`, the loop exits without appending an explicit stop marker and returns the latest assistant content if any.
9. `task_cancelled_exception`: if the coroutine itself receives `asyncio.CancelledError`, tracer is flushed and the exception is re-raised.

After any exit path except `asyncio.CancelledError`, `run_loop()` returns the most recent assistant message with non-empty `content`; otherwise it returns `""`.

Primary references: `session.py:124-166`, `_step()` completion branch at `session.py:285-290`.

## 3. _step 的职责有哪些？

`_step()` is one complete model turn. Its current responsibilities are broader than a pure "LLM call":

- Increment `step_count` immediately.
- Emit `step_start` with the new step number.
- Build current tool schemas from `agent.tool_schemas()`, converting empty lists to `None`.
- Call `self._llm.complete()` with current `messages`, tools, and `agent.temperature`.
- Measure LLM latency.
- Accumulate `response.usage.total_tokens` into `used_tokens`.
- Log the LLM call to `tracer`, including model, finish reason, assistant content, tool call id/name/arguments, tokens, and latency.
- Append the assistant message to `messages`, including `content` and/or `tool_calls`.
- Emit `text_delta` when assistant content exists. This is currently whole-response text, not token streaming.
- Dispatch tool calls through `_process_tool_calls()` when present.
- Set `is_done = True` when no tool calls are present.
- Emit `step_end` with step number and LLM latency.
- Auto-save after the step completes.

Primary references: `session.py:230-293`.

## 4. _process_tool_calls 的职责有哪些？

`_process_tool_calls()` executes every tool call returned by the assistant in order and turns each one into a `tool` message.

Current responsibilities:

- Iterate all tool calls sequentially.
- Extract `function.name`, `function.arguments`, and tool call id.
- Parse JSON arguments; on JSON parse failure, append a tool error message and continue to the next call.
- Hash normalized `{"name": tool_name, "args": args}` for loop detection.
- Maintain `_recent_call_hashes` with a max window of `MAX_CALL_HASH_WINDOW`.
- Detect repeated identical calls over the recent window slice. On detection, append a warning tool message, emit `loop_detected`, and skip execution.
- Resolve the tool through `agent.find_tool(tool_name)`. If missing, append an unknown-tool error message listing available tool names.
- Emit `tool_start` with tool name and parsed args.
- Execute the tool as `await tool.execute(args, env=self.env, confirm_fn=self.confirm_fn)`.
- Convert `PermissionError` and general exceptions into string results instead of raising them.
- Measure tool latency.
- Truncate oversized tool output to `MAX_TOOL_OUTPUT_CHARS`, preserving the start and end with a truncation marker.
- Log tool execution to `tracer`, with trace result capped separately to about 4 KiB.
- Append the final tool result message with `role: "tool"`, `tool_call_id`, and `content`.
- Emit `tool_end` with tool name and latency.

Primary references: `session.py:295-377`.

## 5. _compact 的职责有哪些？

`_compact()` reduces context size by replacing older conversation history with a summary system message.

Current responsibilities:

- Emit `compaction(reason=context_overflow)` before doing any length/no-op check.
- Return without changing messages if there are not enough messages to compact.
- Preserve the first message as the original system message.
- Preserve the last `COMPACTION_KEEP_RECENT` messages exactly.
- Summarize all messages between system and recent messages.
- Build compaction input from older message content, capped to 2000 chars per message. Assistant tool calls are summarized as `[tool_call]: name(...)`.
- Call the same `_llm.complete()` with a special two-message summary prompt and `temperature=0.0`.
- Add successful summary LLM usage to `used_tokens`.
- On compaction LLM failure, fall back to raw joined `older_text[:5000]`.
- Rebuild `messages` as original system message, one compaction-summary system message, then the preserved recent messages.
- Log compaction to tracer with compacted count and summary length.
- Auto-save after compaction.

Primary references: `session.py:381-436`.

## 6. Session 和 TUI / callback 的耦合点在哪里？

Direct callback surface:

- `SessionEvent` defines the event shape consumed by UI code.
- `on_event` accepts sync or async callbacks.
- `_emit()` swallows callback exceptions so UI failures do not crash the loop.

Event names and payloads used by TUI:

- `text_delta`: expects `data.content`.
- `tool_start`: expects `data.tool`, optional `data.args`, optional `data.role`.
- `tool_end`: expects `data.tool`, optional `data.role`, `data.latency`.
- `step_start`: expects `data.step`.
- `compaction`: currently no payload is needed by TUI, but Session sends `reason`.
- `loop_detected`: expects `data.tool` and `data.count`.
- `error`: expects `data.reason`.
- `budget_warning` exists in TUI and event comments, but `Session` currently does not emit it.
- `step_end` is emitted by Session but currently ignored by TUI.

CLI coupling:

- Chat mode passes `tui.event_handler` to `Session(on_event=...)`.
- Chat mode passes `_confirm_prompt` as `confirm_fn` unless yolo mode is enabled.
- Chat mode calls `add_user_message()` then `run_loop(cancel_event=...)` for each user turn.
- CLI prints `session.used_tokens` and `session.step_count` after each turn.
- CLI relies on `save()`, `load()`, and `_auto_save_path` JSONL hydration.

Team coupling:

- `Team` passes its `on_event` and `confirm_fn` through to lead and teammate sessions.
- `Team` reads teammate `used_tokens` and lead `step_count`.
- `Team` also emits its own `SessionEvent` objects for delegate lifecycle, so UI event compatibility is shared beyond `Session`.

Tool callback coupling:

- `confirm_fn` is passed into every tool execution.
- Safety tools use it for risky command approval.
- `AskUserTool` has its own prompt path and uses `confirm_fn is None` as a non-interactive/yolo signal.

Primary references: `session.py:41-50`, `session.py:346-351`, `session.py:440-448`, `cli/main.py:240-291`, `cli/tui.py:50-108`, `team/orchestrator.py:194-202`, `team/orchestrator.py:241-245`.

## 7. 哪些行为必须保持不变？

- The message protocol must remain compatible with the LLM adapters: system/user/assistant/tool roles, assistant `tool_calls`, and tool `tool_call_id`.
- A user message must reset `is_done` to `False` and clear loop detection state.
- `run_loop()` must continue until no tool calls, cancellation, budget exhaustion, or `max_steps`.
- No-tool-call assistant response means the current turn is complete.
- Tool-call assistant response means append tool results and continue another LLM step.
- Budget is checked before each step and `used_tokens` is accumulated from LLM responses.
- Context compaction is checked before each step and must preserve the first system message plus recent messages.
- Tool execution errors must become tool-result content rather than crash the whole loop, except coroutine cancellation.
- Callback failures must not crash the agent loop.
- Tool output truncation must keep context from exploding while preserving both beginning and ending content.
- Loop detection must prevent repeated identical tool calls from executing indefinitely.
- Auto-save must run after user messages, completed steps, and compaction when configured.
- `run_loop()` must return the latest assistant content for CLI/team callers.
- JSONL `save()`/`load()` must continue to support session resumption with current role validation.
- `confirm_fn` must continue to be passed into tools unchanged.
- TUI event names and expected payload keys should remain backward compatible.

## 8. 哪些地方最容易被重构改坏？

- `is_done` semantics across turns. If `add_user_message()` does not reset it, new user input will not run. If `_step()` sets it too early after tool calls, multi-step tool workflows break.
- `step_count` lifetime. It currently accumulates across the session, not per user turn. Resetting it per turn would change max-step behavior and CLI stats.
- `max_steps` exit behavior. Today it exits silently and returns latest assistant content; adding or removing stop messages changes caller-visible behavior.
- Message ordering around tool calls. Assistant tool-call messages must be followed by corresponding tool messages with matching `tool_call_id`.
- Multiple tool calls. They execute sequentially now; parallelizing them can change ordering, shared environment effects, confirmation prompts, and trace/UI order.
- Invalid JSON and unknown-tool handling. These are represented as tool messages, allowing the model to recover in the next step.
- Loop detection window and hash normalization. Changing the hash input, when hashes are appended, or when clearing occurs can either miss loops or create false positives across user turns.
- Compaction boundaries. The first system message and last `COMPACTION_KEEP_RECENT` messages are preserved exactly; breaking this can drop tool-call/result pairs or important recent context.
- Compaction fallback. If the summary LLM fails, current behavior still rebuilds messages using raw truncated history instead of failing the run.
- Token accounting. `used_tokens` includes LLM call usage and successful compaction summary usage, but not tool execution or failed compaction fallback.
- Auto-save timing. Moving saves earlier/later can persist partial state, miss tool results, or fail to record compaction.
- Callback isolation. `_emit()` intentionally catches callback exceptions; letting UI exceptions propagate would make headless agent behavior depend on rendering.
- TUI event payload compatibility. Renaming `tool`, `args`, `latency`, `step`, `reason`, or `content` breaks current rendering.
- `confirm_fn` meaning. `None` is used as yolo/non-interactive in several tools; changing this contract affects safety prompts.
- `snapshot()` and `load()` partial state behavior. They do not restore every runtime field today. Making them "more complete" may be correct, but it is a behavior change.
- `repo_map` injection. It is only appended during construction and not persisted as a separate field. Reloading/injecting it twice would duplicate project structure in the prompt.
- Tracer behavior. The tracer currently records full LLM response content/tool calls and truncated tool output; changing payload shape can break trajectory consumers.

## 9. Post-refactor ownership boundaries

The session package is now split around explicit state ownership:

- `SessionState` owns lifecycle writes through methods such as `append_message()`, `replace_messages()`, `add_used_tokens()`, `advance_step()`, `mark_done()`, `set_phase()`, and user-turn/tool-hash helpers.
- `Session` remains the public facade and compatibility layer. It owns the `SessionState`, wires runtime dependencies, and keeps existing facade properties and helper methods.
- `SessionRunner` remains the state transition engine, but writes lifecycle state through `SessionState` methods.
- `ToolCallProcessor.process()` returns `ToolProcessingResult`; runner/session compatibility paths apply returned messages and recent tool-call hashes explicitly.
- `ContextCompactor.compact()` returns `CompactResult`; direct calls still apply by default for compatibility, while runner/session paths use `apply=False` and apply the result explicitly.

Compatibility intentionally preserved:

- `SessionEvent(type, data)` stays unchanged for TUI, Team, CLI, and callback users.
- JSONL storage remains messages-only.
- Compatibility exports such as `SessionMachine`, facade properties, and older helper methods remain in place.
- TUI event adapter separation, Team-specific event types, HumanInput ports, and stronger event typing remain future work.
