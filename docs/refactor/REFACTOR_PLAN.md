# Session Refactor Plan

Scope: only `opencollab/opencollab/core/session.py`.

This document analyzes the current `Session` module as-is and proposes a minimal safe refactoring order. It is not an implementation patch.

## 1. 当前模块职责分解

`Session` currently combines several responsibilities in one stateful object:

1. Conversation state container
   - Owns `messages`, `used_tokens`, `step_count`, `is_done`, `_recent_call_hashes`.
   - Injects optional `repo_map` into the initial system prompt.
   - Provides `snapshot()`, `save()`, `load()`, and `_auto_save()`.

2. Agent loop coordinator
   - `run_loop()` controls the outer loop.
   - It checks cancellation, token budget, compaction threshold, max step count, and delegates one iteration to `_step()`.
   - It returns the latest assistant text after the loop exits.

3. LLM turn executor
   - `_step()` builds tool schemas, calls `LLMClient.complete()`, records usage, logs tracing, appends assistant messages, emits UI events, and decides whether the turn is complete.

4. Tool call executor
   - `_process_tool_calls()` parses tool arguments, resolves tools from `Agent`, executes them with `env` and `confirm_fn`, appends tool results, truncates large outputs, detects loops, logs traces, and emits tool events.

5. Context compactor
   - `_compact()` estimates old context, builds a summary prompt, calls the LLM, replaces older messages with one summary system message, logs the compaction, and auto-saves.

6. Event adapter
   - `SessionEvent` and `_emit()` provide a callback protocol for TUI and other consumers.
   - `_emit()` supports sync and async callbacks and intentionally swallows callback failures.

7. Budget and safety guard owner
   - Enforces token budget before each step.
   - Prevents repeated identical tool calls from running indefinitely.
   - Caps tool output length before it re-enters context.

## 2. 隐式状态机

The module has no explicit state enum. The state machine is encoded through `is_done`, `step_count`, `used_tokens`, `messages`, and `_recent_call_hashes`.

Main states:

- `idle/new`: after construction. `messages` contains the system prompt, `is_done == False`, counters are zero.
- `user_message_added`: `add_user_message()` appends a user message, resets `is_done = False`, clears loop detection state, and auto-saves.
- `running`: `run_loop()` continues while `not is_done` and `step_count < max_steps`.
- `needs_compaction`: before each step, if estimated message tokens exceed `compaction_threshold`, `_compact()` runs.
- `llm_step`: `_step()` increments `step_count`, emits `step_start`, calls the LLM, appends an assistant message, emits content if present.
- `tool_continuation`: if the assistant message contains `tool_calls`, `_process_tool_calls()` appends corresponding tool messages. `is_done` remains `False`, so the next loop iteration sends the tool results back to the LLM.
- `completed`: if the assistant message has no tool calls, `_step()` sets `is_done = True`, and `run_loop()` exits.
- `cancelled`: if `cancel_event` is set before a step, `run_loop()` appends `[Session interrupted by user]`, emits `error(cancelled)`, and exits without setting `is_done`.
- `budget_exceeded`: if `used_tokens >= max_budget_tokens`, `run_loop()` appends a budget system message, emits `error(budget_exceeded)`, and exits without setting `is_done`.
- `max_steps_exhausted`: if `step_count >= max_steps`, `run_loop()` exits silently and returns the latest assistant content if available.
- `task_cancelled_exception`: if the coroutine receives `asyncio.CancelledError`, tracer is flushed and the exception is re-raised.

Important transition rules:

- `add_user_message()` is the only public API that reopens a completed session for a new turn.
- Tool errors are not terminal states; they are converted into tool messages so the LLM can recover in the next step.
- Loop detection prevents a specific tool execution but does not terminate the session.
- Compaction is an in-loop maintenance transition, not a user-visible terminal state.

## 3. 副作用点

External side effects:

- LLM calls:
  - `_step()` calls `self._llm.complete(...)`.
  - `_compact()` calls `self._llm.complete(...)` for summarization.

- Tool execution:
  - `_process_tool_calls()` calls `tool.execute(args, env=self.env, confirm_fn=self.confirm_fn)`.
  - Tool code may read/write files, run commands, ask for confirmation, or call external systems depending on the concrete tool.

- File system writes:
  - `save()` writes JSONL to disk.
  - `_auto_save()` invokes `save()` after user messages, completed steps, and compaction.
  - `save()` creates the parent directory when needed.

- File system reads:
  - `load()` reads JSONL from disk.

- Callback/UI effects:
  - `_emit()` invokes `on_event`.
  - Event handlers may render terminal UI or perform other side effects.

- Tracing:
  - `_step()` logs `llm_call`.
  - `_process_tool_calls()` logs `tool_exec`.
  - `_compact()` logs `compaction`.
  - `run_loop()` flushes tracer on `asyncio.CancelledError`.

Internal state mutations:

- `messages` is appended or replaced in `__init__`, `add_user_message()`, `run_loop()`, `_step()`, `_process_tool_calls()`, `_compact()`, and `load()`.
- `used_tokens` changes after LLM calls and successful compaction summaries.
- `step_count` increments at the start of `_step()`.
- `is_done` changes in `add_user_message()` and `_step()`.
- `_recent_call_hashes` changes in `add_user_message()` and `_process_tool_calls()`.

## 4. 与 UI/TUI、LLM、工具、tracer、storage 的耦合点

### UI / TUI

- `SessionEvent` is the shared event DTO.
- `on_event` is passed into `Session` and called through `_emit()`.
- `_emit()` accepts both sync and async callbacks.
- `_emit()` catches all callback exceptions, so UI failures do not crash the agent loop.
- Emitted event types:
  - `step_start`
  - `step_end`
  - `text_delta`
  - `tool_start`
  - `tool_end`
  - `compaction`
  - `loop_detected`
  - `error`
- `budget_warning` is documented in the event comment but not emitted by current code.
- Payload keys are part of the practical API: `content`, `tool`, `args`, `latency`, `step`, `reason`, `count`.

### LLM

- `Session` constructs its own `LLMClient` from `agent.model`, `agent.api_key`, `agent.base_url`, and `agent.provider`.
- `_step()` sends the full current `messages` list to `LLMClient.complete()`.
- Tool schemas come from `agent.tool_schemas()`.
- Temperature comes from `agent.temperature`.
- `_compact()` reuses the same LLM client with a special summarization prompt and `temperature=0.0`.
- `used_tokens` assumes the LLM response has `usage.total_tokens`.
- Message format is tightly coupled to OpenAI-style roles plus assistant `tool_calls` and tool `tool_call_id`.

### 工具

- Tool discovery is done through `agent.find_tool(tool_name)`.
- Available tool names in an unknown-tool error are read from `agent.tools`.
- Tool arguments are expected to be JSON strings in `tc["function"]["arguments"]`.
- Tool identity is expected at `tc["function"]["name"]` and `tc["id"]`.
- Tool execution signature is expected to accept `args`, `env`, and `confirm_fn`.
- Tool result is expected to be a string, because `len(result)` and string slicing are used.
- `PermissionError` and generic `Exception` are converted to tool result text.
- `confirm_fn is None` has behavioral meaning for tools that support yolo or non-interactive mode.

### Tracer

- Tracer is optional but the current payload shapes are implicit contracts.
- `_step()` logs:
  - `step_type="llm_call"`
  - model, finish reason, content, tool call id/name/arguments
  - response token count and latency
- `_process_tool_calls()` logs:
  - `step_type="tool_exec"`
  - tool name, args, result length, truncated result
  - zero tokens and tool latency
- `_compact()` logs:
  - `step_type="compaction"`
  - compacted message count and summary length
  - zero tokens and zero latency
- `run_loop()` flushes tracer on coroutine cancellation.

### Storage

- `save()` persists only `messages`, one JSON object per line.
- `load()` validates only message role membership in `{"system", "user", "assistant", "tool"}`.
- `load()` does not restore counters or runtime state: `used_tokens`, `step_count`, `is_done`, `_recent_call_hashes`.
- `_auto_save_path` is runtime configuration, not serialized session state.
- `snapshot()` copies only part of the state: `messages`, `used_tokens`, and `step_count`.
- `repo_map` is injected into the initial system message, not stored as separate metadata.

## 5. 重构风险清单

High-risk behavior:

- Changing `is_done` semantics can cause completed sessions not to reopen, or tool workflows to terminate too early.
- Resetting `step_count` per user turn would change max-step behavior and CLI statistics.
- Changing `max_steps` exhaustion from silent exit to explicit error is user-visible.
- Reordering assistant/tool messages can break LLM tool-call protocol.
- Parallelizing tool calls can change environment side effects, confirmation order, trace order, and UI rendering order.
- Dropping invalid-JSON or unknown-tool results instead of appending tool messages removes the model's recovery path.
- Letting tool exceptions propagate can crash the loop and skip auto-save/tracing behavior.
- Letting `_emit()` exceptions propagate makes runtime correctness depend on UI code.
- Changing loop detection hash input or clear timing can create false positives across user turns or miss actual loops.
- Compacting across the wrong boundary can separate assistant tool calls from tool results.
- Losing the first system message during compaction can remove the agent prompt and injected repo map.
- Not preserving the last `COMPACTION_KEEP_RECENT` messages exactly can drop current task state.
- Changing token accounting can alter budget behavior in long sessions.
- Moving auto-save earlier can persist partial assistant/tool state; moving it later can lose completed work.
- Duplicating `repo_map` during load/snapshot reconstruction can bloat prompts and alter behavior.
- Assuming tool results are always strings is currently implicit; changing tool contracts without normalizing results will break truncation.
- Broad `except Exception` blocks intentionally preserve loop continuity; replacing them with narrower handling changes failure behavior.

Medium-risk behavior:

- Event payload key changes can break TUI without type errors.
- Tracer payload changes can break downstream trajectory analysis.
- `snapshot()` currently has partial-state behavior; making it deeper may be correct but is a behavior change.
- `load()` currently restores only messages; restoring counters may be desirable but changes current semantics.
- Compaction fallback currently continues with raw truncated content if summarization fails; failing hard would be a regression for availability.
- `text_delta` is currently whole assistant content, not streaming tokens. Renaming it to match actual behavior may break UI consumers.

## 6. 建议的最小安全重构顺序

The safest path is to first make current behavior observable, then extract pure helpers, then separate side-effecting services.

1. Add characterization tests before moving code
   - Cover no-tool completion.
   - Cover tool-call continuation.
   - Cover invalid JSON tool args.
   - Cover unknown tool.
   - Cover tool exception converted to tool message.
   - Cover loop detection.
   - Cover compaction message boundaries.
   - Cover auto-save timing at a coarse level.
   - Cover callback exception isolation.
   - Cover budget exceeded and max-step exit behavior.

2. Introduce small pure helper functions without changing call sites materially
   - Build assistant message from `LLMResponse`.
   - Parse tool arguments into either `args` or an error string.
   - Build loop-detection hash.
   - Truncate tool output.
   - Extract latest assistant content.
   - Build compaction prompt input.

3. Make implicit runtime state explicit inside the module
   - Add an internal state/value object only if tests prove behavior is preserved.
   - Keep public attributes available during this step.
   - Do not change `save()`/`load()` semantics yet.

4. Extract event emission behind a tiny adapter
   - Preserve `SessionEvent` type and payloads.
   - Preserve sync/async callback support.
   - Preserve exception swallowing.
   - Keep event order identical.

5. Extract storage concerns
   - Move JSONL read/write logic behind a storage helper.
   - Keep `Session.save()` and `Session.load()` public API as pass-through methods.
   - Keep "messages only" persistence until a deliberate migration is designed.

6. Extract tool execution orchestration
   - Start with a helper/class that handles one tool call.
   - Preserve sequential execution.
   - Preserve exact tool message shapes and error strings.
   - Preserve `confirm_fn`, `env`, tracing, truncation, and events.

7. Extract compaction orchestration
   - Keep trigger location in `run_loop()` unchanged.
   - Preserve first system message and recent message boundary.
   - Preserve fallback behavior on compaction LLM failure.
   - Preserve `used_tokens` update only on successful summary response.

8. Clarify the outer loop last
   - Only after the above pieces are covered should `run_loop()` become a clearer explicit state machine.
   - Keep terminal behavior unchanged: cancellation marker, budget marker, silent max-step exhaustion, and latest-assistant-content return.

9. Defer behavior-changing cleanup
   - Do not change persistence metadata, per-turn step counts, streaming semantics, parallel tool execution, or compaction strategy in the first refactor.
   - Treat those as separate product decisions after the low-risk structural refactor lands.
