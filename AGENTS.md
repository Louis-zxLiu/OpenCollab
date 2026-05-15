# Session Refactoring Goal

## Target
Session owns all session lifecycle state.
SessionState is the single source of truth for messages, tokens, step count, phase, done status, and recent tool hashes.

## Rules
1. Only Session or SessionState can mutate session lifecycle state.
2. Runner may decide transitions, but should not own state.
3. ToolProcessor may execute tool calls, but should return results/deltas.
4. TUI must consume events through an adapter, not depend on internal state.
5. Team orchestration events should not overload SessionEvent semantics.
6. Tools should depend on Environment/HumanInput ports, not terminal UI directly.

## Non-goals
1. Do not rewrite the whole framework.
2. Do not change public CLI behavior.
3. Do not change tool schemas unless necessary.
4. Do not change provider API behavior.
