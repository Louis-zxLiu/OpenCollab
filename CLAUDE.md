# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenCollab is a research workspace for designing a **minimal multi-agent software development framework** (Python CLI+SDK). It contains four reference projects studied for architectural extraction, plus design documents that synthesize insights from all four into a proposed framework.

### Repository Layout

- `opencollab-极简多智能体框架设计.md` — Primary design document (Chinese). Contains the full architecture proposal extracted from the four reference projects. **Read this first** for project context.
- `claude_code_agen_team.md` — Claude Code agent teams documentation. Covers Lead + Teammates orchestration, subagents vs teams, display modes.
- `self-collaboration_paper_tex/` — Academic paper (LaTeX) on self-collaboration patterns.
- `claude-code/` — Anthropic's Claude Code CLI reference (TypeScript/Node).
- `openclaw/` — Multi-channel AI gateway reference (TypeScript/Node, pnpm). Has its own comprehensive `CLAUDE.md`.
- `kimi-cli/` — Moonshot AI's Kimi Code CLI reference (Python, uv workspace, Makefile-driven).
- `opencode/` — Open source AI coding agent reference (TypeScript/Bun monorepo).

## Core Design Principles (from the design document)

These principles govern the proposed OpenCollab framework:

1. **Agent = LLM + Prompt + Context (Memory) + Tools** — first-principles definition, no unnecessary abstractions.
2. **Context Isolation over Group Chat** — multi-agent cooperation uses isolated message histories per agent to prevent token explosion.
3. **Flat Hierarchy** — Lead + N Teammates only. No nested teams, no complex DAGs.
4. **Tool-Based Delegation** — teammates are invoked as Tool calls by the Lead.
5. **Prompt-Driven Behavior** — collaboration roles (Analyst/Coder/Reviewer) live in system prompts, not hardcoded in framework code.
6. **MCP as Escape Hatch** — only 2 built-in tools (Bash + FileEdit); everything else via MCP servers.

## Reference Project Commands

### kimi-cli (Python — most relevant to OpenCollab's target stack)
```bash
cd kimi-cli
make prepare          # Install deps (uv sync + prek hooks)
make format           # Ruff auto-fix across all packages
make check            # Linting + pyright type checking
make test             # pytest all packages
make build            # Python wheel/sdist + web assets
```

### openclaw (TypeScript/Node)
```bash
cd openclaw
pnpm install
pnpm check            # Oxlint + typecheck
pnpm test             # Vitest
pnpm format           # Oxfmt
```

### opencode (TypeScript/Bun)
```bash
cd opencode
bun install
bun run dev           # CLI dev mode
bun run typecheck     # Turbo monorepo check
```

## Architecture Extraction Summary

The design document distills a shared pattern across all four reference projects:

| Concept | Pattern |
|---------|---------|
| Core loop | REPL/TUI with streaming LLM output |
| Agent model | Stateless agent + stateful session (message list) |
| Multi-agent | Lead delegates via tool calls to teammates with isolated contexts |
| Extensibility | MCP servers for external tools (git, github, db) |
| CLI framework | Typer (Python) or Commander.js (TypeScript) |
| TUI | Rich + prompt_toolkit (Python) or web-based (TypeScript) |

## OpenCollab Framework Implementation

The framework is implemented in `opencollab/` as a Python package (~2900 lines).

### Build & Run

```bash
cd opencollab
pip install -e .                          # Install in dev mode
pip install -e ".[dev]"                   # With dev dependencies
opencollab chat --model gpt-4o           # Interactive single-agent REPL
opencollab team --model gpt-4o           # Multi-agent team mode
opencollab eval tasks.jsonl -o results/  # Headless benchmark eval
python -m opencollab chat                # Alternative entry
```

### Four-Layer Onion Architecture

```
Layer 4 (Interface):    cli/main.py (Typer CLI), cli/tui.py (Rich TUI), harness/evaluator.py
Layer 3 (Boundary):     tools/safety.py (path jail + cmd filter), core/tracer.py (JSONL), core/env.py
Layer 2 (Collaboration): team/orchestrator.py (Team + delegation), team/prompts.py (role prompts)
Layer 1 (Core):         core/agent.py (stateless), core/session.py (stateful loop), core/llm.py
```

### Key Files

- `core/session.py` — The agent loop with context compaction, budget enforcement, and loop-breaking
- `core/env.py` — Environment abstraction: `LocalEnvironment`, `WorktreeEnvironment` (git worktree per teammate), `DockerEnvironment` (sandbox)
- `team/orchestrator.py` — `Team` class with `delegate_task` and `delegate_with_review` (Self-Collaboration loop)
- `tools/safety.py` — `SandboxInterceptor` with path jail and command regex filter
- `tools/mcp.py` — MCP client for dynamic tool discovery via stdio transport
- `harness/evaluator.py` — Headless batch runner for SWE-bench style evaluation

### Critical Design Mechanisms

1. **Context compaction** (`session.py`): auto-summarizes older messages when tokens exceed 64k threshold
2. **Loop breaking** (`session.py`): detects 3+ identical tool calls via content hashing, injects warning
3. **Workspace isolation** (`env.py`): parallel teammates use `git worktree` for physical directory separation
4. **Output truncation** (`bash.py`): stdout/stderr capped at 8k chars (head + tail) to prevent context explosion
5. **Budget enforcement** (`session.py`): hard stop when token budget exhausted
6. **Human-in-the-loop** (`safety.py`): risky commands require confirmation (unless `--yolo` mode)
