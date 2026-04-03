"""CLI entry point — Typer-based with three modes: chat, team, eval.

Ref:
- kimi-cli: Typer app with callback, async main, session management
- opencode: bun run dev with agent mode selection
- Design doc: cli/main.py as the interface layer
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from typing import Any, Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="opencollab",
    help="OpenCollab — Minimal Multi-Agent Software Development Framework",
    add_completion=False,
)
console = Console()
_prompt_session: Any | None = None


def _get_prompt_session() -> Any:
    """Lazy-init prompt_toolkit session for robust terminal line editing (IME-safe)."""
    global _prompt_session
    if _prompt_session is None:
        from prompt_toolkit import PromptSession

        _prompt_session = PromptSession()
    return _prompt_session


async def _read_line(prompt_text: str) -> str:
    """Read one input line; fall back to rich console input if needed."""
    try:
        session = _get_prompt_session()
        return await session.prompt_async(prompt_text)
    except Exception:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: console.input(prompt_text))


def _safe_suspend_live(tui: Any) -> bool:
    """Suspend Live rendering if supported by the current TUI implementation."""
    suspend = getattr(tui, "suspend_live", None)
    if callable(suspend):
        return bool(suspend())
    return False


def _safe_resume_live(tui: Any, was_suspended: bool) -> None:
    """Resume Live rendering if supported by the current TUI implementation."""
    resume = getattr(tui, "resume_live", None)
    if callable(resume):
        resume(was_suspended)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value) if value else default
    except (TypeError, ValueError):
        return default


def _required_env_key(provider: str | None) -> str:
    p = (provider or "openai").lower()
    return "ANTHROPIC_API_KEY" if p == "anthropic" else "OPENAI_API_KEY"


def _missing_api_key(provider: str | None, api_key: str | None) -> bool:
    if api_key:
        return False
    return not bool(os.environ.get(_required_env_key(provider)))


def _print_missing_key_hint(provider: str | None) -> None:
    from opencollab.cli.tui import TUI

    tui = TUI(console)
    tui.print_welcome()
    env_key = _required_env_key(provider)
    console.print(
        f"[red]Missing API key[/red]: pass [bold]--api-key[/bold] or set [bold]{env_key}[/bold]."
    )


def _resolve_config(workspace: str, model: str | None, provider: str | None,
                     api_key: str | None, base_url: str | None, budget: int | None) -> dict:
    """Merge CLI args with .env defaults. CLI args take precedence."""
    from opencollab.core.config import get_config
    cfg = get_config(workspace)
    return {
        "model": model or cfg["model"],
        "provider": provider or cfg["provider"],
        "api_key": api_key or cfg["api_key"],
        "base_url": base_url or cfg["base_url"],
        "budget": budget if budget is not None else _safe_int(cfg["budget"], 200_000),
    }


@app.command()
def chat(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model (default from .env)"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider (default from .env or openai)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key (default from .env)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="API base URL (default from .env)"),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Working directory"),
    budget: Optional[int] = typer.Option(None, "--budget", help="Max token budget (default from .env or 200000)"),
    session_file: Optional[str] = typer.Option(None, "--session", "-s", help="Resume from session JSONL"),
    trace: bool = typer.Option(False, "--trace", help="Enable trajectory recording"),
    yolo: bool = typer.Option(False, "--yolo", help="Auto-approve risky commands"),
):
    """Interactive single-agent chat mode (default)."""
    cfg = _resolve_config(workspace, model, provider, api_key, base_url, budget)
    if _missing_api_key(cfg["provider"], cfg["api_key"]):
        _print_missing_key_hint(cfg["provider"])
        raise typer.Exit(code=1)

    asyncio.run(_chat(
        model=cfg["model"], provider=cfg["provider"], api_key=cfg["api_key"],
        base_url=cfg["base_url"], workspace=workspace, budget=cfg["budget"],
        session_file=session_file, trace=trace, yolo=yolo,
    ))


@app.command()
def team(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model (default from .env)"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider (default from .env or openai)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key (default from .env)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="API base URL (default from .env)"),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Working directory"),
    budget: Optional[int] = typer.Option(None, "--budget", help="Max token budget (default from .env or 500000)"),
    trace: bool = typer.Option(False, "--trace", help="Enable trajectory recording"),
    yolo: bool = typer.Option(False, "--yolo", help="Auto-approve risky commands"),
    no_worktrees: bool = typer.Option(False, "--no-worktrees", help="Disable git worktree isolation"),
):
    """Multi-agent team mode with Lead + Teammates."""
    cfg = _resolve_config(workspace, model, provider, api_key, base_url, budget)
    if _missing_api_key(cfg["provider"], cfg["api_key"]):
        _print_missing_key_hint(cfg["provider"])
        raise typer.Exit(code=1)

    # Team mode gets higher default budget
    team_budget = cfg["budget"] if budget is not None else max(cfg["budget"], 500_000)
    asyncio.run(_team(
        model=cfg["model"], provider=cfg["provider"], api_key=cfg["api_key"],
        base_url=cfg["base_url"], workspace=workspace, budget=team_budget,
        trace=trace, yolo=yolo, use_worktrees=not no_worktrees,
    ))


@app.command(name="eval")
def eval_cmd(
    tasks_file: str = typer.Argument(..., help="JSONL file with eval tasks"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model (default from .env)"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider (default from .env)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key (default from .env)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="API base URL (default from .env)"),
    output_dir: str = typer.Option("eval_results", "--output", "-o"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Parallel tasks"),
    max_tokens: int = typer.Option(100_000, "--max-tokens"),
    timeout: float = typer.Option(600.0, "--timeout"),
):
    """Headless evaluation mode for benchmarks (SWE-bench, etc.)."""
    cfg = _resolve_config(".", model, provider, api_key, base_url, None)
    asyncio.run(_eval(
        tasks_file=tasks_file, model=cfg["model"], provider=cfg["provider"],
        api_key=cfg["api_key"], base_url=cfg["base_url"], output_dir=output_dir,
        concurrency=concurrency, max_tokens=max_tokens, timeout=timeout,
    ))


# ---------------------------------------------------------------------------
# Async implementations
# ---------------------------------------------------------------------------


async def _chat(
    model: str, provider: str, api_key: str | None, base_url: str | None,
    workspace: str, budget: int, session_file: str | None,
    trace: bool, yolo: bool,
):
    from opencollab.core.agent import Agent
    from opencollab.core.session import Session
    from opencollab.core.env import LocalEnvironment
    from opencollab.core.tracer import Tracer
    from opencollab.core.context import get_repo_map
    from opencollab.tools.bash import BashTool
    from opencollab.tools.fs import FileReadTool, FileWriteTool, GrepTool
    from opencollab.tools.human import AskUserTool
    from opencollab.tools.safety import SandboxInterceptor
    from opencollab.cli.tui import TUI

    workspace = os.path.abspath(workspace)
    interceptor = SandboxInterceptor(workspace)
    env = LocalEnvironment(workspace)
    tracer = Tracer(run_id=str(uuid.uuid4())[:8]) if trace else None

    # Human-in-the-loop confirmation (disabled in yolo mode)
    confirm_fn = None if yolo else _confirm_prompt

    # Repo map for project topology (机制一)
    repo_map = get_repo_map(workspace)

    agent = Agent(
        name="assistant",
        system_prompt=(
            "You are a skilled software developer. Use the provided tools to help the user "
            "with coding tasks. Read files before modifying them. Verify your changes."
        ),
        tools=[
            BashTool(interceptor),
            FileReadTool(interceptor),
            FileWriteTool(interceptor),
            GrepTool(interceptor),
            AskUserTool(),
        ],
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
    )

    tui = TUI(console)
    tui.print_welcome()

    # Auto-save path (机制四 — session hydration)
    session_id = uuid.uuid4().hex[:8]
    auto_save_dir = os.path.join(workspace, ".opencollab", "sessions")
    auto_save_path = os.path.join(auto_save_dir, f"{session_id}.jsonl")

    # Create or restore session
    if session_file and os.path.exists(session_file):
        session = Session.load(session_file, agent, env=env, tracer=tracer,
                               max_budget_tokens=budget, on_event=tui.event_handler,
                               confirm_fn=confirm_fn, repo_map=repo_map,
                               auto_save_path=auto_save_path)
        console.print(f"[dim]Restored session from {session_file}[/dim]")
    else:
        session = Session(
            agent=agent, env=env, tracer=tracer,
            max_budget_tokens=budget, on_event=tui.event_handler,
            confirm_fn=confirm_fn, repo_map=repo_map,
            auto_save_path=auto_save_path,
        )
        console.print(f"[dim]Session auto-saving to {auto_save_path}[/dim]")

    # REPL loop
    cancel_event = asyncio.Event()

    while True:
        was_suspended = _safe_suspend_live(tui)
        try:
            user_input = await _read_line("> ")
        except (EOFError, KeyboardInterrupt):
            break
        finally:
            _safe_resume_live(tui, was_suspended)

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
            break
        if user_input.lower() == "/save":
            path = f"session-{uuid.uuid4().hex[:8]}.jsonl"
            session.save(path)
            console.print(f"[dim]Session saved to {path}[/dim]")
            continue

        cancel_event.clear()
        tui.reset()
        tui.start_live()

        try:
            await session.add_user_message(user_input)
            await session.run_loop(cancel_event=cancel_event)
        except KeyboardInterrupt:
            cancel_event.set()
        finally:
            tui.stop_live()
            tui.print_stats(session.used_tokens, session.step_count)

    if tracer:
        tracer.close()
        console.print(f"[dim]Trajectory saved to {tracer.path}[/dim]")

    console.print("[dim]Goodbye.[/dim]")


async def _team(
    model: str, provider: str, api_key: str | None, base_url: str | None,
    workspace: str, budget: int, trace: bool, yolo: bool,
    use_worktrees: bool,
):
    from opencollab.team.orchestrator import Team
    from opencollab.core.tracer import Tracer
    from opencollab.core.context import get_repo_map
    from opencollab.tools.human import AskUserTool
    from opencollab.cli.tui import TUI

    workspace = os.path.abspath(workspace)
    tracer = Tracer(run_id=f"team-{uuid.uuid4().hex[:8]}") if trace else None
    confirm_fn = None if yolo else _confirm_prompt
    repo_map = get_repo_map(workspace)

    tui = TUI(console)
    tui.print_welcome()
    console.print("[bold blue]Team mode[/bold blue] — Lead (Planner) + Specialists (Coder + Tester)\n")

    team_instance = Team(
        workspace=workspace,
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        max_budget_tokens=budget,
        tracer=tracer,
        on_event=tui.event_handler,
        confirm_fn=confirm_fn,
        use_worktrees=use_worktrees,
        repo_map=repo_map,
    )

    # Interactive mode: give Lead the ask_user tool (not added in Team.__init__
    # to keep headless eval clean — ref: SWE-bench regression root cause)
    team_instance.lead_agent.tools.append(AskUserTool())

    cancel_event = asyncio.Event()

    while True:
        was_suspended = _safe_suspend_live(tui)
        try:
            user_input = await _read_line("> ")
        except (EOFError, KeyboardInterrupt):
            break
        finally:
            _safe_resume_live(tui, was_suspended)

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
            break

        tui.reset()
        tui.start_live()

        try:
            result = await team_instance.run(user_input)
        except KeyboardInterrupt:
            cancel_event.set()
        finally:
            tui.stop_live()
            tui.print_stats(
                team_instance.lead_session.used_tokens + team_instance._used_tokens,
                team_instance.lead_session.step_count,
            )

    await team_instance.cleanup()
    if tracer:
        tracer.close()
    console.print("[dim]Goodbye.[/dim]")


async def _eval(
    tasks_file: str, model: str, provider: str,
    api_key: str | None, base_url: str | None,
    output_dir: str, concurrency: int,
    max_tokens: int, timeout: float,
):
    from opencollab.harness.evaluator import EvalTask, run_eval_batch, save_results

    # Load tasks from JSONL
    tasks = []
    with open(tasks_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            tasks.append(EvalTask(
                task_id=data["task_id"],
                description=data["description"],
                repo_path=data.get("repo_path"),
                docker_image=data.get("docker_image"),
                timeout=data.get("timeout", timeout),
                max_tokens=data.get("max_tokens", max_tokens),
            ))

    console.print(f"[bold]Running {len(tasks)} eval tasks[/bold] (concurrency={concurrency})")

    results = await run_eval_batch(
        tasks,
        concurrency=concurrency,
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        output_dir=output_dir,
    )

    # Save results
    results_path = os.path.join(output_dir, "results.jsonl")
    save_results(results, results_path)

    # Print summary
    passed = sum(1 for r in results if r.success)
    console.print(f"\n[bold]Results: {passed}/{len(results)} passed[/bold]")
    console.print(f"Results saved to {results_path}")

    for r in results:
        status = "[green]PASS[/green]" if r.success else "[red]FAIL[/red]"
        console.print(f"  {r.task_id}: {status} ({r.tokens_used:,} tokens, {r.duration:.1f}s)")


async def _confirm_prompt(message: str) -> bool:
    """Ask the user for confirmation (human-in-the-loop)."""
    from opencollab.cli.tui import TUI

    active_tui = TUI.get_active()
    was_suspended = _safe_suspend_live(active_tui) if active_tui else False
    try:
        response = await _read_line(f"{message} [y/N] ")
        return response.strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False
    finally:
        if active_tui:
            _safe_resume_live(active_tui, was_suspended)


def main():
    app()


if __name__ == "__main__":
    main()
