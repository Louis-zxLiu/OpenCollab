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
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="opencollab",
    help="OpenCollab — Minimal Multi-Agent Software Development Framework",
    add_completion=False,
)
console = Console()


@app.command()
def chat(
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model identifier"),
    provider: str = typer.Option("openai", "--provider", "-p", help="LLM provider (openai/anthropic)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY", help="API key"),
    base_url: Optional[str] = typer.Option(None, "--base-url", envvar="OPENAI_BASE_URL", help="API base URL"),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Working directory"),
    budget: int = typer.Option(200_000, "--budget", help="Max token budget"),
    session_file: Optional[str] = typer.Option(None, "--session", "-s", help="Resume from session JSONL"),
    trace: bool = typer.Option(False, "--trace", help="Enable trajectory recording"),
    yolo: bool = typer.Option(False, "--yolo", help="Auto-approve risky commands"),
):
    """Interactive single-agent chat mode (default)."""
    asyncio.run(_chat(
        model=model, provider=provider, api_key=api_key, base_url=base_url,
        workspace=workspace, budget=budget, session_file=session_file,
        trace=trace, yolo=yolo,
    ))


@app.command()
def team(
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model identifier"),
    provider: str = typer.Option("openai", "--provider", "-p", help="LLM provider"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY"),
    base_url: Optional[str] = typer.Option(None, "--base-url", envvar="OPENAI_BASE_URL"),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Working directory"),
    budget: int = typer.Option(500_000, "--budget", help="Max token budget"),
    trace: bool = typer.Option(False, "--trace", help="Enable trajectory recording"),
    yolo: bool = typer.Option(False, "--yolo", help="Auto-approve risky commands"),
    no_worktrees: bool = typer.Option(False, "--no-worktrees", help="Disable git worktree isolation"),
):
    """Multi-agent team mode with Lead + Teammates."""
    asyncio.run(_team(
        model=model, provider=provider, api_key=api_key, base_url=base_url,
        workspace=workspace, budget=budget, trace=trace, yolo=yolo,
        use_worktrees=not no_worktrees,
    ))


@app.command(name="eval")
def eval_cmd(
    tasks_file: str = typer.Argument(..., help="JSONL file with eval tasks"),
    model: str = typer.Option("gpt-4o", "--model", "-m"),
    provider: str = typer.Option("openai", "--provider", "-p"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY"),
    base_url: Optional[str] = typer.Option(None, "--base-url", envvar="OPENAI_BASE_URL"),
    output_dir: str = typer.Option("eval_results", "--output", "-o"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Parallel tasks"),
    max_tokens: int = typer.Option(100_000, "--max-tokens"),
    timeout: float = typer.Option(600.0, "--timeout"),
):
    """Headless evaluation mode for benchmarks (SWE-bench, etc.)."""
    asyncio.run(_eval(
        tasks_file=tasks_file, model=model, provider=provider,
        api_key=api_key, base_url=base_url, output_dir=output_dir,
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
    from opencollab.tools.bash import BashTool
    from opencollab.tools.fs import FileReadTool, FileWriteTool, GrepTool
    from opencollab.tools.safety import SandboxInterceptor
    from opencollab.cli.tui import TUI

    workspace = os.path.abspath(workspace)
    interceptor = SandboxInterceptor(workspace)
    env = LocalEnvironment(workspace)
    tracer = Tracer(run_id=str(uuid.uuid4())[:8]) if trace else None

    # Human-in-the-loop confirmation (disabled in yolo mode)
    confirm_fn = None if yolo else _confirm_prompt

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
        ],
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
    )

    tui = TUI(console)
    tui.print_welcome()

    # Create or restore session
    if session_file and os.path.exists(session_file):
        session = Session.load(session_file, agent, env=env, tracer=tracer,
                               max_budget_tokens=budget, on_event=tui.event_handler,
                               confirm_fn=confirm_fn)
        console.print(f"[dim]Restored session from {session_file}[/dim]")
    else:
        session = Session(
            agent=agent, env=env, tracer=tracer,
            max_budget_tokens=budget, on_event=tui.event_handler,
            confirm_fn=confirm_fn,
        )

    # REPL loop
    cancel_event = asyncio.Event()

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: console.input("[bold green]> [/bold green]")
            )
        except (EOFError, KeyboardInterrupt):
            break

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
    from opencollab.cli.tui import TUI

    workspace = os.path.abspath(workspace)
    tracer = Tracer(run_id=f"team-{uuid.uuid4().hex[:8]}") if trace else None
    confirm_fn = None if yolo else _confirm_prompt

    tui = TUI(console)
    tui.print_welcome()
    console.print("[bold blue]Team mode[/bold blue] — Lead + Specialists\n")

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
    )

    cancel_event = asyncio.Event()

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: console.input("[bold green]> [/bold green]")
            )
        except (EOFError, KeyboardInterrupt):
            break

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
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: console.input(f"[yellow]{message}[/yellow] [y/N] ")
        )
        return response.strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def main():
    app()


if __name__ == "__main__":
    main()
