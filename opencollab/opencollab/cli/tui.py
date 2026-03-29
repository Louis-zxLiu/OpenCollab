"""TUI — Rich-based terminal interface with streaming and nested spinners.

Ref:
- kimi-cli: Wire pattern — bidirectional channel between agent and UI
- opencode: processor.ts events — text_delta, tool_start, tool_end, etc.
- openclaw: Rich terminal palette with dynamic updates
"""

from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from rich.table import Table

from opencollab.core.session import SessionEvent


class TUI:
    """Terminal UI that renders SessionEvents in real-time.

    Consumes the event stream from Session (via on_event callback)
    and renders streaming text, tool execution spinners, and status updates.
    """

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._current_text = ""
        self._active_tools: dict[str, dict] = {}
        self._step = 0
        self._live: Live | None = None

    def event_handler(self, event: SessionEvent) -> None:
        """Synchronous event handler — called by Session.on_event."""
        etype = event.type

        if etype == "text_delta":
            content = event.data.get("content", "")
            self._current_text += content
            self._refresh()

        elif etype == "tool_start":
            tool = event.data.get("tool", "?")
            role = event.data.get("role", "")
            label = f"{role}:{tool}" if role else tool
            self._active_tools[label] = event.data
            self._refresh()

        elif etype == "tool_end":
            tool = event.data.get("tool", "?")
            role = event.data.get("role", "")
            label = f"{role}:{tool}" if role else tool
            self._active_tools.pop(label, None)
            self._refresh()

        elif etype == "step_start":
            self._step = event.data.get("step", 0)

        elif etype == "compaction":
            self.console.print("[dim]Context compacted[/dim]")

        elif etype == "loop_detected":
            tool = event.data.get("tool", "?")
            count = event.data.get("count", 0)
            self.console.print(f"[yellow]Loop detected: {tool} called {count}x with same args[/yellow]")

        elif etype == "budget_warning":
            self.console.print("[yellow]Token budget running low[/yellow]")

        elif etype == "error":
            reason = event.data.get("reason", "unknown")
            self.console.print(f"[red]Error: {reason}[/red]")

    def _refresh(self) -> None:
        """Re-render the current state."""
        if self._live:
            self._live.update(self._build_display())

    def _build_display(self) -> Any:
        """Build the Rich renderable for current state."""
        parts = []

        # Streaming text (render as markdown)
        if self._current_text:
            parts.append(Markdown(self._current_text))

        # Active tool spinners
        if self._active_tools:
            for label, data in self._active_tools.items():
                args_preview = ""
                if "args" in data:
                    args = data["args"]
                    if isinstance(args, dict) and "command" in args:
                        args_preview = f" `{args['command'][:60]}`"
                    elif isinstance(args, dict) and "task" in args:
                        args_preview = f" {args['task'][:60]}"

                spinner_text = Text.assemble(
                    ("  ", ""),
                    (f"[{label}]", "bold cyan"),
                    (args_preview, "dim"),
                )
                parts.append(spinner_text)

        if not parts:
            return Text("Thinking...", style="dim")

        from rich.console import Group
        return Group(*parts)

    def start_live(self) -> None:
        """Start the Live display context."""
        self._live = Live(
            Text("Thinking...", style="dim"),
            console=self.console,
            refresh_per_second=10,
            transient=True,
        )
        self._live.start()

    def stop_live(self) -> None:
        """Stop the Live display and print final output."""
        if self._live:
            self._live.stop()
            self._live = None

        # Print the final accumulated text as markdown
        if self._current_text:
            self.console.print(Markdown(self._current_text))
            self._current_text = ""

    def print_welcome(self) -> None:
        self.console.print(Panel.fit(
            "[bold]OpenCollab[/bold] — Minimal Multi-Agent Framework\n"
            "[dim]Type your message. Ctrl+C to interrupt. 'exit' to quit.[/dim]",
            border_style="blue",
        ))

    def print_stats(self, tokens: int, steps: int) -> None:
        self.console.print(f"\n[dim]({tokens:,} tokens, {steps} steps)[/dim]")

    def reset(self) -> None:
        """Reset state for next turn."""
        self._current_text = ""
        self._active_tools.clear()
        self._step = 0
