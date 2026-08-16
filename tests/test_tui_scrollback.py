"""Scrollback behaviour of the TUI: what reaches the terminal, and when.

Split out of ``test_tui_event_rendering`` — those tests cover event dispatch,
these cover the printing contract: only the focused agent's settled blocks are
written, each exactly once, and focusing an agent redraws it in full.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.text import Text

from opencollab.adapters.tui import TUI
from opencollab.adapters.tui import renderer as renderer_mod
from opencollab.adapters.tui import renderer_display as renderer_display_mod
from opencollab.domain.events import SessionRuntimeEvent


def _make_tui() -> TUI:
    return TUI(Console(file=StringIO(), width=100, color_system=None))


def _scrollback(tui: TUI) -> str:
    """Everything the TUI has committed to the terminal so far."""
    return tui.console.file.getvalue()


def _history_plains(tui: TUI, aid: int) -> list[str]:
    return [
        block.plain
        for block in tui._state_for(aid).history_blocks
        if isinstance(block, Text)
    ]


def test_live_frame_stays_bounded_while_full_history_reaches_scrollback():
    console = Console(file=StringIO(), width=100, height=24, color_system=None)
    tui = TUI(console)
    tui.select_agent(1)

    for index in range(100):
        tui._append_activity((f"activity {index}", tui._STYLE_MUTED))

    frame = console.render_lines(tui._build_live_display(), console.options, pad=False)
    assert len(frame) <= renderer_display_mod.MAX_LIVE_BODY_LINES
    scrollback = _scrollback(tui)
    assert "activity 0" in scrollback
    assert "activity 99" in scrollback


def test_every_settled_block_path_reaches_scrollback_for_the_focused_agent():
    tui = TUI(Console(file=StringIO(), width=100, color_system=None))
    tui.event_handler(SessionRuntimeEvent("text_delta", {"content": "assistant say", "aid": 1}))
    assert tui.select_agent(1) == 1

    tui.event_handler(SessionRuntimeEvent("error", {"reason": "failure here", "aid": 1}))
    tui.record_user_message(1, "user asks")
    tui.event_handler(
        SessionRuntimeEvent("tool_start", {"tool": "bash", "args": {"command": "pwd"}, "aid": 1})
    )

    scrollback = _scrollback(tui)
    for expected in ("assistant say", "failure here", "user asks", "A1:bash started"):
        assert expected in scrollback


def test_agent_history_has_a_global_per_agent_bound():
    tui = TUI(Console(file=StringIO(), width=100, color_system=None))
    state = tui._state_for(1)

    for index in range(renderer_mod.MAX_HISTORY_BLOCKS_PER_AGENT + 20):
        tui._append_activity(
            (f"bounded activity {index}", tui._STYLE_MUTED),
            state=state,
        )

    assert len(state.history_blocks) == renderer_mod.MAX_HISTORY_BLOCKS_PER_AGENT
    tui.select_agent(1)
    scrollback = _scrollback(tui)
    assert "bounded activity 0" not in scrollback
    assert "20 older history blocks omitted" in scrollback
    assert f"bounded activity {renderer_mod.MAX_HISTORY_BLOCKS_PER_AGENT + 19}" in scrollback
def test_user_message_is_recorded_only_in_target_and_printed_only_when_focused():
    console = Console(file=StringIO(), width=100, color_system=None)
    tui = TUI(console)

    tui.record_user_message(2, "please inspect the renderer")

    assert tui._state_for(0).history_blocks == []
    assert len(tui._state_for(2).history_blocks) == 1
    # Focus is still agent 0, so agent 2's line has not reached the terminal.
    assert "please inspect the renderer" not in _scrollback(tui)

    tui.select_agent(2)
    assert "please inspect the renderer" in _scrollback(tui)


def test_stop_live_settles_child_final_text_and_error_into_its_history():
    console = Console(file=StringIO(), width=100, color_system=None)
    tui = TUI(console)
    tui._live_paused = True
    tui.event_handler(SessionRuntimeEvent("text_delta", {"content": "final child text", "aid": 1}))
    tui.event_handler(SessionRuntimeEvent("error", {"reason": "child error", "aid": 1}))

    tui.stop_live()
    tui.select_agent(1)

    assert tui._state_for(1).current_text == ""
    scrollback = _scrollback(tui)
    assert "final child text" in scrollback
    assert "Error: child error" in scrollback


def test_agent_history_accumulates_across_turn_resets():
    console = Console(file=StringIO(), width=100, color_system=None)
    tui = TUI(console)
    tui.event_handler(SessionRuntimeEvent("text_delta", {"content": "turn one", "aid": 1}))
    tui.stop_live()

    tui.reset()
    tui.event_handler(SessionRuntimeEvent("text_delta", {"content": "turn two", "aid": 1}))
    tui.stop_live()
    tui.select_agent(1)

    scrollback = _scrollback(tui)
    assert "turn one" in scrollback
    assert "turn two" in scrollback


def test_scrollback_is_append_only_across_turns_for_the_focused_agent():
    """A new turn must not reprint the previous one — the terminal already has it."""
    console = Console(file=StringIO(), width=100, color_system=None)
    tui = TUI(console)
    tui.event_handler(SessionRuntimeEvent("text_delta", {"content": "old answer", "aid": 0}))
    tui.stop_live()

    tui.reset()
    tui.event_handler(SessionRuntimeEvent("text_delta", {"content": "new answer", "aid": 0}))
    tui.stop_live()

    scrollback = _scrollback(tui)
    assert scrollback.count("old answer") == 1
    assert scrollback.count("new answer") == 1
    assert scrollback.index("old answer") < scrollback.index("new answer")


def test_switching_does_not_lose_partial_text_or_history_order():
    tui = _make_tui()
    tui.event_handler(SessionRuntimeEvent("text_delta", {"content": "lead-a", "aid": 0}))
    tui.event_handler(SessionRuntimeEvent("text_delta", {"content": "child-a", "aid": 1}))
    tui.event_handler(
        SessionRuntimeEvent(
            "tool_start",
            {"tool": "bash", "args": {"command": "pwd"}, "aid": 1},
        )
    )
    tui.event_handler(SessionRuntimeEvent("text_delta", {"content": "lead-b", "aid": 0}))

    assert tui._state_for(0).current_text == "lead-alead-b"
    assert tui._state_for(1).current_text == ""
    assert _history_plains(tui, 1) == ["▸ A1:bash started pwd"]
    assert len(tui._state_for(1).history_blocks) == 2
    tui.select_agent(1)
    assert "A1:bash" in tui._active_tools


def test_legacy_event_without_aid_routes_to_lead_not_current_focus():
    tui = TUI()
    tui.event_handler(SessionRuntimeEvent("text_delta", {"content": "child", "aid": 1}))
    tui.select_agent(1)

    tui.event_handler(SessionRuntimeEvent("text_delta", {"content": "lead"}))

    assert tui._state_for(0).current_text == "lead"
    assert tui._state_for(1).current_text == "child"


def test_stop_live_settles_every_agent_but_prints_only_the_focused_one():
    tui = _make_tui()
    tui.event_handler(SessionRuntimeEvent("text_delta", {"content": "lead answer", "aid": 0}))
    tui.event_handler(SessionRuntimeEvent("text_delta", {"content": "child notes", "aid": 1}))

    tui.stop_live()

    # Both agents' streamed text is committed, so neither is lost on a later switch.
    assert tui._state_for(0).current_text == ""
    assert tui._state_for(1).current_text == ""
    scrollback = _scrollback(tui)
    assert "lead answer" in scrollback
    assert "child notes" not in scrollback

    tui.select_agent(1)
    assert "child notes" in _scrollback(tui)
