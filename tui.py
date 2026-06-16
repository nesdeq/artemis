"""
tui.py — the Artemis TUI frontend.

Backend: ArtemisCore. Commands: /save, /export, /cost, /exit.
Built on Textual; uses Textual's theme system. The parchment look is a
registered theme — switch via the command palette to any other theme.

Run with:  python tui.py   (or the `arti` launcher)
"""

# Silence library logging before any other imports
import logging
logging.disable(logging.CRITICAL)

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.text import Text as RichText
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll, Vertical
from textual.theme import Theme
from textual.widgets import DataTable, Header, Input, Markdown, Static

import _config
import frontend_io
from core import ArtemisCore
from llms.LLMInterface import LLMInterface


# ----------------------------------------------------------------------
# Parchment theme — register once, switchable via command palette
# ----------------------------------------------------------------------

ARTEMIS_PARCHMENT = Theme(
    name="artemis-parchment",
    primary="#e8a85a",     # amber
    secondary="#d97757",   # sepia
    accent="#e8a85a",
    background="#1d1612",
    surface="#241c16",
    panel="#241c16",
    foreground="#e8d6b3",  # parchment cream
    success="#a8c878",
    warning="#e8b85a",
    error="#d96f5a",
    dark=True,
)


# ----------------------------------------------------------------------
# History-aware Input
# ----------------------------------------------------------------------

class HistoryInput(Input):
    """Input with persistent history (prompt_toolkit FileHistory format).

    - Up / Down → cycle through history (preserves the in-flight draft)
    - Ctrl+R    → reverse incremental search; status shown in border_title
    - Escape    → exit search mode

    History is read/written in prompt_toolkit's FileHistory format — the
    established on-disk format for .artemis_history.
    """

    BINDINGS = [
        Binding("up", "history_prev", "prev", show=False),
        Binding("down", "history_next", "next", show=False),
        Binding("ctrl+r", "start_search", "search", show=False),
    ]

    def __init__(self, history_path: Path, **kwargs):
        super().__init__(**kwargs)
        self._history_path = history_path
        self._history: List[str] = []
        self._cursor: int = 0
        self._draft: str = ""
        self._search_mode: bool = False
        self._search_query: str = ""

    def on_mount(self) -> None:
        self._history = self._load_history()
        self._cursor = len(self._history)

    # -- File I/O (prompt_toolkit FileHistory format) -------------------

    def _load_history(self) -> List[str]:
        """Parse prompt_toolkit FileHistory format: blocks of '+'-prefixed lines."""
        if not self._history_path.exists():
            return []
        try:
            text = self._history_path.read_text(encoding="utf-8")
        except OSError:
            return []
        entries: List[str] = []
        current: List[str] = []
        for line in text.splitlines():
            if line.startswith("+"):
                current.append(line[1:])
            else:
                if current:
                    entries.append("\n".join(current))
                    current = []
        if current:
            entries.append("\n".join(current))
        return entries

    def append_to_history(self, entry: str) -> None:
        """Add an entry and persist it in prompt_toolkit-compatible format."""
        if not entry.strip():
            return
        if self._history and self._history[-1] == entry:
            self._cursor = len(self._history)
            return
        self._history.append(entry)
        self._cursor = len(self._history)
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            with self._history_path.open("a", encoding="utf-8") as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                f.write(f"\n# {ts}\n")
                for line in entry.splitlines():
                    f.write(f"+{line}\n")
        except OSError:
            pass

    # -- Navigation -----------------------------------------------------

    def action_history_prev(self) -> None:
        if self._search_mode or not self._history:
            return
        if self._cursor == len(self._history):
            self._draft = self.value
        if self._cursor > 0:
            self._cursor -= 1
            self.value = self._history[self._cursor]
            self.cursor_position = len(self.value)

    def action_history_next(self) -> None:
        if self._search_mode:
            return
        if self._cursor < len(self._history) - 1:
            self._cursor += 1
            self.value = self._history[self._cursor]
            self.cursor_position = len(self.value)
        elif self._cursor == len(self._history) - 1:
            self._cursor = len(self._history)
            self.value = self._draft
            self._draft = ""
            self.cursor_position = len(self.value)

    # -- Reverse incremental search -------------------------------------

    def action_start_search(self) -> None:
        if self._search_mode:
            # Pressing Ctrl+R while already searching → step to previous match
            self._step_match()
            return
        if not self._history:
            return
        self._draft = self.value
        self._search_mode = True
        self._search_query = ""
        self._match_index: Optional[int] = None
        self._refresh_search()

    def _find_match(self, query: str, before: Optional[int] = None) -> Optional[int]:
        if not query:
            return None
        end = before if before is not None else len(self._history)
        for i in range(end - 1, -1, -1):
            if query in self._history[i]:
                return i
        return None

    def _refresh_search(self) -> None:
        idx = self._find_match(self._search_query)
        self._match_index = idx
        if idx is not None:
            self.value = self._history[idx]
        elif not self._search_query:
            self.value = ""
        self.cursor_position = len(self.value)
        self.border_title = f"reverse-i-search ‹{self._search_query}›"

    def _step_match(self) -> None:
        """Jump to the next-older match for the current query."""
        if self._match_index is None:
            return
        idx = self._find_match(self._search_query, before=self._match_index)
        if idx is None:
            return
        self._match_index = idx
        self.value = self._history[idx]
        self.cursor_position = len(self.value)

    def _exit_search(self, restore_draft: bool = False) -> None:
        self._search_mode = False
        self._search_query = ""
        self._match_index = None
        self.border_title = None
        if restore_draft:
            self.value = self._draft
            self.cursor_position = len(self.value)
        self._draft = ""

    # -- Key interception ----------------------------------------------

    async def _on_key(self, event: events.Key) -> None:
        if self._search_mode:
            if event.key == "escape":
                self._exit_search(restore_draft=True)
                event.stop()
                event.prevent_default()
                return
            if event.key == "enter":
                self._exit_search(restore_draft=False)
                await super()._on_key(event)
                return
            if event.key == "backspace":
                if self._search_query:
                    self._search_query = self._search_query[:-1]
                    self._refresh_search()
                event.stop()
                event.prevent_default()
                return
            if event.character and event.is_printable:
                self._search_query += event.character
                self._refresh_search()
                event.stop()
                event.prevent_default()
                return
            event.stop()
            event.prevent_default()
            return

        await super()._on_key(event)


# ----------------------------------------------------------------------
# Message widgets — all colors via Textual theme variables
# ----------------------------------------------------------------------

class WelcomeBanner(Static):
    """Initial welcome line."""

    def __init__(self, message: str):
        body = f"[$text-muted]⊰[/]  [b $primary]{message}[/]  [$text-muted]⊱[/]"
        super().__init__(body, markup=True, classes="welcome")


class WelcomeHint(Static):
    """One-line capability hint shown under the welcome banner on first run."""

    def __init__(self):
        body = (
            "[$text-muted]ask anything · paste a URL or file path · [/]"
            "[$secondary]!news !games !finance[/]"
            "[$text-muted] · [/][$secondary]/save /export /cost /exit[/]"
        )
        super().__init__(body, markup=True, classes="welcome-hint")


class UserMessage(Static):
    """User input echoed with the theme's primary accent."""

    def __init__(self, text: str):
        body = f"[b $primary]you ⟩[/]  [$foreground]{text}[/]"
        super().__init__(body, markup=True, classes="user-msg")


class ArtemisMessage(Markdown):
    """Streaming markdown response from artemis.

    Streaming is driven by a Textual MarkdownStream (created in
    _stream_response), which coalesces incoming bursts and appends only the
    newly-arrived lines. The previous approach called Markdown.update() on every
    chunk, re-parsing the entire growing document each time — O(n²) work that
    made long responses visibly lag. MarkdownStream is the purpose-built fix.
    """

    def __init__(self):
        super().__init__("", classes="artemis-msg")


class ArtemisLabel(Static):
    """Small 'artemis ⟩' label rendered above an ArtemisMessage."""

    def __init__(self):
        super().__init__("[b $secondary]artemis ⟩[/]", markup=True, classes="artemis-label")


class ThinkingIndicator(Static):
    """Vintage reel — a quarter-circle rotates beside 'thinking'."""

    GLYPHS = ["◐", "◓", "◑", "◒"]
    INTERVAL = 0.16  # seconds per frame

    def __init__(self):
        super().__init__(self._frame_text(0), markup=True, classes="thinking")
        self._i = 0
        self._timer = None

    def _frame_text(self, i: int) -> str:
        return f"[$primary]{self.GLYPHS[i % len(self.GLYPHS)]}[/]  [italic $text-muted]thinking[/]"

    def on_mount(self) -> None:
        self._timer = self.set_interval(self.INTERVAL, self._tick)

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()

    def _tick(self) -> None:
        self._i += 1
        self.update(self._frame_text(self._i))


class SourcesPanel(Static):
    """Sources box rendered under an assistant turn."""

    def __init__(self, metadata: Dict[str, Dict[str, Any]]):
        excluded = _config.excluded_metadata_agents
        lines: List[str] = []
        for agent, meta in metadata.items():
            if agent in excluded or not meta:
                continue
            entries = "   ".join(
                f"[$text-muted]{k}[/] [$foreground]{v}[/]" for k, v in meta.items()
            )
            lines.append(f"[b $secondary]{agent}[/]   {entries}")
        body = "\n".join(lines)
        super().__init__(body, markup=True, classes="sources")
        self.display = bool(lines)


class CostPanel(Vertical):
    """Bordered card with a title and a DataTable of session costs."""

    DEFAULT_CSS = """
    CostPanel {
        height: auto;
        margin: 1 0 1 0;
        padding: 1 2;
        border: round $primary 30%;
        background: $surface;
    }
    CostPanel > .cost-title {
        margin: 0 0 1 0;
        height: 1;
    }
    CostPanel DataTable {
        background: $surface;
        height: auto;
        max-height: 14;
        scrollbar-size-vertical: 1;
    }
    CostPanel DataTable > .datatable--header {
        background: $surface;
        color: $primary;
        text-style: bold;
    }
    CostPanel DataTable > .datatable--cursor,
    CostPanel DataTable > .datatable--hover,
    CostPanel DataTable > .datatable--odd-row,
    CostPanel DataTable > .datatable--even-row {
        background: $surface;
    }
    """

    def __init__(self, costs: Dict[str, Dict[str, Any]], totals: tuple):
        super().__init__()
        self._costs = costs
        self._totals = totals
        self._table = DataTable(show_header=True, show_cursor=False, zebra_stripes=False)

    def compose(self) -> ComposeResult:
        yield Static("[b $text-muted]› cost[/]", markup=True, classes="cost-title")
        yield self._table

    def on_mount(self) -> None:
        # Pull theme-specific accent colors at mount time so the table picks up
        # whatever theme is currently active (parchment by default).
        theme = self.app.current_theme
        accent_row_total = theme.secondary or "#d97757"
        accent_grand_total = theme.success or "#a8c878"

        t = self._table
        t.add_columns("context", "model", "in", "out", "$ total")

        for ctx, d in frontend_io.sorted_cost_contexts(self._costs):
            tot = d["input_cost"] + d["output_cost"]
            t.add_row(
                RichText(ctx),
                RichText(frontend_io.short_model_name(d["model"]), style="dim"),
                RichText(f"{d['input_tokens']:,}", justify="right"),
                RichText(f"{d['output_tokens']:,}", justify="right"),
                RichText(f"${tot:.4f}", justify="right", style=f"bold {accent_row_total}"),
            )

        total_in, total_out, ic, oc = self._totals
        t.add_row(
            RichText("total", style="bold"),
            RichText(""),
            RichText(f"{total_in:,}", justify="right", style="bold"),
            RichText(f"{total_out:,}", justify="right", style="bold"),
            RichText(f"${ic + oc:.4f}", justify="right", style=f"bold {accent_grand_total}"),
        )


class SystemNotice(Static):
    """Output of /save, /export, /cost, or errors — distinct styled card."""

    # Map severity to a theme-variable name for the title colour.
    SEVERITY_VAR = {
        "info":    "text-muted",
        "success": "success",
        "warning": "warning",
        "error":   "error",
    }

    def __init__(self, title: str, body: str, severity: str = "info"):
        var = self.SEVERITY_VAR.get(severity, "text-muted")
        text = f"[b ${var}]› {title}[/]\n{body}"
        super().__init__(text, markup=True, classes=f"notice notice-{severity}")


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

class ArtemisTUI(App):
    """Cozy parchment TUI for Artemis. Theme switchable via command palette."""

    TITLE = "artemis"
    SUB_TITLE = ""
    ICON = "⊰"

    # All colours via Textual theme variables; no hardcoded hex.
    CSS = """
    Screen {
        background: $background;
        color: $foreground;
    }

    Header {
        background: $surface;
        color: $primary;
        height: 1;
    }

    #chat {
        background: $background;
        padding: 1 3;
        /* Hide the scrollbar entirely. show_vertical_scrollbar stays True on
           overflow (it is overflow-driven, not size-driven), so mouse-wheel and
           keyboard scrolling keep working — only the visible bar is gone. */
        scrollbar-size: 0 0;
    }

    .welcome {
        text-align: center;
        margin: 2 0 0 0;
        color: $primary;
    }

    .welcome-hint {
        text-align: center;
        margin: 0 0 2 0;
    }

    UserMessage {
        height: auto;
        margin: 1 0 1 0;
    }

    ArtemisLabel {
        height: auto;
        margin: 0;
        padding: 0;
    }

    ArtemisMessage {
        height: auto;
        margin: 0 0 1 0;
        padding: 0;
        background: $background;
    }

    ArtemisMessage MarkdownBlock {
        background: $background;
        color: $foreground;
    }

    ArtemisMessage MarkdownH1, ArtemisMessage MarkdownH2,
    ArtemisMessage MarkdownH3, ArtemisMessage MarkdownH4 {
        color: $primary;
        background: $background;
        text-style: bold;
    }

    ArtemisMessage MarkdownFence, ArtemisMessage MarkdownCode {
        background: $surface;
        color: $foreground;
    }

    ArtemisMessage MarkdownBlockQuote {
        border-left: thick $primary 30%;
        color: $text-muted;
        background: $background;
    }

    ArtemisMessage MarkdownH5, ArtemisMessage MarkdownH6 {
        color: $primary;
        background: $background;
        text-style: bold;
    }

    ArtemisMessage MarkdownBullet {
        color: $secondary;
    }

    ArtemisMessage MarkdownHorizontalRule {
        border-bottom: dashed $primary 30%;
    }

    ArtemisMessage MarkdownTable {
        border: round $primary 30%;
    }

    ArtemisMessage MarkdownTH {
        color: $primary;
        text-style: bold;
    }

    .thinking {
        height: auto;
        margin: 0 0 1 0;
        color: $text-muted;
    }

    .sources {
        height: auto;
        margin: 0 0 1 2;
        padding: 0 0 0 2;
        border-left: thick $primary 30%;
        color: $text-muted;
    }

    .notice {
        height: auto;
        margin: 1 0 1 0;
        padding: 1 2;
        border: round $primary 30%;
        background: $surface;
    }

    .notice-warning { border: round $warning; }
    .notice-error   { border: round $error; }
    .notice-success { border: round $success; }

    #prompt {
        background: $background !important;
        background-tint: transparent !important;
        color: $foreground;
        border: round $primary 30% !important;
        margin: 0 3 0 3;
        padding: 0 1;
    }

    #prompt:focus {
        border: round $primary !important;
        background-tint: transparent !important;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "request_quit", "quit"),
        Binding("ctrl+l", "clear_chat", "clear"),
    ]

    def __init__(self):
        super().__init__()
        self.core = ArtemisCore()
        self.session_turns: List[Dict[str, Any]] = []
        self.welcome_message: Optional[str] = None
        self._busy = False

    # -- Layout -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield VerticalScroll(id="chat")
        history_path = Path(_config.data_directory) / ".artemis_history"
        yield HistoryInput(
            history_path=history_path,
            placeholder=" ⟩  type a message  ·  /save /export /cost /exit",
            id="prompt",
        )

    async def on_mount(self) -> None:
        self.register_theme(ARTEMIS_PARCHMENT)
        self.theme = "artemis-parchment"
        self._refresh_subtitle(0, 0)
        self._init_welcome()
        self.query_one("#prompt", Input).focus()

    @work(exclusive=True, group="welcome")
    async def _init_welcome(self) -> None:
        try:
            welcome = await self.core.ensure_opening_message()
        except Exception as e:
            self.query_one("#chat", VerticalScroll).mount(
                SystemNotice("startup", f"Couldn't fetch welcome: {e}", severity="error")
            )
            return
        self.welcome_message = welcome
        self._mount_welcome(self.query_one("#chat", VerticalScroll))

    def _mount_welcome(self, chat: VerticalScroll) -> None:
        """Mount the welcome banner plus the one-line capability hint."""
        chat.mount(WelcomeBanner(self.welcome_message or ""))
        chat.mount(WelcomeHint())

    # -- Input handling --------------------------------------------------

    @on(Input.Submitted, "#prompt")
    async def _on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            event.input.value = ""
            return
        if self._busy:
            # Still streaming a reply. Keep the user's text in the box (don't
            # clear it or push it to history) so the message isn't silently
            # dropped — they can resend the instant the response lands.
            self.bell()
            return
        self.query_one("#prompt", HistoryInput).append_to_history(text)
        event.input.value = ""
        try:
            if text.startswith("/"):
                await self._handle_command(text)
            else:
                self._send_message(text)
        except Exception as e:
            self._mount_notice("error", str(e), severity="error")

    def _send_message(self, text: str) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        chat.mount(UserMessage(text))
        thinking = ThinkingIndicator()
        chat.mount(thinking)
        chat.scroll_end(animate=False)
        self._stream_response(text, thinking)

    @work(exclusive=True)
    async def _stream_response(self, text: str, thinking: ThinkingIndicator) -> None:
        self._busy = True
        chat = self.query_one("#chat", VerticalScroll)
        # Pin the view to the bottom as content streams in; Textual releases the
        # anchor automatically the moment the user scrolls up. Cheaper and
        # smoother than calling scroll_end() on every chunk.
        chat.anchor()
        msg_widget: Optional[ArtemisMessage] = None
        stream = None

        response = ""
        metadata: Optional[Dict[str, Dict[str, Any]]] = None
        in_tokens = out_tokens = 0

        try:
            async for chunk, meta, it, ot in self.core.get_ai_response(text):
                if meta:
                    metadata = meta
                if it:
                    in_tokens = it
                if ot:
                    out_tokens = ot
                if chunk:
                    if msg_widget is None:
                        if thinking.is_mounted:
                            await thinking.remove()
                        chat.mount(ArtemisLabel())
                        msg_widget = ArtemisMessage()
                        chat.mount(msg_widget)
                        stream = Markdown.get_stream(msg_widget)
                    response += chunk
                    await stream.write(chunk)

            if stream is not None:
                await stream.stop()

            if metadata:
                panel = SourcesPanel(metadata)
                if panel.display:
                    chat.mount(panel)

            self.session_turns.append({
                "user": text,
                "assistant": response,
                "metadata": metadata or {},
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
            })
            self._refresh_subtitle(in_tokens, out_tokens)
        except Exception as e:
            if stream is not None:
                await stream.stop()
            chat.mount(SystemNotice("error", str(e), severity="error"))
        finally:
            # MUST run even on CancelledError (a BaseException, which the
            # except-Exception above cannot catch). When an exclusive worker is
            # preempted by the next message, Textual cancels this coroutine here;
            # without the finally, _busy stayed True forever — silently dropping
            # every subsequent message — and the spinner leaked, still ticking.
            # _busy is cleared first (synchronous, guaranteed) before any await.
            self._busy = False
            if thinking.is_mounted:
                await thinking.remove()

    # -- Commands --------------------------------------------------------

    async def _handle_command(self, cmd: str) -> None:
        lower = cmd.strip().lower()
        if lower == "/exit":
            self.exit()
            return
        if lower.startswith("/save"):
            await self._cmd_save(cmd)
            return
        if lower.startswith("/export"):
            await self._cmd_export(cmd)
            return
        if lower == "/cost":
            await self._cmd_cost()
            return
        self._mount_notice("unknown command", cmd, severity="warning")

    async def _cmd_save(self, command: str) -> None:
        try:
            r = frontend_io.save_last_exchange(command, self.core.messages)
        except ValueError as e:
            self._mount_notice("save", str(e), severity="warning")
            return
        except OSError as e:
            self._mount_notice("save", f"Failed: {e}", severity="error")
            return
        self._mount_notice("save", self._save_body(r, "query", r.preview), severity="success")

    async def _cmd_export(self, command: str) -> None:
        try:
            r = frontend_io.export_session(command, self.welcome_message, self.session_turns)
        except ValueError as e:
            self._mount_notice("export", str(e), severity="warning")
            return
        except OSError as e:
            self._mount_notice("export", f"Failed: {e}", severity="error")
            return
        self._mount_notice("export", self._save_body(r, "turns", str(r.turns)), severity="success")

    @staticmethod
    def _save_body(r, third_label: str, third_value: str) -> str:
        """Three-line summary used by /save and /export success notices."""
        return (
            f"[$text-muted]file[/]   {r.filepath}\n"
            f"[$text-muted]size[/]   {r.size:,} bytes  ·  {r.words:,} words\n"
            f"[$text-muted]{third_label}[/]  {third_value}"
        )

    async def _cmd_cost(self) -> None:
        costs = LLMInterface.get_session_costs()
        totals = LLMInterface.get_total_cost()

        if not costs:
            self._mount_notice("cost", "No usage recorded yet.", severity="warning")
            return

        chat = self.query_one("#chat", VerticalScroll)
        chat.mount(CostPanel(costs, totals))
        chat.scroll_end(animate=False)

    # -- Utility ---------------------------------------------------------

    def _mount_notice(self, title: str, body: str, severity: str = "info") -> None:
        chat = self.query_one("#chat", VerticalScroll)
        chat.mount(SystemNotice(title, body, severity=severity))
        chat.scroll_end(animate=False)

    def _refresh_subtitle(self, in_tok: int, out_tok: int) -> None:
        total_in = sum(t["input_tokens"] for t in self.session_turns)
        total_out = sum(t["output_tokens"] for t in self.session_turns)
        model = _config.llm.split("/")[-1] if "/" in _config.llm else _config.llm
        if total_in + total_out:
            self.sub_title = (
                f"{model}  ·  {total_in + total_out:,} session  ·  "
                f"last {in_tok:,}→{out_tok:,}"
            )
        else:
            self.sub_title = model

    # -- Actions ---------------------------------------------------------

    def action_request_quit(self) -> None:
        self.exit()

    async def action_clear_chat(self) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        await chat.remove_children()
        if self.welcome_message:
            self._mount_welcome(chat)


if __name__ == "__main__":
    ArtemisTUI().run()
