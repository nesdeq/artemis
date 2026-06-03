"""
CLI interface for Artemis AI assistant.

Provides a command-line interface with rich text formatting, markdown rendering,
and streaming responses with enhanced visual styling.
"""

# Silence library logging before any other imports.
import logging
logging.disable(logging.CRITICAL)

import asyncio
import os
from typing import Dict, Any, Iterable, List, Optional, Tuple
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.box import SIMPLE
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.align import Align
from rich.live import Live
from rich import box
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import ANSI

import _config
import frontend_io
from core import ArtemisCore
from llms.LLMInterface import LLMInterface
from tools.utils import UI_THEME, LOGO

REFRESH_RATE = _config.cli_refresh_rate


class ArtemisAI:
    """Main CLI application for Artemis AI assistant."""

    THEME = UI_THEME

    def __init__(self):
        """Initialize the CLI application."""
        self.core = ArtemisCore()
        self.console = Console()
        self.term_width = self.console.width
        self._panel_width = min(_config.cli_panel_max_width, self.term_width - _config.cli_panel_padding)
        self._cost_panel_width = min(_config.cli_cost_panel_max_width, self.term_width - _config.cli_panel_padding)
        self._thinking_spinner = self._build_thinking_spinner()
        self.session_turns: List[Dict[str, Any]] = []
        self.welcome_message: Optional[str] = None

    def create_sources_table(self, agent_metadata: Dict[str, Any]) -> Optional[Panel]:
        """
        Create a formatted panel displaying agent sources and metadata.
        
        Args:
            agent_metadata: Dictionary of agent metadata
            
        Returns:
            Formatted Rich panel or None if no relevant metadata
        """
        if not agent_metadata:
            return None
            
        excluded_agents = _config.excluded_metadata_agents
        
        sources_table = Table(show_header=False, expand=True, box=SIMPLE, show_edge=False)
        sources_table.add_column("Agent", style=f"bold {self.THEME['secondary']}")
        sources_table.add_column("Metadata")
        has_rows = False
        
        for agent_name, metadata in agent_metadata.items():
            if agent_name not in excluded_agents and metadata:
                has_rows = True
                metadata_str = "\n".join([f"{key}: {value}" for key, value in metadata.items()])
                sources_table.add_row(agent_name, metadata_str)
        
        if not has_rows:
            return None
            
        return Panel(
            sources_table,
            title="[bold]Sources[/bold]",
            border_style=self.THEME["dim"],
            box=box.ROUNDED,
            padding=(1, 2),
            title_align="left",
            width=self._panel_width,
        )

    def print_stats(self, input_tokens: int, output_tokens: int) -> None:
        """
        Print token usage statistics for current request.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        total_tokens = input_tokens + output_tokens

        stats_table = Table(box=None, show_header=False, show_edge=False, padding=(0, 1))
        stats_table.add_column("Label", style=f"bold {self.THEME['primary']}")
        stats_table.add_column("Value", style=self.THEME['text'])
        stats_table.add_row("Tokens", f"{input_tokens:,} in  |  {output_tokens:,} out  |  {total_tokens:,} total")

        self.console.print(stats_table)

    def _build_thinking_spinner(self) -> Align:
        """Build the thinking spinner once. Reused across requests to avoid Progress re-creation."""
        progress = Progress(
            SpinnerColumn(spinner_name="dots", style=self.THEME["secondary"]),
            TextColumn(f"[bold {self.THEME['secondary']}]Thinking...[/bold {self.THEME['secondary']}]"),
            transient=True,
            expand=False,
        )
        progress.add_task("Thinking", total=None)
        return Align.center(progress)

    async def process_and_display_response(self, user_input: str) -> None:
        """
        Process user input and stream response with live markdown rendering.
        
        Args:
            user_input: User query text
        """
        # Print a newline before response
        self.console.print("")
        
        # Create response header
        response_header = Text()
        response_header.append("artemis", style=f"bold {self.THEME['secondary']}")
        response_header.append(" ⟩", style=self.THEME["dim"])
        
        # Response accumulation and tracking
        response_text = ""
        agent_metadata = None
        input_tokens = 0
        output_tokens = 0
        first_chunk_received = False
        
        # Function to create current display content
        def get_display():
            if not first_chunk_received:
                return self._thinking_spinner
            elif response_text:
                return Panel(
                    Markdown(response_text),
                    title=response_header,
                    title_align="left",
                    border_style=self.THEME["dim"],
                    box=box.ROUNDED,
                    padding=(1, 2),
                    width=self._panel_width,
                )
            else:
                return ""
        
        with Live(get_display(), console=self.console, refresh_per_second=REFRESH_RATE) as live_display:
            async for chunk, metadata, in_tokens, out_tokens in self.core.get_ai_response(user_input):
                if metadata:
                    agent_metadata = metadata
                if in_tokens:
                    input_tokens = in_tokens
                if out_tokens:
                    output_tokens = out_tokens
                if chunk:
                    if not first_chunk_received:
                        first_chunk_received = True
                    response_text += chunk
                    live_display.update(get_display())
        
        # Display sources panel if available
        sources_panel = self.create_sources_table(agent_metadata)
        if sources_panel:
            self.console.print(sources_panel)

        # Display token statistics
        self.print_stats(input_tokens, output_tokens)

        # Add an empty line after the entire response
        self.console.print("")

        # Record turn for /export
        self.session_turns.append({
            'user': user_input,
            'assistant': response_text,
            'metadata': agent_metadata or {},
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
        })

    async def _display_welcome(self) -> None:
        """Display stylized welcome screen with snarky message."""
        # Clear screen for clean start (works on most terminals)
        self.console.clear()

        # Display ASCII logo with gradient styling
        art_text = Text(LOGO)
        art_text.stylize(f"bold {self.THEME['secondary']}")
        self.console.print(Align.center(art_text))

        # Get welcome message from core (ensures it's the first message)
        welcome_message = await self.core.ensure_opening_message()
        self.welcome_message = welcome_message

        # Welcome message
        welcome_text = Text()
        welcome_text.append("\n⊰ ", style=self.THEME["dim"])
        welcome_text.append(welcome_message, style=f"bold {self.THEME['primary']}")
        welcome_text.append(" ⊱\n", style=self.THEME["dim"])
        
        # Version info
        info = Text("Version 1.0.0 • Type ", style=self.THEME["dim"])
        info.append("/exit", style=self.THEME["warning"])
        info.append(" to quit", style=self.THEME["dim"])
        
        welcome_panel = Panel(
            Align.center(welcome_text + Text("\n") + info),
            box=box.ROUNDED,
            border_style=self.THEME["dim"],
            padding=(1, 2)
        )
        
        self.console.print(Align.center(welcome_panel))
        self.console.print()
    
    def _get_prompt_prefix(self) -> str:
        """Get styled prompt prefix for user input."""
        username = os.environ.get("USER", "user")
        prompt_color = "\033[1;38;5;45m"  # ANSI cyan bold
        prompt_prefix = f"{prompt_color}{username}\033[0m \033[38;5;240m⟩\033[0m "
        return prompt_prefix

    def _print_result_panel(self, title: str, rows: Iterable[Tuple[str, str]]) -> None:
        """Print a label/value summary panel (used by /save, /export)."""
        table = Table(box=None, show_header=False, show_edge=False, padding=(0, 1))
        table.add_column("Label", style=f"bold {self.THEME['primary']}")
        table.add_column("Value", style=self.THEME['text'])
        for label, value in rows:
            table.add_row(label, value)

        self.console.print("")
        self.console.print(Panel(
            table,
            title=title,
            title_align="left",
            border_style=self.THEME["dim"],
            box=box.ROUNDED,
            padding=(1, 2),
            width=self._panel_width,
        ))
        self.console.print("")

    def _warn(self, message: str) -> None:
        self.console.print(f"[{self.THEME['warning']}]{message}[/{self.THEME['warning']}]")

    def _error(self, message: str) -> None:
        self.console.print(f"[{self.THEME['error']}]{message}[/{self.THEME['error']}]")

    def save_last_exchange(self, command: str) -> None:
        """Save the last user/assistant exchange to a markdown file."""
        try:
            r = frontend_io.save_last_exchange(command, self.core.messages)
        except ValueError as e:
            self._warn(str(e))
            return
        except OSError as e:
            self._error(f"Failed to save: {e}")
            return

        self._print_result_panel("save", [
            ("File", str(r.filepath)),
            ("Size", f"{r.size:,} bytes  |  {r.words:,} words"),
            ("Query", r.preview),
        ])

    def export_session(self, command: str) -> None:
        """Export the whole session to a markdown file."""
        try:
            r = frontend_io.export_session(command, self.welcome_message, self.session_turns)
        except ValueError as e:
            self._warn(str(e))
            return
        except OSError as e:
            self._error(f"Failed to export: {e}")
            return

        self._print_result_panel("export", [
            ("File", str(r.filepath)),
            ("Size", f"{r.size:,} bytes  |  {r.words:,} words"),
            ("Turns", str(r.turns)),
        ])

    def show_session_cost(self) -> None:
        """Display the cost of the current session with per-context breakdown."""
        costs = LLMInterface.get_session_costs()
        total_in, total_out, total_in_cost, total_out_cost = LLMInterface.get_total_cost()

        if not costs:
            self.console.print(f"[{self.THEME['dim']}]No usage recorded yet.[/{self.THEME['dim']}]")
            return

        # Build cost table with columns for context breakdown
        cost_table = Table(
            box=None,
            show_header=True,
            header_style=f"bold {self.THEME['secondary']}",
            padding=(0, 1)
        )
        cost_table.add_column("Context", style=f"bold {self.THEME['text']}")
        cost_table.add_column("Model", style=self.THEME['dim'])
        cost_table.add_column("In", justify="right")
        cost_table.add_column("Out", justify="right")
        cost_table.add_column("$ In", justify="right", style=self.THEME['text'])
        cost_table.add_column("$ Out", justify="right", style=self.THEME['text'])
        cost_table.add_column("$ Total", justify="right", style=f"bold {self.THEME['primary']}")

        for ctx, data in frontend_io.sorted_cost_contexts(costs):
            ctx_total = data['input_cost'] + data['output_cost']
            cost_table.add_row(
                ctx,
                frontend_io.short_model_name(data['model']),
                f"{data['input_tokens']:,}",
                f"{data['output_tokens']:,}",
                f"${data['input_cost']:.4f}",
                f"${data['output_cost']:.4f}",
                f"${ctx_total:.4f}"
            )

        # Add totals row
        grand_total = total_in_cost + total_out_cost
        cost_table.add_row(
            "[bold]total[/bold]",
            "",
            f"[bold]{total_in:,}[/bold]",
            f"[bold]{total_out:,}[/bold]",
            f"[bold]${total_in_cost:.4f}[/bold]",
            f"[bold]${total_out_cost:.4f}[/bold]",
            f"[bold green]${grand_total:.4f}[/bold green]"
        )

        self.console.print("")
        cost_panel = Panel(
            cost_table,
            title="cost",
            title_align="left",
            border_style=self.THEME["dim"],
            box=box.ROUNDED,
            padding=(1, 2),
            width=self._cost_panel_width,
        )
        self.console.print(cost_panel)
        self.console.print("")

    async def main(self) -> None:
        """Run the main CLI application loop."""
        # Display welcome screen with ASCII art and snarky message
        await self._display_welcome()
        
        # Setup command history
        history_path = Path(_config.data_directory) / '.artemis_history'
        history_path.parent.mkdir(parents=True, exist_ok=True)
        session = PromptSession(
            history=FileHistory(history_path),
            message=ANSI(self._get_prompt_prefix())
        )

        while True:
            try:
                # Get user input with styled prompt
                user_input = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: session.prompt()
                )

                # Check for exit command
                if user_input.strip().lower() == '/exit':
                    farewell_panel = Panel(
                        "Thank you for using Artemis AI",
                        title="Goodbye!",
                        title_align="center",
                        border_style=self.THEME["primary"],
                        box=box.ROUNDED,
                        padding=(1, 2)
                    )
                    self.console.print(Align.center(farewell_panel))
                    break

                # Handle /save command
                if user_input.strip().lower().startswith('/save'):
                    self.save_last_exchange(user_input)
                    continue

                # Handle /export command
                if user_input.strip().lower().startswith('/export'):
                    self.export_session(user_input)
                    continue

                # Handle /cost command
                if user_input.strip().lower() == '/cost':
                    self.show_session_cost()
                    continue

                # Process the input and display response
                await self.process_and_display_response(user_input)
                
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                error_panel = Panel(
                    f"{str(e)}", 
                    title="Error",
                    border_style=self.THEME["error"],
                    title_align="left",
                    box=box.ROUNDED
                )
                self.console.print(error_panel)

if __name__ == "__main__":
    artemis = ArtemisAI()
    asyncio.run(artemis.main())