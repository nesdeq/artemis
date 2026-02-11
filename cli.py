"""
CLI interface for Artemis AI assistant.

Provides a command-line interface with rich text formatting, markdown rendering,
and streaming responses with enhanced visual styling.
"""

# Put logging disable at the very top before other imports
import logging; logging.basicConfig(level=logging.CRITICAL); logging.getLogger().setLevel(logging.CRITICAL); logging.disable(logging.CRITICAL)

import asyncio
import os
import re
import datetime
from typing import Dict, Any, Optional
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
from core import ArtemisCore
from llms.LLMInterface import LLMInterface
from tools.utils import UI_THEME, LOGO

# Use centralized configuration
REFRESH_RATE = _config.cli_refresh_rate


class ArtemisAI:
    """Main CLI application for Artemis AI assistant."""

    THEME = UI_THEME

    def __init__(self):
        """Initialize the CLI application."""
        self.core = ArtemisCore()
        self.console = Console()
        self.term_width = self.console.width
        
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
            width=min(100, self.term_width - 4)
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

    def _get_thinking_spinner(self) -> Align:
        """Generate an elegant thinking spinner with text."""
        with Progress(
            SpinnerColumn(spinner_name="dots", style=self.THEME["secondary"]),
            TextColumn(f"[bold {self.THEME['secondary']}]Thinking...[/bold {self.THEME['secondary']}]"),
            transient=True, 
            expand=False,
        ) as progress:
            progress.add_task("Thinking", total=None)
            # Just prepare the layout, don't actually run animation here
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
        
        # Set up a single live display for the entire response process
        
        # Panel width calculation (done once)
        panel_width = min(100, self.term_width - 4)
        
        # Function to create current display content
        def get_display():
            if not first_chunk_received:
                return self._get_thinking_spinner()
            elif response_text:
                return Panel(
                    Markdown(response_text),
                    title=response_header,
                    title_align="left",
                    border_style=self.THEME["dim"],
                    box=box.ROUNDED,
                    padding=(1, 2),
                    width=panel_width
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

    def save_last_exchange(self, command: str) -> None:
        """
        Save the last user/assistant exchange to a markdown file.

        Args:
            command: The full /save command, optionally with filename
        """

        # Check if there's an exchange to save
        if len(self.core.messages) < 2:
            self.console.print(
                f"[{self.THEME['warning']}]Nothing to save yet.[/{self.THEME['warning']}]"
            )
            return

        # Find last user and assistant messages
        last_user = None
        last_assistant = None
        for msg in reversed(self.core.messages):
            if msg['role'] == 'assistant' and last_assistant is None:
                last_assistant = msg['content']
            elif msg['role'] == 'user' and last_user is None:
                last_user = msg['content']
            if last_user and last_assistant:
                break

        if not last_user or not last_assistant:
            self.console.print(
                f"[{self.THEME['warning']}]No complete exchange to save.[/{self.THEME['warning']}]"
            )
            return

        # Parse optional filename from command
        parts = command.strip().split(maxsplit=1)
        if len(parts) > 1:
            filename = parts[1].strip()
            # Sanitize filename
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            if not filename.endswith('.md'):
                filename += '.md'
        else:
            # Auto-generate filename
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"artemis_{timestamp}.md"

        # Ensure save directory exists
        save_dir = Path(_config.save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Handle filename collision
        filepath = save_dir / filename
        if filepath.exists():
            base = filepath.stem
            ext = filepath.suffix
            counter = 2
            while filepath.exists():
                filepath = save_dir / f"{base}_{counter}{ext}"
                counter += 1

        # Format markdown content
        timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        content = f"""# Query

{last_user}

# Response

{last_assistant}

---
*Saved: {timestamp_str}*
"""

        # Write file
        try:
            filepath.write_text(content, encoding='utf-8')

            # Calculate stats
            file_size = filepath.stat().st_size
            word_count = len(content.split())
            query_preview = last_user[:50] + "..." if len(last_user) > 50 else last_user

            # Build save panel
            save_table = Table(box=None, show_header=False, show_edge=False, padding=(0, 1))
            save_table.add_column("Label", style=f"bold {self.THEME['primary']}")
            save_table.add_column("Value", style=self.THEME['text'])
            save_table.add_row("File", str(filepath))
            save_table.add_row("Size", f"{file_size:,} bytes  |  {word_count:,} words")
            save_table.add_row("Query", query_preview)

            self.console.print("")
            save_panel = Panel(
                save_table,
                title="save",
                title_align="left",
                border_style=self.THEME["dim"],
                box=box.ROUNDED,
                padding=(1, 2),
                width=min(100, self.term_width - 4)
            )
            self.console.print(save_panel)
            self.console.print("")
        except OSError as e:
            self.console.print(
                f"[{self.THEME['error']}]Failed to save: {e}[/{self.THEME['error']}]"
            )

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

        # Sort: "main" first, then agents alphabetically
        sorted_contexts = sorted(costs.keys(), key=lambda x: (x != "main", x))

        for ctx in sorted_contexts:
            data = costs[ctx]
            ctx_total = data['input_cost'] + data['output_cost']
            # Shorten model name for display
            model_short = data['model'].split('/')[-1] if '/' in data['model'] else data['model']
            cost_table.add_row(
                ctx,
                model_short,
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
            width=min(90, self.term_width - 4)
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
                user_input = await asyncio.get_event_loop().run_in_executor(
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