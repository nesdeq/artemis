"""
Shared frontend I/O for cli.py and tui.py.

Pure functions for save / export file resolution and content generation, plus
helpers for /cost row preparation. No rendering — the frontends do that
their own way (rich Panels vs Textual widgets).
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

import _config


_FILENAME_SANITIZE_RE = re.compile(r'[<>:"/\\|?*]')
_PREVIEW_CHARS = 60  # truncation length for /save query preview


# ---------------------------------------------------------------- Path helpers

def resolve_save_path(command: str, default_prefix: str,
                      data_dir: Optional[Path] = None) -> Path:
    """Resolve a target path from a /save-like command.

    Parses an optional filename arg, sanitizes it, ensures .md extension,
    generates a timestamped default if absent, and uniquifies on collision.
    """
    parts = command.strip().split(maxsplit=1)
    if len(parts) > 1:
        filename = _FILENAME_SANITIZE_RE.sub("_", parts[1].strip())
        if not filename.endswith(".md"):
            filename += ".md"
    else:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{default_prefix}_{ts}.md"

    save_dir = Path(data_dir if data_dir is not None else _config.data_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    filepath = save_dir / filename
    if filepath.exists():
        base, ext = filepath.stem, filepath.suffix
        counter = 2
        while filepath.exists():
            filepath = save_dir / f"{base}_{counter}{ext}"
            counter += 1
    return filepath


# ---------------------------------------------------------------- /save

class SaveResult(NamedTuple):
    filepath: Path
    size: int
    words: int
    preview: str


def find_last_exchange(messages: List[Dict[str, str]]
                       ) -> Tuple[Optional[str], Optional[str]]:
    """Return (last_user, last_assistant) walking back through messages."""
    last_user = last_assistant = None
    for msg in reversed(messages):
        if msg["role"] == "assistant" and last_assistant is None:
            last_assistant = msg["content"]
        elif msg["role"] == "user" and last_user is None:
            last_user = msg["content"]
        if last_user and last_assistant:
            break
    return last_user, last_assistant


def save_last_exchange(command: str,
                       messages: List[Dict[str, str]]) -> SaveResult:
    """Persist the last user/assistant exchange as markdown.

    Raises ValueError if there is no complete exchange yet.
    """
    last_user, last_assistant = find_last_exchange(messages)
    if not last_user or not last_assistant:
        raise ValueError("Nothing to save yet.")

    filepath = resolve_save_path(command, "artemis")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = (
        f"# Query\n\n{last_user}\n\n"
        f"# Response\n\n{last_assistant}\n\n"
        f"---\n*Saved: {ts}*\n"
    )
    filepath.write_text(content, encoding="utf-8")

    preview = (last_user[:_PREVIEW_CHARS] + "…"
               if len(last_user) > _PREVIEW_CHARS else last_user)
    return SaveResult(filepath, filepath.stat().st_size,
                      len(content.split()), preview)


# ---------------------------------------------------------------- /export

class ExportResult(NamedTuple):
    filepath: Path
    size: int
    words: int
    turns: int


def build_session_markdown(welcome: Optional[str],
                           turns: List[Dict[str, Any]],
                           excluded_agents: Optional[Iterable[str]] = None
                           ) -> str:
    """Render the full session as markdown for /export."""
    excluded = set(excluded_agents if excluded_agents is not None
                   else _config.excluded_metadata_agents)
    sections: List[str] = []

    if welcome:
        sections.append(f"# Welcome\n\n{welcome}")

    for i, turn in enumerate(turns, 1):
        block = [
            f"# Turn {i}",
            f"## Query\n\n{turn['user']}",
            f"## Response\n\n{turn['assistant']}",
        ]
        srcs = [
            f"**{name}**\n" + "\n".join(f"- {k}: {v}" for k, v in meta.items())
            for name, meta in (turn.get("metadata") or {}).items()
            if name not in excluded and meta
        ]
        if srcs:
            block.append("## Sources\n\n" + "\n\n".join(srcs))
        total = turn["input_tokens"] + turn["output_tokens"]
        block.append(
            f"## Tokens\n\n{turn['input_tokens']:,} in  |  "
            f"{turn['output_tokens']:,} out  |  {total:,} total"
        )
        sections.append("\n\n".join(block))

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections.append(f"---\n*Exported: {ts}*")
    return "\n\n".join(sections) + "\n"


def export_session(command: str,
                   welcome: Optional[str],
                   turns: List[Dict[str, Any]]) -> ExportResult:
    """Persist the full session as markdown.

    Raises ValueError if there's nothing to export yet.
    """
    if not turns:
        raise ValueError("Nothing to export yet.")

    filepath = resolve_save_path(command, "artemis_session")
    content = build_session_markdown(welcome, turns)
    filepath.write_text(content, encoding="utf-8")
    return ExportResult(filepath, filepath.stat().st_size,
                        len(content.split()), len(turns))


# ---------------------------------------------------------------- /cost

def sorted_cost_contexts(costs: Dict[str, Dict[str, Any]]
                         ) -> List[Tuple[str, Dict[str, Any]]]:
    """Sort: 'main' first, then agents alphabetically."""
    return sorted(costs.items(), key=lambda kv: (kv[0] != "main", kv[0]))


def short_model_name(model: str) -> str:
    """Strip provider prefix: 'openai/gpt-5.1' → 'gpt-5.1'."""
    return model.split("/")[-1] if "/" in model else model
