"""FileReader agent: read text/document files referenced by path in user input."""
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import _config
from .Agent import Agent
from tools.utils import format_blocks, shared_markitdown

logger = logging.getLogger(__name__)


class FileReaderAgent(Agent):
    """Read file contents when the user mentions an absolute or home path."""

    TEXT_EXTENSIONS = {
        '.txt', '.py', '.js', '.json', '.yml', '.yaml', '.conf',
        '.ini', '.log', '.md', '.rst', '.csv', '.sh',
        '.bash', '.zsh', '.fish', '.sql', '.xml', '.html', '.css',
    }

    # Never read these — they routinely hold credentials and must not be fed to
    # the LLM. Checked by extension AND exact name (a bare ".env" has no suffix).
    _SENSITIVE_EXTENSIONS = {'.env', '.pem', '.key', '.p12', '.pfx', '.crt', '.keystore'}
    _SENSITIVE_NAMES = {'.env', '.netrc', '.htpasswd', '.pgpass'}

    # Quoted or bare absolute / home-relative path. Stops at whitespace, ', ", :.
    _PATH_RE = re.compile(r'(?<![:/])(?:\'|")?((?:/|~/)(?:[^\'"\s/:]+/?)+)(?:\'|")?')

    _SENSITIVE_DIRS = ('/etc', '/root', '/var/log', '/proc', '/sys',
                       '/boot', '/dev', '/run', '/snap')

    def __init__(self, name: str, user: Optional[str] = None) -> None:
        super().__init__(name, user)
        self.markitdown = shared_markitdown

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        return bool(self._extract_filenames(user_input))

    def process(self, user_input: str, last_response: Optional[str] = None) -> Optional[str]:
        self.metadata = {"files": []}

        filenames = self._extract_filenames(user_input)
        if not filenames:
            return None

        files = self._read_files(filenames)
        if not files:
            return None

        self.metadata["files"] = [os.path.basename(f['filename']) for f in files]
        return self._create_context(files)

    def _is_safe_path(self, path: str) -> bool:
        """Reject path-traversal targets and sensitive system locations."""
        try:
            resolved = str(Path(path).resolve())
            for sensitive in self._SENSITIVE_DIRS:
                if resolved.startswith(sensitive):
                    logger.warning(f"Blocked access to sensitive path: {path}")
                    return False
            return True
        except Exception as e:
            logger.warning(f"Path validation error for {path}: {e}")
            return False

    def _extract_filenames(self, text: str) -> List[str]:
        """Extract paths that look like /abs/path or ~/path and resolve to readable files."""
        matches = self._PATH_RE.findall(text)
        out: List[str] = []
        for raw in matches:
            clean = raw.strip("'\"")
            expanded = os.path.expanduser(clean)
            if not os.path.exists(expanded) or not self._is_safe_path(expanded):
                continue
            if os.path.isfile(expanded):
                out.append(clean)
        return out

    def _read_files(self, filenames: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for filename in filenames:
            expanded = os.path.expanduser(filename)
            # Re-validate at read time — the check in _extract_filenames is
            # subject to TOCTOU — and bound the read size before touching it.
            if not self._is_safe_path(expanded):
                continue
            try:
                size = os.path.getsize(expanded)
                if size > _config.file_reader_max_bytes:
                    logger.warning(
                        f"Skipping {expanded}: {size} bytes exceeds "
                        f"{_config.file_reader_max_bytes} byte limit"
                    )
                    continue
                content = self._read_file_content(expanded)
                if content is None:
                    continue
                out.append({
                    "filename": expanded,
                    "content": content,
                    "size": size,
                })
            except Exception as e:
                logger.error(f"Error reading file {expanded}: {e}")
        return out

    def _read_file_content(self, filename: str) -> Optional[str]:
        p = Path(filename)
        suffix = p.suffix.lower()
        if suffix in self._SENSITIVE_EXTENSIONS or p.name.lower() in self._SENSITIVE_NAMES:
            logger.warning(f"Refusing to read sensitive file: {filename}")
            return None
        if suffix in self.TEXT_EXTENSIONS:
            return self._read_text_file(filename)
        return self._read_with_markitdown(filename)

    def _read_text_file(self, filename: str) -> Optional[str]:
        for encoding in ('utf-8', 'latin-1', 'cp1252'):
            try:
                with open(filename, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading text file {filename} with {encoding}: {e}")
                break
        return None

    def _read_with_markitdown(self, filename: str) -> Optional[str]:
        try:
            return self.markitdown.convert(filename).text_content
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {e}")
            return None

    _CONTEXT_FIELDS = [
        ("Filename", "filename"),
        ("Size", "size_display"),
        ("Content", "content", "block"),
    ]

    def _create_context(self, files: List[Dict[str, Any]]) -> str:
        records = [{**f, "size_display": f"{f['size']} bytes"} for f in files]
        return format_blocks(records, self._CONTEXT_FIELDS)
