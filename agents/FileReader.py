# FileReader.py
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .Agent import Agent
from tools.utils import shared_markitdown

# Configure logger
logger = logging.getLogger(__name__)


class FileReaderAgent(Agent):
    """Agent for reading and processing file contents."""

    TEXT_EXTENSIONS = {
        '.txt', '.py', '.js', '.json', '.yml', '.yaml', '.conf',
        '.ini', '.log', '.md', '.rst', '.csv', '.env', '.sh',
        '.bash', '.zsh', '.fish', '.sql', '.xml', '.html', '.css'
    }

    def __init__(self, name: str, user: Optional[str] = None) -> None:
        super().__init__(name, user)
        self.markitdown = shared_markitdown

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        """Should we read files?"""
        return bool(self.extract_filenames(user_input))

    def process(self, user_input: str, last_response: Optional[str] = None) -> Optional[str]:
        """Read and process file contents."""
        self.metadata = {"files": []}

        filenames = self.extract_filenames(user_input)
        if not filenames:
            return None

        file_contents = self.read_files(filenames)
        if not file_contents:
            return None

        context = self.create_context(file_contents)
        self.metadata["files"] = [os.path.basename(f['filename']) for f in file_contents]
        return context

    def _is_safe_path(self, path: str) -> bool:
        """
        Validate that a path is safe to read (no path traversal).

        Args:
            path: Path to validate

        Returns:
            bool: True if path is safe
        """
        try:
            # Resolve to absolute path (handles .., symlinks, etc.)
            resolved = Path(path).resolve()

            # Block sensitive system directories
            sensitive_dirs = [
                '/etc', '/root', '/var/log', '/proc', '/sys',
                '/boot', '/dev', '/run', '/snap'
            ]
            for sensitive in sensitive_dirs:
                if str(resolved).startswith(sensitive):
                    logger.warning(f"Blocked access to sensitive path: {path}")
                    return False

            # Block hidden files in system directories
            if any(part.startswith('.') for part in resolved.parts[1:3]):
                if str(resolved).startswith(('/etc', '/root', '/var')):
                    logger.warning(f"Blocked access to hidden system file: {path}")
                    return False

            return True
        except Exception as e:
            logger.warning(f"Path validation error for {path}: {e}")
            return False

    def extract_filenames(self, text: str) -> List[str]:
        """Extract valid filenames from the input text.

        Args:
            text: Input text potentially containing filenames

        Returns:
            List[str]: Valid filenames that exist in the filesystem
        """
        # Match filenames that start with / or ~/ and may be quoted
        filename_pattern = r'(?<![:/])(?:\'|")?((?:/|~/)(?:[^\'"\s/:]+/?)+)(?:\'|")?'
        matches = re.findall(filename_pattern, text)

        filtered_matches = []
        for match in matches:
            clean_path = match.strip("'\"")
            expanded_path = os.path.expanduser(clean_path)

            # Validate path safety and existence
            if os.path.exists(expanded_path) and self._is_safe_path(expanded_path):
                # Only allow files, not directories
                if os.path.isfile(expanded_path):
                    filtered_matches.append(clean_path)
                else:
                    logger.debug(f"Skipping directory: {expanded_path}")

        return filtered_matches

    def read_files(self, filenames: List[str]) -> List[Dict[str, Any]]:
        """Read the content of all valid files.
        
        Args:
            filenames: List of filenames to read
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing file information and content
        """
        file_contents = []
        
        for filename in filenames:
            expanded_filename = os.path.expanduser(filename)
            try:
                content = self.read_file_content(expanded_filename)
                
                if content is None:
                    continue
                    
                file_size = os.path.getsize(expanded_filename)
                file_contents.append({
                    "filename": expanded_filename,
                    "content": content,
                    "size": file_size
                })
            except Exception as e:
                logger.error(f"Error reading file {expanded_filename}: {str(e)}")
                
        return file_contents

    def read_file_content(self, filename: str) -> Optional[str]:
        """Read content from a file based on its type.
        
        Args:
            filename: Path to the file
            
        Returns:
            Optional[str]: File content as text or None if reading fails
        """
        suffix = Path(filename).suffix.lower()
        
        # Handle text files directly for better performance
        if suffix in self.TEXT_EXTENSIONS:
            return self._read_text_file(filename)
        # Use MarkItDown for all other file types
        else:
            return self._read_with_markitdown(filename)

    def _read_text_file(self, filename: str) -> Optional[str]:
        """Read content from a text file with encoding fallback.
        
        Args:
            filename: Path to the text file
            
        Returns:
            Optional[str]: File content as text or None if reading fails
        """
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(filename, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading text file {filename} with {encoding}: {str(e)}")
                break
                
        return None

    def _read_with_markitdown(self, filename: str) -> Optional[str]:
        """Read content from a file using MarkItDown.
        
        Args:
            filename: Path to the file
            
        Returns:
            Optional[str]: File content as text or error message
        """
        try:
            result = self.markitdown.convert(filename)
            return result.text_content
        except Exception as e:
            logger.error(f"Error extracting text from {filename} with MarkItDown: {str(e)}")
            return f"Error: Unable to extract text from {filename}"

    def create_context(self, file_contents: List[Dict[str, Any]]) -> str:
        """Create a formatted context from the file contents.
        
        Args:
            file_contents: List of dictionaries containing file information
            
        Returns:
            str: Formatted context string containing file details and content
        """
        context_parts = []
        
        for file_info in file_contents:
            context_parts.append(f"Filename: {file_info['filename']}")
            context_parts.append(f"File size: {file_info['size']} bytes")
            context_parts.append(f"Content:\n{file_info['content']}")
            context_parts.append("")  # Empty line as separator
            
        return "\n".join(context_parts).strip()