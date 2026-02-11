"""
Shared utility functions for Artemis AI.
Centralizes common operations to eliminate code duplication.
"""
import re
import json
import hashlib
import base64
import io
import os
from typing import Optional, Any, List

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
from markitdown import MarkItDown


# =============================================================================
# SHARED UI CONSTANTS
# =============================================================================

# Theme colors for all interfaces (CLI, Audio, etc.)
UI_THEME = {
    "primary": "deep_sky_blue1",
    "secondary": "purple3",
    "accent": "spring_green2",
    "warning": "gold1",
    "error": "red1",
    "background": "grey11",
    "text": "white",
    "dim": "grey50",
}

# ASCII art logo
LOGO = """
  █████╗ ██████╗ ████████╗███████╗███╗   ███╗██╗███████╗
 ██╔══██╗██╔══██╗╚══██╔══╝██╔════╝████╗ ████║██║██╔════╝
 ███████║██████╔╝   ██║   █████╗  ██╔████╔██║██║███████╗
 ██╔══██║██╔══██╗   ██║   ██╔══╝  ██║╚██╔╝██║██║╚════██║
 ██║  ██║██║  ██║   ██║   ███████╗██║ ╚═╝ ██║██║███████║
 ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝     ╚═╝╚═╝╚══════╝
        """

# Shared MarkItDown instance (thread-safe, stateless)
shared_markitdown = MarkItDown(enable_plugins=False)


# Default User-Agent for HTTP requests (consistent across agents)
# Windows Chrome is most accepted - update version periodically (check chromereleases.googleblog.com)
DEFAULT_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'


# =============================================================================
# TRAFILATURA CONFIG (shared across agents)
# =============================================================================

def _build_trafilatura_config():
    """Build a shared trafilatura config with custom user agent and timeouts."""
    from copy import deepcopy
    from trafilatura.settings import DEFAULT_CONFIG
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg.set("DEFAULT", "USER_AGENTS", DEFAULT_USER_AGENT)
    cfg.set("DEFAULT", "DOWNLOAD_TIMEOUT", "10")
    cfg.set("DEFAULT", "MIN_EXTRACTED_SIZE", "100")
    cfg.set("DEFAULT", "MIN_OUTPUT_SIZE", "50")
    return cfg

_trafilatura_config = None

def get_trafilatura_config():
    """Get the shared trafilatura config (lazy-initialized)."""
    global _trafilatura_config
    if _trafilatura_config is None:
        _trafilatura_config = _build_trafilatura_config()
    return _trafilatura_config


def fetch_and_extract(url: str, favor_precision: bool = False,
                      include_links: bool = False) -> Optional[dict]:
    """
    Fetch a URL and extract its main content using trafilatura.

    Returns a dict with keys: title, text, author, date, description, hostname
    or None if extraction fails.
    """
    import trafilatura

    config = get_trafilatura_config()

    # Fetch HTML (trafilatura handles encoding, retries, size limits)
    html = trafilatura.fetch_url(url, config=config)
    if not html:
        return None

    # Extract structured content
    doc = trafilatura.bare_extraction(
        html,
        url=url,
        include_tables=True,
        include_links=include_links,
        include_comments=False,
        include_images=False,
        favor_precision=favor_precision,
        deduplicate=True,
        config=config,
    )

    if not doc or not doc.text:
        return None

    return {
        "title": doc.title or "",
        "text": doc.text,
        "author": doc.author or "",
        "date": doc.date or "",
        "description": doc.description or "",
        "hostname": doc.hostname or "",
    }


def clean_html(html_content: str) -> str:
    """
    Convert HTML to clean markdown text using MarkItDown.

    Args:
        html_content: HTML content to convert

    Returns:
        Clean markdown text
    """
    if not html_content or not html_content.strip():
        return ""

    try:
        result = shared_markitdown.convert_stream(
            io.BytesIO(html_content.encode('utf-8')),
            file_extension='.html'
        )
        return result.text_content.strip() if result.text_content else ""
    except Exception:
        # Fallback to basic regex stripping if markitdown fails
        clean = re.sub(r'<[^>]+>', '', html_content)
        return re.sub(r'\s+', ' ', clean).strip()


# Compiled URL pattern for performance (used by both extract_urls and contains_urls)
_URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.

    Args:
        text: Text to extract URLs from

    Returns:
        List of extracted URLs
    """
    return _URL_PATTERN.findall(text)


def contains_urls(text: str) -> bool:
    """
    Check if text contains URLs.

    Args:
        text: Text to check

    Returns:
        True if URLs found, False otherwise
    """
    return bool(_URL_PATTERN.search(text))


def derive_encryption_key(username: str) -> Optional[bytes]:
    """
    Derive encryption key using ENCKEY environment variable and username.

    Used by PersonalInfoAgent and readpinf tool for encrypted storage.

    Args:
        username: Username to derive salt from

    Returns:
        Base64-encoded derived key or None if ENCKEY not set
    """
    enckey = os.environ.get("ENCKEY")
    if not enckey:
        return None
    try:
        salt = hashlib.sha256(f"artemis_salt_{username}".encode()).digest()[:16]
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(enckey.encode()))
    except Exception:
        return None


def encrypt_data(data: dict, key: bytes) -> bytes:
    """
    Encrypt dictionary data using Fernet.

    Args:
        data: Dictionary to encrypt
        key: Encryption key from derive_encryption_key()

    Returns:
        Encrypted bytes
    """
    json_data = json.dumps(data)
    return Fernet(key).encrypt(json_data.encode())


def decrypt_data(encrypted: bytes, key: bytes) -> dict:
    """
    Decrypt Fernet-encrypted data back to dictionary.

    Args:
        encrypted: Encrypted bytes
        key: Encryption key from derive_encryption_key()

    Returns:
        Decrypted dictionary
    """
    return json.loads(Fernet(key).decrypt(encrypted).decode())


def extract_json(text: str) -> Optional[Any]:
    """
    Extract and parse JSON from text that may contain other content.

    Tries direct parsing first, then extracts JSON array or object via regex.

    Args:
        text: Text potentially containing JSON

    Returns:
        Parsed JSON data or None if parsing fails
    """
    # Try direct JSON parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array
    json_match = re.search(r"\[.*?\]", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try to find a JSON object
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None
