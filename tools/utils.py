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
import socket
import ipaddress
import logging
from concurrent.futures import as_completed, TimeoutError as FuturesTimeoutError
from typing import Optional, Any, Dict, List, Callable, Sequence, Tuple, TypeVar, Union
from urllib.parse import urlparse


from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
from markitdown import MarkItDown

import _config

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


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
    cfg.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(_config.trafilatura_download_timeout))
    cfg.set("DEFAULT", "MIN_EXTRACTED_SIZE", str(_config.trafilatura_min_extracted_size))
    cfg.set("DEFAULT", "MIN_OUTPUT_SIZE", str(_config.trafilatura_min_output_size))
    return cfg

_trafilatura_config = None

def get_trafilatura_config():
    """Get the shared trafilatura config (lazy-initialized)."""
    global _trafilatura_config
    if _trafilatura_config is None:
        _trafilatura_config = _build_trafilatura_config()
    return _trafilatura_config


def _is_safe_public_url(url: str) -> bool:
    """True only for an http(s) URL whose host resolves entirely to public IPs.

    URLs reaching fetch_and_extract come from web-search result links and
    user-pasted URLs, not just trusted config. Block non-http(s) schemes and any
    host that resolves to a loopback / private / link-local / reserved address
    (e.g. 127.0.0.1, 10.0.0.0/8, 169.254.169.254 cloud metadata) so a hostile
    link can't turn the fetcher into an SSRF probe of the local network.

    Note: this resolves the host and checks every returned address, but the
    actual fetch resolves again, so a determined DNS-rebinding attacker could
    still race it. It raises the bar; it is not a hard sandbox.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except Exception:
        return False
    if not infos:
        return False
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (addr.is_loopback or addr.is_private or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            logger.warning(f"Blocked fetch to non-public address {addr} for {url}")
            return False
    return True


def fetch_and_extract(url: str, favor_precision: bool = False,
                      include_links: bool = False) -> Optional[dict]:
    """
    Fetch a URL and extract its main content using trafilatura,
    falling back to Jina Reader API for JS-rendered pages.

    Returns a dict with keys: title, text, author, date, description, hostname
    or None if extraction fails or the URL is not a safe public http(s) target.
    """
    if not _is_safe_public_url(url):
        logger.warning(f"Refusing to fetch unsafe or non-public URL: {url}")
        return None

    result = _fetch_with_trafilatura(url, favor_precision, include_links)
    if result:
        return result

    return _fetch_with_jina(url)


def _fetch_with_trafilatura(url: str, favor_precision: bool = False,
                            include_links: bool = False) -> Optional[dict]:
    """Extract content using trafilatura (fast, no JS support)."""
    import trafilatura

    config = get_trafilatura_config()

    html = trafilatura.fetch_url(url, config=config)
    if not html:
        return None

    doc = trafilatura.bare_extraction(
        html,
        url=url,
        with_metadata=True,  # populate title/author/date; defaults to False in trafilatura 2.x
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


def _fetch_with_jina(url: str) -> Optional[dict]:
    """Fallback: extract content via Jina Reader API (handles JS-rendered pages)."""
    import requests

    try:
        resp = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT},
            timeout=_config.url_fetch_timeout,
        )
        if resp.status_code != 200:
            return None

        data = resp.json().get("data", {})
        text = data.get("content", "")
        if not text:
            return None

        return {
            "title": data.get("title", ""),
            "text": text,
            "author": data.get("author", ""),
            "date": data.get("publishedTime", ""),
            "description": data.get("description", ""),
            "hostname": urlparse(url).hostname or "",
        }
    except Exception as e:
        logger.warning(f"Jina fetch failed for {url}: {e}")
        return None


def clean_html(html_content: str) -> str:
    """Convert HTML to clean text via MarkItDown, falling back to regex strip."""
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


def maybe_summarize(llm: Any, text: str, max_words: int) -> str:
    """Summarize fetched web content when summarize_fetched_content is enabled.

    Shared by OnlineSearch and ReadURLs. On summarization failure the original
    text is kept — a failed summary must never drop the content.
    """
    if not _config.summarize_fetched_content:
        return text
    try:
        return llm.summarize(text, max_words=max_words)
    except Exception as e:
        logger.error(f"Error summarizing content: {e}")
        return text


# Compiled URL pattern for performance (used by both extract_urls and contains_urls)
_URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def extract_urls(text: str) -> List[str]:
    """Extract all URLs from text."""
    return _URL_PATTERN.findall(text)


def contains_urls(text: str) -> bool:
    """Check if text contains any URLs."""
    return bool(_URL_PATTERN.search(text))


def derive_encryption_key(username: str) -> Optional[bytes]:
    """Derive Fernet key from ENCKEY env var + username salt. Returns None if ENCKEY not set."""
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
    """Encrypt dict to Fernet bytes."""
    json_data = json.dumps(data)
    return Fernet(key).encrypt(json_data.encode())


def decrypt_data(encrypted: bytes, key: bytes) -> dict:
    """Decrypt Fernet bytes back to dict."""
    return json.loads(Fernet(key).decrypt(encrypted).decode())


def parallel_map(items: List[T], fn: Callable[[T], Optional[R]], executor,
                 timeout: float) -> List[R]:
    """Run `fn` over `items` on `executor`, drop None results.

    `timeout` is the wall-clock budget for the whole batch. Any future that
    doesn't finish in time is cancelled and its slot is dropped.
    """
    if not items:
        return []
    futures = [executor.submit(fn, item) for item in items]
    results: List[R] = []
    try:
        for future in as_completed(futures, timeout=timeout):
            try:
                r = future.result()
                if r is not None:
                    results.append(r)
            except Exception as e:
                logger.error(f"parallel_map task failed: {e}")
    except FuturesTimeoutError:
        unfinished = sum(1 for f in futures if not f.done())
        logger.warning(
            f"parallel_map: {timeout}s timeout reached, "
            f"{unfinished}/{len(futures)} unfinished, returning partial"
        )
        for f in futures:
            if not f.done():
                f.cancel()
    return results


# A field spec is (label, key) → "Label: value", or (label, key, mode) where
# mode ∈ {"block" (value on its own line), "optional" (skip if falsy),
# "optional_block" (both)}.
FieldSpec = Union[Tuple[str, str], Tuple[str, str, str]]


def format_record(record: Dict[str, Any], fields: Sequence[FieldSpec]) -> str:
    """Render one record as labelled lines. See FieldSpec for field modes."""
    lines: List[str] = []
    for spec in fields:
        label, key = spec[0], spec[1]
        mode = spec[2] if len(spec) > 2 else ""
        value = record.get(key, "")
        if mode in ("optional", "optional_block") and not value:
            continue
        block = mode in ("block", "optional_block")
        lines.append(f"{label}:\n{value}" if block else f"{label}: {value}")
    return "\n".join(lines)


def format_blocks(records: Sequence[Dict[str, Any]], fields: Sequence[FieldSpec],
                  header: Optional[str] = None) -> str:
    """Render records as blank-line-separated labelled blocks, with optional header."""
    blocks = [b for b in (format_record(r, fields) for r in records) if b]
    body = "\n\n".join(blocks)
    if header:
        body = f"{header}\n\n{body}" if body else header
    return body.strip()


_token_encoder = None


def _cl100k_encoder():
    """Lazily build and cache the shared cl100k tokenizer."""
    global _token_encoder
    if _token_encoder is None:
        import tiktoken
        _token_encoder = tiktoken.get_encoding("cl100k_base")
    return _token_encoder


def take_within_token_budget(items: Sequence[T], render: Callable[[T], str],
                             max_tokens: int) -> Tuple[List[T], int]:
    """Return (prefix_of_items_that_fit, total_tokens) under max_tokens.

    Stops at the first item that would overflow the budget — shared by the
    web-search and URL-reader context builders.
    """
    enc = _cl100k_encoder()
    kept: List[T] = []
    total = 0
    for item in items:
        n = len(enc.encode(render(item)))
        if total + n > max_tokens:
            break
        kept.append(item)
        total += n
    return kept, total


def _iter_balanced_spans(text: str, open_ch: str, close_ch: str):
    """Yield (start_index, span) for every balanced open_ch..close_ch span.

    Nesting- and string-literal-aware, so delimiters inside JSON strings don't
    affect balance. Scanning from every opener (not just the first) lets
    extract_json skip a decoy bracket in prose and still recover valid JSON
    further along.
    """
    n = len(text)
    search = 0
    while True:
        start = text.find(open_ch, search)
        if start == -1:
            return
        depth = 0
        in_string = False
        escaped = False
        end = None
        for i in range(start, n):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == '\\':
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is not None:
            yield start, text[start:end + 1]
            search = end + 1   # continue past the consumed span
        else:
            search = start + 1  # unbalanced from here; try the next opener


def extract_json(text: str) -> Optional[Any]:
    """Extract and parse JSON from text.

    Tries a direct parse, then recovers an embedded array or object from
    surrounding prose / markdown fences. EVERY balanced array- and object-span
    is considered (string-literal- and nesting-aware) and tried in order of
    position, so a decoy bracket in leading prose can't shadow valid JSON that
    follows, and an object that merely contains an array isn't mis-extracted as
    that inner array (the enclosing object opens at a lower index, so it wins).
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidates = (list(_iter_balanced_spans(text, '[', ']'))
                  + list(_iter_balanced_spans(text, '{', '}')))
    for _start, span in sorted(candidates, key=lambda s: s[0]):
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            continue
    return None
