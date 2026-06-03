# ReadURLs.py
import logging
from typing import Dict, List, Optional

import _config
from .Agent import Agent
from tools.utils import (
    extract_urls, fetch_and_extract, format_blocks, format_record,
    parallel_map, take_within_token_budget,
)

logger = logging.getLogger(__name__)


class URLReaderAgent(Agent):
    """Fetch and extract content from URLs found in user input."""

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        return bool(extract_urls(user_input))

    def process(self, user_input: str, last_response: Optional[str] = None) -> Optional[str]:
        self.metadata = {"urls": []}

        urls = extract_urls(user_input)
        if not urls:
            return None

        enriched = parallel_map(
            urls, self._fetch_url, self.get_executor(), _config.url_fetch_timeout
        )
        if not enriched:
            return None

        self.metadata["urls"] = [r['url'] for r in enriched]
        return self._create_context(enriched)

    def _fetch_url(self, url: str) -> Optional[Dict[str, str]]:
        try:
            extracted = fetch_and_extract(url, include_links=True)
            if not extracted:
                logger.warning(f"No content extracted from {url}")
                return None

            content = extracted["text"]
            if _config.summarize_fetched_content:
                try:
                    content = self.llm.summarize(
                        content, max_words=_config.url_content_summary_words
                    )
                except Exception as e:
                    logger.error(f"Error summarizing content: {e}")

            return {
                "url": url,
                "domain": extracted["hostname"],
                "title": extracted["title"],
                "author": extracted["author"],
                "date": extracted["date"],
                "content": content,
            }
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return None

    _CONTEXT_FIELDS = [
        ("URL", "url"),
        ("Domain", "domain"),
        ("Title", "title"),
        ("Author", "author", "optional"),
        ("Date", "date", "optional"),
        ("Content", "content"),
    ]

    def _create_context(self, results: List[Dict[str, str]]) -> str:
        # Bound the injected context the same way OnlineSearch does — pasting
        # several large pages must not blow past the model's context window.
        kept, _ = take_within_token_budget(
            results,
            lambda r: format_record(r, self._CONTEXT_FIELDS),
            _config.max_context_tokens,
        )
        return format_blocks(kept, self._CONTEXT_FIELDS)
