# ReadURLs.py
import logging
from typing import Dict, List, Optional
from concurrent.futures import as_completed

import _config
from .Agent import Agent
from tools.utils import extract_urls, fetch_and_extract

# Configure logger
logger = logging.getLogger(__name__)


class URLReaderAgent(Agent):
    """Agent for reading and processing content from URLs using trafilatura."""

    def __init__(self, name: str, user: Optional[str] = None) -> None:
        super().__init__(name, user)

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        """Should we read URLs?"""
        return bool(extract_urls(user_input))

    def process(self, user_input: str, last_response: Optional[str] = None) -> Optional[str]:
        """Fetch and process content from URLs."""
        self.metadata = {"urls": []}

        urls = extract_urls(user_input)
        if not urls:
            return None

        enriched_results = self.enrich_urls(urls)
        if not enriched_results:
            return None

        context = self.create_context(enriched_results)
        self.metadata["urls"] = [r['url'] for r in enriched_results]
        return context

    def enrich_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        """Fetch content from URLs using trafilatura.

        Args:
            urls: List of URLs to process

        Returns:
            List[Dict[str, str]]: List of enriched URL data
        """
        enrichment_tasks = [self.get_executor().submit(self._fetch_url, url) for url in urls]

        enriched_results = []
        for future in as_completed(enrichment_tasks):
            try:
                result = future.result(timeout=15)
                if result:
                    enriched_results.append(result)
            except Exception as e:
                logger.error(f"Error retrieving result from future: {e}")

        return enriched_results

    def _fetch_url(self, url: str) -> Optional[Dict[str, str]]:
        """Fetch and extract content from a single URL.

        Args:
            url: URL to fetch

        Returns:
            Extracted content dict or None on failure
        """
        try:
            extracted = fetch_and_extract(url, include_links=True)
            if not extracted:
                logger.warning(f"No content extracted from {url}")
                return None

            content = extracted["text"]

            if _config.summarize_fetched_content:
                try:
                    content = self.summarize_content(content, max_words=200)
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

    def create_context(self, enriched_results: List[Dict[str, str]]) -> str:
        """Create a formatted context from enriched URL results."""
        if not enriched_results:
            return ""

        context_parts = []

        for result in enriched_results:
            context_parts.append(f"URL: {result['url']}")
            context_parts.append(f"Domain: {result['domain']}")
            context_parts.append(f"Title: {result['title']}")
            if result.get('author'):
                context_parts.append(f"Author: {result['author']}")
            if result.get('date'):
                context_parts.append(f"Date: {result['date']}")
            context_parts.append(f"Content: {result['content']}")
            context_parts.append("")

        return "\n".join(context_parts).strip()