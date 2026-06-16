"""News agent: aggregate RSS stories via !news / !games / !finance bang commands."""
import logging
import time
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

import feedparser
import requests

import _config
from .Agent import Agent
from tools.utils import (
    DEFAULT_USER_AGENT, clean_html, format_blocks, format_record,
    parallel_map, take_within_token_budget,
)

REQUEST_HEADERS = {'User-Agent': DEFAULT_USER_AGENT}

logger = logging.getLogger(__name__)


class DailyStoriesAgent(Agent):
    """Fetch news stories from RSS feeds via !news / !games / !finance bang commands."""

    _BANG_TO_CONFIG_FEEDS = {
        'news': 'news_feeds',
        'games': 'gaming_feeds',
        'finance': 'finance_feeds',
    }

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        return bool(self._detect_news_types(user_input))

    def process(self, user_input: str, last_response: Optional[str] = None) -> Optional[str]:
        news_types = self._detect_news_types(user_input)
        if not news_types:
            return None

        feeds: List[str] = []
        for t in news_types:
            feeds.extend(getattr(_config, self._BANG_TO_CONFIG_FEEDS[t]))

        stories = self.fetch_stories_from_feeds(feeds)
        if not stories:
            return "Unable to retrieve stories at this time. Please try again later."

        self.metadata = {
            'stories_fetched': len(stories),
            'category': ', '.join(sorted(news_types)),
            'domains': [self._get_domain(f) for f in feeds],
            'timestamp': time.strftime('%Y-%m-%d', time.localtime()),
        }
        return self._create_context(stories)

    def _detect_news_types(self, user_input: str) -> Set[str]:
        lower = user_input.lower()
        return {t for t in self._BANG_TO_CONFIG_FEEDS if f'!{t}' in lower}

    def _get_domain(self, url: str) -> str:
        """Extract a clean domain name from a URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc

            if 'feedburner.com' in domain:
                path_parts = parsed.path.strip('/').split('/')
                if path_parts:
                    domain = path_parts[0]
            else:
                if domain.startswith('www.'):
                    domain = domain[4:]
                parts = domain.split('.')
                if len(parts) > 2:
                    domain = '.'.join(parts[-2:])
            return domain
        except Exception as e:
            logger.warning(f"Error extracting domain from {url}: {e}")
            return url.replace('https://', '').replace('http://', '').split('/')[0]

    def fetch_stories_from_feeds(self, feeds: List[str]) -> List[Dict[str, str]]:
        """Fetch stories from multiple feeds concurrently."""
        per_feed = parallel_map(
            feeds, self._fetch_feed, self.get_executor(), _config.feed_result_timeout
        )
        return [story for stories in per_feed for story in stories]

    def _fetch_feed(self, feed_url: str) -> List[Dict[str, str]]:
        """Fetch and parse a single RSS feed."""
        try:
            resp = requests.get(
                feed_url, headers=REQUEST_HEADERS, timeout=_config.feed_fetch_timeout
            )
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)

            if feed.bozo and feed.bozo_exception:
                logger.warning(f"Feed error for {feed_url}: {feed.bozo_exception}")

            domain = self._get_domain(feed_url)
            return [
                {
                    'title': entry.get('title', 'No title'),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', entry.get('updated', 'No date')),
                    'summary': clean_html(entry.get('summary', 'No summary available')),
                    'source_domain': domain,
                }
                for entry in feed.entries
            ]
        except Exception as e:
            logger.error(f"Error fetching feed {feed_url}: {e}")
            return []

    _CONTEXT_FIELDS = [
        ("Title", "title"),
        ("Source", "source_domain"),
        ("Link", "link"),
        ("Published", "published"),
        ("Summary", "summary"),
    ]

    def _create_context(self, stories: List[Dict[str, str]]) -> str:
        if not stories:
            return "No stories available at this time."
        # Bound the injected context the same way OnlineSearch / ReadURLs do — a
        # single bang can pull hundreds of entries across dozens of feeds, which
        # would otherwise dump unbounded text into the turn.
        kept, _ = take_within_token_budget(
            stories, lambda s: format_record(s, self._CONTEXT_FIELDS),
            _config.max_context_tokens,
        )
        self.metadata["stories_included"] = len(kept)
        domains = sorted({s['source_domain'] for s in kept})
        return format_blocks(kept, self._CONTEXT_FIELDS,
                             header="Sources: " + ", ".join(domains))
