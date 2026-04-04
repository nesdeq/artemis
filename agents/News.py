# News.py
import logging
import time
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse
from concurrent.futures import as_completed

import feedparser
import requests

import _config
from .Agent import Agent
from tools.utils import DEFAULT_USER_AGENT

# Some feeds block default user agents
REQUEST_HEADERS = {'User-Agent': DEFAULT_USER_AGENT}

# Configure logger
logger = logging.getLogger(__name__)


class DailyStoriesAgent(Agent):
    """Agent for fetching and providing news stories from various sources.

    This agent retrieves news stories from multiple RSS feeds based on
    the user's request for general news, gaming news, or finance news.
    Supports !news, !games, and !finance bang commands.
    """
    
    def __init__(self, name: str, user: Optional[str] = None) -> None:
        """Initialize the DailyStoriesAgent.

        Args:
            name: The name of the agent
            user: Optional user identifier
        """
        super().__init__(name, user)

        # Load feeds from config
        self.news_feeds = _config.news_feeds
        self.gaming_feeds = _config.gaming_feeds
        self.finance_feeds = _config.finance_feeds

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        """Should we fetch news?"""
        return len(self._detect_news_types(user_input)) > 0

    def process(self, user_input: str, last_response: Optional[str] = None) -> Optional[str]:
        """Fetch news stories."""
        news_types = self._detect_news_types(user_input)
        if not news_types:
            return None

        # Combine feeds from all requested types
        feeds = []
        for news_type in news_types:
            if news_type == 'games':
                feeds.extend(self.gaming_feeds)
            elif news_type == 'finance':
                feeds.extend(self.finance_feeds)
            elif news_type == 'news':
                feeds.extend(self.news_feeds)

        stories = self.fetch_stories_from_feeds(feeds)

        if not stories:
            return "Unable to retrieve stories at this time. Please try again later."

        context = self.create_context(stories)
        self.metadata = {
            'stories_fetched': len(stories),
            'category': ', '.join(sorted(news_types)),
            'domains': self.get_feed_domains(feeds),
            'timestamp': time.strftime('%Y-%m-%d', time.localtime())
        }
        return context

    def _detect_news_types(self, user_input: str) -> Set[str]:
        """Detect news types requested via bang commands only.

        Returns set of 'news', 'games', 'finance' based on !news, !games, !finance.
        """
        lower_input = user_input.lower()
        types: Set[str] = set()

        if '!news' in lower_input:
            types.add('news')
        if '!games' in lower_input:
            types.add('games')
        if '!finance' in lower_input:
            types.add('finance')

        return types

    def get_domain(self, url: str) -> str:
        """Extract a clean domain name from a URL.
        
        Args:
            url: The URL to extract domain from
            
        Returns:
            str: The cleaned domain name showing only the main domain parts
        """
        try:
            # Parse the URL
            parsed = urlparse(url)
            # Extract the domain
            domain = parsed.netloc
            
            # Handle special cases like feedburner
            if 'feedburner.com' in domain:
                # Try to extract the original source from the path
                path_parts = parsed.path.strip('/').split('/')
                if path_parts:
                    domain = path_parts[0]
            else:
                # Remove www. prefix if present
                if domain.startswith('www.'):
                    domain = domain[4:]
                
                # Get only the last two parts of the domain (e.g., nytimes.com from rss.nytimes.com)
                domain_parts = domain.split('.')
                if len(domain_parts) > 2:
                    domain = '.'.join(domain_parts[-2:])
            
            return domain
        except Exception as e:
            logger.warning(f"Error extracting domain from {url}: {e}")
            # Return a fallback
            return url.replace('https://', '').replace('http://', '').split('/')[0]

    def get_feed_domains(self, feeds: List[str]) -> List[str]:
        """Get list of domain names from feed URLs.
        
        Args:
            feeds: List of feed URLs
            
        Returns:
            List[str]: List of domain names
        """
        return [self.get_domain(feed) for feed in feeds]

    def fetch_stories_from_feeds(self, feeds: List[str]) -> List[Dict[str, str]]:
        """Fetch stories from multiple feeds concurrently.

        Args:
            feeds: List of RSS feed URLs

        Returns:
            List[Dict[str, str]]: List of story dictionaries
        """
        # Submit feed fetching tasks to shared thread pool
        futures = [self.get_executor().submit(self.fetch_feed, feed_url) for feed_url in feeds]
        
        # Collect results as they complete
        stories = []
        for future in as_completed(futures):
            try:
                feed_stories = future.result(timeout=20)
                if feed_stories:
                    stories.extend(feed_stories)
            except Exception as e:
                logger.error(f"Error fetching feed: {e}")
        
        return stories

    def fetch_feed(self, feed_url: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Fetch stories from a single RSS feed.

        Args:
            feed_url: URL of the RSS feed
            limit: Maximum number of stories to fetch per feed

        Returns:
            List[Dict[str, str]]: List of story dictionaries
        """
        try:
            # Use requests with proper headers to avoid blocks
            resp = requests.get(feed_url, headers=REQUEST_HEADERS, timeout=15)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)

            if feed.bozo and feed.bozo_exception:
                logger.warning(f"Feed error for {feed_url}: {feed.bozo_exception}")
                
            feed_stories = []
            entries = feed.entries if limit is None else feed.entries[:limit]
            
            # Get domain for the feed
            domain = self.get_domain(feed_url)
            
            for entry in entries:
                story = {
                    'title': entry.get('title', 'No title'),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', entry.get('updated', 'No date')),
                    'summary': entry.get('summary', 'No summary available'),
                    'source_domain': domain
                }

                # Clean up HTML if present in summary
                story['summary'] = self.clean_html(story['summary'])
                
                feed_stories.append(story)
                
            return feed_stories
            
        except Exception as e:
            logger.error(f"Error fetching feed {feed_url}: {e}")
            
        return []

    def create_context(self, stories: List[Dict[str, str]]) -> str:
        """Create a formatted context from the stories.
        
        Args:
            stories: List of story dictionaries
            
        Returns:
            str: Formatted stories
        """
        if not stories:
            return "No stories available at this time."
            
        # Start with a list of source domains
        domains = sorted(set(story['source_domain'] for story in stories))
        context_parts = ["Sources: " + ", ".join(domains), ""]
            
        # Include all stories, not just 25
        for story in stories:
            context_parts.append(f"Title: {story['title']}")
            context_parts.append(f"Source: {story['source_domain']}")
            context_parts.append(f"Link: {story['link']}")
            context_parts.append(f"Published: {story['published']}")
            context_parts.append(f"Summary: {story['summary']}")
            context_parts.append("")  # Empty line as separator
            
        return "\n".join(context_parts).strip()