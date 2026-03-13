# OnlineSearch.py
import logging
import re
import requests
import json
import os
import tiktoken

from typing import Dict, List, Optional, Tuple
from concurrent.futures import as_completed
from datetime import datetime
import _config
from .Agent import Agent
from tools.utils import contains_urls, extract_json, fetch_and_extract

logger = logging.getLogger(__name__)

# Config
MAX_RESULTS_PER_SEARCH = _config.max_search_results
MAX_CONTEXT_TOKENS = _config.max_context_tokens

# Tokenizer for accurate token counting (cl100k_base works for most modern models)
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# SERP API Configuration
SERP_API_URL = "https://google.serper.dev/search"
SERP_NEWS_API_URL = "https://google.serper.dev/news"
SERP_API_KEY = os.getenv("SERP_API_KEY")

# Search type configurations: (endpoint, tbs_filter, description)
SEARCH_TYPES_FULL = [
    (SERP_API_URL, None, "web_all_time"),
    (SERP_API_URL, "qdr:d", "web_past_day"),
    (SERP_NEWS_API_URL, None, "news"),
]

SEARCH_TYPES_QUICK = [
    (SERP_API_URL, "qdr:d", "web_past_day"),     # Web search - past day only
]

# Pre-compiled patterns for news request detection
_NEWS_PATTERNS = [
    re.compile(p) for p in [
        r"^(the\s+)?news[\s\?\!\.]*$",
        r"^what'?s?\s+(the\s+)?(latest\s+)?news[\s\?\!\.]*$",
        r"^(give|show|tell)\s+me\s+(the\s+)?(latest\s+)?news[\s\?\!\.]*$",
        r"^(news|daily)\s+summary[\s\?\!\.]*$",
        r"^(today'?s?\s+)?headlines[\s\?\!\.]*$",
        r"^nachrichten[\s\?\!\.]*$",
        r"^was\s+gibt'?s?\s+(es\s+)?neues[\s\?\!\.]*$",
        r"^schlagzeilen[\s\?\!\.]*$",
    ]
]


class OnlineSearchAgent(Agent):
    """Agent for performing online searches and retrieving information.

    This agent analyzes user input to determine if an online search is needed,
    formulates appropriate search queries, fetches search results from SERP API,
    and retrieves and processes the content of relevant web pages.

    Uses smart search depth: quick searches for simple queries (weather, time-sensitive),
    thorough searches for research topics requiring comprehensive coverage.
    """

    def __init__(self, name: str, user: Optional[str] = None) -> None:
        super().__init__(name, user)

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        """Should we search? Quick checks first, then LLM decision."""
        if re.search(r'!([a-z]+)\b', user_input) or contains_urls(user_input):
            return False
        if self._is_news_request(user_input):
            return False
        return self._needs_search(user_input, last_response)

    def process(self, user_input: str, last_response: Optional[str] = None) -> Optional[str]:
        """Do the search with smart depth selection."""
        self.metadata = {"urls": [], "queries": [], "search_depth": ""}

        # Generate search plan (depth + queries) in single LLM call
        search_depth, search_queries = self._generate_search_plan(user_input, last_response)
        self.metadata["search_depth"] = search_depth
        self.metadata["queries"] = search_queries

        # Perform all query searches in parallel
        all_search_results = []
        seen_urls = set()

        query_futures = [
            self.get_executor().submit(self.perform_searches, query, search_depth)
            for query in search_queries
        ]

        for future in as_completed(query_futures):
            try:
                results = future.result(timeout=3)
                for result in results:
                    url = result.get('href', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_search_results.append(result)
            except Exception as e:
                logger.error(f"Error in query search: {e}")

        logger.info(f"Total unique results from {len(search_queries)} queries ({search_depth}): {len(all_search_results)}")

        # Enrich results (fetch and summarize content)
        enriched_results = self.enrich_results(all_search_results)

        if not enriched_results:
            return None

        context = self.create_context(enriched_results)
        self.metadata["urls"] = [result['url'] for result in enriched_results]

        return f"\n{context}"

    def _is_news_request(self, user_input: str) -> bool:
        """Check if this is a general news/headlines request (handled by DailyStoriesAgent)."""
        text = user_input.lower().strip()
        return any(p.match(text) for p in _NEWS_PATTERNS)

    def _needs_search(self, user_input: str, last_response: Optional[str] = None) -> bool:
        """Determine if the user input requires an online search using LLM."""
        prompt = f"""Need internet search? Output JSON only.

        User: "{user_input}"
        Previous: {last_response[:300] + '...' if last_response and len(last_response) > 300 else last_response or 'None'}

        YES (search=true):
        - facts, current events, people/companies/products
        - software, games, apps, services, platforms (features, guides, how-to)
        - verify/confirm/check previous response
        - anything that could be outdated or version-specific

        NO (search=false):
        - greetings, chitchat
        - pure opinion/creative writing
        - basic math, code execution
        - reformat/rephrase previous response

        When in doubt: YES

        Format: {{"search":true}} or {{"search":false}}"""

        try:
            response = self.llm.generate_single_response(prompt, max_tokens=32).strip()
            data = extract_json(response)
            if isinstance(data, dict):
                result = data.get("search", True)
                logger.info(f"Search decision: '{user_input}' -> {result}")
                return result
            return True
        except Exception as e:
            logger.error(f"Error determining search need: {e}")
            return True

    def _generate_search_plan(self, user_input: str, last_response: Optional[str] = None) -> Tuple[str, List[str]]:
        """Generate search depth and queries in a single LLM call."""
        current_year = datetime.now().strftime("%Y")

        # Build context section if we have previous response
        context_section = ""
        if last_response:
            # Truncate if too long, keep first ~500 chars for topic extraction
            truncated = last_response[:500] + "..." if len(last_response) > 500 else last_response
            context_section = f"\n        Context (previous response): {truncated}"

        prompt = f"""Plan search. Output JSON only.

        User: {user_input}{context_section}

        Format: {{"depth":"quick|thorough","queries":["keywords1","keywords2"]}}
        - depth quick: simple lookup, weather, single fact
        - depth thorough: research, comparison, analysis
        - queries: keywords only, +required, "exact", -exclude, {current_year} if timely, max 3
        - CRITICAL: if user references previous context (this/that/it/verify/check), extract the ACTUAL TOPIC from context and search for THAT topic, not meta-queries about verification"""

        try:
            response = self.llm.generate_single_response(prompt, max_tokens=256).strip()
            data = extract_json(response)
            if not isinstance(data, dict):
                return "thorough", [user_input]
            depth = data.get("depth", "thorough")
            if depth not in ("quick", "thorough"):
                depth = "thorough"

            queries = data.get("queries", [])
            if not isinstance(queries, list):
                queries = [user_input]
            queries = [q for q in queries if isinstance(q, str) and len(q) > 3][:3]

            if not queries:
                queries = [user_input]

            logger.info(f"Search plan: depth={depth}, queries={queries}")
            return depth, queries

        except Exception as e:
            logger.error(f"Error generating search plan: {e}")
            return "thorough", [user_input]

    def perform_searches(self, query: str, depth: str = "thorough") -> List[Dict[str, str]]:
        """Perform searches using SERP API with configurable depth.

        Quick search: past day only (1 search type)
        Thorough search: all time, past day, news (3 search types)

        Args:
            query: Search query
            depth: 'quick' or 'thorough' (default: 'thorough')

        Returns:
            List[Dict[str, str]]: Combined search results
        """
        all_results = []

        # Select search types based on depth
        search_types = SEARCH_TYPES_QUICK if depth == "quick" else SEARCH_TYPES_FULL

        # Execute search types in parallel
        search_futures = []
        for endpoint, tbs_filter, search_type in search_types:
            future = self.get_executor().submit(
                self._execute_single_search, query, endpoint, tbs_filter, search_type
            )
            search_futures.append((future, search_type))

        # Collect results from all searches
        for future, search_type in search_futures:
            try:
                results = future.result(timeout=3)
                logger.info(f"Search type '{search_type}' returned {len(results)} results")
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error in search type '{search_type}': {e}")

        logger.info(f"Total results from {len(search_types)} search types: {len(all_results)}")
        return all_results

    def _execute_single_search(
        self, query: str, endpoint: str, tbs_filter: Optional[str], search_type: str
    ) -> List[Dict[str, str]]:
        """Execute a single search request to SERP API.

        Args:
            query: Search query
            endpoint: API endpoint URL
            tbs_filter: Time-based search filter (e.g., 'qdr:d' for past day)
            search_type: Description of the search type for logging

        Returns:
            List[Dict[str, str]]: Search results
        """
        try:
            payload = {
                "q": query,
                "num": min(MAX_RESULTS_PER_SEARCH, 20)
            }

            # Add time filter if specified
            if tbs_filter:
                payload["tbs"] = tbs_filter

            headers = {
                'X-API-KEY': SERP_API_KEY,
                'Content-Type': 'application/json'
            }

            response = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=3)
            response.raise_for_status()

            search_data = response.json()
            results = []

            # Parse results (news and organic share the same structure)
            raw_results = search_data.get('news') or search_data.get('organic') or []
            for result in raw_results[:MAX_RESULTS_PER_SEARCH]:
                results.append({
                    'title': result.get('title', ''),
                    'href': result.get('link', ''),
                    'body': result.get('snippet', ''),
                    'source': search_type
                })

            # Include knowledge graph if available (only for web searches)
            if 'knowledgeGraph' in search_data and search_type == "web_all_time":
                kg = search_data['knowledgeGraph']
                if 'title' in kg:
                    attributes_text = ""
                    if 'attributes' in kg:
                        attributes_text = " ".join([f"{k}: {v}" for k, v in kg['attributes'].items()])

                    results.insert(0, {
                        'title': kg['title'],
                        'href': '#knowledge_graph',
                        'body': attributes_text,
                        'source': 'knowledge_graph'
                    })

            return results

        except Exception as e:
            logger.error(f"Error performing {search_type} search: {e}")
            return []

    def enrich_results(self, search_results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Fetch content from search result URLs and enrich with trafilatura.

        Args:
            search_results: List of search result dictionaries

        Returns:
            List[Dict[str, str]]: Enriched results with extracted content
        """
        def process_result(result: Dict[str, str]) -> Optional[Dict[str, str]]:
            """Fetch and extract content from a single search result."""
            # Handle knowledge graph results (no URL to fetch)
            if result['href'] == '#knowledge_graph':
                return {
                    'title': result['title'],
                    'url': result['href'],
                    'snippet': result['body'],
                    'content': result['body'],
                }

            try:
                extracted = fetch_and_extract(result['href'], favor_precision=True)
                if not extracted:
                    return None

                content = extracted["text"]

                if _config.summarize_fetched_content:
                    try:
                        content = self.summarize_content(content, max_words=2000)
                    except Exception as e:
                        logger.error(f"Error summarizing content: {e}")

                return {
                    'title': extracted['title'] or result['title'],
                    'url': result['href'],
                    'snippet': result['body'],
                    'content': content,
                }

            except Exception as e:
                logger.error(f"Error processing {result['href']}: {e}")
                return None

        # Process results in parallel using shared executor
        enrichment_tasks = [self.get_executor().submit(process_result, result) for result in search_results]

        enriched_results = []
        for future in as_completed(enrichment_tasks):
            try:
                result = future.result(timeout=15)
                if result:
                    enriched_results.append(result)
            except Exception as e:
                logger.error(f"Error retrieving result from future: {e}")

        return enriched_results

    def create_context(self, enriched_results: List[Dict[str, str]]) -> str:
        """Create a formatted context from enriched search results.

        Respects MAX_CONTEXT_TOKENS limit - stops adding results when the next
        entry would exceed the token budget.

        Args:
            enriched_results: List of enriched result dictionaries

        Returns:
            str: Formatted context
        """
        context_parts = []
        total_tokens = 0
        results_included = 0

        for result in enriched_results:
            # Format this result entry
            entry = (
                f"Title: {result['title']}\n"
                f"URL: {result['url']}\n"
                f"Snippet: {result['snippet']}\n"
                f"Content: {result['content']}\n"
            )

            # Count tokens for this entry
            entry_tokens = len(TOKENIZER.encode(entry))

            # Check if adding this entry would exceed the limit
            if total_tokens + entry_tokens > MAX_CONTEXT_TOKENS:
                logger.info(f"Token limit reached: {total_tokens}/{MAX_CONTEXT_TOKENS}, stopping at {results_included} results")
                break

            context_parts.append(entry)
            total_tokens += entry_tokens
            results_included += 1

        # Store token usage in metadata
        self.metadata["tokens_used"] = total_tokens
        self.metadata["results_included"] = results_included
        self.metadata["results_total"] = len(enriched_results)

        logger.info(f"Context created: {results_included}/{len(enriched_results)} results, {total_tokens} tokens")

        return "\n".join(context_parts).strip()