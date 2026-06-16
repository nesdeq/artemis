"""OnlineSearch agent: SERP-API web search with depth selection and content extraction."""
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

import _config
from .Agent import Agent
from tools.utils import (
    contains_urls, extract_json, fetch_and_extract, format_record,
    maybe_summarize, parallel_map, take_within_token_budget,
)

logger = logging.getLogger(__name__)

# SERP API
SERP_API_URL = "https://google.serper.dev/search"
SERP_NEWS_API_URL = "https://google.serper.dev/news"
SERP_API_KEY = os.getenv("SERP_API_KEY")

# Search type configurations: (endpoint, tbs_filter, label)
SEARCH_TYPES_FULL = [
    (SERP_API_URL, None, "web_all_time"),
    (SERP_API_URL, "qdr:d", "web_past_day"),
    (SERP_NEWS_API_URL, None, "news"),
]
SEARCH_TYPES_QUICK = [
    (SERP_API_URL, "qdr:d", "web_past_day"),
]

_BANG_RE = re.compile(r'!([a-z]+)\b')


class OnlineSearchAgent(Agent):
    """Web search via SERP API with smart depth selection.

    - Quick: past-day web search only (1 endpoint).
    - Thorough: web all-time + web past-day + news (3 endpoints).
    """

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        # Without a SERP key the agent cannot search — disable it up front instead
        # of failing per-request deep inside requests (and burning an LLM
        # decision call to decide on a search it can't run).
        if not SERP_API_KEY:
            logger.warning("SERP_API_KEY not set — OnlineSearchAgent disabled")
            return False
        # Cheap, deterministic exits first; only then ask the LLM.
        if _BANG_RE.search(user_input) or contains_urls(user_input):
            return False
        return self._needs_search(user_input, last_response)

    def process(self, user_input: str, last_response: Optional[str] = None) -> Optional[str]:
        self.metadata = {"urls": [], "queries": [], "search_depth": ""}

        depth, queries = self._generate_search_plan(user_input, last_response)
        self.metadata["search_depth"] = depth
        self.metadata["queries"] = queries

        # Run all queries in parallel; flatten results, deduping URLs.
        per_query = parallel_map(
            queries,
            lambda q: self._perform_searches(q, depth),
            self.get_executor(),
            _config.search_query_timeout,
        )
        all_results: List[Dict[str, str]] = []
        seen: set = set()
        for results in per_query:
            for r in results:
                url = r.get('href', '')
                if url and url not in seen:
                    seen.add(url)
                    all_results.append(r)

        logger.info(f"Total unique results from {len(queries)} queries ({depth}): {len(all_results)}")

        enriched = parallel_map(
            all_results, self._enrich_one, self.get_executor(), _config.url_fetch_timeout
        )
        if not enriched:
            return None

        self.metadata["urls"] = [r['url'] for r in enriched]
        return self._create_context(enriched)

    def _needs_search(self, user_input: str, last_response: Optional[str] = None) -> bool:
        prompt = f"""Need internet search? Output JSON only.

        User: "{user_input}"
        Previous: {last_response or 'None'}

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
            response = self.llm.generate_single_response(
                prompt, max_tokens=_config.search_decision_max_tokens
            ).strip()
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
        current_year = datetime.now().strftime("%Y")
        max_q = _config.max_search_queries

        context_section = (
            f"\n        Context (previous response): {last_response}" if last_response else ""
        )

        prompt = f"""Plan search. Output JSON only.

        User: {user_input}{context_section}

        Format: {{"depth":"quick|thorough","queries":["keywords1","keywords2"]}}
        - depth quick: simple lookup, weather, single fact
        - depth thorough: research, comparison, analysis
        - queries: keywords only, +required, "exact", -exclude, {current_year} if timely, max {max_q}
        - CRITICAL: if user references previous context (this/that/it/verify/check), extract the ACTUAL TOPIC from context and search for THAT topic, not meta-queries about verification"""

        try:
            response = self.llm.generate_single_response(
                prompt, max_tokens=_config.search_plan_max_tokens
            ).strip()
            data = extract_json(response)
            if not isinstance(data, dict):
                return "thorough", [user_input]

            depth = data.get("depth", "thorough")
            if depth not in ("quick", "thorough"):
                depth = "thorough"

            queries = data.get("queries", [])
            if not isinstance(queries, list):
                queries = [user_input]
            queries = [q for q in queries if isinstance(q, str) and len(q) > 3][:max_q]
            if not queries:
                queries = [user_input]

            logger.info(f"Search plan: depth={depth}, queries={queries}")
            return depth, queries

        except Exception as e:
            logger.error(f"Error generating search plan: {e}")
            return "thorough", [user_input]

    def _perform_searches(self, query: str, depth: str = "thorough") -> List[Dict[str, str]]:
        """Run all SERP search types for a single query in parallel."""
        search_types = SEARCH_TYPES_QUICK if depth == "quick" else SEARCH_TYPES_FULL
        results = parallel_map(
            search_types,
            lambda st: self._execute_single_search(query, *st),
            self.get_executor(),
            _config.search_request_timeout,
        )
        flat = [r for batch in results for r in batch]
        logger.info(f"Total results from {len(search_types)} search types: {len(flat)}")
        return flat

    def _execute_single_search(
        self, query: str, endpoint: str, tbs_filter: Optional[str], search_type: str
    ) -> List[Dict[str, str]]:
        try:
            payload: Dict[str, Any] = {
                "q": query,
                "num": min(_config.max_search_results, _config.serp_results_cap),
            }
            if tbs_filter:
                payload["tbs"] = tbs_filter

            response = requests.post(
                endpoint,
                headers={'X-API-KEY': SERP_API_KEY, 'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=_config.search_request_timeout,
            )
            response.raise_for_status()
            data = response.json()

            raw = data.get('news') or data.get('organic') or []
            results = [
                {
                    'title': r.get('title', ''),
                    'href': r.get('link', ''),
                    'body': r.get('snippet', ''),
                    'source': search_type,
                }
                for r in raw[:_config.max_search_results]
            ]

            # Knowledge graph (only on web_all_time)
            kg = data.get('knowledgeGraph') if search_type == "web_all_time" else None
            if kg and 'title' in kg:
                attributes = " ".join(f"{k}: {v}" for k, v in kg.get('attributes', {}).items())
                results.insert(0, {
                    'title': kg['title'],
                    'href': '#knowledge_graph',
                    'body': attributes,
                    'source': 'knowledge_graph',
                })
            return results

        except Exception as e:
            logger.error(f"Error performing {search_type} search: {e}")
            return []

    def _enrich_one(self, result: Dict[str, str]) -> Optional[Dict[str, str]]:
        # Knowledge graph entries have no URL to fetch
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

            content = maybe_summarize(
                self.llm, extracted["text"], _config.search_content_summary_words
            )

            return {
                'title': extracted['title'] or result['title'],
                'url': result['href'],
                'snippet': result['body'],
                'content': content,
            }
        except Exception as e:
            logger.error(f"Error processing {result['href']}: {e}")
            return None

    # Field order matches the legacy output exactly (trailing newline preserved
    # so token accounting is unchanged).
    _CONTEXT_FIELDS = [("Title", "title"), ("URL", "url"),
                       ("Snippet", "snippet"), ("Content", "content")]

    def _render_entry(self, r: Dict[str, str]) -> str:
        return format_record(r, self._CONTEXT_FIELDS) + "\n"

    def _create_context(self, results: List[Dict[str, str]]) -> str:
        """Format results, stopping when token budget is hit."""
        kept, total_tokens = take_within_token_budget(
            results, self._render_entry, _config.max_context_tokens
        )

        self.metadata["tokens_used"] = total_tokens
        self.metadata["results_included"] = len(kept)
        self.metadata["results_total"] = len(results)

        logger.info(f"Context created: {len(kept)}/{len(results)} results, {total_tokens} tokens")
        return "\n".join(self._render_entry(r) for r in kept).strip()
