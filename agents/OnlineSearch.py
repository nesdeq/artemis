"""OnlineSearch agent: SERP-API web search. The LLM decides whether to search
and plans every search parameter (queries, result count, country, language,
recency, news)."""
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

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

# Generic bang (!news / !games / !finance ...): handled by other agents, so a
# bang means "don't auto-search". _FORCE_BANG is the exception: an explicit
# request TO search, which overrides the conservative decision call.
_BANG_RE = re.compile(r'!([a-z]+)\b')
_FORCE_BANG_RE = re.compile(r'!(search|web)\b', re.IGNORECASE)

# Planner "recency" label -> SERP `tbs` value. "any" means no time filter.
RECENCY_TO_TBS = {
    "any": None,
    "hour": "qdr:h",
    "day": "qdr:d",
    "week": "qdr:w",
    "month": "qdr:m",
    "year": "qdr:y",
}


class OnlineSearchAgent(Agent):
    """Web search via SERP API. The decision call is conservative (default NO,
    conversation-aware); the execution call lets the LLM plan all search params."""

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        # Without a SERP key the agent cannot search, so disable it up front instead
        # of failing per-request deep inside requests (and burning an LLM
        # decision call to decide on a search it can't run).
        if not SERP_API_KEY:
            logger.warning("SERP_API_KEY not set, OnlineSearchAgent disabled")
            return False
        # Explicit force (!search / !web): the user asked to search; binary
        # pre-filter, skip the decision call (architecture law 5).
        if _FORCE_BANG_RE.search(user_input):
            return True
        # Cheap, deterministic exits: other bangs and pasted URLs belong to the
        # News / URLReader agents, never to an auto-search.
        if _BANG_RE.search(user_input) or contains_urls(user_input):
            return False
        return self._needs_search(user_input, last_response)

    def process(self, user_input: str, last_response: Optional[str] = None) -> Optional[str]:
        self.metadata = {"queries": [], "urls": []}

        # Drop the force token so it doesn't pollute the planned queries.
        query_input = _FORCE_BANG_RE.sub(" ", user_input).strip() or user_input

        plan = self._generate_search_plan(query_input, last_response)
        self.metadata["queries"] = plan["queries"]
        self.metadata["results_per_query"] = plan["num"]
        if plan["gl"]:
            self.metadata["country"] = plan["gl"]
        if plan["hl"]:
            self.metadata["language"] = plan["hl"]
        if plan["recency"] != "any":
            self.metadata["recency"] = plan["recency"]
        if plan["include_news"]:
            self.metadata["news"] = True

        # Run all queries in parallel; flatten results, deduping URLs.
        per_query = parallel_map(
            plan["queries"],
            lambda q: self._perform_searches(q, plan),
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

        logger.info(f"Total unique results from {len(plan['queries'])} queries: {len(all_results)}")

        enriched = parallel_map(
            all_results, self._enrich_one, self.get_executor(), _config.url_fetch_timeout
        )
        if not enriched:
            return None

        self.metadata["urls"] = [r['url'] for r in enriched]
        return self._create_context(enriched)

    # ------------------------------------------------------------------ Decision
    def _needs_search(self, user_input: str, last_response: Optional[str] = None) -> bool:
        """DECISION LLM call: does answering THIS turn need a fresh web search?

        Conservative and conversation-aware: defaults to NO, and to NO on any
        parse failure or error: searching is the exception, not the reflex.
        """
        context = f'"{last_response}"' if last_response else "None"
        prompt = f"""Decide whether answering the user's latest message needs a fresh web search.

Default to NO. A web search is the exception, not the reflex. Answer YES only when this turn genuinely needs facts that are external AND current, and that you could not provide from general knowledge or from the conversation so far.

YES when the turn turns on:
- current events or news, or anything time-bound ("today", "now", "latest", "this week")
- a specific real-world entity, product, library, API, version, price, score, or release whose CURRENT state matters
- an explicit request to search, look up, check online, or verify against the web

NO when the message is:
- a reaction, acknowledgement, or aside ("thanks", "lol", "interesting", "go on")
- an opinion, a joke, or banter
- a request to explain, rephrase, summarize, analyze, or continue YOUR OWN previous answer
- math, logic, coding, or reasoning you can do yourself
- creative writing or a hypothetical
- already answered by the previous assistant turn, with the user merely building on it

User message: "{user_input}"
Previous assistant turn: {context}

Output JSON only: {{"search": true}} or {{"search": false}}"""

        try:
            response = self.llm.generate_single_response(
                prompt, max_tokens=_config.search_decision_max_tokens
            ).strip()
            data = extract_json(response)
            if isinstance(data, dict):
                result = bool(data.get("search", False))
                logger.info(f"Search decision: '{user_input[:60]}' -> {result}")
                return result
            return False
        except Exception as e:
            logger.error(f"Error determining search need: {e}")
            return False

    # ------------------------------------------------------------------ Planning
    def _default_plan(self, user_input: str) -> Dict[str, Any]:
        return {
            "queries": [user_input],
            "num": _config.search_default_results,
            "gl": None,
            "hl": None,
            "recency": "any",
            "tbs": None,
            "include_news": False,
        }

    @staticmethod
    def _valid_code(value: Any) -> Optional[str]:
        """Two-letter ISO code (gl/hl), lowercased; None for anything else."""
        if isinstance(value, str):
            code = value.strip().lower()
            if len(code) == 2 and code.isalpha():
                return code
        return None

    def _generate_search_plan(self, user_input: str, last_response: Optional[str] = None) -> Dict[str, Any]:
        """EXECUTION LLM call: plan every search parameter, then validate it.

        The LLM picks queries, per-query result count, country, language,
        recency, and whether to include news. Every field is bounded to its
        documented range before use (same discipline as HueLights._validate_action).
        """
        today = datetime.now().strftime("%Y-%m-%d")
        max_q = _config.max_search_queries
        max_results = _config.search_max_results_per_query

        context_section = (
            f'\n        Context (previous response): "{last_response}"' if last_response else ""
        )

        prompt = f"""Plan web searches for the user's request. Output JSON only.

        User: {user_input}{context_section}
        Today: {today}.

        Return a JSON object:
        {{
          "queries": ["keywords", ...],
          "num": <integer 1-{max_results}>,
          "gl": "<2-letter country code>" or null,
          "hl": "<2-letter language code>" or null,
          "recency": "any" | "hour" | "day" | "week" | "month" | "year",
          "include_news": true or false
        }}

        Guidance:
        - queries: up to {max_q} focused keyword queries (keywords, not questions); use +required, "exact phrase", -exclude. If the user refers to earlier context (this/that/it/verify/check), pull the ACTUAL topic from context, not a meta-query about verifying.
        - num: results to fetch PER query. Few (1-3) for a single fact; more (up to {max_results}) for research or comparison.
        - gl / hl: set only when the request is locale- or language-specific; otherwise null.
        - recency: "hour"/"day"/"week" for breaking or fast-moving topics, "month"/"year" for recent-but-not-breaking, "any" for evergreen facts.
        - include_news: true only for current-events / breaking topics; false otherwise."""

        try:
            response = self.llm.generate_single_response(
                prompt, max_tokens=_config.search_plan_max_tokens
            ).strip()
            data = extract_json(response)
            if not isinstance(data, dict):
                return self._default_plan(user_input)

            raw_queries = data.get("queries")
            if not isinstance(raw_queries, list):
                raw_queries = []
            queries = [q.strip() for q in raw_queries
                       if isinstance(q, str) and len(q.strip()) > 3][:max_q]
            if not queries:
                queries = [user_input]

            num = data.get("num", _config.search_default_results)
            if not isinstance(num, int) or isinstance(num, bool):
                num = _config.search_default_results
            num = max(1, min(num, max_results))

            recency = data.get("recency", "any")
            if not isinstance(recency, str) or recency not in RECENCY_TO_TBS:
                recency = "any"

            plan = {
                "queries": queries,
                "num": num,
                "gl": self._valid_code(data.get("gl")),
                "hl": self._valid_code(data.get("hl")),
                "recency": recency,
                "tbs": RECENCY_TO_TBS[recency],
                "include_news": bool(data.get("include_news", False)),
            }
            logger.info(
                f"Search plan: queries={plan['queries']}, num={plan['num']}, "
                f"gl={plan['gl']}, hl={plan['hl']}, recency={plan['recency']}, "
                f"news={plan['include_news']}"
            )
            return plan

        except Exception as e:
            logger.error(f"Error generating search plan: {e}")
            return self._default_plan(user_input)

    # ------------------------------------------------------------------ Execution
    def _perform_searches(self, query: str, plan: Dict[str, Any]) -> List[Dict[str, str]]:
        """Run the web search (and optionally news) for a single query."""
        endpoints = [(SERP_API_URL, "web")]
        if plan["include_news"]:
            endpoints.append((SERP_NEWS_API_URL, "news"))
        results = parallel_map(
            endpoints,
            lambda ep: self._execute_single_search(query, ep[0], ep[1], plan),
            self.get_executor(),
            _config.search_request_timeout,
        )
        flat = [r for batch in results for r in batch]
        logger.info(f"Total results from {len(endpoints)} endpoint(s): {len(flat)}")
        return flat

    def _execute_single_search(
        self, query: str, endpoint: str, search_type: str, plan: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        try:
            payload: Dict[str, Any] = {
                "q": query,
                "num": min(plan["num"], _config.serp_results_cap),
            }
            if plan["gl"]:
                payload["gl"] = plan["gl"]
            if plan["hl"]:
                payload["hl"] = plan["hl"]
            if plan["tbs"]:
                payload["tbs"] = plan["tbs"]

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
                for r in raw[:plan["num"]]
            ]

            # Knowledge graph (web search only)
            kg = data.get('knowledgeGraph') if search_type == "web" else None
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

    # Trailing newline is part of each entry's rendered form, so
    # take_within_token_budget counts it when measuring against the budget.
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
