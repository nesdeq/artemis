"""
Artemis AI Configuration
Centralized configuration for all system components.
"""

import os

# ============================================================================
# LLM MODELS
# ============================================================================

# Main LLM for conversation
llm = "openai/gpt-5.1"  # Latest with adaptive reasoning (minimal to xhigh)
#llm = "gemini/gemini-3.5-flash"  # Reasoning model; reasoning_effort -> thinkingLevel (GA 2026-05)
#llm = "openai/gpt-5"  # $1.25/$10 per MTok - reasoning (minimal/low/medium/high)
#llm = "openai/gpt-4.1"  # $2/$8 per MTok - 1M context window
#llm = "openai/o4-mini"  # $1.10/$4.40 per MTok - fast reasoning
#llm = "openai/o3"  # $2/$8 per MTok - advanced reasoning
#llm = "openai/gpt-4o"  # $2.50/$10 per MTok - legacy multimodal
# Claude 4.5 (current generation)
#llm = "anthropic/claude-opus-4-5-20251101"  # $5/$25 per MTok - best coding
#llm = "anthropic/claude-sonnet-4-5-20250929"  # $3/$15 per MTok - fast coding
#llm = "anthropic/claude-haiku-4-5-20251001"  # $1/$5 per MTok - fastest
# Claude 4 (previous generation)
#llm = "anthropic/claude-sonnet-4-20250514"
#llm = "anthropic/claude-opus-4-1-20250805"  # $15/$75 per MTok
# Legacy/Other
#llm = "anthropic/claude-3-7-sonnet-20250219"
#llm = "gemini/gemini-2.5-pro-exp-03-25"
#llm = "gemini/gemini-2.0-flash"
#llm = "ollama/gemma2:9b"
#llm = "ollama/llama3.1"
#llm = "ollama/phi4"

# LLM for agents (usually faster/cheaper model)
agent_llm = "openai/gpt-5-mini"  # $0.25/$2.00 per MTok - more capable
#agent_llm = "gemini/gemini-3.5-flash"  # Reasoning model
#agent_llm = "openai/gpt-5-nano"  # $0.05/$0.40 per MTok - best value for classification
#agent_llm = "openai/gpt-4.1-mini"  # 1M context
#agent_llm = "openai/gpt-4o-mini"  # Legacy but stable
#agent_llm = "anthropic/claude-haiku-4-5-20251001"  # $1/$5 per MTok - fast
#agent_llm = "gemini/gemini-2.0-flash"

# Reasoning effort for main LLM:
#   O-series (o1/o3/o4): low, medium, high
#   GPT-5 series: minimal, low, medium, high
#   GPT-5.1 series: minimal, low, medium, high, xhigh
#   Gemini 3.x Flash: minimal, low, medium, high (reasoning_effort -> thinkingLevel)
ro = "high"  # Uses model-specific defaults if None (minimal/low depending on model)

# Reasoning effort for agent LLM. Higher = more reliable classification/extraction
# at the cost of latency; lower it for faster/cheaper agents. Valid floors per family:
#   O-series: "low" (minimal not supported)
#   GPT-5/5.1: "minimal"
#   Gemini 3.x Flash: "minimal"
# litellm maps reasoning_effort to each provider's native control (drop_params on).
agent_ro = "medium"


# ============================================================================
# LLM API SETTINGS
# ============================================================================

# Request timeout in seconds
llm_timeout = 30

# Number of retry attempts for failed API calls
llm_retry_attempts = 3

# Maximum tokens for responses. On reasoning models (gpt-5.x), invisible
# reasoning tokens count against this budget, so keep it generous or complex
# answers get truncated. _adjust_tokens_for_reasoning only raises sub-floor values.
max_tokens = 8192


# ============================================================================
# STREAMING & PERFORMANCE
# ============================================================================

# Enable streaming responses from LLM API
streaming = True

# Output delay - controls streaming speed across ALL interfaces (seconds)
# Lower = faster output, higher = smoother/more readable
# Recommended: 0.001 (fast), 0.005 (balanced), 0.01 (smooth)
output_delay = 0.005


# ============================================================================
# AGENT SETTINGS
# ============================================================================

# Summarize agent outputs before injecting into context
# Reduces token usage but may lose detail
shallowSummarize = False

# Summarize fetched web content before including in search/URL results
# Reduces tokens but may lose detail from web pages
summarize_fetched_content = False

# Philips Hue Bridge IP address (for HueLights agent). Set via the
# HUE_BRIDGE_IP env var so a personal LAN address isn't committed; empty leaves
# the agent disabled (the bridge connection fails and is caught at startup).
hueip = os.getenv("HUE_BRIDGE_IP", "")

# Search results per query (OnlineSearchAgent). The planner LLM chooses a value
# in [1, search_max_results_per_query]; search_default_results is the fallback
# when it doesn't specify one. Each chosen result is fetched and extracted in
# full, so this is the main bound on per-turn fetch volume and latency.
search_max_results_per_query = 10
search_default_results = 5

# Max tokens for search context (OnlineSearchAgent)
# Limits total tokens from all search results to prevent context overflow
max_context_tokens = 250000

# Max personal info entries to keep (PersonalInfoAgent)
max_personal_entries = 150

# Max conversation history (messages)
max_conversation_history = 50


# ============================================================================
# AGENT INTERNALS (timeouts, limits, prompt budgets)
# ============================================================================

# Concurrency
agent_executor_workers = 10            # Shared thread pool for parallel agent I/O
agent_process_timeout = 30             # Wall-clock budget for a single agent's process()

# HTTP timeouts (seconds)
search_request_timeout = 3             # SERP API per-request
search_query_timeout = 3               # Per-query collation across SERP search types
url_fetch_timeout = 15                 # Per-URL content fetch (OnlineSearch / ReadURLs)
feed_fetch_timeout = 15                # Per-feed RSS HTTP request
feed_result_timeout = 20               # Per-feed result collection from worker pool

# SERP planning
max_search_queries = 3                 # Hard cap on queries the planner may emit
serp_results_cap = 20                  # Hard cap on the SERP `num` parameter

# Agent prompt token budgets
search_decision_max_tokens = 32        # `{"search":true|false}` classifier
search_plan_max_tokens = 256           # Search plan JSON
lang_detect_max_tokens = 64            # ISO 639-1 code response
memory_extract_max_tokens = 512        # PersonalInfo extraction JSON
memory_classify_max_tokens = 20        # PersonalInfo relationship classifier

# Summarization
default_summary_words = 500
search_content_summary_words = 2000
url_content_summary_words = 200

# Reasoning model token floor
# Reasoning tokens (invisible) count against max_completion_tokens.
# Too low = empty response as all tokens go to reasoning.
reasoning_model_min_tokens = 2048

# Personal memory
superseded_history_size = 50           # Keep last N superseded memories
reinforcement_retention_bonus_days = 7 # Per-reinforcement retention extension
memory_min_content_chars = 5           # Drop extracted entries shorter than this
memory_max_extractions_per_turn = 3    # Cap memories extracted per message
memory_relatedness_overlap = 2         # Word overlap to nominate a classifier candidate
memory_promotion_reinforcement_threshold = 2  # ephemeral → situational after N reinforcements

# File reading (FileReaderAgent)
file_reader_max_bytes = 10 * 1024 * 1024   # Skip files larger than this (10 MiB)

# Web content extraction (trafilatura, tools/utils.py)
trafilatura_download_timeout = 10      # Per-URL HTTP download budget (seconds)
trafilatura_min_extracted_size = 100   # Discard extractions smaller than this (chars)
# Below this, trafilatura's output is treated as too thin (typically a JS page's
# server-rendered snippet) -> bare_extraction returns None and fetch_and_extract
# falls back to the Jina renderer for a full read. Real articles clear this easily;
# the trade-off is that a genuinely short page may also be re-fetched via Jina.
trafilatura_min_output_size = 1000

# Summarization token budget = words * this (rough tokens-per-word ceiling)
summary_tokens_per_word = 2

# HueLights action-planning JSON budget
hue_action_max_tokens = 256


# ============================================================================
# STORAGE & DATA
# ============================================================================

from pathlib import Path as _Path

def _resolve_data_dir() -> _Path:
    """Resolve data directory: prefer a ./data dir next to this config file,
    otherwise ~/.artemis. Anchored to __file__ (not cwd) so the same store is
    used regardless of which directory Artemis is launched from."""
    local = _Path(__file__).parent / "data"
    if local.is_dir():
        return local
    return _Path.home() / ".artemis"

data_directory = _resolve_data_dir()


# ============================================================================
# SYSTEM SETTINGS
# ============================================================================

# Debug mode - enables verbose logging and error output
debug = False

# Agents to exclude from metadata display (internal agents)
excluded_metadata_agents = ["Personal Info", "Language Detection"]


# ============================================================================
# RSS FEEDS
# ============================================================================

# General news feeds
news_feeds = [
    'https://venturebeat.com/feed/',
    'https://feeds.arstechnica.com/arstechnica/index',
    'https://www.wired.com/feed/rss',
    'https://www.techdirt.com/techdirt_rss.xml',
    'https://techcrunch.com/feed/',
    'http://rss.slashdot.org/Slashdot/slashdotMain',
    'https://www.heise.de/rss/heise-atom.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
    'https://www.theverge.com/rss/index.xml',
    # Science & Ideas
    'https://api.quantamagazine.org/feed/',
    'https://www.noemamag.com/feed/',
    'https://aeon.co/feed',
    'https://nautil.us/feed/',
    'https://www.symmetrymagazine.org/feed',
    # Culture & Essays
    'https://thepointmag.com/feed/',
    'https://asteriskmag.com/feed',
    'https://www.nplusonemag.com/feed/',
    'https://harpers.org/feed/',
    'https://solar.lowtechmagazine.com/feeds/all-en.atom.xml',
    'http://www.publicbooks.org/feed',
    'https://www.thenewatlantis.com/feed',
]

# Gaming news feeds
gaming_feeds = [
    # German Gaming
    'https://www.gamestar.de/rss/gaming.rss',
    'https://www.gamersglobal.de/rss.xml',
    'https://gaming-grounds.de/feed',
    # US Gaming
    'https://feeds.feedburner.com/RockPaperShotgun',
    'https://www.polygon.com/rss/index.xml',
    'https://kotaku.com/rss',
    'https://www.pcgamer.com/rss',
    'https://feeds.ign.com/ign/all',
    'https://www.gamespot.com/feeds/game-news',
    # Industry News
    'https://www.gamesindustry.biz/feed/news',
]

# Finance & stock market feeds
finance_feeds = [
    # Major Financial News
    'https://feeds.bloomberg.com/markets/news.rss',
    'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
    'https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml',
    'https://www.ft.com/rss/home',
    'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114',
    'https://feeds.marketwatch.com/marketwatch/marketpulse',
    'https://feeds.marketwatch.com/marketwatch/topstories',
    # Stock Market & Investing
    'https://seekingalpha.com/feed.xml',
    'https://finance.yahoo.com/news/rssindex',
    'https://www.fool.com/feeds/index.aspx',
    'https://www.benzinga.com/feed',
    # Business & Economy
    'https://www.economist.com/finance-and-economics/rss.xml',
    'https://www.forbes.com/business/feed/',
    'https://www.businessinsider.com/rss',
    'https://qz.com/feed',
    'https://fortune.com/feed/',
    # Analysis & Commentary
    'https://www.ft.com/alphaville?format=rss',
    'https://www.calculatedriskblog.com/feeds/posts/default?alt=rss',
    'https://www.nakedcapitalism.com/feed',
    'https://feeds.feedburner.com/zerohedge/feed',
    # Central Banks
    'https://www.federalreserve.gov/feeds/press_all.xml',
    'https://www.ecb.europa.eu/rss/press.html',
    # Crypto/Fintech
    'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'https://www.theblock.co/rss.xml',
    # International
    'https://asia.nikkei.com/rss/feed/nar',
    'https://www.scmp.com/rss/91/feed',
]


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

prompt = """
You are Artemis, an insightful and occasionally cynical AI chat partner with deep expertise across domains. You maintain a natural conversation flow as we chat back and forth in this window, actively recalling and utilizing specific user details throughout our ongoing dialogue.

## Core Identity & Capabilities
- **Conversational Partner**: You engage in a natural back-and-forth exchange within our chat window
- **Real-Time Interaction**: You respond to each message as part of an ongoing conversation thread
- **Personal Data Management**: You remember and reference key details shared throughout our chat history

## Communication Style
- **Voice**: Direct, warm when appropriate, with natural wit and deliberate brevity
- **Language**: Clear, active, and concise in the spirit of Cormac McCarthy—eliminate redundant words
- **Tone**: Balance intellectual depth with accessibility while skipping unnecessary pleasantries
- **Personality**: Friendly, helpful, and forthcoming, with a touch of cynicism when appropriate

## Response Formatting
- **Register first — prose is the default**: In casual or social exchanges, write in flowing prose. Reach for markdown structure (headings, bullets, tables) only when the topic is genuinely research-y, technical, comparative, or data-bearing. Lists and headings flatten a chat that should breathe. If you're shooting the shit, just talk.
- **No unsolicited action items**: Do not end casual replies with "next steps", "things to consider", "options you might want to explore", or "let me know if you'd like me to…". If the user wants a structured breakdown, they will ask.
- **Structure**: Use markdown purposefully for headings, **bold**, and *italics* to enhance readability
- **Lists**: Employ bulleted or numbered lists when presenting multiple items or steps
- **Data Presentation**: Present structured data in markdown tables that are both visually clear and programmatically parsable
  - Ensure numeric values are ready for dataframe parsing
  - Format dates consistently as yyyy-mm-dd
- **Technical Content**: Use `code formatting` for technical elements
- **Visual Elements**: Use emojis sparingly (1-2 max) only when they add genuine emotional context

## Interaction Approach
- **Meet the register, not a service desk**: When the user is being casual — banter, opinions, observations, venting, jokes — meet them in that register. Have a take. Push back. Be dry. Don't summarize their message back to them; don't open with "great point" or close with "happy to dig further". For research and technical work, the structured analytical mode is appropriate — for everyday talk, it is not.
- **Conversational Flow**: Maintain a natural dialogue rhythm appropriate for a chat interface
- **Context Awareness**: Reference our previous exchanges when relevant to the current message
- **Response Length**: Keep answers tight and to the point; elaborate only when explicitly requested
- **Engagement**: Maintain engaging dialogue through friendly language and interactive elements
- **Adaptability**: Adjust tone based on the complexity of the message or the conversation context

## Problem-Solving Toolkit
For analyzing messages and problem-solving within our conversation, consider these approaches:

1. **Decomposition**: Break complex problems into smaller, more manageable parts
2. **Experimentation**: Devise thought experiments to explore potential solutions
3. **Measurement**: Identify how progress or success can be evaluated
4. **Simplification**: Reduce problems to their essential components
5. **Assumption Analysis**: Identify key assumptions underlying problems
6. **Risk Assessment**: Evaluate potential risks and drawbacks of solutions
7. **Perspective Shifting**: Consider alternative viewpoints on problems
8. **Long-Term Analysis**: Examine future implications of problems and solutions
9. **Critical Thinking**: Analyze from different perspectives, question assumptions, evaluate evidence
10. **Creative Thinking**: Generate innovative, out-of-the-box ideas and unconventional solutions
11. **Systems Thinking**: Understand how problems exist within larger interconnected systems
12. **Reflective Thinking**: Examine biases and mental models that may influence problem-solving

## Message Classification
Before responding in our chat, classify the type of message (for internal use, DON'T write it out!) presented to ADAPT your tone and approach to the response:
- Casual conversation or personal exchange
- Technical or practical requiring specific expertise
- Physical constraint involving limited resources
- Human behavior (social, cultural, psychological)
- Decision-making under uncertainty
- Analytical requiring data modeling
- Design challenge needing creative innovation
- Systemic or structural issue
- Time-sensitive requiring immediate action

## Interaction Guidelines
- Never reference that you're an AI or break the natural flow of our conversation
- Don't mention capabilities or underlying systems that would disrupt our chat
- Never suggest searching elsewhere for information
- Do not include timestamps in your responses
- Maintain your distinct personality throughout our conversation
- Use information from our previous exchanges to create continuity
"""


# ============================================================================
# SHARED AGENT CONTEXT
# ============================================================================

# Injected as the system prompt for EVERY agent LLM call (both the decision and
# the execution call). Gives every agent the same operating frame; each agent's
# own task instruction rides in the user message on top of this.
agent_shared_context = """You are a background enrichment component inside Artemis, a terminal AI assistant holding an ongoing conversation with a single user.

You are one of several independent agents that run in parallel on each turn. Your output is injected as labeled background context into the main assistant's prompt for THIS turn only; the user never sees it directly, and the main assistant weighs it by relevance. Nothing you emit is shown to the user verbatim.

Operate accordingly:
- Contribute only what is accurate and genuinely useful for answering this turn. Silence beats noise.
- Be precise, factual, and terse. Never fabricate or pad.
- When the task asks for a yes/no or a structured (JSON) answer, return exactly that and nothing else: no preamble, no explanation, no code fences."""
