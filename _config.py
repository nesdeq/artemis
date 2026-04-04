"""
Artemis AI Configuration
Centralized configuration for all system components.
"""

# ============================================================================
# LLM MODELS
# ============================================================================

# Main LLM for conversation
llm = "openai/gpt-5.1"  # Latest with adaptive reasoning (minimal to xhigh)
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
#agent_llm = "openai/gpt-5-nano"  # $0.05/$0.40 per MTok - best value for classification
agent_llm = "openai/gpt-5-mini"  # $0.25/$2.00 per MTok - more capable
#agent_llm = "openai/gpt-4.1-mini"  # 1M context
#agent_llm = "openai/gpt-4o-mini"  # Legacy but stable
#agent_llm = "anthropic/claude-haiku-4-5-20251001"  # $1/$5 per MTok - fast
#agent_llm = "gemini/gemini-2.0-flash"

# Reasoning effort for main LLM:
#   O-series (o1/o3/o4): low, medium, high
#   GPT-5 series: minimal, low, medium, high
#   GPT-5.1 series: minimal, low, medium, high, xhigh
ro = "medium"  # Uses model-specific defaults if None (minimal/low depending on model)

# Reasoning effort for agent LLM
# Agents need fast responses (classification, extraction), so use lowest valid:
#   O-series: "low" (minimal not supported)
#   GPT-5/5.1: "minimal"
# LLMInterface auto-validates and falls back to lowest valid if misconfigured
agent_ro = "medium"


# ============================================================================
# LLM API SETTINGS
# ============================================================================

# Request timeout in seconds
llm_timeout = 30

# Number of retry attempts for failed API calls
llm_retry_attempts = 3

# Maximum tokens for responses
max_tokens = 4096


# ============================================================================
# STREAMING & PERFORMANCE
# ============================================================================

# Enable streaming responses from LLM API
streaming = True

# Output delay - controls streaming speed across ALL interfaces (seconds)
# Lower = faster output, higher = smoother/more readable
# Recommended: 0.001 (fast), 0.005 (balanced), 0.01 (smooth)
output_delay = 0.005

# CLI refresh rate (Hz) - how often terminal display updates
# Separate from output_delay - controls terminal rendering frequency
cli_refresh_rate = 120


# ============================================================================
# AGENT SETTINGS
# ============================================================================

# Summarize agent outputs before injecting into context
# Reduces token usage but may lose detail
shallowSummarize = False

# Summarize fetched web content before including in search/URL results
# Reduces tokens but may lose detail from web pages
summarize_fetched_content = False

# Philips Hue Bridge IP address (for HueLights agent)
hueip = "192.168.0.23"

# Max search results per query (OnlineSearchAgent)
max_search_results = 3

# Max tokens for search context (OnlineSearchAgent)
# Limits total tokens from all search results to prevent context overflow
max_context_tokens = 250000

# Max personal info entries to keep (PersonalInfoAgent)
max_personal_entries = 150

# Max conversation history (messages)
max_conversation_history = 50


# ============================================================================
# STORAGE & DATA
# ============================================================================

from pathlib import Path as _Path

def _resolve_data_dir() -> _Path:
    """Resolve data directory: use local ./data if it exists, otherwise ~/.artemis."""
    local = _Path.cwd() / "data"
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
- **Structure**: Use markdown purposefully for headings, **bold**, and *italics* to enhance readability
- **Lists**: Employ bulleted or numbered lists when presenting multiple items or steps
- **Data Presentation**: Present structured data in markdown tables that are both visually clear and programmatically parsable
  - Ensure numeric values are ready for dataframe parsing
  - Format dates consistently as yyyy-mm-dd
- **Technical Content**: Use `code formatting` for technical elements
- **Visual Elements**: Use emojis sparingly (1-2 max) only when they add genuine emotional context

## Interaction Approach
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
