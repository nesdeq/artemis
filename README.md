# Artemis

A terminal AI assistant built around **parallel, independent agents**. No router, no manager, no tool-call schema — every agent decides for itself whether it has something to add to the current turn, and the ones that do run concurrently. Their outputs are stitched into the message context before the main LLM ever sees it.

## The idea

Most assistant frameworks centralise control: user input goes through a router, the router picks tools, results funnel back through the same bottleneck. That works, but it makes the router the bottleneck for both latency and capability.

Artemis flips it. Every agent receives every message simultaneously. Each one answers a single question — *can I add something useful here?* — and runs only if the answer is yes. Independent agents, parallel execution, no shared state during a turn.

```
You: "What's the current state of the Rust job market?"

                    ┌─ PersonalInfo ──── always on, extracts user info to memory
                    ├─ LangDetect ────── always on, returns "en"
 user input ───────►├─ OnlineSearch ──── LLM yes → searches, fetches, extracts
                    ├─ URLReader ─────── no URLs found → skip
                    ├─ FileReader ────── no file paths → skip
                    └─ DailyNews ─────── no bang command → skip

                            │ (parallel; ≤ agent_process_timeout)
                            ▼
              ┌──────────────────────────┐
              │  enriched message array  │
              │  system + history +      │
              │  user msg + agent ctx    │
              └──────────────────────────┘
                            │
                            ▼
                    main LLM streams response
```

The main LLM doesn't know agents exist. It sees a message with markdown-headed context blocks already attached. Swap the main model freely — agents are decoupled from it.

## How a turn is assembled

```markdown
<user query>

---

## Background context for this turn

Each section below was gathered automatically by a background agent. Weight
them by query type: minimal for chitchat, selective for topical discussion,
fully for research / factual / time-sensitive questions. Do not surface this
scaffolding to the user.

### Personal Info
<profile>

### Online Research
<extracted page content>

### Language Detection
<directive>
```

Two important properties:

1. **Empty agents stay quiet.** A casual greeting has zero context blocks — no overhead, no token waste.
2. **Context is ephemeral.** The enriched message is what the LLM sees for this turn only; conversation history persists only the raw user input. Agent output enriches one turn and vanishes — the next turn gets fresh signals based on fresh input.

## Decisions go to the LLM, except when they shouldn't

Each agent's `should_process` gate is the cheapest check that's still correct:

- `URLReader`, `FileReader`, `News` — pure regex / filesystem checks. URL detection is binary; no need for an LLM.
- `OnlineSearch` — `!search`/`!web` forces a search; other bangs and pasted URLs short-circuit to skip; otherwise the LLM decides whether a search is warranted at all (conservatively — not on follow-up chatter or commentary on its own prior answer), then plans the queries, result count, country, language, and recency.
- `PersonalInfo`, `LangDetect` — always on. They return whatever they have.

Routing-by-text-interpretation is avoided. The LLM is reserved for actual judgement: *do you need fresh information?*, *is this English?*, *which memories should I update?*.

## Interface

One engine (`core.py`), one frontend:

- **TUI** (`tui.py`) — a [Textual](https://textual.textualize.io/) interface: streaming markdown, switchable themes, history search, live per-agent status while agents run, and click-to-expand sources showing the exact context each agent injected. This is what the `arti` launcher runs after install. Start with `arti` or `python tui.py`.

## Install

One-line install (clones into `~/.local/share/artemis`, sets up a venv, drops an `arti` launcher into `~/.local/bin`):

```bash
curl -fsSL https://raw.githubusercontent.com/nesdeq/artemis/main/install.sh | bash
```

Then run `arti` (the TUI). Make sure `~/.local/bin` is on your `PATH`.

Prefer to inspect before running (you should):

```bash
curl -fsSL https://raw.githubusercontent.com/nesdeq/artemis/main/install.sh -o install.sh
less install.sh
bash install.sh
```

Override locations with env vars: `ARTEMIS_HOME=/custom/path BIN_DIR=$HOME/bin bash install.sh`. Re-run the installer to update an existing checkout.

### Manual install

```bash
git clone https://github.com/nesdeq/artemis.git
cd artemis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set credentials:

```bash
export OPENAI_API_KEY="..."        # or any provider litellm supports
export SERP_API_KEY="..."          # for web search (serper.dev)
export ENCKEY="..."                # encryption key for personal memory
export HUE_BRIDGE_IP="..."         # optional, only for the latent HueLights agent
```

Run it:

```bash
python tui.py    # Textual TUI (what `arti` launches)
```

## Models

Two tiers. Configure in `_config.py`:

```python
llm = "openai/gpt-5.1"           # main conversation model
agent_llm = "openai/gpt-5-mini"  # cheaper, faster — used by agents

ro = "high"                    # main reasoning effort
agent_ro = "medium"            # agent reasoning effort
```

Anything [litellm](https://github.com/BerriAI/litellm) supports works — OpenAI, Anthropic, Gemini, Ollama, local models. Run agents on a small local model and the main loop on a frontier model, or all-local, or all-cloud.

## Built-in agents

| Agent | Trigger | Output |
|---|---|---|
| `PersonalInfo` | always | Encrypted user profile (Fernet/AES). One LLM call per turn extracts memorable info and marks which existing entries it replaces. Retention buckets: core (forever) / situational (30d) / ephemeral (2d). |
| `LangDetect` | always | ISO 639-1 directive — main LLM responds in the user's language. |
| `OnlineSearch` | LLM-decided (`!search` forces) | SERP-API web search. The LLM decides whether a search is warranted, then plans every parameter: queries, results per query (1-10), country (`gl`), language (`hl`), recency (`tbs`), and whether to include news. Pages fetched and extracted via trafilatura, with Jina Reader fallback for JS pages. |
| `URLReader` | URL in input | Fetches and extracts main content from any URL. |
| `FileReader` | absolute path in input | Reads text, PDF, DOCX, XLSX, CSV, and more (MarkItDown). Path-traversal protected. |
| `DailyNews` | `!news` `!games` `!finance` | RSS aggregation across 50+ feeds, fetched concurrently. |
| `HueLights` | LLM-decided (latent — not registered by default) | Controls Philips Hue via natural language. |

## Writing your own agent

Inherit, implement two methods, register. That's it.

```python
# agents/Weather.py
from typing import Optional
from .Agent import Agent

class WeatherAgent(Agent):
    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        # Cheap. Runs for every message. No LLM unless you really need one.
        return "weather" in user_input.lower()

    def process(self, user_input: str, last_response: Optional[str] = None) -> Optional[str]:
        self.metadata = {"source": "weather-api"}
        return "Current weather: 18°C, partly cloudy."
```

Add it to `core.py`:

```python
AGENT_REGISTRY = [
    # ... existing agents ...
    (WeatherAgent, "Weather"),
]
```

Done. Your agent's gate is called in parallel with every other agent's, and if it passes, `process` runs in parallel too. The string you return becomes a `### Weather` section in the next turn's context.

Each agent gets:
- `self.llm` — `LLMInterface` configured for the agent model, with cost tracking under the agent's name
- `self.metadata` — surfaced in the sources panel
- `self.user` — current user identifier
- `Agent.get_executor()` — shared `ThreadPoolExecutor` for fan-out I/O
- `tools.utils.parallel_map(items, fn, executor, timeout)` — drop-in helper for concurrent I/O with real timeout enforcement

See `agents/_Agents.md` for the full guide and `agents/OnlineSearch.py` for a non-trivial reference.

## Commands

Available in the TUI:

| Command | Action |
|---|---|
| `/exit` | Quit |
| `/save [name]` | Save the last exchange as markdown |
| `/export [name]` | Export the full session — every turn, sources, and token stats — as markdown |
| `/cost` | Per-context token + cost breakdown (main + each agent) |

## Project layout

```
artemis/
├── tui.py                  # Textual TUI (launched by `arti`)
├── frontend_io.py          # save/export/cost logic for the TUI
├── core.py                 # orchestrator: agent lifecycle, context assembly, streaming
├── _config.py              # all configuration in one place
├── install.sh              # one-line installer (clone + venv + `arti` launcher)
├── llms/
│   └── LLMInterface.py     # litellm wrapper, exact usage tracking via stream_options
├── agents/
│   ├── Agent.py            # base class + shared executor
│   ├── PersonalInfo.py     # encrypted user memory
│   ├── LangDetect.py       # language detection
│   ├── OnlineSearch.py     # web search + content extraction
│   ├── ReadURLs.py         # URL content fetcher
│   ├── FileReader.py       # multi-format file reader
│   ├── News.py             # RSS aggregator
│   └── HueLights.py        # smart-home control (latent)
├── tools/
│   ├── utils.py            # shared utilities, encryption, web extraction, parallel_map
│   └── readpinf.py         # personal-memory inspector / debugger
└── requirements.txt
```

## Data and privacy

Everything persistent lives in `~/.artemis/` (or `./data/` if it exists in the project root):

| File | Contents |
|---|---|
| `.pinf` / `.pinf_<hash>` | Encrypted personal memory store (Fernet) |
| `.artemis_history` | Command history |
| `artemis_*.md` | Saved chat exports |

Personal memory is encrypted at rest with a key derived from `ENCKEY` via PBKDF2-SHA256. No cloud sync, no telemetry. Inspect what's stored:

```bash
python tools/readpinf.py --stats
python tools/readpinf.py --list
python tools/readpinf.py --search python
```

## License

MIT
