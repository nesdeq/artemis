# Writing Agents

Agents are independent enrichers. Each one looks at the user's input, decides whether it has anything to add, and (if so) returns a string that gets injected into the LLM context for the current turn. No router, no schema, no tool-calls — just parallel functions.

## Architecture Laws

**These are non-negotiable. Don't optimize past them. Don't compromise on them.** The same text lives at the top of `agents/Agent.py` — the two MUST stay in sync.

### 1. The LLM judges. Always.

When an agent's job requires judgement — *is this English?*, *does this need a web search?*, *is this a memorable fact?* — the LLM makes that judgement, **every turn**. Routing-by-text-interpretation is forbidden. Heuristics cannot replace judgement.

### 2. Two-call pattern: one decision, one execution.

- `should_process` → the **DECISION** LLM call: *should this agent contribute this turn?*
- `process` → the **EXECUTION** LLM call: *do the work.*

These are separate calls. **NEVER merged.**

### 3. Execution may use multiple LLM calls.

When the execution has complex sub-tasks (plan + per-result refinement, batched extraction across items, etc.), multiple LLM calls inside `process` are fine. The **decision** remains a single call, separate from execution.

### 4. No call-count optimization. Ever.

- No caching of LLM output across turns.
- No heuristic substitution for LLM judgement.
- No merging two LLM decisions into one to "save a call".

**The call count IS the architecture.** Cost is not your problem; correctness is.

### 5. Cheap pre-filters before the decision call are allowed — when binary.

If the answer is structurally yes/no — *URL present? bang command? file on disk?* — a regex / filesystem check may short-circuit `should_process` **before** the decision LLM call. It does **not** replace the decision call when the question requires judgement.

Examples:

| Agent | `should_process` |
|---|---|
| `URLReader` | `bool(extract_urls(text))` — binary, no judgement → **no LLM call**. |
| `FileReader` | regex + filesystem check — binary → **no LLM call**. |
| `News` | bang-command lookup (`!news`, `!games`, `!finance`) — binary → **no LLM call**. |
| `OnlineSearch` | bangs and URLs short-circuit to `False` first, then **LLM call** *"do you need to search?"* — judgement required. |
| `LangDetect` | always-on; decision is degenerate (returns `True`). Execution does the LLM call. |
| `PersonalInfo` | LLM call *"is there memorable personal info here?"* — judgement required. |
| `HueLights` | keyword regex short-circuits, then **LLM call** *"is this a light-control command?"* — judgement required. |

If your agent's "should I run?" requires judgement, you must call the LLM. If it is structurally binary, you must not.

## The base class

`agents/Agent.py`:

```python
class Agent:
    _shared_executor: Optional[ThreadPoolExecutor] = None  # 10 workers, lazy-init

    def __init__(self, name: Optional[str] = None, user: Optional[str] = None) -> None:
        self.name = name
        self.user = user
        self.llm = LLMInterface(
            _config.agent_llm,
            reasoning_effort=_config.agent_ro,
            context=name,
        )
        self.metadata: Dict[str, Any] = {}

    def should_process(self, user_input, last_response=None) -> bool: ...   # override
    def process(self, user_input, last_response=None) -> Optional[str]: ... # override

    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor: ...
    def get_metadata(self) -> Dict[str, Any]: ...
```

You override two methods.

| Method | Returns | Purpose |
|---|---|---|
| `should_process` | `bool` | Cheap gate. Called for every message. Keep it fast — no LLM calls unless you must. |
| `process` | `Optional[str]` | Do the work. Return a string to inject; return `None` to contribute nothing. |

## Lifecycle

1. **Init** — `core.py` instantiates each registered agent on first user input (lazy, parallel via `asyncio.to_thread`).
2. **Gate** — for every user turn, `should_process(...)` runs in parallel across all agents.
3. **Run** — only agents that returned `True` from the gate get `process(...)` called, in parallel.
4. **Inject** — non-`None` returns get formatted as a `### <agent name>` markdown section under a "Background context" header and appended to the user message for that turn only.
5. **Metadata** — `agent.metadata` is read after `process` and surfaced in the sources panel.

Agent context is ephemeral: it lives for one request only. Conversation history persists; agent enrichment does not.

## Minimal example

```python
# agents/Weather.py
import logging
from typing import Optional
from .Agent import Agent

logger = logging.getLogger(__name__)


class WeatherAgent(Agent):
    """Inject current weather when the user asks about it."""

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        return "weather" in user_input.lower()

    def process(self, user_input: str, last_response: Optional[str] = None) -> Optional[str]:
        self.metadata = {"source": "weather-api"}
        try:
            return "Current weather: 18°C, partly cloudy."
        except Exception as e:
            logger.error(f"Weather lookup failed: {e}")
            return None
```

Register in `core.py`:

```python
AGENT_REGISTRY = [
    (PersonalInfoAgent,    "Personal Info"),
    (DetectLanguageAgent,  "Language Detection"),
    (OnlineSearchAgent,    "Online Research"),
    (URLReaderAgent,       "URL Reader"),
    (FileReaderAgent,      "File Reader"),
    (DailyStoriesAgent,    "Daily News"),
    (WeatherAgent,         "Weather"),  # add here
]
```

## Patterns in the existing agents

Look at how the built-ins gate work:

| Agent | `should_process` strategy |
|---|---|
| `PersonalInfo`, `LangDetect` | Always returns `True` (always-on). |
| `URLReader` | `bool(extract_urls(user_input))` — pure regex, free. |
| `FileReader` | Regex filename match + filesystem existence check. |
| `News` | Bang-command lookup: `!news`, `!games`, `!finance`. |
| `OnlineSearch` | Cheap exclusions first (bang, URL present, news request) — only then an LLM yes/no. |

The pattern: free signals first, paid signals last. Don't burn an agent LLM call on every keystroke if a regex can answer.

## Parallel I/O

`Agent.get_executor()` returns the shared `ThreadPoolExecutor` (size from `_config.agent_executor_workers`). For fan-out work — fetching multiple URLs, parsing multiple feeds — use the helper in `tools/utils.py`:

```python
from tools.utils import parallel_map

results = parallel_map(
    items,
    fetch_one,                 # callable: item -> Optional[result]
    self.get_executor(),
    timeout=_config.url_fetch_timeout,
)
```

`parallel_map` submits one task per item, drops `None`s, swallows exceptions (logged), and returns whatever finished within the timeout.

## The agent's LLM

`self.llm` is an `LLMInterface` configured for `_config.agent_llm` — typically a smaller, cheaper model than the main conversation model. Two helpers:

```python
self.llm.generate_single_response(prompt, max_tokens=64)        # complete-and-return
self.llm.summarize(text, max_words=200)                         # convenience
```

Costs are tracked per-context (the `context` kwarg defaults to the agent's name). Use `/cost` in the TUI to see the breakdown.

## Configuration

Tunable knobs live in `_config.py` — timeouts, token budgets, retry counts, summary lengths. Read from there rather than hardcoding numbers in agents:

```python
self.llm.generate_single_response(prompt, max_tokens=_config.lang_detect_max_tokens)

resp = requests.get(url, timeout=_config.feed_fetch_timeout)
```

If you need a new tunable, add it to `_config.py` rather than introducing a literal in agent code.

## Metadata

`self.metadata` is a free-form dict surfaced in the sources panel. Reset it at the top of `process` and populate it with anything the user might want to see — URLs hit, queries run, files read, stories fetched. Keep keys lowercase and descriptive.

```python
def process(self, user_input, last_response=None):
    self.metadata = {}
    urls = extract_urls(user_input)
    if not urls:
        return None
    enriched = parallel_map(urls, self._fetch, self.get_executor(),
                            timeout=_config.url_fetch_timeout)
    self.metadata["urls"] = [r["url"] for r in enriched]
    return self._format(enriched)
```

To hide an agent from the metadata panel (e.g. internal-only context), add its registered name to `_config.excluded_metadata_agents`.

## Persistence

If your agent stores per-user data, follow `PersonalInfo`'s pattern:

- Use `_config.data_directory` as the root.
- Hash `self.user` to build a per-user filename.
- Encrypt with `tools.utils.encrypt_data` / `decrypt_data` (Fernet, key derived from `ENCKEY` env var via `derive_encryption_key`).
- Save lazily — accumulate dirty state and flush at the end of `process`.

## Output formatting

Agent output is plain text, joined as `### <name>` markdown sections under a "Background context for this turn" header before being passed to the main LLM. Structured strings work well — labels, bullets, tables. The main LLM is told to use this context proportionally to query complexity, so don't worry about it overwhelming a casual reply.

## Checklist

Before adding an agent:

- [ ] Inherits from `Agent`.
- [ ] `should_process` returns quickly with cheap signals (no LLM call unless necessary).
- [ ] `process` resets `self.metadata` at the top.
- [ ] Returns `None` (not `""`) when the agent has nothing to add.
- [ ] Tunable values live in `_config.py`, not as literals.
- [ ] External I/O uses `Agent.get_executor()` + `parallel_map` for fan-out.
- [ ] Exceptions in worker functions are caught and logged — never let one URL kill the whole batch.
- [ ] Registered in `core.py`'s `AGENT_REGISTRY`.
