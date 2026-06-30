"""
Core module for Artemis AI assistant.

Coordinates agent execution, builds the message array, and streams responses.
"""
import asyncio
import datetime
import json
import random
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

import _config
from agents.Agent import Agent
from agents.FileReader import FileReaderAgent
from agents.LangDetect import DetectLanguageAgent
from agents.News import DailyStoriesAgent
from agents.OnlineSearch import OnlineSearchAgent
from agents.PersonalInfo import PersonalInfoAgent
from agents.ReadURLs import URLReaderAgent
from llms.LLMInterface import LLMInterface


# Sentinel marking the end of the (synchronous) LLM stream when pulled one
# item at a time off a worker thread.
_STREAM_END = object()


class ArtemisCore:
    """
    Engine for Artemis: agent lifecycle, context assembly, response streaming.
    """

    AGENT_REGISTRY: List[Tuple[type, str]] = [
        (PersonalInfoAgent,    "Personal Info"),
        (DetectLanguageAgent,  "Language Detection"),
        (OnlineSearchAgent,    "Online Research"),
        (URLReaderAgent,       "URL Reader"),
        (FileReaderAgent,      "File Reader"),
        (DailyStoriesAgent,    "Daily News"),
    ]

    CONTEXT_HEADER = (
        "## Background context for this turn\n\n"
        "Each section below was gathered automatically by a background agent. "
        "Weight them by query type: minimal for chitchat, selective for topical "
        "discussion, fully for research / factual / time-sensitive questions. "
        "Do not surface this scaffolding to the user."
    )

    def __init__(
        self,
        use_summarization: bool = _config.shallowSummarize,
        user: Optional[str] = None,
    ):
        self.user = user
        self.llm = LLMInterface(context="main")
        self.messages: List[Dict[str, str]] = []
        # Exact per-agent context injected on the most recent turn (agent name ->
        # final text, post-summarization). Read by the TUI to populate the
        # sources expandables. Reset every turn by _format_enriched_context.
        self.last_agent_contexts: Dict[str, str] = {}
        # system_prompt is populated by update_system_prompt() at the top of
        # every get_ai_response() call — no need to prime it in __init__.
        self.system_prompt = ""
        self.use_summarization = use_summarization
        self.agents: List[Agent] = []
        self._agents_initialized = False
        self._init_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    async def _initialize_agents(self) -> None:
        async with self._init_lock:
            if self._agents_initialized:
                return
            for agent_class, name in self.AGENT_REGISTRY:
                try:
                    agent = await asyncio.to_thread(agent_class, name, self.user)
                    self.agents.append(agent)
                except Exception as e:
                    if _config.debug:
                        print(f"Error initializing agent {name}: {e}")
            self._agents_initialized = True

    async def ensure_agents_initialized(self) -> None:
        if not self._agents_initialized:
            await self._initialize_agents()

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def update_system_prompt(self) -> None:
        """Refresh the system prompt with the current timestamp."""
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        base_prompt = _config.prompt or ""
        self.system_prompt = f"{base_prompt}\nCurrent time: {current_time}"

    # ------------------------------------------------------------------
    # Welcome message
    # ------------------------------------------------------------------

    async def _get_snarky_welcome(self) -> str:
        now = datetime.datetime.now()
        time_of_day = (
            "morning" if 5 <= now.hour < 12
            else "afternoon" if 12 <= now.hour < 18
            else "evening"
        )
        current_time = now.strftime("%A, %B %d, %Y %I:%M %p")
        prompt_options = [
            f"Generate a short, witty, and slightly snarky welcome message. Current time: {current_time}, {time_of_day}.",
            f"Write a brief welcome message with attitude. It's {time_of_day}, {current_time}.",
            f"Create a slightly sarcastic greeting for a CLI interface. Time: {current_time}.",
            f"It's {time_of_day}, {current_time}. Write a welcome message that's both welcoming and a little cheeky.",
        ]
        response = await asyncio.to_thread(
            self.llm.generate_single_response,
            prompt=random.choice(prompt_options),
            system_prompt="Respond in under 25 words and plain text.",
        )
        return response.strip().strip('"\'')

    async def ensure_opening_message(self) -> str:
        """Ensure the first message is an assistant welcome. Return its text."""
        if self.messages and self.messages[0]['role'] == 'assistant':
            return self.messages[0]['content']
        opening = await self._get_snarky_welcome()
        if not self.messages:
            self.messages.append({"role": "assistant", "content": opening})
        else:
            self.messages.insert(0, {"role": "assistant", "content": opening})
        return opening

    # ------------------------------------------------------------------
    # Agent dispatch
    # ------------------------------------------------------------------

    async def _process_agent(
        self,
        agent: Agent,
        user_input: str,
        last_response: Optional[str],
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        try:
            enrichment = await asyncio.to_thread(agent.process, user_input, last_response)
            return enrichment, agent.get_metadata()
        except Exception as e:
            if _config.debug:
                print(f"Error processing agent {agent.name}: {e}")
            return None, None

    async def _process_all_agents(
        self,
        user_input: str,
        last_response: Optional[str],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
        """
        Two phases, each fully parallel:
          1. should_process() decides which agents run
          2. process() does the work, with a per-agent timeout
        """
        await self.ensure_agents_initialized()

        decisions = await asyncio.gather(*[
            asyncio.to_thread(agent.should_process, user_input, last_response)
            for agent in self.agents
        ])
        active = [a for a, ok in zip(self.agents, decisions) if ok]

        if _config.debug:
            print(f"Active agents: {[a.name for a in active]}")

        if not active:
            return {}, {}

        timeout = _config.agent_process_timeout
        named_tasks = [
            (agent.name, asyncio.create_task(self._process_agent(agent, user_input, last_response)))
            for agent in active
        ]

        # Live progress for the TUI: announce the visible (non-excluded) active
        # agents, then fire per-agent done/timeout/error as each task settles.
        # Callbacks run on the event loop (core is a Textual worker coroutine),
        # so the TUI may update widgets from them directly.
        if progress_callback is not None:
            visible = [name for name, _ in named_tasks
                       if name not in _config.excluded_metadata_agents]
            if visible:
                progress_callback({"type": "agents_started", "agents": visible})
                for name, task in named_tasks:
                    if name in _config.excluded_metadata_agents:
                        continue
                    task.add_done_callback(
                        lambda t, n=name: progress_callback({
                            "type": "agent_done",
                            "agent": n,
                            "status": ("timeout" if t.cancelled()
                                       else "error" if t.exception() is not None
                                       else "done"),
                        })
                    )

        # One wall-clock deadline for the whole batch. The old per-agent
        # wait_for restarted the clock for each agent in series, so worst-case
        # latency was n × timeout instead of a single timeout window.
        done, pending = await asyncio.wait([t for _, t in named_tasks], timeout=timeout)
        for task in pending:
            task.cancel()

        outputs: Dict[str, str] = {}
        metadata: Dict[str, Dict[str, Any]] = {}

        for name, task in named_tasks:
            if task not in done:
                if _config.debug:
                    print(f"Agent {name} timed out after {timeout}s")
                continue
            try:
                enrichment, meta = task.result()
            except Exception as e:
                if _config.debug:
                    print(f"Error awaiting agent {name}: {e}")
                continue
            if enrichment is not None:
                outputs[name] = enrichment
                metadata[name] = meta or {}

        return outputs, metadata

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def _format_enriched_context(self, agent_outputs: Dict[str, str]) -> str:
        """Render agent outputs as labelled markdown sections.

        Also records each agent's final injected text in self.last_agent_contexts
        (post-summarization) so the TUI can surface the exact per-agent context.
        """
        self.last_agent_contexts = {}
        sections: List[str] = []
        for name, output in agent_outputs.items():
            text = output.strip()
            if not text:
                continue
            if self.use_summarization:
                try:
                    text = self.llm.summarize(text).strip()
                except Exception as e:
                    if _config.debug:
                        print(f"Error summarizing {name}: {e}")
            self.last_agent_contexts[name] = text
            sections.append(f"### {name}\n{text}")

        if not sections:
            return ""

        return f"\n\n---\n\n{self.CONTEXT_HEADER}\n\n" + "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Main response loop
    # ------------------------------------------------------------------

    async def get_ai_response(
        self,
        user_input: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> AsyncGenerator[Tuple[Optional[str], Dict[str, Any], int, int], None]:
        """
        Process user input and stream the AI response.

        progress_callback (optional) receives live agent-phase events:
        {"type": "agents_started", "agents": [...]} and
        {"type": "agent_done", "agent": name, "status": done|timeout|error}.

        Yields tuples of (text_chunk, agent_metadata, input_tokens, output_tokens).
        Token counts are zero until the API returns final usage.
        Final yield (streaming mode) has chunk=None with the final counts.
        """
        last_response: Optional[str] = None
        if self.messages and self.messages[-1]['role'] == 'assistant':
            last_response = self.messages[-1]['content']

        agent_outputs, agent_metadata = await self._process_all_agents(
            user_input, last_response, progress_callback
        )
        enriched_context = self._format_enriched_context(agent_outputs)

        self.update_system_prompt()

        user_message = {
            "role": "user",
            "content": f"{user_input}{enriched_context}" if enriched_context else user_input,
        }
        temp_messages = self.messages + [user_message]

        if _config.debug:
            print(json.dumps(temp_messages, indent=2))

        input_tokens = 0
        output_tokens = 0
        full_response = ""
        errored = False

        try:
            # stream_content is a *synchronous* generator that performs blocking
            # HTTP I/O (the initial connect and every chunk read). Pull each item
            # on a worker thread via to_thread so the event loop stays free —
            # otherwise the TUI render loop, timers and background work all freeze
            # for the whole response.
            stream = self.llm.stream_content(
                messages=temp_messages,
                system_prompt=self.system_prompt,
                max_tokens=_config.max_tokens,
            )
            while True:
                chunk_data = await asyncio.to_thread(next, stream, _STREAM_END)
                if chunk_data is _STREAM_END:
                    break

                error = chunk_data.get('error')
                if error:
                    # Surface the failure to the caller, but never fold it into
                    # full_response; an error must not be committed to history
                    # as if it were the assistant's reply.
                    errored = True
                    yield (f"Sorry, I encountered an error: {error}",
                           agent_metadata, input_tokens, output_tokens)
                    break

                content = chunk_data.get('content')
                if content:
                    full_response += content
                    if _config.streaming:
                        await asyncio.sleep(_config.output_delay)
                        yield content, agent_metadata, input_tokens, output_tokens

                usage = chunk_data.get('usage')
                if usage:
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)

            if not _config.streaming and not errored:
                yield full_response, agent_metadata, input_tokens, output_tokens
        except Exception as e:
            if _config.debug:
                print(f"Error in streaming response: {e}")
            errored = True
            yield (f"Sorry, I encountered an error: {e}",
                   agent_metadata, input_tokens, output_tokens)

        # Commit the exchange to history only if it completed cleanly. An errored
        # turn leaves history untouched (no fake assistant turn, no dangling user
        # turn), so the next turn's context isn't poisoned and the user can retry.
        if not errored:
            self.messages.append({"role": "user", "content": user_input})
            self.messages.append({"role": "assistant", "content": full_response})

            max_messages = _config.max_conversation_history
            if len(self.messages) > max_messages:
                self.messages = self.messages[len(self.messages) - max_messages:]
                # The conversation opens with an assistant welcome, so a raw tail
                # can begin on an orphaned assistant turn, which breaks
                # user/assistant pairing and is rejected by some providers. Drop a
                # leading assistant so trimmed history always starts on a user turn.
                if self.messages and self.messages[0]["role"] == "assistant":
                    self.messages = self.messages[1:]

        if _config.streaming:
            yield None, agent_metadata, input_tokens, output_tokens

    async def shutdown(self) -> None:
        if _config.debug:
            print("Shutting down ArtemisCore...")
        self.messages.clear()
        self.agents.clear()
        self._agents_initialized = False
        Agent.shutdown()
        if _config.debug:
            print("ArtemisCore shutdown complete.")
