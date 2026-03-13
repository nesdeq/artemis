"""
Core module for Artemis AI assistant.

This module coordinates agent interactions, handles user queries, and manages context.
"""
import json
import datetime
import asyncio
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator

from litellm import token_counter

import _config
from llms.LLMInterface import LLMInterface
from agents.Agent import Agent
from agents.OnlineSearch import OnlineSearchAgent
from agents.PersonalInfo import PersonalInfoAgent
from agents.ReadURLs import URLReaderAgent
from agents.LangDetect import DetectLanguageAgent
from agents.FileReader import FileReaderAgent
from agents.News import DailyStoriesAgent

class ArtemisCore:
    """
    Core engine for Artemis AI, coordinating agents and handling responses.
    
    Manages conversation context, agent dispatching, and content generation.
    """
    
    def __init__(
        self,
        use_summarization: bool = _config.shallowSummarize,
        user: Optional[str] = None
    ):
        """
        Initialize the Artemis core system.

        Args:
            use_summarization: Whether to summarize agent outputs
            user: Username for personalized interactions
        """
        self.user = user
        self.llm = LLMInterface(context="main")
        self.messages: List[Dict[str, str]] = []
        self.system_prompt = ""
        self.use_summarization = use_summarization
        self.agents = []
        self._agents_initialized = False
        self._init_lock = asyncio.Lock()

        self.update_system_prompt()

    async def _initialize_agents(self) -> None:
        """Initialize all agents asynchronously."""
        async with self._init_lock:
            if self._agents_initialized:
                return

            # Define agent classes and their configurations
            agent_configs = [
                (PersonalInfoAgent, "Personal Info"),
                (DetectLanguageAgent, "Language Detection"),
                (OnlineSearchAgent, "Online Research"),
                (URLReaderAgent, "URL Reader"),
                (FileReaderAgent, "File Reader"),
                (DailyStoriesAgent, "Daily News")
            ]

            for agent_class, name in agent_configs:
                try:
                    agent = await asyncio.to_thread(agent_class, name, self.user)
                    self.agents.append(agent)
                except Exception as e:
                    if _config.debug:
                        print(f"Error initializing agent {name}: {str(e)}")

            self._agents_initialized = True

    async def ensure_agents_initialized(self) -> None:
        """Ensure agents are initialized before proceeding."""
        if not self._agents_initialized:
            await self._initialize_agents()

    def update_system_prompt(self) -> None:
        """Update the system prompt with current context and time."""
        current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        base_prompt = _config.prompt or ""

        self.system_prompt = f"{base_prompt}\nCurrent Time: {current_time}"

    async def _get_snarky_welcome(self) -> str:
        """Get a snarky welcome message from the LLM."""
        import random
        now = datetime.datetime.now()
        time_of_day = "morning" if 5 <= now.hour < 12 else "afternoon" if 12 <= now.hour < 18 else "evening"

        current_time = now.strftime("%A, %B %d, %Y %I:%M %p")
        systemPrompt = "Respond in under 25 words and plain text."
        prompt_options = [
            f"Generate a short, witty, and slightly snarky welcome message. Current time: {current_time}, {time_of_day}.",
            f"Write a brief welcome message with attitude. It's {time_of_day}, {current_time}.",
            f"Create a slightly sarcastic greeting for a CLI interface. Time: {current_time}.",
            f"It's {time_of_day}, {current_time}. Write a welcome message that's both welcoming and a little cheeky."
        ]

        response = await asyncio.to_thread(
            self.llm.generate_single_response,
            prompt=random.choice(prompt_options),
            system_prompt=systemPrompt
        )

        return response.strip().strip('"\'')

    async def ensure_opening_message(self) -> str:
        """Ensure opening message exists as first assistant message. Returns the message."""
        if not self.messages:
            opening_message = await self._get_snarky_welcome()
            self.messages.append({"role": "assistant", "content": opening_message})
            return opening_message
        elif self.messages[0]['role'] == 'assistant':
            return self.messages[0]['content']
        else:
            # Messages exist but first isn't assistant, insert opening message
            opening_message = await self._get_snarky_welcome()
            self.messages.insert(0, {"role": "assistant", "content": opening_message})
            return opening_message

    async def _process_agent(
        self,
        agent: Any,
        input_text: str,
        last_response: Optional[str],
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Process a single agent and return (enrichment, metadata)."""
        try:
            agent_enrichment = await asyncio.to_thread(agent.process, input_text, last_response)
            return agent_enrichment, agent.get_metadata()
        except Exception as e:
            if _config.debug:
                print(f"Error processing agent {agent.name}: {str(e)}")
            return None, None

    async def _process_all_agents(
        self,
        input_text: str,
        last_response: Optional[str],
    ) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
        """Process agents in parallel: first check should_process, then process active ones."""
        await self.ensure_agents_initialized()

        # Phase 1: Check which agents should run (parallel)
        decisions = await asyncio.gather(*[
            asyncio.to_thread(agent.should_process, input_text, last_response)
            for agent in self.agents
        ])

        active_agents = [agent for agent, should in zip(self.agents, decisions) if should]

        if _config.debug:
            print(f"Active agents: {[a.name for a in active_agents]}")

        # Phase 2: Process only active agents (parallel)
        tasks = [
            (agent.name, asyncio.create_task(
                self._process_agent(agent, input_text, last_response)
            ))
            for agent in active_agents
        ]

        agent_outputs = {}
        agent_metadata = {}

        for agent_name, task in tasks:
            try:
                enrichment, metadata = await task
                if enrichment is not None:
                    agent_outputs[agent_name] = enrichment
                    agent_metadata[agent_name] = metadata
            except Exception as e:
                if _config.debug:
                    print(f"Error awaiting agent {agent_name}: {e}")

        return agent_outputs, agent_metadata

    def _format_enriched_context(self, agent_outputs: Dict[str, str]) -> str:
        """
        Format agent outputs into a unified context string.

        Args:
            agent_outputs: Dictionary of agent name to output text

        Returns:
            Formatted context string
        """
        if not agent_outputs:
            return ""

        context_parts = ["\n\nAgent context (use minimally for basic greetings/small talk, moderately for topical discussion, fully for complex queries):\n"]

        for agent_name, output in agent_outputs.items():
            if not output.strip():
                continue

            if self.use_summarization:
                try:
                    summarized = self.llm.summarize(output)
                    context_parts.append(f"\n[{agent_name}::\n{summarized}]\n")
                except Exception as e:
                    if _config.debug:
                        print(f"Error summarizing output from {agent_name}: {str(e)}")
                    context_parts.append(f"\n[{agent_name}::\n{output}]\n")
            else:
                context_parts.append(f"\n[{agent_name}::\n{output}]\n")

        return "".join(context_parts)

    async def get_ai_response(
        self,
        user_input: str,
    ) -> AsyncGenerator[Tuple[Optional[str], Dict[str, Any], int, int], None]:
        """
        Process user input and generate AI response.

        Yields:
            Tuples of (text_chunk, agent_metadata, input_tokens, output_tokens)
            Final yield has chunk=None with final token counts.
        """
        dated_input = f"{user_input}\nTimestamp: {datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"

        last_response = None
        if self.messages and self.messages[-1]['role'] == 'assistant':
            last_response = self.messages[-1]['content']

        agent_outputs, agent_metadata = await self._process_all_agents(dated_input, last_response)
        enriched_context = self._format_enriched_context(agent_outputs)

        self.update_system_prompt()

        user_message = {
            "role": "user",
            "content": f"{dated_input}\n\n{enriched_context}" if enriched_context else dated_input
        }
        temp_messages = self.messages + [user_message]

        if _config.debug:
            print(json.dumps(temp_messages, indent=2))

        input_tokens = 0
        output_tokens = 0
        full_response = ""

        try:
            for chunk_data in self.llm.stream_content(
                messages=temp_messages,
                system_prompt=self.system_prompt,
                max_tokens=_config.max_tokens
            ):
                chunk = chunk_data.get('content', '')
                full_response += chunk
                if _config.streaming:
                    await asyncio.sleep(_config.output_delay)
                    yield chunk, agent_metadata, input_tokens, output_tokens

            input_tokens = token_counter(model=self.llm.model, messages=temp_messages)
            output_tokens = token_counter(model=self.llm.model, text=full_response)
            self.llm.record_usage(input_tokens, output_tokens)

            if not _config.streaming:
                yield full_response, agent_metadata, input_tokens, output_tokens
        except Exception as e:
            if _config.debug:
                print(f"Error in streaming response: {str(e)}")
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            yield error_msg, agent_metadata, input_tokens, output_tokens
            full_response = error_msg

        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": full_response})

        # Trim conversation history
        max_messages = _config.max_conversation_history
        if len(self.messages) > max_messages:
            excess = len(self.messages) - max_messages
            excess += excess % 2  # Round up to even for user/assistant pairs
            self.messages = self.messages[excess:]

        if _config.streaming:
            yield None, agent_metadata, input_tokens, output_tokens

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the ArtemisCore instance.

        Cleans up agents, closes connections, and releases resources.
        """
        if _config.debug:
            print("Shutting down ArtemisCore...")

        # Clear conversation history
        self.messages.clear()

        # Clear agent list
        self.agents.clear()
        self._agents_initialized = False

        # Shutdown shared agent resources (executor, HTTP session)
        Agent.shutdown()

        if _config.debug:
            print("ArtemisCore shutdown complete.")

