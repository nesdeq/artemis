# Agent.py
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from llms.LLMInterface import LLMInterface
import _config
from tools.utils import clean_html


class Agent:
    """Base class for all agents.

    Override:
    - should_process() -> bool: Should this agent run?
    - process() -> Optional[str]: Do the work.
    """

    _shared_executor: Optional[ThreadPoolExecutor] = None

    def __init__(self, name: Optional[str] = None, user: Optional[str] = None) -> None:
        self.name = name
        self.user = user
        agent_ro = getattr(_config, 'agent_ro', None)
        self.llm = LLMInterface(_config.agent_llm, reasoning_effort=agent_ro, context=name)
        self.metadata: Dict[str, Any] = {}

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        """Should this agent run? Override in subclasses."""
        return True

    def process(self, user_input: str, last_response: Optional[str] = None) -> Optional[str]:
        """Do the work. Override in subclasses."""
        raise NotImplementedError

    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        if Agent._shared_executor is None:
            Agent._shared_executor = ThreadPoolExecutor(max_workers=10)
        return Agent._shared_executor

    @classmethod
    def shutdown(cls) -> None:
        """Shutdown shared executor."""
        if Agent._shared_executor is not None:
            Agent._shared_executor.shutdown(wait=True)
            Agent._shared_executor = None

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

    def clean_html(self, html_content: str) -> str:
        return clean_html(html_content)

    def summarize_content(self, content: str, max_words: int = 500) -> str:
        return self.llm.summarize(content, max_words=max_words)
