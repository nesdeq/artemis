# Agent.py
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from llms.LLMInterface import LLMInterface
import _config


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
        self.llm = LLMInterface(
            _config.agent_llm,
            reasoning_effort=getattr(_config, 'agent_ro', None),
            context=name,
        )
        self.metadata: Dict[str, Any] = {}

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        """Should this agent run? Override in subclasses."""
        return True

    def process(self, user_input: str, last_response: Optional[str] = None) -> Optional[str]:
        """Do the work. Override in subclasses."""
        raise NotImplementedError

    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        if cls._shared_executor is None:
            cls._shared_executor = ThreadPoolExecutor(max_workers=_config.agent_executor_workers)
        return cls._shared_executor

    @classmethod
    def shutdown(cls) -> None:
        """Shutdown shared executor."""
        if cls._shared_executor is not None:
            cls._shared_executor.shutdown(wait=True)
            cls._shared_executor = None

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata
