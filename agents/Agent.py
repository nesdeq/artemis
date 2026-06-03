"""
Agent base class.

================================ ARCHITECTURE LAWS ================================
These are non-negotiable. Don't optimize past them. Don't compromise on them.
The canonical text lives in agents/_Agents.md — these MUST stay in sync.

1. THE LLM JUDGES. ALWAYS.
   When an agent's job requires judgement — is this English? does this need a
   web search? is this a memorable fact? — the LLM makes that judgement,
   EVERY turn. Routing-by-text-interpretation is forbidden. Heuristics cannot
   replace judgement.

2. TWO-CALL PATTERN: ONE DECISION, ONE EXECUTION.
   - should_process is the DECISION LLM call: "should this agent contribute?"
   - process       is the EXECUTION LLM call: "do the work."
   These are SEPARATE calls. NEVER merged into one.

3. EXECUTION MAY USE MULTIPLE LLM CALLS.
   When execution has complex subtasks (plan + per-result refinement, batched
   extraction across items), multiple LLM calls inside process() are fine.
   The DECISION is always its own single call, separate from execution.

4. NO CALL-COUNT OPTIMIZATION. EVER.
   - No caching of LLM output across turns.
   - No heuristic substitution for LLM judgement.
   - No merging two LLM decisions into one to "save a call".
   The call count IS the architecture. Cost is not your problem; correctness is.

5. CHEAP PRE-FILTERS BEFORE THE DECISION CALL ARE ALLOWED — WHEN BINARY.
   If the answer is structurally yes/no (URL present? bang command? file on
   disk?), a regex / filesystem check may short-circuit should_process BEFORE
   the decision LLM call. It does NOT replace the decision call when the
   question requires judgement.
====================================================================================
"""
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from llms.LLMInterface import LLMInterface
import _config


class Agent:
    """Base class for all agents.

    Subclasses MUST follow the architecture laws above.

    Override:
    - should_process() -> bool: the DECISION call (binary pre-filter and/or LLM).
    - process() -> Optional[str]: the EXECUTION call (LLM, may use sub-calls).
    """

    _shared_executor: Optional[ThreadPoolExecutor] = None

    def __init__(self, name: Optional[str] = None, user: Optional[str] = None) -> None:
        self.name = name
        self.user = user
        self.llm = LLMInterface(
            _config.agent_llm,
            reasoning_effort=_config.agent_ro,
            context=name,
        )
        # Every agent MUST populate this. Surfaced in the CLI sources panel.
        self.metadata: Dict[str, Any] = {}

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        """DECISION call. Override in subclasses. Must not merge with process()."""
        return True

    def process(self, user_input: str, last_response: Optional[str] = None) -> Optional[str]:
        """EXECUTION call. Override in subclasses. May invoke multiple LLM sub-calls."""
        raise NotImplementedError

    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        # Bind to Agent, not cls: using cls would give each concrete subclass its
        # own pool (cls._shared_executor resolves per-subclass), multiplying
        # thread pools by the number of agents and defeating the shared design.
        if Agent._shared_executor is None:
            Agent._shared_executor = ThreadPoolExecutor(max_workers=_config.agent_executor_workers)
        return Agent._shared_executor

    @classmethod
    def shutdown(cls) -> None:
        """Shutdown the one shared executor."""
        if Agent._shared_executor is not None:
            Agent._shared_executor.shutdown(wait=True)
            Agent._shared_executor = None

    def get_metadata(self) -> Dict[str, Any]:
        """Read by core.py after process(). Surfaced in CLI sources panel."""
        return self.metadata
