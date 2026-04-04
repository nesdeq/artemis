"""
LLM Interface Module

Thin wrapper around LiteLLM - delegates model detection, parameter translation,
and retry logic to LiteLLM's native capabilities.
"""
from typing import List, Dict, Optional, Generator, Any, Tuple
import logging
import threading
import warnings

# Suppress Pydantic serialization warnings from LiteLLM streaming responses
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

from litellm import completion, supports_reasoning, cost_per_token
from litellm.utils import trim_messages
import _config

logger = logging.getLogger(__name__)

# Minimum tokens for reasoning models to prevent starvation
# Reasoning tokens (invisible) count against max_completion_tokens
# Too low = empty responses as all tokens go to reasoning
REASONING_MODEL_MIN_TOKENS = 2048


class LLMInterfaceError(Exception):
    """Exception for LLM interface errors"""
    pass


class LLMInterface:
    """
    Lightweight interface for LLM interactions via LiteLLM.

    LiteLLM handles automatically:
    - Model type detection and parameter adaptation
    - max_tokens vs max_completion_tokens translation
    - reasoning_effort to Claude thinking/budget_tokens mapping
    - Unsupported parameter removal (via drop_params)
    - Retry with exponential backoff (via num_retries)
    - Response normalization across providers

    Cost Tracking:
    - Class-level tracking of token usage and costs per context
    - Use get_session_costs() to retrieve breakdown
    - Use reset_session_costs() to clear tracking
    """

    # Class-level cost tracking: {context: {model, input_tokens, output_tokens, input_cost, output_cost}}
    _session_costs: Dict[str, Dict[str, Any]] = {}
    _cost_lock = threading.Lock()

    def __init__(
        self,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        retry_attempts: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        context: Optional[str] = None
    ):
        """
        Initialize the LLM interface.

        Args:
            model: LLM model identifier, defaults to _config.llm
            timeout: Request timeout in seconds, defaults to _config.llm_timeout
            retry_attempts: Retry count for failed calls, defaults to _config.llm_retry_attempts
            reasoning_effort: Reasoning effort level (low/medium/high), defaults to _config.ro
            context: Identifier for cost tracking (e.g., "main" or agent name)
        """
        self.model = model or _config.llm
        self.timeout = timeout if timeout is not None else _config.llm_timeout
        self.num_retries = retry_attempts if retry_attempts is not None else _config.llm_retry_attempts
        self.streaming = _config.streaming
        self.reasoning_effort = reasoning_effort or getattr(_config, 'ro', None)
        self.context = context or "main"

        # Check if model supports reasoning (for token floor logic)
        self._is_reasoning_model = self._check_reasoning_support()

        logger.debug(
            f"LLMInterface initialized: model={self.model}, context={self.context}, "
            f"streaming={self.streaming}, reasoning_effort={self.reasoning_effort}, "
            f"is_reasoning_model={self._is_reasoning_model}"
        )

    def _check_reasoning_support(self) -> bool:
        """Check if model supports reasoning using LiteLLM's API."""
        try:
            return supports_reasoning(model=self.model)
        except Exception:
            # Fallback: check model name patterns
            model_lower = self.model.lower()
            return any(p in model_lower for p in ['o1', 'o3', 'o4', 'gpt-5', 'claude-sonnet-4', 'claude-opus-4'])

    def _adjust_tokens_for_reasoning(self, max_tokens: Optional[int]) -> Optional[int]:
        """
        Ensure sufficient tokens for reasoning models.

        Reasoning models use tokens for internal thinking (invisible) plus output.
        If max_tokens is too low, all tokens go to reasoning = empty response.
        """
        if not self._is_reasoning_model or max_tokens is None:
            return max_tokens

        if max_tokens < REASONING_MODEL_MIN_TOKENS:
            logger.debug(
                f"Adjusting max_tokens from {max_tokens} to {REASONING_MODEL_MIN_TOKENS} "
                f"for reasoning model {self.model}"
            )
            return REASONING_MODEL_MIN_TOKENS

        return max_tokens

    def _call(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None
    ):
        """
        Call LiteLLM completion with automatic parameter handling.

        LiteLLM automatically:
        - Translates max_tokens to max_completion_tokens for reasoning models
        - Maps reasoning_effort to Claude thinking/budget_tokens
        - Drops unsupported params (e.g., temperature on o-series)
        """
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        use_stream = stream if stream is not None else self.streaming
        effort = reasoning_effort or self.reasoning_effort

        # Ensure reasoning models have enough tokens to avoid starvation
        adjusted_tokens = self._adjust_tokens_for_reasoning(max_tokens)

        try:
            return completion(
                model=self.model,
                messages=trim_messages(messages=messages, model=self.model),
                max_tokens=adjusted_tokens,
                temperature=temperature,
                reasoning_effort=effort,
                stream=use_stream,
                timeout=self.timeout,
                num_retries=self.num_retries,
                drop_params=True,  # Auto-remove unsupported params per model
            )
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise LLMInterfaceError(f"Completion failed: {e}") from e

    def stream_content(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        include_reasoning: bool = False
    ) -> Generator[Dict[str, Optional[str]], None, None]:
        """
        Stream content from LLM response.

        Args:
            messages: List of message dictionaries
            system_prompt: Optional system prompt to prepend
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (auto-dropped if unsupported)
            reasoning_effort: Override reasoning effort level
            include_reasoning: Include reasoning_content in output

        Yields:
            Dict with 'content' and optionally 'reasoning_content' keys
        """
        try:
            response = self._call(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort
            )

            if self.streaming:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    result = {}
                    if getattr(delta, 'content', None):
                        result['content'] = delta.content
                    if include_reasoning and getattr(delta, 'reasoning_content', None):
                        result['reasoning_content'] = delta.reasoning_content
                    if result:
                        yield result
            else:
                # Non-streaming: yield complete response
                if response.choices:
                    msg = response.choices[0].message
                    result = {}
                    if getattr(msg, 'content', None):
                        result['content'] = msg.content
                    if include_reasoning and getattr(msg, 'reasoning_content', None):
                        result['reasoning_content'] = msg.reasoning_content
                    if result:
                        yield result
                    else:
                        yield {'content': ''}
                else:
                    yield {'content': ''}

        except LLMInterfaceError:
            raise
        except Exception as e:
            logger.error(f"Error in stream_content: {e}")
            yield {'content': f"\n[Error: {e}]"}

    def generate_single_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        include_reasoning: bool = False
    ) -> str:
        """
        Generate a single complete response (non-streaming).

        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (auto-dropped if unsupported)
            reasoning_effort: Override reasoning effort level
            include_reasoning: Append reasoning_content to response

        Returns:
            Generated text response
        """
        messages = [{"role": "user", "content": prompt}]

        response = self._call(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            stream=False,
            temperature=temperature,
            reasoning_effort=reasoning_effort
        )

        if not response.choices:
            return ""

        # Record token usage for cost tracking
        if hasattr(response, 'usage') and response.usage:
            self.record_usage(
                getattr(response.usage, 'prompt_tokens', 0),
                getattr(response.usage, 'completion_tokens', 0)
            )

        msg = response.choices[0].message
        content = getattr(msg, 'content', '') or ''
        reasoning = getattr(msg, 'reasoning_content', None)

        if include_reasoning and reasoning:
            return f"{content.strip()}\n\n[Reasoning: {reasoning}]"

        return content.strip()

    def summarize(self, text: str, max_words: int = 500, temperature: float = 0.3) -> str:
        """
        Summarize text content.

        Args:
            text: Text to summarize
            max_words: Maximum words in summary
            temperature: Sampling temperature (auto-dropped if unsupported)

        Returns:
            Summarized text
        """
        prompt = f"""Summarize the following text in under {max_words} words while keeping the most important information.
        Focus on facts and key points.

        Text to summarize:
        {text}"""

        return self.generate_single_response(
            prompt=prompt,
            max_tokens=max_words * 2,
            temperature=temperature
        )

    # =========================================================================
    # Cost Tracking
    # =========================================================================

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        """
        Record token usage and calculate costs for this context.

        Args:
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
        """
        if input_tokens == 0 and output_tokens == 0:
            return

        # Calculate costs using LiteLLM's pricing
        try:
            input_cost, output_cost = cost_per_token(
                model=self.model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens
            )
        except Exception:
            input_cost = output_cost = 0.0

        with self._cost_lock:
            if self.context not in self._session_costs:
                self._session_costs[self.context] = {
                    'model': self.model,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'input_cost': 0.0,
                    'output_cost': 0.0,
                }

            self._session_costs[self.context]['input_tokens'] += input_tokens
            self._session_costs[self.context]['output_tokens'] += output_tokens
            self._session_costs[self.context]['input_cost'] += input_cost
            self._session_costs[self.context]['output_cost'] += output_cost

    @classmethod
    def get_session_costs(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get cost breakdown for all contexts in this session.

        Returns:
            Dict mapping context name to cost details:
            {
                "main": {"model": "...", "input_tokens": N, "output_tokens": N,
                         "input_cost": $, "output_cost": $},
                "Agent Name": {...},
                ...
            }
        """
        with cls._cost_lock:
            return {k: v.copy() for k, v in cls._session_costs.items()}

    @classmethod
    def get_total_cost(cls) -> Tuple[int, int, float, float]:
        """
        Get total session costs across all contexts.

        Returns:
            Tuple of (total_input_tokens, total_output_tokens, total_input_cost, total_output_cost)
        """
        with cls._cost_lock:
            total_in_tokens = sum(c['input_tokens'] for c in cls._session_costs.values())
            total_out_tokens = sum(c['output_tokens'] for c in cls._session_costs.values())
            total_in_cost = sum(c['input_cost'] for c in cls._session_costs.values())
            total_out_cost = sum(c['output_cost'] for c in cls._session_costs.values())
            return total_in_tokens, total_out_tokens, total_in_cost, total_out_cost

    @classmethod
    def reset_session_costs(cls) -> None:
        """Clear all session cost tracking."""
        with cls._cost_lock:
            cls._session_costs.clear()
