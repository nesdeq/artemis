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
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.model = model or _config.llm
        self.timeout = timeout if timeout is not None else _config.llm_timeout
        self.num_retries = retry_attempts if retry_attempts is not None else _config.llm_retry_attempts
        self.streaming = _config.streaming
        self.reasoning_effort = reasoning_effort or _config.ro
        self.context = context or "main"
        # Prepended to generate_single_response / summarize calls that don't pass
        # their own system_prompt. Agents set this to _config.agent_shared_context
        # so every agent call carries the same operating frame.
        self.default_system_prompt = system_prompt

        self._is_reasoning_model = self._check_reasoning_support()

        logger.debug(
            f"LLMInterface initialized: model={self.model}, context={self.context}, "
            f"streaming={self.streaming}, reasoning_effort={self.reasoning_effort}, "
            f"is_reasoning_model={self._is_reasoning_model}"
        )

    def _check_reasoning_support(self) -> bool:
        try:
            return supports_reasoning(model=self.model)
        except Exception:
            model_lower = self.model.lower()
            return any(p in model_lower for p in ['o1', 'o3', 'o4', 'gpt-5', 'claude-sonnet-4', 'claude-opus-4'])

    def _adjust_tokens_for_reasoning(self, max_tokens: Optional[int]) -> Optional[int]:
        if not self._is_reasoning_model or max_tokens is None:
            return max_tokens
        floor = _config.reasoning_model_min_tokens
        if max_tokens < floor:
            logger.debug(f"Adjusting max_tokens {max_tokens} -> {floor} for {self.model}")
            return floor
        return max_tokens

    def _call(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
    ):
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        use_stream = self.streaming if stream is None else stream
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=trim_messages(messages=messages, model=self.model),
            max_tokens=self._adjust_tokens_for_reasoning(max_tokens),
            temperature=temperature,
            reasoning_effort=self.reasoning_effort,
            stream=use_stream,
            timeout=self.timeout,
            num_retries=self.num_retries,
            drop_params=True,
        )
        if use_stream:
            # Ask the provider to emit a final usage chunk so we don't have to
            # reconstruct token counts client-side.
            kwargs["stream_options"] = {"include_usage": True}

        try:
            return completion(**kwargs)
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise LLMInterfaceError(f"Completion failed: {e}") from e

    def stream_content(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream content chunks.

        Yields {'content': str} for each text delta.
        Yields {'usage': {'input_tokens': N, 'output_tokens': M}} once per call
        when the provider returns final usage (typically at end-of-stream).
        Yields {'error': str} if streaming fails mid-flight, so the caller can
        surface the failure without mistaking it for model output.
        Records usage automatically against this interface's cost context.
        """
        try:
            response = self._call(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )

            if self.streaming:
                for chunk in response:
                    # Final usage chunk: choices is empty, usage is populated.
                    usage = getattr(chunk, 'usage', None)
                    if usage:
                        in_tok = getattr(usage, 'prompt_tokens', 0) or 0
                        out_tok = getattr(usage, 'completion_tokens', 0) or 0
                        if in_tok or out_tok:
                            self.record_usage(in_tok, out_tok)
                            yield {'usage': {'input_tokens': in_tok, 'output_tokens': out_tok}}

                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    content = getattr(delta, 'content', None)
                    if content:
                        yield {'content': content}
            else:
                if response.choices:
                    content = getattr(response.choices[0].message, 'content', '') or ''
                    yield {'content': content}
                else:
                    yield {'content': ''}

                usage = getattr(response, 'usage', None)
                if usage:
                    in_tok = getattr(usage, 'prompt_tokens', 0) or 0
                    out_tok = getattr(usage, 'completion_tokens', 0) or 0
                    if in_tok or out_tok:
                        self.record_usage(in_tok, out_tok)
                        yield {'usage': {'input_tokens': in_tok, 'output_tokens': out_tok}}

        except LLMInterfaceError:
            raise
        except Exception as e:
            logger.error(f"Error in stream_content: {e}")
            yield {'error': str(e)}

    def generate_single_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a single complete response (non-streaming).

        Falls back to this interface's default_system_prompt (the shared agent
        context, for agent instances) when the caller passes none.
        """
        response = self._call(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt if system_prompt is not None else self.default_system_prompt,
            max_tokens=max_tokens,
            stream=False,
            temperature=temperature,
        )

        if not response.choices:
            return ""

        if hasattr(response, 'usage') and response.usage:
            self.record_usage(
                getattr(response.usage, 'prompt_tokens', 0),
                getattr(response.usage, 'completion_tokens', 0)
            )

        content = getattr(response.choices[0].message, 'content', '') or ''
        return content.strip()

    def summarize(self, text: str, max_words: Optional[int] = None,
                  temperature: float = 0.3) -> str:
        """Summarize text content."""
        if max_words is None:
            max_words = _config.default_summary_words
        prompt = (
            f"Summarize the following text in under {max_words} words while keeping the most "
            f"important information. Focus on facts and key points.\n\nText to summarize:\n{text}"
        )
        # Reasoning models (the default) reject temperature; only pass it to
        # non-reasoning models. drop_params would strip it anyway — be explicit.
        return self.generate_single_response(
            prompt=prompt,
            max_tokens=max_words * _config.summary_tokens_per_word,
            temperature=None if self._is_reasoning_model else temperature,
        )

    # =========================================================================
    # Cost Tracking
    # =========================================================================

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage and calculate costs for this context."""
        if input_tokens == 0 and output_tokens == 0:
            return

        try:
            input_cost, output_cost = cost_per_token(
                model=self.model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
        except Exception as e:
            logger.debug(f"cost_per_token unavailable for {self.model}: {e}")
            input_cost = output_cost = 0.0

        with self._cost_lock:
            entry = self._session_costs.setdefault(self.context, {
                'model': self.model,
                'input_tokens': 0,
                'output_tokens': 0,
                'input_cost': 0.0,
                'output_cost': 0.0,
            })
            entry['input_tokens'] += input_tokens
            entry['output_tokens'] += output_tokens
            entry['input_cost'] += input_cost
            entry['output_cost'] += output_cost

    @classmethod
    def get_session_costs(cls) -> Dict[str, Dict[str, Any]]:
        """Get cost breakdown for all contexts in this session."""
        with cls._cost_lock:
            return {k: v.copy() for k, v in cls._session_costs.items()}

    @classmethod
    def get_total_cost(cls) -> Tuple[int, int, float, float]:
        """Total session costs across all contexts: (in_tokens, out_tokens, in_cost, out_cost)."""
        with cls._cost_lock:
            costs = cls._session_costs.values()
            return (
                sum(c['input_tokens'] for c in costs),
                sum(c['output_tokens'] for c in costs),
                sum(c['input_cost'] for c in costs),
                sum(c['output_cost'] for c in costs),
            )
