"""Abstract base class for LLM providers and standardized response type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# Provider-native stop reasons meaning "generation ended normally".
# Anything else (e.g. 'max_tokens', 'length', 'refusal', 'content_filter')
# is surfaced in the run log's stop_reason.
NORMAL_STOP_REASONS: frozenset[str] = frozenset({"end_turn", "stop_sequence", "stop"})

# Provider-native stop reasons meaning "output was cut off at the token cap".
TRUNCATION_STOP_REASONS: frozenset[str] = frozenset({"max_tokens", "length"})


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider.

    Every provider maps its SDK-specific response into this dataclass
    so that architectures never depend on provider internals.
    """

    text: str
    """The generated text content."""

    input_tokens: int
    """Number of tokens in the prompt."""

    output_tokens: int
    """Number of tokens in the completion."""

    model: str
    """Actual model string used (as reported by the provider)."""

    stop_reason: str
    """Why generation stopped (e.g. 'end_turn', 'max_tokens', 'stop')."""

    raw_response: Any = None
    """Original SDK response object, kept for debugging. Not serialized."""

    request_params: dict[str, Any] = field(default_factory=dict)
    """Sampling parameters actually sent to the API, after provider adjustments.

    Standard keys: temperature, max_output_tokens, seed, reasoning_effort,
    thinking_budget_tokens. A value of None means the parameter was not sent
    (the provider's default applied).
    """

    usage_details: dict[str, int] = field(default_factory=dict)
    """Provider-specific token breakdowns (e.g. reasoning_tokens, cached_input_tokens).

    These are subsets of input_tokens/output_tokens, not additions to them.
    """

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens


class LLMProvider(ABC):
    """Abstract interface that all LLM providers must implement.

    Architectures call `provider.complete(...)` and receive an `LLMResponse`.
    They never import or interact with provider SDKs directly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'anthropic', 'openai')."""
        ...

    @abstractmethod
    async def complete(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int | None = None,
        thinking_enabled: bool = False,
    ) -> LLMResponse:
        """Send a completion request and return a standardized response.

        Args:
            model: Model identifier (e.g. 'claude-sonnet-4-6', 'gpt-4o').
            system_prompt: System-level instructions.
            user_prompt: The user message / task prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).
            seed: Random seed for reproducibility (provider support varies).
            thinking_enabled: Enable extended thinking / reasoning mode.

        Returns:
            LLMResponse with text, token counts, and metadata.

        Raises:
            ProviderError: On authentication failure, rate limits, or API errors.
        """
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


class ProviderError(Exception):
    """Raised when a provider encounters an API error."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")
