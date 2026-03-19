"""Anthropic Claude provider implementation."""

from __future__ import annotations

import os

import httpx

from providers.base import LLMProvider, LLMResponse, ProviderError

_MAX_API_TOKENS = 16384


class AnthropicProvider(LLMProvider):
    """Wraps the Anthropic SDK to provide Claude models."""

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("anthropic", "ANTHROPIC_API_KEY environment variable is not set")

        try:
            import anthropic
        except ImportError:
            raise ProviderError(
                "anthropic", "anthropic SDK is not installed (pip install anthropic)"
            )

        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "anthropic"

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
        """Send a completion request to the Anthropic API.

        Note: The Anthropic API does not currently support a `seed` parameter.
        The parameter is accepted for interface compatibility but is not passed
        to the API. Temperature=0.0 provides near-deterministic output.

        When thinking_enabled=True, extended thinking is activated with a
        budget of 10000 tokens. Temperature must be 1.0 for thinking mode.
        """
        try:
            import anthropic

            # Cap max_tokens to avoid SDK ValueError requiring streaming
            # for large values. Budget enforcement is handled by the executor.
            api_max_tokens = min(max_tokens, _MAX_API_TOKENS)

            kwargs: dict = dict(
                model=model,
                max_tokens=api_max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=httpx.Timeout(300.0, connect=5.0),
            )

            if thinking_enabled:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}
                # Anthropic requires temperature=1.0 when thinking is enabled
                kwargs["temperature"] = 1.0
            else:
                kwargs["temperature"] = temperature

            response = await self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise ProviderError("anthropic", f"Authentication failed: {e}") from e
        except anthropic.APIError as e:
            raise ProviderError("anthropic", f"API error: {e}") from e

        # Extract text from content blocks
        text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )

        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            stop_reason=response.stop_reason,
            raw_response=response,
        )
