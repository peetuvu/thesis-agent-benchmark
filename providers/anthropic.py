"""Anthropic Claude provider implementation."""

from __future__ import annotations

import os

from providers.base import LLMProvider, LLMResponse, ProviderError


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
    ) -> LLMResponse:
        """Send a completion request to the Anthropic API.

        Note: The Anthropic API does not currently support a `seed` parameter.
        The parameter is accepted for interface compatibility but is not passed
        to the API. Temperature=0.0 provides near-deterministic output.
        """
        try:
            import anthropic

            response = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
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
