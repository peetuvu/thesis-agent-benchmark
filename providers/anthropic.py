"""Anthropic Claude provider implementation."""

from __future__ import annotations

import os

from providers.base import LLMProvider, LLMResponse, ProviderError
from providers.models import check_reasoning_effort

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
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """Send a completion request to the Anthropic API.

        Note: The Anthropic API does not currently support a `seed` parameter.
        The parameter is accepted for interface compatibility but is not passed
        to the API. Temperature=0.0 provides near-deterministic output.

        Two request shapes:

        - reasoning_effort set (required for the 5.5 generation): sends
          thinking={"type": "adaptive"} and output_config={"effort": ...}, and
          NO temperature — Claude 5.5 models reject temperature/top_p/top_k and
          thinking.budget_tokens with HTTP 400.
        - reasoning_effort None (legacy models, unchanged): temperature via
          extra_body; when thinking_enabled=True, extended thinking with a
          budget of 10000 tokens and temperature forced to 1.0.

        SDK 1.x removed temperature from the messages.create() signature but
        still sends anthropic-version 2023-06-01, where the API accepts it, so
        it is passed via extra_body. Do not upgrade to SDK 2.x: it sends
        2026-10-21, which rejects temperature with HTTP 400.
        """
        try:
            check_reasoning_effort(model, reasoning_effort, thinking_enabled)
        except ValueError as e:
            raise ProviderError("anthropic", str(e)) from e

        try:
            import anthropic

            # Cap max_tokens to avoid SDK ValueError requiring streaming
            # for large values. Budget enforcement is handled by the executor.
            # Adaptive thinking tokens count toward this cap.
            api_max_tokens = min(max_tokens, _MAX_API_TOKENS)

            kwargs: dict = dict(
                model=model,
                max_tokens=api_max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=anthropic.Timeout(300.0, connect=5.0),
            )

            if reasoning_effort is not None:
                kwargs["thinking"] = {"type": "adaptive"}
                kwargs["output_config"] = {"effort": reasoning_effort}
                sent_temperature = None
            elif thinking_enabled:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}
                # Anthropic requires temperature=1.0 when thinking is enabled
                sent_temperature = 1.0
            else:
                sent_temperature = temperature
            if sent_temperature is not None:
                kwargs["extra_body"] = {"temperature": sent_temperature}

            response = await self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise ProviderError("anthropic", f"Authentication failed: {e}") from e
        except anthropic.APIError as e:
            raise ProviderError("anthropic", f"API error: {e}") from e

        # Extract text from content blocks (thinking blocks have no .text)
        text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )

        thinking = kwargs.get("thinking")
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            stop_reason=response.stop_reason,
            raw_response=response,
            request_params={
                "temperature": sent_temperature,
                "max_output_tokens": api_max_tokens,
                "seed": None,  # not supported by the Anthropic API
                "reasoning_effort": reasoning_effort,
                "thinking_budget_tokens": thinking.get("budget_tokens") if thinking else None,
                "thinking_type": thinking["type"] if thinking else None,
                "requested_reasoning_effort": reasoning_effort,
                "sent_params": _sent_params(kwargs),
            },
        )


def _sent_params(kwargs: dict) -> dict:
    """Request body fields sent to the API, minus model, prompts and client options.

    extra_body fields are merged in, since that is how they reach the wire.
    """
    excluded = {"model", "system", "messages", "timeout", "extra_body"}
    sent = {k: v for k, v in kwargs.items() if k not in excluded}
    sent.update(kwargs.get("extra_body") or {})
    return sent
