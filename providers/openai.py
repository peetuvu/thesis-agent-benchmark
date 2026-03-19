"""OpenAI GPT provider implementation."""

from __future__ import annotations

import os

from providers.base import LLMProvider, LLMResponse, ProviderError


class OpenAIProvider(LLMProvider):
    """Wraps the OpenAI SDK to provide GPT models."""

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("openai", "OPENAI_API_KEY environment variable is not set")

        try:
            import openai
        except ImportError:
            raise ProviderError(
                "openai", "openai SDK is not installed (pip install openai)"
            )

        self._client = openai.AsyncOpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return "openai"

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
        """Send a completion request to the OpenAI API.

        OpenAI supports the `seed` parameter for reproducible outputs
        (beta feature). When provided, it is passed directly to the API.

        Note: thinking_enabled is accepted for interface compatibility but
        ignored. OpenAI o-series models (o3, o3-pro, o4-mini) reason by
        default — no flag needed.
        """
        try:
            import openai

            kwargs: dict = dict(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if seed is not None:
                kwargs["seed"] = seed

            response = await self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as e:
            raise ProviderError("openai", f"Authentication failed: {e}") from e
        except openai.APIError as e:
            raise ProviderError("openai", f"API error: {e}") from e

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
            stop_reason=choice.finish_reason or "unknown",
            raw_response=response,
        )
