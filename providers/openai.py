"""OpenAI GPT provider implementation."""

from __future__ import annotations

import os
from typing import Any

from providers.base import LLMProvider, LLMResponse, ProviderError
from providers.models import get_model_by_api_string

# Same per-call output cap as the Anthropic provider, so the per-call output
# limit is not a cross-provider confound. Budget enforcement is handled by the
# executor. For reasoning models, reasoning tokens count inside this cap.
_MAX_API_TOKENS = 16384

# Effort used when thinking_enabled=True (OpenAI's default effort level).
_THINKING_EFFORT = "medium"


def _int_or_zero(value: Any) -> int:
    """Return value if it is an int, else 0 (usage detail fields are optional)."""
    return value if isinstance(value, int) else 0


def _reasoning_effort(model: str, thinking_enabled: bool) -> str | None:
    """Choose the reasoning_effort to send, or None to omit the parameter.

    Registered reasoning models get their lowest supported effort (ideally
    "none", i.e. effectively non-reasoning), or _THINKING_EFFORT when thinking
    is enabled. Unregistered models get no reasoning_effort at all.
    """
    info = get_model_by_api_string(model)
    min_effort = info.get("min_reasoning_effort") if info else None
    if min_effort is None:
        return None
    return _THINKING_EFFORT if thinking_enabled else min_effort


class OpenAIProvider(LLMProvider):
    """Wraps the OpenAI SDK (Chat Completions API) to provide GPT models."""

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
        """Send a completion request to the OpenAI Chat Completions API.

        - Uses max_completion_tokens (max_tokens is deprecated and rejected by
          reasoning models), capped at _MAX_API_TOKENS like the Anthropic provider.
        - For registered reasoning models, sends reasoning_effort (see
          _reasoning_effort). OpenAI rejects temperature when effort is not
          "none", so temperature is then omitted and recorded as None.
        - seed is passed through when given (best-effort determinism, beta).

        output_tokens is usage.completion_tokens, which includes reasoning
        tokens; this matches Anthropic, whose output_tokens includes thinking.
        """
        api_max_tokens = min(max_tokens, _MAX_API_TOKENS)
        effort = _reasoning_effort(model, thinking_enabled)
        sent_temperature = temperature if effort in (None, "none") else None

        kwargs: dict = dict(
            model=model,
            max_completion_tokens=api_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=300.0,
        )
        if sent_temperature is not None:
            kwargs["temperature"] = sent_temperature
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        if seed is not None:
            kwargs["seed"] = seed

        try:
            import openai

            response = await self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as e:
            raise ProviderError("openai", f"Authentication failed: {e}") from e
        except openai.APIError as e:
            raise ProviderError("openai", f"API error: {e}") from e

        if not response.choices:
            raise ProviderError("openai", f"Response contained no choices (model={model})")

        choice = response.choices[0]
        message = choice.message
        text = message.content or ""
        stop_reason = choice.finish_reason or "unknown"
        # A refusal arrives as finish_reason="stop" with message.refusal set;
        # report it the way Anthropic does (stop_reason="refusal").
        if not text and getattr(message, "refusal", None):
            stop_reason = "refusal"

        usage = response.usage
        usage_details: dict[str, int] = {}
        if usage is not None:
            completion_details = getattr(usage, "completion_tokens_details", None)
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            usage_details = {
                "reasoning_tokens": _int_or_zero(
                    getattr(completion_details, "reasoning_tokens", None)
                ),
                "cached_input_tokens": _int_or_zero(
                    getattr(prompt_details, "cached_tokens", None)
                ),
            }

        return LLMResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
            stop_reason=stop_reason,
            raw_response=response,
            request_params={
                "temperature": sent_temperature,
                "max_output_tokens": api_max_tokens,
                "seed": seed,
                "reasoning_effort": effort,
                "thinking_budget_tokens": None,
            },
            usage_details=usage_details,
        )
