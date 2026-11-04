"""Tests for the provider abstraction layer.

All tests use mocks — no real API calls are made.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers import get_provider
from providers.base import LLMProvider, LLMResponse, ProviderError


# ---------------------------------------------------------------------------
# LLMResponse dataclass
# ---------------------------------------------------------------------------


class TestLLMResponse:
    def test_creation(self) -> None:
        resp = LLMResponse(
            text="Hello, world!",
            input_tokens=10,
            output_tokens=5,
            model="test-model",
            stop_reason="end_turn",
        )
        assert resp.text == "Hello, world!"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.model == "test-model"
        assert resp.stop_reason == "end_turn"
        assert resp.raw_response is None

    def test_total_tokens_property(self) -> None:
        resp = LLMResponse(
            text="hi",
            input_tokens=100,
            output_tokens=50,
            model="m",
            stop_reason="stop",
        )
        assert resp.total_tokens == 150

    def test_raw_response_preserved(self) -> None:
        raw = {"some": "object"}
        resp = LLMResponse(
            text="x",
            input_tokens=1,
            output_tokens=1,
            model="m",
            stop_reason="stop",
            raw_response=raw,
        )
        assert resp.raw_response is raw


# ---------------------------------------------------------------------------
# get_provider registry
# ---------------------------------------------------------------------------


class TestGetProvider:
    def test_returns_anthropic_provider(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = get_provider("anthropic")
            assert isinstance(provider, LLMProvider)
            assert provider.name == "anthropic"

    def test_returns_openai_provider(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = get_provider("openai")
            assert isinstance(provider, LLMProvider)
            assert provider.name == "openai"

    def test_raises_for_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider 'nonexistent'"):
            get_provider("nonexistent")

    def test_error_message_lists_available_providers(self) -> None:
        with pytest.raises(ValueError, match="anthropic"):
            get_provider("bogus")


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    def test_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            # Remove the key if it exists
            import os
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
                get_provider("anthropic")

    def test_complete_returns_llm_response(self) -> None:
        """Mock the Anthropic SDK and verify field mapping."""
        # Build a mock response matching Anthropic's structure
        mock_text_block = MagicMock()
        mock_text_block.text = "Generated code here"

        mock_usage = MagicMock()
        mock_usage.input_tokens = 200
        mock_usage.output_tokens = 100

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.usage = mock_usage
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = get_provider("anthropic")

            # Patch the client's messages.create to return our mock
            provider._client.messages.create = AsyncMock(return_value=mock_response)

            result = asyncio.get_event_loop().run_until_complete(
                provider.complete(
                    model="claude-sonnet-4-20250514",
                    system_prompt="You are helpful.",
                    user_prompt="Write hello world.",
                )
            )

        assert isinstance(result, LLMResponse)
        assert result.text == "Generated code here"
        assert result.input_tokens == 200
        assert result.output_tokens == 100
        assert result.total_tokens == 300
        assert result.model == "claude-sonnet-4-20250514"
        assert result.stop_reason == "end_turn"
        assert result.raw_response is mock_response


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    def test_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
                get_provider("openai")

    def test_complete_returns_llm_response(self) -> None:
        """Mock the OpenAI SDK and verify field mapping."""
        # Build a mock response matching OpenAI's structure
        mock_message = MagicMock()
        mock_message.content = "def hello(): print('hi')"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 150
        mock_usage.completion_tokens = 80

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "gpt-4o"

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = get_provider("openai")

            # Patch the client's chat.completions.create to return our mock
            provider._client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            result = asyncio.get_event_loop().run_until_complete(
                provider.complete(
                    model="gpt-4o",
                    system_prompt="You are helpful.",
                    user_prompt="Write hello world.",
                    seed=42,
                )
            )

        assert isinstance(result, LLMResponse)
        assert result.text == "def hello(): print('hi')"
        assert result.input_tokens == 150
        assert result.output_tokens == 80
        assert result.total_tokens == 230
        assert result.model == "gpt-4o"
        assert result.stop_reason == "stop"
        assert result.raw_response is mock_response

    def test_seed_passed_to_api(self) -> None:
        """Verify that the seed parameter is forwarded to OpenAI."""
        mock_message = MagicMock()
        mock_message.content = "ok"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "gpt-4o"

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = get_provider("openai")
            mock_create = AsyncMock(return_value=mock_response)
            provider._client.chat.completions.create = mock_create

            asyncio.get_event_loop().run_until_complete(
                provider.complete(
                    model="gpt-4o",
                    system_prompt="sys",
                    user_prompt="usr",
                    seed=42,
                )
            )

            # Verify seed was passed in the call
            call_kwargs = mock_create.call_args
            assert call_kwargs[1].get("seed") == 42 or call_kwargs.kwargs.get("seed") == 42


# ---------------------------------------------------------------------------
# ProviderError
# ---------------------------------------------------------------------------


class TestProviderError:
    def test_error_includes_provider_name(self) -> None:
        err = ProviderError("anthropic", "something went wrong")
        assert "anthropic" in str(err)
        assert "something went wrong" in str(err)
        assert err.provider == "anthropic"


# ---------------------------------------------------------------------------
# OpenAIProvider — request construction and response parsing
# ---------------------------------------------------------------------------
#
# These use SimpleNamespace rather than MagicMock so that a missing attribute
# fails loudly instead of silently returning a mock, mirroring the shape of
# openai.types.chat.ChatCompletion.


from types import SimpleNamespace



def _run_async(coro):
    """Run a coroutine on a private loop without touching the global event loop.

    (asyncio.run() unsets the current loop, which breaks the older tests that
    use asyncio.get_event_loop().)
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _openai_response(
    content: str | None = "ok",
    finish_reason: str | None = "stop",
    refusal: str | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    reasoning_tokens: int | None = 0,
    cached_tokens: int | None = 0,
    model: str = "gpt-5.6-terra",
    with_usage: bool = True,
    with_choices: bool = True,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, refusal=refusal, role="assistant")
    choice = SimpleNamespace(message=message, finish_reason=finish_reason, index=0)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )
    return SimpleNamespace(
        choices=[choice] if with_choices else [],
        usage=usage if with_usage else None,
        model=model,
    )


def _openai_call(response: SimpleNamespace, **complete_kwargs: object):
    """Run OpenAIProvider.complete against a mocked client.

    Returns (LLMResponse, kwargs sent to chat.completions.create).
    """
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        provider = get_provider("openai")
    mock_create = AsyncMock(return_value=response)
    provider._client.chat.completions.create = mock_create
    args = dict(model="gpt-5.6-terra", system_prompt="sys", user_prompt="usr")
    args.update(complete_kwargs)
    result = _run_async(provider.complete(**args))
    return result, mock_create.call_args.kwargs


class TestOpenAIRequest:
    def test_uses_max_completion_tokens_not_max_tokens(self) -> None:
        _, sent = _openai_call(_openai_response(), max_tokens=1000)
        assert sent["max_completion_tokens"] == 1000
        assert "max_tokens" not in sent

    def test_max_tokens_sent_unchanged(self) -> None:
        """No provider-side clamp: the run's per-call cap reaches the API as-is."""
        result, sent = _openai_call(_openai_response(), max_tokens=32_000)
        assert sent["max_completion_tokens"] == 32_000
        assert result.request_params["max_output_tokens"] == 32_000

    def test_messages_have_system_and_user_roles(self) -> None:
        _, sent = _openai_call(_openai_response())
        assert sent["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]

    def test_registered_model_runs_without_reasoning(self) -> None:
        result, sent = _openai_call(_openai_response(), temperature=0.0)
        assert sent["reasoning_effort"] == "none"
        assert sent["temperature"] == 0.0
        assert result.request_params["reasoning_effort"] == "none"
        assert result.request_params["temperature"] == 0.0

    def test_thinking_enabled_omits_temperature(self) -> None:
        result, sent = _openai_call(_openai_response(), thinking_enabled=True)
        assert sent["reasoning_effort"] == "medium"
        assert "temperature" not in sent
        assert result.request_params["temperature"] is None

    def test_unregistered_model_sends_no_reasoning_effort(self) -> None:
        result, sent = _openai_call(_openai_response(model="gpt-4o"), model="gpt-4o")
        assert "reasoning_effort" not in sent
        assert sent["temperature"] == 0.0
        assert result.request_params["reasoning_effort"] is None

    def test_seed_omitted_when_none(self) -> None:
        result, sent = _openai_call(_openai_response(), seed=None)
        assert "seed" not in sent
        assert result.request_params["seed"] is None

    def test_seed_recorded_when_sent(self) -> None:
        result, sent = _openai_call(_openai_response(), seed=7)
        assert sent["seed"] == 7
        assert result.request_params["seed"] == 7

    def test_request_params_have_standard_keys(self) -> None:
        result, _ = _openai_call(_openai_response(), max_tokens=500)
        assert set(result.request_params) == {
            "temperature", "max_output_tokens", "seed",
            "reasoning_effort", "thinking_budget_tokens",
            "thinking_type", "requested_reasoning_effort", "sent_params",
        }
        assert result.request_params["max_output_tokens"] == 500


class TestOpenAIResponseParsing:
    def test_token_mapping(self) -> None:
        result, _ = _openai_call(
            _openai_response(prompt_tokens=1200, completion_tokens=800)
        )
        assert result.input_tokens == 1200
        assert result.output_tokens == 800
        assert result.total_tokens == 2000

    def test_model_from_response_not_request(self) -> None:
        result, _ = _openai_call(_openai_response(model="gpt-5.6-terra-2026-09-01"))
        assert result.model == "gpt-5.6-terra-2026-09-01"

    def test_usage_details_extracted(self) -> None:
        result, _ = _openai_call(
            _openai_response(completion_tokens=800, reasoning_tokens=300, cached_tokens=1024)
        )
        assert result.usage_details == {"reasoning_tokens": 300, "cached_input_tokens": 1024}
        # reasoning tokens are a subset of completion tokens, not added on top
        assert result.output_tokens == 800

    def test_usage_details_tolerate_missing_fields(self) -> None:
        result, _ = _openai_call(_openai_response(reasoning_tokens=None, cached_tokens=None))
        assert result.usage_details == {"reasoning_tokens": 0, "cached_input_tokens": 0}

    def test_missing_usage_gives_zero_tokens(self) -> None:
        result, _ = _openai_call(_openai_response(with_usage=False))
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.usage_details == {}

    def test_length_finish_reason_preserved(self) -> None:
        result, _ = _openai_call(_openai_response(finish_reason="length"))
        assert result.stop_reason == "length"

    def test_content_filter_finish_reason_preserved(self) -> None:
        result, _ = _openai_call(_openai_response(content=None, finish_reason="content_filter"))
        assert result.text == ""
        assert result.stop_reason == "content_filter"

    def test_none_content_becomes_empty_string(self) -> None:
        result, _ = _openai_call(_openai_response(content=None))
        assert result.text == ""

    def test_refusal_reported_as_refusal(self) -> None:
        result, _ = _openai_call(
            _openai_response(content=None, refusal="I can't help with that.")
        )
        assert result.text == ""
        assert result.stop_reason == "refusal"

    def test_missing_finish_reason_is_unknown(self) -> None:
        result, _ = _openai_call(_openai_response(finish_reason=None))
        assert result.stop_reason == "unknown"

    def test_empty_choices_raises_provider_error(self) -> None:
        with pytest.raises(ProviderError, match="no choices"):
            _openai_call(_openai_response(with_choices=False))

    def test_labeled_code_block_survives_round_trip(self) -> None:
        from architectures.utils import parse_code_blocks

        text = (
            "Here is the implementation.\n\n"
            "```snake/game.py\nclass SnakeGame:\n    pass\n```\n"
        )
        result, _ = _openai_call(_openai_response(content=text))
        assert parse_code_blocks(result.text) == {
            "snake/game.py": "class SnakeGame:\n    pass\n"
        }


def _sdk_error(cls: type) -> Exception:
    """Instantiate an SDK exception without its version-specific constructor."""
    err = cls.__new__(cls)
    Exception.__init__(err, "boom")
    return err


class TestOpenAIErrors:
    def test_authentication_error_wrapped(self) -> None:
        import openai

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = get_provider("openai")
        provider._client.chat.completions.create = AsyncMock(
            side_effect=_sdk_error(openai.AuthenticationError)
        )
        with pytest.raises(ProviderError, match="Authentication failed"):
            _run_async(provider.complete(model="gpt-5.6-terra", system_prompt="s", user_prompt="u"))

    def test_api_error_wrapped(self) -> None:
        import openai

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = get_provider("openai")
        provider._client.chat.completions.create = AsyncMock(
            side_effect=_sdk_error(openai.APIConnectionError)
        )
        with pytest.raises(ProviderError, match=r"\[openai\] API error"):
            _run_async(provider.complete(model="gpt-5.6-terra", system_prompt="s", user_prompt="u"))


# ---------------------------------------------------------------------------
# AnthropicProvider — request_params (temperature logging)
# ---------------------------------------------------------------------------


def _anthropic_call(**complete_kwargs: object):
    """Run AnthropicProvider.complete against a mocked client."""
    block = SimpleNamespace(type="text", text="ok")
    response = SimpleNamespace(
        content=[block],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        model="claude-sonnet-4-6",
        stop_reason="end_turn",
    )
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        provider = get_provider("anthropic")
    mock_create = AsyncMock(return_value=response)
    provider._client.messages.create = mock_create
    args = dict(model="claude-sonnet-4-6", system_prompt="s", user_prompt="u")
    args.update(complete_kwargs)
    result = _run_async(provider.complete(**args))
    return result, mock_create.call_args.kwargs


class TestAnthropicRequestParams:
    def test_temperature_recorded_matches_sent(self) -> None:
        result, sent = _anthropic_call(temperature=0.0)
        assert sent["extra_body"]["temperature"] == 0.0
        assert result.request_params["temperature"] == 0.0
        assert result.request_params["thinking_budget_tokens"] is None

    def test_thinking_records_forced_temperature(self) -> None:
        result, sent = _anthropic_call(temperature=0.0, thinking_enabled=True)
        assert sent["extra_body"]["temperature"] == 1.0
        assert result.request_params["temperature"] == 1.0
        assert result.request_params["thinking_budget_tokens"] == 10000

    def test_seed_recorded_as_not_sent(self) -> None:
        result, sent = _anthropic_call(seed=42)
        assert "seed" not in sent
        assert result.request_params["seed"] is None

    def test_max_output_tokens_sent_unchanged(self) -> None:
        """No provider-side clamp: the run's per-call cap reaches the API as-is."""
        result, sent = _anthropic_call(max_tokens=32_000)
        assert sent["max_tokens"] == 32_000
        assert result.request_params["max_output_tokens"] == 32_000

    def test_same_param_keys_as_openai(self) -> None:
        anthropic_result, _ = _anthropic_call()
        openai_result, _ = _openai_call(_openai_response())
        assert set(anthropic_result.request_params) == set(openai_result.request_params)


# ---------------------------------------------------------------------------
# Explicit reasoning effort — current lineup (Claude 5.5, GPT-6)
# ---------------------------------------------------------------------------


_CLAUDE_55 = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
_GPT_6 = ["gpt-6-luna", "gpt-6.1-sol", "gpt-6-astra"]


class TestAnthropicReasoningEffort:
    @pytest.mark.parametrize("model", _CLAUDE_55)
    def test_sends_adaptive_thinking_and_effort_without_temperature(self, model: str) -> None:
        result, sent = _anthropic_call(model=model, temperature=0.0, reasoning_effort="low")
        assert sent["thinking"] == {"type": "adaptive"}
        assert sent["output_config"] == {"effort": "low"}
        assert "extra_body" not in sent  # no temperature anywhere in the request
        assert "temperature" not in sent
        assert set(sent) == {
            "model", "max_tokens", "system", "messages", "timeout", "thinking", "output_config",
        }

    def test_logs_requested_and_sent(self) -> None:
        result, _ = _anthropic_call(model="claude-opus-5-5", reasoning_effort="low")
        rp = result.request_params
        assert rp["requested_reasoning_effort"] == "low"
        assert rp["reasoning_effort"] == "low"
        assert rp["thinking_type"] == "adaptive"
        assert rp["temperature"] is None
        assert rp["thinking_budget_tokens"] is None
        assert rp["sent_params"] == {
            "max_tokens": 4096,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
        }

    def test_legacy_sent_params_include_extra_body_temperature(self) -> None:
        result, _ = _anthropic_call(temperature=0.0)
        assert result.request_params["sent_params"] == {"max_tokens": 4096, "temperature": 0.0}
        assert result.request_params["requested_reasoning_effort"] is None

    @pytest.mark.parametrize("kwargs,match", [
        ({"model": "claude-opus-5-5"}, "requires an explicit reasoning effort"),
        ({"model": "claude-opus-5-5", "thinking_enabled": True}, "requires an explicit"),
        ({"model": "claude-opus-5-5", "reasoning_effort": "none"}, "does not accept"),
        ({"model": "claude-sonnet-4-6", "reasoning_effort": "low"}, "no verified"),
    ])
    def test_invalid_combination_raises_before_api_call(self, kwargs: dict, match: str) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = get_provider("anthropic")
        provider._client.messages.create = AsyncMock()
        args = dict(system_prompt="s", user_prompt="u", **kwargs)
        with pytest.raises(ProviderError, match=match):
            _run_async(provider.complete(**args))
        provider._client.messages.create.assert_not_called()


class TestOpenAIReasoningEffort:
    @pytest.mark.parametrize("model", _GPT_6)
    def test_sends_low_effort_without_temperature(self, model: str) -> None:
        result, sent = _openai_call(
            _openai_response(model=model), model=model, temperature=0.0, reasoning_effort="low",
        )
        assert sent["reasoning_effort"] == "low"
        assert "temperature" not in sent
        assert set(sent) == {"model", "max_completion_tokens", "messages", "timeout", "reasoning_effort"}
        rp = result.request_params
        assert rp["requested_reasoning_effort"] == "low"
        assert rp["reasoning_effort"] == "low"
        assert rp["temperature"] is None
        assert rp["sent_params"] == {"max_completion_tokens": 4096, "reasoning_effort": "low"}

    def test_luna_at_none_sends_temperature(self) -> None:
        _, sent = _openai_call(
            _openai_response(model="gpt-6-luna"), model="gpt-6-luna", reasoning_effort="none",
        )
        assert sent["reasoning_effort"] == "none"
        assert sent["temperature"] == 0.0

    def test_legacy_requested_is_none_sent_is_registry_min(self) -> None:
        result, _ = _openai_call(_openai_response())
        assert result.request_params["requested_reasoning_effort"] is None
        assert result.request_params["reasoning_effort"] == "none"

    @pytest.mark.parametrize("kwargs,match", [
        ({"model": "gpt-6-astra"}, "requires an explicit reasoning effort"),
        ({"model": "gpt-6-astra", "reasoning_effort": "none"}, "does not accept"),
        ({"model": "gpt-6.1-sol", "reasoning_effort": "none"}, "does not accept"),
        ({"model": "gpt-6-luna", "reasoning_effort": "low", "thinking_enabled": True}, "mutually exclusive"),
        ({"model": "gpt-5.6-terra", "reasoning_effort": "low"}, "no verified"),
    ])
    def test_invalid_combination_raises_before_api_call(self, kwargs: dict, match: str) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = get_provider("openai")
        provider._client.chat.completions.create = AsyncMock()
        with pytest.raises(ProviderError, match=match):
            _run_async(provider.complete(system_prompt="s", user_prompt="u", **kwargs))
        provider._client.chat.completions.create.assert_not_called()
