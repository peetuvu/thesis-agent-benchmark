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
