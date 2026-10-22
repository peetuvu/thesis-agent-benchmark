"""SDK drift tests: provider requests checked against the real installed SDKs.

Unlike test_providers.py, these tests do not trust a mock's view of the SDK.
The kwargs each provider builds are bound against the installed SDK method
signature (inspect.signature), so an SDK upgrade that removes or renames a
parameter fails here instead of in a paid run. The wire-level test sends a
request through the real Anthropic client into an in-process mock transport;
no network traffic and no API calls are made.
"""

from __future__ import annotations

import inspect
import json
import tomllib
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anthropic
import httpx2
import pytest
from anthropic.resources.messages import AsyncMessages
from openai.resources.chat.completions import AsyncCompletions

from providers import get_provider

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _run_async(coro):
    """Run a coroutine on a private event loop."""
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _assert_binds(method: object, kwargs: dict) -> None:
    """Fail if kwargs would raise TypeError when passed to the real SDK method."""
    signature = inspect.signature(method)
    try:
        signature.bind(None, **kwargs)  # None stands in for self
    except TypeError as e:
        accepted = sorted(p for p in signature.parameters if p != "self")
        pytest.fail(f"{method.__qualname__} rejects provider kwargs: {e}. Accepted: {accepted}")


def _captured_anthropic_kwargs(**complete_kwargs: object) -> dict:
    """Return the kwargs AnthropicProvider passes to messages.create."""
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        model="claude-sonnet-4-6",
        stop_reason="end_turn",
    )
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        provider = get_provider("anthropic")
    mock_create = AsyncMock(return_value=response)
    provider._client.messages.create = mock_create
    args = dict(model="claude-sonnet-4-6", system_prompt="s", user_prompt="u")
    args.update(complete_kwargs)
    _run_async(provider.complete(**args))
    return mock_create.call_args.kwargs


def _captured_openai_kwargs(**complete_kwargs: object) -> dict:
    """Return the kwargs OpenAIProvider passes to chat.completions.create."""
    message = SimpleNamespace(content="ok", refusal=None, role="assistant")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop", index=0)],
        usage=None,
        model="gpt-5.6-terra",
    )
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        provider = get_provider("openai")
    mock_create = AsyncMock(return_value=response)
    provider._client.chat.completions.create = mock_create
    args = dict(model="gpt-5.6-terra", system_prompt="s", user_prompt="u")
    args.update(complete_kwargs)
    _run_async(provider.complete(**args))
    return mock_create.call_args.kwargs


class TestInstalledVersionsMatchPins:
    """A stale venv (e.g. on the other machine) must not silently run benchmarks."""

    @pytest.mark.parametrize("package", ["anthropic", "openai"])
    def test_installed_version_matches_pyproject_pin(self, package: str) -> None:
        deps = tomllib.loads(_PYPROJECT.read_text())["project"]["dependencies"]
        pins = [d.split("==", 1)[1] for d in deps if d.startswith(f"{package}==")]
        assert pins, f"{package} is not pinned with == in pyproject.toml"
        assert version(package) == pins[0], (
            f"installed {package} {version(package)} != pinned {pins[0]}; run `uv sync`"
        )


class TestAnthropicKwargsMatchSdkSignature:
    @pytest.mark.parametrize("thinking_enabled", [False, True])
    def test_kwargs_bind_to_messages_create(self, thinking_enabled: bool) -> None:
        kwargs = _captured_anthropic_kwargs(thinking_enabled=thinking_enabled)
        _assert_binds(AsyncMessages.create, kwargs)


class TestOpenAIKwargsMatchSdkSignature:
    @pytest.mark.parametrize(
        "complete_kwargs",
        [
            {},  # registered model, reasoning_effort="none", temperature sent
            {"thinking_enabled": True},  # reasoning_effort sent, no temperature
            {"model": "unregistered-model"},  # no reasoning_effort
            {"seed": 42},
        ],
        ids=["default", "thinking", "unregistered", "seed"],
    )
    def test_kwargs_bind_to_chat_completions_create(self, complete_kwargs: dict) -> None:
        kwargs = _captured_openai_kwargs(**complete_kwargs)
        _assert_binds(AsyncCompletions.create, kwargs)


class TestAnthropicWireRequest:
    """Send a request through the real SDK client into a mock transport.

    This catches drift that a signature check cannot: extra_body only works
    while the SDK sends an API version that accepts temperature (2023-06-01).
    """

    def _send(self, **complete_kwargs: object) -> httpx2.Request:
        captured: list[httpx2.Request] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            captured.append(request)
            return httpx2.Response(200, json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 1},
            })

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = get_provider("anthropic")
        provider._client = anthropic.AsyncAnthropic(
            api_key="test-key",
            max_retries=0,
            http_client=anthropic.DefaultAsyncHttpxClient(
                transport=httpx2.MockTransport(handler)
            ),
        )
        args = dict(model="claude-sonnet-4-6", system_prompt="s", user_prompt="u")
        args.update(complete_kwargs)
        result = _run_async(provider.complete(**args))
        assert result.text == "ok"
        assert len(captured) == 1
        return captured[0]

    def test_api_version_accepts_temperature(self) -> None:
        request = self._send()
        assert request.headers["anthropic-version"] == "2023-06-01", (
            "SDK now sends a different API version; 2026-10-21 rejects temperature"
        )

    def test_temperature_reaches_request_body(self) -> None:
        body = json.loads(self._send(temperature=0.0).content)
        assert body["temperature"] == 0.0
        assert body["model"] == "claude-sonnet-4-6"
        assert body["system"] == "s"

    def test_thinking_sends_forced_temperature(self) -> None:
        body = json.loads(self._send(thinking_enabled=True).content)
        assert body["temperature"] == 1.0
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 10000}

