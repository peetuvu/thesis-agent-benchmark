"""Tests for the single-agent baseline architecture.

All tests use mocks — no real API calls are made.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from architectures.base import RunMetrics, RunResult, RunStatus, TaskContext
from architectures.single_agent import (
    SingleAgent,
    format_source_files,
    parse_code_blocks,
)
from providers.base import LLMResponse, ProviderError


# ---------------------------------------------------------------------------
# parse_code_blocks
# ---------------------------------------------------------------------------


class TestParseCodeBlocks:
    def test_single_block(self) -> None:
        text = "Here is the code:\n\n```snake/game.py\nclass Snake:\n    pass\n```\n"
        result = parse_code_blocks(text)
        assert result == {"snake/game.py": "class Snake:\n    pass\n"}

    def test_multiple_blocks(self) -> None:
        text = (
            "```snake/__init__.py\n# init\n```\n\n"
            "```snake/game.py\nclass Snake:\n    pass\n```\n"
        )
        result = parse_code_blocks(text)
        assert len(result) == 2
        assert "snake/__init__.py" in result
        assert "snake/game.py" in result
        assert result["snake/__init__.py"] == "# init\n"
        assert result["snake/game.py"] == "class Snake:\n    pass\n"

    def test_no_code_blocks(self) -> None:
        text = "I couldn't solve this task. Sorry!"
        result = parse_code_blocks(text)
        assert result == {}

    def test_language_identifier_not_matched(self) -> None:
        text = "```python\ndef hello():\n    pass\n```\n"
        result = parse_code_blocks(text)
        assert result == {}

    def test_javascript_identifier_not_matched(self) -> None:
        text = "```javascript\nconsole.log('hi');\n```\n"
        result = parse_code_blocks(text)
        assert result == {}

    def test_mixed_labels_and_identifiers(self) -> None:
        text = (
            "```python\n# some example\n```\n\n"
            "```snake/game.py\nclass Snake:\n    pass\n```\n"
        )
        result = parse_code_blocks(text)
        assert len(result) == 1
        assert "snake/game.py" in result

    def test_file_without_directory(self) -> None:
        text = "```game.py\nclass Game:\n    pass\n```\n"
        result = parse_code_blocks(text)
        assert result == {"game.py": "class Game:\n    pass\n"}

    def test_empty_content(self) -> None:
        text = "```snake/game.py\n```\n"
        result = parse_code_blocks(text)
        assert result == {"snake/game.py": ""}

    def test_whitespace_after_label(self) -> None:
        text = "```snake/game.py   \nclass Snake:\n    pass\n```\n"
        result = parse_code_blocks(text)
        assert "snake/game.py" in result


# ---------------------------------------------------------------------------
# format_source_files
# ---------------------------------------------------------------------------


class TestFormatSourceFiles:
    def test_basic_formatting(self) -> None:
        files = {"snake/game.py": "class Snake:\n    pass\n"}
        result = format_source_files(files)
        assert "```snake/game.py" in result
        assert "class Snake:" in result
        assert result.endswith("```")

    def test_empty_dict(self) -> None:
        result = format_source_files({})
        assert result == "(no source files)"

    def test_adds_trailing_newline(self) -> None:
        files = {"game.py": "pass"}
        result = format_source_files(files)
        # Content should have newline added before closing ```
        assert "pass\n```" in result

    def test_multiple_files(self) -> None:
        files = {
            "snake/__init__.py": "# init\n",
            "snake/game.py": "# game\n",
        }
        result = format_source_files(files)
        assert "```snake/__init__.py" in result
        assert "```snake/game.py" in result


# ---------------------------------------------------------------------------
# SingleAgent prompt formatting
# ---------------------------------------------------------------------------


def _make_context(**overrides: object) -> TaskContext:
    """Create a TaskContext with sensible defaults for testing."""
    defaults = dict(
        task_dir=Path("/tmp/test_task"),
        task_spec="Implement a snake game.",
        source_files={
            "snake/__init__.py": "# init\n",
            "snake/game.py": "# stub\nclass SnakeGame:\n    pass\n",
        },
        prompt_regime="minimal",
        prompt_template=(
            "Task: {task_spec}\n\n"
            "Files:\n{source_files}\n\n"
            "Editable: {editable_files}\n\n"
            "Test: {test_cmd}\n"
        ),
        model="test-model",
    )
    defaults.update(overrides)
    return TaskContext(**defaults)


def _make_mock_provider(response_text: str = "") -> MagicMock:
    """Create a mock LLMProvider that returns a canned response."""
    provider = MagicMock()
    provider.name = "mock"
    provider.complete = AsyncMock(
        return_value=LLMResponse(
            text=response_text,
            input_tokens=100,
            output_tokens=50,
            model="test-model",
            stop_reason="end_turn",
        )
    )
    return provider


class TestSingleAgentPromptFormatting:
    def test_all_placeholders_filled(self) -> None:
        provider = _make_mock_provider("```snake/game.py\ncode\n```")
        ctx = _make_context(provider=provider)

        agent = SingleAgent()
        asyncio.get_event_loop().run_until_complete(agent.run(ctx))

        call_kwargs = provider.complete.call_args
        user_prompt = call_kwargs.kwargs.get("user_prompt") or call_kwargs[1].get("user_prompt")

        assert "Implement a snake game." in user_prompt
        assert "snake/game.py" in user_prompt
        assert "snake/__init__.py" in user_prompt
        assert "python -m pytest tests/ -v" in user_prompt

    def test_minimal_template_works(self) -> None:
        """Minimal template only uses task_spec and source_files — no errors."""
        provider = _make_mock_provider("```snake/game.py\ncode\n```")
        ctx = _make_context(
            provider=provider,
            prompt_template="Do this: {task_spec}\n\nCode:\n{source_files}\n",
        )

        agent = SingleAgent()
        result = asyncio.get_event_loop().run_until_complete(agent.run(ctx))
        assert result.status == RunStatus.SUCCESS


# ---------------------------------------------------------------------------
# SingleAgent.run() — full integration with mock provider
# ---------------------------------------------------------------------------


class TestSingleAgentRun:
    def test_happy_path(self) -> None:
        response_text = (
            "Here is the implementation:\n\n"
            "```snake/game.py\n"
            "class SnakeGame:\n"
            "    def __init__(self):\n"
            "        self.score = 0\n"
            "```\n"
        )
        provider = _make_mock_provider(response_text)
        ctx = _make_context(provider=provider)

        agent = SingleAgent()
        result = asyncio.get_event_loop().run_until_complete(agent.run(ctx))

        assert result.status == RunStatus.SUCCESS
        assert "snake/game.py" in result.modified_files
        assert "class SnakeGame:" in result.modified_files["snake/game.py"]
        assert result.metrics.input_tokens == 100
        assert result.metrics.output_tokens == 50
        assert result.metrics.api_calls == 1
        assert result.error_message is None

    def test_no_code_blocks_returns_failure(self) -> None:
        provider = _make_mock_provider("I don't know how to do this.")
        ctx = _make_context(provider=provider)

        agent = SingleAgent()
        result = asyncio.get_event_loop().run_until_complete(agent.run(ctx))

        assert result.status == RunStatus.FAILURE
        assert result.modified_files == {}
        assert "No labeled code blocks" in result.error_message

    def test_provider_error_returns_error(self) -> None:
        provider = MagicMock()
        provider.name = "mock"
        provider.complete = AsyncMock(
            side_effect=ProviderError("mock", "API key invalid")
        )
        ctx = _make_context(provider=provider)

        agent = SingleAgent()
        result = asyncio.get_event_loop().run_until_complete(agent.run(ctx))

        assert result.status == RunStatus.ERROR
        assert "API key invalid" in result.error_message
        assert result.metrics.api_calls == 1

    def test_no_provider_returns_error(self) -> None:
        ctx = _make_context(provider=None)

        agent = SingleAgent()
        result = asyncio.get_event_loop().run_until_complete(agent.run(ctx))

        assert result.status == RunStatus.ERROR
        assert "No provider" in result.error_message

    def test_agent_log_recorded(self) -> None:
        provider = _make_mock_provider("```snake/game.py\ncode\n```")
        ctx = _make_context(provider=provider)

        agent = SingleAgent()
        result = asyncio.get_event_loop().run_until_complete(agent.run(ctx))

        assert len(result.agent_log) == 2
        assert result.agent_log[0]["role"] == "user"
        assert result.agent_log[1]["role"] == "assistant"
        assert result.agent_log[1]["model"] == "test-model"
        assert result.agent_log[1]["stop_reason"] == "end_turn"

    def test_metrics_from_llm_response(self) -> None:
        provider = MagicMock()
        provider.name = "mock"
        provider.complete = AsyncMock(
            return_value=LLMResponse(
                text="```snake/game.py\npass\n```",
                input_tokens=500,
                output_tokens=200,
                model="gpt-4o",
                stop_reason="stop",
            )
        )
        ctx = _make_context(provider=provider)

        agent = SingleAgent()
        result = asyncio.get_event_loop().run_until_complete(agent.run(ctx))

        assert result.metrics.input_tokens == 500
        assert result.metrics.output_tokens == 200
        assert result.metrics.total_tokens == 700
        assert result.metrics.api_calls == 1

    def test_name_property(self) -> None:
        agent = SingleAgent()
        assert agent.name == "single-agent"
