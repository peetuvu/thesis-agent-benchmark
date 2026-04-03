"""Tests for the sequential pipeline architecture.

All tests use mocks — no real API calls are made.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from architectures.base import RunMetrics, RunResult, RunStatus, TaskContext
from architectures.sequential import SequentialPipeline
from architectures.utils import parse_code_blocks
from providers.base import LLMResponse, ProviderError


# ---------------------------------------------------------------------------
# Helpers
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
        prompt_template="",
        model="test-model",
        max_tokens=40_000,
    )
    defaults.update(overrides)
    return TaskContext(**defaults)


def _make_response(text: str, input_tokens: int = 100, output_tokens: int = 50) -> LLMResponse:
    """Create a canned LLMResponse."""
    return LLMResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model="test-model",
        stop_reason="end_turn",
    )


def _make_mock_provider(responses: list[LLMResponse]) -> MagicMock:
    """Create a mock provider that returns responses in sequence."""
    provider = MagicMock()
    provider.name = "mock"
    provider.complete = AsyncMock(side_effect=responses)
    return provider


def _four_step_provider(
    plan: str = "Here is the plan.",
    implement: str = "```snake/game.py\nclass SnakeGame:\n    pass\n```",
    review: str = "Looks good, no issues found.",
    fix: str = "```snake/game.py\nclass SnakeGame:\n    pass\n```",
) -> MagicMock:
    """Create a mock provider with responses for all 4 steps."""
    return _make_mock_provider([
        _make_response(plan, input_tokens=200, output_tokens=100),
        _make_response(implement, input_tokens=300, output_tokens=150),
        _make_response(review, input_tokens=250, output_tokens=80),
        _make_response(fix, input_tokens=350, output_tokens=120),
    ])


def _run(ctx: TaskContext) -> RunResult:
    """Run the pipeline synchronously."""
    pipeline = SequentialPipeline()
    return asyncio.get_event_loop().run_until_complete(pipeline.run(ctx))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNameProperty:
    def test_name(self) -> None:
        pipeline = SequentialPipeline()
        assert pipeline.name == "sequential-pipeline"


class TestFourStepExecution:
    def test_makes_exactly_four_calls(self) -> None:
        provider = _four_step_provider()
        ctx = _make_context(provider=provider)
        _run(ctx)
        assert provider.complete.call_count == 4

    def test_step1_prompt_contains_task_spec_and_source_files(self) -> None:
        provider = _four_step_provider()
        ctx = _make_context(provider=provider)
        _run(ctx)

        step1_call = provider.complete.call_args_list[0]
        user_prompt = step1_call.kwargs["user_prompt"]
        assert "Implement a snake game." in user_prompt
        assert "snake/game.py" in user_prompt
        assert "snake/__init__.py" in user_prompt

    def test_step2_prompt_contains_plan_from_step1(self) -> None:
        plan_text = "My detailed plan: do X then Y."
        provider = _four_step_provider(plan=plan_text)
        ctx = _make_context(provider=provider)
        _run(ctx)

        step2_call = provider.complete.call_args_list[1]
        user_prompt = step2_call.kwargs["user_prompt"]
        assert plan_text in user_prompt

    def test_step3_prompt_contains_code_from_step2(self) -> None:
        impl_text = "```snake/game.py\nclass SnakeGame:\n    def move(self): ...\n```"
        provider = _four_step_provider(implement=impl_text)
        ctx = _make_context(provider=provider)
        _run(ctx)

        step3_call = provider.complete.call_args_list[2]
        user_prompt = step3_call.kwargs["user_prompt"]
        assert "class SnakeGame:" in user_prompt
        assert "def move(self):" in user_prompt

    def test_step4_prompt_contains_review_from_step3(self) -> None:
        review_text = "Bug found: off-by-one error in boundary check."
        provider = _four_step_provider(review=review_text)
        ctx = _make_context(provider=provider)
        _run(ctx)

        step4_call = provider.complete.call_args_list[3]
        user_prompt = step4_call.kwargs["user_prompt"]
        assert review_text in user_prompt


class TestMetricsAggregation:
    def test_total_tokens_sum_of_all_steps(self) -> None:
        provider = _four_step_provider()
        ctx = _make_context(provider=provider)
        result = _run(ctx)

        # 200+300+250+350 = 1100 input, 100+150+80+120 = 450 output
        assert result.metrics.input_tokens == 1100
        assert result.metrics.output_tokens == 450
        assert result.metrics.total_tokens == 1550

    def test_api_calls_is_four(self) -> None:
        provider = _four_step_provider()
        ctx = _make_context(provider=provider)
        result = _run(ctx)
        assert result.metrics.api_calls == 4


class TestAgentLog:
    def test_four_entries(self) -> None:
        provider = _four_step_provider()
        ctx = _make_context(provider=provider)
        result = _run(ctx)
        assert len(result.agent_log) == 4

    def test_step_numbers_and_roles(self) -> None:
        provider = _four_step_provider()
        ctx = _make_context(provider=provider)
        result = _run(ctx)

        expected = [
            (1, "plan"),
            (2, "implement"),
            (3, "review"),
            (4, "fix"),
        ]
        for entry, (step, role) in zip(result.agent_log, expected):
            assert entry["step"] == step
            assert entry["role"] == role
            assert "prompt" in entry
            assert "response" in entry
            assert "input_tokens" in entry
            assert "output_tokens" in entry


class TestProviderError:
    def test_error_on_step2_stops_pipeline(self) -> None:
        responses = [
            _make_response("Plan text", input_tokens=200, output_tokens=100),
        ]
        provider = MagicMock()
        provider.name = "mock"
        provider.complete = AsyncMock(
            side_effect=[
                responses[0],
                ProviderError("mock", "Rate limit exceeded"),
            ]
        )
        ctx = _make_context(provider=provider)
        result = _run(ctx)

        assert result.status == RunStatus.ERROR
        assert "Step 2" in result.error_message
        assert "implement" in result.error_message
        assert "Rate limit" in result.error_message
        assert provider.complete.call_count == 2

    def test_error_on_step1_stops_immediately(self) -> None:
        provider = MagicMock()
        provider.name = "mock"
        provider.complete = AsyncMock(
            side_effect=ProviderError("mock", "Auth failed")
        )
        ctx = _make_context(provider=provider)
        result = _run(ctx)

        assert result.status == RunStatus.ERROR
        assert "Step 1" in result.error_message
        assert provider.complete.call_count == 1

    def test_no_provider_returns_error(self) -> None:
        ctx = _make_context(provider=None)
        result = _run(ctx)
        assert result.status == RunStatus.ERROR
        assert "No provider" in result.error_message


class TestModifiedFiles:
    def test_parsed_from_step4(self) -> None:
        fix_code = "```snake/game.py\nclass SnakeGame:\n    fixed = True\n```"
        provider = _four_step_provider(fix=fix_code)
        ctx = _make_context(provider=provider)
        result = _run(ctx)

        assert "snake/game.py" in result.modified_files
        assert "fixed = True" in result.modified_files["snake/game.py"]

    def test_fallback_to_step2_when_step4_has_no_code(self) -> None:
        impl_code = "```snake/game.py\nclass SnakeGame:\n    impl = True\n```"
        fix_text = "No issues found, code is correct."
        provider = _four_step_provider(implement=impl_code, fix=fix_text)
        ctx = _make_context(provider=provider)
        result = _run(ctx)

        assert "snake/game.py" in result.modified_files
        assert "impl = True" in result.modified_files["snake/game.py"]

    def test_no_code_blocks_at_all_returns_failure(self) -> None:
        provider = _four_step_provider(
            implement="I cannot solve this.",
            fix="Nothing to fix.",
        )
        ctx = _make_context(provider=provider)
        result = _run(ctx)

        assert result.status == RunStatus.FAILURE
        assert result.modified_files == {}

    def test_success_status_on_happy_path(self) -> None:
        provider = _four_step_provider()
        ctx = _make_context(provider=provider)
        result = _run(ctx)
        assert result.status == RunStatus.SUCCESS


class TestTokenBudgetSplit:
    def test_per_step_max_tokens(self) -> None:
        provider = _four_step_provider()
        ctx = _make_context(provider=provider, max_tokens=40_000)
        _run(ctx)

        for call in provider.complete.call_args_list:
            assert call.kwargs["max_tokens"] == 10_000


class TestStepModels:
    def test_step_models_stored(self) -> None:
        models = {"plan": "gpt-4o", "implement": "claude-sonnet-4-6"}
        pipeline = SequentialPipeline(step_models=models)
        assert pipeline._step_models == models

    def test_default_step_models_is_none(self) -> None:
        pipeline = SequentialPipeline()
        assert pipeline._step_models is None
