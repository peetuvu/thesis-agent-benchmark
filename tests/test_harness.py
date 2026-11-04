"""Tests for the benchmarking harness: config, sandbox, and logger."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from runner.config import RunConfig, TaskPaths
from runner.logger import (
    ActualMetrics,
    BudgetConfig,
    EvalResults,
    RunLog,
    get_git_info,
    make_run_filename,
    save_run_log,
    shorten_model,
)
from runner.sandbox import check_policy, snapshot_directory


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


class TestRunConfig:
    def test_defaults(self) -> None:
        cfg = RunConfig(task_name="snake", architecture_name="single_agent")
        assert cfg.model == "claude-sonnet-4-6"
        assert cfg.provider == "anthropic"
        assert cfg.prompt_regime == "minimal"
        assert cfg.max_tokens == 100_000
        assert cfg.max_tool_calls == 10
        assert cfg.max_wall_clock_seconds == 1200.0
        assert cfg.seed is None

    def test_auto_fields(self) -> None:
        cfg = RunConfig(task_name="snake", architecture_name="single_agent")
        # run_id is a UUID4 string
        assert len(cfg.run_id) == 36
        assert cfg.run_id.count("-") == 4
        # timestamp is a recent UTC datetime
        assert isinstance(cfg.timestamp, datetime)
        assert cfg.timestamp.tzinfo is not None

    def test_two_configs_have_different_ids(self) -> None:
        a = RunConfig(task_name="snake", architecture_name="x")
        b = RunConfig(task_name="snake", architecture_name="x")
        assert a.run_id != b.run_id


# ---------------------------------------------------------------------------
# TaskPaths
# ---------------------------------------------------------------------------


class TestTaskPaths:
    def test_resolves_snake(self) -> None:
        tp = TaskPaths("snake")
        assert tp.task_dir.name == "snake"
        assert tp.task_dir.parent.name == "tasks"
        assert tp.source_dir == tp.task_dir / "snake"
        assert tp.public_tests_dir == tp.task_dir / "tests"
        assert tp.hidden_tests_dir.parts[-2:] == ("hidden_tests", "snake")
        assert tp.readme_path == tp.task_dir / "README.md"

    def test_allowed_edit_files_snake(self) -> None:
        tp = TaskPaths("snake")
        allowed = tp.allowed_edit_files
        # game.py should be allowed, __init__.py should not
        assert "snake/game.py" in allowed
        assert "snake/__init__.py" not in allowed


# ---------------------------------------------------------------------------
# Snapshot & policy
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("world")

        snap = snapshot_directory(tmp_path)
        assert "a.txt" in snap
        assert "sub/b.txt" in snap
        assert len(snap) == 2

    def test_snapshot_ignores_pycache(self, tmp_path: Path) -> None:
        (tmp_path / "ok.py").write_text("pass")
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "ok.cpython-311.pyc").write_bytes(b"\x00")

        snap = snapshot_directory(tmp_path)
        assert "ok.py" in snap
        assert len(snap) == 1

    def test_snapshot_empty_dir(self, tmp_path: Path) -> None:
        snap = snapshot_directory(tmp_path)
        assert snap == {}

    def test_snapshot_nonexistent(self, tmp_path: Path) -> None:
        snap = snapshot_directory(tmp_path / "nope")
        assert snap == {}


class TestPolicy:
    def test_no_violations(self) -> None:
        pre = {"a.py": "abc123", "b.py": "def456"}
        post = {"a.py": "abc123", "b.py": "def456"}
        ok, violations = check_policy(pre, post, allowed_files=[])
        assert ok is True
        assert violations == []

    def test_detects_modification(self) -> None:
        pre = {"a.py": "abc123"}
        post = {"a.py": "changed"}
        ok, violations = check_policy(pre, post, allowed_files=[])
        assert ok is False
        assert len(violations) == 1
        assert "modified" in violations[0]

    def test_detects_added_file(self) -> None:
        pre = {"a.py": "abc123"}
        post = {"a.py": "abc123", "evil.py": "new"}
        ok, violations = check_policy(pre, post, allowed_files=[])
        assert ok is False
        assert any("added" in v for v in violations)

    def test_detects_deleted_file(self) -> None:
        pre = {"a.py": "abc123", "b.py": "def456"}
        post = {"a.py": "abc123"}
        ok, violations = check_policy(pre, post, allowed_files=[])
        assert ok is False
        assert any("deleted" in v for v in violations)

    def test_allows_permitted_file(self) -> None:
        pre = {"game.py": "old"}
        post = {"game.py": "new"}
        ok, violations = check_policy(pre, post, allowed_files=["game.py"])
        assert ok is True
        assert violations == []

    def test_mixed_allowed_and_disallowed(self) -> None:
        pre = {"game.py": "old", "secret.py": "x"}
        post = {"game.py": "new", "secret.py": "y"}
        ok, violations = check_policy(pre, post, allowed_files=["game.py"])
        assert ok is False
        assert len(violations) == 1
        assert "secret.py" in violations[0]


# ---------------------------------------------------------------------------
# RunLog & logger
# ---------------------------------------------------------------------------


def _make_run_log(**overrides: object) -> RunLog:
    """Create a RunLog with sensible defaults for testing."""
    defaults = dict(
        run_id="test-uuid-1234",
        timestamp=datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc),
        git_commit="abc123",
        git_branch="main",
        task_name="snake",
        architecture_name="single_agent",
        prompt_regime="minimal",
        model="claude-sonnet-4-6",
        provider="anthropic",
        budget_config=BudgetConfig(),
        actual_metrics=ActualMetrics(
            tokens_prompt=500,
            tokens_completion=200,
            tokens_total=700,
            api_calls=3,
            wall_clock_seconds=12.5,
            estimated_cost_usd=0.0042,
        ),
        test_results=EvalResults(
            public_passed=4,
            public_total=5,
            hidden_passed=7,
            hidden_total=8,
            failing_test_names=["test_hidden.py::test_wrap"],
        ),
        status="failure",
        policy_compliant=True,
        stop_reason=None,
        notes=None,
    )
    defaults.update(overrides)
    return RunLog(**defaults)


class TestRunLog:
    def test_serialization_json(self) -> None:
        log = _make_run_log()
        data = json.loads(log.model_dump_json())
        assert data["run_id"] == "test-uuid-1234"
        assert data["task_name"] == "snake"
        assert data["actual_metrics"]["tokens_total"] == 700
        assert data["test_results"]["hidden_passed"] == 7

    def test_serialization_roundtrip(self) -> None:
        log = _make_run_log()
        json_str = log.model_dump_json()
        restored = RunLog.model_validate_json(json_str)
        assert restored.run_id == log.run_id
        assert restored.actual_metrics.tokens_total == 700

    def test_nullable_fields(self) -> None:
        log = _make_run_log(git_commit=None, git_branch=None, stop_reason=None, notes=None)
        data = json.loads(log.model_dump_json())
        assert data["git_commit"] is None
        assert data["stop_reason"] is None

    def test_save_run_log_creates_files(self, tmp_path: Path) -> None:
        log = _make_run_log()
        json_path, md_path = save_run_log(log, tmp_path)

        assert json_path.exists()
        assert md_path.exists()
        assert json_path.suffix == ".json"
        assert md_path.suffix == ".md"

        # Filenames use the new descriptive format
        # UTC 12:00 -> Helsinki 14:00 (EET, UTC+2; DST starts last Sun of March)
        expected_stem = "26-03-05_1400_anthropic_sonnet-4-6_single_agent_snake_minimal_test-uui"
        assert json_path.stem == expected_stem
        assert md_path.stem == expected_stem

        # JSON is valid and has expected content (full run_id preserved inside)
        data = json.loads(json_path.read_text())
        assert data["run_id"] == "test-uuid-1234"

        # Markdown has human-readable content
        md = md_path.read_text()
        assert "test-uuid-1234" in md
        assert "snake" in md
        assert "Hidden" in md

    def test_markdown_includes_failing_tests(self, tmp_path: Path) -> None:
        log = _make_run_log()
        _, md_path = save_run_log(log, tmp_path)
        md = md_path.read_text()
        assert "test_wrap" in md


class TestShortenModel:
    def test_claude_with_date(self) -> None:
        assert shorten_model("claude-haiku-4-5-20251001") == "haiku-4-5"

    def test_gpt_with_date(self) -> None:
        assert shorten_model("gpt-4o-2024-08-06") == "gpt-4o"

    def test_claude_no_date(self) -> None:
        assert shorten_model("claude-opus-4-6") == "opus-4-6"

    def test_plain_model(self) -> None:
        assert shorten_model("gemini-pro") == "gemini-pro"


class TestMakeRunFilename:
    def test_format(self) -> None:
        log = _make_run_log()
        filename = make_run_filename(log)
        assert filename == "26-03-05_1400_anthropic_sonnet-4-6_single_agent_snake_minimal_test-uui"

    def test_different_provider(self) -> None:
        log = _make_run_log(provider="openai", model="gpt-4o-2024-08-06")
        filename = make_run_filename(log)
        assert "openai_gpt-4o_" in filename


class TestGitInfo:
    def test_get_git_info(self) -> None:
        commit, branch = get_git_info()
        # We're in a git repo, so these should be non-None
        assert commit is not None
        assert len(commit) == 40  # full SHA
        assert branch is not None


# ---------------------------------------------------------------------------
# Sampling config + LLM call logging
# ---------------------------------------------------------------------------

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from architectures.base import RunMetrics, RunResult, RunStatus
from providers.base import LLMProvider, LLMResponse, ProviderError
from runner.executor import (
    _COLLECTION_ERROR,
    _NO_RESULTS,
    Executor,
    _derive_outcome,
    _RecordingProvider,
)
from runner.logger import LLMCallRecord, SamplingConfig, is_run_log_file


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


def _call(stop_reason: str = "end_turn", **params: object) -> LLMCallRecord:
    request_params = {
        "temperature": 0.0, "max_output_tokens": 16384, "seed": None,
        "reasoning_effort": None, "thinking_budget_tokens": None,
    }
    request_params.update(params)
    return LLMCallRecord(model="m", stop_reason=stop_reason, request_params=request_params)


class TestSamplingConfig:
    def test_from_single_call(self) -> None:
        cfg = SamplingConfig.from_calls([_call(temperature=0.0, reasoning_effort="none")])
        assert cfg.temperature == 0.0
        assert cfg.reasoning_effort == "none"
        assert cfg.max_output_tokens == 16384
        assert cfg.consistent_across_calls

    def test_consistent_calls_collapse(self) -> None:
        cfg = SamplingConfig.from_calls([_call(), _call(), _call()])
        assert cfg.temperature == 0.0
        assert cfg.consistent_across_calls

    def test_inconsistent_calls_flagged(self) -> None:
        cfg = SamplingConfig.from_calls([_call(temperature=0.0), _call(temperature=1.0)])
        assert cfg.temperature is None
        assert not cfg.consistent_across_calls
        assert cfg.max_output_tokens == 16384  # other keys still agree

    def test_temperature_not_sent_stays_none(self) -> None:
        cfg = SamplingConfig.from_calls([_call(temperature=None, reasoning_effort="medium")])
        assert cfg.temperature is None
        assert cfg.consistent_across_calls

    def test_no_calls(self) -> None:
        cfg = SamplingConfig.from_calls([])
        assert cfg.temperature is None
        assert cfg.consistent_across_calls

    def test_failed_calls_ignored(self) -> None:
        failed = LLMCallRecord(model="m", error="boom")
        cfg = SamplingConfig.from_calls([failed, _call(temperature=0.0)])
        assert cfg.temperature == 0.0
        assert cfg.consistent_across_calls


class TestRunLogSampling:
    def test_temperature_in_json(self) -> None:
        calls = [_call(temperature=0.0)]
        log = _make_run_log(sampling=SamplingConfig.from_calls(calls), llm_calls=calls)
        data = json.loads(log.model_dump_json())
        assert data["sampling"]["temperature"] == 0.0
        assert data["llm_calls"][0]["stop_reason"] == "end_turn"
        assert data["llm_calls"][0]["request_params"]["temperature"] == 0.0

    def test_old_logs_without_sampling_still_load(self) -> None:
        data = json.loads(_make_run_log().model_dump_json())
        del data["sampling"], data["llm_calls"]
        restored = RunLog.model_validate(data)
        assert restored.sampling.temperature is None
        assert restored.llm_calls == []

    def test_markdown_shows_sampling(self, tmp_path: Path) -> None:
        calls = [_call(temperature=0.0, reasoning_effort="none")]
        log = _make_run_log(sampling=SamplingConfig.from_calls(calls), llm_calls=calls)
        _, md_path = save_run_log(log, tmp_path)
        md = md_path.read_text()
        assert "| Temperature | 0.0 |" in md
        assert "| Reasoning effort (sent) | none |" in md
        assert "| Reasoning effort (requested) | not sent |" in md
        assert "| Seed | not sent |" in md
        assert "## LLM calls" in md


class _FakeProvider(LLMProvider):
    def __init__(self, response: LLMResponse | None = None, error: str | None = None) -> None:
        self._response = response
        self._error = error

    @property
    def name(self) -> str:
        return "fake"

    async def complete(self, model, system_prompt, user_prompt, max_tokens=4096,
                       temperature=0.0, seed=None, thinking_enabled=False,
                       reasoning_effort=None) -> LLMResponse:
        if self._error:
            raise ProviderError("fake", self._error)
        return self._response


class TestRecordingProvider:
    def test_records_successful_call(self) -> None:
        response = LLMResponse(
            text="x", input_tokens=3, output_tokens=2, model="m-snap", stop_reason="stop",
            request_params={"temperature": 0.0}, usage_details={"reasoning_tokens": 0},
        )
        rec = _RecordingProvider(_FakeProvider(response))
        out = _run_async(rec.complete(model="m", system_prompt="s", user_prompt="u"))
        assert out is response
        assert rec.name == "fake"
        assert len(rec.calls) == 1
        call = rec.calls[0]
        assert (call.model, call.stop_reason, call.input_tokens, call.output_tokens) == (
            "m-snap", "stop", 3, 2,
        )
        assert call.request_params == {"temperature": 0.0}
        assert call.usage_details == {"reasoning_tokens": 0}

    def test_records_and_reraises_provider_error(self) -> None:
        rec = _RecordingProvider(_FakeProvider(error="rate limited"))
        with pytest.raises(ProviderError):
            _run_async(rec.complete(model="m", system_prompt="s", user_prompt="u"))
        assert len(rec.calls) == 1
        assert "rate limited" in rec.calls[0].error


def _result(status: RunStatus = RunStatus.SUCCESS, files: dict | None = None,
            error: str | None = None) -> RunResult:
    return RunResult(
        status=status, metrics=RunMetrics(),
        modified_files={"snake/game.py": "x"} if files is None else files,
        error_message=error,
    )


_ALLOWED = ["snake/game.py"]


class TestDeriveOutcome:
    def test_full_pass_is_success(self) -> None:
        status, reason = _derive_outcome(_result(), 15, 15, [], _ALLOWED, [_call()])
        assert (status, reason) == ("success", None)

    def test_partial_pass_is_failure(self) -> None:
        status, reason = _derive_outcome(_result(), 10, 15, ["t"], _ALLOWED, [_call()])
        assert status == "failure"
        assert reason is None

    def test_collection_error_is_failure_not_success(self) -> None:
        status, reason = _derive_outcome(
            _result(), 0, 0, [_COLLECTION_ERROR], _ALLOWED, [_call()],
        )
        assert status == "failure"
        assert "syntax or import error" in reason

    def test_no_results_is_error_not_success(self) -> None:
        status, reason = _derive_outcome(
            _result(), 0, 0, [f"{_NO_RESULTS}: exit 1: No module named pytest"],
            _ALLOWED, [_call()],
        )
        assert status == "error"
        assert "No module named pytest" in reason

    def test_wrong_file_label_explained(self) -> None:
        status, reason = _derive_outcome(
            _result(files={"game.py": "x"}), 0, 15, ["t"] * 15, _ALLOWED, [_call()],
        )
        assert status == "failure"
        assert "none target an editable file" in reason
        assert "game.py" in reason

    def test_truncation_appended_to_no_blocks_message(self) -> None:
        status, reason = _derive_outcome(
            _result(RunStatus.FAILURE, files={}, error="No labeled code blocks found in LLM response"),
            0, 15, ["t"] * 15, _ALLOWED, [_call(stop_reason="length")],
        )
        assert status == "failure"
        assert reason.startswith("No labeled code blocks found")
        assert "output token cap (stop_reason=length)" in reason

    def test_anthropic_truncation_detected(self) -> None:
        _, reason = _derive_outcome(
            _result(RunStatus.FAILURE, files={}, error="No labeled code blocks found"),
            0, 15, [], _ALLOWED, [_call(stop_reason="max_tokens")],
        )
        assert "stop_reason=max_tokens" in reason

    def test_refusal_surfaced(self) -> None:
        _, reason = _derive_outcome(
            _result(RunStatus.FAILURE, files={}, error="No labeled code blocks found"),
            0, 15, [], _ALLOWED, [_call(stop_reason="refusal")],
        )
        assert "stop_reason=refusal" in reason

    def test_architecture_error_preserved(self) -> None:
        status, reason = _derive_outcome(
            _result(RunStatus.ERROR, files={}, error="[openai] API error: boom"),
            0, 0, [_NO_RESULTS], _ALLOWED, [LLMCallRecord(model="m", error="boom")],
        )
        assert status == "error"
        assert reason == "[openai] API error: boom"


class TestHiddenEvalNoResults:
    def _executor_with_eval_script(self, tmp_path: Path, script: str) -> Executor:
        (tmp_path / "eval").mkdir()
        (tmp_path / "eval" / "run_eval.sh").write_text(script)
        return Executor(repo_root=tmp_path)

    def test_collection_error_detected(self, tmp_path: Path) -> None:
        ex = self._executor_with_eval_script(
            tmp_path,
            'echo "ERROR test_hidden.py"\n'
            'echo "!!!! Interrupted: 1 error during collection !!!!"\nexit 2\n',
        )
        assert ex._run_hidden_eval("snake", tmp_path) == (0, 0, [_COLLECTION_ERROR])

    def test_missing_pytest_detected(self, tmp_path: Path) -> None:
        ex = self._executor_with_eval_script(
            tmp_path, 'echo "/x/.venv/bin/python: No module named pytest" >&2\nexit 1\n',
        )
        passed, total, failing = ex._run_hidden_eval("snake", tmp_path)
        assert (passed, total) == (0, 0)
        assert failing == [f"{_NO_RESULTS}: exit 1: /x/.venv/bin/python: No module named pytest"]

    def test_normal_output_unchanged(self, tmp_path: Path) -> None:
        ex = self._executor_with_eval_script(
            tmp_path,
            'echo "test_hidden.py::test_a PASSED"\necho "test_hidden.py::test_b FAILED"\n',
        )
        assert ex._run_hidden_eval("snake", tmp_path) == (1, 2, ["test_hidden.py::test_b"])


class TestExecutorEndToEndOpenAI:
    """Full executor run with the real OpenAIProvider and SingleAgent.

    Only the OpenAI HTTP call and the pytest subprocesses are mocked.
    """

    def _run(self, tmp_path: Path, content: str, finish_reason: str = "stop",
             hidden: tuple = (15, 15, [])) -> RunLog:
        from architectures.single_agent import SingleAgent
        from providers.openai import OpenAIProvider

        usage = SimpleNamespace(
            prompt_tokens=1500, completion_tokens=900,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content, refusal=None),
                finish_reason=finish_reason,
            )],
            usage=usage,
            model="gpt-5.6-terra",
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()
        provider._client.chat.completions.create = AsyncMock(return_value=response)

        executor = Executor()
        executor.runs_dir = tmp_path
        config = RunConfig(
            task_name="snake", architecture_name="single_agent",
            model="gpt-5.6-terra", provider="openai",
        )
        with patch("runner.executor.get_provider", return_value=provider), \
             patch.object(Executor, "_run_hidden_eval", return_value=hidden), \
             patch.object(Executor, "_run_public_tests", return_value=(3, 3, [])):
            return _run_async(executor.run(config, SingleAgent()))

    def test_success_logs_sampling_and_calls(self, tmp_path: Path) -> None:
        log = self._run(tmp_path, "```snake/game.py\nclass SnakeGame:\n    pass\n```\n")
        assert log.status == "success"
        assert log.sampling.temperature == 0.0
        assert log.sampling.reasoning_effort == "none"
        assert log.sampling.max_output_tokens == 32_000  # RunConfig default, sent unchanged
        assert len(log.llm_calls) == 1
        assert log.llm_calls[0].stop_reason == "stop"
        assert log.actual_metrics.tokens_prompt == 1500
        assert log.actual_metrics.tokens_completion == 900
        assert log.policy_compliant is True
        saved = json.loads(next(f for f in tmp_path.glob("*.json") if is_run_log_file(f)).read_text())
        assert saved["sampling"]["temperature"] == 0.0

    def test_truncated_response_explained_in_log(self, tmp_path: Path) -> None:
        log = self._run(
            tmp_path, "```snake/game.py\nclass SnakeGame:\n    def move(self):\n",
            finish_reason="length", hidden=(0, 15, ["t"] * 15),
        )
        assert log.status == "failure"
        assert "No labeled code blocks found" in log.stop_reason
        assert "stop_reason=length" in log.stop_reason

    def test_unlabeled_block_explained_in_log(self, tmp_path: Path) -> None:
        log = self._run(
            tmp_path, "```python\nclass SnakeGame:\n    pass\n```\n",
            hidden=(0, 15, ["t"] * 15),
        )
        assert log.status == "failure"
        assert "No labeled code blocks found" in log.stop_reason

    def test_broken_code_not_logged_as_success(self, tmp_path: Path) -> None:
        log = self._run(
            tmp_path, "```snake/game.py\ndef broken(:\n```\n",
            hidden=(0, 0, [_COLLECTION_ERROR]),
        )
        assert log.status == "failure"
        assert "syntax or import error" in log.stop_reason


# ---------------------------------------------------------------------------
# Git working-tree state, reasoning-effort config, cost estimation
# ---------------------------------------------------------------------------

import subprocess

from runner.logger import get_git_dirty


class TestGitDirty:
    def _repo(self, tmp_path: Path) -> Path:
        def git(*args: str) -> None:
            subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
        git("init", "-q")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "t")
        (tmp_path / ".gitignore").write_text("runs/\n")
        (tmp_path / "a.py").write_text("x = 1\n")
        git("add", ".")
        git("commit", "-q", "-m", "init")
        return tmp_path

    def test_clean_tree(self, tmp_path: Path) -> None:
        assert get_git_dirty(cwd=self._repo(tmp_path)) is False

    def test_modified_tracked_file(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        (repo / "a.py").write_text("x = 2\n")
        assert get_git_dirty(cwd=repo) is True

    def test_untracked_file_counts(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        (repo / "new.py").write_text("")
        assert get_git_dirty(cwd=repo) is True

    def test_ignored_run_logs_do_not_count(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        (repo / "runs").mkdir()
        (repo / "runs" / "log.json").write_text("{}")
        assert get_git_dirty(cwd=repo) is False

    def test_not_a_repo_is_unknown(self, tmp_path: Path) -> None:
        assert get_git_dirty(cwd=tmp_path) is None

    def test_old_logs_without_field_still_load(self) -> None:
        data = _make_run_log().model_dump(mode="json")
        del data["git_dirty"], data["requested_reasoning_effort"]
        assert RunLog.model_validate(data).git_dirty is None


class TestRunConfigReasoningEffort:
    def test_default_is_none(self) -> None:
        assert RunConfig(task_name="snake", architecture_name="x").reasoning_effort is None

    def test_valid_effort_accepted(self) -> None:
        cfg = RunConfig(task_name="snake", architecture_name="x",
                        model="claude-opus-5-5", reasoning_effort="low")
        assert cfg.reasoning_effort == "low"

    @pytest.mark.parametrize("kwargs", [
        {"model": "claude-opus-5-5"},
        {"model": "gpt-6-astra", "reasoning_effort": "none"},
        {"model": "claude-sonnet-4-6", "reasoning_effort": "low"},
        {"model": "gpt-6-luna", "reasoning_effort": "low", "thinking_enabled": True},
    ])
    def test_invalid_rejected_at_config_time(self, kwargs: dict) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RunConfig(task_name="snake", architecture_name="x", **kwargs)


class TestCliReasoningEffort:
    def test_flag_parsed(self) -> None:
        from runner.cli import _build_parser
        args = _build_parser().parse_args(
            ["run", "--task", "snake", "--arch", "single_agent",
             "--model", "opus-5.5", "--reasoning-effort", "low"]
        )
        assert args.reasoning_effort == "low"

    def test_invalid_combination_exits_before_run(self, capsys) -> None:
        from runner import cli
        argv = ["bench", "run", "--task", "snake", "--arch", "single_agent", "--model", "opus-5.5"]
        with patch("sys.argv", argv), patch.object(cli.Executor, "run") as run, \
             pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 2
        run.assert_not_called()
        assert "requires an explicit reasoning effort" in capsys.readouterr().err


class TestExecutorEndToEndAnthropicEffort:
    """Full executor run, real AnthropicProvider + SingleAgent, Claude 5.5 at effort low."""

    def _run(self, tmp_path: Path, dirty: bool | None = True) -> RunLog:
        from architectures.single_agent import SingleAgent
        from providers.anthropic import AnthropicProvider

        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="", signature="sig"),
                SimpleNamespace(type="text", text="```snake/game.py\nclass SnakeGame:\n    pass\n```\n"),
            ],
            usage=SimpleNamespace(input_tokens=2000, output_tokens=1000),
            model="claude-opus-5-5",
            stop_reason="end_turn",
        )
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicProvider()
        provider._client.messages.create = AsyncMock(return_value=response)

        executor = Executor()
        executor.runs_dir = tmp_path
        config = RunConfig(
            task_name="snake", architecture_name="single_agent",
            model="claude-opus-5-5", provider="anthropic", reasoning_effort="low",
        )
        with patch("runner.executor.get_provider", return_value=provider), \
             patch("runner.executor.get_git_dirty", return_value=dirty), \
             patch.object(Executor, "_run_hidden_eval", return_value=(15, 15, [])), \
             patch.object(Executor, "_run_public_tests", return_value=(3, 3, [])):
            return _run_async(executor.run(config, SingleAgent()))

    def test_effort_logged_requested_and_sent(self, tmp_path: Path) -> None:
        log = self._run(tmp_path)
        assert log.status == "success"
        assert log.requested_reasoning_effort == "low"
        assert log.sampling.requested_reasoning_effort == "low"
        assert log.sampling.reasoning_effort == "low"
        assert log.sampling.thinking_type == "adaptive"
        assert log.sampling.temperature is None
        sent = log.llm_calls[0].request_params["sent_params"]
        assert sent["output_config"] == {"effort": "low"}
        assert "temperature" not in sent

    def test_cost_estimated(self, tmp_path: Path) -> None:
        log = self._run(tmp_path)
        expected = (2000 * 4.00 + 1000 * 20.00) / 1e6
        assert log.llm_calls[0].estimated_cost_usd == pytest.approx(expected)
        assert log.actual_metrics.estimated_cost_usd == pytest.approx(expected)
        assert "platform.claude.com" in log.actual_metrics.price_source

    def test_git_dirty_recorded(self, tmp_path: Path) -> None:
        log = self._run(tmp_path, dirty=True)
        assert log.git_dirty is True
        saved = json.loads(next(f for f in tmp_path.glob("*.json") if is_run_log_file(f)).read_text())
        assert saved["git_dirty"] is True
        md = next(tmp_path.glob("*.md")).read_text()
        assert "Uncommitted changes at start: yes" in md
        assert "| Reasoning effort (requested) | low |" in md


class TestUnpricedModelCost:
    def test_legacy_model_not_priced(self, tmp_path: Path) -> None:
        log = TestExecutorEndToEndOpenAI()._run(
            tmp_path, "```snake/game.py\nclass SnakeGame:\n    pass\n```\n",
        )
        assert log.actual_metrics.estimated_cost_usd == 0.0
        assert log.actual_metrics.price_source is None
        assert log.llm_calls[0].estimated_cost_usd is None
        md = next(tmp_path.glob("*.md")).read_text()
        assert "| Est. cost (USD) | not priced |" in md
