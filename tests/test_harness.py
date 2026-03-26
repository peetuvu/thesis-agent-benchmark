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
        assert cfg.max_wall_clock_seconds == 300.0
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
