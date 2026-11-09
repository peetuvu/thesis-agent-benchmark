"""Tests for the 2026-11-04 harness fixes (one class per fix, numbered as in the request).

No real API calls: providers are fakes, and the hidden/public pytest
subprocesses are patched out in executor-level tests.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from architectures.base import Architecture, RunMetrics, RunResult, RunStatus, TaskContext
from architectures.sequential import SequentialPipeline
from architectures.single_agent import SingleAgent
from architectures.utils import (
    TEST_CMD,
    format_source_files,
    render_prompt_template,
    resolve_editable_files,
    strip_template_header,
)
from providers.base import LLMProvider, LLMResponse, ProviderError
from runner.config import RunConfig, TaskPaths
from runner.executor import Executor, _derive_outcome
from runner.logger import LLMCallRecord, RunArtifacts, RunLog, is_run_log_file

REPO = Path(__file__).resolve().parents[1]
PROMPTS = REPO / "agent-bench" / "prompts"
TASKS = sorted(p.name for p in (REPO / "agent-bench" / "tasks").iterdir() if p.is_dir())
REGIMES = ("minimal", "production")

sys.path.insert(0, str(REPO / "scripts"))
import aggregate_results  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _response(text: str, inp: int = 100, out: int = 50) -> LLMResponse:
    return LLMResponse(
        text=text, input_tokens=inp, output_tokens=out, model="test-model",
        stop_reason="end_turn", request_params={"max_output_tokens": 1},
    )


def _mock_provider(texts: list[str]) -> MagicMock:
    provider = MagicMock()
    provider.name = "mock"
    provider.complete = AsyncMock(side_effect=[_response(t) for t in texts])
    return provider


def _task_context(task: str, regime: str, provider: object | None = None) -> TaskContext:
    """A TaskContext built by the real executor code from the real task files."""
    executor = Executor()
    paths = TaskPaths(task, executor.repo_root)
    config = RunConfig(
        task_name=task, architecture_name="x", model="claude-sonnet-4-6", prompt_regime=regime,
    )
    return executor._build_context(config, paths, paths.task_dir, provider)


def _user_prompts(provider: MagicMock) -> list[str]:
    return [c.kwargs["user_prompt"] for c in provider.complete.call_args_list]


class _ScriptedProvider(LLMProvider):
    """Returns canned responses; optionally hangs (to trigger the timeout) or
    raises a non-provider exception after a given number of calls."""

    def __init__(self, texts: list[str], hang_after: int | None = None,
                 crash_after: int | None = None) -> None:
        self._texts = list(texts)
        self._hang_after = hang_after
        self._crash_after = crash_after
        self.n = 0

    @property
    def name(self) -> str:
        return "anthropic"

    async def complete(self, model, system_prompt, user_prompt, max_tokens=4096,
                       temperature=0.0, seed=None, thinking_enabled=False,
                       reasoning_effort=None) -> LLMResponse:
        if self._hang_after is not None and self.n >= self._hang_after:
            await asyncio.sleep(60)
        if self._crash_after is not None and self.n >= self._crash_after:
            raise RuntimeError("boom")
        self.n += 1
        return LLMResponse(
            text=self._texts.pop(0), input_tokens=1000, output_tokens=500,
            model=model, stop_reason="end_turn",
            request_params={"max_output_tokens": max_tokens, "temperature": None},
        )


def _execute(tmp_path: Path, provider: LLMProvider, arch: Architecture, *,
             model: str = "claude-sonnet-4-6", wall: float = 30.0,
             hidden: tuple = (15, 15, []), regime: str = "minimal") -> RunLog:
    executor = Executor()
    executor.runs_dir = tmp_path
    config = RunConfig(
        task_name="snake", architecture_name="single_agent", model=model,
        provider="anthropic", prompt_regime=regime, max_wall_clock_seconds=wall,
        reasoning_effort="low" if model.endswith("-5-5") else None,
    )
    with patch("runner.executor.get_provider", return_value=provider), \
         patch.object(Executor, "_run_hidden_eval", return_value=hidden), \
         patch.object(Executor, "_run_public_tests", return_value=(3, 3, [])):
        return _run_async(executor.run(config, arch))


GOOD_GAME = "```snake/game.py\nclass SnakeGame:\n    pass\n```\n"


# ---------------------------------------------------------------------------
# 1. Sequential pipeline uses the prompt regime in the implement step
# ---------------------------------------------------------------------------


class TestFix1SequentialRegime:
    def _run(self, regime: str) -> list[str]:
        provider = _mock_provider(["THE PLAN", GOOD_GAME, "REVIEW", GOOD_GAME])
        ctx = _task_context("snake", regime, provider)
        _run_async(SequentialPipeline().run(ctx))
        return _user_prompts(provider)

    @pytest.mark.parametrize("regime", REGIMES)
    def test_implement_prompt_is_rendered_template_plus_plan(self, regime: str) -> None:
        prompts = self._run(regime)
        ctx = _task_context("snake", regime)
        rendered = render_prompt_template(ctx)
        implement = prompts[1]
        assert implement.startswith(rendered.rstrip())
        assert implement.rstrip().endswith("THE PLAN")
        assert implement.index("THE PLAN") > len(rendered.rstrip())

    def test_regimes_produce_different_implement_prompts(self) -> None:
        minimal, production = self._run("minimal"), self._run("production")
        assert minimal[1] != production[1]
        assert "You are an implementation agent" in production[1]
        assert "You are an implementation agent" not in minimal[1]

    def test_plan_review_fix_prompts_do_not_depend_on_regime(self) -> None:
        minimal, production = self._run("minimal"), self._run("production")
        for step in (0, 2, 3):
            assert minimal[step] == production[step]

    def test_implement_prompt_matches_single_agent_prompt(self) -> None:
        single = _mock_provider([GOOD_GAME])
        _run_async(SingleAgent().run(_task_context("snake", "production", single)))
        seq_implement = self._run("production")[1]
        assert seq_implement.startswith(_user_prompts(single)[0].rstrip())


# ---------------------------------------------------------------------------
# 2. Template rendering: no header, spec/sources once, real example label
# ---------------------------------------------------------------------------


class TestFix2TemplateRendering:
    @pytest.mark.parametrize("task", TASKS)
    @pytest.mark.parametrize("regime", REGIMES)
    def test_spec_and_sources_exactly_once(self, task: str, regime: str) -> None:
        ctx = _task_context(task, regime)
        prompt = render_prompt_template(ctx)
        assert prompt.count(ctx.task_spec) == 1
        assert prompt.count(format_source_files(ctx.source_files)) == 1

    @pytest.mark.parametrize("task", TASKS)
    def test_production_header_not_sent(self, task: str) -> None:
        prompt = render_prompt_template(_task_context(task, "production"))
        for marker in ("Production Prompt Template", "Kimara", "Placeholders", "{"):
            assert marker not in prompt.split("## Task Specification")[0]
        assert prompt.startswith("You are an implementation agent.")

    @pytest.mark.parametrize("task", TASKS)
    @pytest.mark.parametrize("regime", REGIMES)
    def test_example_label_is_task_editable_file(self, task: str, regime: str) -> None:
        ctx = _task_context(task, regime)
        prompt = render_prompt_template(ctx)
        expected = TaskPaths(task).allowed_edit_files[0]
        assert f"```{expected}\n# complete file contents here" in prompt
        if task != "snake":
            assert "snake/game.py" not in prompt

    def test_no_template_hardcodes_snake(self) -> None:
        for regime in REGIMES:
            assert "snake/game.py" not in (PROMPTS / f"{regime}.md").read_text()

    def test_strip_keeps_headings_and_hash_lines_in_body(self) -> None:
        template = "# header\n# more\n\n## Task\n{task_spec}\n```x.py\n# comment in code\n```\n"
        assert strip_template_header(template) == (
            "## Task\n{task_spec}\n```x.py\n# comment in code\n```\n"
        )

    def test_template_without_header_unchanged(self) -> None:
        template = "Implement.\n\n## Task\n{task_spec}\n"
        assert strip_template_header(template) == template

    @pytest.mark.parametrize("template, message", [
        ("{task_spec}\n", "{source_files} exactly once"),
        ("{task_spec}{task_spec}{source_files}", "{task_spec} exactly once"),
        ("{task_spec}{source_files}{bogus}", "Unknown placeholder"),
        ("{task_spec}{source_files}{test_cmd}{test_cmd}", "more than once"),
        ("", "{task_spec} exactly once"),
    ])
    def test_bad_templates_rejected(self, template: str, message: str) -> None:
        ctx = TaskContext(task_dir=Path("."), task_spec="s", source_files={"p/a.py": ""},
                          prompt_template=template)
        with pytest.raises(ValueError, match=message.replace("{", r"\{").replace("}", r"\}")):
            render_prompt_template(ctx)

    @pytest.mark.parametrize("arch", [SingleAgent(), SequentialPipeline()])
    def test_bad_template_errors_before_any_call(self, arch: Architecture) -> None:
        provider = _mock_provider([])
        ctx = TaskContext(task_dir=Path("."), task_spec="s", source_files={"p/a.py": ""},
                          prompt_template="no placeholders", provider=provider)
        result = _run_async(arch.run(ctx))
        assert result.status == RunStatus.ERROR
        assert "Prompt template error" in result.error_message
        assert provider.complete.call_count == 0
        assert result.metrics.api_calls == 0


# ---------------------------------------------------------------------------
# 3. Fix step merges over implement step
# ---------------------------------------------------------------------------


class TestFix3MergeFixOverImplement:
    def _run(self, implement: str, fix: str) -> RunResult:
        provider = _mock_provider(["plan", implement, "review", fix])
        ctx = TaskContext(
            task_dir=Path("."), task_spec="s",
            source_files={"pkg/a.py": "", "pkg/b.py": ""},
            prompt_template="{task_spec}{source_files}", provider=provider,
        )
        return _run_async(SequentialPipeline().run(ctx))

    def test_file_not_returned_by_fixer_is_kept(self) -> None:
        result = self._run(
            "```pkg/a.py\nA1\n```\n```pkg/b.py\nB1\n```\n",
            "```pkg/a.py\nA2\n```\n",
        )
        assert result.modified_files == {"pkg/a.py": "A2\n", "pkg/b.py": "B1\n"}

    def test_fixer_can_add_a_file(self) -> None:
        result = self._run("```pkg/a.py\nA1\n```\n", "```pkg/b.py\nB2\n```\n")
        assert result.modified_files == {"pkg/a.py": "A1\n", "pkg/b.py": "B2\n"}

    def test_fixer_without_code_keeps_implementation(self) -> None:
        result = self._run("```pkg/a.py\nA1\n```\n", "No changes needed.")
        assert result.modified_files == {"pkg/a.py": "A1\n"}
        assert result.status == RunStatus.SUCCESS


# ---------------------------------------------------------------------------
# 4. Per-call output cap
# ---------------------------------------------------------------------------


class TestFix4OutputCap:
    def test_config_default_and_validation(self) -> None:
        cfg = RunConfig(task_name="snake", architecture_name="single_agent")
        assert cfg.max_output_tokens == 32_000
        with pytest.raises(ValueError):
            RunConfig(task_name="snake", architecture_name="single_agent", max_output_tokens=0)

    def test_cli_flag(self) -> None:
        from runner.cli import _build_parser
        args = _build_parser().parse_args(
            ["run", "--task", "snake", "--arch", "single_agent", "--model", "opus-5.5",
             "--max-output-tokens", "777"])
        assert args.max_output_tokens == 777
        default = _build_parser().parse_args(
            ["run", "--task", "snake", "--arch", "single_agent", "--model", "opus-5.5"])
        assert default.max_output_tokens == 32_000

    def test_single_agent_uses_cap_not_total_budget(self) -> None:
        provider = _mock_provider([GOOD_GAME])
        ctx = _task_context("snake", "minimal", provider)
        ctx.max_tokens, ctx.max_output_tokens = 100_000, 4_321
        _run_async(SingleAgent().run(ctx))
        assert provider.complete.call_args.kwargs["max_tokens"] == 4_321

    @pytest.mark.parametrize("arch", [SingleAgent(), SequentialPipeline()])
    def test_executor_cap_reaches_every_call_and_log(self, tmp_path: Path,
                                                     arch: Architecture) -> None:
        provider = _ScriptedProvider(["plan", GOOD_GAME, "review", GOOD_GAME])
        log = _execute(tmp_path, provider, arch)
        assert log.budget_config.max_output_tokens_per_call == 32_000
        assert {c.request_params["max_output_tokens"] for c in log.llm_calls} == {32_000}

    def test_no_enforcement_claims_left(self) -> None:
        for rel in ("providers/anthropic.py", "providers/openai.py"):
            assert "handled by the executor" not in (REPO / rel).read_text()


# ---------------------------------------------------------------------------
# 5. Prompts, responses and files saved per run
# ---------------------------------------------------------------------------


class TestFix5Artifacts:
    def _artifacts(self, tmp_path: Path, log: RunLog) -> RunArtifacts:
        assert log.artifacts_file and log.artifacts_file.endswith(".artifacts.json")
        saved = json.loads((tmp_path / log.artifacts_file).read_text())
        return RunArtifacts(**saved)

    def test_single_agent(self, tmp_path: Path) -> None:
        log = _execute(tmp_path, _ScriptedProvider([GOOD_GAME]), SingleAgent())
        art = self._artifacts(tmp_path, log)
        assert art.run_id == log.run_id
        assert len(art.calls) == len(log.llm_calls) == 1
        assert "## Current Source Files" in art.calls[0].user_prompt
        assert art.calls[0].system_prompt == "You are a programming assistant."
        assert art.calls[0].response_text == GOOD_GAME
        assert art.modified_files == {"snake/game.py": "class SnakeGame:\n    pass\n"}
        assert art.evaluated_files == {"snake/game.py": "class SnakeGame:\n    pass\n"}

    def test_sequential_keeps_all_steps(self, tmp_path: Path) -> None:
        log = _execute(tmp_path, _ScriptedProvider(["P", GOOD_GAME, "R", GOOD_GAME]),
                       SequentialPipeline())
        art = self._artifacts(tmp_path, log)
        assert [c.response_text for c in art.calls] == ["P", GOOD_GAME, "R", GOOD_GAME]
        assert [e["role"] for e in art.agent_log] == ["plan", "implement", "review", "fix"]

    def test_markdown_links_sidecar_and_sidecar_not_a_run_log(self, tmp_path: Path) -> None:
        log = _execute(tmp_path, _ScriptedProvider([GOOD_GAME]), SingleAgent())
        md = next(tmp_path.glob("*.md")).read_text()
        assert log.artifacts_file in md
        logs = [p for p in tmp_path.glob("*.json") if is_run_log_file(p)]
        assert len(logs) == 1 and not logs[0].name.endswith(".artifacts.json")

    def test_old_log_without_field_loads(self) -> None:
        log = RunLog(
            run_id="x", timestamp="2026-01-01T00:00:00Z", task_name="snake",
            architecture_name="a", prompt_regime="minimal", model="m", provider="p",
            budget_config={}, actual_metrics={}, test_results={}, status="success",
        )
        assert log.artifacts_file is None


# ---------------------------------------------------------------------------
# 6. Timeout / exception runs log real tokens, calls, cost and wall time
# ---------------------------------------------------------------------------


class _CrashAfterOneCall(Architecture):
    @property
    def name(self) -> str:
        return "crash"

    async def run(self, ctx: TaskContext) -> RunResult:
        await ctx.provider.complete(model=ctx.model, system_prompt="s", user_prompt="u")
        raise RuntimeError("architecture bug")


class TestFix6TimeoutAndExceptionMetrics:
    def test_timeout_uses_recorded_calls(self, tmp_path: Path) -> None:
        provider = _ScriptedProvider(["plan", GOOD_GAME], hang_after=2)
        log = _execute(tmp_path, provider, SequentialPipeline(),
                       model="claude-sonnet-5-5", wall=0.5)
        assert log.status == "timeout"
        am = log.actual_metrics
        assert am.tokens_prompt == 2000 and am.tokens_completion == 1000
        assert am.tokens_total == 3000
        # two completed calls + the in-flight one cancelled by the timeout
        assert am.api_calls == 3 == len(log.llm_calls)
        assert "cancelled" in log.llm_calls[2].error
        assert am.wall_clock_seconds >= 0.5
        assert am.estimated_cost_usd == pytest.approx((2000 * 2.0 + 1000 * 10.0) / 1e6)

    def test_exception_uses_recorded_calls(self, tmp_path: Path) -> None:
        log = _execute(tmp_path, _ScriptedProvider(["x"]), _CrashAfterOneCall(),
                       model="claude-sonnet-5-5")
        assert log.status == "error"
        assert "architecture bug" in log.stop_reason
        am = log.actual_metrics
        assert (am.tokens_prompt, am.tokens_completion, am.api_calls) == (1000, 500, 1)
        assert am.wall_clock_seconds > 0
        assert am.estimated_cost_usd > 0

    def test_non_provider_exception_in_provider_is_recorded(self, tmp_path: Path) -> None:
        log = _execute(tmp_path, _ScriptedProvider([], crash_after=0), SingleAgent())
        assert log.status == "error"
        assert log.actual_metrics.api_calls == 1
        assert "RuntimeError: boom" in log.llm_calls[0].error

    def test_normal_run_metrics_unchanged(self, tmp_path: Path) -> None:
        log = _execute(tmp_path, _ScriptedProvider([GOOD_GAME]), SingleAgent())
        am = log.actual_metrics
        assert (am.tokens_prompt, am.tokens_completion, am.api_calls) == (1000, 500, 1)


# ---------------------------------------------------------------------------
# 7. Paths outside the workspace are rejected
# ---------------------------------------------------------------------------


class TestFix7PathContainment:
    def test_apply_rejects_escaping_paths(self, tmp_path: Path) -> None:
        work = tmp_path / "task"
        work.mkdir()
        outside = tmp_path / "escape.py"
        violations = Executor()._apply_modified_files(
            {
                "snake/../../escape.py": "evil",
                str(tmp_path / "abs.py"): "evil",
                "snake/../..": "evil",
                "snake/game.py": "ok",
            },
            work,
        )
        assert not outside.exists() and not (tmp_path / "abs.py").exists()
        assert (work / "snake" / "game.py").read_text() == "ok"
        assert len(violations) == 3
        assert all(v.startswith("rejected (outside workspace): ") for v in violations)

    def test_apply_rejects_symlink_escape(self, tmp_path: Path) -> None:
        work = tmp_path / "task"
        work.mkdir()
        (tmp_path / "elsewhere").mkdir()
        (work / "link").symlink_to(tmp_path / "elsewhere")
        violations = Executor()._apply_modified_files({"link/x.py": "evil"}, work)
        assert violations and not (tmp_path / "elsewhere" / "x.py").exists()

    def test_allows_internal_dotdot(self, tmp_path: Path) -> None:
        work = tmp_path / "task"
        (work / "snake").mkdir(parents=True)
        assert Executor()._apply_modified_files({"snake/../snake/game.py": "ok"}, work) == []
        assert (work / "snake" / "game.py").read_text() == "ok"

    def test_executor_records_violation(self, tmp_path: Path) -> None:
        text = GOOD_GAME + "```snake/../../escape.py\nevil\n```\n"
        log = _execute(tmp_path, _ScriptedProvider([text]), SingleAgent())
        assert log.policy_compliant is False
        assert "rejected (outside workspace): snake/../../escape.py" in log.notes
        assert log.status == "policy_violation"


# ---------------------------------------------------------------------------
# 8. Policy violations get their own status
# ---------------------------------------------------------------------------


def _ok_result(files: dict | None = None) -> RunResult:
    return RunResult(status=RunStatus.SUCCESS, metrics=RunMetrics(),
                     modified_files=files or {"snake/game.py": "x"})


class TestFix8PolicyStatus:
    _CALLS = [LLMCallRecord(model="m", stop_reason="end_turn")]

    def test_violation_overrides_success(self) -> None:
        status, reason = _derive_outcome(
            _ok_result(), 15, 15, [], ["snake/game.py"], self._CALLS, ["added: x.py"])
        assert status == "policy_violation"
        assert "would otherwise have been 'success'" in reason

    def test_violation_overrides_failure(self) -> None:
        status, reason = _derive_outcome(
            _ok_result(), 3, 15, ["t"], ["snake/game.py"], self._CALLS, ["modified: tests/t.py"])
        assert status == "policy_violation"
        assert "'failure'" in reason

    def test_no_violation_unchanged(self) -> None:
        assert _derive_outcome(
            _ok_result(), 15, 15, [], ["snake/game.py"], self._CALLS, [])[0] == "success"

    def test_workspace_violation_end_to_end(self, tmp_path: Path) -> None:
        text = GOOD_GAME + "```tests/test_public_sanity.py\ndef test_x(): pass\n```\n"
        log = _execute(tmp_path, _ScriptedProvider([text]), SingleAgent())
        assert log.status == "policy_violation"
        assert log.policy_compliant is False
        assert "modified: tests/test_public_sanity.py" in log.notes

    def test_status_enum(self) -> None:
        assert RunStatus("policy_violation") is RunStatus.POLICY_VIOLATION


# ---------------------------------------------------------------------------
# 9. editable_files shown to the model == allowed_edit_files
# ---------------------------------------------------------------------------


class TestFix9EditableFiles:
    @pytest.mark.parametrize("task", TASKS)
    def test_context_and_prompt_match_allowed(self, task: str) -> None:
        ctx = _task_context(task, "production")
        allowed = TaskPaths(task).allowed_edit_files
        assert ctx.editable_files == allowed
        assert resolve_editable_files(ctx) == allowed
        prompt = render_prompt_template(ctx)
        section = prompt.split("## Editable Files")[1].split("## Steps")[0]
        listed = [ln[2:] for ln in section.splitlines() if ln.startswith("- ")]
        assert listed == allowed
        assert not any(p.endswith("__init__.py") for p in listed)

    def test_fallback_excludes_init(self) -> None:
        ctx = TaskContext(task_dir=Path("."), task_spec="", source_files={
            "pkg/__init__.py": "", "pkg/a.py": "", "pkg/sub/__init__.py": "", "pkg/sub/b.py": ""})
        assert resolve_editable_files(ctx) == ["pkg/a.py", "pkg/sub/b.py"]

    def test_test_cmd_still_rendered(self) -> None:
        assert TEST_CMD in render_prompt_template(_task_context("snake", "production"))


# ---------------------------------------------------------------------------
# 10. aggregate_results.py
# ---------------------------------------------------------------------------


def _log(status: str = "success", task: str = "snake", provider: str = "anthropic",
         model: str = "claude-sonnet-5-5", arch: str = "single_agent", regime: str = "minimal",
         effort: str | None = "low", hidden: tuple = (15, 15), compliant: bool | None = True,
         cost: float = 0.01, priced: bool = True, tokens: int = 1000, wall: float = 2.0,
         run_id: str = "r") -> dict:
    return {
        "run_id": run_id, "task_name": task, "provider": provider, "model": model,
        "architecture_name": arch, "prompt_regime": regime,
        "requested_reasoning_effort": effort, "status": status,
        "policy_compliant": compliant,
        "test_results": {"hidden_passed": hidden[0], "hidden_total": hidden[1],
                         "public_passed": 0, "public_total": 0},
        "actual_metrics": {"tokens_total": tokens, "wall_clock_seconds": wall,
                           "estimated_cost_usd": cost,
                           "price_source": "src" if priced else None},
    }


def _write(runs_dir: Path, logs: list[dict]) -> None:
    for i, data in enumerate(logs):
        (runs_dir / f"run_{i}.json").write_text(json.dumps(data))


class TestFix10Aggregate:
    def test_groups_by_all_six_dimensions(self, tmp_path: Path) -> None:
        _write(tmp_path, [
            _log(), _log(task="calculator"), _log(provider="openai", model="gpt-6-luna"),
            _log(model="claude-opus-5-5"), _log(arch="sequential"), _log(regime="production"),
            _log(effort="medium"), _log(),
        ])
        runs, _ = aggregate_results.load_runs(tmp_path)
        rows = aggregate_results.summarize(runs)
        assert len(rows) == 7
        assert max(r["n"] for r in rows) == 2

    def test_error_runs_counted_not_dropped(self, tmp_path: Path) -> None:
        _write(tmp_path, [
            _log(), _log(status="error", hidden=(0, 15)), _log(status="timeout", hidden=(0, 0)),
            _log(status="failure", hidden=(5, 10)),
        ])
        runs, _ = aggregate_results.load_runs(tmp_path)
        (row,) = aggregate_results.summarize(runs)
        assert row["n"] == 4
        assert (row["errors"], row["timeouts"]) == (1, 1)
        assert row["pass_at_1"] == pytest.approx(0.25)
        assert row["hidden_mean"] == pytest.approx((1.0 + 0 + 0 + 0.5) / 4)

    def test_non_compliant_counted_and_not_a_pass(self, tmp_path: Path) -> None:
        _write(tmp_path, [
            _log(),
            _log(status="policy_violation", compliant=False),
            # legacy log: success status but non-compliant
            _log(status="success", compliant=False),
        ])
        runs, _ = aggregate_results.load_runs(tmp_path)
        (row,) = aggregate_results.summarize(runs)
        assert row["non_compliant"] == 2
        assert row["pass_at_1"] == pytest.approx(1 / 3)  # legacy non-compliant "success" is not a pass

    def test_cost_tokens_latency(self, tmp_path: Path) -> None:
        _write(tmp_path, [_log(cost=0.02, tokens=100, wall=1.0),
                          _log(cost=0.0, priced=False, tokens=300, wall=3.0)])
        runs, _ = aggregate_results.load_runs(tmp_path)
        (row,) = aggregate_results.summarize(runs)
        assert row["cost_mean"] == pytest.approx(0.02)
        assert row["cost_priced_runs"] == 1
        assert aggregate_results._fmt_cost(row).endswith("*")
        assert (row["tokens_mean"], row["latency_mean"]) == (200, 2.0)

    def test_hidden_std(self, tmp_path: Path) -> None:
        _write(tmp_path, [_log(hidden=(15, 15)), _log(status="failure", hidden=(0, 15))])
        runs, _ = aggregate_results.load_runs(tmp_path)
        (row,) = aggregate_results.summarize(runs)
        assert row["hidden_std"] == pytest.approx(0.7071, abs=1e-3)

    def test_skips_sidecars_batches_and_reports_junk(self, tmp_path: Path) -> None:
        _write(tmp_path, [_log()])
        (tmp_path / "x.artifacts.json").write_text(json.dumps({"run_id": "r"}))
        (tmp_path / "batch_26-01-01_000000.json").write_text(json.dumps({"outcomes": []}))
        (tmp_path / "junk.json").write_text(json.dumps({"hello": 1}))
        runs, skipped = aggregate_results.load_runs(tmp_path)
        assert len(runs) == 1
        assert skipped == ["junk.json (not a run log)"]

    def test_old_logs_without_effort(self, tmp_path: Path) -> None:
        old = _log()
        del old["requested_reasoning_effort"]
        del old["actual_metrics"]["price_source"]
        _write(tmp_path, [old])
        runs, _ = aggregate_results.load_runs(tmp_path)
        (row,) = aggregate_results.summarize(runs)
        assert row["effort"] == "-" and row["cost_mean"] is None

    def test_task_filter_and_main(self, tmp_path: Path, capsys) -> None:
        _write(tmp_path, [_log(), _log(task="calculator", status="error")])
        csv_path = tmp_path / "out.csv"
        assert aggregate_results.main(
            ["snake", "--runs-dir", str(tmp_path), "--csv", str(csv_path)]) == 0
        out = capsys.readouterr().out
        assert "Loaded 1 runs for task=snake" in out
        assert csv_path.read_text().startswith("task,provider,model")
        assert aggregate_results.main(["--runs-dir", str(tmp_path)]) == 0
        assert "1 error" in capsys.readouterr().out
