"""Benchmark executor: orchestrates a single run of an architecture on a task."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from architectures.base import Architecture, RunMetrics, RunResult, RunStatus, TaskContext
from providers import get_provider
from runner.config import RunConfig, TaskPaths
from runner.logger import (
    ActualMetrics,
    BudgetConfig,
    EvalResults,
    RunLog,
    get_git_info,
    save_run_log,
)
from runner.sandbox import check_policy, snapshot_directory


def _parse_pytest_output(stdout: str, stderr: str) -> tuple[int, int, list[str]]:
    """Parse pytest -v output and return (passed, total, failing_test_names)."""
    passed = 0
    failed = 0
    failing_names: list[str] = []

    combined = stdout + "\n" + stderr

    for line in combined.splitlines():
        # Match lines like "test_hidden.py::test_name PASSED" or "FAILED"
        m = re.match(r".*?(\S+::test_\S+)\s+(PASSED|FAILED)", line)
        if m:
            test_name = m.group(1)
            outcome = m.group(2)
            if outcome == "PASSED":
                passed += 1
            else:
                failed += 1
                failing_names.append(test_name)

    total = passed + failed
    return passed, total, failing_names


class Executor:
    """Runs an architecture on a task and produces a RunLog.

    The real repository is never modified during a run.  The architecture
    works inside a temporary copy of the task directory, and both public
    and hidden evaluation run against that same temp copy.  The only
    side-effect is the run log written to ``runs/``.
    """

    def __init__(self, repo_root: Path | None = None) -> None:
        from runner.config import _find_repo_root
        self.repo_root = repo_root or _find_repo_root()
        self.runs_dir = self.repo_root / "runs"

    async def run(self, config: RunConfig, architecture: Architecture) -> RunLog:
        """Execute the full benchmark pipeline for one run."""
        paths = TaskPaths(config.task_name, self.repo_root)

        # --- 1. Pre-snapshot eval/ ---
        eval_pre = snapshot_directory(paths.eval_dir)

        # --- 2. Copy task dir to temp ---
        tmp_dir = tempfile.mkdtemp(prefix=f"bench_{config.task_name}_")
        temp_task_dir = Path(tmp_dir) / "task"
        try:
            shutil.copytree(paths.task_dir, temp_task_dir)
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to copy task dir: {e}") from e

        try:
            # --- 3. Snapshot temp dir before architecture runs ---
            task_pre = snapshot_directory(temp_task_dir)

            # --- 4. Run architecture in temp dir ---
            result = await self._run_architecture(config, paths, temp_task_dir, architecture)

            # --- 5. Write modified files into the temp dir ---
            self._apply_modified_files(result.modified_files, temp_task_dir)

            # --- 6. Snapshot temp dir after architecture, check workspace policy ---
            task_post = snapshot_directory(temp_task_dir)
            task_ok, workspace_violations = check_policy(
                task_pre, task_post, allowed_files=paths.allowed_edit_files,
            )

            # --- 7. Run hidden eval against temp dir ---
            hidden_passed, hidden_total, hidden_failing = self._run_hidden_eval(
                config.task_name, temp_task_dir,
            )

            # --- 8. Run public tests against temp dir ---
            public_passed, public_total, public_failing = self._run_public_tests(
                temp_task_dir,
            )

            # --- 9. Post-snapshot eval/ (safety: nothing should have changed) ---
            eval_post = snapshot_directory(paths.eval_dir)
            eval_ok, eval_violations = check_policy(eval_pre, eval_post, allowed_files=[])

            # --- 10. Policy compliance ---
            all_violations = eval_violations + workspace_violations
            policy_compliant = len(all_violations) == 0

            # --- 11. Assemble RunLog ---
            git_commit, git_branch = get_git_info()

            status = result.status.value
            stop_reason = result.error_message
            if result.status == RunStatus.SUCCESS and hidden_total > 0:
                if hidden_passed < hidden_total:
                    status = RunStatus.FAILURE.value

            log = RunLog(
                run_id=config.run_id,
                timestamp=config.timestamp,
                git_commit=git_commit,
                git_branch=git_branch,
                task_name=config.task_name,
                architecture_name=config.architecture_name,
                prompt_regime=config.prompt_regime,
                model=config.model,
                provider=config.provider,
                budget_config=BudgetConfig(
                    max_tokens=config.max_tokens,
                    max_tool_calls=config.max_tool_calls,
                    max_wall_clock_seconds=config.max_wall_clock_seconds,
                ),
                actual_metrics=ActualMetrics(
                    tokens_prompt=result.metrics.input_tokens,
                    tokens_completion=result.metrics.output_tokens,
                    tokens_total=result.metrics.total_tokens,
                    api_calls=result.metrics.api_calls,
                    wall_clock_seconds=result.metrics.wall_time_seconds,
                    estimated_cost_usd=result.metrics.estimated_cost_usd,
                ),
                test_results=EvalResults(
                    public_passed=public_passed,
                    public_total=public_total,
                    hidden_passed=hidden_passed,
                    hidden_total=hidden_total,
                    failing_test_names=hidden_failing + public_failing,
                ),
                status=status,
                policy_compliant=policy_compliant,
                stop_reason=stop_reason,
                notes="; ".join(all_violations) if all_violations else None,
            )

            save_run_log(log, self.runs_dir)
            return log

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_architecture(
        self,
        config: RunConfig,
        paths: TaskPaths,
        temp_task_dir: Path,
        architecture: Architecture,
    ) -> RunResult:
        """Build TaskContext and call architecture.run_timed() with timeout."""
        provider = get_provider(config.provider)
        ctx = self._build_context(config, paths, temp_task_dir, provider)

        try:
            result = await asyncio.wait_for(
                architecture.run_timed(ctx),
                timeout=config.max_wall_clock_seconds,
            )
        except asyncio.TimeoutError:
            result = RunResult(
                status=RunStatus.TIMEOUT,
                metrics=RunMetrics(),
                modified_files={},
                error_message=f"Timed out after {config.max_wall_clock_seconds}s",
            )
        except Exception as e:
            result = RunResult(
                status=RunStatus.ERROR,
                metrics=RunMetrics(),
                modified_files={},
                error_message=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )

        return result

    def _build_context(
        self, config: RunConfig, paths: TaskPaths, temp_task_dir: Path,
        provider: object | None = None,
    ) -> TaskContext:
        """Assemble a TaskContext from config and temp directory."""
        # Read task spec
        task_spec = ""
        readme = temp_task_dir / "README.md"
        if readme.exists():
            task_spec = readme.read_text()

        # Read source files
        source_files: dict[str, str] = {}
        temp_source = temp_task_dir / config.task_name
        if temp_source.exists():
            for py_file in sorted(temp_source.rglob("*.py")):
                rel = py_file.relative_to(temp_task_dir).as_posix()
                source_files[rel] = py_file.read_text()

        # Read prompt template
        prompt_template = ""
        prompt_file = paths.prompt_path / f"{config.prompt_regime}.md"
        if prompt_file.exists():
            prompt_template = prompt_file.read_text()

        # Public tests dir in temp
        temp_tests = temp_task_dir / "tests"
        public_tests = temp_tests if temp_tests.exists() else None

        return TaskContext(
            task_dir=temp_task_dir,
            task_spec=task_spec,
            source_files=source_files,
            public_tests_dir=public_tests,
            prompt_regime=config.prompt_regime,
            prompt_template=prompt_template,
            max_tokens=config.max_tokens,
            max_tool_calls=config.max_tool_calls,
            timeout_seconds=config.max_wall_clock_seconds,
            model=config.model,
            seed=config.seed,
            provider=provider,
        )

    def _apply_modified_files(self, modified: dict[str, str], temp_task_dir: Path) -> None:
        """Write the architecture's modified files into the temp task directory."""
        for rel_path, content in modified.items():
            target = temp_task_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)

    def _run_hidden_eval(
        self, task_name: str, temp_task_dir: Path
    ) -> tuple[int, int, list[str]]:
        """Run hidden evaluation via eval/run_eval.sh against the temp dir."""
        eval_script = self.repo_root / "eval" / "run_eval.sh"
        try:
            proc = subprocess.run(
                ["bash", str(eval_script), task_name, str(temp_task_dir)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.repo_root,
            )
        except subprocess.TimeoutExpired:
            return 0, 0, ["eval_timeout"]
        except FileNotFoundError:
            return 0, 0, ["eval_script_not_found"]

        return _parse_pytest_output(proc.stdout, proc.stderr)

    def _run_public_tests(self, temp_task_dir: Path) -> tuple[int, int, list[str]]:
        """Run public sanity tests against the temp dir."""
        tests_dir = temp_task_dir / "tests"
        if not tests_dir.exists():
            return 0, 0, []

        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "pytest", "-v", "--tb=line",
                    str(tests_dir),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_task_dir,
                env={**os.environ, "PYTHONPATH": str(temp_task_dir)},
            )
        except subprocess.TimeoutExpired:
            return 0, 0, ["public_tests_timeout"]

        return _parse_pytest_output(proc.stdout, proc.stderr)
