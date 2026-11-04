"""Benchmark executor: orchestrates a single run of an architecture on a task."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from architectures.base import Architecture, RunMetrics, RunResult, RunStatus, TaskContext
from providers import get_provider
from providers.pricing import estimate_call_cost, get_price
from providers.base import (
    NORMAL_STOP_REASONS,
    TRUNCATION_STOP_REASONS,
    LLMProvider,
    LLMResponse,
    ProviderError,
)
from runner.config import RunConfig, TaskPaths
from runner.logger import (
    ActualMetrics,
    BudgetConfig,
    CallTranscript,
    EvalResults,
    LLMCallRecord,
    RunArtifacts,
    RunLog,
    SamplingConfig,
    artifacts_filename,
    get_git_dirty,
    get_git_info,
    save_run_artifacts,
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


# Sentinel "failing test names" returned when the hidden eval yields no results.
_COLLECTION_ERROR = "hidden_eval_collection_error"
_NO_RESULTS = "hidden_eval_no_results"


class _RecordingProvider(LLMProvider):
    """Transparent wrapper that records every call made through a provider.

    Lets the executor log what was actually sent, why each call stopped and
    what it cost, for any architecture, without architectures having to
    report it. Cost is priced by the requested model string (the response
    may name a dated snapshot).
    """

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.calls: list[LLMCallRecord] = []
        self.transcript: list[CallTranscript] = []
        """Prompts and response text per call, index-aligned with calls."""

    @property
    def name(self) -> str:
        return self._inner.name

    async def complete(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int | None = None,
        thinking_enabled: bool = False,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        try:
            response = await self._inner.complete(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                thinking_enabled=thinking_enabled,
                reasoning_effort=reasoning_effort,
            )
        except asyncio.CancelledError:
            # Cancelled by the wall-clock timeout while the request was in
            # flight: usage is unknown (the provider may still bill it).
            msg = "cancelled before a response arrived (wall-clock timeout); usage unknown"
            self.calls.append(LLMCallRecord(model=model, error=msg))
            self.transcript.append(CallTranscript(
                system_prompt=system_prompt, user_prompt=user_prompt, error=msg,
            ))
            raise
        except Exception as e:  # ProviderError, or anything else the provider raised
            msg = str(e) if isinstance(e, ProviderError) else f"{type(e).__name__}: {e}"
            self.calls.append(LLMCallRecord(model=model, error=msg))
            self.transcript.append(CallTranscript(
                system_prompt=system_prompt, user_prompt=user_prompt, error=msg,
            ))
            raise
        self.transcript.append(CallTranscript(
            system_prompt=system_prompt, user_prompt=user_prompt, response_text=response.text,
        ))
        self.calls.append(LLMCallRecord(
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            request_params=response.request_params,
            usage_details=response.usage_details,
            estimated_cost_usd=estimate_call_cost(
                model,
                response.input_tokens,
                response.output_tokens,
                response.usage_details.get("cached_input_tokens", 0),
            ),
        ))
        return response


def _derive_outcome(
    result: RunResult,
    hidden_passed: int,
    hidden_total: int,
    hidden_failing: list[str],
    allowed_edit_files: list[str],
    calls: list[LLMCallRecord],
    violations: list[str] | None = None,
) -> tuple[str, str | None]:
    """Determine the run's final (status, stop_reason).

    Guarantees a run never ends as "success" without a positive hidden score,
    and that the stop_reason explains why when the output was unusable.
    Any sandbox policy violation turns the status into "policy_violation"
    (whatever it would otherwise have been, which is kept in stop_reason),
    so a non-compliant run can never count as a success.
    """
    status = result.status.value
    reasons: list[str] = [result.error_message] if result.error_message else []

    if result.status == RunStatus.SUCCESS:
        editable = set(allowed_edit_files)
        if editable and not editable.intersection(result.modified_files):
            status = RunStatus.FAILURE.value
            reasons.append(
                "Labeled code blocks found but none target an editable file "
                f"(got: {', '.join(sorted(result.modified_files))}; "
                f"editable: {', '.join(sorted(editable))})"
            )
        if hidden_total == 0:
            detail = next(
                (n for n in hidden_failing if n.startswith((_COLLECTION_ERROR, _NO_RESULTS))),
                _NO_RESULTS,
            )
            if detail.startswith(_COLLECTION_ERROR):
                # The generated code broke test collection (syntax/import error).
                status = RunStatus.FAILURE.value
                reasons.append(
                    "Hidden eval could not collect tests: generated code likely "
                    "has a syntax or import error"
                )
            else:
                # No test outcomes and no collection error: the eval itself failed.
                status = RunStatus.ERROR.value
                reasons.append(f"Hidden eval produced no test results ({detail})")
        elif hidden_passed < hidden_total:
            status = RunStatus.FAILURE.value

    last = next((c for c in reversed(calls) if c.error is None), None)
    if last is not None and last.stop_reason not in NORMAL_STOP_REASONS:
        if last.stop_reason in TRUNCATION_STOP_REASONS:
            reasons.append(
                f"Last LLM call hit the output token cap (stop_reason={last.stop_reason}); "
                "output may be truncated"
            )
        else:
            reasons.append(f"Last LLM call ended with stop_reason={last.stop_reason}")

    if violations:
        reasons.append(
            f"Sandbox policy violated ({len(violations)} violation(s), listed in notes); "
            f"status would otherwise have been '{status}'"
        )
        status = RunStatus.POLICY_VIOLATION.value

    return status, "; ".join(reasons) if reasons else None


def _metrics_from_calls(calls: list[LLMCallRecord], wall_time_seconds: float) -> RunMetrics:
    """Metrics reconstructed from the recorded calls, for runs whose
    architecture never returned (timeout or unhandled exception)."""
    return RunMetrics(
        input_tokens=sum(c.input_tokens for c in calls),
        output_tokens=sum(c.output_tokens for c in calls),
        api_calls=len(calls),
        wall_time_seconds=wall_time_seconds,
        estimated_cost_usd=sum(c.estimated_cost_usd or 0.0 for c in calls),
    )


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

        # --- 0. Record code version as of run start ---
        git_commit, git_branch = get_git_info()
        git_dirty = get_git_dirty(cwd=self.repo_root)

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
            result, llm_calls, transcript = await self._run_architecture(
                config, paths, temp_task_dir, architecture,
            )

            # --- 5. Write modified files into the temp dir (paths that would
            #        resolve outside it are rejected as policy violations) ---
            apply_violations = self._apply_modified_files(
                result.modified_files, temp_task_dir,
            )
            evaluated_files = {
                rel: (temp_task_dir / rel).read_text()
                for rel in paths.allowed_edit_files
                if (temp_task_dir / rel).is_file()
            }

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
            all_violations = eval_violations + apply_violations + workspace_violations
            policy_compliant = len(all_violations) == 0

            # --- 11. Assemble RunLog ---
            price = get_price(config.model)
            estimated_cost = sum(c.estimated_cost_usd or 0.0 for c in llm_calls)

            status, stop_reason = _derive_outcome(
                result, hidden_passed, hidden_total, hidden_failing,
                paths.allowed_edit_files, llm_calls, all_violations,
            )

            log = RunLog(
                run_id=config.run_id,
                timestamp=config.timestamp,
                git_commit=git_commit,
                git_branch=git_branch,
                git_dirty=git_dirty,
                task_name=config.task_name,
                architecture_name=config.architecture_name,
                prompt_regime=config.prompt_regime,
                model=config.model,
                provider=config.provider,
                requested_reasoning_effort=config.reasoning_effort,
                budget_config=BudgetConfig(
                    max_tokens=config.max_tokens,
                    max_tool_calls=config.max_tool_calls,
                    max_wall_clock_seconds=config.max_wall_clock_seconds,
                    max_output_tokens_per_call=config.max_output_tokens,
                ),
                sampling=SamplingConfig.from_calls(llm_calls),
                actual_metrics=ActualMetrics(
                    tokens_prompt=result.metrics.input_tokens,
                    tokens_completion=result.metrics.output_tokens,
                    tokens_total=result.metrics.total_tokens,
                    api_calls=result.metrics.api_calls,
                    wall_clock_seconds=result.metrics.wall_time_seconds,
                    estimated_cost_usd=(
                        estimated_cost if price else result.metrics.estimated_cost_usd
                    ),
                    price_source=price.source if price else None,
                ),
                test_results=EvalResults(
                    public_passed=public_passed,
                    public_total=public_total,
                    hidden_passed=hidden_passed,
                    hidden_total=hidden_total,
                    failing_test_names=hidden_failing + public_failing,
                ),
                llm_calls=llm_calls,
                status=status,
                policy_compliant=policy_compliant,
                stop_reason=stop_reason,
                notes="; ".join(all_violations) if all_violations else None,
            )

            # --- 12. Save prompts/responses/files sidecar, then the log ---
            log.artifacts_file = artifacts_filename(log)
            save_run_artifacts(
                RunArtifacts(
                    run_id=config.run_id,
                    calls=transcript,
                    agent_log=result.agent_log,
                    modified_files=result.modified_files,
                    evaluated_files=evaluated_files,
                ),
                log,
                self.runs_dir,
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
    ) -> tuple[RunResult, list[LLMCallRecord], list[CallTranscript]]:
        """Build TaskContext and call architecture.run_timed() with timeout.

        Returns the result, a record of every LLM call the architecture made,
        and the prompts/responses of those calls. If the architecture times
        out or raises, the result's metrics are reconstructed from the
        recorded calls and the executor's own wall-clock measurement.
        """
        provider = _RecordingProvider(get_provider(config.provider))
        ctx = self._build_context(config, paths, temp_task_dir, provider)

        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                architecture.run_timed(ctx),
                timeout=config.max_wall_clock_seconds,
            )
        except asyncio.TimeoutError:
            result = RunResult(
                status=RunStatus.TIMEOUT,
                metrics=_metrics_from_calls(provider.calls, time.monotonic() - start),
                modified_files={},
                error_message=f"Timed out after {config.max_wall_clock_seconds}s",
            )
        except Exception as e:
            result = RunResult(
                status=RunStatus.ERROR,
                metrics=_metrics_from_calls(provider.calls, time.monotonic() - start),
                modified_files={},
                error_message=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )

        return result, provider.calls, provider.transcript

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
            editable_files=paths.allowed_edit_files,
            public_tests_dir=public_tests,
            prompt_regime=config.prompt_regime,
            prompt_template=prompt_template,
            max_tokens=config.max_tokens,
            max_output_tokens=config.max_output_tokens,
            max_tool_calls=config.max_tool_calls,
            timeout_seconds=config.max_wall_clock_seconds,
            model=config.model,
            seed=config.seed,
            thinking_enabled=config.thinking_enabled,
            reasoning_effort=config.reasoning_effort,
            provider=provider,
        )

    def _apply_modified_files(
        self, modified: dict[str, str], temp_task_dir: Path,
    ) -> list[str]:
        """Write the architecture's modified files into the temp task directory.

        A path that resolves outside the temp task directory (via '..',
        an absolute path or a symlink) is not written; it is returned as a
        policy violation instead. Returns the list of violations.
        """
        root = temp_task_dir.resolve()
        violations: list[str] = []
        for rel_path, content in modified.items():
            target = (temp_task_dir / rel_path).resolve()
            if target == root or not target.is_relative_to(root):
                violations.append(f"rejected (outside workspace): {rel_path}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        return violations

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

        passed, total, failing = _parse_pytest_output(proc.stdout, proc.stderr)
        if total == 0:
            combined = proc.stdout + "\n" + proc.stderr
            if "during collection" in combined:
                failing = failing + [_COLLECTION_ERROR]
            else:
                lines = [ln for ln in combined.splitlines() if ln.strip()]
                last_line = lines[-1].strip() if lines else "no output"
                failing = failing + [f"{_NO_RESULTS}: exit {proc.returncode}: {last_line}"]
        return passed, total, failing

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
