"""Structured run-log schema and persistence."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field


class BudgetConfig(BaseModel):
    """Budget limits configured for a run.

    Only max_wall_clock_seconds and max_output_tokens_per_call take effect;
    max_tokens and max_tool_calls are recorded ceilings, not enforced.
    """

    max_tokens: int = 100_000
    max_tool_calls: int = 10
    max_wall_clock_seconds: float = 300.0
    max_output_tokens_per_call: int | None = None  # None in logs written before it existed


class ActualMetrics(BaseModel):
    """Metrics actually observed during a run."""

    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    api_calls: int = 0
    wall_clock_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    price_source: str | None = None  # pricing table entry used; None = model not priced


class EvalResults(BaseModel):
    """Test outcomes from public and hidden evaluation."""

    public_passed: int = 0
    public_total: int = 0
    hidden_passed: int = 0
    hidden_total: int = 0
    failing_test_names: list[str] = Field(default_factory=list)


class LLMCallRecord(BaseModel):
    """One provider API call as observed by the executor."""

    model: str | None = None  # as reported by the provider (may be a dated snapshot)
    stop_reason: str | None = None  # provider-native (e.g. 'end_turn', 'stop', 'length')
    input_tokens: int = 0
    output_tokens: int = 0
    request_params: dict[str, Any] = Field(default_factory=dict)
    usage_details: dict[str, int] = Field(default_factory=dict)
    estimated_cost_usd: float | None = None  # None = model not priced
    error: str | None = None  # set if the call raised instead of returning


class SamplingConfig(BaseModel):
    """Sampling parameters actually sent to the provider API.

    Derived from the per-call request_params. A value of None means the
    parameter was not sent (provider default applied), or that calls in the
    run disagreed — in which case consistent_across_calls is False and
    llm_calls holds the per-call values.
    """

    temperature: float | None = None
    max_output_tokens: int | None = None
    seed: int | None = None
    reasoning_effort: str | None = None
    thinking_budget_tokens: int | None = None
    thinking_type: str | None = None
    requested_reasoning_effort: str | None = None
    consistent_across_calls: bool = True

    @classmethod
    def from_calls(cls, calls: list[LLMCallRecord]) -> SamplingConfig:
        """Collapse per-call request params into a run-level summary."""
        params = [c.request_params for c in calls if c.request_params]
        values: dict[str, Any] = {}
        consistent = True
        for key in ("temperature", "max_output_tokens", "seed",
                    "reasoning_effort", "thinking_budget_tokens",
                    "thinking_type", "requested_reasoning_effort"):
            distinct = {repr(p.get(key)) for p in params}
            if len(distinct) == 1:
                values[key] = params[0].get(key)
            elif len(distinct) > 1:
                consistent = False
        return cls(**values, consistent_across_calls=consistent)


class RunLog(BaseModel):
    """Complete record of a single benchmark run."""

    run_id: str
    timestamp: datetime
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    """True if the working tree had uncommitted changes (tracked or untracked,
    excluding gitignored files) when the run started; None if unknown."""

    task_name: str
    architecture_name: str
    prompt_regime: str
    model: str
    provider: str
    requested_reasoning_effort: str | None = None  # RunConfig.reasoning_effort

    budget_config: BudgetConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    actual_metrics: ActualMetrics
    test_results: EvalResults
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)

    status: str  # matches RunStatus enum values
    policy_compliant: bool | None = None
    stop_reason: str | None = None
    notes: str | None = None
    artifacts_file: str | None = None
    """Filename (in the same directory) of the RunArtifacts sidecar holding
    prompts, responses and generated files; None for older logs."""


class CallTranscript(BaseModel):
    """Prompts and response text of one provider call (index-aligned with RunLog.llm_calls)."""

    system_prompt: str
    user_prompt: str
    response_text: str | None = None  # None if the call raised or was cancelled
    error: str | None = None


class RunArtifacts(BaseModel):
    """Bulky per-run content kept out of the main log: everything the model
    was sent, everything it returned, and the files that were evaluated."""

    run_id: str
    calls: list[CallTranscript] = Field(default_factory=list)
    agent_log: list[dict[str, Any]] = Field(default_factory=list)
    """RunResult.agent_log as returned by the architecture (e.g. step/role labels)."""
    modified_files: dict[str, str] = Field(default_factory=dict)
    """Files exactly as parsed from the model output (including rejected paths)."""
    evaluated_files: dict[str, str] = Field(default_factory=dict)
    """Contents of the editable files in the workspace when the tests ran."""


def get_git_info() -> tuple[str | None, str | None]:
    """Return (commit_hash, branch_name) or (None, None) if not in a git repo."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        commit_hash = commit.stdout.strip() if commit.returncode == 0 else None
        branch_name = branch.stdout.strip() if branch.returncode == 0 else None
        return commit_hash, branch_name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, None


def get_git_dirty(cwd: Path | None = None) -> bool | None:
    """Return True if the working tree has uncommitted changes, None if unknown.

    Counts staged, unstaged and untracked files; gitignored files (e.g. run
    logs under runs/) do not count.
    """
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True, text=True, timeout=10, cwd=cwd,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return bool(proc.stdout.strip())


def _fmt_param(value: Any) -> str:
    """Render a sampling parameter, making 'not sent' explicit."""
    return "not sent" if value is None else str(value)


def _render_markdown(log: RunLog) -> str:
    """Render a human-readable markdown summary of a run."""
    tr = log.test_results
    am = log.actual_metrics
    bc = log.budget_config

    lines = [
        f"# Run {log.run_id}",
        "",
        f"**Timestamp:** {log.timestamp.isoformat()}  ",
        f"**Status:** {log.status}  ",
        f"**Task:** {log.task_name}  ",
        f"**Architecture:** {log.architecture_name}  ",
        f"**Model:** {log.model} ({log.provider})  ",
        f"**Prompt regime:** {log.prompt_regime}  ",
        "",
        "## Test results",
        "",
        f"| Suite | Passed | Total |",
        f"|-------|--------|-------|",
        f"| Public | {tr.public_passed} | {tr.public_total} |",
        f"| Hidden | {tr.hidden_passed} | {tr.hidden_total} |",
        "",
    ]

    if tr.failing_test_names:
        lines.append("**Failing tests:**")
        for name in tr.failing_test_names:
            lines.append(f"- `{name}`")
        lines.append("")

    lines.extend([
        "## Metrics",
        "",
        f"| Metric | Value | Budget |",
        f"|--------|-------|--------|",
        f"| Tokens (prompt) | {am.tokens_prompt} | — |",
        f"| Tokens (completion) | {am.tokens_completion} | — |",
        f"| Tokens (total) | {am.tokens_total} | {bc.max_tokens} (not enforced) |",
        f"| API calls | {am.api_calls} | {bc.max_tool_calls} (not enforced) |",
        f"| Wall clock (s) | {am.wall_clock_seconds:.1f} | {bc.max_wall_clock_seconds:.1f} |",
        f"| Est. cost (USD) | "
        f"{f'${am.estimated_cost_usd:.4f}' if am.price_source else 'not priced'} | — |",
        "",
        "## Sampling (as sent to API)",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Temperature | {_fmt_param(log.sampling.temperature)} |",
        f"| Max output tokens / call | {_fmt_param(log.sampling.max_output_tokens)} |",
        f"| Seed | {_fmt_param(log.sampling.seed)} |",
        f"| Reasoning effort (requested) | {_fmt_param(log.requested_reasoning_effort)} |",
        f"| Reasoning effort (sent) | {_fmt_param(log.sampling.reasoning_effort)} |",
        f"| Thinking type | {_fmt_param(log.sampling.thinking_type)} |",
        f"| Thinking budget tokens | {_fmt_param(log.sampling.thinking_budget_tokens)} |",
        "",
    ])

    if not log.sampling.consistent_across_calls:
        lines.extend(["*Parameters differed between calls; see llm_calls in the JSON log.*", ""])

    if log.llm_calls:
        lines.extend([
            "## LLM calls",
            "",
            "| # | Model | Stop reason | In | Out |",
            "|---|-------|-------------|----|-----|",
        ])
        for i, call in enumerate(log.llm_calls, 1):
            stop = call.stop_reason if call.error is None else f"error: {call.error}"
            lines.append(
                f"| {i} | {call.model} | {stop} | {call.input_tokens} | {call.output_tokens} |"
            )
        lines.append("")

    lines.extend([
        "## Policy",
        "",
        f"**Compliant:** {log.policy_compliant}  ",
    ])

    if log.stop_reason:
        lines.append(f"**Stop reason:** {log.stop_reason}  ")
    if log.notes:
        lines.extend(["", "## Notes", "", log.notes])
    if log.artifacts_file:
        lines.extend(["", f"**Prompts, responses and files:** `{log.artifacts_file}`"])
    if log.git_commit:
        lines.extend(["", "## Git", "", f"- Commit: `{log.git_commit}`"])
    if log.git_branch:
        lines.append(f"- Branch: `{log.git_branch}`")
    if log.git_dirty is not None:
        lines.append(f"- Uncommitted changes at start: {'yes' if log.git_dirty else 'no'}")

    lines.append("")  # trailing newline
    return "\n".join(lines)


def shorten_model(model: str) -> str:
    """Extract a short model name by stripping vendor prefix and date suffix.

    Examples:
        'claude-haiku-4-5-20251001' -> 'haiku-4-5'
        'gpt-4o-2024-08-06' -> 'gpt-4o'
        'claude-sonnet-4-6' -> 'sonnet-4-6'
    """
    name = re.sub(r"^claude-", "", model)
    name = re.sub(r"-\d{8}$", "", name)
    name = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name)
    return name


def make_run_filename(log: RunLog) -> str:
    """Build a descriptive filename stem from run metadata.

    Format: {timestamp}_{provider}_{model_short}_{architecture}_{task}_{prompt_regime}_{run_id_short}
    Timestamp: YY-MM-DD_HHMM in Finnish time (Europe/Helsinki). Run ID short: first 8 chars.
    """
    helsinki = ZoneInfo("Europe/Helsinki")
    local_ts = log.timestamp.astimezone(helsinki)
    ts = local_ts.strftime("%y-%m-%d_%H%M")
    model_short = shorten_model(log.model)
    run_id_short = log.run_id[:8]
    return (
        f"{ts}_{log.provider}_{model_short}_{log.architecture_name}"
        f"_{log.task_name}_{log.prompt_regime}_{run_id_short}"
    )


def artifacts_filename(log: RunLog) -> str:
    """Sidecar filename for a run's artifacts (next to its JSON log)."""
    return f"{make_run_filename(log)}.artifacts.json"


def is_run_log_file(path: Path) -> bool:
    """True for a run's main JSON log (not a batch summary or artifacts sidecar)."""
    name = path.name
    return (
        name.endswith(".json")
        and not name.endswith(".artifacts.json")
        and not name.startswith("batch_")
    )


def save_run_artifacts(artifacts: RunArtifacts, log: RunLog, runs_dir: Path) -> Path:
    """Write the artifacts sidecar for *log* and return its path."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / artifacts_filename(log)
    path.write_text(json.dumps(artifacts.model_dump(mode="json"), indent=2) + "\n")
    return path


def save_run_log(log: RunLog, runs_dir: Path) -> tuple[Path, Path]:
    """Write run log as JSON and markdown. Returns (json_path, md_path)."""
    runs_dir.mkdir(parents=True, exist_ok=True)

    stem = make_run_filename(log)
    json_path = runs_dir / f"{stem}.json"
    md_path = runs_dir / f"{stem}.md"

    json_path.write_text(
        json.dumps(log.model_dump(mode="json"), indent=2, default=str) + "\n"
    )
    md_path.write_text(_render_markdown(log))

    return json_path, md_path
