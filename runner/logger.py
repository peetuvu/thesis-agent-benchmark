"""Structured run-log schema and persistence."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field


class BudgetConfig(BaseModel):
    """Budget limits configured for a run."""

    max_tokens: int = 100_000
    max_tool_calls: int = 10
    max_wall_clock_seconds: float = 300.0


class ActualMetrics(BaseModel):
    """Metrics actually observed during a run."""

    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    api_calls: int = 0
    wall_clock_seconds: float = 0.0
    estimated_cost_usd: float = 0.0


class EvalResults(BaseModel):
    """Test outcomes from public and hidden evaluation."""

    public_passed: int = 0
    public_total: int = 0
    hidden_passed: int = 0
    hidden_total: int = 0
    failing_test_names: list[str] = Field(default_factory=list)


class RunLog(BaseModel):
    """Complete record of a single benchmark run (STAGES.md D1 schema)."""

    run_id: str
    timestamp: datetime
    git_commit: str | None = None
    git_branch: str | None = None

    task_name: str
    architecture_name: str
    prompt_regime: str
    model: str
    provider: str

    budget_config: BudgetConfig
    actual_metrics: ActualMetrics
    test_results: EvalResults

    status: str  # matches RunStatus enum values
    policy_compliant: bool | None = None
    stop_reason: str | None = None
    notes: str | None = None


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
        f"| Tokens (total) | {am.tokens_total} | {bc.max_tokens} |",
        f"| API calls | {am.api_calls} | {bc.max_tool_calls} |",
        f"| Wall clock (s) | {am.wall_clock_seconds:.1f} | {bc.max_wall_clock_seconds:.1f} |",
        f"| Est. cost (USD) | ${am.estimated_cost_usd:.4f} | — |",
        "",
        "## Policy",
        "",
        f"**Compliant:** {log.policy_compliant}  ",
    ])

    if log.stop_reason:
        lines.append(f"**Stop reason:** {log.stop_reason}  ")
    if log.notes:
        lines.extend(["", "## Notes", "", log.notes])
    if log.git_commit:
        lines.extend(["", "## Git", "", f"- Commit: `{log.git_commit}`"])
    if log.git_branch:
        lines.append(f"- Branch: `{log.git_branch}`")

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
