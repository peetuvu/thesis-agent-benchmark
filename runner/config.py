"""Run configuration and task path resolution."""

from __future__ import annotations

import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


def _generate_run_id() -> str:
    return str(uuid.uuid4())


def _generate_timestamp() -> datetime:
    return datetime.now(timezone.utc)


def _find_repo_root() -> Path:
    """Find the repository root by looking for pyproject.toml walking upward."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root (no pyproject.toml found)")


class RunConfig(BaseModel):
    """Configuration for a single benchmark run."""

    task_name: str
    architecture_name: str
    model: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"
    prompt_regime: str = "minimal"
    max_tokens: int = 100_000
    max_tool_calls: int = 10
    max_wall_clock_seconds: float = 300.0
    seed: int | None = None
    run_id: str = Field(default_factory=_generate_run_id)
    timestamp: datetime = Field(default_factory=_generate_timestamp)


class TaskPaths:
    """Resolves all filesystem paths for a given task."""

    def __init__(self, task_name: str, repo_root: Path | None = None) -> None:
        self.task_name = task_name
        self.repo_root = repo_root or _find_repo_root()

    @property
    def task_dir(self) -> Path:
        """Root directory of the task (e.g. agent-bench/tasks/snake/)."""
        return self.repo_root / "agent-bench" / "tasks" / self.task_name

    @property
    def source_dir(self) -> Path:
        """Source package directory (e.g. agent-bench/tasks/snake/snake/)."""
        return self.task_dir / self.task_name

    @property
    def public_tests_dir(self) -> Path:
        """Public sanity tests directory."""
        return self.task_dir / "tests"

    @property
    def hidden_tests_dir(self) -> Path:
        """Hidden evaluation tests directory."""
        return self.repo_root / "eval" / "hidden_tests" / self.task_name

    @property
    def eval_dir(self) -> Path:
        """Root eval directory."""
        return self.repo_root / "eval"

    @property
    def readme_path(self) -> Path:
        """Task specification README."""
        return self.task_dir / "README.md"

    @property
    def prompt_path(self) -> Path:
        """Prompt templates directory."""
        return self.repo_root / "agent-bench" / "prompts"

    @property
    def allowed_edit_files(self) -> list[str]:
        """List of relative paths (relative to task_dir) that the agent may edit.

        Convention: all .py files in the source package, excluding __init__.py.
        """
        if not self.source_dir.exists():
            return []
        files = []
        for py_file in sorted(self.source_dir.rglob("*.py")):
            if py_file.name == "__init__.py":
                continue
            rel = py_file.relative_to(self.task_dir)
            files.append(rel.as_posix())
        return files
