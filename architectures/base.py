"""Abstract base class for multi-agent architectures."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RunStatus(Enum):
    """Outcome status of a benchmark run."""

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class TaskContext:
    """Everything the architecture needs to attempt a task.

    This is what the runner hands to an architecture's `run` method.
    """

    task_dir: Path
    """Root directory of the task (e.g. agent-bench/tasks/snake/)."""

    task_spec: str
    """Contents of the task README / specification."""

    source_files: dict[str, str]
    """Mapping of relative path -> file contents for editable source files."""

    public_tests_dir: Path | None = None
    """Path to public sanity tests the agent is allowed to run."""

    prompt_regime: str = "minimal"
    """Which prompt template to use (e.g. 'minimal', 'production')."""

    prompt_template: str = ""
    """Resolved prompt template contents."""

    max_tokens: int = 100_000
    """Total token budget for this run."""

    max_tool_calls: int = 50
    """Maximum number of tool/API calls allowed."""

    timeout_seconds: float = 300.0
    """Wall-clock time limit for the run."""

    model: str = "claude-sonnet-4-20250514"
    """Model identifier to use."""

    seed: int | None = None
    """Random seed for reproducibility where supported."""


@dataclass
class RunMetrics:
    """Metrics collected during a single benchmark run."""

    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    tool_calls: int = 0
    wall_time_seconds: float = 0.0
    estimated_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class RunResult:
    """Complete result of running an architecture on a task."""

    status: RunStatus
    """How the run ended."""

    metrics: RunMetrics
    """Token/time/cost metrics."""

    modified_files: dict[str, str]
    """Mapping of relative path -> final file contents after the run."""

    agent_log: list[dict[str, Any]] = field(default_factory=list)
    """Chronological log of agent actions (messages, tool calls, etc.)."""

    error_message: str | None = None
    """If status is ERROR, a description of what went wrong."""


class Architecture(ABC):
    """Base class that all benchmark architectures must implement.

    Each architecture defines a strategy for solving a software development
    task (e.g. single-agent, sequential pipeline, debate, etc.). The runner
    calls `run()` with a `TaskContext` and expects a `RunResult` back.

    Subclasses must implement:
        - `name` (property): human-readable architecture name
        - `run(ctx)`: execute the architecture on the given task
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this architecture (e.g. 'sequential-pipeline')."""
        ...

    @abstractmethod
    async def run(self, ctx: TaskContext) -> RunResult:
        """Execute this architecture on a task and return the result.

        Implementations must:
        - Respect ctx.max_tokens, ctx.max_tool_calls, and ctx.timeout_seconds
        - Populate RunResult.metrics accurately
        - Return modified_files with the final state of all edited source files
        - Catch and report errors rather than raising (set status=ERROR)
        """
        ...

    async def run_timed(self, ctx: TaskContext) -> RunResult:
        """Wrapper around `run` that enforces wall-clock timing.

        Subclasses should generally not override this — override `run` instead.
        """
        start = time.monotonic()
        result = await self.run(ctx)
        result.metrics.wall_time_seconds = time.monotonic() - start
        return result

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
