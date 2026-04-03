"""Sequential pipeline architecture: plan → implement → review → fix."""

from __future__ import annotations

from architectures.base import Architecture, RunMetrics, RunResult, RunStatus, TaskContext
from architectures.utils import format_source_files, parse_code_blocks
from providers.base import ProviderError

_STEP_CONFIGS = [
    {"step": 1, "role": "plan", "system": "You are a planning agent."},
    {"step": 2, "role": "implement", "system": "You are an implementation agent."},
    {"step": 3, "role": "review", "system": "You are a code review agent."},
    {"step": 4, "role": "fix", "system": "You are a fixing agent."},
]


def _build_plan_prompt(task_spec: str, source_files: dict[str, str]) -> str:
    """Build the user prompt for the planning step."""
    files_str = format_source_files(source_files)
    return (
        f"## Task specification\n\n{task_spec}\n\n"
        f"## Current source files\n\n{files_str}\n\n"
        "## Your job\n\n"
        "Produce a structured implementation plan. Include:\n"
        "1. Overall approach\n"
        "2. Key functions/classes needed\n"
        "3. Edge cases to handle\n"
        "4. Suggested order of implementation\n\n"
        "Do NOT write code — only plan."
    )


def _build_implement_prompt(
    task_spec: str, source_files: dict[str, str], plan: str
) -> str:
    """Build the user prompt for the implementation step."""
    files_str = format_source_files(source_files)
    return (
        f"## Task specification\n\n{task_spec}\n\n"
        f"## Current source files\n\n{files_str}\n\n"
        f"## Implementation plan\n\n{plan}\n\n"
        "## Your job\n\n"
        "Implement the solution following the plan above. "
        "Return your code in labeled markdown blocks like:\n\n"
        "```path/to/file.py\n<code>\n```\n\n"
        "Include ALL modified files with their complete contents."
    )


def _build_review_prompt(task_spec: str, implementation: str) -> str:
    """Build the user prompt for the review step."""
    return (
        f"## Task specification\n\n{task_spec}\n\n"
        f"## Implementation to review\n\n{implementation}\n\n"
        "## Your job\n\n"
        "Review the code above against the specification. Check for:\n"
        "1. Missed requirements\n"
        "2. Bugs or logic errors\n"
        "3. Edge case handling\n"
        "4. Correctness\n\n"
        "Return a structured review listing any issues found. "
        "If the code is correct, say so explicitly."
    )


def _build_fix_prompt(
    task_spec: str, implementation: str, review: str
) -> str:
    """Build the user prompt for the fix step."""
    return (
        f"## Task specification\n\n{task_spec}\n\n"
        f"## Current implementation\n\n{implementation}\n\n"
        f"## Code review\n\n{review}\n\n"
        "## Your job\n\n"
        "If the review found issues, fix them and return the corrected code. "
        "If no issues were found, return the code unchanged.\n\n"
        "Return your code in labeled markdown blocks like:\n\n"
        "```path/to/file.py\n<code>\n```\n\n"
        "Include ALL modified files with their complete contents."
    )


class SequentialPipeline(Architecture):
    """Fixed 4-step sequential pipeline: plan → implement → review → fix.

    Each step is one LLM call. The output of each step feeds into the
    next step's prompt. The final code comes from the fix step (or the
    implement step if fix returns no code blocks).
    """

    def __init__(
        self,
        step_models: dict[str, str] | None = None,
    ) -> None:
        # TODO: per-step model selection — when step_models is set,
        # each step should use its specified model instead of ctx.model.
        # Not implemented yet; stored for future use.
        self._step_models = step_models

    @property
    def name(self) -> str:
        return "sequential-pipeline"

    async def run(self, ctx: TaskContext) -> RunResult:
        """Execute the 4-step pipeline and return the result."""
        if ctx.provider is None:
            return RunResult(
                status=RunStatus.ERROR,
                metrics=RunMetrics(),
                modified_files={},
                error_message="No provider configured in TaskContext",
            )

        agent_log: list[dict] = []
        total_input_tokens = 0
        total_output_tokens = 0
        api_calls = 0

        # Rough even split of token budget across steps — could be made smarter later
        per_step_tokens = ctx.max_tokens // 4

        # ---- Step 1: PLAN ----
        plan_prompt = _build_plan_prompt(ctx.task_spec, ctx.source_files)
        plan_text, err = await self._call_step(
            ctx, _STEP_CONFIGS[0], plan_prompt, per_step_tokens, agent_log
        )
        if err is not None:
            return self._error_result(err, agent_log, api_calls + 1)
        api_calls += 1
        total_input_tokens += agent_log[-1]["input_tokens"]
        total_output_tokens += agent_log[-1]["output_tokens"]

        # ---- Step 2: IMPLEMENT ----
        implement_prompt = _build_implement_prompt(
            ctx.task_spec, ctx.source_files, plan_text
        )
        implement_text, err = await self._call_step(
            ctx, _STEP_CONFIGS[1], implement_prompt, per_step_tokens, agent_log
        )
        if err is not None:
            return self._error_result(err, agent_log, api_calls + 1)
        api_calls += 1
        total_input_tokens += agent_log[-1]["input_tokens"]
        total_output_tokens += agent_log[-1]["output_tokens"]

        implement_code = parse_code_blocks(implement_text)

        # ---- Step 3: REVIEW ----
        review_prompt = _build_review_prompt(ctx.task_spec, implement_text)
        review_text, err = await self._call_step(
            ctx, _STEP_CONFIGS[2], review_prompt, per_step_tokens, agent_log
        )
        if err is not None:
            return self._error_result(err, agent_log, api_calls + 1)
        api_calls += 1
        total_input_tokens += agent_log[-1]["input_tokens"]
        total_output_tokens += agent_log[-1]["output_tokens"]

        # ---- Step 4: FIX ----
        fix_prompt = _build_fix_prompt(ctx.task_spec, implement_text, review_text)
        fix_text, err = await self._call_step(
            ctx, _STEP_CONFIGS[3], fix_prompt, per_step_tokens, agent_log
        )
        if err is not None:
            return self._error_result(err, agent_log, api_calls + 1)
        api_calls += 1
        total_input_tokens += agent_log[-1]["input_tokens"]
        total_output_tokens += agent_log[-1]["output_tokens"]

        fix_code = parse_code_blocks(fix_text)

        # Use fix step's code blocks, fall back to implement step's
        modified_files = fix_code if fix_code else implement_code

        metrics = RunMetrics(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            api_calls=api_calls,
        )

        if not modified_files:
            return RunResult(
                status=RunStatus.FAILURE,
                metrics=metrics,
                modified_files={},
                agent_log=agent_log,
                error_message="No labeled code blocks found in pipeline output",
            )

        return RunResult(
            status=RunStatus.SUCCESS,
            metrics=metrics,
            modified_files=modified_files,
            agent_log=agent_log,
        )

    async def _call_step(
        self,
        ctx: TaskContext,
        step_config: dict,
        user_prompt: str,
        max_tokens: int,
        agent_log: list[dict],
    ) -> tuple[str, str | None]:
        """Execute a single pipeline step. Returns (response_text, error_or_None)."""
        try:
            response = await ctx.provider.complete(
                model=ctx.model,
                system_prompt=step_config["system"],
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                seed=ctx.seed,
                thinking_enabled=ctx.thinking_enabled,
            )
        except ProviderError as e:
            agent_log.append({
                "step": step_config["step"],
                "role": step_config["role"],
                "prompt": user_prompt,
                "response": None,
                "input_tokens": 0,
                "output_tokens": 0,
            })
            return "", f"Step {step_config['step']} ({step_config['role']}): {e}"

        agent_log.append({
            "step": step_config["step"],
            "role": step_config["role"],
            "prompt": user_prompt,
            "response": response.text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        })
        return response.text, None

    @staticmethod
    def _error_result(
        error_message: str, agent_log: list[dict], api_calls: int
    ) -> RunResult:
        """Build an ERROR RunResult from the accumulated state."""
        total_in = sum(entry["input_tokens"] for entry in agent_log)
        total_out = sum(entry["output_tokens"] for entry in agent_log)
        return RunResult(
            status=RunStatus.ERROR,
            metrics=RunMetrics(
                input_tokens=total_in,
                output_tokens=total_out,
                api_calls=api_calls,
            ),
            modified_files={},
            agent_log=agent_log,
            error_message=error_message,
        )
