"""Single-agent baseline architecture: one LLM call, one shot."""

from __future__ import annotations

from architectures.base import Architecture, RunMetrics, RunResult, RunStatus, TaskContext
from architectures.utils import (  # noqa: F401 (format_source_files re-exported)
    format_source_files,
    parse_code_blocks,
    render_prompt_template,
)
from providers.base import ProviderError

_SYSTEM_PROMPT = "You are a programming assistant."


class SingleAgent(Architecture):
    """Single-agent baseline: one prompt, one LLM call, one shot.

    Formats the prompt template with task context, makes a single call
    to the configured LLM provider, and parses labeled code blocks from
    the response to extract modified files.

    No conversation history, no tool use, no iteration.
    """

    @property
    def name(self) -> str:
        return "single-agent"

    async def run(self, ctx: TaskContext) -> RunResult:
        """Execute a single-shot LLM call and parse the response."""
        if ctx.provider is None:
            return RunResult(
                status=RunStatus.ERROR,
                metrics=RunMetrics(),
                modified_files={},
                error_message="No provider configured in TaskContext",
            )

        # 1. Format the prompt (a bad template fails here, before any API call)
        try:
            prompt = self._format_prompt(ctx)
        except ValueError as e:
            return RunResult(
                status=RunStatus.ERROR,
                metrics=RunMetrics(),
                modified_files={},
                error_message=f"Prompt template error: {e}",
            )

        # 2. Make ONE LLM call
        try:
            response = await ctx.provider.complete(
                model=ctx.model,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_tokens=ctx.max_output_tokens,
                temperature=0.0,
                seed=ctx.seed,
                thinking_enabled=ctx.thinking_enabled,
                reasoning_effort=ctx.reasoning_effort,
            )
        except ProviderError as e:
            return RunResult(
                status=RunStatus.ERROR,
                metrics=RunMetrics(api_calls=1),
                modified_files={},
                error_message=str(e),
            )

        # 3. Parse code blocks from response
        modified_files = parse_code_blocks(response.text)

        # 4. Build metrics
        metrics = RunMetrics(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            api_calls=1,
        )

        # 5. Determine status
        if not modified_files:
            status = RunStatus.FAILURE
            error_msg = "No labeled code blocks found in LLM response"
        else:
            status = RunStatus.SUCCESS
            error_msg = None

        return RunResult(
            status=status,
            metrics=metrics,
            modified_files=modified_files,
            agent_log=[
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": response.text,
                    "model": response.model,
                    "stop_reason": response.stop_reason,
                },
            ],
            error_message=error_msg,
        )

    def _format_prompt(self, ctx: TaskContext) -> str:
        """Render the prompt template (see architectures.utils.render_prompt_template)."""
        return render_prompt_template(ctx)
