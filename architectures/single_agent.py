"""Single-agent baseline architecture: one LLM call, one shot."""

from __future__ import annotations

import re

from architectures.base import Architecture, RunMetrics, RunResult, RunStatus, TaskContext
from providers.base import ProviderError


# Matches labeled code blocks: ```path/to/file.ext\n<content>\n```
# The label must contain a dot to distinguish from language identifiers
# (e.g. "python", "javascript" won't match but "game.py" will).

_CODE_BLOCK_RE = re.compile(
    r"```([\w][\w/.-]*\.[\w]+)\s*\n(.*?)```",
    re.DOTALL,
)

_SYSTEM_PROMPT = "You are a programming assistant."


def parse_code_blocks(text: str) -> dict[str, str]:
    """Extract labeled code blocks from LLM response text.

    Matches blocks of the form::

        ```path/to/file.ext
        <content>
        ```

    Returns:
        Dict mapping file paths to their content.
    """
    result: dict[str, str] = {}
    for match in _CODE_BLOCK_RE.finditer(text):
        filepath = match.group(1)
        content = match.group(2)
        result[filepath] = content
    return result


def format_source_files(source_files: dict[str, str]) -> str:
    """Format source files as labeled markdown code blocks for the prompt."""
    if not source_files:
        return "(no source files)"
    blocks = []
    for path, content in source_files.items():
        if content and not content.endswith("\n"):
            content += "\n"
        blocks.append(f"```{path}\n{content}```")
    return "\n\n".join(blocks)


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

        # 1. Format the prompt
        prompt = self._format_prompt(ctx)

        # 2. Make ONE LLM call
        try:
            response = await ctx.provider.complete(
                model=ctx.model,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_tokens=ctx.max_tokens,
                temperature=0.0,
                seed=ctx.seed,
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
        """Fill placeholders in the prompt template."""
        source_files_str = format_source_files(ctx.source_files)
        editable_files_str = "\n".join(
            f"- {path}" for path in ctx.source_files.keys()
        )
        test_cmd = "python -m pytest tests/ -v"

        return ctx.prompt_template.format(
            task_spec=ctx.task_spec,
            source_files=source_files_str,
            editable_files=editable_files_str,
            test_cmd=test_cmd,
        )
