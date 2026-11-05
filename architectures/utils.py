"""Shared utilities for architecture implementations."""

from __future__ import annotations

import re
import string
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from architectures.base import TaskContext

# Opening fence of a labeled code block: ```path/to/file.ext followed by a
# newline. The label must contain a dot to distinguish from language
# identifiers (e.g. "python", "javascript" won't match but "game.py" will).
# group(1) is the backtick run, so the closing fence can be matched against
# its length.
_OPEN_FENCE_RE = re.compile(r"(?<!`)(`{3,})([\w][\w/.-]*\.[\w]+)\s*\n")

# Command the templates show as the public sanity test command.
TEST_CMD = "python -m pytest tests/ -v"

# Placeholders a prompt template may use. task_spec and source_files must each
# appear exactly once; the others at most once.
_REQUIRED_ONCE = ("task_spec", "source_files")
_OPTIONAL = ("editable_files", "test_cmd", "example_file")

# A template header line: a single '#' (not '##', which is a markdown heading
# meant for the model) followed by whitespace or end of line.
_HEADER_LINE_RE = re.compile(r"#(?!#)(\s|$)")


def parse_code_blocks(text: str) -> dict[str, str]:
    """Extract labeled code blocks from LLM response text.

    Matches blocks of the form::

        ```path/to/file.ext
        <content>
        ```

    The block ends at the first closing fence: a line consisting only of
    backticks, at least as many as the opening fence, plus optional
    whitespace. A ``` in the middle of a line (e.g. ``s.startswith("```")``)
    or a line starting with ``` followed by other characters (e.g. ```python)
    never closes a block. A block without a closing fence is ignored. The
    content runs up to the start of the closing fence's line, so it ends with
    a newline unless it is empty. A later block for the same path wins.

    Returns:
        Dict mapping file paths to their content.
    """
    result: dict[str, str] = {}
    pos = 0
    while (opening := _OPEN_FENCE_RE.search(text, pos)) is not None:
        n = len(opening.group(1))
        closing = re.compile(rf"^[ \t]*`{{{n},}}[ \t\r]*$", re.MULTILINE).search(
            text, opening.end(),
        )
        if closing is None:
            pos = opening.end()
            continue
        result[opening.group(2)] = text[opening.end():closing.start()]
        pos = closing.end()
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


def strip_template_header(template: str) -> str:
    """Remove the documentation header at the top of a prompt template.

    The header is the leading run of lines that start with a single '#'
    (e.g. "# Production Prompt Template", "# Placeholders: ..."), plus the
    blank lines after it. Markdown headings ('## ...') and '#' lines later in
    the template (e.g. inside code examples) are kept.
    """
    lines = template.splitlines(keepends=True)
    i = 0
    while i < len(lines) and _HEADER_LINE_RE.match(lines[i]):
        i += 1
    if i:
        while i < len(lines) and not lines[i].strip():
            i += 1
    return "".join(lines[i:])


def resolve_editable_files(ctx: TaskContext) -> list[str]:
    """Paths the model may edit: ctx.editable_files, or (when unset) the
    source files except __init__.py, matching TaskPaths.allowed_edit_files."""
    if ctx.editable_files:
        return list(ctx.editable_files)
    return [p for p in ctx.source_files if p.rsplit("/", 1)[-1] != "__init__.py"]


def _check_placeholders(body: str) -> None:
    """Raise ValueError unless the template's placeholders are well-formed."""
    counts: dict[str, int] = {}
    try:
        fields = [f for _, f, _, _ in string.Formatter().parse(body) if f is not None]
    except ValueError as e:
        raise ValueError(f"Malformed prompt template: {e}") from e
    for name in fields:
        if name not in _REQUIRED_ONCE + _OPTIONAL:
            raise ValueError(f"Unknown placeholder {{{name}}} in prompt template")
        counts[name] = counts.get(name, 0) + 1
    for name in _REQUIRED_ONCE:
        if counts.get(name, 0) != 1:
            raise ValueError(
                f"Prompt template must contain {{{name}}} exactly once "
                f"(found {counts.get(name, 0)})"
            )
    for name in _OPTIONAL:
        if counts.get(name, 0) > 1:
            raise ValueError(f"Prompt template contains {{{name}}} more than once")


def render_prompt_template(ctx: TaskContext) -> str:
    """Render ctx.prompt_template for the model.

    Strips the template's header block, checks that {task_spec} and
    {source_files} each occur exactly once, and fills:
      task_spec, source_files  - from the context
      editable_files           - resolve_editable_files(ctx), one "- path" per line
      test_cmd                 - TEST_CMD
      example_file             - the first editable file (the label used in the
                                 template's output-format example)

    Raises ValueError for a missing, duplicated or unknown placeholder, so a
    bad template fails before any API call is made.
    """
    body = strip_template_header(ctx.prompt_template)
    _check_placeholders(body)
    editable = resolve_editable_files(ctx)
    example = editable[0] if editable else next(iter(ctx.source_files), "path/to/file.py")
    return body.format(
        task_spec=ctx.task_spec,
        source_files=format_source_files(ctx.source_files),
        editable_files="\n".join(f"- {path}" for path in editable),
        test_cmd=TEST_CMD,
        example_file=example,
    )
