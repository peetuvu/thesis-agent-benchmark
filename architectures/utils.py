"""Shared utilities for architecture implementations."""

from __future__ import annotations

import re

# Matches labeled code blocks: ```path/to/file.ext\n<content>\n```
# The label must contain a dot to distinguish from language identifiers
# (e.g. "python", "javascript" won't match but "game.py" will).
_CODE_BLOCK_RE = re.compile(
    r"```([\w][\w/.-]*\.[\w]+)\s*\n(.*?)```",
    re.DOTALL,
)


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
