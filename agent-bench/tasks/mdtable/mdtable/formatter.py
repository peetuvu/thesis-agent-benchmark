def parse_table(lines: list[str]) -> dict | None:
    """Attempt to parse a sequence of lines as a markdown table.

    Returns a dict with keys: headers, alignments, rows. Or None if not a table.
    """
    raise NotImplementedError("Not implemented")


def format_table(table: dict) -> list[str]:
    """Format a parsed table into clean, aligned markdown lines."""
    raise NotImplementedError("Not implemented")


def format_markdown(text: str) -> str:
    """Process a full markdown document: find tables, reformat them,
    preserve everything else unchanged."""
    raise NotImplementedError("Not implemented")
